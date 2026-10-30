"""Perdida COMPLETA de Fase 3: masking `$m_i$` + pesos de incertidumbre
homocedastica `$\\sigma_i$` (README principal S4.4):

    L = sum_i m_i * ( 1/(2*sigma_i^2) * L_i + log(sigma_i) )

A diferencia de Fase 2 (pesos fijos, sin sigma_i - ver discusion de diseño
en el chat de planeacion: activarlos desde el arranque con modulos recien
inicializados era propenso a valores ruidosos), en Fase 3 los modulos ya
vienen calibrados por el warm-start de Fase 2, asi que sigma_i tiene una
base estable desde la cual aprender a balancear las 3 tareas.

Parametrizacion: se aprende `log_var_i = log(sigma_i^2)` directamente (no
sigma_i), truco estandar (Kendall & Gal, 2017) que evita tener que
restringir sigma_i > 0 explicitamente - log_var_i puede ser cualquier real,
sigma_i^2 = exp(log_var_i) siempre es positivo. Con esa sustitucion:
    1/(2*sigma_i^2) = 0.5 * exp(-log_var_i)
    log(sigma_i)     = 0.5 * log_var_i
Inicializado en log_var_i=0 (sigma_i=1, sin reponderar al arrancar - mismo
punto de partida que la suma simple de Fase 2).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_mean(per_sample_loss: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    valid = valid.float()
    n_valid = valid.sum()
    if n_valid == 0:
        return per_sample_loss.sum() * 0.0
    return (per_sample_loss * valid).sum() / n_valid


class MaskedFocalLossWithDistance(nn.Module):
    """Identica a la de Fase 2 - ver system/training/phase2_warmup/losses.py para el detalle."""

    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0, distance_weight: float = 0.1):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma
        self.distance_weight = distance_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        target_safe = target.clamp(min=0, max=logits.shape[1] - 1)
        ce = F.cross_entropy(logits, target_safe, reduction="none")
        pt = torch.exp(-ce)
        alpha_t = self.alpha[target_safe]
        focal = alpha_t * (1 - pt) ** self.gamma * ce

        probs = torch.softmax(logits, dim=1)
        classes = torch.arange(logits.shape[1], device=logits.device, dtype=probs.dtype)
        expected_distance = (probs * (classes.unsqueeze(0) - target_safe.unsqueeze(1).float()).abs()).sum(dim=1)

        per_sample = focal * (1 + self.distance_weight * expected_distance)
        return _masked_mean(per_sample, valid)


class MaskedWeightedCrossEntropy(nn.Module):
    def __init__(self, class_weights: torch.Tensor):
        super().__init__()
        self.register_buffer("class_weights", class_weights)

    def forward(self, logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        target_safe = target.clamp(min=0, max=logits.shape[1] - 1)
        per_sample = F.cross_entropy(logits, target_safe, weight=self.class_weights, reduction="none")
        return _masked_mean(per_sample, valid)


class MaskedBCELoss(nn.Module):
    def __init__(self, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else torch.tensor(1.0))

    def forward(self, logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        per_sample = F.binary_cross_entropy_with_logits(
            logits, target.float(), pos_weight=self.pos_weight, reduction="none",
        )
        return _masked_mean(per_sample, valid)


class Phase3Loss(nn.Module):
    """Suma de las 3 perdidas enmascaradas, CON pesos de incertidumbre
    homocedastica aprendidos (`log_var`, uno por tarea - ver docstring del modulo).
    """

    TASK_ORDER = ("stage", "plus", "ap_rop")

    def __init__(self, stage_alpha: torch.Tensor, plus_weights: torch.Tensor, ap_rop_pos_weight: torch.Tensor):
        super().__init__()
        self.stage_loss = MaskedFocalLossWithDistance(alpha=stage_alpha)
        self.plus_loss = MaskedWeightedCrossEntropy(class_weights=plus_weights)
        self.ap_rop_loss = MaskedBCELoss(pos_weight=ap_rop_pos_weight)
        self.log_var = nn.Parameter(torch.zeros(len(self.TASK_ORDER)))  # sigma_i=1 al arrancar

    def forward(self, outputs: dict, targets: dict, valid: dict) -> tuple[torch.Tensor, dict]:
        raw = {
            "stage": self.stage_loss(outputs["stage_logits"], targets["stage"], valid["stage"]),
            "plus": self.plus_loss(outputs["plus_logits"], targets["plus"], valid["plus"]),
            "ap_rop": self.ap_rop_loss(outputs["ap_rop_logits"], targets["ap_rop"], valid["ap_rop"]),
        }
        total = raw["stage"].new_zeros(())
        sigmas = {}
        for i, name in enumerate(self.TASK_ORDER):
            log_var_i = self.log_var[i]
            weighted = 0.5 * torch.exp(-log_var_i) * raw[name] + 0.5 * log_var_i
            total = total + weighted
            sigmas[name] = torch.exp(0.5 * log_var_i).item()  # sigma_i, para loguear

        parts = {k: v.item() for k, v in raw.items()}
        parts["sigma"] = sigmas
        return total, parts
