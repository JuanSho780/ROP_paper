"""Perdidas por cabezal (Estadio, Plus, AP-ROP), cada una enmascarada por
muestra via `$m_i$` (README principal S4.4) - una fila sin ground truth
para una tarea no aporta gradiente a esa tarea, sin importar que prediga el
modelo.

Version de Fase 2 de la perdida total (sin pesos de incertidumbre sigma_i -
ver discusion de diseño en el chat de planeacion: activar sigma_i desde el
arranque, con los modulos nuevos recien inicializados y una cobertura por
tarea muy desigual entre ROP-VL/FARFUM, es propenso a que sigma_i aprenda
valores ruidosos. Sigma_i se activa recien en Fase 3):

    L_fase2 = L_estadio_enmascarada + L_plus_enmascarada + L_ap_rop_enmascarada

(la version completa con sigma_i, para Fase 3, es un modulo aparte que se
escribe cuando llegue esa fase).
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
    """Focal Loss + penalizacion por distancia ordinal, enmascarada - para `stage`.

    Focal Loss (Lin et al. 2017): baja el peso de los ejemplos ya bien
    clasificados (1-p_t)^gamma, deja que el entrenamiento se concentre en
    los dificiles/raros (Estadio 4/5, muy escasos - ver EDA del manifiesto
    unificado). `alpha` son pesos por clase (inversamente proporcionales a
    frecuencia, calculados en train.py desde el pool de entrenamiento real
    de cada fold - no hardcodeados aqui).

    Matriz de penalizacion por distancia: ademas de la clase correcta,
    penaliza que el modelo ponga probabilidad lejos del Estadio real en la
    escala ordinal 0-5 (confundir Estadio 0 con 5 es peor que confundir 2
    con 3) - se implementa como una distancia esperada suave:
    `sum_j P(j) * |j - y|`, multiplicando (no reemplazando) el Focal Loss.
    """

    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0, distance_weight: float = 0.1):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma
        self.distance_weight = distance_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """logits: (B, 6). target: (B,) long, valores 0-5 (irrelevante si valid=False). valid: (B,) bool."""
        target_safe = target.clamp(min=0, max=logits.shape[1] - 1)
        ce = F.cross_entropy(logits, target_safe, reduction="none")  # (B,)
        pt = torch.exp(-ce)
        alpha_t = self.alpha[target_safe]
        focal = alpha_t * (1 - pt) ** self.gamma * ce

        probs = torch.softmax(logits, dim=1)
        classes = torch.arange(logits.shape[1], device=logits.device, dtype=probs.dtype)
        expected_distance = (probs * (classes.unsqueeze(0) - target_safe.unsqueeze(1).float()).abs()).sum(dim=1)

        per_sample = focal * (1 + self.distance_weight * expected_distance)
        return _masked_mean(per_sample, valid)


class MaskedWeightedCrossEntropy(nn.Module):
    """Entropia cruzada ponderada por clase, enmascarada - para `plus`."""

    def __init__(self, class_weights: torch.Tensor):
        super().__init__()
        self.register_buffer("class_weights", class_weights)

    def forward(self, logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        target_safe = target.clamp(min=0, max=logits.shape[1] - 1)
        per_sample = F.cross_entropy(logits, target_safe, weight=self.class_weights, reduction="none")
        return _masked_mean(per_sample, valid)


class MaskedBCELoss(nn.Module):
    """BCE con logits, enmascarada - para `ap_rop` (binario)."""

    def __init__(self, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else torch.tensor(1.0))

    def forward(self, logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        per_sample = F.binary_cross_entropy_with_logits(
            logits, target.float(), pos_weight=self.pos_weight, reduction="none",
        )
        return _masked_mean(per_sample, valid)


class Phase2Loss(nn.Module):
    """Suma de las 3 perdidas enmascaradas, sin ponderacion por incertidumbre (ver docstring del modulo)."""

    def __init__(self, stage_alpha: torch.Tensor, plus_weights: torch.Tensor, ap_rop_pos_weight: torch.Tensor):
        super().__init__()
        self.stage_loss = MaskedFocalLossWithDistance(alpha=stage_alpha)
        self.plus_loss = MaskedWeightedCrossEntropy(class_weights=plus_weights)
        self.ap_rop_loss = MaskedBCELoss(pos_weight=ap_rop_pos_weight)

    def forward(self, outputs: dict, targets: dict, valid: dict) -> tuple[torch.Tensor, dict]:
        l_stage = self.stage_loss(outputs["stage_logits"], targets["stage"], valid["stage"])
        l_plus = self.plus_loss(outputs["plus_logits"], targets["plus"], valid["plus"])
        l_ap_rop = self.ap_rop_loss(outputs["ap_rop_logits"], targets["ap_rop"], valid["ap_rop"])
        total = l_stage + l_plus + l_ap_rop
        return total, {"stage": l_stage.item(), "plus": l_plus.item(), "ap_rop": l_ap_rop.item()}
