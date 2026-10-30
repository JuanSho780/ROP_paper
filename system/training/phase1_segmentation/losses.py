"""Dice Loss (+ variante combinada con BCE) multi-canal, enmascarada por
canal y por muestra.

Cada imagen de HVDROPDB solo tiene mascara real para UNA estructura (o dos,
en los 38/300 casos de fotos compartidas entre estructuras - pero como
dataset.py trata cada fila como una muestra independiente, siempre es una
sola mascara por muestra en la practica). Los otros canales no deben
aportar gradiente - mismo principio de `$m_i$` de la perdida clinica
enmascarada (README principal S4.4), aplicado aqui a nivel de canal.

`MaskedDiceBCELoss` (revision 2, tras la primera corrida): Dice Loss puro
tiene gradiente debil/inestable cuando la prediccion casi no se solapa con
la mascara real - un problema conocido para estructuras finas y dispersas
como los vasos (BV), que en la primera corrida quedo con Dice muy por
debajo de OD pese a tener mas pixeles positivos (ver README de esta
carpeta). Sumar BCE (penaliza error pixel a pixel de forma mas estable,
independiente del solape global) es la correccion estandar para este
problema - se usa junto al decoder multi-escala de model.py, no en vez de.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor, channel_valid: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, C, H, W) - salida cruda del modelo.
        target: (B, C, H, W) - mascara binaria (0/1); canales sin dato pueden
            tener cualquier valor, se ignoran via channel_valid.
        channel_valid: (B, C) bool - True donde ese canal tiene ground truth.
        """
        probs = torch.sigmoid(logits)
        dims = (-1, -2)  # reducir sobre H, W; queda (B, C)
        intersection = (probs * target).sum(dim=dims)
        union = probs.sum(dim=dims) + target.sum(dim=dims)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        loss_per_bc = 1 - dice  # (B, C)

        valid = channel_valid.float()
        n_valid = valid.sum()
        if n_valid == 0:
            return logits.sum() * 0.0  # no deberia pasar nunca; evita NaN si pasara
        return (loss_per_bc * valid).sum() / n_valid


class MaskedDiceBCELoss(nn.Module):
    """0.5 * MaskedDiceLoss + 0.5 * BCE-con-logits enmascarada por canal."""

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice = MaskedDiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor, channel_valid: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice(logits, target, channel_valid)

        bce_per_bc = F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean(dim=(-1, -2))  # (B, C)
        valid = channel_valid.float()
        n_valid = valid.sum()
        bce_loss = (bce_per_bc * valid).sum() / n_valid if n_valid > 0 else logits.sum() * 0.0

        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


@torch.no_grad()
def masked_dice_score(logits: torch.Tensor, target: torch.Tensor, channel_valid: torch.Tensor,
                       threshold: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    """Dice score (no loss) por canal, promediado solo sobre (muestra, canal) validos.

    Devuelve un tensor (C,) con el Dice promedio de cada estructura, para
    poder reportar y elegir el mejor checkpoint por estructura y en
    promedio - ver train.py.
    """
    preds = (torch.sigmoid(logits) >= threshold).float()
    dims = (-1, -2)
    intersection = (preds * target).sum(dim=dims)
    union = preds.sum(dim=dims) + target.sum(dim=dims)
    dice = (2 * intersection + eps) / (union + eps)  # (B, C)

    valid = channel_valid.float()
    per_channel_sum = (dice * valid).sum(dim=0)
    per_channel_n = valid.sum(dim=0).clamp(min=1)
    return per_channel_sum / per_channel_n
