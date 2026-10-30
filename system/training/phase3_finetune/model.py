"""Fase 3: backbone ViT DESCONGELADO + MLP clinico (con PMA) + Cross-Attention
+ Attention Pooling (MIL) + 3 cabezales - todo entrenable a la vez.

A diferencia de Fase 2 (backbone congelado, tokens precalculados), aqui el
backbone forma parte del modelo y se afina junto con el resto - cada imagen
del bag pasa por el ViT completo (forward+backward) en cada paso, no hay
cacheo posible porque el backbone cambia entre pasos.

Reutiliza identicos los bloques de system/training/common/mil_modules.py
(MLP clinico, Cross-Attention, Attention Pooling, cabezales) - la unica
diferencia de arquitectura respecto a Fase 2 es `CLINICAL_IN_DIM=5` (se
agrega PMA + pma_available al vector clinico, README principal S5 Fase 3).
"""
import sys
from pathlib import Path

import timm
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent.parent.parent  # system/training/phase3_finetune/ -> ROP_paper/
sys.path.insert(0, str(ROOT / "system" / "training"))
from common.mil_modules import (  # noqa: E402
    ATTN_DIM,
    N_PLUS_CLASSES,
    N_STAGE_CLASSES,
    AttentionPooling,
    ClinicalMLP,
    CrossAttentionBlock,
    make_head,
)

CLINICAL_IN_DIM = 5  # GA, BW, sexo, PMA, pma_available (vs. 3 en Fase 2 - sin PMA)


class FinetuneModel(nn.Module):
    """Modelo completo de Fase 3, backbone incluido y descongelado."""

    def __init__(self, backbone_name: str, pretrained_backbone_state_dict: dict | None = None):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        if pretrained_backbone_state_dict is not None:
            self.backbone.load_state_dict(pretrained_backbone_state_dict)
        self.num_prefix_tokens = self.backbone.num_prefix_tokens
        for p in self.backbone.parameters():
            p.requires_grad = True  # descongelado - a diferencia de Fase 1 (fijo) y Fase 2 (ni se incluye)

        self.clinical_mlp = ClinicalMLP(in_dim=CLINICAL_IN_DIM, out_dim=ATTN_DIM)
        self.cross_attn = CrossAttentionBlock()
        self.pooling = AttentionPooling()
        self.head_stage = make_head(ATTN_DIM, N_STAGE_CLASSES)
        self.head_plus = make_head(ATTN_DIM, N_PLUS_CLASSES)
        self.head_ap_rop = make_head(ATTN_DIM, 1)

    def load_warmup_weights(self, warmup_state_dict: dict) -> None:
        """Carga clinical_mlp/cross_attn/pooling/cabezales desde un checkpoint
        de Fase 2 (del MISMO fold) - el backbone se carga aparte, desde
        Fase 1 (ver load_pretrained_backbone en train.py)."""
        own_state = self.state_dict()
        loaded = 0
        for k, v in warmup_state_dict.items():
            if k in own_state and own_state[k].shape == v.shape:
                own_state[k] = v
                loaded += 1
        self.load_state_dict(own_state)
        print(f"[warm-start] {loaded} tensores cargados desde el checkpoint de Fase 2")

    def forward_visual_tokens(self, images_flat: torch.Tensor) -> torch.Tensor:
        """images_flat: (N, 3, 384, 384) -> tokens de PARCHE (sin CLS), (N, 576, 768)."""
        tokens = self.backbone.forward_features(images_flat)
        return tokens[:, self.num_prefix_tokens:, :]

    def forward(self, clinical: torch.Tensor, images: torch.Tensor, bag_mask: torch.Tensor):
        """
        clinical: (B, CLINICAL_IN_DIM).
        images: (B, N_img, 3, 384, 384) - imagenes crudas preprocesadas (NO tokens -
            a diferencia de Fase 2, acá se calculan en vivo porque el backbone cambia).
        bag_mask: (B, N_img) bool.
        """
        b, n_img, c, h, w = images.shape
        clinical_q = self.clinical_mlp(clinical)  # (B, ATTN_DIM)

        images_flat = images.reshape(b * n_img, c, h, w)
        mask_flat = bag_mask.reshape(b * n_img)

        # El backbone (la parte cara: forward+backward por el ViT completo) SOLO
        # procesa las imagenes reales - las de padding nunca lo tocan. El gradiente
        # de padding ya daba 0 por la mascara en AttentionPooling, pero sin esto se
        # seguia gastando computo real (forward+backward) en imagenes puro relleno.
        real_images = images_flat[mask_flat]
        real_tokens = self.forward_visual_tokens(real_images)  # (n_real, 576, 768)
        visual_flat = torch.zeros(
            b * n_img, real_tokens.shape[1], real_tokens.shape[2],
            device=images.device, dtype=real_tokens.dtype,
        )
        visual_flat[mask_flat] = real_tokens

        clinical_rep = clinical_q.unsqueeze(1).expand(-1, n_img, -1).reshape(b * n_img, -1)
        fused_flat = self.cross_attn(clinical_rep, visual_flat)  # (B*N_img, ATTN_DIM)
        fused = fused_flat.reshape(b, n_img, -1)

        z = self.pooling(fused, bag_mask)  # (B, ATTN_DIM)

        return {
            "stage_logits": self.head_stage(z),
            "plus_logits": self.head_plus(z),
            "ap_rop_logits": self.head_ap_rop(z).squeeze(-1),
        }
