"""Fase 2: MLP clinico + Cross-Attention por imagen + Attention Pooling (MIL)
+ 3 cabezales (Estadio, Plus, AP-ROP).

Arquitectura exacta segun README principal S4.1-S4.4, con el backbone
visual YA CONGELADO (los tokens llegan precalculados desde
precompute_tokens.py, este modulo nunca ve el backbone directamente).

Los bloques (MLP clinico, Cross-Attention, Attention Pooling, cabezales)
viven en system/training/common/mil_modules.py - Fase 3 los reutiliza
identicos, agregando el backbone descongelado y PMA al vector clinico.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent.parent.parent  # system/training/phase2_warmup/ -> ROP_paper/
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

CLINICAL_IN_DIM = 3    # GA (z-score), BW (z-score), sexo (binario) - sin PMA en Fase 2


class WarmupModel(nn.Module):
    """Modelo completo de Fase 2 (backbone NO incluido - ya esta congelado y
    sus tokens vienen precalculados, ver precompute_tokens.py/dataset.py)."""

    def __init__(self):
        super().__init__()
        self.clinical_mlp = ClinicalMLP(in_dim=CLINICAL_IN_DIM, out_dim=ATTN_DIM)
        self.cross_attn = CrossAttentionBlock()
        self.pooling = AttentionPooling()
        self.head_stage = make_head(ATTN_DIM, N_STAGE_CLASSES)
        self.head_plus = make_head(ATTN_DIM, N_PLUS_CLASSES)
        self.head_ap_rop = make_head(ATTN_DIM, 1)

    def forward(self, clinical: torch.Tensor, visual_tokens: torch.Tensor, bag_mask: torch.Tensor):
        """
        clinical: (B, CLINICAL_IN_DIM) - un vector por bag.
        visual_tokens: (B, N_img, N_tokens, VISUAL_DIM) - tokens precalculados
            de cada imagen del bag (padding en N_img segun bag_mask).
        bag_mask: (B, N_img) bool.
        """
        b, n_img, n_tok, d = visual_tokens.shape
        clinical_q = self.clinical_mlp(clinical)  # (B, ATTN_DIM)

        # Cross-Attention se aplica por imagen - "aplanamos" (B, N_img) en un solo
        # eje de batch, repitiendo el Query clinico para cada imagen del bag.
        visual_flat = visual_tokens.reshape(b * n_img, n_tok, d)
        clinical_rep = clinical_q.unsqueeze(1).expand(-1, n_img, -1).reshape(b * n_img, -1)
        fused_flat = self.cross_attn(clinical_rep, visual_flat)  # (B*N_img, ATTN_DIM)
        fused = fused_flat.reshape(b, n_img, -1)  # (B, N_img, ATTN_DIM)

        z = self.pooling(fused, bag_mask)  # (B, ATTN_DIM)

        return {
            "stage_logits": self.head_stage(z),
            "plus_logits": self.head_plus(z),
            "ap_rop_logits": self.head_ap_rop(z).squeeze(-1),
        }
