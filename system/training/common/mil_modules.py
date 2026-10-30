"""Piezas de arquitectura MIL compartidas entre Fase 2 y Fase 3: MLP clinico,
Cross-Attention por imagen, Attention Pooling, cabezales de clasificacion.

Fase 2 las usa con el backbone congelado (tokens precalculados). Fase 3 las
reutiliza IDENTICAS, agregando el backbone descongelado y el vector clinico
extendido con PMA - ver system/training/phase3_finetune/model.py.
"""
import torch
import torch.nn as nn

VISUAL_DIM = 768       # dim de los tokens del ViT (fijo, viene del backbone de Fase 1)
ATTN_DIM = 256         # dim del embedding clinico / dim interna de Cross-Attention
NUM_HEADS = 4
FF_DIM = 512           # dim oculta del feed-forward del bloque Cross-Attention
HEAD_HIDDEN = 128

N_STAGE_CLASSES = 6    # Estadio 0-5
N_PLUS_CLASSES = 3     # Normal / Pre-Plus / Plus


class ClinicalMLP(nn.Module):
    """Vector clinico -> embedding. `in_dim` varia por fase: 3 en Fase 2
    (GA, BW, sexo - sin PMA), 5 en Fase 3 (+ PMA, pma_available)."""

    def __init__(self, in_dim: int, out_dim: int = ATTN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_dim),
        )

    def forward(self, clinical: torch.Tensor) -> torch.Tensor:
        return self.net(clinical)  # (B, ATTN_DIM)


class CrossAttentionBlock(nn.Module):
    """Query=embedding clinico (1 por bag), Key/Value=tokens visuales de UNA imagen.

    Bloque estilo Transformer (residual + LayerNorm + feed-forward), no una
    operacion de atencion aislada - README principal S4.2. Unidireccional:
    el vector clinico "pregunta" sobre la imagen, nunca al reves (el lado
    clinico es un solo vector, no una secuencia - atencion en la direccion
    contraria se degeneraria a una simple inyeccion lineal, no atencion real).
    """

    def __init__(self, attn_dim: int = ATTN_DIM, visual_dim: int = VISUAL_DIM,
                 num_heads: int = NUM_HEADS, ff_dim: int = FF_DIM):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=attn_dim, num_heads=num_heads, kdim=visual_dim, vdim=visual_dim,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(attn_dim)
        self.ff = nn.Sequential(
            nn.Linear(attn_dim, ff_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ff_dim, attn_dim),
        )
        self.norm2 = nn.LayerNorm(attn_dim)

    def forward(self, clinical_query: torch.Tensor, visual_tokens: torch.Tensor) -> torch.Tensor:
        """
        clinical_query: (B, ATTN_DIM) - un embedding por bag.
        visual_tokens: (B, N_tokens, VISUAL_DIM) - tokens de UNA imagen del bag.
        -> (B, ATTN_DIM) vector fusionado para esa imagen.
        """
        q = clinical_query.unsqueeze(1)  # (B, 1, ATTN_DIM)
        attn_out, _ = self.attn(query=q, key=visual_tokens, value=visual_tokens, need_weights=False)
        x = self.norm1(q + attn_out)
        x = self.norm2(x + self.ff(x))
        return x.squeeze(1)  # (B, ATTN_DIM)


class AttentionPooling(nn.Module):
    """Agrega los vectores fusionados de un bag (tamano variable, con
    padding+mascara) en un unico vector z - README principal S4.3.

    Formula estandar de Attention-MIL (Ilse et al. 2018): score_i =
    w^T tanh(V h_i), softmax SOLO sobre las posiciones reales del bag
    (bag_mask), suma ponderada. Las posiciones de padding reciben score
    -inf antes del softmax -> peso exactamente 0, no contaminan z.
    """

    def __init__(self, dim: int = ATTN_DIM, hidden: int = 128):
        super().__init__()
        self.V = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)

    def forward(self, fused_per_image: torch.Tensor, bag_mask: torch.Tensor) -> torch.Tensor:
        """
        fused_per_image: (B, N_img, ATTN_DIM) - vector fusionado por imagen del bag.
        bag_mask: (B, N_img) bool - True donde la posicion es una imagen real (no padding).
        -> z: (B, ATTN_DIM)
        """
        scores = self.w(torch.tanh(self.V(fused_per_image))).squeeze(-1)  # (B, N_img)
        scores = scores.masked_fill(~bag_mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)  # (B, N_img), posiciones de padding -> 0
        z = (weights.unsqueeze(-1) * fused_per_image).sum(dim=1)  # (B, ATTN_DIM)
        return z


def make_head(in_dim: int, out_dim: int, hidden: int = HEAD_HIDDEN) -> nn.Sequential:
    """MLP pequeño por cabezal (Linear -> ReLU -> Linear), README principal S4.4."""
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, out_dim))
