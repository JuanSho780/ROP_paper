"""Backbone ViT (timm, preentrenado ImageNet) + decoder de segmentacion multi-escala.

El ViT devuelve una secuencia de tokens por parche (N x D), no un vector
pooled - necesario para que, mas adelante en Fase 2/3, la Cross-Attention
pueda atender a zonas especificas de la retina (ver README principal S4.1).
`forward_tokens()` (tokens de la ULTIMA capa solamente) es el metodo que
Fase 2/3 reutiliza - no cambia con esta revision.

Decoder multi-escala (revision 2, tras la primera corrida de 30 epocas):
la primera version del decoder solo usaba la ultima capa del ViT, y BV
(vasos, estructura fina y ramificada) quedo con Dice muy por debajo de OD
(mancha compacta) pese a tener MAS pixeles positivos (6.8% vs 0.6%) -
evidencia real de que el problema era perdida de detalle espacial fino al
decodificar desde una sola capa ya muy abstraida, no desbalance de clases.
Solucion: se extraen tokens de 4 profundidades del ViT (bloques 2, 5, 8, 11
de 12 - es conocido en la literatura de interpretabilidad de ViT que las
capas tempranas retienen mas detalle local/espacial, las tardias se vuelven
mas semanticas/globales), se proyectan y fusionan ANTES del mismo camino de
upsampling que ya existia - estilo DPT (Dense Prediction Transformer)
simplificado, sin la reasignacion de resolucion completa del paper original.
Sigue siendo liviano a proposito: la mascara de Fase 1 se descarta despues,
solo importa darle al backbone una senal de gradiente mas rica.
"""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

BACKBONE_NAME = "vit_base_patch16_384.augreg_in21k_ft_in1k"
NUM_STRUCTURES = 3
INTERMEDIATE_BLOCK_INDICES = [2, 5, 8, 11]  # de 12 bloques (0-indexado); temprano -> tardio
PROJ_DIM = 128  # dim por capa tras proyectar, antes de concatenar las 4


class ConvUpBlock(nn.Module):
    """Duplica la resolucion espacial (interpolacion + conv de refinamiento)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv(x)


class MultiScaleSegmentationDecoder(nn.Module):
    """Lista de K grillas de tokens (B, N, D), misma resolucion 24x24, distinta
    profundidad del ViT -> mascara multi-canal (B, 3, 384, 384).

    Cada capa se proyecta por separado a PROJ_DIM canales (1x1 conv + BN +
    ReLU), se concatenan en el eje de canal, y se funden con una conv final
    antes de entrar al mismo camino de upsampling que la version anterior.
    """

    def __init__(self, in_dim: int, grid_size: tuple[int, int], n_layers: int, out_size: int = 384):
        super().__init__()
        self.grid_size = grid_size
        self.out_size = out_size

        self.layer_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_dim, PROJ_DIM, kernel_size=1),
                nn.BatchNorm2d(PROJ_DIM),
                nn.ReLU(inplace=True),
            )
            for _ in range(n_layers)
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(PROJ_DIM * n_layers, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        # grid_size tipico (24, 24) -> 4 duplicaciones llegan a 384
        self.up1 = ConvUpBlock(256, 128)   # 24  -> 48
        self.up2 = ConvUpBlock(128, 64)    # 48  -> 96
        self.up3 = ConvUpBlock(64, 32)     # 96  -> 192
        self.up4 = ConvUpBlock(32, 16)     # 192 -> 384
        self.head = nn.Conv2d(16, NUM_STRUCTURES, kernel_size=1)

    def _to_grid(self, tokens: torch.Tensor) -> torch.Tensor:
        b, n, d = tokens.shape
        h, w = self.grid_size
        assert n == h * w, f"se esperaban {h*w} tokens de parche, llegaron {n}"
        return tokens.transpose(1, 2).reshape(b, d, h, w)

    def forward(self, multi_layer_tokens: list[torch.Tensor]) -> torch.Tensor:
        """multi_layer_tokens: lista de K tensores (B, N, D) -> logits (B, 3, out_size, out_size)."""
        grids = [self._to_grid(t) for t in multi_layer_tokens]
        projected = [proj(g) for proj, g in zip(self.layer_proj, grids)]
        x = torch.cat(projected, dim=1)
        x = self.fuse(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        logits = self.head(x)
        if logits.shape[-1] != self.out_size:
            logits = F.interpolate(logits, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
        return logits


class ViTSegmentationModel(nn.Module):
    """Backbone ViT + decoder multi-escala, para Fase 1 (segmentacion en HVDROPDB).

    El backbone (`self.backbone`) es el modulo que se guarda al final de
    Fase 1 y se reutiliza (congelado) en Fase 2/3 - el decoder se descarta.
    """

    def __init__(self, backbone_name: str = BACKBONE_NAME, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        self.num_prefix_tokens = self.backbone.num_prefix_tokens
        grid_size = self.backbone.patch_embed.grid_size  # ej. (24, 24)
        embed_dim = self.backbone.embed_dim
        self.decoder = MultiScaleSegmentationDecoder(
            in_dim=embed_dim, grid_size=grid_size, n_layers=len(INTERMEDIATE_BLOCK_INDICES),
        )

    def forward_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """Devuelve solo los tokens de PARCHE de la ULTIMA capa (sin CLS) - (B, N, D).

        Es el mismo metodo que usara Fase 2/3 para alimentar la
        Cross-Attention (README principal S4.1/S4.2) - no participa del
        decoder multi-escala, que es interno solo a Fase 1.
        """
        tokens = self.backbone.forward_features(images)
        return tokens[:, self.num_prefix_tokens:, :]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        multi_layer_tokens = self.backbone.forward_intermediates(
            images, indices=INTERMEDIATE_BLOCK_INDICES, norm=False,
            output_fmt="NLC", intermediates_only=True,
        )
        return self.decoder(multi_layer_tokens)
