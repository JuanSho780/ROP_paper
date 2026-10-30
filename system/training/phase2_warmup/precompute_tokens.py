"""Precalcula y cachea los tokens de parche del backbone (congelado) de Fase 1
para cada imagen de ROP-VL + FARFUM, antes de entrenar Fase 2.

Por que cachear: el backbone esta congelado en Fase 2 (README principal S5,
"Optimizacion de computo: como el visual esta congelado, los tokens del ViT
pueden precalcularse y cachearse una sola vez - no cambian entre epocas").
Como consecuencia, Fase 2 NO aplica augmentation de imagen (si la imagen
cambiara de una epoca a otra, los tokens cacheados dejarian de ser
validos) - el unico preprocesamiento es el mismo pipeline determinista de
Fase 1 (recorte + CLAHE verde + resize-con-padding + normalizacion),
importado de system/training/common/image_preprocessing.py para
garantizar que sea EXACTAMENTE el mismo que vio el backbone al entrenarse.

Salida: un archivo .pt por imagen en cached_tokens/<sample_id>.pt, tensor
float16 (576, 768) - los tokens de parche de la ULTIMA capa del ViT
(model.forward_tokens(), el mismo metodo que usa Fase 1 para exponerlos -
no los multi-capa del decoder de segmentacion, eso es interno solo a
Fase 1).

Requiere: el checkpoint de Fase 1 ya entrenado
(system/training/phase1_segmentation/checkpoints/backbone_best.pt).
"""
import sys
from pathlib import Path

import pandas as pd
import timm
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent.parent.parent  # system/training/phase2_warmup/ -> ROP_paper/
EXTRACTED_DIR = ROOT / "datasets" / "extracted"
MANIFEST_PATH = ROOT / "datasets" / "manifest" / "rop_unified_manifest.parquet"
PHASE1_CHECKPOINT = ROOT / "system" / "training" / "phase1_segmentation" / "checkpoints" / "backbone_best.pt"
CACHE_DIR = Path(__file__).resolve().parent / "cached_tokens"

sys.path.insert(0, str(ROOT / "system" / "training"))
from common.image_preprocessing import build_geometric_transform, load_and_preprocess_image  # noqa: E402

BATCH_SIZE = 32
NUM_WORKERS = 4


class ImageOnlyDataset(Dataset):
    """Carga + preprocesa (sin augmentation) una imagen - para precalculo de tokens."""

    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_t = load_and_preprocess_image(str(EXTRACTED_DIR / row["image_path"]), self.transform)
        return image_t, row["sample_id"]


def load_frozen_backbone(device: torch.device):
    ckpt = torch.load(PHASE1_CHECKPOINT, map_location=device)
    backbone_name = ckpt["backbone_name"]
    print(f"[backbone] {backbone_name} (epoca {ckpt['epoch']} de Fase 1, mean_dice={ckpt['mean_dice']:.4f})")

    backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
    backbone.load_state_dict(ckpt["backbone_state_dict"])
    backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    num_prefix_tokens = backbone.num_prefix_tokens
    data_cfg = timm.data.resolve_data_config({}, model=backbone)
    return backbone, num_prefix_tokens, data_cfg["mean"], data_cfg["std"]


@torch.no_grad()
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    backbone, num_prefix_tokens, mean, std = load_frozen_backbone(device)
    transform = build_geometric_transform(train=False, mean=mean, std=std)

    df = pd.read_parquet(MANIFEST_PATH)
    df = df[df["dataset_origen"].isin(["ROP-VL", "FARFUM"])].reset_index(drop=True)
    print(f"[datos] {len(df)} imagenes a procesar (ROP-VL + FARFUM)")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pending = df[~df["sample_id"].apply(lambda sid: (CACHE_DIR / f"{sid}.pt").exists())]
    if len(pending) < len(df):
        print(f"[cache] {len(df) - len(pending)} ya cacheadas, se omiten; {len(pending)} pendientes")
    if len(pending) == 0:
        print("[cache] nada pendiente, listo.")
        return

    loader = DataLoader(
        ImageOnlyDataset(pending, transform), batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    done = 0
    for images, sample_ids in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            tokens = backbone.forward_features(images)
        tokens = tokens[:, num_prefix_tokens:, :].to(torch.float16).cpu()

        for i, sid in enumerate(sample_ids):
            torch.save(tokens[i].clone(), CACHE_DIR / f"{sid}.pt")
        done += len(sample_ids)
        if done % (BATCH_SIZE * 10) == 0 or done == len(pending):
            print(f"  {done}/{len(pending)}")

    print(f"-> {CACHE_DIR} ({len(df)} archivos en total)")


if __name__ == "__main__":
    main()
