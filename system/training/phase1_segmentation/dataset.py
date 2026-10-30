"""Dataset de Fase 1: pares (imagen, mascara) de HVDROPDB para segmentacion.

Lee datasets/manifest/HVDROPDB.parquet (ya con split train/val asignado por
system/dataset_build/partition/split_hvdropdb.py, agrupado por image_hash -
ver ese script para el porque: 38/300 imagenes se repiten entre estructuras,
agrupar evita que la misma foto caiga en train para una estructura y en val
para otra).

Cada fila del parquet = una imagen + UNA estructura (BV/OD/RIDGE) con mascara
real. El resto de canales de salida del modelo se enmascara (no aporta
gradiente para esa fila) - mismo principio de perdida enmascarada que se usa
en todo el proyecto para las tareas clinicas (ver README principal S4.4),
aplicado aqui a nivel de canal de segmentacion en vez de cabezal clinico.

Preprocesamiento real (verificado contra los datos, no asumido - ver
system/training/phase1_segmentation/README.md para el detalle):
- Recorte de bordes oscuros estaticos (bounding box del contenido no-negro).
  Necesario de verdad: las imagenes RetCam (480x640) NO son cuadradas y
  tienen vineta negra real en los bordes, a diferencia de Neo (2040x2040,
  practicamente sin borde). Sin este paso, estirar 480x640 a 384x384
  distorsiona las proporciones de la retina.
- Resize-con-padding a 384x384 (LongestMaxSize + PadIfNeeded) en vez de un
  resize directo - preserva la proporcion real en vez de deformarla.
- CLAHE restringido al canal verde (realce capilar, canal verde es el de
  mayor contraste vascular en fondo de ojo - mismo criterio que README S3.2).
- Augmentation (solo train): rotacion libre (equivalente a 0-360, parametrizada
  como +-180 en torno al centro), flip horizontal, jitter de brillo/contraste.
- Mascara binarizada al cargar (las mascaras originales tienen antialiasing
  real en los bordes - 89% de pixeles en {0,255}, el resto valores
  intermedios de transicion - se umbraliza a 0/1 con threshold 127) e
  interpolada con nearest (no bilinear) en todo el pipeline geometrico, para
  no introducir valores fraccionarios en una mascara binaria.
- Normalizacion: mean/std exactos que espera el checkpoint preentrenado de
  timm (NO los genericos de ImageNet - este checkpoint en particular usa
  mean=std=0.5, verificado con timm.data.resolve_data_config).
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent.parent.parent.parent  # system/training/phase1_segmentation/ -> ROP_paper/
EXTRACTED_DIR = ROOT / "datasets" / "extracted"
MANIFEST_PATH = ROOT / "datasets" / "manifest" / "HVDROPDB.parquet"

sys.path.insert(0, str(ROOT / "system" / "training"))
from common.image_preprocessing import (  # noqa: E402
    IMG_SIZE,
    build_geometric_transform,
    clahe_green_channel,
    crop_dark_border,
)

STRUCTURES = ["BV", "OD", "RIDGE"]  # orden fijo = orden de canal de salida del modelo
STRUCTURE_TO_CHANNEL = {s: i for i, s in enumerate(STRUCTURES)}

MASK_BINARIZE_THRESH = 127


class HVDROPDBSegmentationDataset(Dataset):
    def __init__(self, split: str, mean: tuple, std: tuple):
        df = pd.read_parquet(MANIFEST_PATH)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.transform = build_geometric_transform(train=(split == "train"), mean=mean, std=std)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        image = cv2.imread(str(EXTRACTED_DIR / row["image_path"]), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_raw = cv2.imread(str(EXTRACTED_DIR / row["mask_path"]), cv2.IMREAD_GRAYSCALE)
        mask_bin = (mask_raw >= MASK_BINARIZE_THRESH).astype(np.uint8)

        r0, r1, c0, c1 = crop_dark_border(image)
        image = image[r0:r1, c0:c1]
        mask_bin = mask_bin[r0:r1, c0:c1]

        image = clahe_green_channel(image)

        out = self.transform(image=image, mask=mask_bin)
        image_t = out["image"]  # float32 CHW, ya normalizada
        mask_t = out["mask"].float()  # HxW, 0/1

        channel = STRUCTURE_TO_CHANNEL[row["structure"]]
        target = torch.zeros((len(STRUCTURES), IMG_SIZE, IMG_SIZE), dtype=torch.float32)
        target[channel] = mask_t
        channel_valid = torch.zeros(len(STRUCTURES), dtype=torch.bool)
        channel_valid[channel] = True

        return image_t, target, channel_valid
