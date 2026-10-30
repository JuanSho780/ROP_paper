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
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent.parent.parent.parent  # system/training/phase1_segmentation/ -> ROP_paper/
EXTRACTED_DIR = ROOT / "datasets" / "extracted"
MANIFEST_PATH = ROOT / "datasets" / "manifest" / "HVDROPDB.parquet"

STRUCTURES = ["BV", "OD", "RIDGE"]  # orden fijo = orden de canal de salida del modelo
STRUCTURE_TO_CHANNEL = {s: i for i, s in enumerate(STRUCTURES)}

IMG_SIZE = 384
MASK_BINARIZE_THRESH = 127


def crop_dark_border(image: np.ndarray, thresh: int = 10) -> tuple[int, int, int, int]:
    """Bounding box (r0, r1, c0, c1) del contenido no-negro (vineta del retinografo)."""
    gray = image.astype(np.float32).mean(axis=2)
    content = gray > thresh
    rows = np.where(content.any(axis=1))[0]
    cols = np.where(content.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return 0, image.shape[0], 0, image.shape[1]
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def clahe_green_channel(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Aplica CLAHE solo al canal verde (RGB) y devuelve la imagen RGB reconstruida."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    out = image.copy()
    out[:, :, 1] = clahe.apply(image[:, :, 1])
    return out


def build_geometric_transform(train: bool, mean: tuple, std: tuple) -> A.Compose:
    ops = [
        A.LongestMaxSize(max_size=IMG_SIZE, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
        A.PadIfNeeded(
            min_height=IMG_SIZE, min_width=IMG_SIZE,
            border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0,
        ),
    ]
    if train:
        ops += [
            A.Rotate(limit=180, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0,
                      interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.0, p=0.5),
        ]
    # Nota: se probo sumar zoom/traslacion leve (Affine) + deformacion elastica suave
    # (ElasticTransform) - bajo mean_dice de 0.6012 a 0.5881 (OD empeoro bastante, BV
    # no mejoro) - revertido. Ver README de esta carpeta, seccion "Revision 3".
    ops += [A.Normalize(mean=mean, std=std), ToTensorV2()]
    return A.Compose(ops)


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
