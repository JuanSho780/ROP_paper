"""Preprocesamiento de imagen COMPARTIDO entre Fase 1 y Fase 2+.

Critico que esto viva en un solo lugar: el backbone de Fase 1 se entreno
viendo exactamente este preprocesamiento (recorte + CLAHE verde + resize-
con-padding + normalizacion especifica del checkpoint). Si Fase 2 usara un
preprocesamiento distinto para generar los tokens, estarian fuera de la
distribucion que el backbone aprendio a interpretar - los tokens cacheados
dejarian de tener sentido.

Ver system/training/phase1_segmentation/README.md para el detalle completo
y la verificacion real (viñeta negra en RetCam, aspecto no cuadrado, etc.)
"""
import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2

IMG_SIZE = 384


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
    """train=True agrega augmentation (rotacion, flip, brillo) - usar SOLO en
    Fase 1 (donde el backbone se sigue afinando). Fase 2+ usa train=False
    siempre (tokens cacheados, deterministicos - ver precompute_tokens.py).
    """
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
    ops += [A.Normalize(mean=mean, std=std), ToTensorV2()]
    return A.Compose(ops)


def load_and_preprocess_image(path: str, transform: A.Compose) -> "torch.Tensor":
    """Crudo en disco -> tensor CHW normalizado, listo para el backbone.

    Mismos 3 pasos deterministicos que Fase 1 (recorte, CLAHE verde,
    resize-con-padding), sin augmentation - se usa tal cual en
    precompute_tokens.py (Fase 2) y es la base de lo que hace
    HVDROPDBSegmentationDataset en Fase 1 (que ademas le agrega augmentation
    si train=True y maneja la mascara en paralelo).
    """
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    r0, r1, c0, c1 = crop_dark_border(image)
    image = image[r0:r1, c0:c1]
    image = clahe_green_channel(image)
    out = transform(image=image)
    return out["image"]
