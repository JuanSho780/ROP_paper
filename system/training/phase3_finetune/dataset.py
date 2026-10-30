"""Dataset de Fase 3: arma bags (MIL) desde rop_unified_manifest.parquet,
las 3 fuentes clinicas (RIDIRP + ROP-VL + FARFUM) - a diferencia de Fase 2
(solo ROP-VL+FARFUM). Carga imagenes CRUDAS (no hay cacheo posible, el
backbone esta descongelado y cambia entre pasos) con el mismo
preprocesamiento determinista de Fase 1/2 (common/image_preprocessing.py)
MAS augmentation (rotacion, flip, brillo) en train - decision acordada en
el chat de planeacion de Fase 3: el backbone vuelve a tener gradiente real,
mismo riesgo de sobreajuste que motivo usar augmentation en Fase 1.

Vector clinico extendido: [GA, BW, sexo, PMA, pma_available] (5 features,
vs. 3 en Fase 2) - PMA real solo en RIDIRP (100% de sus filas),
`pma_available=False` + PMA imputado por mediana (del fold, del subconjunto
con dato real) para ROP-VL/FARFUM. `sexo` usa el mapeo de 3 vocabularios
distintos por fuente (RIDIRP M/F, ROP-VL male/female, FARFUM boy/girl) -
ver system/training/common/clinical_features.py.

MAX_BAG_SIZE: igual que Fase 2 (cap=8, resampleo aleatorio en train, fijo
en val) - PERO ahora el backbone esta descongelado (forward+backward real
por cada imagen del bag), mucha mas memoria de GPU por imagen que en Fase 2.
Si se corre con mas VRAM disponible (ej. Colab), subir o quitar este cap es
seguro - no cambia la logica, solo el costo de memoria/computo por batch.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent.parent.parent.parent  # system/training/phase3_finetune/ -> ROP_paper/
EXTRACTED_DIR = ROOT / "datasets" / "extracted"
MANIFEST_PATH = ROOT / "datasets" / "manifest" / "rop_unified_manifest.parquet"

sys.path.insert(0, str(ROOT / "system" / "training"))
from common.clinical_features import PLUS_TO_IDX, SEXO_TO_BINARY  # noqa: E402
from common.image_preprocessing import build_geometric_transform, load_and_preprocess_image  # noqa: E402

MAX_BAG_SIZE = 8  # subir/quitar si se corre con mas VRAM (ej. Colab) - ver docstring


def load_phase3_rows() -> pd.DataFrame:
    return pd.read_parquet(MANIFEST_PATH)


def compute_clinical_norm_stats(train_df: pd.DataFrame) -> dict:
    """Media/desvio de GA, BW y PMA, calculados SOLO del train del fold.
    PMA se calcula solo sobre las filas con dato real (pma_available=True) -
    el resto se imputa con esa misma mediana/media, ver __getitem__."""
    bags = train_df.drop_duplicates("bag_id")
    pma_real = bags.loc[bags["pma_available"], "PMA"]
    return {
        "ga_mean": bags["GA"].mean(), "ga_std": bags["GA"].std(),
        "bw_mean": bags["BW"].mean(), "bw_std": bags["BW"].std(),
        "pma_mean": pma_real.mean(), "pma_std": pma_real.std(),
    }


def compute_class_weights(train_df: pd.DataFrame) -> dict:
    """Identico a Fase 2 - ver system/training/phase2_warmup/dataset.py."""
    bags = train_df.drop_duplicates("bag_id")

    stage_counts = bags["stage"].dropna().astype(int).value_counts().reindex(range(6), fill_value=0) + 1
    stage_alpha = (1.0 / stage_counts)
    stage_alpha = (stage_alpha / stage_alpha.sum() * 6).values

    plus_idx = bags["plus"].dropna().map(PLUS_TO_IDX)
    plus_counts = plus_idx.value_counts().reindex(range(3), fill_value=0) + 1
    plus_weights = (1.0 / plus_counts)
    plus_weights = (plus_weights / plus_weights.sum() * 3).values

    ap_rop_vals = bags["ap_rop"].dropna()
    n_pos = (ap_rop_vals == True).sum()  # noqa: E712
    n_neg = (ap_rop_vals == False).sum()  # noqa: E712
    ap_rop_pos_weight = (n_neg / max(n_pos, 1))

    return {
        "stage_alpha": torch.tensor(stage_alpha, dtype=torch.float32),
        "plus_weights": torch.tensor(plus_weights, dtype=torch.float32),
        "ap_rop_pos_weight": torch.tensor(ap_rop_pos_weight, dtype=torch.float32),
    }


class FinetuneBagDataset(Dataset):
    def __init__(self, df: pd.DataFrame, norm_stats: dict, mean: tuple, std: tuple, train: bool, seed: int = 42):
        self.train = train
        self.norm_stats = norm_stats
        self.transform = build_geometric_transform(train=train, mean=mean, std=std)
        self.rng = np.random.default_rng(seed)

        self.bags = []
        for bag_id, group in df.groupby("bag_id"):
            self.bags.append(group.reset_index(drop=True))

    def __len__(self) -> int:
        return len(self.bags)

    def _select_rows(self, bag: pd.DataFrame) -> pd.DataFrame:
        if len(bag) <= MAX_BAG_SIZE:
            return bag
        if self.train:
            idx = self.rng.choice(len(bag), size=MAX_BAG_SIZE, replace=False)
        else:
            idx = np.arange(MAX_BAG_SIZE)
        return bag.iloc[sorted(idx)]

    def __getitem__(self, idx: int):
        bag = self._select_rows(self.bags[idx])
        first = bag.iloc[0]

        ga = (first["GA"] - self.norm_stats["ga_mean"]) / self.norm_stats["ga_std"]
        bw = (first["BW"] - self.norm_stats["bw_mean"]) / self.norm_stats["bw_std"]
        sexo = SEXO_TO_BINARY[first["sexo"]]
        pma_available = bool(first["pma_available"])
        if pma_available:
            pma = (first["PMA"] - self.norm_stats["pma_mean"]) / self.norm_stats["pma_std"]
        else:
            pma = 0.0  # imputado: z-score 0 = la media del subconjunto con dato real
        clinical = torch.tensor([ga, bw, sexo, pma, float(pma_available)], dtype=torch.float32)

        images = torch.zeros((MAX_BAG_SIZE, 3, 384, 384), dtype=torch.float32)
        bag_mask = torch.zeros(MAX_BAG_SIZE, dtype=torch.bool)
        for i, (_, row) in enumerate(bag.iterrows()):
            img_path = str(EXTRACTED_DIR / row["image_path"])
            images[i] = load_and_preprocess_image(img_path, self.transform)
            bag_mask[i] = True

        stage = first["stage"]
        stage_valid = pd.notna(stage)
        stage_t = torch.tensor(int(stage) if stage_valid else -1, dtype=torch.long)

        plus = first["plus"]
        plus_valid = pd.notna(plus)
        plus_t = torch.tensor(PLUS_TO_IDX[plus] if plus_valid else -1, dtype=torch.long)

        ap_rop = first["ap_rop"]
        ap_rop_valid = pd.notna(ap_rop)
        ap_rop_t = torch.tensor(float(ap_rop) if ap_rop_valid else 0.0, dtype=torch.float32)

        targets = {"stage": stage_t, "plus": plus_t, "ap_rop": ap_rop_t}
        valid = {
            "stage": torch.tensor(stage_valid), "plus": torch.tensor(plus_valid),
            "ap_rop": torch.tensor(ap_rop_valid),
        }
        return clinical, images, bag_mask, targets, valid
