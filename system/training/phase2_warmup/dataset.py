"""Dataset de Fase 2: arma bags (MIL) desde rop_unified_manifest.parquet,
usando los tokens YA CACHEADOS por precompute_tokens.py (backbone congelado,
sin augmentation de imagen - ver docstring de ese script).

Granularidad: un bag = una visita/sesion (bag_id), puede tener de 1 a 46
imagenes (verificado con datos reales: ROP-VL casi siempre 1-3, FARFUM
promedia ~22 y llega a 46). Se limita a MAX_BAG_SIZE=8 imagenes por bag -
decision acordada en el chat de planeacion: sin limite, un solo bag grande
en un batch obliga a rellenar TODOS los demas bags del batch hasta su
tamano, desperdiciando computo/memoria de forma impredecible. En train, el
subconjunto de 8 se resamplea al azar en cada __getitem__ (distinto por
epoca - a lo largo del entrenamiento el modelo ve casi todas las imagenes
de los bags grandes, solo nunca las 46 juntas). En val, la seleccion es
FIJA (determinista) para que la metrica de validacion no fluctue por puro
azar de que imagenes se sortearon.

Como cada bag queda con exactamente MAX_BAG_SIZE posiciones (reales +
padding), el collate por defecto de PyTorch alcanza - no hace falta
collate_fn custom.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent.parent.parent.parent  # system/training/phase2_warmup/ -> ROP_paper/
MANIFEST_PATH = ROOT / "datasets" / "manifest" / "rop_unified_manifest.parquet"
CACHE_DIR = Path(__file__).resolve().parent / "cached_tokens"

sys.path.insert(0, str(ROOT / "system" / "training"))
from common.clinical_features import PLUS_TO_IDX, SEXO_TO_BINARY  # noqa: E402

MAX_BAG_SIZE = 8
N_TOKENS = 576
TOKEN_DIM = 768


def load_phase2_rows() -> pd.DataFrame:
    df = pd.read_parquet(MANIFEST_PATH)
    return df[df["dataset_origen"].isin(["ROP-VL", "FARFUM"])].reset_index(drop=True)


def compute_clinical_norm_stats(train_df: pd.DataFrame) -> dict:
    """Media/desvio de GA y BW calculados SOLO del train del fold (nunca de
    val/test - evita fuga de estadisticas entre splits)."""
    bags = train_df.drop_duplicates("bag_id")
    return {
        "ga_mean": bags["GA"].mean(), "ga_std": bags["GA"].std(),
        "bw_mean": bags["BW"].mean(), "bw_std": bags["BW"].std(),
    }


def compute_class_weights(train_df: pd.DataFrame) -> dict:
    """Pesos de clase (inversa de frecuencia, normalizados) desde el train
    del fold - para Focal Loss (stage) y CE ponderada (plus)."""
    bags = train_df.drop_duplicates("bag_id")

    stage_counts = bags["stage"].dropna().astype(int).value_counts().reindex(range(6), fill_value=0) + 1  # +1 suaviza
    stage_alpha = (1.0 / stage_counts)
    stage_alpha = (stage_alpha / stage_alpha.sum() * 6).values  # normalizado, promedio 1.0

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


class WarmupBagDataset(Dataset):
    def __init__(self, df: pd.DataFrame, norm_stats: dict, train: bool, seed: int = 42):
        self.train = train
        self.norm_stats = norm_stats
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
            idx = np.arange(MAX_BAG_SIZE)  # seleccion fija y determinista para validacion
        return bag.iloc[sorted(idx)]

    def __getitem__(self, idx: int):
        bag = self._select_rows(self.bags[idx])
        first = bag.iloc[0]

        ga = (first["GA"] - self.norm_stats["ga_mean"]) / self.norm_stats["ga_std"]
        bw = (first["BW"] - self.norm_stats["bw_mean"]) / self.norm_stats["bw_std"]
        sexo = SEXO_TO_BINARY[first["sexo"]]
        clinical = torch.tensor([ga, bw, sexo], dtype=torch.float32)

        visual = torch.zeros((MAX_BAG_SIZE, N_TOKENS, TOKEN_DIM), dtype=torch.float32)
        bag_mask = torch.zeros(MAX_BAG_SIZE, dtype=torch.bool)
        for i, (_, row) in enumerate(bag.iterrows()):
            tokens = torch.load(CACHE_DIR / f"{row['sample_id']}.pt").float()
            visual[i] = tokens
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
        return clinical, visual, bag_mask, targets, valid
