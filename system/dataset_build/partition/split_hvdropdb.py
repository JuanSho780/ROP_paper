"""Lee datasets/clean/HVDROPDB.parquet, asigna split train/val (Fase 1) y
escribe el resultado en datasets/manifest/HVDROPDB.parquet.

Script separado de clean_hvdropdb.py a proposito (ver su docstring y la
discusion en README.md S2 / S5): HVDROPDB no participa del Group K-Fold
clinico (no tiene patient_id, no tiene las etiquetas del modelo final) - es
un split unico interno, solo para monitorear el fine-tuning del ViT en
Fase 1, agrupado por `image_hash` en vez de paciente.

Vive en partition/ (no en clean/) porque no construye el parquet limpio -
toma el que ya existe en datasets/clean/ y produce la version final, lista
para entrenar, en datasets/manifest/ (mismo destino conceptual que el
futuro rop_unified_manifest.parquet de build_manifest.py).

Por que agrupar por hash: se verifico (clean_hvdropdb.py, columna
`image_hash`) que 38 de las 300 filas comparten la foto fuente exacta con
otra fila de distinta estructura/camara (25 BV/RetCam<->OD/RetCam, 10
BV/Neo<->RIDGE/Neo, 3 BV/RetCam<->RIDGE/RetCam). Si el split fuera aleatorio
por fila, esas 38 fotos podrian caer en train para una estructura y en val
para otra - el backbone ya habria visto esa foto en entrenamiento, inflando
el Dice de validacion. Agrupando por hash, cada foto (y todas sus filas)
cae siempre del mismo lado.

No es K-Fold (un solo split, no se rota ni se promedia) - ver README.md S2
para el razonamiento de por que Fase 1 no necesita cross-validation (es una
etapa auxiliar, no el resultado que se reporta en la tesis).

Requiere: pandas, numpy. Ejecutar despues de clean_hvdropdb.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent  # system/dataset_build/partition/ -> ROP_paper/
CLEAN_DIR = ROOT / "datasets" / "clean"
MANIFEST_DIR = ROOT / "datasets" / "manifest"

VAL_FRACTION = 0.2
SEED = 42


def main() -> None:
    df = pd.read_parquet(CLEAN_DIR / "HVDROPDB.parquet")

    groups = df.groupby("image_hash").size().rename("n_filas").reset_index()
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(groups))

    target_val = int(round(len(df) * VAL_FRACTION))
    val_hashes = set()
    running = 0
    for idx in order:
        row = groups.iloc[idx]
        if running >= target_val:
            break
        val_hashes.add(row["image_hash"])
        running += row["n_filas"]

    df["split"] = np.where(df["image_hash"].isin(val_hashes), "val", "train")

    # verificacion: ninguna foto (hash) debe aparecer en ambos splits
    split_per_hash = df.groupby("image_hash")["split"].nunique()
    inconsistent = int((split_per_hash > 1).sum())

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MANIFEST_DIR / "HVDROPDB.parquet"
    df.to_parquet(out_path, index=False)

    counts = df["split"].value_counts()
    print(f"[HVDROPDB] train: {counts.get('train', 0)} filas, val: {counts.get('val', 0)} filas")
    print(f"[HVDROPDB] fotos (hash) en val: {len(val_hashes)} de {len(groups)}")
    print(f"[HVDROPDB] fotos con split inconsistente entre filas: {inconsistent}")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
