"""Asigna `split`/`fold` a datasets/manifest/rop_unified_manifest.parquet.

Tercera y ultima etapa del manifiesto clinico (ver README.md de esta
carpeta y README.md principal S2):
    clean/clean_<dataset>.py -> partition/build_manifest.py -> partition/group_kfold.py (este archivo)

Estrategia (acordada y documentada en system/eda/eda_unified.ipynb y en el
README principal S2 antes de escribir esto):

1. Agrupa por `patient_id_global` (nunca por bag ni por imagen) - ningun
   paciente queda partido entre splits/folds, evita fuga de anatomia real.
2. Estratifica, ademas de agrupar, usando una llave compuesta por paciente:
   `dataset_origen` + `stage_bucket` (peor estadio que tuvo ese paciente
   alguna vez, en 5 cubetas: sin_dato/0/1-2/3/4-5) + `has_plus` (alguna vez
   Plus o Pre-Plus) + `has_ap_rop` (alguna vez ap_rop=True). Verificado con
   datos reales: solo 20 categorias distintas resultan de 1.182 pacientes,
   manejable para StratifiedGroupKFold.
   Motivo: un reparto puramente aleatorio por paciente arriesga dejar un
   fold entero sin casos Plus (38 pacientes en todo el pool) o sin casos
   AP-ROP (102 pacientes, con buena parte concentrada en RIDIRP_043).
3. Primero se aparta un test set fijo (20% = 1 de 5 folds de un
   StratifiedGroupKFold inicial), nunca se vuelve a tocar. Despues se
   recalculan las categorias SOLO sobre el 80% restante y se corre un
   segundo StratifiedGroupKFold(n_splits=5) para el K-Fold real de
   entrenamiento (reentrenamiento final sobre todo train+val, evaluacion
   unica contra el test fijo - ver README principal S2).

Requiere: pandas, pyarrow, scikit-learn. Ejecutar despues de build_manifest.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent.parent.parent  # system/dataset_build/partition/ -> ROP_paper/
MANIFEST_PATH = ROOT / "datasets" / "manifest" / "rop_unified_manifest.parquet"

SEED = 42
N_FOLDS = 5


def bucket_stage(s: pd.Series) -> str:
    if s.isna().all():
        return "sin_dato"
    m = s.max()
    if m == 0:
        return "0"
    if m in (1, 2):
        return "1-2"
    if m == 3:
        return "3"
    return "4-5"


def build_patient_table(df: pd.DataFrame) -> pd.DataFrame:
    patients = df.groupby("patient_id_global").agg(
        dataset_origen=("dataset_origen", "first"),
        stage_bucket=("stage", bucket_stage),
        has_plus=("plus", lambda s: s.isin(["Plus", "Pre-Plus"]).any()),
        has_ap_rop=("ap_rop", lambda s: (s == True).any()),  # noqa: E712
    )
    patients["llave"] = (
        patients["dataset_origen"] + "_" + patients["stage_bucket"] + "_"
        + patients["has_plus"].astype(str) + "_" + patients["has_ap_rop"].astype(str)
    )
    return patients


def stratified_group_assign(patients: pd.DataFrame, n_splits: int, seed: int) -> pd.Series:
    """Devuelve una Series patient_id_global -> indice de fold (0..n_splits-1)."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X = np.zeros(len(patients))
    fold_of = pd.Series(index=patients.index, dtype="int64")
    for fold_idx, (_, test_idx) in enumerate(
        sgkf.split(X, patients["llave"], groups=patients.index)
    ):
        fold_of.iloc[test_idx] = fold_idx
    return fold_of


def main() -> None:
    df = pd.read_parquet(MANIFEST_PATH)
    patients = build_patient_table(df)

    # --- paso 1: test set fijo (1 de 5 = 20%) ---
    initial_fold = stratified_group_assign(patients, N_FOLDS, SEED)
    is_test = initial_fold == 0
    test_patients = patients.index[is_test]
    train_pool_patients = patients.loc[~is_test]

    print(f"[test] {len(test_patients)} pacientes ({100*len(test_patients)/len(patients):.1f}%)")

    # --- paso 2: K-Fold real (K=5) sobre el 80% restante, categorias recalculadas ---
    cv_fold = stratified_group_assign(train_pool_patients, N_FOLDS, SEED)

    split_map = pd.Series("train", index=patients.index)
    split_map.loc[test_patients] = "test"

    fold_map = pd.Series(-1, index=patients.index, dtype="int64")
    fold_map.loc[cv_fold.index] = cv_fold.values

    df["split"] = df["patient_id_global"].map(split_map)
    df["fold"] = df["patient_id_global"].map(fold_map)

    df.to_parquet(MANIFEST_PATH, index=False)

    # --- validacion ---
    bad_groups = df.groupby("patient_id_global")[["split", "fold"]].nunique()
    inconsistent = ((bad_groups["split"] > 1) | (bad_groups["fold"] > 1)).sum()
    print(f"[validacion] pacientes con split/fold inconsistente entre sus filas: {inconsistent}")

    print(f"\n[split] filas por split:\n{df['split'].value_counts()}")
    print(f"\n[fold] pacientes del pool de train por fold:\n{patients.loc[train_pool_patients.index].assign(fold=cv_fold)['fold'].value_counts().sort_index()}")

    print("\n[balance] pacientes con has_plus/has_ap_rop por fold (y test):")
    patients_with_split = patients.copy()
    patients_with_split["split"] = split_map
    patients_with_split["fold"] = fold_map
    resumen = patients_with_split.groupby(["split", "fold"]).agg(
        n_pacientes=("dataset_origen", "size"),
        n_plus=("has_plus", "sum"),
        n_ap_rop=("has_ap_rop", "sum"),
    )
    print(resumen)

    print(f"\n-> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
