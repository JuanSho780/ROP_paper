"""Limpia FARFUM-RoP por separado y escribe datasets/clean/FARFUM.parquet.

Mismo patron que clean_ridirp.py:
    crudo -> clean_<dataset>.py (este archivo) -> build_manifest.py (unificado)

Decisiones aplicadas aqui (detalle completo y verificacion con datos reales
en system/eda/eda_farfum.ipynb):

- `plus` viene directo de la columna "Label" del xlsx (col R, facil de
  perder por el encabezado de 2 filas) - es el consenso oficial del panel
  de 5 medicos, confirmado contra el paper original (Akbari et al. 2024,
  PMCID PMC11525552): 1=Normal, 2=Pre-Plus, 3=Plus. No se recalcula desde
  los `grade_X` individuales, seria redundante.
- `stage` y `treatment` (diagnostic) se calculan con consenso propio:
  severidad MAXIMA entre los raters que opinaron, SIN minimo de votos (no
  mayoria estricta) - un rater que se abstiene (`grade=0` = "no
  especificado", NO "normal") deja stage/diagnostic en blanco, y queda
  excluido del consenso automaticamente por eso, sin necesitar logica
  extra sobre `grade`.
- Como cada imagen puede tener un subconjunto distinto de raters
  opinando, el consenso por imagen puede variar dentro de un mismo
  paciente aunque el caso clinico sea uno solo (verificado: 9/9/3
  pacientes inconsistentes en stage/treatment/label antes de reconciliar).
  Se reconcilia a nivel de PACIENTE (bag) con el mismo criterio de
  severidad maxima, propagando el resultado a todas las imagenes del
  paciente - garantiza 0 inconsistencias dentro de un bag antes de que el
  dato llegue al modelo.
- No se elimina ninguna fila: a diferencia de RIDIRP, `plus` siempre tiene
  valor (via Label), asi que ninguna imagen de FARFUM queda con las 3
  tareas nulas a la vez.
- `stage=NaN` solo ocurre en 2 pacientes (patient26, patient28, 15
  imagenes) donde ningun rater dio un valor real de stage - se mantienen
  en el dataset (si tienen plus/treatment validos), solo se enmascaran en
  la tarea de Estadio.

Requiere: pandas, pyarrow, openpyxl. Ejecutar despues de extract_archives.py.
"""
from pathlib import Path

import pandas as pd
import openpyxl

ROOT = Path(__file__).resolve().parent.parent.parent  # system/dataset_build/ -> ROP_paper/
DATASETS = ROOT / "datasets"
RAW_DIR = DATASETS / "FARFUM_Original"
EXTRACTED_DIR = DATASETS / "extracted" / "FARFUM"
CLEAN_DIR = DATASETS / "clean"

TREAT_SEVERITY = {"revisit": 0, "no_treatment": 1, "treatment": 2}
TREAT_SEVERITY_INV = {v: k for k, v in TREAT_SEVERITY.items()}
LABEL_TO_PLUS = {1: "Normal", 2: "Pre-Plus", 3: "Plus"}
PLUS_SEVERITY = {"Normal": 0, "Pre-Plus": 1, "Plus": 2}
PLUS_SEVERITY_INV = {v: k for k, v in PLUS_SEVERITY.items()}

RATER_COLS = {
    "A": (2, 3, 4), "B": (5, 6, 7), "C": (8, 9, 10), "D": (11, 12, 13), "E": (14, 15, 16),
}
DIAG_NORMALIZE = {"teatment": "treatment"}


def rel(path: Path) -> str:
    return path.relative_to(DATASETS / "extracted").as_posix()


def load_details() -> pd.DataFrame:
    details = pd.read_excel(RAW_DIR / "Dataset_Details.xlsx")
    details.columns = ["patient_id", "bw", "ga", "gender"]
    return details.set_index("patient_id")


def load_labels() -> pd.DataFrame:
    wb = openpyxl.load_workbook(RAW_DIR / "Dataset_Labels.xlsx", data_only=True)
    ws = wb["Labels"]
    rows = list(ws.iter_rows(values_only=True))[2:]
    cols = [
        "patient_id", "image_name",
        "grade_A", "stage_A", "diag_A", "grade_B", "stage_B", "diag_B",
        "grade_C", "stage_C", "diag_C", "grade_D", "stage_D", "diag_D",
        "grade_E", "stage_E", "diag_E", "label",
    ]
    df = pd.DataFrame(rows, columns=cols)
    for name in RATER_COLS:
        df[f"diag_{name}"] = df[f"diag_{name}"].replace(DIAG_NORMALIZE)
    return df


def consensus_max_stage(values) -> float | None:
    valid = [v for v in values if pd.notna(v) and v != "?"]
    return max(valid) if valid else None


def consensus_max_treatment(values) -> str | None:
    valid = [v for v in values if pd.notna(v) and v != "?"]
    return max(valid, key=lambda v: TREAT_SEVERITY[v]) if valid else None


def main() -> None:
    details = load_details()
    df = load_labels()

    stage_cols = [f"stage_{n}" for n in RATER_COLS]
    diag_cols = [f"diag_{n}" for n in RATER_COLS]
    grade_cols = [f"grade_{n}" for n in RATER_COLS]

    df["stage_consensus"] = df[stage_cols].apply(lambda r: consensus_max_stage(list(r)), axis=1)
    df["treatment_consensus"] = df[diag_cols].apply(lambda r: consensus_max_treatment(list(r)), axis=1)
    df["stage_n_raters"] = df[stage_cols].apply(lambda r: sum(pd.notna(v) and v != "?" for v in r), axis=1)
    df["treatment_n_raters"] = df[diag_cols].apply(lambda r: sum(pd.notna(v) and v != "?" for v in r), axis=1)
    df["plus_raw"] = df["label"].map(LABEL_TO_PLUS)

    # reconciliacion a nivel de paciente (bag): gana el mas severo
    df["stage"] = df.groupby("patient_id")["stage_consensus"].transform("max")

    treat_rank = df["treatment_consensus"].map(TREAT_SEVERITY)
    treat_rank_bag = treat_rank.groupby(df["patient_id"]).transform("max")
    df["treatment"] = treat_rank_bag.map(TREAT_SEVERITY_INV)

    plus_rank = df["plus_raw"].map(PLUS_SEVERITY)
    plus_rank_bag = plus_rank.groupby(df["patient_id"]).transform("max")
    df["plus"] = plus_rank_bag.map(PLUS_SEVERITY_INV)

    rows = []
    missing_files = 0
    for _, r in df.iterrows():
        patient_id = r["patient_id"]
        image_name = r["image_name"]
        if patient_id is None or image_name is None:
            continue
        img_path = EXTRACTED_DIR / patient_id / f"{image_name}.jpg"
        if not img_path.exists():
            missing_files += 1
            continue

        patient_id_global = f"FARFUM_{patient_id}"
        detail = details.loc[patient_id] if patient_id in details.index else None
        rows.append(
            {
                "sample_id": f"{patient_id_global}_{image_name}",
                "bag_id": patient_id_global,
                "patient_id_global": patient_id_global,
                "image_path": rel(img_path),
                "sexo": None if detail is None else detail["gender"],
                "GA": None if detail is None else float(detail["ga"]),
                "BW": None if detail is None else float(detail["bw"]),
                "PMA": None,
                "stage": r["stage"],
                "plus": r["plus"],
                "treatment": r["treatment"],
                "stage_n_raters": int(r["stage_n_raters"]),
                "treatment_n_raters": int(r["treatment_n_raters"]),
                "raw_grade_per_rater": ";".join(str(r[c]) for c in grade_cols),
            }
        )

    if missing_files:
        print(f"[aviso] {missing_files} filas del xlsx sin archivo de imagen extraido (omitidas)")

    out = pd.DataFrame(rows)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CLEAN_DIR / "FARFUM.parquet"
    out.to_parquet(out_path, index=False)

    print(f"[FARFUM] {len(out)} imagenes, {out['bag_id'].nunique()} pacientes")
    print(f"[FARFUM] stage nulo: {out['stage'].isna().sum()} ({100*out['stage'].isna().mean():.1f}%)")
    print(f"[FARFUM] plus nulo:  {out['plus'].isna().sum()} ({100*out['plus'].isna().mean():.1f}%)")
    print(f"[FARFUM] treatment nulo: {out['treatment'].isna().sum()} ({100*out['treatment'].isna().mean():.1f}%)")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
