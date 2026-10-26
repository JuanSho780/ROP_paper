"""Limpia RIDIRP por separado y escribe datasets/clean/RIDIRP.parquet.

Primera etapa de la nueva arquitectura de 3 pasos:
    crudo -> clean_<dataset>.py (este archivo) -> build_manifest.py (unificado)

Decisiones aplicadas aqui (documentadas con el detalle completo, incluyendo
el porque de cada una, en system/eda/eda_ridirp.ipynb, celda final
"Decisiones de diseno discutidas"):

- El DG (diagnosis code) se unifica a `stage` 0-5. AP-ROP y status-post-ROP
  quedan con stage=NaN (no son reducibles a la escala ordinal) pero SI
  conservan `plus` (la medicion de Plus sigue siendo valida para ellos).
- Las filas con diagnostico no-ROP (hemorragia, hamartoma, hipoplasia,
  toxoplasmosis) se ELIMINAN por completo, no se enmascaran: quedarian con
  stage=NaN, plus=NaN y treatment=NaN simultaneamente, es decir no aportan
  gradiente a ningun cabezal clinico.
- La columna `series` del xlsx esta mal asignada para ~31 filas del
  dispositivo 3 (numero repetido en vez de incremental) - el cruce con los
  archivos de imagen se hace alineando por PMA ascendente (rank
  cronologico), no por el valor literal de esa columna.
- `device` se conserva como metadata (`device_raw`) en este parquet limpio
  individual (sirve para un futuro analisis leave-one-device-out), pero NO
  es un feature de entrada del modelo y se descarta en el parquet unificado.
- `treatment` queda siempre None (RIDIRP no tiene esta etiqueta) - se
  mantiene la columna vacia por consistencia de esquema con los otros 3
  datasets limpios (FARFUM, ROP-VL, HVDROPDB no aplica), para que la union
  final sea un concat simple.

Requiere: pandas, pyarrow, openpyxl. Ejecutar despues de extract_archives.py.
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent  # system/dataset_build/ -> ROP_paper/
DATASETS = ROOT / "datasets"
RAW_DIR = DATASETS / "RIDIRP_Original"
EXTRACTED_DIR = DATASETS / "extracted" / "RIDIRP" / "images_stack_without_captions"
CLEAN_DIR = DATASETS / "clean"

DG_TO_STAGE = {0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 4, 7: 5}
AP_ROP_CODES = {8}
STATUS_POST_CODES = {9}
NON_ROP_CODES = {10, 11, 12, 13}  # hamartoma, hemorragia, hipoplasia N.II, toxoplasma
PF_TO_PLUS = {0: "Normal", 1: "Pre-Plus", 2: "Plus"}

FILENAME_RE = re.compile(
    r"^(?P<id>\d+)_(?P<sex>[MF])_GA(?P<ga>\d+)_BW(?P<bw>\d+)_PA(?P<pa>\d+)"
    r"_DG(?P<dg>\d+)_PF(?P<pf>\d+)_D(?P<device>\d+)_S(?P<series>\d+)_(?P<idx>\d+)\.jpg$"
)


def rel(path: Path) -> str:
    return path.relative_to(DATASETS / "extracted").as_posix()


def load_xlsx() -> pd.DataFrame:
    db = pd.read_excel(RAW_DIR / "infant_retinal_database_info.xlsx", sheet_name="database")
    db.columns = ["id", "sex", "ga", "bw", "pma", "dg", "pf", "device", "series"]
    return db


def build_join_table(xlsx: pd.DataFrame) -> dict:
    """Mapea (patient_id, rank_cronologico) -> fila del xlsx.

    La columna "series" del xlsx esta mal asignada para varios pacientes del
    dispositivo 3 (confirmado en el EDA). El rank cronologico (PMA
    ascendente) si coincide 1 a 1 con el numero de serie real de los
    archivos para los 188 pacientes - es la llave de union correcta.
    """
    table = {}
    for pid, group in xlsx.groupby("id"):
        ordered = group.sort_values("pma").reset_index(drop=True)
        for rank, row in ordered.iterrows():
            table[(pid, rank + 1)] = row
    return table


def main() -> None:
    xlsx = load_xlsx()
    join_table = build_join_table(xlsx)

    series_by_patient: dict[int, set[int]] = {}
    for p in EXTRACTED_DIR.glob("*.jpg"):
        m = FILENAME_RE.match(p.name)
        if m:
            series_by_patient.setdefault(int(m.group("id")), set()).add(int(m.group("series")))
    series_rank = {
        pid: {s: rank + 1 for rank, s in enumerate(sorted(s_set))}
        for pid, s_set in series_by_patient.items()
    }

    rows = []
    dropped_non_rop = 0
    missing_join = 0
    for img_path in sorted(EXTRACTED_DIR.glob("*.jpg")):
        m = FILENAME_RE.match(img_path.name)
        if not m:
            continue
        pid, series = int(m.group("id")), int(m.group("series"))
        rank = series_rank[pid][series]
        xr = join_table.get((pid, rank))
        if xr is None:
            missing_join += 1
            continue

        dg = int(xr["dg"])
        if dg in NON_ROP_CODES:
            dropped_non_rop += 1
            continue

        patient_id_global = f"RIDIRP_{pid:03d}"
        rows.append(
            {
                "sample_id": f"{patient_id_global}_S{series:02d}_{m.group('idx')}",
                "bag_id": f"{patient_id_global}_S{series:02d}",
                "patient_id_global": patient_id_global,
                "image_path": rel(img_path),
                "sexo": xr["sex"],
                "GA": float(xr["ga"]),
                "BW": float(xr["bw"]),
                "PMA": float(xr["pma"]),
                "stage": DG_TO_STAGE.get(dg),
                "plus": PF_TO_PLUS.get(int(xr["pf"])),
                "treatment": None,
                "ap_rop": dg in AP_ROP_CODES,
                "status_post_rop": dg in STATUS_POST_CODES,
                "device_raw": int(xr["device"]),
                "raw_diagnosis_code": str(dg),
            }
        )

    if missing_join:
        print(f"[aviso] {missing_join} imagenes sin fila correspondiente en el xlsx (omitidas)")
    print(f"[RIDIRP] {dropped_non_rop} filas eliminadas por diagnostico no-ROP")

    df = pd.DataFrame(rows)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CLEAN_DIR / "RIDIRP.parquet"
    df.to_parquet(out_path, index=False)

    print(f"[RIDIRP] {len(df)} imagenes, {df['bag_id'].nunique()} bags, {df['patient_id_global'].nunique()} pacientes")
    print(f"[RIDIRP] stage nulo: {df['stage'].isna().sum()} ({100*df['stage'].isna().mean():.1f}%)")
    print(f"[RIDIRP] plus nulo:  {df['plus'].isna().sum()} ({100*df['plus'].isna().mean():.1f}%)")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
