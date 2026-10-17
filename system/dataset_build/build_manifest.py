"""Construye el manifiesto unificado (Parquet) a partir de los datasets ya
extraidos en datasets/extracted/.

Genera dos archivos en datasets/manifest/:
  - rop_unified_manifest.parquet   -> RIDIRP + FARFUM + ROP-VL (tareas clinicas)
  - hvdropdb_segmentation_manifest.parquet -> pares imagen/mascara (Fase 1)

Las decisiones de mapeo (que codigo de diagnostico cuenta como que estadio,
que se deja NaN, etc.) estan documentadas en README.md - revisar ahi antes
de confiar en las columnas `stage`/`plus`/`treatment`.

Requiere: pandas, pyarrow, openpyxl. Ejecutar despues de extract_archives.py.
"""
import re
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent  # system/dataset_build/ -> ROP_paper/
DATASETS = ROOT / "datasets"
EXTRACTED = DATASETS / "extracted"
MANIFEST_DIR = DATASETS / "manifest"


def rel(path: Path) -> str:
    """Ruta relativa a datasets/extracted/, con separadores '/' (portable)."""
    return path.relative_to(EXTRACTED).as_posix()


# ---------------------------------------------------------------------------
# RIDIRP
# ---------------------------------------------------------------------------

# DG (diagnosis code) -> estadio ordinal 0-5. Ver README §RIDIRP para la
# justificacion de cada caso especial (AP-ROP, status-post, no-ROP).
RIDIRP_DG_TO_STAGE = {0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 4, 7: 5}
RIDIRP_AP_ROP_CODES = {8}
RIDIRP_STATUS_POST_CODES = {9}
RIDIRP_NON_ROP_CODES = {10, 11, 12, 13}  # hamartoma, hemorragia, hipoplasia, toxoplasma

RIDIRP_PF_TO_PLUS = {0: "Normal", 1: "Pre-Plus", 2: "Plus"}

RIDIRP_FILENAME_RE = re.compile(
    r"^(?P<id>\d+)_(?P<sex>[MF])_GA(?P<ga>\d+)_BW(?P<bw>\d+)_PA(?P<pa>\d+)"
    r"_DG(?P<dg>\d+)_PF(?P<pf>\d+)_D(?P<device>\d+)_S(?P<series>\d+)_(?P<idx>\d+)\.jpg$"
)


def load_ridirp_xlsx():
    xlsx_path = DATASETS / "RIDIRP_Original" / "infant_retinal_database_info.xlsx"
    db = pd.read_excel(xlsx_path, sheet_name="database")
    db.columns = ["id", "sex", "ga", "bw", "pma", "dg", "pf", "device", "series"]
    return db


def _ridirp_join_table(xlsx: pd.DataFrame) -> dict:
    """Mapea (patient_id, series_en_archivo) -> fila del xlsx.

    La columna "series" del xlsx esta mal asignada para varios pacientes del
    dispositivo 3 (series repetida en vez de incremental - confirmado en
    ~31 filas). El numero de serie en el NOMBRE DE ARCHIVO si es correcto y
    coincide 1 a 1 en cantidad con las filas del xlsx para los 188 pacientes
    (verificado). Por eso la alineacion real se hace por orden cronologico
    (PMA ascendente) y no por el valor de la columna "series" - ver README
    seccion RIDIRP para el detalle.
    """
    table = {}
    for pid, group in xlsx.groupby("id"):
        ordered = group.sort_values("pma").reset_index(drop=True)
        for rank, row in ordered.iterrows():
            table[(pid, rank + 1)] = row
    return table


def build_ridirp_rows():
    xlsx = load_ridirp_xlsx()
    join_table = _ridirp_join_table(xlsx)
    img_dir = EXTRACTED / "RIDIRP" / "images_stack_without_captions"

    # orden de series realmente presentes por paciente (segun archivos),
    # para mapear el numero de serie del archivo -> rango cronologico (1..N)
    import collections
    series_by_patient = collections.defaultdict(set)
    for img_path in img_dir.glob("*.jpg"):
        m = RIDIRP_FILENAME_RE.match(img_path.name)
        if m:
            series_by_patient[int(m.group("id"))].add(int(m.group("series")))
    series_rank = {
        pid: {s: rank + 1 for rank, s in enumerate(sorted(series_set))}
        for pid, series_set in series_by_patient.items()
    }

    rows = []
    missing_xlsx_match = 0
    for img_path in sorted(img_dir.glob("*.jpg")):
        m = RIDIRP_FILENAME_RE.match(img_path.name)
        if not m:
            continue
        pid, series = int(m.group("id")), int(m.group("series"))
        rank = series_rank[pid][series]
        xr = join_table.get((pid, rank))
        if xr is None:
            missing_xlsx_match += 1
            continue
        dg = int(xr["dg"])
        is_rop_eligible = dg not in RIDIRP_NON_ROP_CODES
        stage = RIDIRP_DG_TO_STAGE.get(dg) if is_rop_eligible else None
        plus = RIDIRP_PF_TO_PLUS.get(int(xr["pf"])) if is_rop_eligible else None
        patient_id_global = f"RIDIRP_{pid:03d}"
        rows.append(
            {
                "sample_id": f"{patient_id_global}_S{series:02d}_{m.group('idx')}",
                "bag_id": f"{patient_id_global}_S{series:02d}",
                "dataset_origen": "RIDIRP",
                "patient_id_global": patient_id_global,
                "image_path": rel(img_path),
                "sexo": xr["sex"],
                "GA": float(xr["ga"]),
                "BW": float(xr["bw"]),
                "PMA": float(xr["pma"]),
                "pma_available": True,
                "stage": stage,
                "plus": plus,
                "treatment": None,
                "ap_rop": dg in RIDIRP_AP_ROP_CODES,
                "status_post_rop": dg in RIDIRP_STATUS_POST_CODES,
                "is_rop_eligible": is_rop_eligible,
                "laser_treated": False,
                "raw_diagnosis_code": str(dg),
            }
        )
    if missing_xlsx_match:
        print(f"[RIDIRP] aviso: {missing_xlsx_match} imagenes sin fila correspondiente en el xlsx (omitidas)")
    print(f"[RIDIRP] {len(rows)} imagenes")
    return rows


# ---------------------------------------------------------------------------
# FARFUM-RoP
# ---------------------------------------------------------------------------

FARFUM_DIAGNOSTIC_NORMALIZE = {"teatment": "treatment"}

# Columna "Label" (R) del xlsx: target oficial de severidad, con cobertura
# completa (1533/1533, sin nulos) y exactamente 3 valores -- a diferencia de
# "grade" (0-3, semantica sin confirmar) y de nuestro propio consenso por
# mayoria (cobertura baja). Se infiere la direccion 1->Normal, 3->Plus por
# correlacion con "diagnostic" (Label=3 casi nunca coincide con "revisit",
# Label=1 rara vez con "treatment") - no viene explicito en el archivo, pero
# es consistente con la descripcion oficial del dataset en Figshare
# ("anotado como Normal/Pre-Plus/Plus"). Ver README seccion FARFUM.
FARFUM_LABEL_TO_PLUS = {1: "Normal", 2: "Pre-Plus", 3: "Plus"}


def _majority_vote(values, min_votes=3):
    # pd.notna() (no "is not None"): pandas representa celda vacia como
    # NaN (float), y "nan is not None" es True -- con "is not None" esos
    # blancos entraban al conteo de votos como si fueran una respuesta real.
    values = [v for v in values if pd.notna(v) and v != "?"]
    if len(values) < min_votes:
        return None
    counts = Counter(values)
    value, n = counts.most_common(1)[0]
    if n * 2 <= len(values):  # exige mayoria estricta, no solo moda
        return None
    return value


def build_farfum_rows():
    details = pd.read_excel(DATASETS / "FARFUM_Original" / "Dataset_Details.xlsx")
    details.columns = ["patient_id", "bw", "ga", "gender"]
    details = details.set_index("patient_id")

    labels = pd.read_excel(
        DATASETS / "FARFUM_Original" / "Dataset_Labels.xlsx", sheet_name="Labels", header=1
    )
    # columnas: Patient.id(0), Image name(1), grade/stage/diagnostic x5 raters (2..16), Label(17)
    rater_cols = {
        "A": (2, 3, 4), "B": (5, 6, 7), "C": (8, 9, 10), "D": (11, 12, 13), "E": (14, 15, 16),
    }

    img_dir = EXTRACTED / "FARFUM"
    rows = []
    missing_files = 0
    for _, r in labels.iterrows():
        patient_id = r.iloc[0]
        image_name = r.iloc[1]
        if patient_id is None or image_name is None:
            continue
        img_path = img_dir / patient_id / f"{image_name}.jpg"
        if not img_path.exists():
            missing_files += 1
            continue

        stages, diagnostics, grades = [], [], []
        for _, (gi, si, di) in rater_cols.items():
            grades.append(r.iloc[gi])
            stages.append(r.iloc[si])
            diag = r.iloc[di]
            if isinstance(diag, str):
                diag = FARFUM_DIAGNOSTIC_NORMALIZE.get(diag, diag)
            diagnostics.append(diag)

        stage_consensus = _majority_vote(stages)
        treatment_consensus = _majority_vote(diagnostics)
        label_raw = r.iloc[17]  # columna "Label", target oficial (ver constante arriba)
        plus = FARFUM_LABEL_TO_PLUS.get(int(label_raw)) if pd.notna(label_raw) else None

        patient_id_global = f"FARFUM_{patient_id}"
        detail = details.loc[patient_id] if patient_id in details.index else None
        rows.append(
            {
                "sample_id": f"{patient_id_global}_{image_name}",
                "bag_id": patient_id_global,
                "dataset_origen": "FARFUM",
                "patient_id_global": patient_id_global,
                "image_path": rel(img_path),
                "sexo": ("boy" if detail is None else detail["gender"]),
                "GA": None if detail is None else float(detail["ga"]),
                "BW": None if detail is None else float(detail["bw"]),
                "PMA": None,
                "pma_available": False,
                "stage": stage_consensus,
                "plus": plus,
                "treatment": treatment_consensus,
                "ap_rop": False,
                "status_post_rop": False,
                "is_rop_eligible": True,
                "laser_treated": False,
                "raw_diagnosis_code": (None if pd.isna(label_raw) else str(int(label_raw))),
                "raw_grade_per_rater": ";".join(str(g) for g in grades),
            }
        )
    if missing_files:
        print(f"[FARFUM] aviso: {missing_files} filas del xlsx sin archivo de imagen extraido (omitidas)")
    print(f"[FARFUM] {len(rows)} imagenes")
    return rows


# ---------------------------------------------------------------------------
# ROP-VL
# ---------------------------------------------------------------------------

ROPVL_CATEGORY_TO_STAGE = {
    "normal": 0,
    "Stage 1 ROP": 1,
    "Stage 2 ROP": 2,
    "Stage 3 ROP": 3,
    "Stage 4 ROP": 4,
    "Stage 5 ROP": 5,
}
ROPVL_AP_ROP_CATEGORY = "A-ROP"
ROPVL_LASER_CATEGORY = "Laser-treated ROP"

ROPVL_FILENAME_RE = re.compile(r"^(?P<pid>\d+)_(?P<date>[\d-]+)_(?P<eye>[LR])_(?P<idx>\d+)\.jpg$")


def build_ropvl_rows():
    xlsx = pd.read_excel(EXTRACTED / "ROP-VL" / "img_info.xlsx")
    xlsx.columns = ["image", "patient_id", "gender", "ga", "bw", "plurality", "delivery_mode", "category"]

    # indice filename -> ruta real extraida (las imagenes viven en subcarpetas por categoria)
    img_index = {p.name: p for p in (EXTRACTED / "ROP-VL" / "Image").rglob("*.jpg")}

    rows = []
    missing_files = 0
    for _, r in xlsx.iterrows():
        fname = r["image"]
        img_path = img_index.get(fname)
        if img_path is None:
            missing_files += 1
            continue
        m = ROPVL_FILENAME_RE.match(fname)
        date, eye = (m.group("date"), m.group("eye")) if m else (None, None)

        category = r["category"]
        is_ap_rop = category == ROPVL_AP_ROP_CATEGORY
        is_laser = category == ROPVL_LASER_CATEGORY
        stage = ROPVL_CATEGORY_TO_STAGE.get(category)  # None para A-ROP y Laser-treated

        patient_id_global = f"ROPVL_{r['patient_id']}"
        bag_id = f"{patient_id_global}_{date}_{eye}" if date and eye else f"{patient_id_global}_{fname}"
        rows.append(
            {
                "sample_id": f"{patient_id_global}_{fname}",
                "bag_id": bag_id,
                "dataset_origen": "ROP-VL",
                "patient_id_global": patient_id_global,
                "image_path": rel(img_path),
                "sexo": r["gender"],
                "GA": float(r["ga"]) if pd.notna(r["ga"]) else None,
                "BW": float(r["bw"]) if pd.notna(r["bw"]) else None,
                "PMA": None,
                "pma_available": False,
                "stage": stage,
                "plus": None,
                "treatment": None,
                "ap_rop": is_ap_rop,
                "status_post_rop": False,
                "is_rop_eligible": True,
                "laser_treated": is_laser,
                "raw_diagnosis_code": category,
            }
        )
    if missing_files:
        print(f"[ROP-VL] aviso: {missing_files} filas del xlsx sin archivo de imagen extraido (omitidas)")
    print(f"[ROP-VL] {len(rows)} imagenes")
    return rows


# ---------------------------------------------------------------------------
# HVDROPDB (Fase 1 - segmentacion, manifiesto separado)
# ---------------------------------------------------------------------------

def build_hvdropdb_rows():
    seg_root = next((EXTRACTED / "HVDROPDB").glob("HVDROPDB_RetCam_Neo_Segmentation"), None)
    if seg_root is None:
        print("[HVDROPDB] no encontrado, se omite")
        return []
    rows = []
    for structure_dir in sorted(seg_root.glob("HVDROPDB-*")):
        structure = structure_dir.name.replace("HVDROPDB-", "")  # BV, OD, RIDGE
        for images_dir in sorted(structure_dir.glob("*_images")):
            camera = "Neo" if images_dir.name.startswith("Neo") else "RetCam"
            masks_dir = images_dir.parent / images_dir.name.replace("_images", "_masks")
            for img_path in sorted(images_dir.glob("*.png")):
                mask_path = masks_dir / img_path.name
                if not mask_path.exists():
                    continue
                rows.append(
                    {
                        "pair_id": f"HVDROPDB_{structure}_{camera}_{img_path.stem}",
                        "structure": structure,
                        "camera": camera,
                        "image_path": rel(img_path),
                        "mask_path": rel(mask_path),
                    }
                )
    print(f"[HVDROPDB] {len(rows)} pares imagen/mascara")
    return rows


PLUS_SEVERITY = {"Normal": 0, "Pre-Plus": 1, "Plus": 2}
PLUS_SEVERITY_INV = {v: k for k, v in PLUS_SEVERITY.items()}


def reconcile_bag_plus(df: pd.DataFrame) -> pd.DataFrame:
    """Fuerza un unico valor de `plus` por bag_id (mismo criterio que
    reconcile_bag_stage: el hallazgo mas severo gana).

    Encontrado en los datos reales: 3 de los 68 pacientes de FARFUM
    (patient44, patient55, patient57) tienen la columna "Label" (ya mapeada
    a plus) con 2 valores distintos entre sus propias imagenes, siempre en
    categorias adyacentes (Normal/Pre-Plus o Pre-Plus/Plus) - mismo patron
    que el hallazgo de `stage` en ROP-VL. Se resuelve igual: el paciente se
    queda con el Plus mas severo detectado en cualquiera de sus imagenes.
    """
    sev = df["plus"].map(PLUS_SEVERITY)
    bag_max_sev = sev.groupby(df["bag_id"]).transform("max")
    changed = sev.notna() & bag_max_sev.notna() & (sev != bag_max_sev)
    if changed.any():
        print(
            f"[reconcile] {changed.sum()} filas de plus ajustadas al mas severo de su bag "
            f"({df.loc[changed, 'bag_id'].nunique()} bags)"
        )
    df.loc[bag_max_sev.notna(), "plus"] = bag_max_sev[bag_max_sev.notna()].map(PLUS_SEVERITY_INV)
    return df


TREATMENT_SEVERITY = {"revisit": 0, "no_treatment": 1, "treatment": 2}
TREATMENT_SEVERITY_INV = {v: k for k, v in TREATMENT_SEVERITY.items()}


def reconcile_bag_treatment(df: pd.DataFrame) -> pd.DataFrame:
    """Mismo criterio que reconcile_bag_plus, para `treatment` en FARFUM:
    el bag se queda con la decision mas urgente (treatment > no_treatment >
    revisit) si hay desacuerdo entre sus imagenes. Orden de severidad
    inferido (no viene declarado en el xlsx), ver README seccion FARFUM.
    """
    sev = df["treatment"].map(TREATMENT_SEVERITY)
    bag_max_sev = sev.groupby(df["bag_id"]).transform("max")
    changed = sev.notna() & bag_max_sev.notna() & (sev != bag_max_sev)
    if changed.any():
        print(
            f"[reconcile] {changed.sum()} filas de treatment ajustadas a la mas urgente de su bag "
            f"({df.loc[changed, 'bag_id'].nunique()} bags)"
        )
    df.loc[bag_max_sev.notna(), "treatment"] = bag_max_sev[bag_max_sev.notna()].map(TREATMENT_SEVERITY_INV)
    return df


def reconcile_bag_stage(df: pd.DataFrame) -> pd.DataFrame:
    """Fuerza un unico valor de `stage` por bag_id.

    Encontrado en los datos reales (no teorico): ~91 bags de ROP-VL (4.9%)
    tienen imagenes del mismo ojo+fecha con `categories` de estadio
    distintas (ej. una toma de polo posterior "normal", otra toma
    periferica "Stage 2 ROP"). Se resuelven tomando el MAXIMO: el hallazgo
    mas avanzado visible en cualquier imagen define el estadio clinico real
    del ojo ese dia (es la misma logica de MIL: basta una instancia
    positiva para que la bolsa sea positiva).

    Aparte, 52 bags mezclan una imagen A-ROP o Laser-treated ROP (que no
    tienen estadio ordinal 0-5 por diseno, stage=NaN) con otra imagen del
    mismo ojo que si tiene estadio numerico o se ve "normal". Ahi NO se
    aplica el maximo -- asumir "normal" solo porque otra foto del mismo ojo
    se ve sana seria un error clinico (un ojo con laser ya tuvo ROP severa).
    Se deja `stage=NaN` para todo el bag, y se marcan `ap_rop`/
    `laser_treated` a nivel de bag. Ver README seccion ROP-VL.
    """
    has_special = (
        df.groupby("bag_id")["ap_rop"].transform("any")
        | df.groupby("bag_id")["laser_treated"].transform("any")
    )
    bag_max_stage = df.groupby("bag_id")["stage"].transform("max")

    resolved_by_max = (~has_special) & df["stage"].notna() & (df["stage"] != bag_max_stage)
    if resolved_by_max.any():
        print(
            f"[reconcile] {resolved_by_max.sum()} filas ajustadas al estadio maximo de su bag "
            f"({df.loc[resolved_by_max, 'bag_id'].nunique()} bags, sin categoria especial)"
        )

    blocked_by_special = has_special & df["stage"].notna()
    if blocked_by_special.any():
        print(
            f"[reconcile] {blocked_by_special.sum()} filas con stage puesto a NaN por mezclar "
            f"categoria especial (A-ROP/Laser-treated) con otra etiqueta en el mismo bag "
            f"({df.loc[blocked_by_special, 'bag_id'].nunique()} bags)"
        )

    df.loc[has_special, "stage"] = None
    df.loc[~has_special, "stage"] = bag_max_stage[~has_special]
    # propagar flags a nivel de bag (por si solo una imagen del bag los traia)
    df["ap_rop"] = df.groupby("bag_id")["ap_rop"].transform("any")
    df["laser_treated"] = df.groupby("bag_id")["laser_treated"].transform("any")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    clinical_rows = build_ridirp_rows() + build_farfum_rows() + build_ropvl_rows()
    df = pd.DataFrame(clinical_rows)
    df = reconcile_bag_stage(df)
    df = reconcile_bag_plus(df)
    df = reconcile_bag_treatment(df)
    df["split"] = None
    df["fold"] = None
    out_path = MANIFEST_DIR / "rop_unified_manifest.parquet"
    df.to_parquet(out_path, index=False)
    print(f"-> {out_path} ({len(df)} filas)")

    seg_rows = build_hvdropdb_rows()
    seg_df = pd.DataFrame(seg_rows)
    seg_out_path = MANIFEST_DIR / "hvdropdb_segmentation_manifest.parquet"
    seg_df.to_parquet(seg_out_path, index=False)
    print(f"-> {seg_out_path} ({len(seg_df)} filas)")


if __name__ == "__main__":
    main()
