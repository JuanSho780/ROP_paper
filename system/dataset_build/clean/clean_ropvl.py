"""Limpia ROP-VL por separado y escribe datasets/clean/ROP-VL.parquet.

Mismo patron que clean_ridirp.py / clean_farfum.py:
    crudo -> clean_<dataset>.py (este archivo) -> build_manifest.py (unificado)

Decisiones aplicadas aqui (detalle completo, incluyendo por que se
descarto la alternativa de "voto mayoritario", en system/eda/eda_ropvl.ipynb,
celda final "Estrategia final para clean_ropvl.py"):

- `bag_id` = (patient_id, fecha, ojo): cada fila del xlsx ya es una imagen
  individual, el bag agrupa las imagenes del mismo ojo tomadas el mismo dia
  (mismo criterio de granularidad que RIDIRP/FARFUM - una "bolsa" = un
  examen, no una serie temporal).
- Regla final (opcion conservadora, decidida tras verificar con datos reales
  que NO existe mayoria en los 54 bags mixtos - exactamente 1 imagen por
  categoria en cada uno, es un empate 1 a 1, no hay "minoria" que descartar):
  cualquier bag que contenga al menos una imagen `Laser-treated ROP` se
  ELIMINA por completo (todas sus imagenes), sin excepcion. Un ojo con
  cicatriz de laser ya tuvo ROP severa tratada - no es seguro asumir
  "normal" ni "A-ROP" a partir de las otras fotos del mismo bag.
- Para los bags que sobreviven, `stage` = maximo entre las categorias
  numericas presentes en el bag (normal=0, Stage 1-5=1-5) - misma logica de
  severidad-maxima que RIDIRP/FARFUM (MIL: basta una instancia positiva
  para que la bolsa sea positiva).
- `ap_rop` = True si el bag tiene al menos una imagen `A-ROP` (ya sin mezcla
  con Laser-treated, porque esos bags se eliminaron arriba). Si el bag no
  tiene ninguna imagen A-ROP, `ap_rop` = False (ROP-VL si distingue A-ROP
  de "sin informacion", a diferencia de RIDIRP donde ambos pueden ser NaN).
- `plus` y `treatment` quedan siempre None - ROP-VL no tiene estas
  etiquetas en su metadata (`img_info.xlsx` solo trae `categories`).

Requiere: pandas, pyarrow, openpyxl. Ejecutar despues de extract_archives.py.
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent  # system/dataset_build/clean/ -> ROP_paper/
DATASETS = ROOT / "datasets"
EXTRACTED_DIR = DATASETS / "extracted" / "ROP-VL"
IMAGE_DIR = EXTRACTED_DIR / "Image"
CLEAN_DIR = DATASETS / "clean"

CAT_TO_STAGE = {
    "normal": 0,
    "Stage 1 ROP": 1,
    "Stage 2 ROP": 2,
    "Stage 3 ROP": 3,
    "Stage 4 ROP": 4,
    "Stage 5 ROP": 5,
}
LASER_CATEGORY = "Laser-treated ROP"
AP_ROP_CATEGORY = "A-ROP"
CAT_TO_FOLDER = {"normal": "Normal"}  # el resto coincide literal con el nombre de carpeta

FILENAME_RE = re.compile(r"^(?P<pid>\d+)_(?P<date>[\d-]+)_(?P<eye>[LR])_(?P<idx>\d+)\.jpg$")


def rel(path: Path) -> str:
    return path.relative_to(DATASETS / "extracted").as_posix()


def load_img_info() -> pd.DataFrame:
    df = pd.read_excel(EXTRACTED_DIR / "img_info.xlsx")
    parsed = df["image"].str.extract(FILENAME_RE)
    df["date"] = parsed["date"]
    df["eye"] = parsed["eye"]
    return df


def main() -> None:
    df = load_img_info()
    df["bag_id"] = "ROPVL_" + df["patient_id"].astype(str) + "_" + df["date"] + "_" + df["eye"]

    bag_has_laser = df.groupby("bag_id")["categories"].transform(lambda s: (s == LASER_CATEGORY).any())
    dropped_bags = df.loc[bag_has_laser, "bag_id"].nunique()
    dropped_images = int(bag_has_laser.sum())
    df = df.loc[~bag_has_laser].copy()

    df["stage_value"] = df["categories"].map(CAT_TO_STAGE)
    df["stage"] = df.groupby("bag_id")["stage_value"].transform("max")
    df["ap_rop"] = df.groupby("bag_id")["categories"].transform(lambda s: (s == AP_ROP_CATEGORY).any())

    rows = []
    missing_files = 0
    for _, r in df.iterrows():
        folder = CAT_TO_FOLDER.get(r["categories"], r["categories"])
        img_path = IMAGE_DIR / folder / r["image"]
        if not img_path.exists():
            missing_files += 1
            continue

        patient_id_global = f"ROPVL_{r['patient_id']}"
        rows.append(
            {
                "sample_id": f"{r['bag_id']}_{r['image']}",
                "bag_id": r["bag_id"],
                "patient_id_global": patient_id_global,
                "image_path": rel(img_path),
                "sexo": r["gender"],
                "GA": float(r["gestational_age"]),
                "BW": float(r["birth_weight"]),
                "PMA": None,
                "stage": r["stage"],
                "plus": None,
                "treatment": None,
                "ap_rop": bool(r["ap_rop"]),
                "raw_category": r["categories"],
            }
        )

    if missing_files:
        print(f"[aviso] {missing_files} filas del xlsx sin archivo de imagen extraido (omitidas)")
    print(f"[ROP-VL] {dropped_bags} bags ({dropped_images} imagenes) eliminados por contener Laser-treated ROP")

    out = pd.DataFrame(rows)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CLEAN_DIR / "ROP-VL.parquet"
    out.to_parquet(out_path, index=False)

    bags = out.drop_duplicates("bag_id")
    print(f"[ROP-VL] {len(out)} imagenes, {out['bag_id'].nunique()} bags, {out['patient_id_global'].nunique()} pacientes")
    print(f"[ROP-VL] stage nulo (por bag): {bags['stage'].isna().sum()} ({100*bags['stage'].isna().mean():.1f}%)")
    print(f"[ROP-VL] ap_rop (por bag): True={int(bags['ap_rop'].sum())} False={int((~bags['ap_rop']).sum())}")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
