"""Construye el manifiesto clinico unificado a partir de los parquets ya
limpios por dataset (datasets/clean/RIDIRP.parquet, FARFUM.parquet,
ROP-VL.parquet), en vez de releer el crudo.

Segunda etapa del pipeline de 3 pasos (ver README.md de esta carpeta):
    crudo -> clean/clean_<dataset>.py (limpio individual, ya hecho)
          -> partition/build_manifest.py (este archivo, unificado)

HVDROPDB NO entra aqui - no es MIL clinico (es segmentacion, Fase 1) y usa
su propio pipeline separado (clean/clean_hvdropdb.py + partition/
split_hvdropdb.py -> datasets/manifest/HVDROPDB.parquet).

Decisiones aplicadas al unificar (cada una ya validada y documentada por
separado en cada clean_<dataset>.py y en el README principal):

- `treatment` se descarta por completo - el cabezal de Tratamiento fue
  reemplazado por AP-ROP (ver README principal S0/S4.4, y el banner de
  "Cambio de alcance" en el README de esta carpeta). Los 3 clean_*.py
  siguen calculando esa columna (registro historico de cuando si era parte
  del alcance), pero no se lleva al manifiesto final.
- `device_raw` (solo RIDIRP) se descarta - es metadata de la capa limpia
  individual (util para un futuro analisis leave-one-device-out), nunca un
  feature de entrada del modelo.
- `ap_rop` se enmascara (None) en los casos donde NO se puede afirmar con
  certeza que el ojo nunca tuvo AP-ROP, en vez de asumir False:
    - RIDIRP: None para las filas `status_post_rop=True` (DG=9, ojo ya
      tratado/regresado - no se sabe si en algun momento paso por AP-ROP
      antes del seguimiento registrado); True/False normal en el resto.
    - FARFUM: None siempre - este dataset no distingue AP-ROP en absoluto.
    - ROP-VL: True/False directo desde el clean (ya sin ambiguedad: los
      bags que tocaban "Laser-treated ROP" se eliminaron por completo en
      clean_ropvl.py, asi que no queda ningun caso "Laser-treated sin
      A-ROP" que enmascarar aqui).
- `pma_available` se deriva (`PMA.notna()`) para los 3 datasets - solo
  RIDIRP tiene PMA real por visita, los otros dos siempre None.
- `split`/`fold` quedan vacios a proposito - los llena el script de Group
  K-Fold clinico (K=5, agrupado por `patient_id_global`), todavia no
  escrito (ver README principal S2 y "Pendientes conocidos" de esta
  carpeta).

Requiere: pandas, pyarrow. Ejecutar despues de clean_ridirp.py, clean_farfum.py
y clean_ropvl.py.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent  # system/dataset_build/partition/ -> ROP_paper/
CLEAN_DIR = ROOT / "datasets" / "clean"
MANIFEST_DIR = ROOT / "datasets" / "manifest"

UNIFIED_COLUMNS = [
    "sample_id", "bag_id", "dataset_origen", "patient_id_global", "image_path",
    "sexo", "GA", "BW", "PMA", "pma_available",
    "stage", "plus", "ap_rop",
    "split", "fold",
]


def load_ridirp() -> pd.DataFrame:
    df = pd.read_parquet(CLEAN_DIR / "RIDIRP.parquet")
    df["dataset_origen"] = "RIDIRP"
    # status-post (DG=9) es ambiguo para AP-ROP: el ojo ya fue tratado/
    # regreso, no hay forma de afirmar que nunca paso por AP-ROP - se
    # enmascara en vez de asumir False.
    df["ap_rop"] = df["ap_rop"].where(~df["status_post_rop"], None)
    return df


def load_farfum() -> pd.DataFrame:
    df = pd.read_parquet(CLEAN_DIR / "FARFUM.parquet")
    df["dataset_origen"] = "FARFUM"
    df["ap_rop"] = None  # FARFUM no distingue AP-ROP en absoluto
    return df


def load_ropvl() -> pd.DataFrame:
    df = pd.read_parquet(CLEAN_DIR / "ROP-VL.parquet")
    df["dataset_origen"] = "ROP-VL"
    return df


def to_unified_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pma_available"] = df["PMA"].notna()
    df["split"] = None
    df["fold"] = None
    return df[UNIFIED_COLUMNS]


def main() -> None:
    sources = {
        "RIDIRP": load_ridirp(),
        "FARFUM": load_farfum(),
        "ROP-VL": load_ropvl(),
    }
    for name, df in sources.items():
        print(f"[{name}] {len(df)} imagenes leidas de datasets/clean/{name}.parquet")

    unified = pd.concat([to_unified_schema(df) for df in sources.values()], ignore_index=True)

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MANIFEST_DIR / "rop_unified_manifest.parquet"
    unified.to_parquet(out_path, index=False)

    print(f"\n[unificado] {len(unified)} imagenes, {unified['bag_id'].nunique()} bags, "
          f"{unified['patient_id_global'].nunique()} pacientes")
    print(f"[unificado] stage nulo: {100*unified['stage'].isna().mean():.1f}%")
    print(f"[unificado] plus nulo:  {100*unified['plus'].isna().mean():.1f}%")
    print(f"[unificado] ap_rop nulo: {100*unified['ap_rop'].isna().mean():.1f}% "
          f"(True={int((unified['ap_rop'] == True).sum())}, False={int((unified['ap_rop'] == False).sum())})")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
