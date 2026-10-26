"""Limpia HVDROPDB (subset de Segmentacion) y escribe datasets/clean/HVDROPDB.parquet.

Mismo patron que clean_ridirp.py / clean_farfum.py / clean_ropvl.py:
    crudo -> clean_<dataset>.py (este archivo) -> build_manifest.py (unificado)

Nota: HVDROPDB es estructuralmente distinto a los otros 3 - no es MIL clinico,
es (imagen, mascara) para el pre-entrenamiento visual de Fase 1. Por eso NO
alimenta el manifiesto clinico unificado ni el Group K-Fold por paciente (ver
README principal, S2 y S5).

Decisiones aplicadas aqui (detalle completo en system/eda/eda_hvdropdb.ipynb):

- `pair_id` = f"{structure}_{camera}_{indice}" (ej. "BV_Neo_1") - una fila por
  cada par (imagen, mascara) real en disco, 300 filas esperadas (50 x 2
  camaras x 3 estructuras).
- `image_hash` (MD5 del archivo de imagen, NO de la mascara) se calcula y se
  guarda aqui a proposito: el hash exhaustivo de las 300 imagenes encontro
  que 38 de ellas (12.7%) son el mismo archivo fuente reciclado entre dos
  carpetas de estructura distintas (25 BV/RetCam<->OD/RetCam, 10 BV/Neo<->
  RIDGE/Neo, 3 BV/RetCam<->RIDGE/RetCam) - las mascaras siempre son
  distintas (0 duplicados), pero la foto de entrada no. Sin agrupar por este
  hash al hacer el split train/val, la misma foto podria terminar en train
  para una estructura y en val para otra (fuga de informacion visual).
  Este script NO arma el split (decision B/opcion "esperar" acordada) - solo
  deja el hash calculado para que el script de split (pendiente, separado de
  este) agrupe por el antes de repartir train/val.
- `split`/`fold` quedan fuera de este parquet (no se agregan columnas vacias
  para ellas) - a diferencia del manifiesto clinico, donde si se reservan
  esas columnas desde ya, aqui se prefiere no asumir el nombre/forma final
  del split hasta que se escriba ese script.

Requiere: pandas, pyarrow. Ejecutar despues de extract_archives.py.
"""
import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent  # system/dataset_build/clean/ -> ROP_paper/
DATASETS = ROOT / "datasets"
EXTRACTED_SEG = DATASETS / "extracted" / "HVDROPDB" / "HVDROPDB_RetCam_Neo_Segmentation"
CLEAN_DIR = DATASETS / "clean"


def rel(path: Path) -> str:
    return path.relative_to(DATASETS / "extracted").as_posix()


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> None:
    rows = []
    missing_masks = 0
    for structure_dir in sorted(EXTRACTED_SEG.glob("HVDROPDB-*")):
        structure = structure_dir.name.replace("HVDROPDB-", "")
        for images_dir in sorted(structure_dir.glob("*_images")):
            camera = "Neo" if images_dir.name.startswith("Neo") else "RetCam"
            masks_dir = images_dir.parent / images_dir.name.replace("_images", "_masks")
            for img_path in sorted(images_dir.glob("*.png"), key=lambda p: int(p.stem)):
                mask_path = masks_dir / img_path.name
                if not mask_path.exists():
                    missing_masks += 1
                    continue
                rows.append(
                    {
                        "pair_id": f"{structure}_{camera}_{img_path.stem}",
                        "structure": structure,
                        "camera": camera,
                        "image_path": rel(img_path),
                        "mask_path": rel(mask_path),
                        "image_hash": md5(img_path),
                    }
                )

    if missing_masks:
        print(f"[aviso] {missing_masks} imagenes sin mascara correspondiente (omitidas)")

    df = pd.DataFrame(rows)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CLEAN_DIR / "HVDROPDB.parquet"
    df.to_parquet(out_path, index=False)

    n_groups = df["image_hash"].nunique()
    n_shared = int((df["image_hash"].value_counts() > 1).sum())
    print(f"[HVDROPDB] {len(df)} pares (imagen, mascara)")
    print(f"[HVDROPDB] {n_groups} fotos fuente distintas ({n_shared} compartidas entre >1 estructura/camara)")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
