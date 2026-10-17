"""Extrae los datasets comprimidos hacia datasets/extracted/<Dataset>/.

No reprocesa ni recorta imagenes, solo descomprime. Ver README.md de esta
carpeta para el detalle de que fuente se usa por dataset y por que.

Requiere 7-Zip instalado (para los .rar de FARFUM). Si "7z" no esta en el
PATH, el script busca en las rutas de instalacion por defecto de Windows.
"""
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # system/dataset_build/ -> ROP_paper/
DATASETS = ROOT / "datasets"
EXTRACTED = DATASETS / "extracted"

SEVEN_ZIP_CANDIDATES = [
    "7z",
    r"C:\Program Files\7-Zip\7z.exe",
    r"C:\Program Files (x86)\7-Zip\7z.exe",
]


def find_7z() -> str:
    for candidate in SEVEN_ZIP_CANDIDATES:
        if shutil.which(candidate) or Path(candidate).exists():
            return candidate
    raise RuntimeError(
        "No se encontro 7-Zip. Instalalo (https://www.7-zip.org/) "
        "o agrega 7z.exe al PATH."
    )


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    print(f"[zip] {zip_path.name} -> {dest_dir}")


def extract_rar(rar_path: Path, dest_dir: Path, seven_zip: str) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [seven_zip, "x", str(rar_path), f"-o{dest_dir}", "-y"],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    print(f"[rar] {rar_path.name} -> {dest_dir}")


def extract_ridirp() -> None:
    # Fuente canonica: images_stack_without_captions.zip, NO images.zip
    # (images.zip e images_stack.zip tienen texto de paciente/fecha quemado
    # en el pixel para los dispositivos 2 y 3 - ver README).
    src = DATASETS / "RIDIRP_Original" / "images_stack_without_captions.zip"
    dest = EXTRACTED / "RIDIRP"
    if dest.exists():
        print("[skip] RIDIRP ya extraido")
        return
    extract_zip(src, dest)


def extract_farfum(seven_zip: str) -> None:
    src_dir = DATASETS / "FARFUM_Original"
    dest_root = EXTRACTED / "FARFUM"
    dest_root.mkdir(parents=True, exist_ok=True)
    for rar_path in sorted(src_dir.glob("patient*.rar")):
        patient_id = rar_path.stem  # "patient01"
        if (dest_root / patient_id).exists():
            continue
        # cada .rar ya trae internamente la carpeta "patientXX/", por eso
        # se extrae directo a dest_root y no a dest_root/patient_id
        extract_rar(rar_path, dest_root, seven_zip)


def extract_ropvl() -> None:
    src = DATASETS / "ROP-VL_Original" / "ROP-VL.zip"
    dest = EXTRACTED / "ROP-VL"
    if dest.exists():
        print("[skip] ROP-VL ya extraido")
        return
    extract_zip(src, dest)


def extract_hvdropdb() -> None:
    # Solo el subset de Segmentacion (insumo de Fase 1). El subset de
    # Clasificacion binaria (Normal/ROP, sin metadata clinica) se deja
    # fuera del scope a proposito - ver README.
    src = DATASETS / "HVDROPDB_Original" / "HVDROPDB_RetCam_Neo_Segmentation.zip"
    dest = EXTRACTED / "HVDROPDB"
    if dest.exists():
        print("[skip] HVDROPDB ya extraido")
        return
    extract_zip(src, dest)


def main() -> None:
    seven_zip = find_7z()
    EXTRACTED.mkdir(parents=True, exist_ok=True)
    extract_ridirp()
    extract_farfum(seven_zip)
    extract_ropvl()
    extract_hvdropdb()
    print("Extraccion completa.")


if __name__ == "__main__":
    main()
