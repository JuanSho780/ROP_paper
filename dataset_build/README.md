# Construcción del Manifiesto Unificado

Scripts que extraen los 4 datasets comprimidos y construyen el Parquet unificado que alimenta el pipeline de entrenamiento. Ejecutar en orden:

```
python dataset_build/extract_archives.py
python dataset_build/build_manifest.py
```

Salida: `datasets/manifest/rop_unified_manifest.parquet` (RIDIRP + FARFUM + ROP-VL, tareas clínicas) y `datasets/manifest/hvdropdb_segmentation_manifest.parquet` (pares imagen/máscara, Fase 1). Ninguno de los dos se versiona en git (están dentro de `datasets/`, ignorado).

## Requisitos

- **7-Zip instalado** (necesario para extraer los `.rar` de FARFUM — no hay backend puro-Python para RAR5 comprimido). El script busca `7z` en el PATH y en `C:\Program Files\7-Zip\7z.exe` como fallback.
- `pip install pandas pyarrow openpyxl`

## Extracción — qué se extrae de cada dataset y por qué

| Dataset | Archivo fuente usado | Destino | Notas |
|---|---|---|---|
| RIDIRP | `images_stack_without_captions.zip` | `datasets/extracted/RIDIRP/` | **No usar `images.zip`.** Verifiqué por hash MD5 que `images.zip` e `images_stack.zip` son *bit-a-bit idénticos* entre sí, y que para los dispositivos 2 y 3 (Phoenix ICON, Natus RetCam Envision) traen texto del paciente/fecha quemado en el píxel. `images_stack_without_captions.zip` es la única variante limpia para todos los dispositivos. |
| FARFUM-RoP | los 68 `patientXX.rar` | `datasets/extracted/FARFUM/patientXX/` | Cada `.rar` ya trae internamente la carpeta `patientXX/`, por eso se extraen todos a la misma carpeta destino. |
| ROP-VL | `ROP-VL.zip` | `datasets/extracted/ROP-VL/` | Incluye `img_info.xlsx` y `disease_description.xlsx` en la raíz además de `Image/<categoría>/`. |
| HVDROPDB | solo `HVDROPDB_RetCam_Neo_Segmentation.zip` | `datasets/extracted/HVDROPDB/` | El subset de Clasificación (`HVDROPDB_RetCam_Neo_Classification.rar`, 189 imágenes Normal/ROP) **se deja sin extraer a propósito** — no trae patient ID ni metadata clínica, así que no puede alimentar el vector clínico de la Cross-Attention. Queda disponible como posible tarea auxiliar de Fase 1 si se decide usarlo más adelante. |

Los `.xlsx` de metadata (`infant_retinal_database_info.xlsx`, `Dataset_Details.xlsx`, `Dataset_Labels.xlsx`) **no se extraen** porque no están comprimidos — `build_manifest.py` los lee directo desde `datasets/<Dataset>_Original/`.

## Granularidad por dataset (qué es un "bag" de MIL)

Confirmado abriendo los archivos reales (no por supuesto):

| Dataset | `bag_id` | Evidencia |
|---|---|---|
| RIDIRP | `(patient_id, series)` — una visita real | Verificado: 484 carpetas paciente/serie = 484 filas del xlsx = 6.004 imágenes. Un paciente (043, AP-ROP) tiene 16 visitas longitudinales reales (PMA 32→56 semanas). Sin distinción de ojo (limitación del dataset). |
| FARFUM | `patient_id` (sesión única) | Verificado: 0 pacientes con más de un UUID de sesión en sus 68 `.rar`. 1-46 imágenes/paciente, ambos ojos mezclados sin distinción. |
| ROP-VL | `(patient_id, date, eye)` | Nuevo hallazgo: 190/1116 pacientes tienen más de una fecha (hasta 5 visitas), y 225 pacientes tienen categorías *distintas* entre sus propias imágenes (ej. un ojo Stage 3 y el otro normal el mismo día) — la etiqueta es por ojo+fecha, no replicada a nivel paciente como en FARFUM. Mayoría de bags son de 1 imagen (1719/1867), hasta 3 en algunos casos. |
| HVDROPDB | N/A — no es MIL clínico, es `(imagen, máscara)` para segmentación | 300 pares = 50 imágenes × (Vasos/Disco/Cresta) × (Neo/RetCam). Pareo imagen↔máscara verificado 1:1 sin huérfanos. |

## Problema real encontrado y corregido: columna "series" de RIDIRP

El xlsx de RIDIRP tiene la columna `SERIE NUMBER (S)` **mal asignada para ~31 filas**, todas del dispositivo 3 (Natus RetCam Envision): varias visitas distintas (con PMA distinto) comparten el mismo número de serie en vez de incrementar. Ejemplo real: paciente 146 tiene 4 filas con `series=4` y PMA 34.4/35.3/36.0/37.0, pero en los archivos extraídos esas mismas 4 visitas están correctamente numeradas `S01, S02, S03, S04`.

**Verificación:** el conteo de filas del xlsx por paciente coincide exactamente con el conteo de series distintas en los nombres de archivo para los 188 pacientes (0 discrepancias).

**Solución aplicada en `build_manifest.py`:** no se usa la columna `series` del xlsx como llave de join. En su lugar, para cada paciente se ordenan las filas del xlsx por PMA ascendente y los números de serie reales (del nombre de archivo) también ascendente, y se alinean por posición (rank cronológico). El `bag_id` final sí usa el número de serie real del archivo (que es el que agrupa las imágenes físicamente en disco).

## Mapeo de etiquetas — decisiones y pendientes

### RIDIRP (`DIAGNOSIS CODE` → `stage`)

```
DG 0 (physiological) -> stage 0      DG 5 (ROP 4A) -> stage 4
DG 1 (ROP 0)          -> stage 0      DG 6 (ROP 4B) -> stage 4
DG 2 (ROP 1)          -> stage 1      DG 7 (ROP 5)  -> stage 5
DG 3 (ROP 2)          -> stage 2      DG 8 (AP-ROP) -> stage=None, ap_rop=True
DG 4 (ROP 3)          -> stage 3      DG 9 (status post ROP) -> stage=None, status_post_rop=True
DG 10-13 (hamartoma, hemorragia, hipoplasia n.II, toxoplasma) -> stage=None, plus=None, is_rop_eligible=False
```

AP-ROP y status-post no se fuerzan a un estadio 0-5 porque no son comparables con la escala ordinal clásica (AP-ROP es agresiva/atípica por definición; status-post es un ojo ya tratado, no una evaluación activa). Los diagnósticos no-ROP (10-13) quedan completamente excluidos de `stage` y `plus` — son hallazgos oculares distintos, no ROP.

`PLUS FORM (PF)` se mapea directo: `0→Normal, 1→Pre-Plus, 2→Plus`. En los datos reales **no hay ningún caso Pre-Plus** (0 filas con PF=1) — confirmado ya en la auditoría inicial, se mantiene igual en el manifiesto (4.080 Normal, 629 Plus, resto None por ser no-ROP).

RIDIRP no tiene columna de tratamiento → `treatment` siempre `None` para este origen.

### FARFUM-RoP — consenso entre 5 raters, y un pendiente importante

Cada imagen tiene hasta 5 opiniones independientes (`grade`, `stage`, `diagnostic` por rater A-E). Se calculó consenso por **mayoría estricta** (>50% de los votos no-nulos, mínimo 3 votos) para `stage` y `treatment`.

**Cobertura real tras aplicar mayoría estricta** (verificado corriendo el script, no estimado):
- `stage`: 159 de 1.533 imágenes alcanzan consenso (13 stage=1, 146 stage=2). El resto son `grade=0` (normal, sin opinión de stage — 762 imágenes) o desacuerdo real entre raters sin mayoría (1.123 imágenes). Hay desacuerdo inter-rater genuino incluso cuando coinciden en que hay ROP.
- `treatment`: 266 de 1.533 alcanzan consenso (119 revisit, 147 treatment). **Ningún caso alcanza consenso estricto en "no_treatment"** — no es un bug: el análisis muestra que "no_treatment" es la opción con más desacuerdo entre raters (gana por pluralidad simple solo 74 veces, nunca por mayoría estricta).

Esto es una limitación real a documentar en la tesis: con mayoría estricta, el cabezal de Tratamiento entrena con ~17% de las imágenes de FARFUM. Alternativas a evaluar más adelante (no aplicadas todavía): bajar el umbral a pluralidad simple (más cobertura, menos confianza), o tratar las 5 etiquetas como *soft labels* en vez de forzar un consenso duro.

**Pendiente sin resolver:** la columna `grade` (valores 0-3 por rater) no tiene semántica confirmada. La ficha de Figshare describe FARFUM-RoP como anotado en 3 clases "Normal/Pre-Plus/Plus", pero la columna trae 4 valores distintos (0,1,2,3) — no es un mapeo 1:1 obvio y no se encontró el texto exacto del paper (Akbari et al. 2024, DOI `10.1038/s41597-024-03897-7`, no verificado de forma independiente — ver `datasets_download/README.md`) que lo aclare. **Por eso la columna unificada `plus` queda en `None` para todo FARFUM** en vez de adivinar el mapeo. Se preserva el valor crudo en `raw_grade_per_rater` (los 5 valores separados por `;`) para poder resolverlo sin re-extraer nada en cuanto se confirme la semántica contra el paper original.

### ROP-VL (`categories` → `stage`)

```
normal -> 0          Stage 4 ROP -> 4
Stage 1 ROP -> 1      Stage 5 ROP -> 5
Stage 2 ROP -> 2      A-ROP -> stage=None, ap_rop=True
Stage 3 ROP -> 3      Laser-treated ROP -> stage=None, laser_treated=True
```

`Laser-treated ROP` **no se mapea a la columna `treatment`** aunque conceptualmente esté relacionado — es una categoría retrospectiva (el ojo ya fue tratado en algún momento, no sabemos con qué estadio/zona estaba *antes* del láser), mientras que `treatment` en FARFUM es una decisión prospectiva tomada en el momento de la foto. Mezclar ambas definiciones de "tratamiento" bajo la misma columna introduciría ruido de etiqueta. Se preserva como flag `laser_treated` + `raw_diagnosis_code` por si se quiere usar distinto más adelante. ROP-VL no tiene columna de Plus explícita → `plus` siempre `None` para este origen.

## Esquema del manifiesto unificado

`rop_unified_manifest.parquet` (9.557 filas: 6.004 RIDIRP + 1.533 FARFUM + 2.020 ROP-VL):

| Columna | Tipo | Notas |
|---|---|---|
| `sample_id` | str | única por imagen |
| `bag_id` | str | agrupador de MIL, ver tabla de granularidad arriba |
| `dataset_origen` | str | `RIDIRP` / `FARFUM` / `ROP-VL` |
| `patient_id_global` | str | prefijado por dataset, sin colisiones |
| `image_path` | str | relativa a `datasets/extracted/` |
| `sexo`, `GA`, `BW`, `PMA` | — | `PMA` solo poblado para RIDIRP; `pma_available` indica si aplica |
| `stage`, `plus`, `treatment` | float/str, nullable | `None` = sin ground truth para esa tarea en esa fila — así se implementa el masking de la pérdida multitarea |
| `ap_rop`, `status_post_rop`, `is_rop_eligible`, `laser_treated` | bool | flags de casos especiales, ver mapeos arriba |
| `raw_diagnosis_code`, `raw_grade_per_rater` | str, nullable | trazabilidad a la etiqueta original sin procesar |
| `split`, `fold` | — | **vacíos a propósito** — los llena el siguiente script (partición Group K-Fold), todavía no escrito |

`hvdropdb_segmentation_manifest.parquet` (300 filas): `pair_id`, `structure` (BV/OD/RIDGE), `camera` (Neo/RetCam), `image_path`, `mask_path`.

## Pendientes conocidos

1. Confirmar semántica de la columna `grade` de FARFUM contra el paper original y completar `plus` para ese origen.
2. Decidir si la cobertura de consenso de FARFUM (~17% en `treatment`, ~10% en `stage`) es aceptable o si conviene relajar el umbral de mayoría estricta.
3. Escribir el script de partición Group K-Fold que llena `split`/`fold` (siguiente paso, no incluido aquí).
