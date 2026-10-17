# Construcción del Manifiesto Unificado

Scripts que extraen los 4 datasets comprimidos y construyen el Parquet unificado que alimenta el pipeline de entrenamiento. Ejecutar en orden:

```
python system/dataset_build/extract_archives.py
python system/dataset_build/build_manifest.py
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

## Problema real encontrado y corregido: `stage` inconsistente dentro de un mismo bag (ROP-VL)

Verificando el parquet ya construido (no en la planificación) se encontraron **91 bags de ROP-VL (4.9% de 1.867)** donde dos imágenes del mismo ojo+fecha traían `categories` de estadio distintas en el dataset original — ej. `ROPVL_173_2015-07-29_R` tiene una foto etiquetada `normal` y dos fotos `Stage 2 ROP` el mismo día, mismo ojo. La mayoría son estadios adyacentes (delta=1 en 76/91 casos), consistente con que distintas tomas capturan distintos campos de la retina (polo posterior vs. periferia) y no necesariamente un error de etiquetado.

**Corrección aplicada en `build_manifest.py` (`reconcile_bag_stage`):** el `stage` final de un bag es el **máximo** entre sus imágenes — el hallazgo más avanzado visible en cualquier imagen define el estadio clínico real de ese ojo ese día (misma lógica de MIL: basta una instancia positiva para que la bolsa sea positiva).

**Caso aparte, sin resolver por máximo:** 52 de esos bags mezclaban una imagen `A-ROP` o `Laser-treated ROP` (que ya tienen `stage=NaN` por diseño, no son comparables a la escala ordinal) con otra imagen del mismo ojo etiquetada `normal` o con un estadio numérico. Ahí **no** se aplica el máximo — asumir "normal" solo porque otra foto del mismo ojo se ve sana sería un error clínico real (un ojo con tratamiento láser ya tuvo ROP severa). Para estos 52 bags se deja `stage=NaN` en todas sus imágenes, y las flags `ap_rop`/`laser_treated` se propagan a nivel de bag (si una sola imagen del bag las traía, ahora las tienen todas). Cobertura final de `stage` en ROP-VL tras la corrección: 73% (antes de la corrección, antes de validar, el run original tenía datos per-imagen inconsistentes dentro del mismo bag).

Se verificó con un script de consistencia que, tras la corrección, **0 bags** en los 3 datasets (RIDIRP, FARFUM, ROP-VL) tienen valores clínicos (`stage`/`plus`/`treatment`) distintos entre imágenes del mismo `bag_id`.

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

### FARFUM-RoP — columna "Label" (corrección importante, no estaba en la primera versión)

**Corrección a lo documentado antes:** el xlsx de `Dataset_Labels.xlsx` tiene un encabezado de 2 filas, y la fila 1 (la que se usaba como `header` al leer con pandas) deja el nombre de la última columna (R) en blanco — el nombre real, "Label", está en la fila 0. El primer intento de este manifiesto leía las columnas por posición pero nunca tocaba la columna R, así que esa columna quedó completamente ignorada.

`Label` es un **target oficial, con cobertura completa (1.533/1.533, cero nulos)** y exactamente 3 valores (`1`: 782, `2`: 479, `3`: 272) — a diferencia de `grade` (0-3, semántica sin confirmar) y de nuestro propio consenso por mayoría entre los 5 raters (cobertura baja, ver abajo). No es una función simple de ningún rater individual ni de la mayoría exacta de `grade`/`stage`, así que probablemente es la etiqueta de referencia que fijaron los propios curadores del dataset. Se mapeó `1→Normal, 2→Pre-Plus, 3→Plus` por correlación con `diagnostic` (`Label=3` casi nunca coincide con `revisit`, `Label=1` rara vez con `treatment`) y porque coincide con la descripción oficial de Figshare ("anotado como Normal/Pre-Plus/Plus") — la dirección del mapeo es una inferencia de alta confianza, no viene explícita en el archivo, así que **sigue valiendo la pena confirmarla contra el texto del paper** si se consigue acceso.

Con esta corrección, **`plus` ahora tiene 100% de cobertura en FARFUM** (antes 0%, quedaba en `None` para todo el dataset).

**Hallazgo adicional corregido:** 3 de los 68 pacientes (`patient44`, `patient55`, `patient57`) tenían la columna `Label` con 2 valores distintos entre sus propias imágenes (siempre categorías adyacentes: Normal/Pre-Plus o Pre-Plus/Plus) — mismo patrón de inconsistencia que se encontró en ROP-VL. Se resolvió igual: `reconcile_bag_plus` deja el paciente con el hallazgo más severo detectado en cualquiera de sus imágenes.

### FARFUM-RoP — consenso entre 5 raters para `stage` y `treatment`

Aparte de `Label`, cada imagen tiene hasta 5 opiniones independientes (`grade`, `stage`, `diagnostic` por rater A-E) que **sí están separadas por rater** (A, B, C, D, E — 5 oftalmólogos, consistente con la descripción oficial del dataset). Se calculó consenso por **mayoría estricta** (>50% de los votos no-nulos, mínimo 3 votos) para `stage` y `treatment`.

**Bug real encontrado y corregido (no era desacuerdo entre médicos, como se documentó en una versión anterior de este archivo):** `_majority_vote` filtraba votos vacíos con `v is not None`, pero `pandas` representa una celda vacía de Excel como `NaN` (float), no como `None` — y `float('nan') is not None` es `True`. Es decir, **cada celda en blanco estaba entrando al conteo de votos como si fuera una respuesta real**, inflando el denominador y haciendo casi imposible alcanzar mayoría estricta. Se corrigió reemplazando el filtro por `pd.notna(v)`, que sí reconoce tanto `None` como `NaN`.

**Impacto medido (antes → después de la corrección, mismo umbral de mayoría estricta, sin cambiar criterio):**
- `stage`: 14.7% → 19.5% de cobertura.
- `treatment`: 17.4% → **55.0%** de cobertura, y por primera vez aparece la categoría `no_treatment` (antes nunca ganaba consenso porque el "voto" `NaN` competía artificialmente contra ella).

`stage` mejoró menos porque ahí la causa principal de los vacíos es otra, real y no un bug: cuando un médico marca `grade=0` (normal), casi nunca anota `stage` (verificado: >98% de las veces que `grade=0`, ese mismo rater deja `stage` y `diagnostic` en blanco para esa imagen, en los 5 raters). Eso es consistente — un ojo sin ROP no tiene estadio que anotar — pero reduce el número de votos disponibles para alcanzar el mínimo de 3. **Pendiente de decidir con el usuario:** si conviene imputar `stage=0` (y `diagnostic="revisit"`) por rater cuando su `grade==0`, para recuperar esa cobertura sin inventar nada — simulación preliminar sugiere que `stage` podría subir a ~61% y `treatment` a ~95% con ese criterio. No aplicado todavía en el script.

**Nota sobre `grade` (distinto de `Label`):** la columna `grade` (valores 0-3 por rater, una de las 5×3 columnas crudas) sigue sin semántica confirmada — no se usa para construir `plus` (eso ahora viene de `Label`, ver arriba). No se encontró el texto exacto del paper (Akbari et al. 2024, DOI `10.1038/s41597-024-03897-7`, no verificado de forma independiente — ver `system/datasets_download/README.md`) que aclare qué mide `grade`. Se conserva sin usar en `raw_grade_per_rater` (los 5 valores separados por `;`) por si resulta útil más adelante.

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

1. Confirmar contra el paper original (si se consigue acceso) que la columna `Label` de FARFUM efectivamente corresponde a Normal/Pre-Plus/Plus en ese orden — el mapeo usado es una inferencia de alta confianza por correlación, no viene declarado explícitamente en el archivo.
2. Decidir si se imputa `stage=0`/`diagnostic="revisit"` por rater cuando ese rater marcó `grade==0` (normal) y dejó `stage`/`diagnostic` en blanco — recuperaría cobertura real de `stage` en FARFUM (actualmente 19.5%) sin inventar nada, pero es una inferencia sobre semántica clínica que conviene confirmar antes de aplicarla. No afecta a `plus` (ya 100% vía `Label`).
3. Escribir el script de partición Group K-Fold que llena `split`/`fold` (siguiente paso, no incluido aquí).

## Cobertura final por dataset y tarea

Verificado corriendo el script contra los datos reales (`df.groupby('dataset_origen')[['stage','plus','treatment']].notna().mean()`):

| Dataset | `stage` | `plus` | `treatment` |
|---|---|---|---|
| RIDIRP | 64.3% | 78.4% | 0% (no tiene esta etiqueta) |
| FARFUM | 19.5% | **100%** | 55.0% |
| ROP-VL | 73.0% | 0% (no tiene esta etiqueta) | 0% (no tiene esta etiqueta) |
