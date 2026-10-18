# Construcción del Manifiesto Unificado

Scripts que extraen los 4 datasets comprimidos y construyen el Parquet unificado que alimenta el pipeline de entrenamiento, en 3 etapas: **crudo → `clean_<dataset>.py` (limpio individual) → `build_manifest.py` (unificado)**.

```
python system/dataset_build/extract_archives.py
python system/dataset_build/clean_ridirp.py     # -> datasets/clean/RIDIRP.parquet
python system/dataset_build/clean_farfum.py     # -> datasets/clean/FARFUM.parquet
python system/dataset_build/clean_ropvl.py      # -> datasets/clean/ROP-VL.parquet
# clean_hvdropdb.py: pendiente
# build_manifest.py: TODAVIA NO actualizado para leer de datasets/clean/ - ver "Pendientes conocidos"
```

Salida (una vez completa la migración): `datasets/manifest/rop_unified_manifest.parquet` (RIDIRP + FARFUM + ROP-VL, tareas clínicas) y `datasets/manifest/hvdropdb_segmentation_manifest.parquet` (pares imagen/máscara, Fase 1). Nada de esto se versiona en git (está dentro de `datasets/`, ignorado).

## ⚠️ Cambio de alcance: `treatment` ya no es un cabezal del modelo

**Decisión tomada después de escribir `clean_ridirp.py` y `clean_farfum.py`:** el cabezal de Tratamiento se **reemplazó** por AP-ROP (detalle completo y razonamiento en el README principal, §0 y §4.4) — no se agrega como cabezal adicional, se sustituye. Motivo resumido: Tratamiento solo tenía ground truth en FARFUM (68 pacientes, una sola fuente), mientras que AP-ROP tiene soporte de RIDIRP (470 img.) + ROP-VL (171 img.); y clínicamente es más apropiado que el modelo describa hallazgos (estadio, plus, AP-ROP) y el médico decida el tratamiento, no que el modelo lo prescriba.

Las secciones de abajo sobre `clean_ridirp.py`/`clean_farfum.py` **se dejan tal como quedaron documentadas cuando se escribieron** (sí calculan y validan una columna `treatment`, porque en ese momento todavía era parte del alcance) — es un registro histórico correcto de lo que se construyó, no una inconsistencia. La columna `treatment` simplemente no se lleva al parquet unificado final; se puede quitar de los scripts limpios más adelante si se quiere, pero no es urgente (no le hace daño a nada dejarla calculada en la capa intermedia).

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

## Problema real encontrado: `stage` inconsistente dentro de un mismo bag (ROP-VL)

> ⚠️ **Nota histórica, superada:** esta sección documenta el hallazgo original (era de `build_manifest.py` monolítico, antes de `clean_ropvl.py` y antes del cabezal AP-ROP). Los números (91/52/73%) **ya no son los que se usan** — la regla y las cifras finales están en "ROP-VL — regla final: cualquier `Laser-treated` elimina el bag completo (decisión conservadora)" más abajo. Se deja aquí solo como registro de cómo se descubrió el problema.

Verificando el parquet ya construido (no en la planificación) se encontraron **91 bags de ROP-VL (4.9% de 1.867)** donde dos imágenes del mismo ojo+fecha traían `categories` de estadio distintas en el dataset original — ej. `ROPVL_173_2015-07-29_R` tiene una foto etiquetada `normal` y dos fotos `Stage 2 ROP` el mismo día, mismo ojo. La mayoría son estadios adyacentes (delta=1 en 76/91 casos), consistente con que distintas tomas capturan distintos campos de la retina (polo posterior vs. periferia) y no necesariamente un error de etiquetado.

Primera corrección intentada (`reconcile_bag_stage` en el `build_manifest.py` original): `stage` final de un bag = **máximo** entre sus imágenes (misma lógica de MIL: basta una instancia positiva para que la bolsa sea positiva), excepto en 52 bags donde la mezcla incluía `A-ROP`/`Laser-treated ROP` junto con `normal` o un estadio numérico — ahí aplicar el máximo era clínicamente incorrecto (un ojo con láser ya tuvo ROP severa, no es "normal"), así que esos 52 quedaban con `stage=NaN`. Esto daba una cobertura de `stage` del 73%, pero dejaba el caso `A-ROP`/`Laser-treated` sin una regla limpia y motivó rediseñar todo el tratamiento de ROP-VL — ver la sección final más abajo.

Se verificó con un script de consistencia que, con la regla final aplicada, **0 bags** en los 3 datasets (RIDIRP, FARFUM, ROP-VL) tienen valores clínicos (`stage`/`plus`/`treatment`) distintos entre imágenes del mismo `bag_id`.

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

`Label` es un **target oficial, con cobertura completa (1.533/1.533, cero nulos)** y exactamente 3 valores (`1`: 782, `2`: 479, `3`: 272). **Confirmado contra el paper original** (Akbari et al. 2024, Scientific Data, DOI `10.1038/s41597-024-03897-7`, PMCID `PMC11525552` — verificado vía Europe PMC, cita completa en `system/datasets_download/README.md`): la sección de Data Records del paper describe `Label` textualmente como "el grade colectivo" / "la etiqueta de enfermedad asignada por el equipo de 5 médicos", con valores `1 = normal, 2 = preplus, 3 = plus`. El mapeo `1→Normal, 2→Pre-Plus, 3→Plus` ya no es una inferencia estadística nuestra, viene declarado en el paper. El paper no especifica el método exacto de consenso (mayoría simple, unanimidad, etc.) que usaron los 5 médicos para llegar a este valor.

Con esta corrección, **`plus` ahora tiene 100% de cobertura en FARFUM** (antes 0%, quedaba en `None` para todo el dataset).

**Hallazgo adicional corregido:** 3 de los 68 pacientes (`patient44`, `patient55`, `patient57`) tenían la columna `Label` con 2 valores distintos entre sus propias imágenes (siempre categorías adyacentes: Normal/Pre-Plus o Pre-Plus/Plus) — mismo patrón de inconsistencia que se encontró en ROP-VL. Se resolvió igual: `reconcile_bag_plus` deja el paciente con el hallazgo más severo detectado en cualquiera de sus imágenes.

### FARFUM-RoP — consenso entre 5 raters para `stage` y `treatment`

Aparte de `Label`, cada imagen tiene hasta 5 opiniones independientes (`grade`, `stage`, `diagnostic` por rater A-E) que **sí están separadas por rater** (A, B, C, D, E — 5 oftalmólogos, consistente con la descripción oficial del dataset). Se calculó consenso por **mayoría estricta** (>50% de los votos no-nulos, mínimo 3 votos) para `stage` y `treatment`.

**Bug real encontrado y corregido (no era desacuerdo entre médicos, como se documentó en una versión anterior de este archivo):** `_majority_vote` filtraba votos vacíos con `v is not None`, pero `pandas` representa una celda vacía de Excel como `NaN` (float), no como `None` — y `float('nan') is not None` es `True`. Es decir, **cada celda en blanco estaba entrando al conteo de votos como si fuera una respuesta real**, inflando el denominador y haciendo casi imposible alcanzar mayoría estricta. Se corrigió reemplazando el filtro por `pd.notna(v)`, que sí reconoce tanto `None` como `NaN`.

**Impacto medido (antes → después de la corrección, mismo umbral de mayoría estricta, sin cambiar criterio):**
- `stage`: 14.7% → 19.5% de cobertura.
- `treatment`: 17.4% → **55.0%** de cobertura, y por primera vez aparece la categoría `no_treatment` (antes nunca ganaba consenso porque el "voto" `NaN` competía artificialmente contra ella).

`stage` mejoró menos porque ahí la causa principal de los vacíos es otra, real y no un bug — pero la explicación original de esto estaba mal y ya se corrigió (ver nota sobre `grade` abajo): **no es que el rater marque "normal" y por eso no anote stage** — es que el rater se abstiene / no especifica opinión alguna para esa imagen (`grade=0` = "no especificado", confirmado por el paper), y al no opinar, tampoco anota `stage` ni `diagnostic`. Verificado: >98% de las veces que `grade=0`, ese mismo rater deja `stage` y `diagnostic` en blanco, en los 5 raters. Esto reduce el número de votos disponibles para alcanzar el mínimo de 3.

**Idea descartada:** se había dejado pendiente evaluar si convenía imputar `stage=0` cuando `grade==0`, asumiendo que `grade=0` significaba "normal". Con la semántica real confirmada (`grade=0` = "no especificado", no "normal"), esa imputación **queda descartada** — estaría mal fundamentada: imputar "sano" cuando el rater simplemente se abstuvo podría introducir ceros falsos en casos que sí tenían ROP pero ese médico en particular no evaluó esa imagen. No se implementa.

**Nota sobre `grade` (distinto de `Label`):** la columna `grade` (valores 0-3 por rater) **ya tiene semántica confirmada** contra el paper original (Akbari et al. 2024, PMCID `PMC11525552`): `0 = no especificado, 1 = normal, 2 = pre-plus, 3 = plus` — es decir, es la opinión individual de cada rater sobre Plus Disease, previa al consenso. No se usa para construir `plus` (eso ya viene del consenso oficial en `Label`, ver arriba) porque sería redundante. Se conserva sin usar en `raw_grade_per_rater` (los 5 valores separados por `;`) por si resulta útil más adelante (ej. análisis de acuerdo inter-rater).

### ROP-VL (`categories` → `stage`)

```
normal -> 0          Stage 4 ROP -> 4
Stage 1 ROP -> 1      Stage 5 ROP -> 5
Stage 2 ROP -> 2      A-ROP -> stage=None, ap_rop=True
Stage 3 ROP -> 3      Laser-treated ROP -> stage=None, laser_treated=True
```

`Laser-treated ROP` **no se mapea a la columna `treatment`** aunque conceptualmente esté relacionado — es una categoría retrospectiva (el ojo ya fue tratado en algún momento, no sabemos con qué estadio/zona estaba *antes* del láser), mientras que `treatment` en FARFUM es una decisión prospectiva tomada en el momento de la foto. Mezclar ambas definiciones de "tratamiento" bajo la misma columna introduciría ruido de etiqueta. Se preserva como flag `laser_treated` + `raw_diagnosis_code` por si se quiere usar distinto más adelante. ROP-VL no tiene columna de Plus explícita → `plus` siempre `None` para este origen.

### ROP-VL — regla final: cualquier `Laser-treated` elimina el bag completo (decisión conservadora)

Se evaluaron dos opciones para los bags donde una imagen es `Laser-treated ROP` y otra del mismo bag trae un Estadio numérico real o `A-ROP`:
- **A)** usar el Estadio/AP-ROP real igual, ignorando la imagen `Laser-treated` (matemáticamente válido: como `Laser-treated` no tiene estadio numérico, tomar el máximo del bag ya la ignora sola).
- **B)** eliminar el bag completo apenas aparezca `Laser-treated`, sin excepciones.

**Se eligió B** (decisión conservadora, más simple de justificar) después de verificar el costo real: solo 13 bags con Estadio real y 3 de los 148 bags de A-ROP de ROP-VL (2.0%) se pierden — RIDIRP aporta 470 imágenes de AP-ROP independientes, no afectadas por esta decisión. Verificación adicional que llevó a esta decisión: en los 54 bags con categorías mezcladas, **nunca hay mayoría de imágenes** (todos son empate 1 contra 1 por categoría, verificado contando imágenes no solo categorías) — no existía una alternativa de "quedarse con la categoría mayoritaria" que aplicar. Detalle completo y celdas de verificación en `system/eda/eda_ropvl.ipynb`.

**Regla final:**
- Sin `Laser-treated` en el bag: `stage` = máximo entre las imágenes del bag (resuelve los 91 bags de conflicto puramente numérico).
- Bag con `Laser-treated` en cualquiera de sus imágenes (345 bags, 381 imágenes, 18.9% del dataset): **se elimina el bag completo**, sin excepciones.
- Bag puramente `A-ROP` (sin Laser-treated): `stage=NaN`, pero se mantiene para el cabezal AP-ROP.
- `ap_rop`: `True` si el bag (ya filtrado) contiene alguna imagen `A-ROP`, `False` en caso contrario — sin casos ambiguos, porque la única fuente de ambigüedad (`Laser-treated`) ya quedó eliminada.

**Resultado final:** 1.522 de 1.867 bags, 1.639 de 2.020 imágenes. `stage` con 91.7% de cobertura sobre los bags finales; `ap_rop=True` en 145 bags, `False` en 1.377.

## ⚠️ Secciones desactualizadas (pipeline anterior, antes de `clean_ridirp.py`/`clean_farfum.py`)

Las dos secciones siguientes describen la salida de `build_manifest.py` tal como estaba **antes** de separar la limpieza por dataset — usan el consenso viejo de FARFUM (mínimo 3 votos, mayoría estricta: 19.5%/55.0%) en vez del actual (severidad máxima, sin mínimo: 99.0%/100%, ver arriba), y no reflejan que RIDIRP ahora elimina las filas no-ROP. Se quedan como referencia histórica hasta que `build_manifest.py` se reescriba para consumir `datasets/clean/*.parquet` (pendiente #5 arriba) — ahí se documentará el esquema final completo, con descripciones de columna.

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

## Resultados: `clean_ridirp.py`

Esquema y decisiones completas (DG→stage, filas no-ROP eliminadas, `device_raw` solo en esta capa, etc.) documentadas en `system/eda/eda_ridirp.ipynb`, celda final "Decisiones de diseño discutidas". Validado corriendo el script contra los datos reales:

| | Valor |
|---|---|
| Imágenes | 4.709 (de 6.004 crudas — se eliminan 1.295 por diagnóstico no-ROP) |
| Bags (visitas) | 374 (de 484) |
| Pacientes | 150 (de 188 — 38 pacientes quedan sin ninguna visita ROP) |
| `stage` nulo | 18.0% (849 imágenes, AP-ROP + status-post) |
| `plus` nulo | 0.0% |

Verificado: 0 `sample_id`/`image_path` duplicados, 0 estadios fuera de 0-5, 0 bags inconsistentes, 0 paths faltantes.

## Resultados: `clean_farfum.py`

Esquema y decisiones completas (consenso por severidad máxima sin mínimo de votos, reconciliación a nivel de paciente, columna `Label` para `plus`) documentadas en `system/eda/eda_farfum.ipynb`. Validado corriendo el script contra los datos reales:

| | Valor |
|---|---|
| Imágenes | 1.533 (ninguna eliminada — `plus` siempre tiene valor vía `Label`, ninguna fila queda con las 3 tareas nulas a la vez) |
| Pacientes (bags) | 68 |
| `stage` nulo | 1.0% (15 imágenes, 2 pacientes: `patient26` y `patient28` — ningún rater dio estadio para ellos, pero sí tienen `plus`/`treatment` válidos, se mantienen) |
| `plus` nulo | 0.0% |
| `treatment` nulo | 0.0% |

Verificado: 0 duplicados, 0 bags inconsistentes, 0 paths faltantes, exactamente los 3 valores esperados en cada columna categórica.

## Resultados: `clean_ropvl.py`

Esquema y decisiones completas (regla conservadora de eliminación por `Laser-treated`, `stage` por máximo, `ap_rop` por bag) documentadas en "ROP-VL — regla final" arriba y en `system/eda/eda_ropvl.ipynb`. Validado corriendo el script contra los datos reales:

| | Valor |
|---|---|
| Imágenes | 1.639 (de 2.020 crudas — se eliminan 381, el 18.9%, por pertenecer a un bag con `Laser-treated ROP`) |
| Bags | 1.522 (de 1.867 — se eliminan 345) |
| Pacientes | 964 |
| `stage` nulo (por bag) | 8.3% (126 de 1.522 bags — todos `A-ROP` puro, sin estadio numérico) |
| `ap_rop` (por bag) | `True` en 145, `False` en 1.377 |
| `plus` / `treatment` | siempre `None` (ROP-VL no tiene estas etiquetas) |

Verificado: 0 `sample_id`/`image_path` duplicados, `stage` solo toma valores 0-5, 0 bags con `stage`/`ap_rop` inconsistente entre sus imágenes, 0 paths faltantes, 0 imágenes `Laser-treated ROP` sobreviven en el parquet final.

## Pendientes conocidos

1. ~~Confirmar contra el paper original que la columna `Label` de FARFUM corresponde a Normal/Pre-Plus/Plus~~ — **resuelto**: confirmado textualmente contra Akbari et al. 2024 (PMCID `PMC11525552`), ver sección FARFUM arriba.
2. ~~Decidir si se imputa `stage=0` cuando `grade==0`~~ — **resuelto, descartado**: `grade=0` significa "no especificado" según el paper, no "normal" — imputar en base a esa premisa habría sido incorrecto.
3. ~~Escribir `clean_farfum.py`~~ — **resuelto**, ver resultados arriba.
4. ~~Decidir la regla de ROP-VL para bags mixtos (`A-ROP`/`Laser-treated ROP` combinados con `normal` o estadios numéricos) y para el cabezal AP-ROP~~ — **resuelto**: se descartó la idea de "voto mayoritario" (verificado con datos reales: los 54 bags mixtos tienen empate 1 a 1, no existe mayoría) y se eligió la opción conservadora — cualquier bag que toque `Laser-treated ROP` se elimina completo. Ver "ROP-VL — regla final" arriba y `system/eda/eda_ropvl.ipynb`.
5. ~~Escribir `clean_ropvl.py`~~ — **resuelto**, ver resultados arriba. Falta `clean_hvdropdb.py` (siguiente paso; estructura distinta — pares imagen/máscara de segmentación, no filas clínicas).
6. Reescribir `build_manifest.py` para que combine `datasets/clean/*.parquet` en vez de leer el crudo directamente — las secciones "Esquema del manifiesto unificado" y "Cobertura final por dataset y tarea" más abajo describen el pipeline **anterior** (monolítico, consenso con mínimo de 3 votos) y están desactualizadas respecto a `clean_ridirp.py`/`clean_farfum.py`/`clean_ropvl.py`. Se reescriben completas (con descripción de columnas) cuando se arme el parquet unificado final.
7. Escribir el script de partición Group K-Fold que llena `split`/`fold`.

## Cobertura final por dataset y tarea

Verificado corriendo el script contra los datos reales (`df.groupby('dataset_origen')[['stage','plus','treatment']].notna().mean()`):

| Dataset | `stage` | `plus` | `treatment` |
|---|---|---|---|
| RIDIRP | 64.3% | 78.4% | 0% (no tiene esta etiqueta) |
| FARFUM | 19.5% | **100%** | 55.0% |
| ROP-VL | 73.0% | 0% (no tiene esta etiqueta) | 0% (no tiene esta etiqueta) |
