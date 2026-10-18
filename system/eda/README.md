# EDA - Analisis Exploratorio por Dataset

Un notebook por dataset, cada uno explorando el **dato crudo** (xlsx/zip/rar originales en `datasets/<Nombre>_Original/`, o ya extraidos en `datasets/extracted/<Nombre>/`), independiente de las decisiones de mapeo que aplica `system/dataset_build/`. El objetivo es entender cada fuente por separado antes de unificarlas.

Requiere: `pandas`, `matplotlib`, `pillow`, `openpyxl`, `rarfile` (solo para HVDROPDB), y el kernel de Jupyter del `.venv` del proyecto (`../../.venv`). Cada notebook asume que se ejecuta con el directorio de trabajo en `system/eda/` (abierto normalmente desde Jupyter/VSCode, o vía `jupyter nbconvert --execute` desde esta carpeta).

## `eda_ridirp.ipynb`

188 pacientes, 6.004 imagenes, 484 visitas longitudinales (hasta 16 visitas en un mismo paciente). Hallazgos clave:
- Hay que usar `images_stack_without_captions.zip`, no `images.zip` (esta ultima tiene texto de paciente/fecha quemado en el pixel para los dispositivos 2 y 3 - demostrado con comparacion de hashes).
- La columna `series` del xlsx esta mal asignada para ~31 filas del dispositivo 3 (numero repetido en vez de incremental) - hay que alinear por PMA ascendente, no por esa columna.
- 0 casos "Pre-Plus" en todo el dataset, y de los 150 pacientes que quedan tras `clean_ridirp.py` (se eliminan 38 por ser 100% diagnosticos no-ROP) solo **3 tienen `Plus` alguna vez** - ver "Plus/Pre-Plus a nivel de paciente real" mas abajo.

## `eda_farfum.ipynb`

68 pacientes, 1.533 imagenes, 5 oftalmologos (A-E) por imagen. Hallazgos clave:
- El xlsx de labels tiene un encabezado de 2 filas; la columna `Label` (target oficial de Plus, 100% poblada) queda con nombre en blanco si se lee mal y es facil perderla.
- Bug de `NaN` vs `None` en el filtrado de votos: corregirlo subio la cobertura de consenso de `treatment` de 17.4% a 55.0% y de `stage` de 14.7% a 19.5%.
- 3 de 68 pacientes tienen `Label` inconsistente entre sus propias imagenes.

## `eda_ropvl.ipynb`

1.116 pacientes, 2.020 imagenes, 1.867 bags (paciente, fecha, ojo). Hallazgos clave:
- No es puramente transversal: 190 pacientes tienen mas de una fecha (hasta 5 visitas).
- 145 bags (7.8%) tienen imagenes del mismo ojo+fecha con categoria distinta - 91 se resuelven tomando el estadio maximo, 54 mezclan A-ROP/Laser-treated y no se resuelven asi (quedan `stage=NaN`).

## `eda_hvdropdb.ipynb`

300 pares imagen/mascara de segmentacion (50 x 2 camaras x 3 estructuras), pareo verificado 1:1. El subset de Clasificacion (189 imagenes Normal/ROP) se documenta pero se deja fuera del manifiesto por no tener metadatos clinicos.

## Plus/Pre-Plus a nivel de paciente real (hallazgo cross-dataset)

El conteo de **imagenes** con Plus/Pre-Plus puede ser enganoso sobre cuanta diversidad real de pacientes hay detras de esa etiqueta - un paciente con muchas visitas/fotos infla el conteo de imagenes sin aportar diversidad. Contando por **paciente real** (no por imagen) en las 3 fuentes clinicas:

| Dataset | Normal (pacientes) | Pre-Plus | Plus |
|---|---|---|---|
| RIDIRP | 147 | **0** | 3 |
| FARFUM | 33 | 20 | 15 |
| ROP-VL | — (no tiene esta etiqueta) | — | — |
| **Unificado** | 180 | **20** | **18** |

**RIDIRP por si solo tiene un problema real de diversidad**: toda su clase Plus depende de 3 bebes (principalmente uno, el paciente 043 con 16 visitas de AP-ROP, que por si solo genera cientos de las imagenes "Plus"). Si se entrenara solo con RIDIRP, un Group K-Fold mal hecho podria dejar folds enteros sin ningun ejemplo de Plus.

**Al unificar con FARFUM, el problema se diluye de verdad** (no solo en apariencia): FARFUM aporta 35 pacientes reales distintos con Pre-Plus/Plus (no solo imagenes repetidas de pocos bebes), llevando el pool combinado a 38 pacientes con alguna forma de Plus. Sigue habiendo desbalance de *frecuencia* (38 vs 180 Normal), resoluble con las tecnicas ya planteadas (Focal Loss con pesos, `WeightedRandomSampler`), pero ya no es un problema de diversidad casi inexistente.

**ROP-VL no aporta nada a esta tarea** (no tiene columna de Plus, confirmado: 0 pacientes con esta etiqueta de los 1.116).

**Implicacion para el script de particion (pendiente, no resuelto aqui):** al hacer el Group K-Fold, conviene verificar explicitamente que los 38 pacientes con Plus/Pre-Plus (de RIDIRP+FARFUM combinados) queden razonablemente repartidos entre folds, en vez de asumir que el azar lo resuelve - con un grupo tan chico, un mal reparto es facil que ocurra sin querer.

Reproducible con: `clean_ridirp.parquet` (RIDIRP) + `rop_unified_manifest.parquet` filtrado por `dataset_origen` (FARFUM/ROP-VL), agrupando por `patient_id_global` y viendo si `plus` fue alguna vez "Plus"/"Pre-Plus" en cualquiera de sus visitas/imagenes.

## Relacion con `system/dataset_build/`

Estos notebooks son exploratorios y **no escriben nada** - las decisiones de mapeo, consenso y reconciliacion que sí se aplican al manifiesto final estan en `system/dataset_build/build_manifest.py` y documentadas en su README. Donde los numeros de este EDA difieren levemente de los de `dataset_build` (ej. la particion exacta de bags inconsistentes en ROP-VL), se debe a que el EDA mide el dato crudo o un paso intermedio, no el resultado final tras reconciliacion - se deja la nota explicita en el notebook correspondiente.
