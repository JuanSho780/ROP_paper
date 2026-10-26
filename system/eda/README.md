# EDA - Analisis Exploratorio por Dataset

Cuatro notebooks, uno por dataset, cada uno explorando el **dato crudo** (xlsx/zip/rar originales en `datasets/<Nombre>_Original/`, o ya extraidos en `datasets/extracted/<Nombre>/`), independiente de las decisiones de mapeo que aplica `system/dataset_build/`. El objetivo es entender cada fuente por separado antes de unificarlas.

Un quinto notebook, `eda_unified.ipynb`, es distinto en naturaleza: explora `datasets/manifest/rop_unified_manifest.parquet`, el resultado **ya unificado** (RIDIRP+FARFUM+ROP-VL, cada uno ya pasado por su `clean_<dataset>.py` y por `build_manifest.py`) - ver su propia seccion mas abajo.

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

300 pares imagen/mascara de segmentacion (50 x 2 camaras x 3 estructuras), pareo verificado 1:1. El subset de Clasificacion (189 imagenes Normal/ROP) se documenta pero se deja fuera del manifiesto por no tener metadatos clinicos. Hallazgo corregido en este notebook: las 3 estructuras NO son colecciones de fotos independientes - 38 de las 300 imagenes (12.7%) son el mismo archivo fuente reciclado entre dos carpetas de estructura (verificado por hash MD5 exhaustivo, no solo un indice de muestra). Por eso el split train/val de Fase 1 (`split_hvdropdb.py`) agrupa por `image_hash`, no por fila.

## `eda_unified.ipynb` (sobre el manifiesto YA unificado, no el crudo)

7.881 imagenes, 1.964 bags, 1.182 pacientes (RIDIRP 150 + FARFUM 68 + ROP-VL 964). Objetivo: desbalance de clases y patrones de nulos a nivel del **pool combinado**, como insumo directo para el script de Group K-Fold (pendiente). Hallazgos clave:
- `stage`: las 3 fuentes tienen formas muy distintas, no solo volumen - RIDIRP es 85.5% Estadio 0 (bags con stage valido), FARFUM es **0% Estadio 0** (nunca "normal"), ROP-VL es la unica con Estadio 4/5 (84 bags) y la mas balanceada. Estratificar solo por `dataset_origen` no alcanza para el K-Fold.
- `plus`: confirma el hallazgo cross-dataset de abajo (38 pacientes reales en todo el pool) directamente desde el manifiesto unificado, sin tener que cruzar archivos por separado.
- `ap_rop`: 10.6% positivo entre filas con dato (634/5.969). El combo `stage=NaN, plus=valido, ap_rop=True` (16 bags) es **exclusivamente** el paciente `RIDIRP_043` (verificado 16/16) - la señal de esa combinacion depende de un solo bebe.
- Patron de nulos cruzado: 0 bags sin ninguna de las 3 tareas (confirma que la limpieza de no-ROP de RIDIRP funciono). Las 6 combinaciones de nulos observadas son 100% explicables por el diseño de cada fuente.
- `RIDIRP_043` (16 bags, hasta 470 imagenes) es el paciente mas grande del pool combinado - riesgo real de que un fold se quede corto de señal AP-ROP si lo excluye.

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

Reproducible directo desde `rop_unified_manifest.parquet` agrupando por `patient_id_global` y viendo si `plus` fue alguna vez "Plus"/"Pre-Plus" en cualquiera de sus visitas/imagenes - ya no hace falta cruzar `clean_ridirp.parquet` por separado (RIDIRP ya esta dentro del manifiesto unificado). Verificado con codigo real en `eda_unified.ipynb`, celda "Distribucion de `plus`".

## Relacion con `system/dataset_build/`

Los 4 notebooks por dataset son exploratorios y **no escriben nada** - las decisiones de mapeo, consenso y reconciliacion que sí se aplican al manifiesto final estan en `system/dataset_build/clean/clean_<dataset>.py` y `system/dataset_build/partition/build_manifest.py`, documentadas en el README de esa carpeta. Donde los numeros de este EDA difieren levemente de los de `dataset_build` (ej. la particion exacta de bags inconsistentes en ROP-VL), se debe a que el EDA mide el dato crudo o un paso intermedio, no el resultado final tras reconciliacion - se deja la nota explicita en el notebook correspondiente. `eda_unified.ipynb` es la excepcion: lee directamente la salida final de `build_manifest.py`, asi que sus numeros son exactos, no aproximados.
