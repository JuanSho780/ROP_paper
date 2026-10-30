# Fase 2 — Warm-up Multimodal (visual congelado)

Entrena MLP clínico + Cross-Attention por imagen + Attention Pooling (MIL) + 3 cabezales (Estadio/Plus/AP-ROP) sobre ROP-VL + FARFUM, con el backbone visual de Fase 1 **congelado**. Se repite una vez por cada uno de los 5 folds del Group K-Fold clínico (README principal §2: "solo Fase 2/3 se repite K veces") — cada fold entrena módulos nuevos desde cero, usa solo su propio train para normalización/pesos de clase, y guarda su propio checkpoint (punto de partida de la Fase 3 de ese mismo fold).

```
python system/training/phase2_warmup/precompute_tokens.py   # cachea tokens del backbone congelado (una vez)
python system/training/phase2_warmup/train.py                # entrena los 5 folds
```

## Decisiones de diseño

- **Preprocesamiento de imagen compartido con Fase 1** (`system/training/common/image_preprocessing.py`, extraído de `phase1_segmentation/dataset.py` para esta fase) — crítico que sea idéntico al que vio el backbone al entrenarse, o los tokens quedan fuera de distribución.
- **Tokens precalculados y cacheados** (`precompute_tokens.py`, 3.172 archivos `.pt`, 2.7GB en float16) — el backbone está congelado y los tokens no cambian entre épocas (README principal §5), así que correrlo una sola vez evita recalcularlo en cada época. Consecuencia directa: **sin augmentation de imagen en Fase 2** (si la imagen cambiara por época, los tokens cacheados dejarían de ser válidos).
- **Cross-Attention unidireccional** (Query=clínico, Key/Value=visual, nunca al revés) — el lado clínico es un solo vector (no una secuencia), así que la dirección contraria se degenera a una inyección lineal, no atención real. Implementado con `nn.MultiheadAttention(kdim=768, vdim=768)` para manejar dimensiones distintas entre Query y Key/Value.
- **Cap de tamaño de bag = 8** (`MAX_BAG_SIZE`, `dataset.py`): verificado con datos reales que FARFUM promedia ~22.5 imágenes/bag y llega a 46 (ROP-VL casi siempre 1-3). Sin límite, un solo bag grande en un batch obliga a rellenar todos los demás hasta su tamaño — desperdicio de cómputo impredecible. En train, el subconjunto de 8 se resamplea al azar en cada `__getitem__` (distinto por época, sin pérdida permanente de datos a lo largo del entrenamiento); en val, la selección es fija para que la métrica no fluctúe por azar.
- **Sin pesos de incertidumbre `$\sigma_i$`** (`losses.py`) — pérdida = suma simple de las 3 tareas enmascaradas. Se activan recién en Fase 3 (ver discusión de diseño: activarlos desde el arranque, con módulos recién inicializados y cobertura muy desigual por tarea entre ROP-VL/FARFUM, es propenso a que `$\sigma_i$` aprenda valores ruidosos).
- **Optimizador: AdamW, un solo learning rate** para todos los módulos nuevos (a diferencia de Fase 1, acá nada viene preentrenado — MLP, Cross-Attention, Pooling y cabezales arrancan todos con pesos aleatorios, no hay razón para separar LR por módulo como con backbone-vs-decoder en Fase 1). Backbone fuera del optimizador por completo (`requires_grad=False`, ni se le calcula gradiente).
- **Métrica de validación: balanced accuracy por cabezal** (promedio de recall por clase, no accuracy simple) — más robusta dado el desbalance real de `plus`/`ap_rop` dentro de Fase 2.

## Hallazgos reales durante la implementación

- **`sexo` tiene 4 valores de texto, no 2**: ROP-VL usa `male`/`female`, FARFUM usa `boy`/`girl` para lo mismo (verificado con crosstab real — sin solapamiento entre fuentes). Se normaliza a un binario `{male,boy}=1, {female,girl}=0` antes de pasarlo al MLP — sin esto, el modelo vería 4 categorías donde solo hay 2 reales.
- **Normalización de GA/BW y pesos de clase se calculan por fold, solo del train de ese fold** — nunca del val/test, para no filtrar estadísticas entre splits (mismo cuidado metodológico que ya aplicamos en el Group K-Fold).
- **Bag de tamaño fijo (post-cap) simplifica el collate**: como todo bag queda en exactamente `MAX_BAG_SIZE` posiciones (reales + padding), el collate por defecto de PyTorch alcanza — no hizo falta un `collate_fn` custom.

## Prueba de humo

1 fold, 3 épocas, ~18s/época: `bal_acc` sube en los 3 cabezales (`stage` 0.386→0.494, `plus` 0.333→0.800, `ap_rop` 0.754→0.797) — confirma que el modelo aprende antes de lanzar los 5 folds completos (~44 min estimados, 30 épocas cada uno).
