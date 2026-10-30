# Fase 3 — Fine-Tuning End-to-End (backbone descongelado + RIDIRP + PMA + σᵢ)

Red completa entrenable: backbone ViT (ya no congelado) + MLP clínico (ahora con PMA) + Cross-Attention + Attention Pooling + 3 cabezales, sobre las **3** fuentes clínicas (RIDIRP + ROP-VL + FARFUM). Se repite una vez por cada uno de los 5 folds, continuando desde el checkpoint de Fase 2 de ese mismo fold + el backbone de Fase 1 (warm-start doble).

```
python system/training/phase3_finetune/train.py
```

**Local (8GB VRAM) vs Colab:** este código corre tal cual en ambos — ver `system/training/phase3_finetune_colab/` para el notebook que lo ejecuta en una GPU con más VRAM (`batch_size=8`+`MAX_BAG_SIZE=8` no entra de forma confiable en 8GB con el backbone descongelado, verificado con un `OutOfMemoryError` real).

## Decisiones de diseño

- **Vector clínico ampliado a 5 features**: `[GA, BW, sexo, PMA, pma_available]` (Fase 2 usaba solo 3, sin PMA). `PMA` real solo en RIDIRP (100% de sus filas); imputado por media del subconjunto con dato real + bandera binaria para ROP-VL/FARFUM (README principal §3.2).
- **Pérdida completa con `$\sigma_i$`** (`losses.Phase3Loss`) — a diferencia de Fase 2 (pesos fijos), acá se activa el peso de incertidumbre homocedástica aprendido por tarea. Parametrizado como `log_var_i = log(σᵢ²)` (truco de Kendall & Gal 2017 — evita restringir σᵢ>0 explícitamente), inicializado en 0 (σᵢ=1, mismo punto de partida que la suma simple de Fase 2).
- **Warm-start doble**: backbone desde `phase1_segmentation/checkpoints/backbone_best.pt`, MLP/Cross-Attention/Pooling/cabezales desde `phase2_warmup/checkpoints/fold{k}_best.pt` del **mismo fold** — nunca se mezclan folds.
- **LR diferenciado**: backbone más lento (`1e-5`, ya viene afinado dos veces), resto del modelo más rápido (`5e-5`, calibrado por Fase 2 pero no al azar como en Fase 2).
- **Augmentation de imagen reactivada** (rotación completa, flip horizontal, jitter de brillo) — a diferencia de Fase 2 (sin augmentation, tokens cacheados), acá el backbone vuelve a tener gradiente real, mismo riesgo de sobreajuste que motivó usar augmentation en Fase 1.
- **Backbone solo procesa imágenes reales, nunca el padding**: el padding de bag siempre da gradiente exactamente 0 (por la máscara en Attention Pooling), pero sin este cuidado igual se gastaría cómputo real (forward+backward por el ViT completo) en imágenes puro relleno — ahora se filtran por índice booleano antes de llegar al backbone, y los tokens de padding se rellenan con ceros después, sin pasar por él.
- **`NUM_WORKERS` calculado dinámicamente** (`min(cpu_count-1, 4)`), no hardcodeado — se adapta a la máquina (local o Colab) sin editar código, con un techo de 4 (ver hallazgo de la DLL más abajo).
- **Reanuda folds ya completados** (`main()` revisa si `checkpoints/fold{k}_best.pt` ya existe antes de entrenar) — pensado para Colab gratuito, donde la sesión se puede desconectar a mitad de los 5 folds.

## Hallazgos reales durante la implementación

- **`sexo` tiene un TERCER vocabulario**: RIDIRP usa `M`/`F` (además de `male`/`female` en ROP-VL y `boy`/`girl` en FARFUM, ya conocido de Fase 2) — verificado con un crosstab real antes de asumir. El mapeo se centralizó en `system/training/common/clinical_features.py`.
- **Con RIDIRP sumado, el 16.4% de los bags supera el cap=8** (no solo FARFUM — RIDIRP también llega a bags de hasta 58 imágenes), contra el 3.7% que afectaba en Fase 2 (solo ROP-VL+FARFUM).
- **Memoria de GPU real, no solo estimada**: con warm-start cargado, `batch_size=4`+cap=8 usa ~3.5GB; un micro-benchmark de `batch_size=8` dio ~4GB, pero la corrida real completa hizo `OutOfMemoryError` (`torch.OutOfMemoryError`, "Tried to allocate 110 MiB... 2.54 GiB is free") — la composición de cada batch es aleatoria (cuántos bags tienen casi 8 imágenes reales a la vez), así que el pico de memoria varía batch a batch; `batch_size=8`+cap=8 queda demasiado al límite de 8GB para confiar en que siempre entre.
- **`num_workers` alto rompe DLLs de CUDA en Windows**: con `num_workers=15` (todos los núcleos menos uno), los procesos worker inicializando CUDA/cuDNN al mismo tiempo generaron una condición de carrera real (`WinError 1114`, fallo cargando `cudnn_engines_runtime_compiled64_9.dll`). Un proceso zombie de ese crash quedó con el contexto CUDA corrupto y rompió intentos posteriores incluso con `num_workers` bajo, hasta matarlo manualmente — de ahí el tope de 4 en el cálculo dinámico.
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** (sugerido por el propio mensaje de error de OOM) **no está soportado en Windows** (warning real al usarlo) — no ayuda en este entorno, descartado.

## Prueba de humo

Config reducida (`batch_size=2`, `MAX_BAG_SIZE=4`) solo para validar correctitud, no la corrida real: 2 épocas, `mean_bal_acc` 0.57→0.74, σᵢ moviéndose levemente desde 1.0 — confirma que warm-start, backbone descongelado, PMA y la pérdida con incertidumbre funcionan de punta a punta antes de correr la versión completa en Colab.
