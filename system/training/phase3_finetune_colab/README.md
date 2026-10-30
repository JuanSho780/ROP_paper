# Fase 3 en Colab — misma lógica, solo que para ejecutar con más VRAM

`system/training/phase3_finetune/` corre localmente, tal cual — **no se tocó**. Esta carpeta es aparte, solo contiene lo necesario para correr exactamente ese mismo código (clonado del repo, sin copiarlo/duplicarlo) en una GPU de Colab, porque localmente (RTX 5070 Laptop, 8GB VRAM) `batch_size=8` + `MAX_BAG_SIZE=8` con el backbone descongelado da `OutOfMemoryError` real (verificado) — la composición de cada batch es aleatoria, así que a veces entra y a veces no.

## Por qué hace falta esto

- Backbone descongelado (Fase 3, a diferencia de Fase 2) = cada imagen del bag pasa por el ViT completo (forward+backward) — mucha más memoria de GPU por imagen que Fase 2 (que solo leía tokens cacheados).
- Con `MAX_BAG_SIZE=8` sin reducir (decisión explícita: no comprometer el diseño para que entre en 8GB, preferible correrlo completo en una GPU con más VRAM).

## Qué hay acá

- `ROP_paper_phase3_colab.ipynb` — notebook de Colab, 17 celdas, pensado para correrse de principio a fin sin editar nada salvo la ruta de Drive si usaste otro nombre de carpeta.

## Qué subir a Google Drive antes de correrlo

Una carpeta `MyDrive/ROP_paper_colab_data/` con:

```
ROP_paper_colab_data/
├── extracted/
│   ├── RIDIRP/      (2.5GB)
│   ├── FARFUM/       (2.1GB)
│   └── ROP-VL/        (673MB)
├── manifest/
│   └── rop_unified_manifest.parquet
├── checkpoints_phase1/
│   └── backbone_best.pt   (329MB)
└── checkpoints_phase2/
    └── fold0_best.pt ... fold4_best.pt   (18MB los 5)
```

Total ≈ 5.65GB. **No hace falta `HVDROPDB` ni `cached_tokens`** (eran solo para Fase 1/2 — Fase 3 no cachea tokens, calcula en vivo porque el backbone cambia).

## Qué hace el notebook, en orden

1. Verifica GPU asignada.
2. Monta Drive.
3. Clona el repo de GitHub (`JuanSho780/ROP_paper`) — **el código tiene que estar pusheado antes de correr esto**.
4. Instala solo lo que falta (`timm`, `albumentations`, `opencv-python-headless`) — no reinstala `torch`/`torchvision` para no romper la build de CUDA que Colab ya trae preinstalada y emparejada con su propio driver.
5. Copia los datos de Drive al disco local de Colab (más rápido para I/O de miles de imágenes que leer directo desde Drive montado).
6. Verifica que el manifiesto, los 3 datasets y los 6 checkpoints de warm-start estén en su lugar — antes de lanzar horas de entrenamiento, no después.
7. Corre `train.py` **sin modificar**, con `BATCH_SIZE=8`/`MAX_BAG_SIZE=8` tal cual — `NUM_WORKERS` se recalcula solo según los núcleos reales de la máquina de Colab (misma lógica que local).
8. Copia los checkpoints resultantes de vuelta a Drive (el disco local de Colab se borra al cerrar la sesión).

## Resiliencia a desconexión (Colab gratuito)

`phase3_finetune/train.py` ahora **reanuda** folds ya completados — si Colab se desconecta a mitad de los 5 folds, basta con volver a correr el notebook desde el paso 1 (montar Drive) hasta el paso 7 (entrenar): los folds que ya tienen checkpoint guardado se omiten automáticamente, no se reentrenan de cero. Por eso el paso 8 (copiar a Drive) conviene correrlo periódicamente, no solo al final.
