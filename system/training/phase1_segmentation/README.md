# Fase 1 — Pre-entrenamiento Visual (Segmentación, HVDROPDB)

Fine-tuning del backbone ViT en segmentación (vasos/BV, disco óptico/OD, cresta/RIDGE) antes de congelarlo y pasarlo a Fase 2/3 (ver README principal §5). El backbone resultante es el único artefacto que sobrevive de esta fase — el decoder de segmentación se descarta.

```
python system/training/phase1_segmentation/train.py
```

Requiere `datasets/manifest/HVDROPDB.parquet` con la columna `split` ya asignada (`system/dataset_build/partition/split_hvdropdb.py`).

## Decisiones de diseño

- **Librería del backbone: `timm`**, no `torchvision` — tiene variantes ViT preentrenadas directo a 384×384 (`vit_base_patch16_384.augreg_in21k_ft_in1k`), evita interpolar los position embeddings de un checkpoint a 224×224 para estirarlo a 384.
- **Decoder liviano pero multi-escala** (no una U-Net completa, pero tampoco una sola capa del ViT): ver "Revisión 2" abajo — se descartó la primera versión (una sola capa) tras la primera corrida real.
- **Pérdida: Dice Loss + BCE enmascarada por canal** (`MaskedDiceBCELoss`) — cada imagen de HVDROPDB solo tiene máscara real de una estructura; los otros 2 canales no aportan gradiente para esa fila. Mismo principio de enmascarado que la pérdida clínica multitarea del proyecto (README principal §4.4), aplicado aquí a nivel de canal en vez de cabezal. Empezó como Dice puro, se agregó BCE en la Revisión 2 (ver abajo).
- **`model.forward_tokens()`** expone la secuencia completa de tokens de parche (sin CLS) — es el mismo método que usará Fase 2/3 para alimentar la Cross-Attention (README principal §4.1/§4.2).
- **Sin K-Fold** — split único train/val ya calculado (ver README principal §2): Fase 1 es una etapa auxiliar, no el resultado que se reporta en la tesis.
- **LR distinto para backbone vs. decoder** (1e-5 vs. 1e-4, grupos de parámetros en `AdamW`): el backbone ya viene preentrenado y solo necesita afinarse suave; el decoder es nuevo (inicialización aleatoria) y necesita aprender más rápido.

## Hallazgos reales durante la implementación (verificados con código, no asumidos)

- **Las imágenes de HVDROPDB tienen viñeta negra real y las dos cámaras no son cuadradas**: Neo es 2040×2040 (prácticamente sin borde), pero **RetCam es 480×640** (rectangular) con viñeta negra real en los bordes. Un resize directo a 384×384 habría distorsionado la proporción de RetCam. Solución: recorte por bounding-box del contenido no-negro + resize-con-padding (`LongestMaxSize`+`PadIfNeeded` de albumentations) en vez de un resize directo — preserva proporción real, nunca deforma.
- **Las máscaras no son binarias puras**: 89% de los píxeles están exactamente en `{0, 255}`, el resto son valores de transición por antialiasing en los bordes (confirmado con histograma real). Se umbralizan a 0/1 (threshold 127) al cargar, antes de cualquier transformación geométrica.
- **Normalización del checkpoint ≠ ImageNet genérico**: `vit_base_patch16_384.augreg_in21k_ft_in1k` usa `mean=std=0.5` (no los `(0.485, 0.456, 0.406)`/`(0.229, 0.224, 0.225)` típicos) — se obtiene con `timm.data.resolve_data_config`, no se hardcodea.
- **GPU real verificada, no solo instalada**: la máquina tiene una RTX 5070 Laptop (Blackwell, compute capability 12.0). `torch`/`torchvision` instalados desde PyPI normal dan build CPU-only — hubo que instalar explícitamente desde el índice `cu128` de PyTorch, y se verificó con una multiplicación de matrices real en GPU (no solo `torch.cuda.is_available()`) antes de dar por buena la instalación.

## Prueba de humo y primera corrida

Antes de lanzar las 30 épocas completas se verificó, con una muestra y un batch reales:
- `dataset[0]` produce shapes correctos (`imagen: (3,384,384)`, `target: (3,384,384)`, `channel_valid` con exactamente 1 `True`).
- Forward → loss → `backward()` corre sin error.
- Dice sin entrenar es cercano a 0 (confirma que la métrica no está trivialmente satisfecha).
- 2 épocas de prueba: ~21s/época en la RTX 5070, Dice promedio sube de 0.305 a 0.334 ya en la segunda época — confirma que el modelo aprende antes de comprometerse a la corrida completa.

**Resultado de la primera corrida (30 épocas, decoder de 1 sola capa, Dice puro):** mejor `mean_dice=0.4376` (época 15) — `BV=0.289, OD=0.565, RIDGE=0.459`. `BV` quedó prácticamente estancado todo el entrenamiento (~0.26-0.31 desde la época 1 hasta la 30).

## Revisión 2: decoder multi-escala + Dice+BCE (por qué y qué cambió)

**El hallazgo que motivó esto:** se verificó la fracción real de píxeles positivos por estructura — `BV=6.78%`, `OD=0.57%`, `RIDGE=2.03%`. **`BV` tiene MÁS píxeles positivos que `OD`, no menos**, y aun así le fue mucho peor (0.289 vs 0.565) — descarta el desbalance de clases como explicación principal. La explicación real: `OD` es una mancha compacta (fácil de localizar incluso con poca resolución espacial), `BV` es una red de vasos finos y ramificados dispersos por toda la imagen — exactamente el tipo de estructura que más sufre cuando el decoder solo usa la última capa del ViT, donde el detalle espacial fino ya se perdió (es conocido en la literatura de interpretabilidad de ViT que las capas tempranas retienen más detalle local, las tardías se vuelven más semánticas/globales).

**Cambios aplicados (los 3 juntos, no en corridas separadas — ver discusión en el chat de planeación):**
1. **Decoder multi-escala** (`model.MultiScaleSegmentationDecoder`): en vez de decodificar solo desde la última capa, se extraen tokens de 4 profundidades del ViT (bloques 2, 5, 8, 11 de 12, vía `timm`'s `forward_intermediates`), se proyectan por separado y se fusionan antes del mismo camino de upsampling — estilo DPT (*Dense Prediction Transformer*) simplificado, sin la reasignación de resolución completa del paper original. El backbone sigue intacto para Fase 2/3 (`forward_tokens()`, que solo usa la última capa, no cambió).
2. **Pérdida combinada Dice+BCE** (`losses.MaskedDiceBCELoss`, 50/50): el Dice Loss puro tiene gradiente débil cuando la predicción casi no se solapa con la máscara real — problema conocido para estructuras finas/dispersas. BCE penaliza error píxel a píxel de forma más estable.
3. **Más épocas** (30 → 60): la curva de `train_loss` de la primera corrida seguía bajando al final, sin señal clara de convergencia total.

**Se descartó (por ahora) cambiar de backbone a una CNN** — contradiría la decisión arquitectónica de usar ViT (README principal §4.1): Fase 1 existe justo para afinar el backbone que Fase 2/3 reutiliza vía Cross-Attention, una CNN no da la secuencia de tokens que ese mecanismo necesita. El decoder multi-escala captura, en espíritu, el beneficio de "mirar múltiples resoluciones" de una U-Net, sin abandonar el ViT.

**Smoke test de la revisión 2** (2 épocas): `mean_dice=0.4260` ya en la época 2 — comparable al MEJOR resultado de las 30 épocas completas de la primera corrida, alcanzado en apenas 2 épocas.

**Resultado de la corrida completa (60 épocas, decoder multi-escala, Dice+BCE):** mejor `mean_dice=0.6012` (época 50) — `BV=0.407, OD=0.815, RIDGE=0.581`. Mejora sobre la revisión 1 en las 3 estructuras (BV +41%, OD +44%, RIDGE +27% relativo). `val_loss` pasó de fluctuar sin control (0.80-0.91) a bajar suave y consistente (0.73→0.56) — convergencia real, no solo un número final mejor. Las últimas ~10 épocas se estabilizan en `mean_dice≈0.59-0.60`, sin señal de que más épocas por sí solas sigan ayudando mucho.

**Lo que no se resolvió del todo:** `BV` sigue siendo la estructura más débil de las 3 por un margen claro (0.41 vs 0.58-0.82) — mejoró sustancialmente, pero la dificultad relativa de segmentar vasos finos y ramificados no desapareció.

## Revisión 3 (probada y revertida): augmentation adicional no ayudó

Se probó sumar contraste (ya estaba la opción, deshabilitada), zoom/traslación leve (`A.Affine`, escala 0.9-1.1) y deformación elástica suave (`A.ElasticTransform`, alpha=15/sigma=5, p=0.3) — con la hipótesis de que más variedad de encuadre/forma ayudaría específicamente a `BV`. **Resultado: empeoró el promedio** (`mean_dice` 0.6012 → 0.5881): `BV` prácticamente no cambió (0.407→0.403, no logró el objetivo), `OD` bajó notablemente (0.815→0.744, -8.7%), `RIDGE` mejoró algo (0.581→0.618, +6.4%) pero no compensó la caída de `OD`. Interpretación: el zoom/deformación le costó a `OD` (mancha compacta, menos tolerante a verse distorsionada/movida que a mejorar por variedad extra) más de lo que le aportó a `BV`. **Revertido** — `dataset.py` volvió a la augmentation de la Revisión 2 (solo rotación completa, flip horizontal, jitter de brillo).

Checkpoint final (vigente, Revisión 2): `checkpoints/backbone_best.pt` (época 50, `mean_dice=0.6012`) — es el backbone que se reutiliza (congelado) en Fase 2/3.
