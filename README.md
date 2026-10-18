# Arquitectura y Estrategia de Entrenamiento Multimodal para ROP

Documento de diseño para un modelo de IA multimodal basado en *Cross-Attention* y *Multiple Instance Learning* (MIL), diseñado para clasificar la Retinopatía del Prematuro (ROP) fusionando imágenes de fondo de ojo y variables clínicas tabulares.

## Caso de Uso (Despliegue)

El escenario de producción confirmado: un médico sube **1 o varias fotos del examen de hoy** (mismo encuentro clínico, mismo paciente, mismo día) y el sistema predice el estado actual (Estadio, Plus, y si hay signos de AP-ROP). La decisión de tratamiento queda en manos del médico — el sistema describe hallazgos, no prescribe manejo clínico (ver §0 para el razonamiento). Esto es exactamente lo que representa **un bag de MIL** (ver §4) — el mecanismo de Cross-Attention por imagen + Attention Pooling ya está diseñado para bags de tamaño variable (1 a N fotos), así que no se necesita ningún componente adicional para soportar "una o varias fotos" en producción.

**Explícitamente fuera de scope:** el sistema no usa historial de exámenes anteriores del mismo bebé para predecir el estado actual — no hay modelo de secuencia entre visitas (sin RNN/Transformer entre bags). Cada examen se evalúa de forma independiente. Esto es intencional y simplifica el despliegue: no se requiere que el médico tenga acceso a fotos previas del mismo paciente, ni que el sistema mantenga un historial por paciente entre sesiones. Si en el futuro se quisiera incorporar historial, la opción más simple sería agregar features clínicos derivados (ej. "último estadio registrado") al vector de la visita actual, no construir un modelo de secuencia.

Esto también aclara por qué el PMA (edad postmenstrual, disponible solo en RIDIRP — ver §1) **no es un requisito** del sistema: es un campo opcional del vector clínico que, cuando está disponible, enriquece el Query de la Cross-Attention, pero el modelo funciona igual (con un vector clínico reducido) cuando no está presente.

## Estructura del proyecto

```
ROP_paper/
├── README.md              # este documento: arquitectura y estrategia
├── .gitignore
├── datasets/               # datos crudos + extraídos + manifiesto (no versionado)
└── system/                 # todo el código del pipeline de datos
    ├── datasets_download/  # notebooks de descarga, uno por dataset (ver su README)
    ├── dataset_build/       # extracción + construcción del parquet unificado (ver su README)
    └── eda/                 # análisis exploratorio, un notebook por dataset
```

## 0. Alcance (resultado de la auditoría de datasets)

El modelo optimiza **3 cabezales**:

- **Estadio** (0-5, ordinal).
- **Enfermedad Plus** (Normal / Pre-Plus / Plus).
- **AP-ROP** (binario: ¿hay signos de Retinopatía Agresiva Posterior?).

**Decisión de alcance: se reemplazó Tratamiento por AP-ROP** (no se agregó como 4º cabezal, se sustituyó). Dos razones:
1. **Soporte de datos:** Tratamiento solo tenía ground truth en FARFUM-ROP (68 pacientes, una sola fuente — límite de generalización real). AP-ROP tiene soporte de **2 fuentes independientes** (RIDIRP: 470 imágenes: ROP-VL: 171 imágenes), más del doble de respaldo y sin depender de un solo sitio/población.
2. **Alcance clínico más seguro y útil:** un sistema que **describe hallazgos** (estadio, plus, AP-ROP) es más defendible que uno que **prescribe una decisión de manejo** (tratar o no) — la decisión de tratamiento real depende de más factores que la imagen sola (criterios ETROP con Zona incluida, juicio clínico, seguimiento disponible). Dejar la decisión de tratamiento al médico, dándole toda la información diagnóstica necesaria, es el diseño más apropiado para el caso de uso (ver "Caso de Uso" arriba).

**Zona (I/II/III) queda fuera del scope formal del modelo.** Ninguno de los cuatro datasets públicos disponibles (RIDIRP, FARFUM-ROP, ROP-VL, HVDROPDB) incluye esta etiqueta, y no es derivable de forma confiable a partir de las otras variables. Se documenta como limitación y posible trabajo futuro (vía un dataset adicional con anotación clínica de Zona, o vía estimación semi-automática a partir de las máscaras de vascularización de HVDROPDB).

**"AP-ROP" (RIDIRP) y "A-ROP" (ROP-VL) son la misma entidad clínica** — verificado contra la descripción textual que cada dataset trae de sí mismo, no asumido por similitud de nombre:
- **RIDIRP** (hoja `diagnosis`, DG=8) la describe literalmente como: *"aggressive posterior retinopathy of prematurity"* — el nombre completo de "AP-ROP" deletreado.
- **ROP-VL** (`disease_description.xlsx`, categoría "A-ROP") la describe con los hallazgos clínicos exactos que definen a la AP-ROP según ICROP: *"severe vasoconstriction; capillary nonperfusion; nonphysiologic dilated vascular loops; arteriovenous shunts; plus disease in zone 1; (...) diffuse flat extraretinal neovascularization (...) and severe plus disease; (...) ill-defined junction between vascular and avascular retina in zone 1"* — afectación en zona 1, plus severo, neovascularización **plana** (no la cresta clásica), progresión agresiva.

Son dos convenciones de abreviación distintas ("AP-ROP" vs "A-ROP") que usaron los dos equipos de investigación para la misma entidad clínica estándar — se fusionan con confianza en un solo cabezal `ap_rop`.

## 1. Ecosistema de Datos (Ground Truth)

| Dataset | Pacientes / Imágenes | Qué etiqueta aporta | Qué NO aporta |
|---|---|---|---|
| **RIDIRP (2024)** | 188 pac. / ~6.677 imágenes (series longitudinales) | Estadio (0-5, desbalanceado), Plus (normal/plus — sin casos pre-plus), **AP-ROP** (470 imágenes), **Edad Postmenstrual real por visita** | Zona |
| **FARFUM-ROP (2023)** | 68 pac. / 1.533 imágenes (multi-ángulo, sesión única) | Plus (columna `Label`, consenso oficial del panel de 5 médicos, confirmado contra el paper — Akbari et al. 2024, PMCID `PMC11525552`), Estadio (parcial, por consenso entre raters) | Zona, Edad Postmenstrual, AP-ROP (no se distingue en este dataset) — ya no aporta una etiqueta exclusiva propia tras retirar Tratamiento del alcance, pero sigue siendo la única fuente con 100% de cobertura en Plus |
| **ROP-VL (2025)** | 1.116 pac. / 2.020 imágenes | Estadio (0-5 completo, la fuente más balanceada), **AP-ROP** (171 imágenes) | Zona, Edad Postmenstrual, Plus explícito |
| **HVDROPDB v3 (2024)** | ~50 imágenes únicas × 2 cámaras | Máscaras de segmentación (vasos, cresta, disco óptico) — **sin metadatos clínicos** | Cualquier etiqueta clínica/diagnóstica |

Los datos textuales presentes en ROP-VL se descartan en el preprocesamiento; solo se usan sus imágenes y metadatos biológicos estructurados.

## 2. Partición de Datos y Prevención de Fuga

- **Group K-Fold por `Patient_ID` global** (`dataset_origen + "_" + id_original`, para evitar colisión de IDs entre datasets), aplicado **dentro de cada una de las 3 fuentes clínicas** (RIDIRP, FARFUM-ROP, ROP-VL). Todas las imágenes, ojos y visitas longitudinales de un mismo paciente van al mismo bloque.
- **Test set principal:** ~15-20% estratificado por Estadio (para no perder las clases raras 4/5), reservado de las 3 fuentes clínicas combinadas, nunca usado en ninguna fase de entrenamiento.
- **Análisis secundario (robustez/dominio):** evaluación *leave-one-dataset-out* (entrenar sin una fuente, probar exclusivamente en ella) para reportar explícitamente la caída de desempeño por cambio de dominio (cámara, población, sitio clínico) — no es la métrica principal, pero es evidencia relevante para la discusión del paper.
- **HVDROPDB** no participa de esta partición: su split train/val es interno y solo sirve para monitorear la Fase 1 (no tiene las etiquetas que el modelo final evalúa).

## 3. Pipeline de Preprocesamiento

### 3.1. Gestión y Almacenamiento

- **Manifiesto unificado en Parquet** (no CSVs sueltos por dataset): una sola tabla con columnas estandarizadas (`sample_id`, `dataset_origen`, `patient_id_global`, `image_path`, `visita_id`, `ojo`, `GA`, `BW`, `PMA` [nullable], `sexo`, `stage`, `plus`, `ap_rop` [los 3 nullable, uno por cabezal], `split`, `fold`). Permite hacer el Group K-Fold global y el masking de pérdida con un solo join, sin reimplementar lógica por dataset.
- Las carpetas originales de cada dataset se mantienen intactas y separadas (sin fusionar ni renombrar archivos); el manifiesto apunta a ellas con rutas relativas. Esto preserva trazabilidad y permite regenerar el manifiesto si cambia el criterio de preprocesamiento.
- **Almacenamiento local** (filesystem + manifiesto) como opción por defecto para el volumen actual (~10-15k imágenes); MinIO/bucket privado (S3/GCS/Azure) solo si el entrenamiento se mueve a una VM en la nube, y siempre **privado**, nunca público — aunque los datasets de origen son públicos, conviene verificar la licencia de cada uno antes de redistribuirlos en otro storage.
- **Fidelidad:** conservación de imágenes en formatos sin pérdida (PNG/TIFF) para no destruir la microvasculatura periférica; MozJPEG calidad 95+ solo si hay restricción real de espacio.

### 3.2. Procesamiento de Entradas

- **Flujo Visual:**
  - Recorte (*cropping*) de bordes oscuros estáticos del retinógrafo.
  - CLAHE restringido al canal verde para realce capilar.
  - Redimensionamiento uniforme (ej. 384×384 para ViT, 224×224 para CNN).
  - Augmentation: rotación 360°, flip horizontal, jitter de brillo.
  - El preprocesamiento (CLAHE/resize) se aplica en el `Dataset.__getitem__` del data loader, no como paso destructivo fijo en disco — las imágenes originales permanecen intactas.
- **Flujo Tabular (GA, BW, PMA cuando exista, sexo):**
  - Estandarización Z-score para variables continuas.
  - One-Hot encoding para variables categóricas.
  - **La ausencia de PMA en FARFUM y ROP-VL es estructural (falta la fuente completa), no aleatoria** — no se imputa con KNN cross-dataset (filtraría identidad de dataset en la señal). Se usa un indicador binario "PMA disponible" + imputación por mediana *dentro de cada dataset*, y el vector clínico reducido (GA+BW+sexo) es el que alimenta la Cross-Attention en Fase 2; el PMA real solo entra en juego en Fase 3 con RIDIRP.

## 4. Arquitectura del Modelo (Fusión Cross-Attention + MIL)

El sistema procesa la información de forma asimétrica: la biometría del infante guía la atención de la red visual, y la agregación de múltiples imágenes por visita ocurre **después** de la fusión por imagen (MIL con atención a nivel de imagen, no de bolsa cruda).

### 4.1. Codificadores

1. **Codificador Visual (ViT, preferido sobre CNN):** preentrenado en ImageNet y luego afinado en Fase 1 (segmentación vascular en HVDROPDB). Devuelve la **secuencia completa de tokens por parche** (N × D), no un vector pooled — se necesita la granularidad espacial para que la Cross-Attention pueda atender a zonas específicas de la retina.
2. **Codificador Clínico (MLP Tabular):** procesa el vector clínico estandarizado y genera **un único embedding denso** por visita — este es el Query.

### 4.2. Cross-Attention por imagen

Para cada imagen individual dentro de la bolsa (bag) de una visita:

- **Query (Q):** embedding clínico de esa visita (compartido por todas las imágenes de la bolsa).
- **Key (K) / Value (V):** tokens visuales de esa imagen.
- El bloque se implementa con conexión residual + LayerNorm + feed-forward (estilo bloque Transformer), no como una operación de atención aislada, para estabilidad de gradiente y mayor capacidad expresiva.
- Salida: un vector fusionado por imagen.

### 4.3. Pooling de MIL (Opción A)

Las imágenes de una misma visita (periféricas y centrales, capturadas en la misma fecha) forman una bolsa que comparte el mismo vector clínico. Cada imagen pasa primero por su propio bloque de Cross-Attention (§4.2); los vectores fusionados resultantes se agregan con **attention pooling** (no promedio simple, para no diluir la imagen más informativa de la bolsa) en un único vector `z` por visita. Si hay menos imágenes que el máximo permitido, se aplica zero-padding con attention mask.

```
Imágenes de la visita → ViT → tokens por imagen
Vector clínico → MLP → embedding clínico (Query, compartido en la bolsa)

por cada imagen:
    Cross-Attention(Q=embedding clínico, K/V=tokens de esa imagen) → vector fusionado

Attention Pooling(vectores fusionados de la bolsa) → z (vector único de la visita)
```

### 4.4. Cabezales de Clasificación (Multitasking)

`z` alimenta 3 cabezales, cada uno un MLP pequeño (Linear → ReLU → Linear → softmax/sigmoid) sobre el tronco compartido:

- **Estadio:** clasificación ordinal (0-5), Focal Loss + matriz de penalización por distancia.
- **Enfermedad Plus:** Normal / Pre-Plus / Plus, entropía cruzada ponderada.
- **AP-ROP (binario):** ¿hay signos de AP-ROP (Aggressive Posterior ROP)? — cabezal separado del de Estadio, no una 7ª clase dentro de él (reemplaza al cabezal de Tratamiento que estaba en versiones anteriores de este documento, ver §0 para el razonamiento del cambio). **Por qué separado de Estadio:** AP-ROP no es "más severo que Estadio 5" en una recta ordinal — es una entidad clínica distinta (localización posterior + progresión rápida + simetría, no extensión de la cresta), así que forzarla como categoría final de la escala 0-5 rompería la matriz de penalización por distancia que sí tiene sentido para 0-5. Separarlo evita rediseñar esa matriz con un criterio clínico ad-hoc. Etiqueta positiva (`ap_rop=True`) desde RIDIRP (DG=8) y ROP-VL (categoría "A-ROP", misma entidad verificada en §0); negativa desde las imágenes con Estadio numérico válido en ambas fuentes; enmascarada (sin dato) en FARFUM, que no distingue AP-ROP en absoluto, y en los casos ambiguos (RIDIRP status-post, ROP-VL Laser-treated sin AP-ROP) donde no se puede afirmar con certeza que nunca hubo AP-ROP.

**Pérdida global**, balanceada dinámicamente por incertidumbre homocedástica, con máscara por tarea y por muestra:

$$
\mathcal{L}_{total} = \sum_{i \in \{estadio,\ plus,\ ap\text{-}rop\}} m_i \left( \frac{1}{2\sigma_i^2} \mathcal{L}_i + \log(\sigma_i) \right)
$$

donde $m_i \in \{0,1\}$ indica si la muestra tiene ground truth disponible para la tarea $i$.

## 5. Flujo de Entrenamiento (Curriculum Learning)

El entrenamiento se divide en tres fases para evitar que la rama tabular (de convergencia rápida) "apague" a la rama visual (de convergencia lenta) si se entrenara todo end-to-end desde pesos aleatorios.

### Fase 1: Pre-entrenamiento Visual (Segmentación Vascular)

- **Datos:** exclusivamente HVDROPDB.
- **Procedimiento:** el codificador visual (ImageNet-pretrained, aislado del resto de la red) se afina para predecir máscaras de vasos sanguíneos, cresta y disco óptico (*Dice Loss*).
- **Objetivo:** dotar al backbone de una representación anatómica real antes de competir con la rama clínica.

### Fase 2: Warm-up Multimodal (visual congelado)

- **Datos:** ROP-VL + FARFUM-ROP.
- **Procedimiento:** congelar el encoder visual de Fase 1. Conectar el MLP clínico (vector reducido GA+BW+sexo, sin PMA) y el bloque Cross-Attention + Pooling de MIL. Entrenar MLP, fusión y cabezales.
- **Objetivo:** calibrar la rama clínica y los pesos de atención contra un backbone visual ya estable, sin la complejidad temporal todavía.
- *Optimización de cómputo:* como el visual está congelado, los tokens del ViT pueden precalcularse y cachearse una sola vez (no cambian entre épocas), reduciendo drásticamente el costo de esta fase.

### Fase 3: Fine-Tuning End-to-End (con PMA dinámico)

- **Datos:** RIDIRP + ROP-VL + FARFUM-ROP (red completa descongelada).
- **Procedimiento:** se incorpora RIDIRP, única fuente con PMA real por visita — aquí el Query dinámico (PMA variable por visita) tiene sentido pleno. Se activan los pesos de incertidumbre ($\sigma_i$) y la pérdida enmascarada completa.
- **Objetivo:** ajuste conjunto de toda la red con el vector clínico completo disponible. El mecanismo de MIL (Cross-Attention por imagen + Attention Pooling, §4.3) es el mismo en todas las fases y para las 3 fuentes — no hay un componente de secuencia entre visitas; cada bag (visita) se entrena de forma independiente, incluso en RIDIRP donde un mismo paciente aporta varios bags. Ver "Caso de Uso (Despliegue)" al inicio del documento.
