# Descarga de Datasets

Scripts usados para obtener los 4 datasets públicos del proyecto. Cada notebook descarga su dataset hacia `../../datasets/<Nombre>_Original/` (dos niveles arriba — `system/datasets_download/` → `system/` → raíz del proyecto — y dentro de `datasets/`). Ejecutar desde dentro de `system/datasets_download/` para que las rutas relativas funcionen.

## RIDIRP

- **Notebook:** [`ridirp_dataset.ipynb`](ridirp_dataset.ipynb)
- **Fuente:** Figshare, colección `6626162` — API pública de Figshare (`GET /v2/collections/{id}/articles`, luego `GET /v2/articles/{id}/files`).
- **Título del recurso (Figshare):** "Retinal Image Dataset of Infants and Retinopathy of Prematurity 2" — DOI de colección `10.6084/m9.figshare.c.6626162.v1`.
- **Paper asociado (DOI reportado por la propia API de Figshare como `resource_doi`):** `10.1038/s41597-024-03409-7` (Scientific Data, 2024). *No verificado de forma independiente fuera de la metadata de Figshare — confirmar título exacto/autores antes de citarlo en la tesis.*
- **Origen clínico:** cribado de ROP en el University Hospital Ostrava, República Checa.
- **Tamaño (según metadata de Figshare):** 188 recién nacidos, 6.004 imágenes retinianas (series longitudinales con múltiples visitas por paciente).
- **Contenido descargado:** `infant_retinal_database_info.xlsx` (metadatos clínicos y diccionario de diagnósticos), `images.zip`, `images_stack.zip` (con texto superpuesto/"captions" del equipo RetCam), `images_stack_without_captions.zip`.

## FARFUM-RoP

- **Notebook:** [`farfum_dataset.ipynb`](farfum_dataset.ipynb)
- **Fuente:** Figshare, colección `6721269` — misma API pública que RIDIRP.
- **Título del recurso (Figshare):** "FARFUM-RoP" (Farabi and Ferdowsi University of Mashhad's ROP dataset) — DOI de colección `10.6084/m9.figshare.c.6721269.v2`.
- **Paper asociado (identificado por búsqueda, no por metadata directa de la API):** Akbari et al., *"FARFUM-RoP, A dataset for computer-aided detection of Retinopathy of Prematurity"*, Scientific Data, DOI `10.1038/s41597-024-03897-7` (2024). Coincide en cifras exactas (1.533 imágenes, 68 pacientes) con la descripción de la colección — *verificar antes de citar formalmente.*
- **Tamaño:** 68 pacientes, 1.533 imágenes de fondo de ojo, anotadas independientemente por 5 oftalmólogos pediatras (Normal / Pre-Plus / Plus).
- **Contenido descargado:** `Dataset_Details.xlsx` (datos biométricos por paciente), `Dataset_Labels.xlsx` (etiquetas por imagen y anotador), `patientXX.rar` (imágenes, uno por paciente).

## HVDROPDB (v3)

- **Notebook:** [`hvdropdb_dataset.ipynb`](hvdropdb_dataset.ipynb)
- **Fuente:** Mendeley Data, dataset `xw5xc7xrmp`, versión 3 — API pública de Mendeley.
- **Nota:** la descarga automatizada vía API **falló en este proyecto** (HTTP 403, bloqueo Cloudflare — ver output guardado en el notebook). Los archivos presentes en `../datasets/HVDROPDB_Original/` se obtuvieron por la vía manual que el propio notebook imprime como alternativa: `https://data.mendeley.com/datasets/xw5xc7xrmp/3`.
- **Autor:** Ranjana Agrawal.
- **Origen clínico:** cribado de ROP en PBMA's H.V. Desai Eye Hospital, Pune, India; imágenes de los sistemas RetCam y Neo.
- **Paper relacionado (identificado por búsqueda):** Agrawal et al., Multimedia Tools and Applications (2022), DOI `10.1007/s11042-022-12396-z`, sobre segmentación de fondo de ojo retiniano para ROP mediante deep learning. *Verificar antes de citar formalmente.*
- **Contenido descargado:** `HVDROPDB_RetCam_Neo_Classification.rar`, `HVDROPDB_RetCam_Neo_Segmentation.zip` (máscaras de vasos, cresta y disco óptico, sin metadatos clínicos).

## ROP-VL

- **Sin notebook.** Descargado manualmente como `.zip` directamente desde la página del dataset (no vía API).
- **Fuente exacta:** *pendiente — completar con la URL/plataforma real desde la que se descargó.*
- **Título probable (identificado por búsqueda, no confirmado):** *"ROP-VL: An Open Dataset Integrating Fundus Photograph and Clinical Knowledge for Retinopathy of Prematurity"*, reportado como publicado en Scientific Data. **DOI no confirmado** — no usar como cita formal hasta verificarlo directamente en la fuente.
- **Tamaño:** 1.116 pacientes, 2.020 imágenes, 8 categorías (Normal, Stage 1-5 ROP, A-ROP, Laser-treated ROP).
- **Contenido descargado:** `ROP-VL.zip` → `disease_description.xlsx`, `img_info.xlsx` (metadatos por imagen: género, edad gestacional, peso al nacer, pluralidad, vía de parto, categoría), carpetas de imágenes por categoría.

## Pendientes

- Confirmar/corregir los DOIs marcados como "identificados por búsqueda, no verificados" contra la fuente original antes de usarlos en la bibliografía de la tesis.
- Completar la fuente exacta de descarga de ROP-VL.
