"""Mapeos de variables clinicas categoricas, compartidos entre Fase 2 y Fase 3.

`sexo` tiene VOCABULARIO DISTINTO por dataset (verificado con datos reales,
no asumido) - sin este mapeo el modelo veria 6 categorias donde solo hay 2
reales:
    RIDIRP:  "M" / "F"
    ROP-VL:  "male" / "female"
    FARFUM:  "boy" / "girl"
"""
SEXO_TO_BINARY = {
    "M": 1, "male": 1, "boy": 1,
    "F": 0, "female": 0, "girl": 0,
}

PLUS_TO_IDX = {"Normal": 0, "Pre-Plus": 1, "Plus": 2}
