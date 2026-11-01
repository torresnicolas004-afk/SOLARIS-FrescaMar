from pathlib import Path
import re
import unicodedata

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURACIÓN
# ============================================================

RANDOM_SEED = 42
THRESHOLD = 0.59

DATA_PATH = Path("data/raw/SOLARIS_FrescaMar_cartera_credito.csv")
MODEL_PATH = Path("models/artifacts/solaris_hgb_v1_20260915.pkl")
OUTPUT_PATH = Path("dashboard/data/solaris_predicciones.csv")

FEATURES = [
    "edad",
    "ingreso_mensual",
    "anio_vinculacion",
    "score_externo",
    "nivel_educativo",
    "ciudad",
]


# ============================================================
# FUNCIONES DE NORMALIZACIÓN
# ============================================================

def normalize_colname(value):
    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def normalize_text(value):
    if pd.isna(value):
        return np.nan

    text = unicodedata.normalize(
        "NFKD",
        str(value).strip()
    )

    text = "".join(
        ch for ch in text
        if not unicodedata.combining(ch)
    )

    text = re.sub(
        r"[^A-Za-z0-9]+",
        " ",
        text
    )

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip().upper()


CANONICAL_CITIES = {
    "BOGOTA",
    "MEDELLIN",
    "CALI",
    "BARRANQUILLA",
    "BUCARAMANGA",
    "CARTAGENA",
    "SANTA MARTA",
    "MANIZALES",
    "PEREIRA",
    "CUCUTA",
    "VILLAVICENCIO",
    "IBAGUE",
}


CITY_EXACT_ALIASES = {
    "BTA": "BOGOTA",
    "BOG": "BOGOTA",
    "BOGOTA D C": "BOGOTA",
    "BOGOTA DC": "BOGOTA",
    "MDE": "MEDELLIN",
    "MDELLIN": "MEDELLIN",
    "MEDELIN": "MEDELLIN",
    "CAL": "CALI",
    "BAQ": "BARRANQUILLA",
    "BQ": "BARRANQUILLA",
    "B QUILLA": "BARRANQUILLA",
    "BQUILLA": "BARRANQUILLA",
    "BARRANQUILA": "BARRANQUILLA",
    "BGA": "BUCARAMANGA",
    "BUCARA": "BUCARAMANGA",
    "CTG": "CARTAGENA",
    "SMR": "SANTA MARTA",
    "STA MARTA": "SANTA MARTA",
    "MZL": "MANIZALES",
    "PEI": "PEREIRA",
    "CUC": "CUCUTA",
    "VVC": "VILLAVICENCIO",
    "VILLAVO": "VILLAVICENCIO",
    "IBE": "IBAGUE",
    "IBG": "IBAGUE",
    "B MANGA": "BUCARAMANGA",
    "BARRANQILLA": "BARRANQUILLA",
    "MDL": "MEDELLIN",
    "MED": "MEDELLIN",
}


CITY_NAME_RULES = [
    ("BOGOTA", "BOGOTA"),
    ("MEDELLIN", "MEDELLIN"),
    ("CALI", "CALI"),
    ("BARRANQUILLA", "BARRANQUILLA"),
    ("BUCARAMANGA", "BUCARAMANGA"),
    ("CARTAGENA", "CARTAGENA"),
    ("SANTA MARTA", "SANTA MARTA"),
    ("STA MARTA", "SANTA MARTA"),
    ("MANIZALES", "MANIZALES"),
    ("PEREIRA", "PEREIRA"),
    ("CUCUTA", "CUCUTA"),
    ("VILLAVICENCIO", "VILLAVICENCIO"),
    ("VILLAVO", "VILLAVICENCIO"),
    ("IBAGUE", "IBAGUE"),
]


def normalize_city(value):
    text = normalize_text(value)

    if pd.isna(text):
        return np.nan

    if text in CITY_EXACT_ALIASES:
        return CITY_EXACT_ALIASES[text]

    for token, canonical in CITY_NAME_RULES:
        if token in text:
            return canonical

    return text


# ============================================================
# CARGA Y LIMPIEZA DETERMINÍSTICA
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"No se encontró el dataset: {DATA_PATH}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"No se encontró el pipeline: {MODEL_PATH}"
    )


print("Cargando dataset...")

df_raw = pd.read_csv(DATA_PATH)

print(
    f"Dataset original: "
    f"{df_raw.shape[0]:,} filas x "
    f"{df_raw.shape[1]} columnas"
)


df = df_raw.copy()

df.columns = [
    normalize_colname(c)
    for c in df.columns
]


for col in [
    "edad",
    "ingreso_mensual",
    "saldo_actual",
    "cuotas_pagadas",
    "cuotas_totales",
    "score_externo",
]:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# Edad = 0 fue definida como valor inválido en M1/M3.
df.loc[
    df["edad"] == 0,
    "edad"
] = np.nan


df["fecha_vinculacion"] = pd.to_datetime(
    df["fecha_vinculacion"],
    errors="coerce"
)


df["nivel_educativo"] = (
    df["nivel_educativo"]
    .map(normalize_text)
)


df["ciudad"] = (
    df["ciudad"]
    .map(normalize_city)
)


df["anio_vinculacion"] = (
    df["fecha_vinculacion"]
    .dt.year
)


df["estado_norm"] = (
    df["estado_cuenta"]
    .map(normalize_text)
)


target_map = {
    "MOROSO": 1,
    "AL DIA": 0,
}


mask_model = (
    df["estado_norm"]
    .isin(target_map)
)


df_model = (
    df.loc[mask_model]
    .copy()
)


df_model["target_moroso"] = (
    df_model["estado_norm"]
    .map(target_map)
    .astype(int)
)


print(
    f"Filas para modelado binario: "
    f"{len(df_model):,}"
)


# ============================================================
# REPRODUCCIÓN EXACTA DEL SPLIT ORIGINAL
# ============================================================

X = df_model[FEATURES].copy()
y = df_model["target_moroso"].copy()


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_SEED,
    stratify=y,
    shuffle=True,
)


print(
    f"Train: {len(X_train):,}"
)

print(
    f"Test : {len(X_test):,}"
)


if len(X_test) != 2460:
    raise RuntimeError(
        "El test no tiene las 2.460 filas esperadas."
    )


# ============================================================
# CARGA DEL PIPELINE CONGELADO
# ============================================================

print("Cargando pipeline final...")

pipeline = joblib.load(MODEL_PATH)


# ============================================================
# PREDICCIONES
# ============================================================

probabilidad_mora = (
    pipeline
    .predict_proba(X_test)[:, 1]
)


prediccion = (
    probabilidad_mora >= THRESHOLD
).astype(int)


# ============================================================
# DATASET PARA DASHBOARD
# ============================================================

dashboard_df = (
    df_model.loc[X_test.index, [
        "id_cliente",
        "edad",
        "ingreso_mensual",
        "anio_vinculacion",
        "score_externo",
        "nivel_educativo",
        "ciudad",
    ]]
    .copy()
)


dashboard_df["score_disponible"] = np.where(
    dashboard_df["score_externo"].isna(),
    "FALTANTE",
    "DISPONIBLE",
)


dashboard_df["target_real"] = (
    y_test.astype(int)
)


dashboard_df["estado_real"] = np.where(
    dashboard_df["target_real"] == 1,
    "MOROSO",
    "AL DIA",
)


dashboard_df["probabilidad_mora"] = (
    probabilidad_mora
)


dashboard_df["prediccion"] = (
    prediccion
)


dashboard_df["estado_predicho"] = np.where(
    dashboard_df["prediccion"] == 1,
    "MOROSO",
    "AL DIA",
)


dashboard_df["threshold"] = THRESHOLD


def clasificar_resultado(row):
    real = row["target_real"]
    pred = row["prediccion"]

    if real == 1 and pred == 1:
        return "VERDADERO POSITIVO"

    if real == 0 and pred == 1:
        return "FALSO POSITIVO"

    if real == 0 and pred == 0:
        return "VERDADERO NEGATIVO"

    return "FALSO NEGATIVO"


dashboard_df["resultado"] = (
    dashboard_df.apply(
        clasificar_resultado,
        axis=1
    )
)


dashboard_df["acierto"] = np.where(
    dashboard_df["target_real"]
    == dashboard_df["prediccion"],
    "SI",
    "NO",
)


# Ordenar de mayor a menor riesgo
dashboard_df = (
    dashboard_df
    .sort_values(
        "probabilidad_mora",
        ascending=False
    )
    .reset_index(drop=True)
)


# ============================================================
# VALIDACIONES
# ============================================================

assert len(dashboard_df) == 2460

assert dashboard_df[
    "id_cliente"
].nunique() == 2460


conteo_resultados = (
    dashboard_df["resultado"]
    .value_counts()
)


print("\nMatriz de resultados:")

print(conteo_resultados)


esperados = {
    "VERDADERO NEGATIVO": 1787,
    "FALSO POSITIVO": 313,
    "FALSO NEGATIVO": 107,
    "VERDADERO POSITIVO": 253,
}


for categoria, esperado in esperados.items():

    obtenido = int(
        conteo_resultados.get(
            categoria,
            0
        )
    )

    if obtenido != esperado:

        raise RuntimeError(
            f"{categoria}: "
            f"se esperaban {esperado} "
            f"y se obtuvieron {obtenido}."
        )


print(
    "\nValidación contra Módulo 4: APROBADA"
)


# ============================================================
# EXPORTACIÓN
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


dashboard_df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)


print(
    f"\nArchivo generado correctamente:"
    f"\n{OUTPUT_PATH}"
)


print(
    f"\nShape final: "
    f"{dashboard_df.shape}"
)


print(
    "\nPrimeros 5 registros:"
)


print(
    dashboard_df.head()
)