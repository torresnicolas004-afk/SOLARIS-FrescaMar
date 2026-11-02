from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from utils.graficos import (
    grafico_riesgo_ciudad,
    grafico_matriz_confusion,
    grafico_importancia_shap,
    grafico_disponibilidad_score,
)


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="SOLARIS | Riesgo de mora",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)


THRESHOLD_OFICIAL = 0.59


# ============================================================
# RUTAS DEL PROYECTO
# ============================================================

ROOT = Path(__file__).resolve().parents[1]


PREDICTIONS_PATH = (
    ROOT
    / "dashboard"
    / "data"
    / "solaris_predicciones.csv"
)


SHAP_PATH = (
    ROOT
    / "outputs"
    / "m4"
    / "shap_importance_original.csv"
)


SHAP_CASE_PATH = (
    ROOT
    / "outputs"
    / "m4"
    / "shap_caso_individual.csv"
)


SHAP_WATERFALL_PATH = (
    ROOT
    / "reports"
    / "figures"
    / "m4"
    / "shap_waterfall_caso_individual.png"
)


# ============================================================
# CARGA DE DATOS
# ============================================================

@st.cache_data
def cargar_predicciones():

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: "
            f"{PREDICTIONS_PATH}"
        )

    datos = pd.read_csv(
        PREDICTIONS_PATH
    )

    columnas_obligatorias = {
        "id_cliente",
        "edad",
        "ingreso_mensual",
        "anio_vinculacion",
        "score_externo",
        "nivel_educativo",
        "ciudad",
        "score_disponible",
        "target_real",
        "estado_real",
        "probabilidad_mora",
    }

    faltantes = (
        columnas_obligatorias
        - set(datos.columns)
    )

    if faltantes:
        raise ValueError(
            "Faltan columnas requeridas: "
            f"{sorted(faltantes)}"
        )

    return datos


@st.cache_data
def cargar_shap():

    if not SHAP_PATH.exists():
        return None

    return pd.read_csv(
        SHAP_PATH
    )


@st.cache_data
def cargar_caso_shap():

    if not SHAP_CASE_PATH.exists():
        return None

    datos = pd.read_csv(
        SHAP_CASE_PATH
    )

    if datos.empty:
        return None

    return datos


df = cargar_predicciones()
shap_df = cargar_shap()
caso_shap_df = cargar_caso_shap()


# ============================================================
# ENCABEZADO
# ============================================================

st.title(
    "SOLARIS | Priorización de riesgo de mora"
)


st.caption(
    "Prototipo ejecutivo para FrescaMar S.A.S. · "
    "Evaluación histórica sobre el conjunto de prueba."
)


st.info(
    "SOLARIS es una herramienta de apoyo. "
    "Las alertas no deben utilizarse para aprobar, "
    "negar o reducir crédito de forma automática."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "Filtros de análisis"
    )

    st.caption(
        "Los filtros actualizan indicadores, gráficos "
        "y la lista de clientes."
    )


    ciudades = (
        ["TODAS"]
        +
        sorted(
            df["ciudad"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
    )


    ciudad = st.selectbox(
        "Ciudad",
        ciudades,
    )


    niveles = (
        ["TODOS"]
        +
        sorted(
            df["nivel_educativo"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
    )


    nivel = st.selectbox(
        "Nivel educativo",
        niveles,
    )


    disponibilidad = st.selectbox(
        "Disponibilidad de score externo",
        [
            "TODOS",
            "DISPONIBLE",
            "FALTANTE",
        ],
    )


    st.divider()


    st.subheader(
        "Escenario de decisión"
    )


    threshold = st.slider(
        "Umbral de alerta",
        min_value=0.30,
        max_value=0.80,
        value=THRESHOLD_OFICIAL,
        step=0.01,
        help=(
            "El umbral oficial validado es 0,59. "
            "Mover este control permite explorar "
            "escenarios alternativos sin modificar "
            "el modelo oficial."
        ),
    )


    st.caption(
        "Umbral oficial del proyecto: 0,59"
    )


# ============================================================
# FILTROS
# ============================================================

df_filtrado = df.copy()


if ciudad != "TODAS":

    df_filtrado = df_filtrado[
        df_filtrado["ciudad"]
        == ciudad
    ]


if nivel != "TODOS":

    df_filtrado = df_filtrado[
        df_filtrado["nivel_educativo"]
        == nivel
    ]


if disponibilidad != "TODOS":

    df_filtrado = df_filtrado[
        df_filtrado["score_disponible"]
        == disponibilidad
    ]


if df_filtrado.empty:

    st.warning(
        "La combinación seleccionada "
        "no contiene registros."
    )

    st.stop()


# ============================================================
# PREDICCIÓN DINÁMICA SEGÚN THRESHOLD
# ============================================================

df_filtrado = df_filtrado.copy()


df_filtrado[
    "prediccion_dashboard"
] = (
    df_filtrado[
        "probabilidad_mora"
    ]
    >= threshold
).astype(int)


df_filtrado[
    "estado_predicho_dashboard"
] = np.where(
    df_filtrado[
        "prediccion_dashboard"
    ] == 1,
    "MOROSO",
    "AL DIA",
)


def clasificar_resultado(row):

    real = row[
        "target_real"
    ]

    pred = row[
        "prediccion_dashboard"
    ]

    if real == 1 and pred == 1:
        return "VERDADERO POSITIVO"

    if real == 0 and pred == 1:
        return "FALSO POSITIVO"

    if real == 0 and pred == 0:
        return "VERDADERO NEGATIVO"

    return "FALSO NEGATIVO"


df_filtrado[
    "resultado_dashboard"
] = df_filtrado.apply(
    clasificar_resultado,
    axis=1,
)


# ============================================================
# ADVERTENCIA DE THRESHOLD EXPLORATORIO
# ============================================================

if abs(
    threshold
    - THRESHOLD_OFICIAL
) > 1e-9:

    st.warning(
        f"Escenario exploratorio con umbral "
        f"{threshold:.2f}. "
        "El modelo oficial del proyecto utiliza 0,59."
    )


# ============================================================
# SECCIÓN 1 — SEÑAL DE DECISIÓN
# ============================================================

st.subheader(
    "1. Señal de decisión"
)


total_clientes = len(
    df_filtrado
)


alertas = int(
    df_filtrado[
        "prediccion_dashboard"
    ].sum()
)


morosos_reales = int(
    df_filtrado[
        "target_real"
    ].sum()
)


tp = int(
    (
        (
            df_filtrado[
                "target_real"
            ] == 1
        )
        &
        (
            df_filtrado[
                "prediccion_dashboard"
            ] == 1
        )
    ).sum()
)


fp = int(
    (
        (
            df_filtrado[
                "target_real"
            ] == 0
        )
        &
        (
            df_filtrado[
                "prediccion_dashboard"
            ] == 1
        )
    ).sum()
)


recall = (
    tp / morosos_reales
    if morosos_reales > 0
    else np.nan
)


precision = (
    tp / alertas
    if alertas > 0
    else np.nan
)


tasa_alerta = (
    alertas
    / total_clientes
)


c1, c2, c3, c4 = st.columns(4)


with c1:

    st.metric(
        "Clientes evaluados",
        f"{total_clientes:,}"
        .replace(
            ",",
            ".",
        ),
    )

    st.caption(
        "Registros del segmento analizado"
    )


with c2:

    st.metric(
        "Alertas para revisión",
        f"{alertas:,}"
        .replace(
            ",",
            ".",
        ),
    )

    st.caption(
        f"{tasa_alerta:.1%} del segmento"
    )


with c3:

    valor_recall = (
        f"{recall:.1%}"
        if not np.isnan(recall)
        else "N/A"
    )

    st.metric(
        "Morosos detectados",
        f"{tp}/{morosos_reales}",
    )

    st.caption(
        f"Recall: {valor_recall}"
    )


with c4:

    valor_precision = (
        f"{precision:.1%}"
        if not np.isnan(precision)
        else "N/A"
    )

    st.metric(
        "Falsas alertas",
        f"{fp:,}"
        .replace(
            ",",
            ".",
        ),
    )

    st.caption(
        f"Precision: {valor_precision}"
    )


st.caption(
    "Con el umbral oficial 0,59 y sin filtros, "
    "SOLARIS identifica aproximadamente 7 de cada "
    "10 clientes que realmente presentan mora."
)


# ============================================================
# SECCIÓN 2 — CONTEXTO DEL RIESGO
# ============================================================

st.subheader(
    "2. ¿Dónde conviene enfocar la revisión?"
)


col_izq, col_der = st.columns(2)


with col_izq:

    fig_ciudad = (
        grafico_riesgo_ciudad(
            df_filtrado
        )
    )

    st.plotly_chart(
        fig_ciudad,
        use_container_width=True,
    )


with col_der:

    fig_confusion = (
        grafico_matriz_confusion(
            df_filtrado
        )
    )

    st.plotly_chart(
        fig_confusion,
        use_container_width=True,
    )


# ============================================================
# INTERPRETACIÓN DE NEGOCIO DE LA CIUDAD
# ============================================================

riesgo_ciudad = (
    df_filtrado
    .dropna(
        subset=["ciudad"]
    )
    .groupby(
        "ciudad"
    )["probabilidad_mora"]
    .mean()
    .sort_values(
        ascending=False
    )
)


if not riesgo_ciudad.empty:

    ciudad_mayor_riesgo = (
        riesgo_ciudad.index[0]
    )

    riesgo_mayor = (
        riesgo_ciudad.iloc[0]
    )

    st.caption(
        f"En la selección actual, "
        f"{ciudad_mayor_riesgo} presenta "
        f"la mayor probabilidad promedio "
        f"estimada de mora "
        f"({riesgo_mayor:.1%}). "
        "Esta diferencia sirve para orientar "
        "el análisis, pero no debe utilizarse "
        "por sí sola para restringir crédito."
    )


# ============================================================
# SECCIÓN 3 — INTERPRETABILIDAD Y RIESGOS
# ============================================================

st.subheader(
    "3. ¿Por qué debemos interpretar las alertas con cuidado?"
)


col_shap, col_score = st.columns(2)


with col_shap:

    if shap_df is not None:

        try:

            fig_shap = (
                grafico_importancia_shap(
                    shap_df
                )
            )

            st.plotly_chart(
                fig_shap,
                use_container_width=True,
            )

        except Exception as error:

            st.warning(
                "No fue posible generar "
                "el gráfico SHAP."
            )

            st.caption(
                str(error)
            )

    else:

        st.warning(
            "No se encontró "
            "shap_importance_original.csv."
        )


with col_score:

    fig_score = (
        grafico_disponibilidad_score(
            df_filtrado
        )
    )

    st.plotly_chart(
        fig_score,
        use_container_width=True,
    )


st.warning(
    "Hallazgo de validación: con el umbral oficial 0,59, "
    "los 212 casos del conjunto de prueba sin "
    "score_externo recibieron alerta. "
    "La ausencia del score debe activar revisión humana "
    "y no una decisión crediticia automática."
)

st.caption(
    "El modelo depende principalmente de score_externo. "
    "Por eso, cuando este dato falta, la alerta debe interpretarse "
    "con especial cautela y pasar a revisión humana."
)

# ============================================================
# CASO INDIVIDUAL AUDITADO
# ============================================================

st.markdown(
    "### Caso individual auditado"
)


if (
    caso_shap_df is not None
    and SHAP_WATERFALL_PATH.exists()
):

    caso = (
        caso_shap_df
        .iloc[0]
    )


    def obtener_valor(
        fila,
        nombres,
        defecto=None,
    ):

        for nombre in nombres:

            if nombre in fila.index:

                valor = fila[
                    nombre
                ]

                if pd.notna(valor):
                    return valor

        return defecto


    probabilidad_caso = obtener_valor(
        caso,
        [
            "probabilidad_mora",
            "probabilidad",
            "prob_moroso",
            "probabilidad_predicha",
            "proba",
        ],
    )


    clase_real_caso = obtener_valor(
        caso,
        [
            "clase_real",
            "estado_real",
            "target_real",
            "y_real",
        ],
        "Moroso",
    )


    prediccion_caso = obtener_valor(
        caso,
        [
            "prediccion",
            "estado_predicho",
            "y_pred",
        ],
    )


    threshold_caso = obtener_valor(
        caso,
        [
            "threshold",
            "umbral",
        ],
        THRESHOLD_OFICIAL,
    )


    col_caso, col_explicacion = (
        st.columns(
            [1, 2]
        )
    )


    with col_caso:

        st.markdown(
            "**Ejemplo del conjunto de prueba**"
        )


        if probabilidad_caso is not None:

            try:

                st.metric(
                    "Probabilidad estimada de mora",
                    f"{float(probabilidad_caso):.1%}",
                )

            except (TypeError, ValueError):

                st.write(
                    "**Probabilidad estimada:** "
                    f"{probabilidad_caso}"
                )


        if (
            str(clase_real_caso)
            in {"1", "1.0"}
        ):

            clase_real_texto = (
                "MOROSO"
            )

        elif (
            str(clase_real_caso)
            in {"0", "0.0"}
        ):

            clase_real_texto = (
                "AL DIA"
            )

        else:

            clase_real_texto = (
                str(
                    clase_real_caso
                )
            )


        st.write(
            "**Clase real:** "
            f"{clase_real_texto}"
        )


        if prediccion_caso is not None:

            if (
                str(prediccion_caso)
                in {"1", "1.0"}
            ):

                prediccion_texto = (
                    "MOROSO"
                )

            elif (
                str(prediccion_caso)
                in {"0", "0.0"}
            ):

                prediccion_texto = (
                    "AL DIA"
                )

            else:

                prediccion_texto = (
                    str(
                        prediccion_caso
                    )
                )


            st.write(
                "**Predicción:** "
                f"{prediccion_texto}"
            )


        try:

            st.write(
                "**Threshold:** "
                f"{float(threshold_caso):.2f}"
            )

        except (TypeError, ValueError):

            st.write(
                "**Threshold:** "
                f"{threshold_caso}"
            )


        st.caption(
            "Este caso permite explicar una "
            "predicción concreta de SOLARIS. "
            "La explicación SHAP describe el "
            "comportamiento del modelo, no una "
            "relación causal."
        )


    with col_explicacion:

        st.image(
            str(
                SHAP_WATERFALL_PATH
            ),
            caption=(
                "Explicación SHAP del caso individual "
                "auditado durante el Módulo 4."
            ),
            use_container_width=True,
        )


else:

    st.info(
        "No se encontraron todos los artefactos "
        "necesarios para mostrar el caso SHAP individual."
    )


# ============================================================
# SECCIÓN 4 — CLIENTES PRIORIZADOS
# ============================================================

st.subheader(
    "4. Clientes priorizados para revisión"
)


prioritarios = (
    df_filtrado[
        df_filtrado[
            "prediccion_dashboard"
        ] == 1
    ]
    .sort_values(
        "probabilidad_mora",
        ascending=False,
    )
    .copy()
)


tabla = prioritarios[
    [
        "id_cliente",
        "ciudad",
        "nivel_educativo",
        "edad",
        "ingreso_mensual",
        "score_disponible",
        "probabilidad_mora",
        "estado_real",
        "resultado_dashboard",
    ]
].copy()


tabla = tabla.rename(
    columns={

        "id_cliente":
            "Cliente",

        "ciudad":
            "Ciudad",

        "nivel_educativo":
            "Nivel educativo",

        "edad":
            "Edad",

        "ingreso_mensual":
            "Ingreso mensual",

        "score_disponible":
            "Score externo",

        "probabilidad_mora":
            "Probabilidad de mora",

        "estado_real":
            "Estado real",

        "resultado_dashboard":
            "Resultado de evaluación",
    }
)


st.caption(
    f"Se muestran {len(tabla):,} clientes "
    "clasificados como alerta bajo el escenario actual."
    .replace(
        ",",
        ".",
    )
)


st.dataframe(
    tabla,
    use_container_width=True,
    hide_index=True,
    column_config={

        "Probabilidad de mora":
            st.column_config.ProgressColumn(
                "Probabilidad de mora",
                min_value=0.0,
                max_value=1.0,
                format="percent",
            ),

        "Ingreso mensual":
            st.column_config.NumberColumn(
                "Ingreso mensual",
                format="$ %.0f",
            ),
    },
)


csv_descarga = (
    tabla
    .to_csv(
        index=False
    )
    .encode(
        "utf-8-sig"
    )
)


st.download_button(
    label="Descargar clientes priorizados",
    data=csv_descarga,
    file_name=(
        "solaris_clientes_priorizados.csv"
    ),
    mime="text/csv",
)


# ============================================================
# NOTA METODOLÓGICA
# ============================================================

with st.expander(
    "Cómo interpretar este dashboard"
):

    st.markdown(
        """
Este dashboard utiliza los **2.460 registros del conjunto
de prueba de SOLARIS**. Estos clientes no fueron utilizados
para entrenar el modelo.

La columna de estado real está disponible porque esta
visualización representa una **evaluación histórica**.
En una aplicación operativa futura, la clase real todavía
no estaría disponible en el momento de generar la alerta.

El **threshold oficial es 0,59**. El control lateral permite
explorar escenarios alternativos, pero no modifica el
modelo aprobado durante la validación.

Las variables mostradas por SHAP indican qué información
utiliza con mayor intensidad el modelo para realizar sus
predicciones. No deben interpretarse como relaciones
causales.

SOLARIS debe utilizarse como **herramienta de apoyo a la
revisión humana**, no como mecanismo autónomo para aprobar,
reducir o negar crédito.
        """
    )