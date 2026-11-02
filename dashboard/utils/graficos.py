import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# PALETA SEMÁNTICA SOLARIS
# ============================================================

AZUL = "#2980B9"
ROJO = "#E74C3C"
AMARILLO = "#F39C12"
GRIS = "#95A5A6"


# ============================================================
# RIESGO PROMEDIO POR CIUDAD
# ============================================================

def grafico_riesgo_ciudad(df):
    """
    Muestra la probabilidad promedio estimada de mora por ciudad.
    """

    datos = df.copy()

    datos["ciudad"] = (
        datos["ciudad"]
        .fillna("SIN DATO")
        .astype(str)
    )

    resumen = (
        datos
        .groupby("ciudad", dropna=False)
        .agg(
            probabilidad_promedio=(
                "probabilidad_mora",
                "mean",
            ),
            clientes=(
                "id_cliente",
                "count",
            ),
        )
        .reset_index()
        .sort_values(
            "probabilidad_promedio",
            ascending=True,
        )
    )

    fig = px.bar(
        resumen,
        x="probabilidad_promedio",
        y="ciudad",
        orientation="h",
        text="probabilidad_promedio",
        custom_data=["clientes"],
    )

    fig.update_traces(
        marker_color=AZUL,
        texttemplate="%{text:.1%}",
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Riesgo promedio: %{x:.1%}<br>"
            "Clientes analizados: %{customdata[0]}"
            "<extra></extra>"
        ),
    )

    maximo = resumen[
        "probabilidad_promedio"
    ].max()

    limite = (
        max(0.30, maximo * 1.18)
        if pd.notna(maximo)
        else 1
    )

    ciudad_top = resumen.iloc[-1]["ciudad"]
    riesgo_top = resumen.iloc[-1]["probabilidad_promedio"]

    fig.update_layout(
        title=(
            f"{ciudad_top} concentra el mayor riesgo promedio "
            f"estimado ({riesgo_top:.1%})"
                ),
        xaxis_title="Probabilidad promedio de mora",
        yaxis_title="",
        showlegend=False,
        margin=dict(
            l=10,
            r=60,
            t=60,
            b=20,
        ),
        height=430,
    )

    fig.update_xaxes(
        tickformat=".0%",
        range=[0, limite],
    )

    return fig


# ============================================================
# MATRIZ DE CONFUSIÓN
# ============================================================

def grafico_matriz_confusion(df):
    """
    Matriz de confusión dinámica utilizando el threshold
    seleccionado en el dashboard.
    """

    matriz = pd.crosstab(
        df["target_real"],
        df["prediccion_dashboard"],
    )

    matriz = matriz.reindex(
        index=[0, 1],
        columns=[0, 1],
        fill_value=0,
    )

    valores = matriz.values

    tp = int(
    (
        (df["target_real"] == 1)
        &
        (df["prediccion_dashboard"] == 1)
    ).sum()
    )

    morosos_reales = int(
    (df["target_real"] == 1).sum()
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=valores,
            x=[
                "Predicho: Al día",
                "Predicho: Moroso",
            ],
            y=[
                "Real: Al día",
                "Real: Moroso",
            ],
            text=valores,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
            hovertemplate=(
                "%{y}<br>"
                "%{x}<br>"
                "Clientes: %{z}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=(
            f"SOLARIS identifica {tp} de "
            f"{morosos_reales} casos reales de mora"
            ),
        xaxis_title="Predicción de SOLARIS",
        yaxis_title="Estado real",
        margin=dict(
            l=10,
            r=10,
            t=60,
            b=20,
        ),
        height=430,
    )

    return fig


# ============================================================
# IMPORTANCIA GLOBAL SHAP
# ============================================================

def grafico_importancia_shap(shap_df):
    """
    Muestra únicamente las tres variables originales
    con mayor importancia global SHAP.
    """

    if shap_df is None or shap_df.empty:
        raise ValueError(
            "El archivo de importancia SHAP está vacío."
        )

    if shap_df.shape[1] < 2:
        raise ValueError(
            "El archivo SHAP debe contener al menos "
            "dos columnas."
        )

    # Primera columna = variable.
    variable_col = shap_df.columns[0]

    # Buscar automáticamente una columna de importancia.
    candidatos = [
        columna
        for columna in shap_df.columns[1:]
        if (
            "shap" in columna.lower()
            or "importance" in columna.lower()
            or "importancia" in columna.lower()
        )
    ]

    if candidatos:
        valor_col = candidatos[0]
    else:
        valor_col = shap_df.columns[1]

    datos = shap_df[
        [
            variable_col,
            valor_col,
        ]
    ].copy()

    datos.columns = [
        "variable",
        "importancia",
    ]

    datos["importancia"] = pd.to_numeric(
        datos["importancia"],
        errors="coerce",
    )

    datos = (
        datos
        .dropna(subset=["importancia"])
        .sort_values(
            "importancia",
            ascending=False,
        )
        .head(3)
        .sort_values(
            "importancia",
            ascending=True,
        )
    )

    fig = px.bar(
        datos,
        x="importancia",
        y="variable",
        orientation="h",
        text="importancia",
    )

    fig.update_traces(
        marker_color=AZUL,
        texttemplate="%{text:.3f}",
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Importancia media |SHAP|: %{x:.3f}"
            "<extra></extra>"
        ),
    )

    fig.update_layout(
        title=(
            "score_externo concentra "
            "la mayor señal predictiva"
        ),
        xaxis_title="Importancia media |SHAP|",
        yaxis_title="",
        showlegend=False,
        margin=dict(
            l=10,
            r=60,
            t=60,
            b=20,
        ),
        height=380,
    )

    return fig


# ============================================================
# DISPONIBILIDAD DE SCORE EXTERNO
# ============================================================

def grafico_disponibilidad_score(df):
    """
    Compara la tasa de alerta según la disponibilidad
    de score_externo.
    """

    resumen = (
        df
        .groupby(
            "score_disponible",
            dropna=False,
        )
        .agg(
            clientes=(
                "id_cliente",
                "count",
            ),
            tasa_alerta=(
                "prediccion_dashboard",
                "mean",
            ),
        )
        .reset_index()
    )

    colores = {
        "DISPONIBLE": AZUL,
        "FALTANTE": ROJO,
    }

    fig = px.bar(
        resumen,
        x="score_disponible",
        y="tasa_alerta",
        color="score_disponible",
        color_discrete_map=colores,
        text="tasa_alerta",
        custom_data=["clientes"],
    )

    fig.update_traces(
        texttemplate="%{text:.1%}",
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Tasa de alerta: %{y:.1%}<br>"
            "Clientes: %{customdata[0]}"
            "<extra></extra>"
        ),
    )

    fig.update_layout(
        title=(
            "La ausencia de score_externo "
            "requiere atención especial"
        ),
        xaxis_title="Disponibilidad del score",
        yaxis_title="Tasa de alerta",
        showlegend=False,
        margin=dict(
            l=10,
            r=20,
            t=60,
            b=20,
        ),
        height=380,
    )

    fig.update_yaxes(
        tickformat=".0%",
        range=[0, 1.10],
    )

    return fig