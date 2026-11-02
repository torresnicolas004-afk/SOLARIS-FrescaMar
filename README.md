# Proyecto SOLARIS

**Sistema de apoyo para la priorización del riesgo de mora en FrescaMar S.A.S.**

Proyecto desarrollado para la asignatura **Proyecto de Ciencia de Datos 2**.

## Descripción

FrescaMar S.A.S. necesita identificar anticipadamente qué clientes presentan
mayor riesgo de mora para apoyar las decisiones de crédito y priorizar la revisión
de cartera.

SOLARIS transforma la información histórica disponible en una probabilidad
estimada de mora mediante un pipeline reproducible de Machine Learning.

La solución no reemplaza el juicio humano. Su propósito es apoyar la priorización
y revisión de clientes antes de tomar decisiones crediticias.

## Dataset

El dataset original contiene:

- 15.000 registros.
- 11 variables.
- 15.000 identificadores únicos.

Para el problema de clasificación binaria se excluyeron los registros con estado
`Cancelado`, obteniendo:

- 12.300 registros para modelado.
- 10.500 clientes `Al día`.
- 1.800 clientes `Moroso`.
- Prevalencia de mora: 14,63 %.

La clase positiva del proyecto es:

`Moroso = 1`

## Variables utilizadas por el modelo

El modelo final utiliza únicamente:

- `edad`
- `ingreso_mensual`
- `anio_vinculacion`
- `score_externo`
- `nivel_educativo`
- `ciudad`

Variables como `saldo_actual`, `cuotas_pagadas`, `cuotas_totales` y
`estado_cuenta` fueron excluidas como predictores debido a riesgo de data leakage
o problemas de temporalidad.

## Modelo final

El modelo seleccionado es:

`HistGradientBoostingClassifier`

El proceso incluye:

- pipeline de preprocesamiento;
- imputación y codificación;
- validación cruzada estratificada;
- comparación con Regresión Logística;
- optimización mediante Optuna;
- 50 trials;
- validación de 5 folds;
- selección del threshold mediante predicciones Out-of-Fold;
- interpretación mediante SHAP;
- análisis de robustez y equidad.

Semilla global:

`42`

Threshold oficial:

`0.59`

## Resultados principales

| Métrica | Resultado |
|---|---:|
| AUC CV | 0.8692 |
| AUC Test | 0.8681 |
| Precision Moroso | 0.4470 |
| Recall Moroso | 0.7028 |
| F1 Moroso | 0.5464 |
| Threshold | 0.59 |
| Gap Train-Test | 0.0254 |

Matriz de confusión sobre el conjunto de prueba:

| Resultado | Clientes |
|---|---:|
| Verdadero negativo | 1.787 |
| Falso positivo | 313 |
| Falso negativo | 107 |
| Verdadero positivo | 253 |

En términos de negocio, SOLARIS identifica aproximadamente 7 de cada 10 clientes
que realmente presentan mora.

## Interpretabilidad

El análisis SHAP identifica como variables de mayor importancia global:

1. `score_externo`
2. `ingreso_mensual`
3. `edad`

La validación también mostró una sensibilidad importante frente a la ausencia
de `score_externo`.

En el conjunto de prueba, los 212 registros sin este dato fueron clasificados
como alerta utilizando el threshold oficial.

Por esta razón, la ausencia de `score_externo` debe activar revisión humana y
no una decisión crediticia automática.

## Uso responsable

SOLARIS debe utilizarse como una herramienta de apoyo.

No debe utilizarse de manera autónoma para:

- aprobar crédito;
- negar crédito;
- reducir cupos;
- modificar condiciones contractuales;
- realizar acciones adversas únicamente por ausencia de `score_externo`.

La recomendación actual es avanzar únicamente mediante un piloto controlado,
con supervisión humana, monitoreo de desempeño y validación temporal antes de
una implementación productiva.

## Dashboard

El proyecto incluye un dashboard ejecutivo desarrollado en Streamlit.

Permite:

- visualizar KPIs de riesgo;
- filtrar por ciudad;
- filtrar por nivel educativo;
- analizar disponibilidad de `score_externo`;
- explorar diferentes thresholds;
- comparar predicción y realidad;
- consultar importancia SHAP;
- revisar un caso individual;
- identificar clientes priorizados;
- descargar los resultados.

El dashboard utiliza los 2.460 registros del conjunto de prueba para mostrar una
evaluación histórica del modelo.

## Estructura del repositorio

```text
SOLARIS-FrescaMar/
│
├── data/
│   ├── README.md
│   └── raw/
│       └── SOLARIS_FrescaMar_cartera_credito.csv
│
├── notebooks/
│   ├── M3_PCD2_TorresCastellanos_Prototipo.ipynb
│   └── M4_PCD2_TorresCastellanos_Validacion.ipynb
│
├── src/
│   ├── README.md
│   └── generar_predicciones_dashboard.py
│
├── models/
│   └── artifacts/
│       └── solaris_hgb_v1_20260915.pkl
│
├── outputs/
│   ├── m3/
│   └── m4/
│
├── reports/
│   ├── figures/
│   │   ├── m3/
│   │   └── m4/
│   ├── modulo_4/
│   └── final/
│
├── dashboard/
│   ├── app.py
│   ├── README.md
│   ├── data/
│   │   └── solaris_predicciones.csv
│   └── utils/
│       ├── __init__.py
│       └── graficos.py
│
├── .gitignore
├── requirements.txt
└── README.md

## Autor

**Nicolás Torres Castellanos**

Proyecto de Ciencia de Datos 2  
Universidad de La Sabana  
Periodo académico 2026-2