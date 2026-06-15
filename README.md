# 🔬 CRISP-DM Lab · Marketing Analytics

Herramienta educativa interactiva que ejecuta **13 modelos ML reales** siguiendo el framework CRISP-DM, con datos sintéticos precargados y soporte para cargar CSV propio.

## Modelos incluidos

| Modelo | Tipo | Algoritmo |
|--------|------|-----------|
| Propensión a Compra | Clasificación | Gradient Boosting |
| Win / Loss de Oportunidades | Clasificación | Random Forest |
| Uplift · Incrementalidad | Causal (two-model) | Regresión Logística |
| Churn Prediction | Clasificación | Random Forest |
| Lead Scoring | Clasificación | Regresión Logística |
| LTV · Customer Value | Regresión | Ridge Regression |
| Elasticidad de Precio | Regresión | Log-log OLS |
| Market Basket Analysis | Asociación | Apriori (mlxtend) |
| Next Best Offer | Clasificación multiclase | Random Forest |
| Segmentación RFM | Clustering | K-Means |
| Segmentación Categórica | Clustering | K-Modes |
| Forecast de Demanda | Series de tiempo | Prophet (Meta) |
| Clustering Jerárquico | Clustering | Agglomerative (Ward) |

## Fases CRISP-DM por modelo

1. **Business Understanding** — Problema de negocio, KPIs, criterios de aceptación
2. **Data Understanding** — EDA completo: estadísticas, histogramas, correlaciones, distribución del target
3. **Data Preparation** — Limpieza, transformaciones, split 80/20, feature engineering
4. **Modeling** — Entrenamiento real con scikit-learn, hiperparámetros ajustables, simulador interactivo
5. **Evaluation** — Métricas en test set, matriz de confusión, curva ROC, validación vs criterios
6. **Deployment** — Arquitectura GCP, plan de monitoreo, ROI estimado, exportación de predicciones

## Instalación local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Despliegue en Streamlit Cloud (gratis)

1. Sube este repositorio a GitHub
2. Ve a [share.streamlit.io](https://share.streamlit.io)
3. Conecta tu repo y selecciona `app.py`
4. Deploy — listo en ~2 minutos

## Estructura del proyecto

```
críspdm_lab/
├── app.py              # App principal Streamlit
├── models.py           # Modelos ML reales (scikit-learn)
├── plots.py            # Visualizaciones Plotly
├── generate_data.py    # Generador de datos sintéticos
├── requirements.txt    # Dependencias
├── README.md
└── .streamlit/
    └── config.toml     # Tema y configuración
```

## Cargar datos propios

En el sidebar: **"Cargar CSV propio"** → reemplaza los datos del modelo activo.  
Todas las fases se recalculan automáticamente con tus datos.

### Columnas requeridas por modelo

| Modelo | Columnas mínimas |
|--------|-----------------|
| Churn | `dias_inactivo, quejas, logins_30d, saldo_prom_k, productos, churn` |
| Lead Scoring | `dias_contacto, interacciones_web, clicks_precio, tiempo_pag_min, convirtio` |
| LTV | `compras_6m, ticket_prom, meses_cliente, categorias, ltv_anual` |
| Elasticidad | `precio, unidades, categoria` |
| Market Basket | `ticket_id, producto` |
| Next Best Offer | `tiene_tarjeta, tiene_cuenta, antiguedad_m, saldo_k, producto_adquirido` |
| Forecast de Demanda | `fecha, ventas` |

## Notebooks

Cada modelo tiene un notebook CRISP-DM completo en [`notebooks/`](notebooks/), listo para abrir en Google Colab o Jupyter:

- [01 · Propensión a Compra — Gradient Boosting](notebooks/01_propension_compra_gbm.ipynb)
- [02 · Win/Loss de Oportunidades — Random Forest](notebooks/02_win_loss_oportunidades.ipynb)
- [03 · Uplift Modeling — Incrementalidad de Campaña](notebooks/03_uplift_incrementalidad.ipynb)
- [04 · Churn Prediction — Random Forest](notebooks/04_churn_prediction_rf.ipynb)
- [05 · Lead Scoring — Regresión Logística](notebooks/05_lead_scoring_logit.ipynb)
- [06 · LTV — Valor del Cliente (Ridge Regression)](notebooks/06_ltv_customer_value.ipynb)
- [07 · Elasticidad de Precio (Regresión Log-Log)](notebooks/07_elasticidad_precio.ipynb)
- [08 · Market Basket Analysis (Apriori)](notebooks/08_market_basket_apriori.ipynb)
- [09 · Next Best Offer (Random Forest Multiclase)](notebooks/09_next_best_offer_rf.ipynb)
- [10 · Segmentación RFM con K-Means](notebooks/10_segmentacion_rfm_kmeans.ipynb)
- [11 · Segmentación Categórica con K-Modes](notebooks/11_segmentacion_kmodes.ipynb)
- [12 · Forecast de Ventas con Prophet](notebooks/12_forecast_ventas_prophet.ipynb)
- [13 · Clustering Jerárquico (Agglomerative)](notebooks/13_clustering_jerarquico.ipynb)
