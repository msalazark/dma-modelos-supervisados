# 🔬 CRISP-DM Lab · Marketing Analytics

Herramienta educativa interactiva que ejecuta **6 modelos ML reales** siguiendo el framework CRISP-DM, con datos sintéticos precargados y soporte para cargar CSV propio.

## Modelos incluidos

| Modelo | Tipo | Algoritmo |
|--------|------|-----------|
| Churn Prediction | Clasificación | Random Forest |
| Lead Scoring | Clasificación | Regresión Logística |
| LTV · Customer Value | Regresión | Ridge Regression |
| Elasticidad de Precio | Regresión | Log-log OLS |
| Market Basket Analysis | Asociación | Apriori (mlxtend) |
| Next Best Offer | Clasificación multiclase | Random Forest |
| Forecast de Demanda | Series de tiempo | Prophet (Meta) |

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

## Notebooks Google Colab

Cada modelo tiene un notebook con el código completo:

- [Churn Prediction](https://colab.research.google.com/drive/1PIPrFDUExENqJt5oQDv_md0o1OE8EquI)
- [Regresión Logística](https://colab.research.google.com/drive/1UfOPvugdQcJuepUy4YWbBrVh-t3qx8Es)
- [Regresión Lineal / LTV](https://colab.research.google.com/drive/1K06-_Moh6PGhsGNVE_p0Oy4qQbLPdYqG)
- [Elasticidad de Precio](https://colab.research.google.com/drive/19UTRJU0ZJnZn9KxvGZhAID_rEXuc1ADG)
- [Market Basket](https://colab.research.google.com/drive/1OKmAKsT3YyFs1WGPaaGy6-GTzWEdTWWo)
- [Next Best Offer](https://colab.research.google.com/drive/1EN-JPEDTf4OU3P5Tr-hKMWOzvrf0vJlE)
