"""
CRISP-DM Lab · Marketing Analytics
Streamlit app —  modelos ML reales con EDA completo y simulador
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings, io, os, sys

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

from generate_data import DATASETS
import models as M
import plots as P

# ── PAGE CONFIG ────────────────────────────────────────────────────
st.set_page_config(
    page_title='CRISP-DM Lab · DMA',
    page_icon='🔬',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ── CUSTOM CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500&family=IBM+Plex+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main .block-container { padding: 1.5rem 2rem; max-width: 1200px; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #f8f7f5;
    border: 1px solid #e8e6e0;
    border-radius: 10px;
    padding: 0.75rem 1rem;
}
[data-testid="metric-container"] label {
    font-size: 11px !important;
    color: #888 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 22px !important;
    font-weight: 500 !important;
    color: #0f0f0f !important;
}

/* Phase headers */
.phase-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #c0392b;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 4px;
}
.phase-title {
    font-size: 22px;
    font-weight: 700;
    color: #0f0f0f;
    margin-bottom: 8px;
}
.phase-desc {
    font-size: 13px;
    color: #666;
    line-height: 1.6;
    margin-bottom: 1.5rem;
}

/* Criteria box */
.criteria-box {
    background: #eff6ff;
    border-left: 4px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    font-size: 13px;
    color: #1e40af;
    margin: 0.5rem 0;
}
.kpi-box {
    background: #f0fdf4;
    border-left: 4px solid #22c55e;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    font-size: 13px;
    color: #166534;
    margin: 0.25rem 0;
}
.warn-box {
    background: #fffbeb;
    border-left: 4px solid #f59e0b;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    font-size: 13px;
    color: #92400e;
    margin: 0.5rem 0;
}
.prob-high { background:#fef2f2;border-left:4px solid #dc2626;border-radius:0 8px 8px 0;padding:.75rem 1rem;color:#7f1d1d;font-size:13px;line-height:1.6 }
.prob-med  { background:#fffbeb;border-left:4px solid #d97706;border-radius:0 8px 8px 0;padding:.75rem 1rem;color:#78350f;font-size:13px;line-height:1.6 }
.prob-low  { background:#f0fdf4;border-left:4px solid #16a34a;border-radius:0 8px 8px 0;padding:.75rem 1rem;color:#14532d;font-size:13px;line-height:1.6 }

/* Sidebar */
.css-1d391kg { background: #0f0f0f !important; }
section[data-testid="stSidebar"] { background: #0f0f0f !important; }
section[data-testid="stSidebar"] * { color: #e8e6e0 !important; }
section[data-testid="stSidebar"] .stRadio label { color: #ccc !important; }

/* Tab headers */
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] {
    font-size: 13px !important;
    padding: 6px 14px !important;
    border-radius: 6px !important;
    font-family: 'DM Sans', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)

# ── MODEL CONFIG ────────────────────────────────────────────────────
MODEL_CONFIG = {
    'Churn Prediction': {
        'key': 'churn', 'data_key': 'churn',
        'badge': '🚨 Clasificación',
        'colab': 'https://colab.research.google.com/drive/1PIPrFDUExENqJt5oQDv_md0o1OE8EquI',
        'runner': M.run_churn, 'target': 'churn',
        'features': ['dias_inactivo','quejas','logins_30d','saldo_prom_k','productos'],
        'criteria': {'auc': 0.75, 'recall': 0.70, 'precision': 0.60},
        'business': {
            'problem': '⚠️ Un banco pierde ~**S/.2,500 de LTV** por cada cliente que se fuga. Con 17% de churn en una cartera de 100,000 clientes, el impacto supera los **S/.42M anuales**.',
            'objective': 'Predecir qué clientes abandonarán en los próximos 30 días para activar campañas de retención proactiva **antes** de la fuga.',
            'kpis': ['Reducir churn mensual de 17% a <10%', 'Priorizar top 500 clientes en riesgo alto', 'ROI del modelo: 10x–30x vs costo'],
            'algo': 'Random Forest (n=200, max_depth=6, class_weight=balanced)',
        }
    },
    'Lead Scoring (Reg. Logística)': {
        'key': 'logit', 'data_key': 'leads',
        'badge': '📊 Clasificación',
        'colab': 'https://colab.research.google.com/drive/1UfOPvugdQcJuepUy4YWbBrVh-t3qx8Es',
        'runner': M.run_logit, 'target': 'convirtio',
        'features': ['dias_contacto','interacciones_web','clicks_precio','tiempo_pag_min'],
        'criteria': {'auc': 0.78, 'recall': 0.60, 'precision': 0.65},
        'business': {
            'problem': '⚠️ Una universidad recibe 5,000 leads por ciclo. El **30% del tiempo de asesores** se desperdicia en leads fríos. Conversión actual: 8%.',
            'objective': 'Predecir la probabilidad de matrícula de cada lead para que los asesores prioricen los de mayor propensión.',
            'kpis': ['Subir conversión de 8% a 12%', 'Reducir llamadas en frío -40%', 'Top 20% leads → 70% matrículas'],
            'algo': 'Regresión Logística (C=1.0, StandardScaler, max_iter=500)',
        }
    },
    'LTV · Regresión Lineal': {
        'key': 'ltv', 'data_key': 'ltv',
        'badge': '💰 Regresión',
        'colab': 'https://colab.research.google.com/drive/1K06-_Moh6PGhsGNVE_p0Oy4qQbLPdYqG',
        'runner': M.run_ltv, 'target': 'ltv_anual',
        'features': ['compras_6m','ticket_prom','meses_cliente','categorias','canal_digital'],
        'criteria': {'r2': 0.70, 'mape': 20},
        'business': {
            'problem': '⚠️ Un retailer invierte el **mismo CAC** para adquirir clientes VIP y clientes de bajo valor. Pérdida de eficiencia estimada: **S/.180K anuales** en presupuesto de adquisición.',
            'objective': 'Estimar el LTV a 12 meses de cada cliente para optimizar la inversión en adquisición y retención por segmento.',
            'kpis': ['MAPE < 18% en predicción LTV', 'Segmentar cartera en Bajo/Regular/VIP', 'Optimizar CAC máximo por segmento'],
            'algo': 'Ridge Regression (alpha=1.0)',
        }
    },
    'Elasticidad de Precio': {
        'key': 'elastic', 'data_key': 'elasticity',
        'badge': '💲 Regresión',
        'colab': 'https://colab.research.google.com/drive/19UTRJU0ZJnZn9KxvGZhAID_rEXuc1ADG',
        'runner': M.run_elasticity, 'target': 'unidades',
        'features': ['precio','promocion'],
        'criteria': {'r2': 0.65},
        'business': {
            'problem': '⚠️ Una cadena retail fija precios de forma **empírica**. No sabe si un descuento del 20% genera más revenue o lo destruye. Cada decisión de pricing es una apuesta.',
            'objective': 'Cuantificar la elasticidad precio-demanda por categoría para optimizar pricing y evaluar el impacto de promociones.',
            'kpis': ['Elasticidad por categoría con IC 95%', 'Precio óptimo por categoría', 'Identificar si promociones son rentables'],
            'algo': 'Regresión log-log (OLS) · ln(Q) = β₀ + β₁·ln(P)',
        }
    },
    'Market Basket Analysis': {
        'key': 'basket', 'data_key': 'basket',
        'badge': '🛒 Asociación',
        'tipo': 'no_supervisado',
        'colab': 'https://colab.research.google.com/drive/1OKmAKsT3YyFs1WGPaaGy6-GTzWEdTWWo',
        'runner': M.run_basket, 'target': None,
        'features': ['ticket_id','producto'],
        'criteria': {'lift': 1.3, 'confidence': 0.40, 'support': 0.05},
        'business': {
            'problem': '⚠️ Librería Crisol tiene ticket promedio bajo porque los clientes compran solo lo que buscan. **No existe sistema de recomendación activo**.',
            'objective': 'Descubrir combinaciones frecuentes para diseñar bundles, recomendaciones y optimizar la disposición de góndola.',
            'kpis': ['Ticket promedio +12% con bundles', 'Top 10 reglas con lift > 1.5', 'Recomendaciones activas en web y POS'],
            'algo': 'Algoritmo Apriori (mlxtend) · Soporte → Confianza → Lift',
        }
    },
    'Next Best Offer': {
        'key': 'nbo', 'data_key': 'nbo',
        'badge': '🎁 Clasificación multiclase',
        'colab': 'https://colab.research.google.com/drive/1EN-JPEDTf4OU3P5Tr-hKMWOzvrf0vJlE',
        'runner': M.run_nbo, 'target': 'producto_adquirido',
        'features': ['tiene_tarjeta','tiene_cuenta','tiene_seguro','tiene_prestamo','antiguedad_m','saldo_k'],
        'criteria': {'accuracy': 0.55, 'f1_macro': 0.50, 'top2_accuracy': 0.75},
        'business': {
            'problem': '⚠️ Un banco ofrece **el mismo portafolio a todos** los clientes sin personalización. Tasa de aceptación cross-sell: 4.2%. Los asesores ofrecen el producto equivocado.',
            'objective': 'Predecir cuál es el próximo producto más relevante para cada cliente, ordenado por probabilidad de adopción.',
            'kpis': ['Cross-sell de 4.2% a 9%+', 'Personalizar top 30% cartera', 'Reducir coste por producto vendido 35%'],
            'algo': 'Random Forest Multiclase (n=150, max_depth=8)',
        }
    },
    'Propensión a Compra (GBM)': {
        'key': 'propension', 'data_key': 'propension',
        'badge': '🎯 Clasificación',
        'colab': 'https://colab.research.google.com/drive/1BSBDIs6ZdSV6OrxVWmJSaQ6paZPwfQgc',
        'runner': M.run_propension, 'target': 'compro',
        'features': ['recencia_dias','n_compras_12m','engagement_score','canal_digital','edad','n_productos'],
        'criteria': {'auc': 0.75, 'recall': 0.65, 'precision': 0.60},
        'business': {
            'problem': '⚠️ La Positiva contacta a toda la cartera para renovar SOAT. El **68% de las llamadas son a clientes con baja propensión**, desperdiciando capacidad del call center.',
            'objective': 'Predecir qué clientes comprarán SOAT en los próximos 15 días para priorizar el contacto del equipo comercial.',
            'kpis': ['Tasa de conversión +35% vs campaña masiva', 'Reducir llamadas en frío -50%', 'Top 20% clientes → 65% de ventas'],
            'algo': 'Gradient Boosting Classifier (n=150, max_depth=4, lr=0.1)',
        }
    },
    'Win / Loss de Oportunidades': {
        'key': 'winloss', 'data_key': 'winloss',
        'badge': '🤝 Clasificación',
        'colab': 'https://colab.research.google.com/drive/1iR7IWH8NwORRN4NgOHEqtziQ1G5f21-t',
        'runner': M.run_winloss, 'target': 'ganado',
        'features': ['monto_oportunidad_k','dias_ciclo','n_reuniones','n_competidores','decision_makers','propuesta_personalizada'],
        'criteria': {'auc': 0.75, 'recall': 0.68, 'precision': 0.62},
        'business': {
            'problem': '⚠️ El equipo comercial de Tecsup/ICPNA tiene **win rate del 28%**. Los gerentes no saben qué oportunidades priorizar y cuáles abandonar para enfocar esfuerzos.',
            'objective': 'Predecir la probabilidad de cierre de cada oportunidad B2B para que el equipo enfoque su tiempo en los deals con mayor chance de ganar.',
            'kpis': ['Win rate de 28% a 40%+', 'Identificar deals en riesgo con 2 semanas de anticipación', 'Reducir ciclo de venta promedio -20%'],
            'algo': 'Random Forest (n=200, max_depth=6, class_weight=balanced)',
        }
    },
    # ── NO SUPERVISADOS ─────────────────────────────────────────────
    'Segmentación RFM (K-Means)': {
        'key': 'rfm_kmeans', 'data_key': 'rfm',
        'tipo': 'no_supervisado',
        'badge': '👥 Clustering',
        'colab': 'https://colab.research.google.com/drive/1SoS96Sr4Cwy-BJxQ6TfOKjn4EHogk3Qc',
        'runner': M.run_kmeans, 'target': None,
        'features': ['recencia_dias','frecuencia_compras','monto_total_k'],
        'criteria': {'silhouette': 0.35, 'n_clusters': 4},
        'business': {
            'problem': '⚠️ Un retailer trata a todos sus clientes igual. El 70% del presupuesto de retención se gasta en clientes que no necesitan activación o que ya están perdidos.',
            'objective': 'Segmentar la cartera en grupos homogéneos por comportamiento RFM para diseñar estrategias diferenciadas por segmento.',
            'kpis': ['4 segmentos accionables (Champions, Potenciales, En riesgo, Perdidos)', 'Reducir costo de retención -30%', 'Incrementar reactivación de dormidos +20%'],
            'algo': 'K-Means (k=4, init=k-means++, StandardScaler sobre R-F-M)',
        }
    },
    'Segmentación Categórica (K-Modes)': {
        'key': 'kmodes', 'data_key': 'kmodes',
        'tipo': 'no_supervisado',
        'badge': '🗂️ Clustering',
        'colab': 'https://colab.research.google.com/drive/1e36DSXM6RJ7WtWY8bAHHdcitSV9QEcSj',
        'runner': M.run_kmodes, 'target': None,
        'features': ['canal_adquisicion','categoria_preferida','frecuencia_visita','region','edad_grupo','dispositivo'],
        'criteria': {'n_clusters': 3},
        'business': {
            'problem': '⚠️ ICPNA/Tecsup no conoce los perfiles de comportamiento de sus prospectos digitales. Todos reciben el mismo mensaje sin importar canal, dispositivo ni región.',
            'objective': 'Descubrir perfiles de cliente basados en variables categóricas de comportamiento para personalizar comunicaciones y canales.',
            'kpis': ['3 perfiles accionables con canal y mensaje diferenciado', 'CTR +25% con comunicación personalizada', 'Reducir costo por lead -20%'],
            'algo': 'K-Modes (init=Huang, distancia de Hamming para variables categóricas)',
        }
    },
    'Clustering Jerárquico': {
        'key': 'hierarchical', 'data_key': 'hierarchical',
        'tipo': 'no_supervisado',
        'badge': '🌳 Clustering',
        'colab': 'https://colab.research.google.com/drive/1SoS96Sr4Cwy-BJxQ6TfOKjn4EHogk3Qc',
        'runner': M.run_hierarchical, 'target': None,
        'features': ['recencia_dias','frecuencia_compras','ticket_prom_k','n_categorias','meses_cliente','canal_digital'],
        'criteria': {'silhouette': 0.30, 'cophenetic': 0.70},
        'business': {
            'problem': '⚠️ Un banco no sabe cuántos segmentos naturales existen en su cartera ni cómo de similares/distintos son entre sí. K-Means requiere fijar k a priori, lo cual es arbitrario.',
            'objective': 'Descubrir la estructura jerárquica de segmentos de clientes sin fijar k a priori, usando el dendrograma para decidir el corte óptimo.',
            'kpis': ['Número de clusters validado por dendrograma + coeficiente cofenético', 'Silhouette score >= 0.30', 'Taxonomía de clientes con distancias interpretables'],
            'algo': 'AgglomerativeClustering (linkage=ward) + dendrograma scipy',
        }
    },
    'Uplift · Incrementalidad': {
        'key': 'uplift', 'data_key': 'uplift',
        'badge': '⚡ Uplift Modeling',
        'colab': 'https://colab.research.google.com/drive/16pPOwZ7DfNlbYfY2sGDxdgcGBsILLRK7',
        'runner': M.run_uplift, 'target': 'convirtio',
        'features': ['edad','recencia_dias','n_compras_prev','ticket_prom_k','canal_digital'],
        'criteria': {'ate': 0.05, 'top10_lift': 0.15},
        'business': {
            'problem': '⚠️ Una campaña de retención bancaria genera **S/.180K de costo mensual**. El 40% del presupuesto se gasta en clientes que habrían comprado igual sin campaña ("falsos positivos de uplift").',
            'objective': 'Identificar qué clientes REALMENTE responden a la campaña (persuadibles) vs. los que comprarían igual, para concentrar el presupuesto donde genera impacto incremental.',
            'kpis': ['Reducir costo de campaña -40% con mismo revenue', 'Identificar top 25% "persuadibles"', 'ROAS incremental +2.5x vs campaña masiva'],
            'algo': 'Two-model approach: LogReg(tratados) − LogReg(control) → uplift score',
        }
    },
}

# ── SESSION STATE ──────────────────────────────────────────────────
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'dataframes' not in st.session_state:
    st.session_state.dataframes = {}

def get_df(model_name):
    cfg = MODEL_CONFIG[model_name]
    key = cfg['data_key']
    if key not in st.session_state.dataframes:
        st.session_state.dataframes[key] = DATASETS[key]()
    return st.session_state.dataframes[key]

def get_results(model_name):
    if model_name not in st.session_state.results:
        cfg = MODEL_CONFIG[model_name]
        df = get_df(model_name)
        with st.spinner('Entrenando modelo...'):
            result = cfg['runner'](df)
        st.session_state.results[model_name] = result
    return st.session_state.results[model_name]

# ── SIDEBAR ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:0 0 1rem'>
      <div style='font-size:18px;font-weight:700;color:#fff;letter-spacing:-.3px'>🔬 CRISP-DM Lab</div>
      <div style='font-size:11px;color:#555;font-family:monospace;margin-top:3px'>UPC - Digital Marketing Analytics</div>
    </div>
    """, unsafe_allow_html=True)

    tipo_filtro = st.radio(
        'Tipo',
        ['Todos', 'Supervisado', 'No supervisado'],
        horizontal=True,
        label_visibility='visible',
    )
    _tipo_key = {'Supervisado': 'supervisado', 'No supervisado': 'no_supervisado'}
    modelos_disp = {
        k: v for k, v in MODEL_CONFIG.items()
        if tipo_filtro == 'Todos'
        or v.get('tipo', 'supervisado') == _tipo_key.get(tipo_filtro, '')
    }

    model_name = st.radio(
        'Selecciona el modelo',
        list(modelos_disp.keys()),
        label_visibility='collapsed',
    )

    st.markdown('---')
    st.markdown('<div style="font-size:11px;color:#555;font-family:monospace;margin-bottom:.5rem">DATOS</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader('Cargar CSV propio', type='csv',
                                help='Reemplaza los datos del modelo activo')
    if uploaded:
        df_up = pd.read_csv(uploaded)
        cfg_up = MODEL_CONFIG[model_name]
        st.session_state.dataframes[cfg_up['data_key']] = df_up
        if model_name in st.session_state.results:
            del st.session_state.results[model_name]
        st.success(f'{len(df_up)} filas cargadas ✓')

    if st.button('↺ Resetear datos y resultados'):
        st.session_state.results = {}
        st.session_state.dataframes = {}
        st.rerun()

    st.markdown('---')
    cfg_cur = MODEL_CONFIG[model_name]
    st.markdown(f'<div style="font-size:11px;color:#555;margin-bottom:4px">{cfg_cur["badge"]}</div>', unsafe_allow_html=True)
    st.markdown(f'[↗ Abrir en Google Colab]({cfg_cur["colab"]})')

# ── MAIN ────────────────────────────────────────────────────────────
cfg = MODEL_CONFIG[model_name]
df = get_df(model_name)

st.markdown(f'<div class="phase-tag">{cfg["badge"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="phase-title">{model_name}</div>', unsafe_allow_html=True)

# CRISP-DM phases as tabs
phases = st.tabs([
    '1 · Business',
    '2 · Data Understanding',
    '3 · Data Preparation',
    '4 · Modeling',
    '5 · Evaluation',
    '6 · Deployment',
])

# ══════════════════════════════════════════════════════════════════
# FASE 1: BUSINESS UNDERSTANDING
# ══════════════════════════════════════════════════════════════════
with phases[0]:
    st.markdown('<div class="phase-desc">Definir el problema de negocio, los objetivos medibles y los criterios de éxito antes de tocar los datos.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader('Problema de negocio')
        st.markdown(cfg['business']['problem'])
        st.markdown('<br>', unsafe_allow_html=True)
        st.subheader('Objetivo analítico')
        st.info(cfg['business']['objective'])

    with col2:
        st.subheader('KPIs de éxito')
        for kpi in cfg['business']['kpis']:
            st.markdown(f'<div class="kpi-box">✓ {kpi}</div>', unsafe_allow_html=True)

        st.markdown('<br>', unsafe_allow_html=True)
        st.subheader('Criterios de aceptación')
        st.markdown('<div class="criteria-box">' +
                    ' · '.join([f'<b>{k.upper()}</b> {">" if k in ["auc","recall","precision","r2","f1_macro","accuracy","top2_accuracy"] else "<"} {v}' for k,v in cfg['criteria'].items()])
                    + '</div>', unsafe_allow_html=True)

    st.markdown('---')
    st.subheader('Algoritmo y variables')
    cola, colb = st.columns([1, 2])
    with cola:
        st.code(cfg['business']['algo'], language='python')
    with colb:
        if cfg['target']:
            feat_tags = ' '.join([f'`{f}`' for f in cfg['features']])
            st.markdown(f'**Variables X:** {feat_tags}')
            st.markdown(f'**Target Y:** `{cfg["target"]}`')
        else:
            feat_tags = ' '.join([f'`{f}`' for f in cfg['features']])
            st.markdown(f'**Variables:** {feat_tags}')
            st.markdown('**Output:** Reglas de asociación (soporte, confianza, lift)')

# ══════════════════════════════════════════════════════════════════
# FASE 2: DATA UNDERSTANDING / EDA
# ══════════════════════════════════════════════════════════════════
with phases[1]:
    st.markdown('<div class="phase-desc">Exploración completa de los datos: dimensiones, distribuciones, valores faltantes y correlaciones.</div>', unsafe_allow_html=True)

    # Overview metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric('Registros', f'{len(df):,}')
    c2.metric('Variables', len(df.columns))
    c3.metric('Variables X', len(cfg['features']))
    c4.metric('Valores nulos', int(df.isnull().sum().sum()))
    c5.metric('Duplicados', int(df.duplicated().sum()))

    st.markdown('---')
    st.subheader('Vista previa del dataset')
    st.dataframe(df.head(10), width="stretch")

    # Download sample
    csv_bytes = df.to_csv(index=False).encode()
    st.download_button('⬇ Descargar dataset completo (CSV)',
                       data=csv_bytes,
                       file_name=f'{cfg["data_key"]}_data.csv',
                       mime='text/csv')

    st.markdown('---')
    st.subheader('Estadísticas descriptivas')
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        st.dataframe(df[num_cols].describe().round(2), width="stretch")
    else:
        st.info('No hay columnas numéricas en el dataset.')

    # Target distribution
    if cfg['target'] and cfg['target'] in df.columns:
        st.markdown('---')
        c_td, c_cb = st.columns(2)
        with c_td:
            st.subheader(f'Distribución del target: `{cfg["target"]}`')
            st.plotly_chart(P.plot_target_dist(df[cfg['target']], cfg['target']),
                            width="stretch", key='eda_target_dist')
        with c_cb:
            st.subheader('Correlación con el target')
            num_feats = [f for f in cfg['features'] if f in df.columns
                         and pd.api.types.is_numeric_dtype(df[f])]
            target_is_numeric = (cfg['target'] is not None
                                  and pd.api.types.is_numeric_dtype(df[cfg['target']]))
            if num_feats and target_is_numeric:
                st.plotly_chart(P.plot_correlation_bar(df, num_feats, cfg['target']),
                                width="stretch", key='eda_corr_bar')
            elif num_feats:
                st.info('El target es categórico — correlación de Pearson no aplica.')

    # Histograms
    st.markdown('---')
    st.subheader('Distribución de variables numéricas')
    num_feats_plot = [f for f in cfg['features']
                      if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
    if num_feats_plot:
        st.plotly_chart(P.plot_histograms(df, num_feats_plot),
                        width="stretch", key='eda_histograms')

    # Boxplots by target for classification
    if cfg['target'] and cfg['target'] in df.columns and model_name != 'LTV · Regresión Lineal':
        st.markdown('---')
        st.subheader('Variables por clase del target')
        figs_box = P.plot_boxplots_by_target(df, num_feats_plot, cfg['target'])
        if figs_box:
            cols_box = st.columns(min(2, len(figs_box)))
            for i, fig in enumerate(figs_box):
                cols_box[i % 2].plotly_chart(fig, width="stretch", key=f'eda_box_{i}')

    # Correlation matrix
    if len(num_feats_plot) > 2:
        st.markdown('---')
        st.subheader('Matriz de correlaciones')
        all_num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
                   and c not in ['cliente_id','lead_id']][:10]
        st.plotly_chart(P.plot_correlation_heatmap(df, all_num),
                        width="stretch", key='eda_corr_heatmap')

    # Category analysis
    cat_cols = [c for c in df.columns
                if df[c].dtype == 'object' and c not in ['cliente_id','lead_id','ticket_id','semana']]
    if cat_cols:
        st.markdown('---')
        st.subheader('Variables categóricas')
        for i, cat in enumerate(cat_cols[:3]):
            vc = df[cat].value_counts().reset_index()
            vc.columns = [cat, 'count']
            fig_cat = P.plot_target_dist(df[cat], cat)
            st.plotly_chart(fig_cat, width="stretch", key=f'eda_cat_{i}')

# ══════════════════════════════════════════════════════════════════
# FASE 3: DATA PREPARATION
# ══════════════════════════════════════════════════════════════════
with phases[2]:
    st.markdown('<div class="phase-desc">Limpieza, transformaciones, feature engineering y división train/test.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Limpieza de datos')
        nulls = df.isnull().sum()
        # Convertir tipos a strings de forma segura para evitar conflictos con PyArrow
        tipo_strings = [str(dtype).replace('Int64', 'int64').replace('Int32', 'int32').replace('Int16', 'int16').replace('Int8', 'int8') 
                        for dtype in df.dtypes]
        clean_df = pd.DataFrame({
            'Variable': nulls.index.astype(str),
            'Nulos': nulls.values.astype('int64'),
            'Tipo': tipo_strings,
            'Estado': ['✓ OK' if n == 0 else f'⚠ {n} nulos' for n in nulls.values]
        })
        st.dataframe(clean_df, width="stretch", hide_index=True)

    with col2:
        st.subheader('División Train / Test (80/20)')
        n_total = len(df)
        n_train = int(n_total * 0.8)
        n_test = n_total - n_train
        st.metric('Total registros', n_total)
        c_tr, c_te = st.columns(2)
        c_tr.metric('Train (80%)', n_train)
        c_te.metric('Test (20%)', n_test)
        st.markdown('''
        <div class="criteria-box">
        División <b>estratificada</b> por la variable target para mantener la misma proporción de clases en train y test.
        </div>''', unsafe_allow_html=True)

    st.markdown('---')
    st.subheader('Transformaciones por variable')
    transforms = {
        'dias_inactivo': ('Normalización Min-Max', 'Escala 0-90, evita dominancia en modelos de distancia'),
        'quejas': ('Sin cambio (escala baja)', 'Variable discreta con rango pequeño (0-5)'),
        'logins_30d': ('Sin cambio', 'Variable discreta, escala natural'),
        'saldo_prom_k': ('Log-transform', 'Distribución sesgada a la derecha'),
        'dias_contacto': ('StandardScaler', 'Requerido por Regresión Logística'),
        'interacciones_web': ('StandardScaler', 'Normalizar para Reg. Logística'),
        'clicks_precio': ('StandardScaler', 'Normalizar para Reg. Logística'),
        'tiempo_pag_min': ('StandardScaler', 'Normalizar para Reg. Logística'),
        'ticket_prom': ('Log-transform', 'Distribución sesgada'),
        'compras_6m': ('Sin cambio', 'Variable discreta, escala natural'),
        'precio': ('Log-transform', 'Para modelo log-log elasticidad'),
        'unidades': ('Log-transform', 'Para modelo log-log elasticidad'),
        'tiene_tarjeta': ('Binaria 0/1', 'Ya codificada, sin cambio'),
        'producto_adquirido': ('Label Encoding', 'Multiclase → enteros 0..N'),
        'recencia_dias': ('Normalización Min-Max', 'Escala 1-120, más reciente = mayor propensión'),
        'n_compras_12m': ('Sin cambio', 'Variable discreta, señal directa de actividad'),
        'engagement_score': ('Sin cambio', 'Ya en escala 0-100'),
        'n_productos': ('Sin cambio', 'Variable discreta, proxy de vínculo con la marca'),
        'monto_oportunidad_k': ('Log-transform', 'Distribución log-normal, alta asimetría'),
        'dias_ciclo': ('Sin cambio', 'Días en pipeline, señal negativa si es muy alto'),
        'n_reuniones': ('Sin cambio', 'Proxy de esfuerzo e interés del cliente'),
        'n_competidores': ('Sin cambio', 'Variable discreta pequeña (0-4)'),
        'decision_makers': ('Sin cambio', 'Número de aprobadores — más = proceso más complejo'),
        'propuesta_personalizada': ('Binaria 0/1', 'Ya codificada'),
        'n_compras_prev': ('Sin cambio', 'Variable discreta, señal de comportamiento pasado'),
        'ticket_prom_k': ('Log-transform', 'Distribución log-normal del gasto'),
        'tratamiento': ('Binaria 0/1', 'Variable de asignación al grupo tratado/control'),
        'convirtio': ('Binaria 0/1', 'Target — conversión observada en el período'),
        'recencia_dias': ('StandardScaler', 'Necesario para K-Means — evita que R domine por escala'),
        'frecuencia_compras': ('StandardScaler', 'Normalizar para K-Means'),
        'monto_total_k': ('StandardScaler', 'Normalizar para K-Means'),
        'canal_adquisicion': ('Sin cambio (categórica)', 'K-Modes trabaja directamente con strings'),
        'categoria_preferida': ('Sin cambio (categórica)', 'K-Modes — distancia de Hamming'),
        'frecuencia_visita': ('Sin cambio (categórica)', 'K-Modes — sin encoding necesario'),
        'edad_grupo': ('Sin cambio (categórica)', 'K-Modes — ordinal tratada como nominal'),
        'dispositivo': ('Sin cambio (categórica)', 'K-Modes — sin encoding'),
        'n_categorias': ('StandardScaler', 'Normalizar para clustering jerárquico'),
        'meses_cliente': ('StandardScaler', 'Normalizar — rango amplio (1-60)'),
    }
    trans_rows = []
    for feat in cfg['features']:
        if feat in transforms:
            trans_rows.append({'Variable': feat,
                               'Transformación': transforms[feat][0],
                               'Motivo': transforms[feat][1]})
        else:
            trans_rows.append({'Variable': feat,
                               'Transformación': 'Sin cambio',
                               'Motivo': 'Variable ya en escala adecuada'})
    if cfg['target'] and cfg['target'] not in [r['Variable'] for r in trans_rows]:
        if cfg['target'] in transforms:
            trans_rows.append({'Variable': f'TARGET: {cfg["target"]}',
                               'Transformación': transforms[cfg['target']][0],
                               'Motivo': transforms[cfg['target']][1]})

    st.dataframe(pd.DataFrame(trans_rows), width="stretch", hide_index=True)

    st.markdown('---')
    st.subheader('Feature Engineering')
    fe_map = {
        'Churn Prediction': [
            ('ratio_logins_dias', 'logins_30d / (dias_inactivo + 1)', 'Frecuencia relativa de uso — señal de compromiso'),
            ('flag_multiquejas', 'int(quejas > 2)', 'Captura clientes con problemas recurrentes'),
        ],
        'Lead Scoring (Reg. Logística)': [
            ('score_engagement', 'interacciones_web × clicks_precio', 'Mide intención combinada de compra'),
            ('log_tiempo', 'log(tiempo_pag_min + 1)', 'Normaliza sesiones muy largas'),
        ],
        'LTV · Regresión Lineal': [
            ('valor_anualizado', 'ticket_prom × compras_6m × 2', 'Proyección de gasto anual simple'),
            ('diversidad', 'categorias / 7', 'Diversificación normalizada (0-1)'),
        ],
        'Elasticidad de Precio': [
            ('log_precio', 'log(precio)', 'Linealiza relación precio-demanda'),
            ('log_unidades', 'log(unidades)', 'Linealiza la variable dependiente'),
        ],
        'Market Basket Analysis': [
            ('co_ocurrencia', 'Matriz producto × producto', 'Conteo de co-aparición en tickets'),
            ('soporte_item', 'freq(item) / total_tickets', 'Frecuencia individual de cada producto'),
        ],
        'Next Best Offer': [
            ('ratio_productos', 'sum(tiene_*) / 4', 'Profundidad relacional del cliente'),
            ('saldo_segmento', 'saldo_k × segmento_encoded', 'Interacción saldo × tier de cliente'),
        ],
        'Propensión a Compra (GBM)': [
            ('recencia_score', '1 / (recencia_dias + 1)', 'Inverso de recencia — más reciente = mayor score'),
            ('rfm_proxy', 'n_compras_12m × engagement_score / 100', 'Combina frecuencia e intensidad de engagement'),
        ],
        'Win / Loss de Oportunidades': [
            ('ratio_reuniones_ciclo', 'n_reuniones / dias_ciclo', 'Velocidad de avance — deals rápidos ganan más'),
            ('flag_competencia_alta', 'int(n_competidores >= 3)', 'Escenario de alta competencia — señal negativa'),
        ],
        'Uplift · Incrementalidad': [
            ('rfm_score', 'n_compras_prev × (1/recencia_dias) × ticket_prom_k', 'Score RFM simplificado — predice base de conversión'),
            ('segmento_valor', 'pd.cut(ticket_prom_k, bins=[0,2,8,20], labels=[0,1,2])', 'Segmento de valor — efecto del tratamiento varía por segmento'),
        ],
        'Segmentación RFM (K-Means)': [
            ('recencia_norm', 'StandardScaler(recencia_dias)', 'Normalización crítica — sin esto R domina el clustering'),
            ('valor_cliente', 'frecuencia_compras × monto_total_k', 'Proxy de valor total — combina F y M del RFM'),
        ],
        'Segmentación Categórica (K-Modes)': [
            ('perfil_digital', 'canal IN (Google, Facebook) AND dispositivo=Móvil', 'Flag de perfil digital — derivado de 2 features'),
            ('intensidad_visita', 'map(frecuencia_visita → {Diaria:4, Semanal:3, Mensual:2, Esporádica:1})', 'Ordinalización de frecuencia para análisis complementario'),
        ],
        'Clustering Jerárquico': [
            ('valor_cliente', 'frecuencia_compras × ticket_prom_k', 'Revenue potencial por cliente'),
            ('engagement_score', '(1/recencia_dias) × frecuencia_compras × canal_digital', 'Score sintético de engagement'),
        ],
    }
    fe_rows = fe_map.get(model_name, [])
    if fe_rows:
        cols_fe = st.columns(len(fe_rows))
        for i, (name, formula, desc) in enumerate(fe_rows):
            with cols_fe[i]:
                st.code(f'{name} = {formula}', language='python')
                st.caption(desc)

# ══════════════════════════════════════════════════════════════════
# FASE 4: MODELING
# ══════════════════════════════════════════════════════════════════
with phases[3]:
    st.markdown('<div class="phase-desc">Entrenamiento real del modelo con scikit-learn, ajuste de hiperparámetros y simulador interactivo de predicciones.</div>', unsafe_allow_html=True)

    # Hyperparameters sidebar in this phase
    col_hp, col_res = st.columns([1, 2])

    with col_hp:
        st.subheader('Hiperparámetros')

        if cfg['key'] in ('churn', 'nbo', 'winloss'):
            n_est = st.slider('n_estimators', 50, 500, 200, 50)
            max_d = st.slider('max_depth', 2, 15, 6)
        elif cfg['key'] == 'propension':
            n_est = st.slider('n_estimators (GBM)', 50, 300, 150, 25)
            max_d = st.slider('max_depth', 2, 8, 4)
            lr_val = st.slider('learning_rate', 0.01, 0.30, 0.10, 0.01)
        elif cfg['key'] == 'logit':
            C_val = st.slider('C (regularización)', 0.1, 10.0, 1.0, 0.1)
            max_iter = st.slider('max_iter', 100, 1000, 300, 100)
        elif cfg['key'] == 'rfm_kmeans':
            n_clusters_km = st.slider('Número de clusters (k)', 2, 8, 4)
            st.info('Usa la curva de codo en Fase 5 para validar el k óptimo.')
        elif cfg['key'] == 'kmodes':
            n_clusters_kmo = st.slider('Número de clusters (k)', 2, 6, 3)
            st.info('K-Modes usa distancia de Hamming — no necesita escalar variables.')
        elif cfg['key'] == 'hierarchical':
            n_clusters_hc = st.slider('Número de clusters', 2, 8, 4)
            linkage_method = st.selectbox('Método de enlace',
                                          ['ward', 'complete', 'average', 'single'])
            st.info('Consulta el dendrograma (Fase 5) para elegir el corte óptimo.')
        elif cfg['key'] == 'uplift':
            C_val = st.slider('C (regularización LogReg)', 0.01, 5.0, 1.0, 0.01)
            st.info('Se entrenan dos modelos independientes: uno sobre el grupo **tratado** y otro sobre el **control**.')
        elif cfg['key'] == 'ltv':
            alpha = st.slider('alpha (Ridge)', 0.01, 10.0, 1.0, 0.01)
        elif cfg['key'] == 'elastic':
            st.info('Modelo log-log OLS. Sin hiperparámetros adicionales.')
        elif cfg['key'] == 'basket':
            min_sup = st.slider('min_support', 0.02, 0.30, 0.05, 0.01)
            min_conf = st.slider('min_confidence', 0.20, 0.90, 0.35, 0.05)
            min_lift = st.slider('min_lift', 1.0, 5.0, 1.0, 0.1)

        train_btn = st.button('▶ Entrenar modelo', type='primary',
                              width="stretch")
        if train_btn:
            if model_name in st.session_state.results:
                del st.session_state.results[model_name]

    with col_res:
        st.subheader('Resultado del entrenamiento')

        if cfg['key'] == 'basket':
            ms = min_sup if 'min_sup' in dir() else 0.05
            mc = min_conf if 'min_conf' in dir() else 0.35
            ml = min_lift if 'min_lift' in dir() else 1.0
            result = M.run_basket(df, min_support=ms,
                                   min_confidence=mc, min_lift=ml)
            if result:
                st.session_state.results[model_name] = result
        elif cfg['key'] == 'rfm_kmeans':
            k = n_clusters_km if 'n_clusters_km' in dir() else 4
            result = M.run_kmeans(df, n_clusters=k)
            st.session_state.results[model_name] = result
        elif cfg['key'] == 'kmodes':
            k = n_clusters_kmo if 'n_clusters_kmo' in dir() else 3
            result = M.run_kmodes(df, n_clusters=k)
            st.session_state.results[model_name] = result
        elif cfg['key'] == 'hierarchical':
            k    = n_clusters_hc    if 'n_clusters_hc'    in dir() else 4
            link = linkage_method   if 'linkage_method'   in dir() else 'ward'
            result = M.run_hierarchical(df, n_clusters=k, linkage=link)
            st.session_state.results[model_name] = result
        else:
            result = get_results(model_name)

        if result is None:
            st.error('No se pudo entrenar el modelo. Verifica que el CSV tenga las columnas requeridas.')
        elif 'error' in result:
            st.warning(f'Error: {result["error"]}')
        else:
            mets = result.get('metrics', {})
            if 'auc_roc' in mets:
                m1,m2,m3,m4 = st.columns(4)
                m1.metric('Accuracy', f'{mets["accuracy"]:.1%}')
                m2.metric('Precision', f'{mets["precision"]:.1%}')
                m3.metric('Recall', f'{mets["recall"]:.1%}')
                m4.metric('AUC-ROC', f'{mets["auc_roc"]:.3f}')
            elif 'r2' in mets and 'mape' in mets:
                m1,m2,m3,m4 = st.columns(4)
                m1.metric('R²', f'{mets["r2"]:.3f}')
                m2.metric('RMSE', f'{mets["rmse"]:,.0f}')
                m3.metric('MAE', f'{mets["mae"]:,.0f}')
                m4.metric('MAPE', f'{mets["mape"]:.1f}%')
            elif 'elasticity' in mets:
                m1,m2,m3 = st.columns(3)
                m1.metric('R² (log-log)', f'{mets["r2"]:.3f}')
                m2.metric('Elasticidad β', f'{mets["elasticity"]:.3f}')
                m3.metric('RMSE (log)', f'{mets["rmse"]:.3f}')
            elif 'f1_macro' in mets:
                m1,m2,m3 = st.columns(3)
                m1.metric('Accuracy', f'{mets["accuracy"]:.1%}')
                m2.metric('F1 Macro', f'{mets["f1_macro"]:.3f}')
                m3.metric('Top-2 Acc.', f'{mets["top2_accuracy"]:.1%}')
            elif 'rules' in result:
                rules_df = result['rules']
                m1,m2,m3 = st.columns(3)
                m1.metric('Transacciones', result.get('n_transactions',0))
                m2.metric('Ítems únicos', result.get('n_items',0))
                m3.metric('Reglas generadas', len(rules_df))
            elif 'inertia' in mets:
                m1,m2,m3,m4 = st.columns(4)
                m1.metric('Clusters', mets['n_clusters'])
                m2.metric('Silhouette', f'{mets["silhouette"]:.3f}')
                m3.metric('Inercia', f'{mets["inertia"]:,.0f}')
                m4.metric('Calinski-Harabasz', f'{mets["calinski_harabasz"]:.0f}')
            elif 'cost' in mets:
                m1,m2 = st.columns(2)
                m1.metric('Clusters', mets['n_clusters'])
                m2.metric('Costo K-Modes', f'{mets["cost"]:,.0f}')
            elif 'cophenetic' in mets:
                m1,m2,m3 = st.columns(3)
                m1.metric('Clusters', mets['n_clusters'])
                m2.metric('Silhouette', f'{mets["silhouette"]:.3f}')
                m3.metric('Coef. Cofenético', f'{mets["cophenetic"]:.3f}')

    # Feature Importance / Coefficients
    st.markdown('---')
    result_now = st.session_state.results.get(model_name)
    if result_now and 'error' not in result_now:
        col_fi, col_sim = st.columns([1, 1])
        with col_fi:
            if 'feature_importance' in result_now:
                st.subheader('Importancia de variables')
                st.plotly_chart(P.plot_feature_importance(result_now['feature_importance']),
                                width="stretch", key='mod_feat_importance')
            elif 'coefficients' in result_now:
                st.subheader('Coeficientes del modelo')
                st.plotly_chart(P.plot_coeff_bar(result_now['coefficients']),
                                width="stretch", key='mod_coeff_bar')
            elif cfg['key'] == 'elastic' and 'elasticity_by_cat' in result_now:
                st.subheader('Elasticidad por categoría')
                ec = result_now['elasticity_by_cat']
                for cat, val in sorted(ec.items(), key=lambda x: x[1]):
                    tipo = '🔴 Elástico' if abs(val)>1 else '🟡 Inelástico'
                    st.metric(cat, f'β = {val:.3f}', delta=tipo, delta_color='off')

        with col_fi:
            if cfg['key'] in ('rfm_kmeans', 'hierarchical') and result_now and 'cluster_profiles' in result_now:
                st.subheader('Perfil de clusters')
                feat_cols = [c for c in result_now['cluster_profiles'].columns
                             if c not in ('cluster', 'n')]
                st.plotly_chart(P.plot_cluster_profiles(result_now['cluster_profiles'], feat_cols),
                                width="stretch", key='mod_cluster_profiles')
            elif cfg['key'] == 'kmodes' and result_now and 'cluster_modes' in result_now:
                st.subheader('Moda por cluster (K-Modes)')
                st.dataframe(result_now['cluster_modes'], width="stretch")

        with col_sim:
            st.subheader('Simulador de predicción' if cfg.get('tipo','supervisado') == 'supervisado' else 'Asignador de cluster')
            st.caption('Ajusta los valores y obtén la predicción del modelo entrenado')

            if cfg['key'] == 'churn' and result_now.get('model'):
                d = st.slider('Días inactivo', 0, 90, 15)
                q = st.slider('Quejas', 0, 10, 1)
                l = st.slider('Logins (30d)', 0, 30, 14)
                s = st.slider('Saldo (S/.k)', 0.0, 50.0, 12.0)
                p = st.slider('Productos activos', 1, 6, 3)
                X_sim = pd.DataFrame([[d,q,l,s,p]], columns=result_now['features'])
                prob = result_now['model'].predict_proba(X_sim)[0][1]
                pct_s = int(prob*100)
                st.metric('Score de churn', f'{pct_s}%')
                if prob >= 0.65:
                    st.markdown(f'<div class="prob-high">🚨 <b>Riesgo alto</b> · Llamada del asesor en 24h. Oferta de retención personalizada.</div>', unsafe_allow_html=True)
                elif prob >= 0.35:
                    st.markdown(f'<div class="prob-med">⚠️ <b>Riesgo medio</b> · Email + beneficio exclusivo. Encuesta de satisfacción.</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="prob-low">✅ <b>Riesgo bajo</b> · Mantener engagement. Cross-sell de producto complementario.</div>', unsafe_allow_html=True)

            elif cfg['key'] == 'logit' and result_now.get('model'):
                d = st.slider('Días desde contacto', 1, 30, 7)
                w = st.slider('Interacciones web', 0, 20, 6)
                c = st.slider('Clicks en precio', 0, 10, 3)
                t = st.slider('Tiempo en página (min)', 0, 30, 8)
                X_sim = result_now['scaler'].transform([[d, w, c, t]])
                prob = result_now['model'].predict_proba(X_sim)[0][1]
                pct_s = int(prob*100)
                st.metric('P(matrícula)', f'{pct_s}%')
                if prob >= 0.65:
                    st.markdown(f'<div class="prob-low">✅ <b>PRIORIDAD ALTA</b> · Llamar hoy. Probabilidad de conversión: {pct_s}%</div>', unsafe_allow_html=True)
                elif prob >= 0.40:
                    st.markdown(f'<div class="prob-med">⚠️ <b>PRIORIDAD MEDIA</b> · Nutrir con contenido. Re-evaluar en 3 días.</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="prob-high">❌ <b>PRIORIDAD BAJA</b> · Automatizar. Re-evaluar en 7 días.</div>', unsafe_allow_html=True)

            elif cfg['key'] == 'ltv' and result_now.get('model'):
                c_s = st.slider('Compras últimos 6m', 1, 20, 6)
                t_s = st.slider('Ticket promedio (S/.)', 50, 800, 200, 10)
                m_s = st.slider('Meses como cliente', 1, 60, 18)
                k_s = st.slider('Categorías distintas', 1, 8, 3)
                digital_s = st.selectbox('Canal digital', [0, 1], format_func=lambda x: 'Sí' if x else 'No')
                X_sim = pd.DataFrame([[c_s,t_s,m_s,k_s,digital_s]], columns=result_now['features'])
                ltv_pred = result_now['model'].predict(X_sim)[0]
                st.metric('LTV estimado (12m)', f'S/.{max(ltv_pred,0):,.0f}')
                seg = 'VIP 🏆' if ltv_pred>5000 else ('Regular' if ltv_pred>1500 else 'Bajo')
                cac_max = 500 if ltv_pred>5000 else (150 if ltv_pred>1500 else 50)
                st.metric('Segmento', seg)
                st.markdown(f'<div class="criteria-box">CAC máximo recomendado: <b>S/.{cac_max}</b> (ratio LTV:CAC ≥ 10:1)</div>', unsafe_allow_html=True)

            elif cfg['key'] == 'elastic' and result_now.get('model'):
                p_base = st.slider('Precio base (S/.)', 50, 500, 150, 5)
                p_new  = st.slider('Nuevo precio (S/.)', 50, 500, 120, 5)
                d_base = st.slider('Demanda base (unidades)', 100, 5000, 1000, 50)
                beta = result_now['metrics']['elasticity']
                d_new = int(d_base * (p_new/p_base)**beta)
                rev_base = p_base * d_base
                rev_new = p_new * d_new
                delta_rev = (rev_new - rev_base) / rev_base * 100
                c1s, c2s = st.columns(2)
                c1s.metric('Nueva demanda', f'{d_new:,} u.')
                c2s.metric('Δ Revenue', f'{delta_rev:+.1f}%',
                            delta=f'S/.{rev_new-rev_base:+,.0f}')
                tipo = 'elástico (sensible al precio)' if abs(beta)>1 else 'inelástico'
                st.caption(f'Elasticidad del modelo: β = {beta:.3f} · Producto {tipo}')

            elif cfg['key'] == 'basket':
                rules_df = result_now.get('rules', pd.DataFrame())
                if not rules_df.empty:
                    st.write(f'**{len(rules_df)} reglas** encontradas con los parámetros actuales.')
                    top3 = rules_df.head(3)
                    for _, row in top3.iterrows():
                        st.markdown(f'`{row["antecedents"]}` → `{row["consequents"]}` · lift {row["lift"]:.2f} · conf {row["confidence"]:.0%}')
                else:
                    st.info('No hay reglas con los umbrales actuales. Reduce los valores.')

            elif cfg['key'] == 'nbo' and result_now.get('model'):
                tc_s = st.selectbox('Tarjeta crédito', [0,1], format_func=lambda x: 'Sí' if x else 'No')
                ca_s = st.selectbox('Cuenta ahorros', [0,1], format_func=lambda x: 'Sí' if x else 'No')
                seg_s = st.selectbox('Seguro vida', [0,1], format_func=lambda x: 'Sí' if x else 'No')
                pr_s = st.selectbox('Préstamo personal', [0,1], format_func=lambda x: 'Sí' if x else 'No')
                ant_s = st.slider('Antigüedad (meses)', 1, 60, 24)
                sal_s = st.slider('Saldo (S/.k)', 1.0, 50.0, 10.0)
                X_sim = pd.DataFrame([[tc_s,ca_s,seg_s,pr_s,ant_s,sal_s]],
                                      columns=result_now['features'])
                probs = result_now['model'].predict_proba(X_sim)[0]
                ranked = sorted(zip(result_now['classes'], probs),
                                key=lambda x: x[1], reverse=True)
                st.markdown('**Top 3 recomendaciones:**')
                for i, (prod, p) in enumerate(ranked[:3]):
                    st.markdown(f'**{i+1}.** {prod} — {p:.0%}')

            elif cfg['key'] == 'propension' and result_now.get('model'):
                rec_s = st.slider('Recencia (días)', 1, 120, 30)
                comp_s = st.slider('Compras últimos 12m', 0, 15, 4)
                eng_s = st.slider('Engagement score', 10, 100, 55)
                cd_s = st.selectbox('Canal digital', [0,1], format_func=lambda x: 'Sí' if x else 'No', key='sim_cd')
                edad_s = st.slider('Edad', 22, 65, 38)
                np_s = st.slider('Nº productos', 1, 5, 2)
                X_sim = pd.DataFrame([[rec_s,comp_s,eng_s,cd_s,edad_s,np_s]],
                                      columns=result_now['features'])
                prob = result_now['model'].predict_proba(X_sim)[0][1]
                pct_s = int(prob * 100)
                st.metric('P(compra próximos 15d)', f'{pct_s}%')
                if prob >= 0.65:
                    st.markdown('<div class="prob-low">✅ <b>ALTA PROPENSIÓN</b> · Contactar esta semana. Canal preferido: digital.</div>', unsafe_allow_html=True)
                elif prob >= 0.35:
                    st.markdown('<div class="prob-med">⚠️ <b>PROPENSIÓN MEDIA</b> · Incluir en campaña email + retargeting.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="prob-high">❌ <b>BAJA PROPENSIÓN</b> · Excluir de campaña. Nutrir con contenido.</div>', unsafe_allow_html=True)

            elif cfg['key'] == 'winloss' and result_now.get('model'):
                mont_s = st.slider('Monto oportunidad (S/.k)', 10, 500, 80, 10)
                dias_s = st.slider('Días en pipeline', 7, 180, 45)
                reun_s = st.slider('Reuniones realizadas', 1, 10, 3)
                comp_s = st.slider('Competidores', 0, 4, 2)
                dm_s = st.slider('Decision makers', 1, 5, 2)
                prop_s = st.selectbox('Propuesta personalizada', [0,1], format_func=lambda x: 'Sí' if x else 'No', key='sim_prop')
                X_sim = pd.DataFrame([[mont_s,dias_s,reun_s,comp_s,dm_s,prop_s]],
                                      columns=result_now['features'])
                prob = result_now['model'].predict_proba(X_sim)[0][1]
                pct_s = int(prob * 100)
                st.metric('P(ganar deal)', f'{pct_s}%')
                if prob >= 0.65:
                    st.markdown('<div class="prob-low">✅ <b>DEAL CALIENTE</b> · Priorizar. Asignar ejecutivo senior.</div>', unsafe_allow_html=True)
                elif prob >= 0.35:
                    st.markdown('<div class="prob-med">⚠️ <b>DEAL EN RIESGO</b> · Activar diferenciadores. Revisar propuesta.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="prob-high">❌ <b>DEAL FRÍO</b> · Evaluar si vale continuar. Costo de oportunidad alto.</div>', unsafe_allow_html=True)

            elif cfg['key'] == 'uplift' and result_now.get('model_t'):
                edad_s = st.slider('Edad', 20, 65, 38)
                rec_s = st.slider('Recencia (días)', 1, 90, 20)
                comp_s = st.slider('Compras previas', 0, 12, 3)
                tick_s = st.slider('Ticket promedio (S/.k)', 0.5, 20.0, 4.0)
                cd_s = st.selectbox('Canal digital', [0,1], format_func=lambda x: 'Sí' if x else 'No', key='sim_upl')
                X_sim_raw = np.array([[edad_s, rec_s, comp_s, tick_s, cd_s]])
                X_sim_s = result_now['scaler'].transform(X_sim_raw)
                p_t = result_now['model_t'].predict_proba(X_sim_s)[0][1]
                p_c = result_now['model_c'].predict_proba(X_sim_s)[0][1]
                uplift = p_t - p_c
                st.metric('P(conv | tratado)', f'{p_t:.1%}')
                st.metric('P(conv | sin campaña)', f'{p_c:.1%}')
                st.metric('Uplift incremental', f'{uplift:+.1%}')
                if uplift >= 0.15:
                    st.markdown('<div class="prob-low">✅ <b>PERSUADIBLE</b> · Incluir en campaña. Alto impacto incremental.</div>', unsafe_allow_html=True)
                elif uplift >= 0.05:
                    st.markdown('<div class="prob-med">⚠️ <b>RESPONDE ALGO</b> · Incluir solo si hay presupuesto disponible.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="prob-high">❌ <b>NO PERSUADIBLE</b> · Excluir de campaña. Compraría igual o no responde.</div>', unsafe_allow_html=True)

            elif cfg['key'] == 'rfm_kmeans' and result_now and result_now.get('model'):
                rec_s  = st.slider('Recencia (días)',      1, 180, 30)
                frec_s = st.slider('Frecuencia compras',   1, 30,  8)
                mont_s = st.slider('Monto total (S/.k)',   0.5, 25.0, 6.0)
                X_raw  = np.array([[rec_s, frec_s, mont_s]])
                X_sc   = result_now['scaler'].transform(X_raw)
                cluster_id = result_now['model'].predict(X_sc)[0]
                st.metric('Cluster asignado', f'Cluster {cluster_id + 1}')
                profile_row = result_now['cluster_profiles'].iloc[cluster_id]
                st.markdown(f'**Perfil medio del cluster:** '
                            f'Recencia {profile_row["recencia_dias"]:.0f}d · '
                            f'Frec {profile_row["frecuencia_compras"]:.0f} · '
                            f'Monto S/.{profile_row["monto_total_k"]:.1f}k')

            elif cfg['key'] == 'hierarchical' and result_now and result_now.get('model'):
                rec_s  = st.slider('Recencia (días)',       1, 180, 30)
                frec_s = st.slider('Frecuencia compras',    1, 25,  8)
                tick_s = st.slider('Ticket promedio (S/.k)',0.5, 20.0, 5.0)
                cat_s  = st.slider('N° categorías',         1, 8,   3)
                ant_s  = st.slider('Antigüedad (meses)',    1, 60,  18)
                dig_s  = st.selectbox('Canal digital', [0,1], format_func=lambda x: 'Sí' if x else 'No', key='sim_hc')
                X_raw  = np.array([[rec_s, frec_s, tick_s, cat_s, ant_s, dig_s]])
                X_sc   = result_now['scaler'].transform(X_raw)
                # Hierarchical no tiene predict → asignar por distancia al centroide
                profiles = result_now['cluster_profiles']
                feat_cols = [c for c in profiles.columns if c not in ('cluster','n')]
                centroids = profiles[feat_cols].values
                centroids_sc = result_now['scaler'].transform(centroids)
                dists = np.linalg.norm(centroids_sc - X_sc, axis=1)
                cluster_id = dists.argmin()
                seg_name = profiles.iloc[cluster_id]['cluster']
                st.metric('Segmento asignado', seg_name)
                st.caption(f'Distancia al centroide: {dists[cluster_id]:.3f}')

            elif cfg['key'] == 'kmodes' and result_now and result_now.get('model'):
                st.info('K-Modes clasifica nuevos registros por moda. '
                        'Ingresa los atributos para asignar el perfil.')
                canal_s = st.selectbox('Canal', ['Google Ads','Facebook','Email','Orgánico','Referido'])
                cat_s   = st.selectbox('Categoría', ['Tecnología','Moda','Hogar','Deportes','Libros'])
                frec_s  = st.selectbox('Frecuencia', ['Diaria','Semanal','Mensual','Esporádica'])
                reg_s   = st.selectbox('Región', ['Lima','Arequipa','Trujillo','Cusco','Otra'])
                edad_s  = st.selectbox('Edad', ['18-24','25-34','35-44','45-54','55+'])
                disp_s  = st.selectbox('Dispositivo', ['Móvil','Desktop','Tablet'])
                X_new   = np.array([[canal_s, cat_s, frec_s, reg_s, edad_s, disp_s]])
                cluster_id = result_now['model'].predict(X_new)[0]
                st.metric('Perfil asignado', f'Perfil {cluster_id + 1}')
                st.dataframe(result_now['cluster_modes'].iloc[[cluster_id]],
                             width="stretch", hide_index=True)

# ══════════════════════════════════════════════════════════════════
# FASE 5: EVALUATION
# ══════════════════════════════════════════════════════════════════
with phases[4]:
    st.markdown('<div class="phase-desc">Métricas en el conjunto de test, matrices de evaluación y validación contra los criterios de aceptación del negocio.</div>', unsafe_allow_html=True)

    result_ev = st.session_state.results.get(model_name)
    if result_ev is None:
        result_ev = get_results(model_name)

    if result_ev is None or 'error' in result_ev:
        st.warning('Entrena el modelo primero (Fase 4 → ▶ Entrenar modelo)')
    else:
        mets = result_ev.get('metrics', {})
        crit = cfg['criteria']

        # ── K-Means evaluation ──
        if 'inertia' in mets:
            st.subheader('Métricas de clustering')
            c1,c2,c3,c4 = st.columns(4)
            c1.metric('Clusters (k)', mets['n_clusters'])
            c2.metric('Silhouette', f'{mets["silhouette"]:.3f}',
                       delta='✓' if mets['silhouette'] >= crit.get('silhouette',0) else '✗')
            c3.metric('Calinski-Harabász', f'{mets["calinski_harabasz"]:.0f}')
            c4.metric('Davies-Bouldin', f'{mets["davies_bouldin"]:.3f}',
                       delta='↓ mejor si bajo', delta_color='off')
            with st.expander('¿Cómo interpretar estas métricas?'):
                st.markdown('''
- **Silhouette [-1, 1]**: cuánto mejor es cada punto en su propio cluster vs el vecino. **>0.25** es aceptable, **>0.50** es bueno.
- **Calinski-Harabász**: razón varianza entre-clusters / intra-clusters. Más alto = clusters más compactos y separados.
- **Davies-Bouldin**: similitud media entre clusters. **Más bajo = mejor** (0 es perfecto).
                ''')

            st.markdown('---')
            col_el, col_sc = st.columns(2)
            with col_el:
                st.subheader('Curva de codo')
                st.plotly_chart(P.plot_elbow_curve(result_ev['elbow_data']),
                                width="stretch", key='eval_elbow')
            with col_sc:
                st.subheader('Scatter RFM por cluster')
                if all(c in result_ev['df_scored'].columns for c in ['recencia_dias','monto_total_k']):
                    st.plotly_chart(P.plot_cluster_scatter_2d(
                        result_ev['df_scored'], 'recencia_dias', 'monto_total_k'),
                        width="stretch", key='eval_rfm_scatter')

            st.markdown('---')
            st.subheader('Perfil de clusters')
            feat_cols = [c for c in result_ev['cluster_profiles'].columns
                         if c not in ('cluster', 'n')]
            st.plotly_chart(P.plot_cluster_profiles(result_ev['cluster_profiles'], feat_cols),
                            width="stretch", key='eval_cluster_profiles')
            st.markdown('**Tamaño de cada cluster:**')
            sizes = result_ev['df_scored']['cluster_label'].value_counts().sort_index()
            st.bar_chart(sizes)

            if 'segmento_rfm' in result_ev['df_scored'].columns:
                st.markdown('---')
                st.subheader('Validación vs segmento real (ground truth)')
                ct = pd.crosstab(result_ev['df_scored']['cluster_label'],
                                 result_ev['df_scored']['segmento_rfm'])
                st.dataframe(ct, width="stretch")

        # ── K-Modes evaluation ──
        elif 'cost' in mets:
            st.subheader('Métricas de clustering categórico')
            c1,c2 = st.columns(2)
            c1.metric('Perfiles encontrados', mets['n_clusters'])
            c2.metric('Costo total (disimilitud)', f'{mets["cost"]:,.0f}',
                       delta='↓ mejor si bajo', delta_color='off')

            st.markdown('---')
            col_elb, col_prof = st.columns(2)
            with col_elb:
                st.subheader('Curva de costo vs k')
                st.plotly_chart(P.plot_elbow_curve(result_ev['elbow_data'],
                                                    metric='cost', label='Costo K-Modes'),
                                width="stretch", key='eval_kmodes_elbow')
            with col_prof:
                st.subheader('Tamaño de perfiles')
                sizes = result_ev['df_scored']['cluster_label'].value_counts().sort_index()
                st.bar_chart(sizes)

            st.markdown('---')
            st.subheader('Distribución de valores por perfil y variable')
            st.plotly_chart(P.plot_kmodes_heatmap(result_ev['profiles_detail'],
                                                    result_ev['features']),
                            width="stretch", key='eval_kmodes_heatmap')

            if 'perfil_real' in result_ev['df_scored'].columns:
                st.markdown('---')
                st.subheader('Validación vs perfil real')
                ct = pd.crosstab(result_ev['df_scored']['cluster_label'],
                                 result_ev['df_scored']['perfil_real'])
                st.dataframe(ct, width="stretch")

        # ── Hierarchical evaluation ──
        elif 'cophenetic' in mets:
            st.subheader('Métricas de clustering jerárquico')
            c1,c2,c3 = st.columns(3)
            c1.metric('Clusters (k)', mets['n_clusters'])
            c2.metric('Silhouette', f'{mets["silhouette"]:.3f}',
                       delta='✓' if mets['silhouette'] >= crit.get('silhouette',0) else '✗')
            c3.metric('Coef. Cofenético', f'{mets["cophenetic"]:.3f}',
                       delta='✓' if mets['cophenetic'] >= crit.get('cophenetic',0) else '✗')

            st.markdown('---')
            st.subheader('Dendrograma — estructura jerárquica de clusters')
            st.plotly_chart(P.plot_dendrogram(result_ev['linkage_matrix']),
                            width="stretch", key='eval_dendrogram')
            with st.expander('¿Cómo leer el dendrograma?'):
                st.markdown('''
- El **eje Y** muestra la distancia de fusión — más alto = grupos más distintos
- Las **ramas** que se unen más arriba indican segmentos más diferentes entre sí
- El **corte horizontal** define el número de clusters: traza una línea donde la distancia de fusión da el salto más grande
- **Coeficiente cofenético > 0.70** indica que el dendrograma representa bien la estructura real de los datos
                ''')

            st.markdown('---')
            st.subheader('Perfil de segmentos')
            feat_cols = [c for c in result_ev['cluster_profiles'].columns
                         if c not in ('cluster', 'n')]
            st.plotly_chart(P.plot_cluster_profiles(result_ev['cluster_profiles'], feat_cols),
                            width="stretch", key='eval_hier_profiles')

            if 'segmento_real' in result_ev['df_scored'].columns:
                st.markdown('---')
                st.subheader('Comparación vs segmentos reales')
                ct = pd.crosstab(result_ev['df_scored']['cluster_label'],
                                 result_ev['df_scored']['segmento_real'])
                st.dataframe(ct, width="stretch")

        # ── Classification evaluation ──
        elif 'auc_roc' in mets:
            st.subheader('Métricas en test set')
            c1,c2,c3,c4,c5 = st.columns(5)
            def pass_fail(v, thr, higher=True):
                return '✓' if (v >= thr if higher else v <= thr) else '✗'
            c1.metric('Accuracy', f'{mets["accuracy"]:.1%}')
            c2.metric('Precision', f'{mets["precision"]:.1%}',
                       delta=pass_fail(mets['precision'], crit.get('precision',0)))
            c3.metric('Recall', f'{mets["recall"]:.1%}',
                       delta=pass_fail(mets['recall'], crit.get('recall',0)))
            c4.metric('F1-Score', f'{mets["f1"]:.3f}')
            c5.metric('AUC-ROC', f'{mets["auc_roc"]:.3f}',
                       delta=pass_fail(mets['auc_roc'], crit.get('auc',0)))

            st.markdown('---')
            col_cm, col_roc = st.columns(2)
            with col_cm:
                st.subheader('Matriz de confusión')
                cm = result_ev['confusion_matrix']
                labels = [str(c) for c in sorted(result_ev['y_test'].unique())]
                st.plotly_chart(P.plot_confusion_matrix(cm, labels),
                                width="stretch", key='eval_confusion_matrix')
                with st.expander('¿Cómo interpretar la matriz?'):
                    st.markdown('''
- **VP (verde)** — Verdaderos Positivos: detectados correctamente
- **VN (verde)** — Verdaderos Negativos: rechazados correctamente
- **FP (rojo)** — Falsos Positivos: falsas alarmas → costo en acciones innecesarias
- **FN (naranja)** — Falsos Negativos: casos reales perdidos → riesgo más alto para el negocio
                    ''')
            with col_roc:
                st.subheader('Curva ROC')
                st.plotly_chart(P.plot_roc_curve(
                    result_ev['roc']['fpr'],
                    result_ev['roc']['tpr'],
                    mets['auc_roc']), width="stretch", key='eval_roc_curve')

        # ── Regression evaluation ──
        elif 'r2' in mets and 'mape' in mets:
            st.subheader('Métricas en test set')
            c1,c2,c3,c4 = st.columns(4)
            c1.metric('R²', f'{mets["r2"]:.3f}',
                       delta='✓ aprobado' if mets['r2'] >= crit.get('r2',0) else '✗ insuficiente')
            c2.metric('RMSE', f'S/.{mets["rmse"]:,.0f}')
            c3.metric('MAE', f'S/.{mets["mae"]:,.0f}')
            c4.metric('MAPE', f'{mets["mape"]:.1f}%',
                       delta='✓ aprobado' if mets['mape'] <= crit.get('mape',100) else '✗ insuficiente',
                       delta_color='normal' if mets['mape'] <= crit.get('mape',100) else 'inverse')

            st.markdown('---')
            col_pv, col_res_dist = st.columns(2)
            with col_pv:
                st.subheader('Predicho vs Real')
                st.plotly_chart(P.plot_actual_vs_predicted(
                    result_ev['y_test'], result_ev['y_pred'],
                    cfg['target']), width="stretch", key='eval_actual_vs_pred')
            with col_res_dist:
                st.subheader('Distribución de residuos')
                residuals = np.array(result_ev['y_test']) - np.array(result_ev['y_pred'])
                fig_res = P.plot_histograms(
                    pd.DataFrame({'residuos': residuals}), ['residuos'])
                st.plotly_chart(fig_res, width="stretch", key='eval_residuals')

        # ── Elasticity evaluation ──
        elif 'elasticity' in mets:
            st.subheader('Métricas del modelo')
            c1,c2,c3 = st.columns(3)
            c1.metric('R² (log-log)', f'{mets["r2"]:.3f}',
                       delta='✓' if mets['r2'] >= crit.get('r2',0) else '✗')
            c2.metric('Elasticidad β', f'{mets["elasticity"]:.3f}')
            c3.metric('Interpretación',
                       'Elástico' if abs(mets['elasticity']) > 1 else 'Inelástico')

            st.markdown('---')
            st.subheader('Curva demanda–precio por categoría')
            st.plotly_chart(P.plot_elasticity_curve(
                result_ev['df_log'],
                result_ev['elasticity'],
                cat_col='categoria' if 'categoria' in result_ev['df_log'].columns else None),
                width="stretch", key='eval_elasticity_curve')

            if result_ev['elasticity_by_cat']:
                st.subheader('Elasticidad por categoría')
                for cat, val in sorted(result_ev['elasticity_by_cat'].items(),
                                        key=lambda x: x[1]):
                    tipo = '🔴 Elástico (sensible al precio)' if abs(val)>1 else '🟡 Inelástico'
                    st.metric(cat, f'β = {val:.3f}', delta=tipo, delta_color='off')

        # ── Basket evaluation ──
        elif 'rules' in result_ev:
            rules_df = result_ev.get('rules', pd.DataFrame())
            st.subheader('Reglas de asociación generadas')
            if rules_df.empty:
                st.warning('Sin reglas. Reduce los umbrales en la Fase 4.')
            else:
                c1,c2,c3,c4 = st.columns(4)
                c1.metric('Reglas totales', len(rules_df))
                c2.metric('Lift máximo', f'{rules_df["lift"].max():.2f}')
                c3.metric('Confianza máxima', f'{rules_df["confidence"].max():.0%}')
                c4.metric('Soporte máximo', f'{rules_df["support"].max():.1%}')

                st.dataframe(
                    rules_df[['antecedents','consequents','support',
                               'confidence','lift']].head(20).round(3),
                    width="stretch")

                fig_lift = P.plot_basket_lift(rules_df)
                if fig_lift:
                    st.plotly_chart(fig_lift, width="stretch", key='eval_basket_lift')

        # ── Uplift evaluation ──
        elif 'ate' in mets:
            st.subheader('Métricas de incrementalidad')
            c1, c2, c3 = st.columns(3)
            c1.metric('ATE (Average Treatment Effect)', f'{mets["ate"]:+.4f}',
                       delta='✓' if mets['ate'] >= crit.get('ate', 0) else '✗')
            c2.metric('Uplift top 10%', f'{mets["top10_lift"]:+.4f}',
                       delta='✓' if mets['top10_lift'] >= crit.get('top10_lift', 0) else '✗')
            c3.metric('Qini coefficient', f'{mets["qini"]:.4f}')

            st.markdown('---')
            col_dec, col_dist = st.columns(2)
            with col_dec:
                st.subheader('Uplift por decil')
                st.plotly_chart(P.plot_uplift_by_decile(result_ev['decile_summary']),
                                width="stretch", key='eval_uplift_decile')
            with col_dist:
                st.subheader('Distribución de scores')
                st.plotly_chart(P.plot_uplift_distribution(result_ev['df_scored']),
                                width="stretch", key='eval_uplift_dist')

            with st.expander('¿Cómo interpretar el uplift score?'):
                st.markdown('''
- **ATE > 0** — En promedio, la campaña genera conversiones incrementales
- **Top 10% (D10)** — Los clientes más "persuadibles": mayor impacto esperado de la campaña
- **Uplift ≈ 0** — Cliente "always buy": compraría igual sin campaña (no gastar en él)
- **Uplift < 0** — Cliente "sleeping dog": la campaña podría tener efecto negativo (evitar)
- **Qini > 0** — El modelo rankea mejor que asignación aleatoria
                ''')

        # ── NBO evaluation ──
        elif 'f1_macro' in mets:
            c1,c2,c3 = st.columns(3)
            c1.metric('Accuracy', f'{mets["accuracy"]:.1%}',
                       delta='✓' if mets['accuracy'] >= crit.get('accuracy',0) else '✗')
            c2.metric('F1 Macro', f'{mets["f1_macro"]:.3f}',
                       delta='✓' if mets['f1_macro'] >= crit.get('f1_macro',0) else '✗')
            c3.metric('Top-2 Accuracy', f'{mets["top2_accuracy"]:.1%}',
                       delta='✓' if mets['top2_accuracy'] >= crit.get('top2_accuracy',0) else '✗')

            st.markdown('---')
            st.subheader('Probabilidad media por producto (test set)')
            probs_mean = result_ev['y_prob'].mean(axis=0)
            st.plotly_chart(P.plot_nbo_probs(result_ev['classes'], probs_mean),
                            width="stretch", key='eval_nbo_probs')

        # ── Pass/Fail summary ──
        st.markdown('---')
        st.subheader('Validación vs criterios de negocio')
        all_pass = True
        for metric_name, threshold in crit.items():
            if metric_name in mets:
                higher = metric_name not in ['mape', 'rmse', 'mae']
                val = mets[metric_name]
                passed = val >= threshold if higher else val <= threshold
                if not passed:
                    all_pass = False
                icon = '✅' if passed else '❌'
                direction = '≥' if higher else '≤'
                st.markdown(
                    f'<div class="{"kpi-box" if passed else "warn-box"}">'
                    f'{icon} <b>{metric_name.upper()}</b> = {val:.3f} '
                    f'(umbral {direction} {threshold}) '
                    f'{"→ APROBADO" if passed else "→ NECESITA MEJORA"}'
                    f'</div>', unsafe_allow_html=True)

        if all_pass:
            st.success('✅ Modelo aprobado. Cumple todos los criterios de aceptación del negocio.')
        else:
            st.warning('⚠️ Modelo necesita ajuste. Prueba con más datos o ajusta los hiperparámetros.')

# ══════════════════════════════════════════════════════════════════
# FASE 6: DEPLOYMENT
# ══════════════════════════════════════════════════════════════════
with phases[5]:
    st.markdown('<div class="phase-desc">Plan de implementación, monitoreo y mantenimiento del modelo en producción.</div>', unsafe_allow_html=True)

    DEPLOY_INFO = {
        'Churn Prediction': {
            'infra': 'Cloud Run + BigQuery ML + Looker Studio',
            'freq': 'Diario (batch nocturno 2am)',
            'output': 'Score de riesgo (0–100%) → CRM → Cola de asesores',
            'roi': [('Clientes rescatados/mes','~120'), ('LTV protegido/mes','S/.300K'), ('ROI modelo','15x–30x')],
        },
        'Lead Scoring (Reg. Logística)': {
            'infra': 'Cloud Run API REST + CRM',
            'freq': 'Tiempo real (por lead nuevo)',
            'output': 'Score 0–100% + etiqueta ALTA/MEDIA/BAJA → CRM',
            'roi': [('Matrículas adicionales/ciclo','+38%'), ('Ahorro llamadas frías','-40%'), ('ROI modelo','12x–20x')],
        },
        'LTV · Regresión Lineal': {
            'infra': 'BigQuery ML + Looker Studio Dashboard',
            'freq': 'Semanal (recálculo portafolio)',
            'output': 'LTV estimado + segmento (Bajo/Regular/VIP) → DMP',
            'roi': [('Optimización CAC','-35%'), ('Revenue incremental','+22%'), ('ROI modelo','8x–15x')],
        },
        'Elasticidad de Precio': {
            'infra': 'Herramienta de pricing + GCP Functions',
            'freq': 'Mensual / por campaña de precio',
            'output': 'Elasticidad β por categoría → Motor de pricing',
            'roi': [('Revenue optimizado','+8–14%'), ('Promociones rentables','3 de 5 identificadas'), ('ROI modelo','5x–10x')],
        },
        'Market Basket Analysis': {
            'infra': 'API de recomendaciones + E-commerce',
            'freq': 'Diario (actualización de reglas)',
            'output': 'Top 3 recomendaciones por transacción → POS + Web',
            'roi': [('Ticket promedio','+S/.28'), ('Revenue bundles/mes','S/.140K'), ('ROI modelo','10x–20x')],
        },
        'Next Best Offer': {
            'infra': 'Cloud Run + Firestore + App asesores',
            'freq': 'Semanal (recálculo portafolio)',
            'output': 'Ranking de productos → Notificación push al asesor',
            'roi': [('Cross-sell','4.2% → 9.1%'), ('Revenue/producto adicional','S/.320'), ('ROI modelo','8x–18x')],
        },
        'Propensión a Compra (GBM)': {
            'infra': 'Cloud Run API + CRM + Call Center',
            'freq': 'Diario (recálculo nocturno)',
            'output': 'Score 0–100% + segmento ALTA/MEDIA/BAJA → Lista priorizada call center',
            'roi': [('Conversión campaña','+35% vs masiva'), ('Ahorro llamadas en frío','-50%'), ('ROI modelo','12x–25x')],
        },
        'Win / Loss de Oportunidades': {
            'infra': 'CRM Plugin (Salesforce/HubSpot) + Cloud Run',
            'freq': 'Tiempo real (por actualización de oportunidad)',
            'output': 'Win probability score → Alerta al gerente comercial',
            'roi': [('Win rate','28% → 40%+'), ('Ciclo de venta','-20% días'), ('ROI modelo','8x–15x')],
        },
        'Uplift · Incrementalidad': {
            'infra': 'Cloud Run + BigQuery (scored_customers) + CDP',
            'freq': 'Pre-campaña (cada ejecución de campaña)',
            'output': 'Uplift score + segmento PERSUADIBLE/ALWAYS-BUY/SLEEPING-DOG → Audiencia filtrada',
            'roi': [('Costo campaña','-40% con mismo revenue'), ('ROAS incremental','+2.5x'), ('ROI modelo','10x–20x')],
        },
        'Segmentación RFM (K-Means)': {
            'infra': 'BigQuery ML + Looker Studio (segmentos en tiempo real)',
            'freq': 'Semanal (recálculo con datos de transacciones)',
            'output': 'Segmento RFM por cliente → CRM → Reglas de automatización',
            'roi': [('Costo retención','-30%'), ('Reactivación dormidos','+20%'), ('ROI modelo','6x–12x')],
        },
        'Segmentación Categórica (K-Modes)': {
            'infra': 'Cloud Run + CDP (Customer Data Platform)',
            'freq': 'Mensual o por campaña',
            'output': 'Perfil K-Modes por cliente → Personalización de mensaje y canal',
            'roi': [('CTR campañas','+25%'), ('Costo por lead','-20%'), ('ROI modelo','5x–10x')],
        },
        'Clustering Jerárquico': {
            'infra': 'Python batch en Cloud Run + BigQuery Export',
            'freq': 'Trimestral (análisis estratégico)',
            'output': 'Segmentos jerárquicos → Informe ejecutivo + Dashboard estratégico',
            'roi': [('Precisión de segmentación','+35% vs RFM simple'), ('Ahorro presupuesto campañas','-25%'), ('ROI modelo','4x–8x')],
        },
    }

    dep = DEPLOY_INFO.get(model_name, {})

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.subheader('Arquitectura de implementación')
        st.markdown(f'**Infraestructura:** `{dep.get("infra","—")}`')
        st.markdown(f'**Frecuencia:** `{dep.get("freq","—")}`')
        st.markdown(f'**Output:** {dep.get("output","—")}')
        st.markdown('<br>', unsafe_allow_html=True)
        st.subheader('Plan de monitoreo')
        st.markdown('<div class="warn-box"><b>Data drift:</b> Comparar distribución de features semanalmente. Alerta si KL-divergencia > 0.2</div>', unsafe_allow_html=True)
        st.markdown('<div class="criteria-box"><b>Model drift:</b> Recalcular AUC/MAPE mensualmente en datos recientes. Re-entrenar si cae >10% del baseline.</div>', unsafe_allow_html=True)
        st.markdown('<div class="kpi-box"><b>Feedback loop:</b> Etiquetar resultados reales para ampliar dataset de entrenamiento en cada ciclo.</div>', unsafe_allow_html=True)

    with col_d2:
        st.subheader('ROI del proyecto ML')
        for label, val in dep.get('roi', []):
            st.metric(label, val)
        st.markdown('<br>', unsafe_allow_html=True)
        st.subheader('Próximos pasos')
        steps = [
            '**Validación histórica** — Contrastar predicciones con resultados reales de los últimos 90 días',
            '**Integración CRM** — Conectar el score del modelo al sistema operativo del equipo comercial',
            '**A/B test controlado** — Comparar grupo con modelo vs grupo sin modelo para medir impacto incremental',
            '**Pipeline de re-entrenamiento** — Automatizar cuando el performance caiga del umbral definido',
        ]
        for i, s in enumerate(steps, 1):
            st.markdown(f'{i}. {s}')

    st.markdown('---')
    st.subheader('Notebook completo en Google Colab')
    st.markdown(f'Abre el notebook para ver el código Python completo con el modelo real, visualizaciones adicionales y experimenta con tus propios datos.')
    st.link_button(f'↗ Abrir en Google Colab — {model_name}',
                    cfg['colab'], width="content")

    # Export model results
    result_exp = st.session_state.results.get(model_name)
    # Export para modelos no supervisados
    if result_exp and 'df_scored' in result_exp and cfg.get('tipo') == 'no_supervisado':
        st.markdown('---')
        st.subheader('Exportar segmentación')
        df_export = result_exp['df_scored'].copy()
        csv_bytes = df_export.to_csv(index=False).encode()
        st.download_button('⬇ Descargar clientes segmentados (CSV)',
                           data=csv_bytes,
                           file_name=f'{cfg["data_key"]}_segmentado.csv',
                           mime='text/csv')

    if result_exp and 'y_pred' in result_exp:
        st.markdown('---')
        st.subheader('Exportar predicciones')
        df_exp = get_df(model_name).copy()
        feats_exp = result_exp['features']
        preds_series = pd.Series(dtype=float)

        if cfg['key'] in ('churn', 'logit', 'propension', 'winloss'):
            X_all = df_exp[feats_exp].fillna(0)
            if cfg['key'] == 'logit' and 'scaler' in result_exp:
                X_all = result_exp['scaler'].transform(X_all)
            probs = result_exp['model'].predict_proba(X_all)[:, 1]
            df_exp['score_predicho'] = (probs * 100).round(1)
        elif cfg['key'] == 'uplift' and 'df_scored' in result_exp:
            df_exp = result_exp['df_scored'].copy()
            df_exp['segmento_uplift'] = pd.cut(
                df_exp['uplift_score'],
                bins=[-1, 0.05, 0.15, 1],
                labels=['No persuadible', 'Responde algo', 'Persuadible'])
        elif cfg['key'] == 'ltv':
            X_all = df_exp[feats_exp].fillna(0)
            preds = result_exp['model'].predict(X_all)
            df_exp['ltv_predicho'] = preds.round(0).astype(int)
            df_exp['segmento_predicho'] = pd.cut(
                df_exp['ltv_predicho'],
                bins=[0, 1500, 5000, 999999],
                labels=['Bajo', 'Regular', 'VIP'])
        elif cfg['key'] == 'nbo':
            X_all = df_exp[feats_exp].fillna(0)
            probs = result_exp['model'].predict_proba(X_all)
            top1_idx = probs.argmax(axis=1)
            df_exp['producto_recomendado'] = result_exp['le'].inverse_transform(top1_idx)
            df_exp['prob_recomendacion'] = probs.max(axis=1).round(3)

        csv_pred = df_exp.to_csv(index=False).encode()
        st.download_button(
            '⬇ Descargar predicciones completas (CSV)',
            data=csv_pred,
            file_name=f'{cfg["data_key"]}_predicciones.csv',
            mime='text/csv')
