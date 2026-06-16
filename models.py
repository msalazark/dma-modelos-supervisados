"""
Módulo de modelos ML reales — CRISP-DM Lab
Cada función recibe un DataFrame y devuelve resultados completos.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingRegressor,
                              GradientBoostingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, roc_curve,
                              confusion_matrix, mean_squared_error,
                              mean_absolute_error, r2_score)
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

# ── CHURN ──────────────────────────────────────────────────────────
def run_churn(df, features: list[str] | None = None, target: str | None = None):
    DEFAULT_FEATURES = ['dias_inactivo', 'quejas', 'logins_30d',
                'saldo_prom_k', 'productos']
    target = target or 'churn'
    feats = features if features else DEFAULT_FEATURES
    feats_available = [f for f in feats if f in df.columns]
    if target not in df.columns:
        return None
    if not feats_available:
        return {'error': 'Selecciona al menos una variable disponible en el dataset.'}
    X = df[feats_available].fillna(0)
    y = df[target].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    model = RandomForestClassifier(n_estimators=200, max_depth=6,
                                   random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    return {
        'model': model,
        'features': feats_available,
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'y_pred': y_pred, 'y_prob': y_prob,
        'metrics': {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_test, y_prob),
        },
        'confusion_matrix': cm,
        'roc': {'fpr': fpr, 'tpr': tpr},
        'feature_importance': dict(zip(feats_available,
                                        model.feature_importances_)),
        'target': target,
        'churn_rate': y.mean(),
        'train_size': len(X_train),
        'test_size': len(X_test),
    }

# ── LOGISTIC REGRESSION / LEAD SCORING ────────────────────────────
def run_logit(df, features: list[str] | None = None, target: str | None = None):
    DEFAULT_FEATURES = ['dias_contacto', 'interacciones_web',
                'clicks_precio', 'tiempo_pag_min']
    target = target or 'convirtio'
    feats = features if features else DEFAULT_FEATURES
    feats_available = [f for f in feats if f in df.columns]
    if target not in df.columns:
        return None
    if not feats_available:
        return {'error': 'Selecciona al menos una variable disponible en el dataset.'}
    X = df[feats_available].fillna(0)
    y = df[target].astype(int)

    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    coef_df = pd.DataFrame({
        'feature': feats_available,
        'coef': model.coef_[0],
        'odds_ratio': np.exp(model.coef_[0])
    }).sort_values('coef', ascending=False)

    return {
        'model': model, 'scaler': scaler,
        'features': feats_available,
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'y_pred': y_pred, 'y_prob': y_prob,
        'metrics': {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_test, y_prob),
        },
        'confusion_matrix': cm,
        'roc': {'fpr': fpr, 'tpr': tpr},
        'coefficients': coef_df,
        'target': target,
        'conv_rate': y.mean(),
        'train_size': len(X_train),
        'test_size': len(X_test),
    }

# ── LTV / LINEAR REGRESSION ────────────────────────────────────────
def run_ltv(df, features: list[str] | None = None, target: str | None = None):
    DEFAULT_FEATURES = ['compras_6m', 'ticket_prom', 'meses_cliente',
                'categorias', 'canal_digital']
    target = target or 'ltv_anual'
    feats = features if features else DEFAULT_FEATURES
    feats_available = [f for f in feats if f in df.columns]
    if target not in df.columns:
        return None
    if not feats_available:
        return {'error': 'Selecciona al menos una variable disponible en el dataset.'}
    X = df[feats_available].fillna(0)
    y = df[target].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-9))) * 100
    r2 = r2_score(y_test, y_pred)

    coef_df = pd.DataFrame({
        'feature': feats_available,
        'coef': model.coef_
    }).sort_values('coef', ascending=False)

    return {
        'model': model,
        'features': feats_available,
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'y_pred': y_pred,
        'metrics': {'r2': r2, 'rmse': rmse, 'mae': mae, 'mape': mape},
        'coefficients': coef_df,
        'target': target,
        'intercept': model.intercept_,
        'train_size': len(X_train),
        'test_size': len(X_test),
    }

# ── PRICE ELASTICITY ───────────────────────────────────────────────
def run_elasticity(df):
    if 'precio' not in df.columns or 'unidades' not in df.columns:
        return None
    from sklearn.linear_model import LinearRegression
    df2 = df.copy()
    df2 = df2[(df2['precio'] > 0) & (df2['unidades'] > 0)].copy()

    # Elasticity by category
    elast_by_cat = {}
    if 'categoria' in df2.columns:
        for cat in df2['categoria'].unique():
            sub = df2[df2['categoria'] == cat].copy()
            if len(sub) > 8:
                sub['log_p'] = np.log(sub['precio'])
                sub['log_q'] = np.log(sub['unidades'])
                m = LinearRegression()
                m.fit(sub[['log_p']], sub['log_q'])
                elast_by_cat[cat] = round(m.coef_[0], 3)

    # Global: normalize price within category to remove category-level noise
    if 'categoria' in df2.columns:
        df2['precio_norm'] = df2.groupby('categoria')['precio'].transform(
            lambda x: x / x.mean())
        df2['log_precio'] = np.log(df2['precio_norm'])
    else:
        df2['log_precio'] = np.log(df2['precio'])
    df2['log_unidades'] = np.log(df2['unidades'])

    features = ['log_precio']
    if 'promocion' in df2.columns:
        features.append('promocion')

    X = df2[features].fillna(0)
    y = df2['log_unidades']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    elasticity = model.coef_[0]
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return {
        'model': model, 'features': features,
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'y_pred': y_pred,
        'metrics': {'r2': r2, 'rmse': rmse, 'elasticity': elasticity},
        'elasticity': elasticity,
        'elasticity_by_cat': elast_by_cat,
        'df_log': df2,
        'train_size': len(X_train),
        'test_size': len(X_test),
    }

# ── MARKET BASKET ──────────────────────────────────────────────────
def run_basket(df, min_support=0.05, min_confidence=0.3, min_lift=1.0):
    try:
        from mlxtend.frequent_patterns import apriori, association_rules
        from mlxtend.preprocessing import TransactionEncoder

        if 'ticket_id' not in df.columns or 'producto' not in df.columns:
            return None

        transactions = df.groupby('ticket_id')['producto'].apply(list).tolist()
        te = TransactionEncoder()
        te_array = te.fit_transform(transactions)
        df_te = pd.DataFrame(te_array, columns=te.columns_)

        frequent = apriori(df_te, min_support=min_support,
                           use_colnames=True, verbose=0)
        if frequent.empty:
            return {'rules': pd.DataFrame(), 'frequent': frequent,
                    'n_transactions': len(transactions),
                    'n_items': len(te.columns_)}

        rules = association_rules(frequent, metric='confidence',
                                  min_threshold=min_confidence)
        rules = rules[rules['lift'] >= min_lift].sort_values(
            'lift', ascending=False)
        rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

        return {
            'rules': rules,
            'frequent': frequent,
            'n_transactions': len(transactions),
            'n_items': len(te.columns_),
            'item_support': frequent[frequent['itemsets'].apply(len) == 1].copy(),
        }
    except Exception as e:
        return {'error': str(e), 'rules': pd.DataFrame()}

# ── PROPENSIÓN A COMPRA ────────────────────────────────────────────
def run_propension(df, features: list[str] | None = None, target: str | None = None):
    DEFAULT_FEATURES = ['recencia_dias', 'n_compras_12m', 'engagement_score',
                'canal_digital', 'edad', 'n_productos']
    target = target or 'compro'
    feats = features if features else DEFAULT_FEATURES
    feats_available = [f for f in feats if f in df.columns]
    if target not in df.columns:
        return None
    if not feats_available:
        return {'error': 'Selecciona al menos una variable disponible en el dataset.'}
    X = df[feats_available].fillna(0)
    y = df[target].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    model = GradientBoostingClassifier(n_estimators=150, max_depth=4,
                                        learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    return {
        'model': model,
        'features': feats_available,
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'y_pred': y_pred, 'y_prob': y_prob,
        'metrics': {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_test, y_prob),
        },
        'confusion_matrix': cm,
        'roc': {'fpr': fpr, 'tpr': tpr},
        'feature_importance': dict(zip(feats_available, model.feature_importances_)),
        'target': target,
        'conv_rate': y.mean(),
        'train_size': len(X_train),
        'test_size': len(X_test),
    }

# ── WIN / LOSS DE OPORTUNIDADES ────────────────────────────────────
def run_winloss(df, features: list[str] | None = None, target: str | None = None):
    DEFAULT_FEATURES = ['monto_oportunidad_k', 'dias_ciclo', 'n_reuniones',
                'n_competidores', 'decision_makers', 'propuesta_personalizada']
    target = target or 'ganado'
    feats = features if features else DEFAULT_FEATURES
    feats_available = [f for f in feats if f in df.columns]
    if target not in df.columns:
        return None
    if not feats_available:
        return {'error': 'Selecciona al menos una variable disponible en el dataset.'}
    X = df[feats_available].fillna(0)
    y = df[target].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    model = RandomForestClassifier(n_estimators=200, max_depth=6,
                                   random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    return {
        'model': model,
        'features': feats_available,
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'y_pred': y_pred, 'y_prob': y_prob,
        'metrics': {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_test, y_prob),
        },
        'confusion_matrix': cm,
        'roc': {'fpr': fpr, 'tpr': tpr},
        'feature_importance': dict(zip(feats_available, model.feature_importances_)),
        'target': target,
        'win_rate': y.mean(),
        'train_size': len(X_train),
        'test_size': len(X_test),
    }

# ── UPLIFT / INCREMENTALIDAD ────────────────────────────────────────
def run_uplift(df, features: list[str] | None = None, target: str | None = None):
    DEFAULT_FEATURES = ['edad', 'recencia_dias', 'n_compras_prev', 'ticket_prom_k', 'canal_digital']
    treatment_col = 'tratamiento'
    target = target or 'convirtio'
    feats = features if features else DEFAULT_FEATURES
    feats_available = [f for f in feats if f in df.columns]
    if target not in df.columns or treatment_col not in df.columns:
        return None
    if not feats_available:
        return {'error': 'Selecciona al menos una variable disponible en el dataset.'}

    treated = df[df[treatment_col] == 1].copy()
    control = df[df[treatment_col] == 0].copy()

    scaler = StandardScaler()
    X_all = scaler.fit_transform(df[feats_available].fillna(0))
    X_t = scaler.transform(treated[feats_available].fillna(0))
    X_c = scaler.transform(control[feats_available].fillna(0))

    model_t = LogisticRegression(C=1.0, max_iter=300, random_state=42)
    model_c = LogisticRegression(C=1.0, max_iter=300, random_state=42)
    model_t.fit(X_t, treated[target].astype(int))
    model_c.fit(X_c, control[target].astype(int))

    p_treat = model_t.predict_proba(X_all)[:, 1]
    p_control = model_c.predict_proba(X_all)[:, 1]
    uplift_scores = p_treat - p_control

    df_scored = df[feats_available + [treatment_col, target]].copy()
    df_scored['p_treated'] = p_treat
    df_scored['p_control'] = p_control
    df_scored['uplift_score'] = uplift_scores
    df_scored['decil'] = pd.qcut(
        uplift_scores, 10,
        labels=[f'D{i}' for i in range(1, 11)],
        duplicates='drop'
    )

    ate = float(uplift_scores.mean())
    top10_n = max(1, len(uplift_scores) // 10)
    top10_lift = float(uplift_scores[np.argsort(uplift_scores)[::-1][:top10_n]].mean())

    # Qini coefficient
    rank = np.argsort(uplift_scores)[::-1]
    cum_conv = np.cumsum(df_scored[target].values[rank])
    random_line = np.linspace(0, cum_conv[-1], len(cum_conv))
    qini = float((cum_conv - random_line).sum() / len(cum_conv))

    decile_summary = (df_scored.groupby('decil', observed=True)['uplift_score']
                      .mean().reset_index()
                      .rename(columns={'uplift_score': 'uplift_medio'}))

    return {
        'model_t': model_t,
        'model_c': model_c,
        'scaler': scaler,
        'features': feats_available,
        'df_scored': df_scored,
        'decile_summary': decile_summary,
        'metrics': {
            'ate': round(ate, 4),
            'top10_lift': round(top10_lift, 4),
            'qini': round(qini, 4),
        },
        'target': target,
        'uplift_scores': uplift_scores,
        'train_size': len(treated),
        'test_size': len(control),
    }

# ── NEXT BEST OFFER ────────────────────────────────────────────────
def run_nbo(df, features: list[str] | None = None, target: str | None = None):
    DEFAULT_FEATURES = ['tiene_tarjeta', 'tiene_cuenta', 'tiene_seguro',
                'tiene_prestamo', 'antiguedad_m', 'saldo_k']
    target = target or 'producto_adquirido'
    feats = features if features else DEFAULT_FEATURES
    feats_available = [f for f in feats if f in df.columns]
    if target not in df.columns:
        return None
    if not feats_available:
        return {'error': 'Selecciona al menos una variable disponible en el dataset.'}

    le = LabelEncoder()
    X = df[feats_available].fillna(0)
    y = le.fit_transform(df[target])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=150, max_depth=8,
                                   random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    # Top-2 accuracy
    top2_correct = sum(y_test[i] in np.argsort(y_prob[i])[-2:]
                       for i in range(len(y_test)))
    top2_acc = top2_correct / len(y_test)

    return {
        'model': model, 'le': le,
        'features': feats_available,
        'classes': le.classes_,
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'y_pred': y_pred, 'y_prob': y_prob,
        'metrics': {'accuracy': acc, 'f1_macro': f1, 'top2_accuracy': top2_acc},
        'feature_importance': dict(zip(feats_available,
                                        model.feature_importances_)),
        'target': target,
        'train_size': len(X_train),
        'test_size': len(X_test),
    }

# ── K-MEANS (RFM) ──────────────────────────────────────────────────
def run_kmeans(df: pd.DataFrame, n_clusters: int = 4, features: list[str] | None = None) -> dict:
    DEFAULT_FEATURES = ['recencia_dias', 'frecuencia_compras', 'monto_total_k']
    feats_in = features if features else DEFAULT_FEATURES
    feats = [f for f in feats_in if f in df.columns]
    if len(feats) < 2:
        return {'error': 'Selecciona al menos 2 variables disponibles en el dataset.'}

    X_raw = df[feats].fillna(0).values
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # Curva de codo: entrenar k=2..9
    elbow = []
    for k in range(2, 10):
        km_tmp = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        km_tmp.fit(X)
        elbow.append({'k': k, 'inertia': km_tmp.inertia_})

    km = KMeans(n_clusters=n_clusters, init='k-means++', n_init=20, random_state=42)
    labels = km.fit_predict(X)

    sil  = silhouette_score(X, labels)
    ch   = calinski_harabasz_score(X, labels)
    db   = davies_bouldin_score(X, labels)

    df_scored = df.copy()
    df_scored['cluster'] = labels
    df_scored['cluster_label'] = [f'Cluster {l+1}' for l in labels]

    centroids_scaled = km.cluster_centers_
    centroids_real   = scaler.inverse_transform(centroids_scaled)
    profiles = pd.DataFrame(centroids_real, columns=feats)
    profiles.insert(0, 'cluster', [f'Cluster {i+1}' for i in range(n_clusters)])
    profiles['n'] = pd.Series(labels).value_counts().sort_index().values

    return {
        'model': km, 'scaler': scaler, 'features': feats,
        'labels': labels, 'df_scored': df_scored,
        'cluster_profiles': profiles,
        'elbow_data': pd.DataFrame(elbow),
        'metrics': {
            'silhouette': round(sil, 4),
            'inertia': round(km.inertia_, 2),
            'calinski_harabasz': round(ch, 2),
            'davies_bouldin': round(db, 4),
            'n_clusters': n_clusters,
        },
        'train_size': len(df), 'test_size': 0,
    }

# ── K-MODES ────────────────────────────────────────────────────────
def run_kmodes(df: pd.DataFrame, n_clusters: int = 3, features: list[str] | None = None) -> dict:
    try:
        from kmodes.kmodes import KModes
    except ImportError:
        return {'error': 'Instala el paquete: pip install kmodes'}

    DEFAULT_FEATURES = ['canal_adquisicion', 'categoria_preferida', 'frecuencia_visita',
                'region', 'edad_grupo', 'dispositivo']
    feats_in = features if features else DEFAULT_FEATURES
    feats = [f for f in feats_in if f in df.columns]
    if len(feats) < 2:
        return {'error': 'Selecciona al menos 2 variables disponibles en el dataset.'}

    X = df[feats].fillna('Desconocido').values

    # Curva de costo
    elbow = []
    for k in range(2, 7):
        km_tmp = KModes(n_clusters=k, init='Huang', n_init=5, random_state=42)
        km_tmp.fit(X)
        elbow.append({'k': k, 'cost': km_tmp.cost_})

    km = KModes(n_clusters=n_clusters, init='Huang', n_init=10, random_state=42)
    labels = km.fit_predict(X)

    df_scored = df.copy()
    df_scored['cluster'] = labels
    df_scored['cluster_label'] = [f'Perfil {l+1}' for l in labels]

    modes_df = pd.DataFrame(km.cluster_centroids_, columns=feats)
    modes_df.insert(0, 'cluster', [f'Perfil {i+1}' for i in range(n_clusters)])
    sizes = pd.Series(labels).value_counts().sort_index()
    modes_df['n'] = sizes.values

    # Distribución de features por cluster
    profiles_detail = {}
    for feat in feats:
        profiles_detail[feat] = df_scored.groupby('cluster_label')[feat].apply(
            lambda x: x.value_counts(normalize=True).head(3).to_dict()
        ).to_dict()

    return {
        'model': km, 'features': feats,
        'labels': labels, 'df_scored': df_scored,
        'cluster_modes': modes_df,
        'profiles_detail': profiles_detail,
        'elbow_data': pd.DataFrame(elbow),
        'metrics': {
            'cost': round(km.cost_, 2),
            'n_clusters': n_clusters,
        },
        'train_size': len(df), 'test_size': 0,
    }

# ── FORECAST DE DEMANDA (PROPHET) ──────────────────────────────────
def run_prophet(df: pd.DataFrame, test_days: int = 30,
                changepoint_prior_scale: float = 0.05,
                seasonality_mode: str = 'additive',
                yearly_seasonality: bool = True,
                weekly_seasonality: bool = True,
                sim_days: int = 90) -> dict:
    try:
        from prophet import Prophet
        import logging
        logging.getLogger('prophet').setLevel(logging.WARNING)
        logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
    except ImportError:
        return {'error': 'Instala el paquete: pip install prophet'}

    date_col = 'fecha' if 'fecha' in df.columns else df.columns[0]
    value_col = 'ventas' if 'ventas' in df.columns else df.columns[1]

    data = df[[date_col, value_col]].dropna().copy()
    data.columns = ['ds', 'y']
    data['ds'] = pd.to_datetime(data['ds'])
    data = data.sort_values('ds').reset_index(drop=True)

    n_test = max(7, min(test_days, len(data) // 4))
    train = data.iloc[:-n_test]
    test = data.iloc[-n_test:]

    model = Prophet(
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_mode=seasonality_mode,
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
    )
    model.fit(train)

    future = model.make_future_dataframe(periods=n_test + sim_days)
    forecast = model.predict(future)

    fc_test = forecast.set_index('ds').loc[test['ds']]
    y_true = test['y'].values
    y_pred = fc_test['yhat'].values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100)

    return {
        'model': model,
        'data': data,
        'train': train,
        'test': test,
        'forecast': forecast,
        'metrics': {
            'mae': round(float(mae), 2),
            'rmse': round(float(rmse), 2),
            'mape': round(mape, 2),
            'horizon': n_test,
        },
        'train_size': len(train),
        'test_size': len(test),
    }

# ── CLUSTERING JERÁRQUICO ──────────────────────────────────────────
def run_hierarchical(df: pd.DataFrame, n_clusters: int = 4,
                     linkage: str = 'ward', features: list[str] | None = None) -> dict:
    from scipy.cluster.hierarchy import linkage as sp_linkage, cophenet
    from scipy.spatial.distance import pdist

    DEFAULT_FEATURES = ['recencia_dias', 'frecuencia_compras', 'ticket_prom_k',
                'n_categorias', 'meses_cliente', 'canal_digital']
    feats_in = features if features else DEFAULT_FEATURES
    feats = [f for f in feats_in if f in df.columns]
    if len(feats) < 2:
        return {'error': 'Selecciona al menos 2 variables disponibles en el dataset.'}

    X_raw = df[feats].fillna(0).values
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # Matriz de enlace para dendrograma
    Z = sp_linkage(X, method=linkage)
    c, _ = cophenet(Z, pdist(X))

    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(X)

    sil = silhouette_score(X, labels)
    ch  = calinski_harabasz_score(X, labels)

    df_scored = df.copy()
    df_scored['cluster'] = labels
    df_scored['cluster_label'] = [f'Segmento {l+1}' for l in labels]

    profiles = df_scored.groupby('cluster')[feats].mean().round(2)
    profiles.index = [f'Segmento {i+1}' for i in profiles.index]
    profiles['n'] = pd.Series(labels).value_counts().sort_index().values

    return {
        'model': model, 'scaler': scaler, 'features': feats,
        'labels': labels, 'df_scored': df_scored,
        'cluster_profiles': profiles.reset_index().rename(columns={'index': 'cluster'}),
        'linkage_matrix': Z,
        'metrics': {
            'silhouette': round(sil, 4),
            'cophenetic': round(float(c), 4),
            'calinski_harabasz': round(ch, 2),
            'n_clusters': n_clusters,
        },
        'train_size': len(df), 'test_size': 0,
    }
