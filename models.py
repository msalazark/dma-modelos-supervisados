"""
Módulo de modelos ML reales — CRISP-DM Lab
Cada función recibe un DataFrame y devuelve resultados completos.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, roc_curve,
                              confusion_matrix, mean_squared_error,
                              mean_absolute_error, r2_score)
import warnings
warnings.filterwarnings('ignore')

# ── CHURN ──────────────────────────────────────────────────────────
def run_churn(df):
    features = ['dias_inactivo', 'quejas', 'logins_30d',
                'saldo_prom_k', 'productos']
    target = 'churn'
    feats_available = [f for f in features if f in df.columns]
    if target not in df.columns:
        return None
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
        'churn_rate': y.mean(),
        'train_size': len(X_train),
        'test_size': len(X_test),
    }

# ── LOGISTIC REGRESSION / LEAD SCORING ────────────────────────────
def run_logit(df):
    features = ['dias_contacto', 'interacciones_web',
                'clicks_precio', 'tiempo_pag_min']
    target = 'convirtio'
    feats_available = [f for f in features if f in df.columns]
    if target not in df.columns:
        return None
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
        'conv_rate': y.mean(),
        'train_size': len(X_train),
        'test_size': len(X_test),
    }

# ── LTV / LINEAR REGRESSION ────────────────────────────────────────
def run_ltv(df):
    features = ['compras_6m', 'ticket_prom', 'meses_cliente',
                'categorias', 'canal_digital']
    target = 'ltv_anual'
    feats_available = [f for f in features if f in df.columns]
    if target not in df.columns:
        return None
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

# ── NEXT BEST OFFER ────────────────────────────────────────────────
def run_nbo(df):
    features = ['tiene_tarjeta', 'tiene_cuenta', 'tiene_seguro',
                'tiene_prestamo', 'antiguedad_m', 'saldo_k']
    target = 'producto_adquirido'
    feats_available = [f for f in features if f in df.columns]
    if target not in df.columns:
        return None

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
        'train_size': len(X_train),
        'test_size': len(X_test),
    }
