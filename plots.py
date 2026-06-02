"""Visualizaciones reutilizables para todas las fases CRISP-DM"""
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

PALETTE = {
    'red':    '#c0392b',
    'blue':   '#1a4c8c',
    'teal':   '#0d9488',
    'orange': '#d97706',
    'green':  '#16a34a',
    'purple': '#7c3aed',
    'gray':   '#6b7280',
}
SEQ = list(PALETTE.values())

def plot_target_dist(series, target_name):
    vc = series.value_counts().reset_index()
    vc.columns = [target_name, 'count']
    vc['pct'] = (vc['count'] / vc['count'].sum() * 100).round(1)
    vc['label'] = vc[target_name].astype(str) + ' (' + vc['pct'].astype(str) + '%)'
    fig = px.bar(vc, x=target_name, y='count',
                 color=target_name,
                 color_discrete_sequence=[PALETTE['red'], PALETTE['teal']],
                 text='label',
                 title=f'Distribución: {target_name}')
    fig.update_traces(textposition='outside')
    fig.update_layout(showlegend=False, height=320,
                      margin=dict(t=40,b=20,l=20,r=20),
                      plot_bgcolor='white', paper_bgcolor='white',
                      font_family='DM Sans')
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor='#f0f0f0')
    return fig

def plot_histograms(df, cols, ncols=3):
    nrows = (len(cols) + ncols - 1) // ncols
    fig = make_subplots(rows=nrows, cols=ncols,
                        subplot_titles=cols)
    for idx, col in enumerate(cols):
        r = idx // ncols + 1
        c = idx % ncols + 1
        data = df[col].dropna()
        fig.add_trace(go.Histogram(x=data, name=col, nbinsx=20,
                                   marker_color=SEQ[idx % len(SEQ)],
                                   opacity=0.8), row=r, col=c)
    fig.update_layout(height=220*nrows, showlegend=False,
                      margin=dict(t=40,b=20,l=20,r=20),
                      plot_bgcolor='white', paper_bgcolor='white',
                      font_family='DM Sans')
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor='#f0f0f0')
    return fig

def plot_correlation_heatmap(df, cols):
    sub = df[cols].dropna()
    corr = sub.corr().round(2)
    fig = px.imshow(corr, text_auto=True, aspect='auto',
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1,
                    title='Matriz de correlaciones')
    fig.update_layout(height=400, margin=dict(t=40,b=20,l=20,r=20),
                      paper_bgcolor='white', font_family='DM Sans')
    return fig

def plot_correlation_bar(df, features, target):
    corrs = [df[f].corr(df[target]) for f in features
             if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
    feats = [f for f in features
             if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
    colors = [PALETTE['teal'] if c > 0 else PALETTE['red'] for c in corrs]
    fig = go.Figure(go.Bar(x=corrs, y=feats, orientation='h',
                           marker_color=colors, text=[f'{c:.2f}' for c in corrs],
                           textposition='outside'))
    fig.update_layout(title=f'Correlación con "{target}"',
                      height=280, margin=dict(t=40,b=20,l=20,r=20),
                      plot_bgcolor='white', paper_bgcolor='white',
                      font_family='DM Sans',
                      xaxis=dict(range=[-1, 1], gridcolor='#f0f0f0'),
                      yaxis=dict(showgrid=False))
    return fig

def plot_boxplots_by_target(df, features, target):
    figs = []
    for feat in features[:4]:
        if feat in df.columns and pd.api.types.is_numeric_dtype(df[feat]):
            fig = px.box(df, x=target, y=feat,
                         color=target,
                         color_discrete_sequence=[PALETTE['teal'], PALETTE['red']],
                         title=f'{feat} por {target}',
                         points='outliers')
            fig.update_layout(height=280, showlegend=False,
                              margin=dict(t=40,b=20,l=20,r=20),
                              plot_bgcolor='white', paper_bgcolor='white',
                              font_family='DM Sans')
            figs.append(fig)
    return figs

def plot_confusion_matrix(cm, labels=None):
    if labels is None:
        labels = [str(i) for i in range(cm.shape[0])]
    fig = px.imshow(cm, text_auto=True, aspect='equal',
                    x=[f'Pred: {l}' for l in labels],
                    y=[f'Real: {l}' for l in labels],
                    color_continuous_scale=[[0,'#f0fdf4'],[0.5,'#86efac'],[1,'#16a34a']],
                    title='Matriz de confusión')
    fig.update_layout(height=320, margin=dict(t=40,b=20,l=20,r=20),
                      paper_bgcolor='white', font_family='DM Sans')
    return fig

def plot_roc_curve(fpr, tpr, auc_score):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                              name=f'ROC (AUC={auc_score:.3f})',
                              line=dict(color=PALETTE['blue'], width=2.5)))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                              name='Random', line=dict(color='#ccc',
                              width=1, dash='dash')))
    fig.update_layout(title='Curva ROC',
                      xaxis_title='False Positive Rate',
                      yaxis_title='True Positive Rate (Recall)',
                      height=340, margin=dict(t=40,b=20,l=20,r=20),
                      plot_bgcolor='white', paper_bgcolor='white',
                      font_family='DM Sans',
                      legend=dict(x=0.6, y=0.1))
    fig.update_xaxes(gridcolor='#f0f0f0')
    fig.update_yaxes(gridcolor='#f0f0f0')
    return fig

def plot_feature_importance(importance_dict, title='Importancia de variables'):
    df = pd.DataFrame(list(importance_dict.items()),
                      columns=['feature', 'importance'])
    df = df.sort_values('importance', ascending=True)
    colors = [PALETTE['red'] if i == len(df)-1
              else PALETTE['blue'] if i >= len(df)-3
              else PALETTE['gray'] for i in range(len(df))]
    fig = go.Figure(go.Bar(x=df['importance'], y=df['feature'],
                           orientation='h', marker_color=colors,
                           text=[f'{v:.3f}' for v in df['importance']],
                           textposition='outside'))
    fig.update_layout(title=title, height=max(280, len(df)*45),
                      margin=dict(t=40,b=20,l=20,r=120),
                      plot_bgcolor='white', paper_bgcolor='white',
                      font_family='DM Sans',
                      xaxis=dict(gridcolor='#f0f0f0'),
                      yaxis=dict(showgrid=False))
    return fig

def plot_coeff_bar(coef_df, title='Coeficientes del modelo'):
    df = coef_df.sort_values('coef', ascending=True)
    colors = [PALETTE['teal'] if v > 0 else PALETTE['red']
              for v in df['coef']]
    fig = go.Figure(go.Bar(x=df['coef'], y=df['feature'],
                           orientation='h', marker_color=colors,
                           text=[f'{v:.3f}' for v in df['coef']],
                           textposition='outside'))
    fig.update_layout(title=title, height=max(280, len(df)*50),
                      margin=dict(t=40,b=20,l=20,r=80),
                      plot_bgcolor='white', paper_bgcolor='white',
                      font_family='DM Sans',
                      xaxis=dict(gridcolor='#f0f0f0', zeroline=True,
                                  zerolinecolor='#aaa'),
                      yaxis=dict(showgrid=False))
    return fig

def plot_actual_vs_predicted(y_test, y_pred, target_name='target'):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(y_test), y=list(y_pred),
                              mode='markers',
                              marker=dict(color=PALETTE['blue'],
                                         opacity=0.5, size=6),
                              name='Predicción'))
    mn = min(min(y_test), min(y_pred))
    mx = max(max(y_test), max(y_pred))
    fig.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode='lines',
                              line=dict(color=PALETTE['red'], width=1,
                                       dash='dash'),
                              name='Ideal (y=x)'))
    fig.update_layout(title='Predicho vs Real (test set)',
                      xaxis_title='Valor real',
                      yaxis_title='Valor predicho',
                      height=340, margin=dict(t=40,b=20,l=20,r=20),
                      plot_bgcolor='white', paper_bgcolor='white',
                      font_family='DM Sans')
    fig.update_xaxes(gridcolor='#f0f0f0')
    fig.update_yaxes(gridcolor='#f0f0f0')
    return fig

def plot_elasticity_curve(df_log, elasticity, cat_col=None):
    fig = go.Figure()
    if cat_col and cat_col in df_log.columns:
        for cat, grp in df_log.groupby(cat_col):
            fig.add_trace(go.Scatter(
                x=grp['precio'], y=grp['unidades'],
                mode='markers', name=cat, opacity=0.6,
                marker=dict(size=6)))
    else:
        fig.add_trace(go.Scatter(
            x=df_log['precio'], y=df_log['unidades'],
            mode='markers', marker=dict(color=PALETTE['blue'],
                                        opacity=0.5, size=6),
            name='Observaciones'))
    fig.update_layout(title=f'Demanda vs Precio (elasticidad global: {elasticity:.2f})',
                      xaxis_title='Precio (S/.)',
                      yaxis_title='Unidades vendidas',
                      height=340, margin=dict(t=40,b=20,l=20,r=20),
                      plot_bgcolor='white', paper_bgcolor='white',
                      font_family='DM Sans')
    fig.update_xaxes(gridcolor='#f0f0f0')
    fig.update_yaxes(gridcolor='#f0f0f0')
    return fig

def plot_basket_lift(rules_df, top_n=10):
    if rules_df.empty:
        return None
    top = rules_df.head(top_n).copy()
    top['regla'] = top['antecedents'] + ' → ' + top['consequents']
    fig = go.Figure(go.Bar(
        x=top['lift'], y=top['regla'], orientation='h',
        marker_color=PALETTE['blue'],
        text=[f'{v:.2f}' for v in top['lift']],
        textposition='outside'))
    fig.update_layout(title='Top reglas por Lift',
                      height=max(300, len(top)*45),
                      margin=dict(t=40,b=20,l=20,r=60),
                      plot_bgcolor='white', paper_bgcolor='white',
                      font_family='DM Sans',
                      xaxis=dict(gridcolor='#f0f0f0'),
                      yaxis=dict(showgrid=False))
    return fig

def plot_nbo_probs(classes, probs_mean):
    df = pd.DataFrame({'producto': classes, 'prob': probs_mean})
    df = df.sort_values('prob', ascending=True)
    colors = [PALETTE['red'] if i == len(df)-1
              else PALETTE['blue'] if i >= len(df)-3
              else PALETTE['gray'] for i in range(len(df))]
    fig = go.Figure(go.Bar(
        x=df['prob'], y=df['producto'], orientation='h',
        marker_color=colors,
        text=[f'{v:.1%}' for v in df['prob']],
        textposition='outside'))
    fig.update_layout(title='Probabilidad media por producto (test set)',
                      height=320, margin=dict(t=40,b=20,l=20,r=80),
                      plot_bgcolor='white', paper_bgcolor='white',
                      font_family='DM Sans',
                      xaxis=dict(gridcolor='#f0f0f0',
                                  tickformat='.0%'),
                      yaxis=dict(showgrid=False))
    return fig
