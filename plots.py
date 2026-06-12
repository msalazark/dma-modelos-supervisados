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

def plot_elbow_curve(elbow_df, metric='inertia', label='Inercia'):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=elbow_df['k'], y=elbow_df[metric],
        mode='lines+markers',
        line=dict(color=PALETTE['blue'], width=2.5),
        marker=dict(size=8, color=PALETTE['blue'])
    ))
    fig.update_layout(
        title=f'Curva de codo — {label} vs número de clusters',
        xaxis_title='Número de clusters (k)',
        yaxis_title=label,
        height=320, margin=dict(t=40,b=20,l=20,r=20),
        plot_bgcolor='white', paper_bgcolor='white',
        font_family='DM Sans',
        xaxis=dict(dtick=1, gridcolor='#f0f0f0'),
        yaxis=dict(gridcolor='#f0f0f0')
    )
    return fig

def plot_cluster_profiles(profiles_df, feature_cols):
    df = profiles_df.copy()
    cluster_col = 'cluster'
    colors = [PALETTE['blue'], PALETTE['teal'], PALETTE['red'],
              PALETTE['orange'], PALETTE['purple'], PALETTE['green']]
    fig = go.Figure()
    for i, row in df.iterrows():
        vals = [row[c] for c in feature_cols]
        fig.add_trace(go.Bar(
            name=str(row[cluster_col]),
            x=feature_cols, y=vals,
            marker_color=colors[i % len(colors)],
            text=[f'{v:.1f}' for v in vals],
            textposition='outside'
        ))
    fig.update_layout(
        barmode='group',
        title='Perfil medio por cluster',
        height=380, margin=dict(t=40,b=20,l=20,r=20),
        plot_bgcolor='white', paper_bgcolor='white',
        font_family='DM Sans',
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor='#f0f0f0')
    )
    return fig

def plot_cluster_scatter_3d(df_scored, x_col, y_col, z_col, cluster_col='cluster_label'):
    colors = [PALETTE['blue'], PALETTE['teal'], PALETTE['red'],
              PALETTE['orange'], PALETTE['purple'], PALETTE['green']]
    fig = go.Figure()
    for i, cluster in enumerate(sorted(df_scored[cluster_col].unique())):
        sub = df_scored[df_scored[cluster_col] == cluster]
        fig.add_trace(go.Scatter3d(
            x=sub[x_col], y=sub[y_col], z=sub[z_col],
            mode='markers', name=str(cluster),
            marker=dict(size=4, color=colors[i % len(colors)], opacity=0.7)
        ))
    fig.update_layout(
        scene=dict(
            xaxis_title=x_col, yaxis_title=y_col, zaxis_title=z_col,
            bgcolor='white'
        ),
        title='Distribución 3D de clusters (RFM)',
        height=460, margin=dict(t=40,b=0,l=0,r=0),
        paper_bgcolor='white', font_family='DM Sans'
    )
    return fig

def plot_cluster_scatter_2d(df_scored, x_col, y_col, cluster_col='cluster_label'):
    fig = px.scatter(
        df_scored, x=x_col, y=y_col, color=cluster_col,
        color_discrete_sequence=list(PALETTE.values()),
        opacity=0.7, title=f'{x_col} vs {y_col} por cluster'
    )
    fig.update_layout(
        height=380, margin=dict(t=40,b=20,l=20,r=20),
        plot_bgcolor='white', paper_bgcolor='white',
        font_family='DM Sans',
        xaxis=dict(gridcolor='#f0f0f0'),
        yaxis=dict(gridcolor='#f0f0f0')
    )
    return fig

def plot_dendrogram(Z, n_last: int = 40):
    import plotly.figure_factory as ff
    fig = ff.create_dendrogram(
        Z, orientation='bottom',
        color_threshold=0.7 * max(Z[:, 2]),
        colorscale=[PALETTE['blue'], PALETTE['teal'],
                    PALETTE['red'], PALETTE['orange'],
                    PALETTE['purple']]
    )
    fig.update_layout(
        title='Dendrograma — distancias de fusión entre clusters',
        height=380, margin=dict(t=40,b=20,l=20,r=20),
        paper_bgcolor='white', font_family='DM Sans',
        xaxis=dict(showticklabels=False, title='Observaciones'),
        yaxis=dict(title='Distancia', gridcolor='#f0f0f0')
    )
    return fig

def plot_kmodes_heatmap(profiles_detail: dict, features: list):
    """Heatmap de valores más frecuentes por cluster y feature."""
    clusters = list(list(profiles_detail.values())[0].keys())
    rows, text_vals = [], []
    for feat in features:
        row, txt = [], []
        for cl in clusters:
            top = profiles_detail[feat].get(cl, {})
            # Asegurar que top es un diccionario
            if not isinstance(top, dict):
                top = {}
            top_val = max(top, key=top.get) if top else '—'
            top_pct = top.get(top_val, 0) if isinstance(top_val, str) or top_val != '—' else 0
            row.append(top_pct)
            txt.append(f'{top_val}<br>{top_pct:.0%}')
        rows.append(row)
        text_vals.append(txt)
    fig = go.Figure(go.Heatmap(
        z=rows, x=clusters, y=features,
        text=text_vals, texttemplate='%{text}',
        colorscale='teal', showscale=False
    ))
    fig.update_layout(
        title='Valor dominante por cluster y variable (K-Modes)',
        height=max(280, len(features)*50),
        margin=dict(t=40,b=20,l=20,r=20),
        paper_bgcolor='white', font_family='DM Sans'
    )
    return fig

def plot_uplift_by_decile(decile_df):
    colors = [PALETTE['teal'] if u > 0 else PALETTE['red']
              for u in decile_df['uplift_medio']]
    fig = go.Figure(go.Bar(
        x=decile_df['decil'].astype(str),
        y=decile_df['uplift_medio'],
        marker_color=colors,
        text=[f'{u:+.3f}' for u in decile_df['uplift_medio']],
        textposition='outside'
    ))
    fig.update_layout(
        title='Uplift medio por decil (D10 = mayor incrementalidad)',
        height=320, margin=dict(t=40, b=20, l=20, r=20),
        plot_bgcolor='white', paper_bgcolor='white',
        font_family='DM Sans',
        xaxis=dict(title='Decil de uplift', showgrid=False),
        yaxis=dict(title='Uplift score medio', gridcolor='#f0f0f0',
                   zeroline=True, zerolinecolor='#aaa')
    )
    return fig

def plot_uplift_distribution(df_scored):
    fig = go.Figure()
    for treat_val, label, color in [(1, 'Grupo tratado', PALETTE['blue']),
                                     (0, 'Grupo control', PALETTE['gray'])]:
        sub = df_scored[df_scored['tratamiento'] == treat_val]['uplift_score']
        fig.add_trace(go.Histogram(x=sub, name=label, opacity=0.7,
                                    marker_color=color, nbinsx=25))
    fig.update_layout(
        title='Distribución de uplift scores por grupo',
        barmode='overlay', height=300,
        margin=dict(t=40, b=20, l=20, r=20),
        plot_bgcolor='white', paper_bgcolor='white',
        font_family='DM Sans',
        xaxis=dict(title='Uplift score', gridcolor='#f0f0f0'),
        yaxis=dict(title='Frecuencia', gridcolor='#f0f0f0')
    )
    return fig

def plot_timeseries(df, date_col, value_col):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[date_col], y=df[value_col], mode='lines',
        line=dict(color=PALETTE['blue'], width=1.5),
        name=value_col))
    fig.update_layout(
        title=f'Serie de tiempo: {value_col}',
        height=360, margin=dict(t=40,b=20,l=20,r=20),
        plot_bgcolor='white', paper_bgcolor='white',
        font_family='DM Sans')
    fig.update_xaxes(gridcolor='#f0f0f0', title='Fecha')
    fig.update_yaxes(gridcolor='#f0f0f0', title=value_col)
    return fig

def plot_forecast(data, forecast, test=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat_upper'], mode='lines',
        line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat_lower'], mode='lines',
        line=dict(width=0), fill='tonexty',
        fillcolor='rgba(26,76,140,0.15)',
        name='Intervalo 95%', hoverinfo='skip'))
    fig.add_trace(go.Scatter(
        x=data['ds'], y=data['y'], mode='markers',
        marker=dict(color=PALETTE['gray'], size=4, opacity=0.6),
        name='Real'))
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat'], mode='lines',
        line=dict(color=PALETTE['blue'], width=2),
        name='Forecast'))
    if test is not None and len(test):
        fig.add_vline(x=test['ds'].iloc[0], line_dash='dash',
                      line_color=PALETTE['red'])
    fig.update_layout(
        title='Forecast vs Real',
        height=420, margin=dict(t=40,b=20,l=20,r=20),
        plot_bgcolor='white', paper_bgcolor='white',
        font_family='DM Sans',
        legend=dict(orientation='h', y=1.12))
    fig.update_xaxes(gridcolor='#f0f0f0', title='Fecha')
    fig.update_yaxes(gridcolor='#f0f0f0', title='Ventas')
    return fig

def plot_forecast_components(forecast):
    comps = []
    if 'trend' in forecast.columns:
        comps.append(('Tendencia', forecast['ds'], forecast['trend']))
    if 'weekly' in forecast.columns:
        wk = forecast[['ds', 'weekly']].copy()
        wk['dow'] = wk['ds'].dt.dayofweek
        wk = wk.drop_duplicates('dow').sort_values('dow')
        comps.append(('Estacionalidad semanal', wk['ds'].dt.strftime('%A'), wk['weekly']))
    if 'yearly' in forecast.columns:
        yr = forecast[['ds', 'yearly']].copy()
        yr['doy'] = yr['ds'].dt.dayofyear
        yr = yr.drop_duplicates('doy').sort_values('doy')
        comps.append(('Estacionalidad anual', yr['ds'], yr['yearly']))

    fig = make_subplots(rows=len(comps), cols=1,
                        subplot_titles=[c[0] for c in comps])
    for i, (title, x, y) in enumerate(comps):
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines',
                                 line=dict(color=SEQ[i % len(SEQ)], width=2)),
                      row=i+1, col=1)
    fig.update_layout(height=260*len(comps), showlegend=False,
                      margin=dict(t=40,b=20,l=20,r=20),
                      plot_bgcolor='white', paper_bgcolor='white',
                      font_family='DM Sans')
    fig.update_xaxes(gridcolor='#f0f0f0')
    fig.update_yaxes(gridcolor='#f0f0f0')
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
