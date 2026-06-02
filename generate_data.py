import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

np.random.seed(42)
N = 300

def gen_churn():
    edad = np.random.randint(22, 67, N)
    antiguedad = np.random.randint(1, 73, N)
    productos = np.random.randint(1, 6, N)
    saldo = np.round(np.random.uniform(1, 50, N), 1)
    logins = np.random.randint(0, 31, N)
    quejas = np.where(np.random.random(N) < 0.25,
                      np.random.randint(1, 6, N), 0)
    dias_inactivo = np.random.randint(0, 91, N)
    z = (-1.8 + 0.05*dias_inactivo + 0.5*quejas
         - 0.1*logins - 0.03*saldo - 0.25*productos
         + np.random.normal(0, 0.8, N))
    prob = 1 / (1 + np.exp(-z))
    churn = (prob > 0.5).astype(int)
    return pd.DataFrame({
        'cliente_id': range(1, N+1),
        'dias_inactivo': dias_inactivo,
        'quejas': quejas,
        'logins_30d': logins,
        'saldo_prom_k': saldo,
        'productos': productos,
        'antiguedad_m': antiguedad,
        'edad': edad,
        'churn': churn
    })

def gen_ltv():
    compras = np.random.randint(1, 21, N)
    ticket = np.round(np.random.uniform(50, 800, N) / 10) * 10
    meses = np.random.randint(1, 61, N)
    cats = np.random.randint(1, 8, N)
    edad = np.random.randint(18, 68, N)
    digital = (np.random.random(N) > 0.4).astype(int)
    ltv = np.round(180*compras + 3.2*ticket + 28*meses
                   + 120*cats + 200*digital - 200
                   + np.random.normal(0, 300, N))
    ltv = np.maximum(ltv, 100)
    return pd.DataFrame({
        'cliente_id': range(1, N+1),
        'compras_6m': compras,
        'ticket_prom': ticket.astype(int),
        'meses_cliente': meses,
        'categorias': cats,
        'edad': edad,
        'canal_digital': digital,
        'ltv_anual': ltv.astype(int)
    })

def gen_leads():
    canales = ['Google', 'Meta', 'Organic', 'Email', 'Referido']
    dias = np.random.randint(1, 30, N)
    web = np.random.randint(0, 21, N)
    clicks = np.random.randint(0, 11, N)
    tiempo = np.random.randint(0, 31, N)
    canal = np.random.choice(canales, N)
    z = (-3.5 - 0.06*dias + 0.18*web + 0.45*clicks
         + 0.08*tiempo + np.random.normal(0, 1.2, N))
    prob = 1 / (1 + np.exp(-z))
    conv = (prob > 0.5).astype(int)
    return pd.DataFrame({
        'lead_id': range(1, N+1),
        'dias_contacto': dias,
        'interacciones_web': web,
        'clicks_precio': clicks,
        'tiempo_pag_min': tiempo,
        'canal': canal,
        'convirtio': conv
    })

def gen_elasticity():
    cats = ['Electronica', 'Ropa', 'Hogar', 'Deportes']
    base_prices = {'Electronica': 350, 'Ropa': 120, 'Hogar': 85, 'Deportes': 200}
    betas = {'Electronica': -1.8, 'Ropa': -1.2, 'Hogar': -0.8, 'Deportes': -1.4}
    rows = []
    for i in range(200):
        cat = np.random.choice(cats)
        promo = int(np.random.random() > 0.8)
        bp = base_prices[cat]
        precio = round(bp * (0.65 + np.random.random() * 0.7) / 5) * 5
        beta = betas[cat]
        dem_base = 1000
        units = round(dem_base * (precio/bp)**beta * (1 + promo*0.15)
                      * (0.9 + np.random.random() * 0.2))
        rows.append({'semana': f'S{i+1}', 'precio': precio,
                     'unidades': max(units, 10), 'categoria': cat, 'promocion': promo})
    return pd.DataFrame(rows)

def gen_basket():
    prods = ['Manga shonen', 'Artbook anime', 'Poster coleccionable',
             'Novela grafica', 'Marcadores premium', 'Cuaderno diseno',
             'Stickers kawaii', 'Bolígrafo premium', 'Novela light',
             'Figura coleccionable']
    rules = [(0,1,0.65),(0,2,0.40),(3,5,0.35),(1,6,0.50),(7,5,0.42),(4,8,0.38)]
    rows = []
    for tid in range(1, 401):
        n_prods = np.random.randint(1, 5)
        basket = set()
        anchor = np.random.randint(0, len(prods))
        basket.add(anchor)
        for a, b, p in rules:
            if a == anchor and np.random.random() < p:
                basket.add(b)
        while len(basket) < n_prods:
            basket.add(np.random.randint(0, len(prods)))
        for p in basket:
            rows.append({'ticket_id': f'T{tid}', 'producto': prods[p]})
    return pd.DataFrame(rows)

def gen_nbo():
    prods = ['Seguro vehicular', 'Fondo inversion', 'Prestamo hipotecario',
             'CTS digital', 'Tarjeta debito', 'SOAT premium']
    rows = []
    for i in range(N):
        tc = int(np.random.random() > 0.4)
        ca = int(np.random.random() > 0.5)
        seg = int(np.random.random() > 0.7)
        pr = int(np.random.random() > 0.6)
        ant = np.random.randint(1, 61)
        saldo = round(np.random.uniform(1, 50), 1)
        segmento = 'Alto' if saldo > 20 else ('Medio' if saldo > 10 else 'Bajo')
        scores = [tc*0.6+0.1, saldo/50*0.8+0.1,
                  (ant/60)*0.5+(pr*0.3), ca*0.5+0.2,
                  (1-ca)*0.6+0.1, tc*0.3+0.1]
        scores = [s + np.random.uniform(-0.05, 0.05) for s in scores]
        prod = prods[np.argmax(scores)]
        rows.append({'cliente_id': i+1, 'tiene_tarjeta': tc, 'tiene_cuenta': ca,
                     'tiene_seguro': seg, 'tiene_prestamo': pr,
                     'antiguedad_m': ant, 'saldo_k': saldo,
                     'segmento': segmento, 'producto_adquirido': prod})
    return pd.DataFrame(rows)

def gen_propension():
    recencia = np.random.randint(1, 121, N)
    n_compras = np.random.poisson(4, N).clip(0, 15)
    engagement = np.random.randint(10, 101, N)
    canal_digital = (np.random.random(N) > 0.35).astype(int)
    edad = np.random.randint(22, 65, N)
    n_productos = np.random.randint(1, 6, N)
    z = (-2.5 - 0.025*recencia + 0.22*n_compras
         + 0.035*engagement + 0.8*canal_digital
         + 0.3*n_productos + np.random.normal(0, 0.8, N))
    prob = 1 / (1 + np.exp(-z))
    compro = (prob > 0.5).astype(int)
    return pd.DataFrame({
        'cliente_id': range(1, N+1),
        'recencia_dias': recencia,
        'n_compras_12m': n_compras,
        'engagement_score': engagement,
        'canal_digital': canal_digital,
        'edad': edad,
        'n_productos': n_productos,
        'compro': compro
    })

def gen_winloss():
    monto = np.round(np.random.lognormal(4.5, 0.8, N)).clip(10, 500).astype(int)
    dias_ciclo = np.random.randint(7, 181, N)
    n_reuniones = np.random.randint(1, 11, N)
    n_competidores = np.random.randint(0, 5, N)
    decision_makers = np.random.randint(1, 6, N)
    propuesta = (np.random.random(N) > 0.45).astype(int)
    z = (-1.0 + 0.004*monto + 0.3*n_reuniones
         - 0.4*n_competidores + 0.6*propuesta
         + 0.15*decision_makers - 0.008*dias_ciclo
         + np.random.normal(0, 0.9, N))
    prob = 1 / (1 + np.exp(-z))
    ganado = (prob > 0.5).astype(int)
    return pd.DataFrame({
        'oportunidad_id': range(1, N+1),
        'monto_oportunidad_k': monto,
        'dias_ciclo': dias_ciclo,
        'n_reuniones': n_reuniones,
        'n_competidores': n_competidores,
        'decision_makers': decision_makers,
        'propuesta_personalizada': propuesta,
        'ganado': ganado
    })

def gen_uplift():
    N2 = 500
    edad = np.random.randint(20, 65, N2)
    recencia = np.random.randint(1, 91, N2)
    n_compras = np.random.poisson(3, N2).clip(0, 12)
    ticket = np.round(np.random.lognormal(1.5, 0.6, N2), 1).clip(0.5, 20)
    canal_digital = (np.random.random(N2) > 0.4).astype(int)
    tratamiento = (np.random.random(N2) > 0.5).astype(int)
    base_z = -2.5 + 0.12*n_compras + 0.04*ticket - 0.01*recencia
    base_prob = 1 / (1 + np.exp(-base_z))
    treatment_effect = (0.20*(n_compras/12) + 0.10*canal_digital
                        + np.random.normal(0, 0.05, N2)).clip(0, 0.5)
    conv_prob = np.where(tratamiento == 1,
                         np.minimum(base_prob + treatment_effect, 1.0),
                         base_prob)
    convirtio = (np.random.random(N2) < conv_prob).astype(int)
    return pd.DataFrame({
        'cliente_id': range(1, N2+1),
        'edad': edad,
        'recencia_dias': recencia,
        'n_compras_prev': n_compras,
        'ticket_prom_k': ticket,
        'canal_digital': canal_digital,
        'tratamiento': tratamiento,
        'convirtio': convirtio
    })

def gen_rfm():
    """RFM con 4 segmentos naturales para K-Means."""
    perfiles = {
        'champion':  {'rec': (8, 5),    'frec': (18, 4),  'monto': (18, 4)},
        'at_risk':   {'rec': (65, 20),  'frec': (7, 3),   'monto': (7, 3)},
        'lost':      {'rec': (130, 30), 'frec': (2, 1),   'monto': (2, 1)},
        'potential': {'rec': (15, 8),   'frec': (4, 2),   'monto': (5, 2)},
    }
    probs = [0.20, 0.25, 0.30, 0.25]
    segmentos = np.random.choice(list(perfiles.keys()), N, p=probs)
    recencia   = np.array([max(1, int(np.random.normal(*perfiles[s]['rec'])))   for s in segmentos])
    frecuencia = np.array([max(1, int(np.random.normal(*perfiles[s]['frec'])))  for s in segmentos])
    monto      = np.array([max(0.5, round(np.random.normal(*perfiles[s]['monto']), 1)) for s in segmentos])
    return pd.DataFrame({
        'cliente_id':        range(1, N+1),
        'recencia_dias':     recencia,
        'frecuencia_compras': frecuencia,
        'monto_total_k':     monto,
        'segmento_rfm':      segmentos,
    })

def gen_kmodes_data():
    """Perfiles de cliente 100% categóricos para K-Modes."""
    perfiles_param = {
        'digital_joven':    {'canal': (['Google Ads','Facebook'], [0.55,0.45]),
                             'cat':   (['Tecnología','Moda','Deportes'], [0.50,0.30,0.20]),
                             'freq':  (['Diaria','Semanal'], [0.35,0.65]),
                             'region':(['Lima','Otra'], [0.80,0.20]),
                             'edad':  (['18-24','25-34'], [0.40,0.60]),
                             'device':(['Móvil','Tablet'], [0.85,0.15])},
        'adulto_tradicional':{'canal': (['Email','Orgánico','Referido'], [0.45,0.35,0.20]),
                             'cat':   (['Hogar','Libros','Tecnología'], [0.50,0.35,0.15]),
                             'freq':  (['Mensual','Semanal'], [0.65,0.35]),
                             'region':(['Lima','Arequipa','Otra'], [0.55,0.25,0.20]),
                             'edad':  (['35-44','45-54','55+'], [0.40,0.35,0.25]),
                             'device':(['Desktop','Móvil'], [0.70,0.30])},
        'regional_esporadico':{'canal':(['Orgánico','Referido','Facebook'], [0.40,0.35,0.25]),
                             'cat':   (['Deportes','Hogar','Moda'], [0.45,0.35,0.20]),
                             'freq':  (['Esporádica','Mensual'], [0.60,0.40]),
                             'region':(['Arequipa','Trujillo','Cusco','Otra'], [0.30,0.30,0.20,0.20]),
                             'edad':  (['25-34','35-44'], [0.50,0.50]),
                             'device':(['Móvil','Desktop'], [0.65,0.35])},
    }
    rows = []
    for i in range(N):
        p = np.random.choice(list(perfiles_param.keys()), p=[0.35, 0.35, 0.30])
        pm = perfiles_param[p]
        rows.append({
            'cliente_id':           i + 1,
            'canal_adquisicion':    np.random.choice(pm['canal'][0], p=pm['canal'][1]),
            'categoria_preferida':  np.random.choice(pm['cat'][0],   p=pm['cat'][1]),
            'frecuencia_visita':    np.random.choice(pm['freq'][0],  p=pm['freq'][1]),
            'region':               np.random.choice(pm['region'][0],p=pm['region'][1]),
            'edad_grupo':           np.random.choice(pm['edad'][0],  p=pm['edad'][1]),
            'dispositivo':          np.random.choice(pm['device'][0],p=pm['device'][1]),
            'perfil_real':          p,
        })
    return pd.DataFrame(rows)

def gen_hierarchical():
    """Comportamiento de cliente para clustering jerárquico."""
    perfiles = {
        'vip':         {'rec': (7, 4),    'frec': (20, 4),  'tick': (12, 3),  'cats': (5, 1),  'ant': (36, 8),  'dig': 0.90},
        'regular':     {'rec': (30, 12),  'frec': (8, 3),   'tick': (5, 2),   'cats': (3, 1),  'ant': (18, 6),  'dig': 0.60},
        'en_riesgo':   {'rec': (90, 20),  'frec': (3, 2),   'tick': (3, 1.5), 'cats': (2, 1),  'ant': (24, 8),  'dig': 0.35},
        'nuevo':       {'rec': (10, 5),   'frec': (3, 2),   'tick': (4, 2),   'cats': (2, 1),  'ant': (4, 2),   'dig': 0.75},
    }
    rows = []
    for seg, pm in perfiles.items():
        n_seg = N // len(perfiles)
        for _ in range(n_seg):
            rows.append({
                'recencia_dias':     max(1,  int(np.random.normal(pm['rec'][0],  pm['rec'][1]))),
                'frecuencia_compras':max(1,  int(np.random.normal(pm['frec'][0], pm['frec'][1]))),
                'ticket_prom_k':     max(0.5,round(np.random.normal(pm['tick'][0], pm['tick'][1]), 1)),
                'n_categorias':      max(1,  int(np.random.normal(pm['cats'][0], pm['cats'][1]))),
                'meses_cliente':     max(1,  int(np.random.normal(pm['ant'][0],  pm['ant'][1]))),
                'canal_digital':     int(np.random.random() < pm['dig']),
                'segmento_real':     seg,
            })
    df_h = pd.DataFrame(rows)
    df_h.insert(0, 'cliente_id', range(1, len(df_h)+1))
    return df_h.sample(frac=1, random_state=42).reset_index(drop=True)

DATASETS = {
    'churn': gen_churn,
    'ltv': gen_ltv,
    'leads': gen_leads,
    'elasticity': gen_elasticity,
    'basket': gen_basket,
    'nbo': gen_nbo,
    'propension': gen_propension,
    'winloss': gen_winloss,
    'uplift': gen_uplift,
    'rfm': gen_rfm,
    'kmodes': gen_kmodes_data,
    'hierarchical': gen_hierarchical,
}

if __name__ == '__main__':
    import os
    os.makedirs('data', exist_ok=True)
    for name, fn in DATASETS.items():
        df = fn()
        df.to_csv(f'data/{name}.csv', index=False)
        print(f"{name}: {df.shape}")
