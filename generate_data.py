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

DATASETS = {
    'churn': gen_churn,
    'ltv': gen_ltv,
    'leads': gen_leads,
    'elasticity': gen_elasticity,
    'basket': gen_basket,
    'nbo': gen_nbo,
}

if __name__ == '__main__':
    import os
    os.makedirs('data', exist_ok=True)
    for name, fn in DATASETS.items():
        df = fn()
        df.to_csv(f'data/{name}.csv', index=False)
        print(f"{name}: {df.shape}")
