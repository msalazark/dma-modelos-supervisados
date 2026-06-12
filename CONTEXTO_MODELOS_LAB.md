# Contexto de casos para los modelos del CRISP-DM Lab

Este documento agrega contexto de negocio y de datos para cada modelo definido en `models.py` del lab `dma-modelos-supervisados`. La idea es que el estudiante no vea cada modelo como un ejercicio aislado de codigo, sino como una situacion realista de Marketing Analytics donde la data representa clientes, leads, ventas, oportunidades o transacciones.

Cada caso sigue una logica CRISP-DM: problema de negocio, data disponible, variable objetivo o criterio de agrupacion, uso del modelo e interpretacion esperada.

## 1. Churn Prediction

**Caso:** Un banco quiere anticipar que clientes tienen mayor probabilidad de abandonar sus productos durante los proximos 30 dias.

**Data usada:** La tabla simula una cartera de clientes bancarios con actividad reciente, quejas, uso digital, saldo promedio y cantidad de productos contratados.

**Variables principales:** `dias_inactivo`, `quejas`, `logins_30d`, `saldo_prom_k`, `productos`.

**Variable objetivo:** `churn`, donde 1 indica cliente fugado o en alto riesgo de fuga y 0 indica cliente retenido.

**Modelo:** Random Forest Classifier.

**Uso analitico:** Priorizar acciones de retencion sobre clientes con mayor riesgo, especialmente cuando el costo de perder un cliente es mayor que el costo de contactarlo.

**Interpretacion:** Un mayor numero de dias inactivo y mas quejas deberian elevar el riesgo. Mas logins, mayor saldo y mas productos suelen indicar mayor vinculacion y menor churn.

## 2. Lead Scoring con Regresion Logistica

**Caso:** Una universidad o institucion educativa recibe muchos leads, pero el equipo comercial no puede llamar con la misma intensidad a todos.

**Data usada:** La tabla representa leads capturados por formularios o campanas digitales, con senales de interes como interacciones web, clicks en precios y tiempo navegando.

**Variables principales:** `dias_contacto`, `interacciones_web`, `clicks_precio`, `tiempo_pag_min`.

**Variable objetivo:** `convirtio`, donde 1 indica que el lead se matriculo o avanzo a conversion y 0 indica que no convirtio.

**Modelo:** Regresion Logistica.

**Uso analitico:** Ordenar leads por probabilidad de conversion y definir reglas de priorizacion comercial.

**Interpretacion:** Los coeficientes y odds ratios ayudan a explicar que senales aumentan o reducen la probabilidad de conversion.

## 3. LTV / Customer Value

**Caso:** Un retailer necesita estimar el valor anual esperado de sus clientes para decidir cuanto invertir en adquisicion, retencion o beneficios.

**Data usada:** La tabla simula clientes con historial de compras, ticket promedio, antiguedad, cantidad de categorias compradas y canal digital.

**Variables principales:** `compras_6m`, `ticket_prom`, `meses_cliente`, `categorias`, `canal_digital`.

**Variable objetivo:** `ltv_anual`.

**Modelo:** Ridge Regression.

**Uso analitico:** Estimar valor futuro del cliente y segmentar la base en clientes de bajo, medio y alto valor.

**Interpretacion:** El modelo permite discutir error de prediccion, R2, MAE, RMSE y MAPE. Tambien permite conectar LTV con decisiones de CAC maximo.

## 4. Elasticidad de Precio

**Caso:** Una cadena retail quiere saber si subir o bajar precios incrementa o destruye ingresos, y si las promociones realmente generan demanda adicional.

**Data usada:** La tabla simula ventas semanales por categoria, precio, unidades vendidas y presencia de promocion.

**Variables principales:** `precio`, `unidades`, `categoria`, `promocion`.

**Variable objetivo:** `log_unidades`, construida a partir de `unidades`.

**Modelo:** Regresion lineal log-log.

**Uso analitico:** Medir la sensibilidad de la demanda frente al precio. La elasticidad se interpreta con el coeficiente de `log_precio`.

**Interpretacion:** Una elasticidad negativa indica que al subir precio bajan unidades. Si el valor absoluto es mayor a 1, la demanda es elastica; si es menor a 1, es inelastica.

## 5. Market Basket Analysis

**Caso:** Una libreria quiere descubrir productos que suelen comprarse juntos para disenar bundles, recomendaciones o acciones de cross-sell.

**Data usada:** La tabla representa tickets de compra. Cada fila contiene un producto dentro de un ticket.

**Variables principales:** `ticket_id`, `producto`.

**Variable objetivo:** No aplica. Es un modelo no supervisado de reglas de asociacion.

**Modelo:** Apriori y reglas de asociacion.

**Uso analitico:** Identificar reglas del tipo "si compra A, tambien suele comprar B".

**Interpretacion:** Soporte mide frecuencia, confianza mide probabilidad condicional y lift mide si la asociacion es mas fuerte que el azar.

## 6. Propension a Compra

**Caso:** Una empresa con base de clientes quiere priorizar a quienes tienen mayor probabilidad de comprar en una campana proxima.

**Data usada:** La tabla simula comportamiento reciente, historial de compras, engagement, uso digital, edad y cantidad de productos.

**Variables principales:** `recencia_dias`, `n_compras_12m`, `engagement_score`, `canal_digital`, `edad`, `n_productos`.

**Variable objetivo:** `compro`.

**Modelo:** Gradient Boosting Classifier.

**Uso analitico:** Crear audiencias de alta propension para mejorar conversion y reducir gasto comercial en clientes frios.

**Interpretacion:** La importancia de variables muestra que factores explican mejor la probabilidad de compra. El ranking de scores sirve para decidir a quien contactar primero.

## 7. Win / Loss de Oportunidades

**Caso:** Un equipo B2B quiere estimar que oportunidades comerciales tienen mayor probabilidad de cierre para priorizar tiempo de ejecutivos.

**Data usada:** La tabla representa oportunidades de venta con monto, duracion del ciclo, reuniones, competidores, decisores y si la propuesta fue personalizada.

**Variables principales:** `monto_oportunidad_k`, `dias_ciclo`, `n_reuniones`, `n_competidores`, `decision_makers`, `propuesta_personalizada`.

**Variable objetivo:** `ganado`.

**Modelo:** Random Forest Classifier.

**Uso analitico:** Detectar deals con alta probabilidad de ganar, oportunidades en riesgo y factores que aumentan o reducen el win rate.

**Interpretacion:** Mas competidores y ciclos largos suelen reducir la probabilidad de cierre. Mas reuniones, decisores involucrados y propuesta personalizada suelen aumentar la probabilidad de ganar.

## 8. Uplift / Incrementalidad

**Caso:** Una campana de marketing no solo quiere saber quien compra, sino quien compra gracias a la campana.

**Data usada:** La tabla simula clientes tratados y de control, con conversion observada y variables de comportamiento previo.

**Variables principales:** `edad`, `recencia_dias`, `n_compras_prev`, `ticket_prom_k`, `canal_digital`, `tratamiento`.

**Variable objetivo:** `convirtio`.

**Modelo:** Two-model approach con dos regresiones logisticas: una para tratados y otra para control.

**Uso analitico:** Separar clientes persuadibles de clientes que comprarian igual sin campana, y evitar gastar presupuesto en audiencias sin impacto incremental.

**Interpretacion:** El `uplift_score` es la diferencia entre probabilidad de conversion con tratamiento y sin tratamiento. Un uplift alto indica mayor impacto incremental esperado.

## 9. Next Best Offer

**Caso:** Un banco quiere recomendar el siguiente producto mas relevante para cada cliente en lugar de ofrecer el mismo producto a toda la cartera.

**Data usada:** La tabla simula clientes con productos actuales, antiguedad, saldo y el producto finalmente adquirido.

**Variables principales:** `tiene_tarjeta`, `tiene_cuenta`, `tiene_seguro`, `tiene_prestamo`, `antiguedad_m`, `saldo_k`.

**Variable objetivo:** `producto_adquirido`.

**Modelo:** Random Forest multiclase.

**Uso analitico:** Recomendar el producto con mayor probabilidad de adopcion y evaluar precision top-1 y top-2.

**Interpretacion:** La exactitud top-2 es especialmente util porque en negocio puede bastar con que el producto correcto este entre las dos mejores recomendaciones.

## 10. Segmentacion RFM con K-Means

**Caso:** Un retailer quiere segmentar clientes por comportamiento de compra para definir estrategias diferenciadas de retencion, reactivacion y fidelizacion.

**Data usada:** La tabla simula clientes con recencia, frecuencia y monto acumulado.

**Variables principales:** `recencia_dias`, `frecuencia_compras`, `monto_total_k`.

**Variable objetivo:** No aplica. Es clustering no supervisado.

**Modelo:** K-Means.

**Uso analitico:** Construir segmentos como champions, potenciales, en riesgo y perdidos.

**Interpretacion:** Los centroides permiten describir cada cluster. Silhouette, inertia y curva de codo ayudan a evaluar si la segmentacion es razonable.

## 11. Segmentacion Categorica con K-Modes

**Caso:** Una institucion educativa o marca digital necesita perfilar prospectos segun variables categoricas como canal, region, dispositivo y categoria preferida.

**Data usada:** La tabla simula perfiles de usuario con atributos 100% categoricos.

**Variables principales:** `canal_adquisicion`, `categoria_preferida`, `frecuencia_visita`, `region`, `edad_grupo`, `dispositivo`.

**Variable objetivo:** No aplica. Es clustering no supervisado.

**Modelo:** K-Modes.

**Uso analitico:** Descubrir perfiles accionables para personalizar comunicacion, canal y mensaje.

**Interpretacion:** A diferencia de K-Means, K-Modes trabaja con modas categoricas y distancia de Hamming. El resultado se interpreta por la categoria dominante en cada variable.

## 12. Forecast de Demanda con Prophet

**Caso:** Un e-commerce quiere anticipar ventas diarias para planificar inventario, compras y acciones comerciales.

**Data usada:** La tabla simula ventas diarias durante dos anos, con tendencia, estacionalidad semanal, estacionalidad anual y ruido.

**Variables principales:** `fecha`, `ventas`.

**Variable objetivo:** `ventas`, proyectada hacia adelante.

**Modelo:** Prophet.

**Uso analitico:** Pronosticar demanda futura y evaluar error en un horizonte temporal.

**Interpretacion:** MAE, RMSE y MAPE muestran la calidad del pronostico. El split debe respetar el orden temporal; no se debe mezclar aleatoriamente train y test.

## 13. Clustering Jerarquico

**Caso:** Un banco quiere explorar la estructura natural de segmentos de clientes sin decidir de antemano cuantos grupos existen.

**Data usada:** La tabla simula comportamiento de cliente con recencia, frecuencia, ticket promedio, categorias, antiguedad y canal digital.

**Variables principales:** `recencia_dias`, `frecuencia_compras`, `ticket_prom_k`, `n_categorias`, `meses_cliente`, `canal_digital`.

**Variable objetivo:** No aplica. Es clustering no supervisado.

**Modelo:** Agglomerative Clustering con linkage jerarquico.

**Uso analitico:** Usar el dendrograma para decidir cortes y construir una taxonomia de clientes.

**Interpretacion:** El coeficiente cofenetico ayuda a evaluar que tan bien el dendrograma preserva las distancias originales. Los perfiles promedio por cluster permiten traducir grupos tecnicos a segmentos de negocio.

## Como usar este contexto en clase

1. Antes de entrenar el modelo, pedir al estudiante que identifique el problema de negocio y la decision que se tomaria con el output.
2. Revisar si las columnas disponibles realmente representan senales utiles para esa decision.
3. Entrenar el modelo y conectar metricas tecnicas con criterios de negocio.
4. Interpretar outputs: probabilidades, coeficientes, importancia de variables, reglas, clusters o forecast.
5. Discutir limitaciones: datos sinteticos, sesgos, costo de errores, estabilidad temporal y necesidad de validacion con datos reales.

## Nota sobre los datos sinteticos

Los datos son generados para fines pedagogicos. Tienen relaciones intencionales entre variables para que los modelos encuentren patrones interpretables. En un caso real se deberia validar calidad de datos, definicion de target, ventana temporal, leakage, representatividad, costo de falsos positivos y falsos negativos, y criterios de aceptacion con stakeholders.
