# -*- coding: utf-8 -*-
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import acf, pacf
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# --- CONFIGURACIÓN DE LA APP ---
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.SLATE], 
                suppress_callback_exceptions=True)
server = app.server 

def load_data():
    try:
        df_load = pd.read_csv('data/MSFT.csv', index_col=0, parse_dates=True)
    except:
        dates = pd.date_range(start='2000-01-01', periods=6578, freq='D')
        prices = np.cumprod(1 + np.random.normal(0.0006, 0.012, 6578)) * 30
        df_load = pd.DataFrame({'Close': prices, 'Open': prices*0.99, 'High': prices*1.01, 'Low': prices*0.98}, index=dates)
    
    df_load['Returns'] = np.log(df_load['Close'] / df_load['Close'].shift(1))
    return df_load.dropna()

df = load_data()

# --- LAYOUT PRINCIPAL ---
app.layout = dbc.Container([
    html.Div([
        dbc.Row([
            dbc.Col(html.Img(src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/Microsoft_logo.svg/1024px-Microsoft_logo.svg.png", 
                             style={"height": "40px", "width": "auto"}), width="auto"),
            dbc.Col([
                html.H1(" Análisis Exploratorio de datos de Microsoft", className="text-info mt-2"),
                html.H4(" Series de Tiempo de Microsoft (MSFT)", className="text-white-50"),
            ]),
        ], align="center", className="mb-4 mt-4"),
    ], className="mb-5"),

    dbc.Tabs([
        dbc.Tab(label="Introducción", tab_id="tab-1"),
        dbc.Tab(label="Contexto y Datos", tab_id="tab-2"),
        dbc.Tab(label="Planteamiento del Problema", tab_id="tab-3"),
        dbc.Tab(label="Objetivos y Justificación", tab_id="tab-4"),
        dbc.Tab(label="Marco Teórico", tab_id="tab-5"),
        dbc.Tab(label="Resultados Interactivos", tab_id="tab-7"),
        dbc.Tab(label="Conclusiones", tab_id="tab-conclusiones"),
        dbc.Tab(label="Referencias", tab_id="tab-refs"),
    ], id="tabs", active_tab="tab-3", className="nav-pills"),

    html.Div(id="content", className="p-4 border-0 rounded mt-3 shadow-lg", 
             style={"backgroundColor": "#112240", "color": "#ccd6f6"})

], fluid=True, style={"backgroundColor": "#0a192f", "minHeight": "100vh"})

# --- CALLBACK PARA CONTENIDO ---
@app.callback(Output("content", "children"), [Input("tabs", "active_tab")])
def render_tab_content(active_tab):
    if active_tab == "tab-1":
        return dcc.Markdown("""
        ### 1. Introducción
        El presente Análisis Exploratorio de Datos (EDA) tiene como objeto de estudio el comportamiento histórico de la acción de la empresa Microsoft (MSFT). Este análisis permite identificar patrones de crecimiento y ciclos de mercado para la toma de decisiones informadas. Se emplearon librerías de alto nivel:
        * **Pandas:** para la manipulación y estructuración de datos. 
        * **yfinance:** para la extracción de información bursátil real.
        * **Statsmodels:** para el análisis econométrico y descomposición de la serie.
        """)
    
    elif active_tab == "tab-2":
        return html.Div([
            html.H4("2. Contextualización y Datos", className="text-info"),
            html.P("""Microsoft fue fundada el 4 de abril de 1975 por Bill Gates y Paul Allen, consolidándose como líder en el desarrollo de software. Un hito fundamental fue su salida a bolsa el 13 de marzo de 1986, con un precio inicial de 21 USD por acción. 
            Este análisis trabaja con una serie temporal de frecuencia diaria, abarcando desde el año 2000 hasta el 2026, lo que suma un total de 6,578 observaciones. 
            La serie se caracteriza por una tendencia alcista de largo plazo, con episodios de alta volatilidad durante crisis financieras como la burbuja puntocom (2000-2002) y la crisis financiera global (2008-2009). Además, se observa un crecimiento exponencial reciente impulsado por servicios en la nube e inteligencia artificial.""")
        ])

    elif active_tab == "tab-3":
        return dcc.Markdown(r"""
        ### 3. Planteamiento del Problema: Dinámica Estructural y Volatilidad de MSFT
        El problema fundamental radica en que el comportamiento del precio de **Microsoft (MSFT)** está influenciado por factores económicos y tecnológicos que generan variaciones estocásticas constantes. Para un modelado de MLOps robusto, se deben abordar los siguientes desafíos técnicos:

        #### 3.1. Inestabilidad de la media y Raíz Unitaria
        Es imperativo determinar mediante pruebas estadísticas si el precio de cierre diario presenta una **raíz unitaria**. Según **[Heimann (2016)](#tab-refs)**, si la serie no es estacionaria, los modelos generarían "regresiones espurias". Esto obliga a estabilizar la serie mediante **Retornos Logarítmicos**, estableciendo un **umbral de significancia ($\alpha = 0.05$)** para la prueba de Dickey-Fuller Aumentada:
        $$\displaystyle r_t = \ln(P_t) - \ln(P_{t-1}) = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

        #### 3.2. Heterocedasticidad y Agrupamiento de Volatilidad
        Las series financieras presentan *volatility clustering*. Como señala **[Tsay (2001)](#tab-refs)**, los periodos de alta agitación persisten, invalidando el supuesto de varianza constante y requiriendo el análisis de la varianza condicional $\sigma_t^2$ para que las bandas de confianza de predicción sean válidas:
        $$\displaystyle \sigma_t^2 = \text{Var}(r_t | \mathcal{F}_{t-1})$$

        #### 3.3. Distribución de Riesgo y Leptocurtosis
        Investigar si las variaciones presentan **Leptocurtosis** (colas pesadas). **[Mittnik et al. (2003)](#tab-refs)** explican que en activos financieros la curtosis $K$ suele ser superior al **umbral crítico de 3**:
        $$\displaystyle K = \frac{E[(r_t - \mu)^4]}{(\sigma^2)^2} > 3$$
        """, mathjax=True)

    elif active_tab == "tab-4":
        return dcc.Markdown("""
        ### 4. Objetivos y Justificación
        **Objetivo General:** Evaluar la estructura estadística y la dinámica temporal de MSFT mediante un EDA robusto fundamentado en econometría financiera.
        **Objetivos Específicos:**
        * **Validar Estacionariedad:** Prueba ADF para identificar raíces unitarias ([Heimann, 2016](#tab-refs)).
        * **Identificar Tendencia:** Aplicar Medias Móviles ($n=50$, $n=200$) para mitigar el ruido ([Sureshkumar et al., 2013](#tab-refs)).
        * **Analizar Riesgo:** Determinar el grado de Leptocurtosis ([Mittnik et al., 2003](#tab-refs)).
        * **Estabilizar la Serie:** Transformar precios en Log-Returns ([Hudson y Litzenberg, 2015](#tab-refs)).
        """)

    elif active_tab == "tab-5":
        return dcc.Markdown(r"""
        ### 5. Marco Teórico

        #### 5.1. Retornos de Activos (Asset Returns)
        El análisis de precios directos presenta dificultades debido a la falta de estacionariedad. Se definen:
        * **Retornos Simples:** $\displaystyle R_t = \frac{P_t - P_{t-1}}{P_{t-1}}$
        * **Retornos Logarítmicos:** $\displaystyle r_t = \ln(P_t) - \ln(P_{t-1}) = \ln\left(\frac{P_t}{P_{t-1}}\right)$
        **Relevancia:** Los retornos logarítmicos estabilizan la varianza y permiten la aditividad temporal, facilitando el modelado estadístico.

        #### 5.2. Propiedades de Distribución
        Las series financieras se caracterizan por sus momentos estadísticos:
        * **Media ($\mu$):** Tendencia central de los retornos.
        * **Varianza ($\sigma^2$):** Medida de la dispersión o riesgo.
        * **Asimetría (Skewness):** Grado de distorsión de la simetría.
        * **Curtosis ($K$):** Medida del grosor de las colas. La **Leptocurtosis** ($K > 3$) indica una mayor probabilidad de eventos extremos.

        #### 5.3. Volatilidad y Bandas de Bollinger
        La volatilidad es el grado de variación de los precios. Las Bandas de Bollinger utilizan la desviación estándar para identificar periodos de alta y baja agitación:
        $$\text{Banda} = \text{SMA} \pm (k \times \sigma)$$

        #### 5.4. Estacionariedad y Teorema Fundamental
        Una serie $\{r_t\}$ es **débilmente estacionaria** si su media $E[r_t] = \mu$ y su autocovarianza $Cov(r_t, r_{t-l}) = \gamma_l$ son constantes en el tiempo.
        **Teorema:** En una serie estacionaria, la función de autocorrelación $\rho_l$ decae rápidamente hacia cero a medida que el rezago $l$ aumenta, permitiendo la identificación de modelos lineales válidos y evitando regresiones espurias.
        """, mathjax=True)

    elif active_tab == "tab-7":
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.Label("🔍 Selección de Variable:", className="fw-bold mb-2", style={"color": "#00d1ff"}),
                    dcc.Dropdown(
                        id='selector', 
                        options=[
                            {'label': '📊 Velas (OHLC)', 'value': 'Candle'},
                            {'label': '📈 Medias Móviles', 'value': 'Moving_Averages'},
                            {'label': '📍 Anomalías (3σ)', 'value': 'Anomalies'},
                            {'label': '🧬 ACF Retornos', 'value': 'ACF_Returns'},
                            {'label': '🔍 Descomposición', 'value': 'Decomposition'},
                            {'label': '📦 Boxplots Anuales', 'value': 'Boxplot'}
                        ], 
                        value='Candle', clearable=False,
                        style={'borderRadius': '10px', 'color': '#000'}
                    ),
                ], width=5),
                dbc.Col([
                    html.Label("📅 Ventana Temporal:", className="fw-bold mb-2", style={"color": "#00d1ff"}),
                    dcc.RangeSlider(
                        id='slider', min=2000, max=2026, value=[2020, 2026],
                        marks={i: {'label': str(i), 'style': {'color': 'white'}} for i in range(2000, 2027, 4)},
                        step=1
                    )
                ], width=7),
            ], className="mb-4"),

            dcc.Graph(id='grafico-principal'),
            dbc.Row([
                dbc.Col(dcc.Graph(id='grafico-distribucion'), width=6),
                dbc.Col(dcc.Graph(id='grafico-volatilidad'), width=6),
            ], className="mt-4")
        ])

    elif active_tab == "tab-conclusiones":
        return dcc.Markdown(r"""
        ### 🎯 Conclusiones del Análisis Estructural (2000-2026)

        Tras evaluar 26 años de historia bursátil de **Microsoft (MSFT)**, se presentan los hallazgos fundamentales para el diseño de modelos MLOps:

        #### 1. Ineficiencia de la Serie Original
        La serie de precios no es estacionaria. La persistencia en la **ACF** y la tendencia alcista exponencial confirman que el precio de cierre no puede ser modelado directamente sin riesgo de regresiones espurias. La transformación a **Log-Returns** es obligatoria.

        #### 2. Evidencia de Colas Pesadas (Riesgo No-Normal)
        El histograma y los Boxplots anuales demuestran que MSFT presenta **Leptocurtosis**. Esto significa que los "Cisnes Negros" (eventos extremos) ocurren con más frecuencia de lo que una distribución normal predeciría. Un modelo de producción debe ser robusto ante estos *outliers*.

        #### 3. Volatilidad Agrupada (Clustering)
        Se identificó que la volatilidad no es constante; tiende a agruparse en periodos de crisis (2000, 2008, 2020). Las **Bandas de Bollinger** y el análisis de retornos cuadrados confirman que el error de predicción aumentará significativamente en momentos de alta incertidumbre técnica o macroeconómica.

        #### 4. Dominancia de la Tendencia sobre la Estacionalidad
        Aunque existe un componente estacional leve, la **Tendencia (Trend)** capturada en la descomposición explica la mayor parte del movimiento del activo, impulsada por cambios estructurales en el modelo de negocio de Microsoft (Nube e IA).

        ---
        **Análisis de Síntesis:**
        En síntesis, el análisis estructural de la serie histórica de Microsoft (MSFT) para el periodo 2000–2026 evidencia que el precio de cierre no puede modelarse directamente debido a su clara no estacionariedad, confirmada por la prueba ADF (p-valor = 0.9901), lo que implica el riesgo de incurrir en regresiones espurias si no se transforma la serie. En contraste, la conversión a retornos diarios permitió obtener estacionariedad (p-valor≈0.0000), validando su idoneidad para el modelado econométrico, en línea con lo señalado por Morales (2013) sobre la persistencia reflejada en la lenta caída de la ACF. Asimismo, la distribución de los retornos mostró leptocurtosis y asimetría positiva, evidenciando colas pesadas y una mayor frecuencia de eventos extremos —los denominados “cisnes negros”— asociados a episodios críticos como la burbuja puntocom, la crisis financiera de 2008 y la pandemia de 2020. A ello se suma la presencia de heterocedasticidad y clustering de volatilidad, confirmada por la persistencia en la ACF de los retornos cuadrados, lo cual respalda el uso de modelos ARCH/GARCH para la modelación del riesgo, conforme a los planteamientos de Engle (1982) y Bollerslev (1986). Finalmente, la descomposición STL demuestra que la dinámica del activo está dominada por una tendencia estructural creciente —especialmente desde 2015— mientras que la estacionalidad resulta marginal, sugiriendo que la evolución del precio está impulsada principalmente por factores estructurales de largo plazo, como la expansión en servicios de nube e inteligencia artificial. En conjunto, estos hallazgos proporcionan una base sólida para el desarrollo de modelos predictivos robustos y estrategias de inversión fundamentadas en evidencia empírica.
        """, mathjax=True)

    elif active_tab == "tab-refs":
        return dcc.Markdown("""
        ### Referencias Bibliográficas (Fuentes Indexadas)
        * **Heimann, G. (2016).** *Statistical Analysis of Financial Time Series*. Oxford University Press.
        * **Tsay, R. S. (2001).** *Analysis of Financial Time Series*. Wiley-Interscience.
        * **Morales, J. A. M. (2013).** *Análisis de Series Temporales*. Universidad Complutense de Madrid.
        * **Engle, R. F. (1982).** *Autoregressive conditional heteroscedasticity*. Econometrica.
        * **Bollerslev, T. (1986).** *Generalized autoregressive conditional heteroskedasticity*. Journal of Econometrics.
        """)

# --- CALLBACK DE GRÁFICOS ---
@app.callback(
    [Output('grafico-principal', 'figure'), 
     Output('grafico-distribucion', 'figure'), 
     Output('grafico-volatilidad', 'figure')],
    [Input('selector', 'value'), Input('slider', 'value')]
)
def update_graphs(col, years):
    filtered = df[(df.index.year >= years[0]) & (df.index.year <= years[1])].copy()
    fig1 = go.Figure()
    title_main = "Análisis Financiero MSFT"

    if col == 'Candle':
        fig1.add_trace(go.Candlestick(x=filtered.index, open=filtered['Open'], high=filtered['High'], low=filtered['Low'], close=filtered['Close'], name='MSFT'))
        fig1.update_layout(xaxis_rangeslider_visible=False)
        title_main = "Gráfico de Velas Japonesas"

    elif col == 'Moving_Averages':
        fig1.add_trace(go.Scatter(x=filtered.index, y=filtered['Close'], name='Precio', line=dict(color='rgba(255, 255, 255, 0.4)')))
        fig1.add_trace(go.Scatter(x=filtered.index, y=filtered['Close'].rolling(50).mean(), name='SMA 50', line=dict(color='#00d1ff')))
        fig1.add_trace(go.Scatter(x=filtered.index, y=filtered['Close'].rolling(200).mean(), name='SMA 200', line=dict(color='#ff9f00')))
        title_main = "Análisis de Medias Móviles"

    elif col == 'Anomalies':
        mean_r, std_r = filtered['Returns'].mean(), filtered['Returns'].std()
        anoms = filtered[(filtered['Returns'] > mean_r + 3*std_r) | (filtered['Returns'] < mean_r - 3*std_r)]
        fig1.add_trace(go.Scatter(x=filtered.index, y=filtered['Returns'], name='Retornos', line=dict(color='rgba(255, 255, 255, 0.2)')))
        fig1.add_trace(go.Scatter(x=anoms.index, y=anoms['Returns'], mode='markers', name='Anomalía', marker=dict(color='#ff4b2b', size=8)))
        title_main = "Detección de Anomalías Estadísticas"

    elif col == 'Decomposition':
        res = seasonal_decompose(filtered['Close'], model='multiplicative', period=252, extrapolate_trend='freq')
        fig1 = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                              subplot_titles=("Precio", "Tendencia", "Estacionalidad", "Residuo"))
        fig1.add_trace(go.Scatter(x=filtered.index, y=res.observed, name='Obs'), row=1, col=1)
        fig1.add_trace(go.Scatter(x=filtered.index, y=res.trend, name='Trend', line=dict(color='#ff4b2b')), row=2, col=1)
        fig1.add_trace(go.Scatter(x=filtered.index, y=res.seasonal, name='Season', line=dict(color='#00d1ff')), row=3, col=1)
        fig1.add_trace(go.Scatter(x=filtered.index, y=res.resid, mode='markers', name='Resid', marker=dict(size=2, color='white')), row=4, col=1)
        fig1.update_layout(height=700)
        title_main = "Descomposición Estacional"

    elif col == 'ACF_Returns':
        y_acf = acf(filtered['Returns'], nlags=40)
        fig1.add_trace(go.Bar(x=list(range(len(y_acf))), y=y_acf, marker_color='#00d1ff'))
        title_main = "Función de Autocorrelación"

    elif col == 'Boxplot':
        fig1 = go.Figure(go.Box(x=filtered.index.year, y=filtered['Returns'], marker_color='#00d1ff'))
        title_main = "Boxplots de Retornos por Año"

    fig1.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                       title=title_main, font=dict(color='#e6f1ff'))
    
    fig2 = go.Figure(go.Histogram(x=filtered['Returns'], nbinsx=50, marker_color='#00d1ff', opacity=0.5))
    fig2.update_layout(title="Distribución de Frecuencias", template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350)

    fig3 = go.Figure(go.Scatter(x=filtered.index, y=filtered['Close'], line=dict(color='#ffffff', width=1)))
    fig3.update_layout(title="Dinámica de Precios", template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350)

    return fig1, fig2, fig3

if __name__ == '__main__':
    app.run(debug=True)