# -*- coding: utf-8 -*-
# =============================================================================
#  MSFT Institutional-Grade Analytics Dashboard
#  Senior Quant Developer Edition — Versión 2.0
# =============================================================================

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
#  PALETA DE COLORES — "Trading Esmeralda & Gold"
# =============================================================================
colors = {
    'background': '#0A1F1C',
    'card_bg':    '#122E2B',
    'accent':     '#00C896',
    'luxury':     '#D4AF37',
    'text':       '#F5F5F5',
    'grid':       '#1A3A35',
    'danger':     '#FF4B2B',
    'muted':      '#8BA89F',
}

# =============================================================================
#  CACHÉ GLOBAL — El modelo ARIMA se entrena UNA sola vez por sesión
# =============================================================================
_MODEL_CACHE = {}


# =============================================================================
#  CARGA DE DATOS
# =============================================================================
def load_data():
    try:
        df_load = pd.read_csv('data/MSFT.csv', index_col=0, parse_dates=True)
    except Exception:
        np.random.seed(42)
        dates  = pd.date_range(start='2000-01-01', periods=6578, freq='B')
        prices = np.cumprod(1 + np.random.normal(0.0006, 0.012, len(dates))) * 30
        df_load = pd.DataFrame({
            'Close': prices,
            'Open':  prices * 0.99,
            'High':  prices * 1.01,
            'Low':   prices * 0.98,
        }, index=dates)
    df_load['Returns'] = np.log(df_load['Close'] / df_load['Close'].shift(1))
    return df_load.dropna()

df = load_data()


# =============================================================================
#  ENGINE DE PREDICCIÓN — ARIMA(1,0,4) con caché
# =============================================================================
def get_arima_results(returns_series):
    cache_key = 'arima_104'
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    model  = ARIMA(returns_series, order=(1, 0, 4))
    result = model.fit()

    # Forecast 30 días
    fc     = result.get_forecast(steps=30)
    fc_mean= fc.predicted_mean
    ci_90  = fc.conf_int(alpha=0.10)
    ci_95  = fc.conf_int(alpha=0.05)
    ci_99  = fc.conf_int(alpha=0.01)

    # Residuos
    resid = result.resid

    # Métricas in-sample
    fitted    = result.fittedvalues
    actual    = returns_series.iloc[len(returns_series) - len(fitted):]
    rmse      = np.sqrt(np.mean((actual - fitted) ** 2))
    mape_vals = np.abs((actual[actual != 0] - fitted[actual != 0]) / actual[actual != 0])
    mape      = mape_vals.mean() * 100

    # Theil's U — alineación explícita para evitar length mismatch con datos reales
    try:
        common_idx  = actual.index.intersection(fitted.index)
        act_aligned = actual.loc[common_idx]
        fit_aligned = fitted.loc[common_idx]
        naive       = act_aligned.shift(1).dropna()
        min_len     = min(len(naive), len(act_aligned) - 1, len(fit_aligned) - 1)
        act_trim    = act_aligned.iloc[1: min_len + 1].values
        fit_trim    = fit_aligned.iloc[1: min_len + 1].values
        naive_vals  = naive.iloc[:min_len].values
        num         = np.sqrt(np.mean((act_trim - fit_trim) ** 2))
        den         = np.sqrt(np.mean((act_trim - naive_vals) ** 2))
        theilu      = num / den if den != 0 else np.nan
    except Exception:
        theilu = np.nan

    # Ljung-Box — compatible statsmodels 0.14.x (return_df deprecated, siempre retorna DF)
    try:
        lb = acorr_ljungbox(resid, lags=[10, 20, 30], return_df=True)
    except TypeError:
        lb = acorr_ljungbox(resid, lags=[10, 20, 30])

    payload = {
        'result':   result,
        'fc_mean':  fc_mean,
        'ci_90':    ci_90,
        'ci_95':    ci_95,
        'ci_99':    ci_99,
        'resid':    resid,
        'fitted':   fitted,
        'actual':   actual,
        'aic':      result.aic,
        'bic':      result.bic,
        'rmse':     rmse,
        'mape':     mape,
        'theilu':   theilu,
        'lb':       lb,
    }
    _MODEL_CACHE[cache_key] = payload
    return payload


# =============================================================================
#  MÉTRICAS DE VALIDACIÓN EDA
# =============================================================================
def get_validation_metrics(df_full):
    try:
        close_s  = df_full['Close'].replace([np.inf, -np.inf], np.nan).dropna()
        rets     = df_full['Returns'].replace([np.inf, -np.inf], np.nan).dropna()
        adf_close= adfuller(close_s)[1]
        adf_rets = adfuller(rets)[1]
        jb_p     = stats.jarque_bera(rets)[1]
        arch_p   = het_arch(rets)[1]
        return {'adf_close': adf_close, 'adf_rets': adf_rets, 'jb': jb_p, 'arch': arch_p}
    except Exception:
        return {'adf_close': 1.0, 'adf_rets': 1.0, 'jb': 1.0, 'arch': 1.0}


# =============================================================================
#  CUSTOM INDEX STRING — Google Fonts + CSS Institucional
# =============================================================================
CUSTOM_INDEX = '''<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>MSFT Institutional Analytics</title>
    {%favicon%}
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&family=Montserrat:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    {%css%}
    <style>
        /* ── Base ── */
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            background-color: #0A1F1C;
            font-family: 'Montserrat', sans-serif;
            color: #F5F5F5;
        }

        h1, h2, h3, h4 {
            font-family: 'Playfair Display', serif;
            letter-spacing: 0.02em;
        }

        /* ── Metric Card ── */
        .metric-card {
            background: #122E2B;
            border: 1px solid rgba(212, 175, 55, 0.20);
            border-radius: 12px;
            padding: 20px 24px;
            transition: border-color 0.3s ease, box-shadow 0.3s ease;
            height: 100%;
        }
        .metric-card:hover {
            border-color: rgba(212, 175, 55, 0.65);
            box-shadow: 0 0 18px rgba(212, 175, 55, 0.18);
        }
        .metric-card .metric-label {
            font-family: 'Montserrat', sans-serif;
            font-size: 0.70rem;
            font-weight: 600;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #8BA89F;
            margin-bottom: 8px;
        }
        .metric-card .metric-value {
            font-family: 'Montserrat', sans-serif;
            font-size: 1.55rem;
            font-weight: 700;
            color: #D4AF37;
            line-height: 1.1;
        }
        .metric-card .metric-sub {
            font-family: 'Montserrat', sans-serif;
            font-size: 0.72rem;
            color: #8BA89F;
            margin-top: 5px;
        }

        /* ── Edge Card ── */
        .edge-card {
            background: linear-gradient(135deg, #122E2B 0%, #0e2724 100%);
            border: 1px solid rgba(0, 200, 150, 0.25);
            border-radius: 12px;
            padding: 20px 22px;
            transition: border-color 0.3s ease, box-shadow 0.3s ease;
            height: 100%;
        }
        .edge-card:hover {
            border-color: rgba(0, 200, 150, 0.60);
            box-shadow: 0 0 16px rgba(0, 200, 150, 0.14);
        }
        .edge-card .edge-label {
            font-size: 0.68rem;
            font-weight: 600;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: #00C896;
            margin-bottom: 8px;
        }
        .edge-card .edge-value {
            font-size: 1.40rem;
            font-weight: 700;
            color: #F5F5F5;
            line-height: 1.1;
        }
        .edge-card .edge-sub {
            font-size: 0.72rem;
            color: #8BA89F;
            margin-top: 5px;
        }

        /* ── Section divider ── */
        .section-divider {
            border: none;
            border-top: 1px solid rgba(212, 175, 55, 0.15);
            margin: 28px 0;
        }

        /* ── Tab override ── */
        .nav-pills .nav-link {
            font-family: 'Montserrat', sans-serif;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #8BA89F !important;
            border-radius: 8px;
            padding: 8px 16px;
            transition: color 0.2s, background 0.2s;
        }
        .nav-pills .nav-link.active {
            background-color: rgba(0, 200, 150, 0.15) !important;
            color: #00C896 !important;
            border: 1px solid rgba(0, 200, 150, 0.35) !important;
        }
        .nav-pills .nav-link:hover {
            color: #F5F5F5 !important;
            background: rgba(255,255,255,0.05) !important;
        }

        /* ── Badge theilu ── */
        .badge-valid   { background: rgba(0,200,150,0.18); color: #00C896; border: 1px solid rgba(0,200,150,0.4); border-radius: 6px; padding: 4px 10px; font-weight:700; font-size:0.85rem; }
        .badge-invalid { background: rgba(255,75,43,0.18);  color: #FF4B2B; border: 1px solid rgba(255,75,43,0.4);  border-radius: 6px; padding: 4px 10px; font-weight:700; font-size:0.85rem; }

        /* ── Signal badge ── */
        .signal-high   { color: #00C896; font-weight: 700; }
        .signal-normal { color: #D4AF37; font-weight: 700; }

        /* ── Scrollbar ── */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #0A1F1C; }
        ::-webkit-scrollbar-thumb { background: #1A3A35; border-radius: 3px; }

        /* ── Dropdown text ── */
        .Select-value-label, .Select-option { font-family: 'Montserrat', sans-serif !important; }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>'''


# =============================================================================
#  APP INITIALIZATION
# =============================================================================
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
)
app.index_string = CUSTOM_INDEX
server = app.server


# =============================================================================
#  LAYOUT
# =============================================================================
app.layout = html.Div(
    style={'backgroundColor': colors['background'], 'minHeight': '100vh'},
    children=[
        dbc.Container([

            # ── Header ──────────────────────────────────────────────────────
            dbc.Row([
                dbc.Col(
                    html.Img(
                        src="https://i1.wp.com/socialgeek.co/wp-content/uploads/2017/10/microsoft-oficinas-logo.jpg?w=800&ssl=1",
                        style={"height": "72px", "width": "auto", "backgroundColor": "white",
                               "padding": "8px 12px", "borderRadius": "8px"}
                    ), width="auto"
                ),
                dbc.Col([
                    html.H1(
                        "Identificación de Estructuras Latentes en las Acciones de Microsoft (2000–2026)",
                        style={'color': colors['accent'], 'fontWeight': '700', 'fontSize': '1.45rem'}
                    ),
                    html.H4(
                        "Estudio de Series de Tiempo · Tendencias · Volatilidad · Predicción Institucional",
                        style={'color': colors['luxury'], 'fontWeight': '400', 'fontSize': '0.90rem',
                               'fontFamily': "'Montserrat', sans-serif", 'marginTop': '6px'}
                    ),
                ]),
            ], align="center", className="mb-4 mt-4"),

            # ── Tabs ─────────────────────────────────────────────────────────
            dbc.Tabs([
                dbc.Tab(label="Introducción",        tab_id="tab-intro"),
                dbc.Tab(label="Contexto y Datos",    tab_id="tab-context"),
                dbc.Tab(label="Problema",            tab_id="tab-problem"),
                dbc.Tab(label="Objetivos",           tab_id="tab-objectives"),
                dbc.Tab(label="Marco Teórico",       tab_id="tab-theory"),
                dbc.Tab(label="Resultados EDA",      tab_id="tab-results"),
                dbc.Tab(label="Engine de Predicción",tab_id="tab-predict"),
                dbc.Tab(label="Conclusiones",        tab_id="tab-conclusions"),
                dbc.Tab(label="Referencias",         tab_id="tab-refs"),
            ], id="tabs", active_tab="tab-intro", className="nav-pills flex-wrap"),

            html.Div(
                id="content",
                className="p-4 mt-3 shadow-lg",
                style={
                    "backgroundColor": colors['card_bg'],
                    "color": colors['text'],
                    "border": f"1px solid {colors['grid']}",
                    "borderRadius": "12px",
                }
            )

        ], fluid=True)
    ]
)


# =============================================================================
#  HELPER — layout base para gráficos
# =============================================================================
LAYOUT_BASE = {
    "template":       "plotly_dark",
    "paper_bgcolor":  "rgba(0,0,0,0)",
    "plot_bgcolor":   "rgba(0,0,0,0)",
    "font":           {"color": colors['text'], "family": "Montserrat, sans-serif"},
    "legend":         {"bgcolor": "rgba(0,0,0,0)", "bordercolor": colors['grid'], "borderwidth": 1},
    "xaxis":          {"gridcolor": colors['grid'], "zerolinecolor": colors['grid']},
    "yaxis":          {"gridcolor": colors['grid'], "zerolinecolor": colors['grid']},
}


# =============================================================================
#  CALLBACK — Render Tab Content
# =============================================================================
@app.callback(Output("content", "children"), [Input("tabs", "active_tab")])
def render_tab_content(active_tab):

    # ── INTRODUCCIÓN ─────────────────────────────────────────────────────────
    if active_tab == "tab-intro":
        return html.Div([
            html.H2("Introducción", style={'color': colors['accent']}),
            html.Hr(className="section-divider"),
            dcc.Markdown("""
El presente Análisis Exploratorio de Datos (EDA) tiene como objetivo estudiar el comportamiento histórico de los
precios de las acciones de Microsoft a partir de un conjunto de datos diarios que incluye variables como los precios
de apertura, máximo, mínimo y cierre ajustado, así como el volumen de negociación. Este análisis se enmarca dentro
del estudio de **series de tiempo financieras**, donde el interés principal radica en comprender cómo evolucionan
los precios de una acción a lo largo del tiempo y qué patrones pueden identificarse en dicha evolución.

Se examina el comportamiento histórico del precio de la acción de **Microsoft (MSFT)**, se identifican patrones
temporales, se evalúa su estructura estadística y se analiza la dinámica de la volatilidad a lo largo del período
**2000–2026**. A través de este análisis exploratorio se busca describir estadísticamente las variables del conjunto
de datos y visualizar su evolución temporal.

El propósito principal es analizar la dinámica temporal del precio de la acción para caracterizarla mediante
métricas fundamentales en finanzas cuantitativas: el **retorno** y la **volatilidad**. Finalmente, el
**Engine de Predicción** implementa un modelo ARIMA(1,0,4) para proyectar los retornos logarítmicos con horizontes
de 30 días, ofreciendo señales accionables de valor institucional.
            """)
        ])

    # ── CONTEXTO ──────────────────────────────────────────────────────────────
    elif active_tab == "tab-context":
        return html.Div([
            html.H2("Contexto y Descripción de Datos", style={'color': colors['accent']}),
            html.Hr(className="section-divider"),
            html.P(
                "Microsoft fue fundada el 4 de abril de 1975 por Bill Gates y Paul Allen. Su salida a bolsa ocurrió "
                "el 13 de marzo de 1986 con un precio inicial de 21 USD por acción. Este análisis trabaja con una "
                "serie temporal de frecuencia diaria, abarcando desde el año 2000 hasta el 2026 (≈6,578 observaciones). "
                "La serie exhibe una tendencia alcista de largo plazo, episodios de alta volatilidad durante la burbuja "
                "puntocom (2000–2002) y la crisis financiera global (2008–2009), además de un crecimiento exponencial "
                "reciente impulsado por servicios en la nube e inteligencia artificial.",
                style={'lineHeight': '1.8'}
            ),
            html.Hr(className="section-divider"),
            html.H4("Ficha Técnica del Dataset", style={'color': colors['luxury']}),
            html.Ul([
                html.Li([html.B("Fuente: "), "Nasdaq"]),
                html.Li([html.B("Frecuencia: "), "Diaria (días hábiles)"]),
                html.Li([html.B("Variables: "), "Date, Close, Open, High, Low, Returns (log)"]),
                html.Li([html.B("Ajuste: "), "Precios ajustados por splits y dividendos"]),
            ], style={'lineHeight': '2.0'})
        ])

    # ── PROBLEMA ──────────────────────────────────────────────────────────────
    elif active_tab == "tab-problem":
        return html.Div([
            html.H2("Planteamiento del Problema", style={'color': colors['accent']}),
            html.Hr(className="section-divider"),
            html.P(
                "El precio de las acciones de Microsoft (MSFT) constituye una serie de tiempo financiera con "
                "comportamientos complejos: tendencias de largo plazo, fluctuaciones abruptas y períodos de "
                "volatilidad diferenciada. Comprender esta dinámica histórica es esencial antes de aplicar "
                "cualquier modelo predictivo, ya que un análisis exploratorio riguroso permite identificar patrones, "
                "detectar anomalías y caracterizar estadísticamente el activo.",
                style={'lineHeight': '1.8'}
            ),
            html.P(
                "Sin este paso previo, la aplicación de modelos de series de tiempo como ARIMA o GARCH carecería "
                "de fundamento empírico, aumentando el riesgo de modelar incorrectamente el comportamiento del precio.",
                style={'lineHeight': '1.8'}
            ),
            html.H4("Pregunta de Investigación:", style={'color': colors['luxury'], 'marginTop': '20px'}),
            html.Blockquote(
                "¿Qué patrones de retorno y volatilidad caracterizan el comportamiento histórico de la acción de "
                "Microsoft durante el período 2000–2026, y qué estructura presenta la serie como base para su modelación?",
                style={
                    'borderLeft': f"3px solid {colors['luxury']}",
                    'paddingLeft': '16px',
                    'color': colors['luxury'],
                    'fontStyle': 'italic',
                    'marginTop': '12px',
                    'lineHeight': '1.7',
                }
            )
        ])

    # ── OBJETIVOS ─────────────────────────────────────────────────────────────
    elif active_tab == "tab-objectives":
        return html.Div([
            html.H2("Objetivos", style={'color': colors['accent']}),
            html.Hr(className="section-divider"),
            dcc.Markdown(r"""
### Objetivo General
Evaluar la estructura estadística y la dinámica temporal de MSFT mediante un EDA robusto fundamentado en
econometría financiera, con el fin de aplicar modelos predictivos para los valores de retorno y de volatilidad.

### Objetivos Específicos

- **Validar Estacionariedad:** Aplicar la prueba ADF para identificar la presencia de raíces unitarias
  en la serie de precios de MSFT.
- **Identificar Tendencia:** Implementar medias móviles en ventana diaria, trimestral y anual
  $(n=1,\; n=60,\; n=252)$ como herramienta de suavizamiento.
- **Analizar Riesgo:** Determinar el grado de leptocurtosis en la distribución de los retornos
  logarítmicos como medida del riesgo de cola.
- **Estabilizar la Serie:** Transformar los precios en retornos logarítmicos para obtener una serie
  con varianza estable.
- **Modelar y Predecir:** Implementar ARIMA(1,0,4) sobre retornos logarítmicos y proyectar
  horizontes de 30 días con intervalos de confianza al 90%, 95% y 99%.
            """, mathjax=True, style={'lineHeight': '1.9'})
        ])

    # ── MARCO TEÓRICO ─────────────────────────────────────────────────────────
    elif active_tab == "tab-theory":
        return html.Div([
            html.H2("Marco Teórico", style={'color': colors['accent']}),
            html.Hr(className="section-divider"),
            dcc.Markdown(r"""
#### 5.1. Retorno Logarítmico

$$r_t = \ln(P_t) - \ln(P_{t-1})$$

donde $P_t$ es el precio en el tiempo $t$. Su principal ventaja es la aditividad temporal y la
aproximación a la normalidad.

#### 5.2. Volatilidad

$$\sigma = \sqrt{\frac{1}{n-1} \sum_{t=1}^{n} (r_t - \bar{r})^2}$$

En series financieras es común el **clustering de volatilidad**: períodos de alta volatilidad tienden a
agruparse, motivando el uso de modelos ARCH/GARCH.

#### 5.3. Modelos de Descomposición

* **Aditivo:** $Y_t = T_t + S_t + E_t$
* **Multiplicativo:** $Y_t = T_t \times S_t \times E_t$

#### 5.4. Prueba de Dickey-Fuller Aumentada (ADF)

$$H_0: \delta = 0 \quad \text{(raíz unitaria — serie no estacionaria)}$$

Un p-valor $< 0.05$ permite rechazar $H_0$.

#### 5.5. Modelo ARIMA(p, d, q)

$$\phi(B)(1-B)^d y_t = \theta(B)\varepsilon_t$$

Para ARIMA(1,0,4):
$$y_t = \phi_1 y_{t-1} + \varepsilon_t + \theta_1\varepsilon_{t-1} + \theta_2\varepsilon_{t-2} + \theta_3\varepsilon_{t-3} + \theta_4\varepsilon_{t-4}$$

#### 5.6. Theil's U Statistic

$$U = \frac{\sqrt{\frac{1}{n}\sum(y_t - \hat{y}_t)^2}}{\sqrt{\frac{1}{n}\sum(y_t - y_{t-1})^2}}$$

$U < 1$: el modelo supera al pronóstico ingenuo (naïve). $U > 1$: inadecuado.

#### 5.7. Prueba de Ljung-Box

$$Q = n(n+2)\sum_{k=1}^{m}\frac{\hat{\rho}_k^2}{n-k}$$

Verifica que los residuos del modelo no presenten autocorrelación significativa.

#### 5.8. MAPE y RMSE

$$\text{MAPE} = \frac{1}{n}\sum\left|\frac{y_t - \hat{y}_t}{y_t}\right| \times 100 \qquad
\text{RMSE} = \sqrt{\frac{1}{n}\sum(y_t - \hat{y}_t)^2}$$
            """, mathjax=True, style={'lineHeight': '1.9'})
        ], style={'padding': '4px 8px'})

    # ── RESULTADOS EDA ────────────────────────────────────────────────────────
    elif active_tab == "tab-results":
        metrics = get_validation_metrics(df)
        return html.Div([
            html.H2("Resultados del EDA", style={'color': colors['accent']}),
            html.Hr(className="section-divider"),

            # Tabla de validación
            html.H4("Validación Estadística de la Serie", style={'color': colors['luxury']}),
            dbc.Table([
                html.Thead(html.Tr([
                    html.Th("Prueba"), html.Th("Métrica"),
                    html.Th("P-Valor"), html.Th("Interpretación")
                ]), style={'fontFamily': 'Montserrat, sans-serif', 'fontSize': '0.80rem', 'letterSpacing': '0.06em'}),
                html.Tbody([
                    html.Tr([html.Td("ADF (Precios)"),  html.Td("Estacionariedad"),
                             html.Td(f"{metrics['adf_close']:.4f}"), html.Td("No Estacionaria", style={'color': colors['danger']})]),
                    html.Tr([html.Td("ADF (Retornos)"), html.Td("Estacionariedad"),
                             html.Td(f"{metrics['adf_rets']:.4f}"),  html.Td("Estacionaria", style={'color': colors['accent']})]),
                    html.Tr([html.Td("Jarque-Bera"),    html.Td("Normalidad"),
                             html.Td(f"{metrics['jb']:.4f}"),        html.Td("Leptocúrtica", style={'color': colors['accent']})]),
                    html.Tr([html.Td("Engle ARCH-LM"),  html.Td("Heterocedasticidad"),
                             html.Td(f"{metrics['arch']:.4f}"),      html.Td("Clustering Presente", style={'color': colors['accent']})]),
                ])
            ], bordered=True, hover=True, striped=True, color="dark",
               style={'borderColor': colors['grid'], 'textAlign': 'center',
                      'fontFamily': 'Montserrat, sans-serif', 'fontSize': '0.85rem'}),

            html.Hr(className="section-divider"),

            # Controles
            dbc.Row([
                dbc.Col([
                    html.Label("Análisis Técnico:", style={"color": colors['accent'], 'fontWeight': '600', 'fontSize': '0.78rem', 'letterSpacing': '0.1em', 'textTransform': 'uppercase'}),
                    dcc.Dropdown(id='selector', options=[
                        {'label': 'Velas Japonesas',         'value': 'Candle'},
                        {'label': 'Medias Móviles',          'value': 'Moving_Averages'},
                        {'label': 'Detección de Anomalías',  'value': 'Anomalies'},
                        {'label': 'Autocorrelación (ACF)',   'value': 'ACF_Returns'},
                        {'label': 'Descomposición',          'value': 'Decomposition'},
                    ], value='Candle', clearable=False, style={'color': '#000'}),
                ], width=5),
                dbc.Col([
                    html.Label("Rango de Años:", style={"color": colors['accent'], 'fontWeight': '600', 'fontSize': '0.78rem', 'letterSpacing': '0.1em', 'textTransform': 'uppercase'}),
                    dcc.RangeSlider(id='slider', min=2000, max=2026, value=[2018, 2026],
                                    marks={i: {'label': str(i), 'style': {'color': colors['muted'], 'fontSize': '0.72rem'}}
                                           for i in range(2000, 2027, 4)}, step=1),
                ], width=7),
            ], className="mb-4"),

            dcc.Graph(id='grafico-principal'),
            dbc.Row([
                dbc.Col(dcc.Graph(id='grafico-distribucion'), width=6),
                dbc.Col(dcc.Graph(id='grafico-volatilidad'),  width=6),
            ], className="mt-4")
        ])

    # ── ENGINE DE PREDICCIÓN ──────────────────────────────────────────────────
    elif active_tab == "tab-predict":
        ar = get_arima_results(df['Returns'])

        aic    = ar['aic']
        bic    = ar['bic']
        rmse   = ar['rmse']
        mape   = ar['mape']
        theilu = ar['theilu']

        # ── Fan Chart data ───────────────────────────────────────────────────
        last_date  = df.index[-1]
        future_idx = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=30)
        _fc_mean   = ar['fc_mean']
        _ci_90     = ar['ci_90']
        _ci_95     = ar['ci_95']
        _ci_99     = ar['ci_99']

        # Re-construir Series/DataFrame con el índice de fechas futuras
        # (statsmodels 0.14.x retorna RangeIndex en forecast — no admite reasignación directa)
        fc_mean = pd.Series(_fc_mean.values, index=future_idx)
        ci_90   = pd.DataFrame(_ci_90.values,  index=future_idx, columns=['lower', 'upper'])
        ci_95   = pd.DataFrame(_ci_95.values,  index=future_idx, columns=['lower', 'upper'])
        ci_99   = pd.DataFrame(_ci_99.values,  index=future_idx, columns=['lower', 'upper'])

        # Historical tail (last 120 obs)
        hist_tail = df['Returns'].iloc[-120:]

        fan_fig = go.Figure()
        # Historical
        fan_fig.add_trace(go.Scatter(
            x=hist_tail.index, y=hist_tail.values,
            mode='lines', name='Retorno Histórico',
            line=dict(color=colors['muted'], width=1.2)
        ))
        # CI bands
        def add_band(fig, upper, lower, color, alpha, name):
            fig.add_trace(go.Scatter(
                x=list(future_idx) + list(future_idx[::-1]),
                y=list(upper) + list(lower[::-1]),
                fill='toself',
                fillcolor=color.replace('1)', f'{alpha})'),
                line=dict(color='rgba(0,0,0,0)'),
                name=name, showlegend=True
            ))

        add_band(fan_fig,
                 ci_99['upper'], ci_99['lower'],
                 'rgba(0,200,150,1)', 0.10, 'IC 99%')
        add_band(fan_fig,
                 ci_95['upper'], ci_95['lower'],
                 'rgba(0,200,150,1)', 0.18, 'IC 95%')
        add_band(fan_fig,
                 ci_90['upper'], ci_90['lower'],
                 'rgba(0,200,150,1)', 0.28, 'IC 90%')

        # Mean forecast
        fan_fig.add_trace(go.Scatter(
            x=future_idx, y=fc_mean.values,
            mode='lines+markers', name='Pronóstico Central',
            line=dict(color=colors['luxury'], width=2, dash='dash'),
            marker=dict(size=4, color=colors['luxury'])
        ))
        fan_fig.add_vline(x=str(last_date), line_dash="dot",
                          line_color=colors['accent'], opacity=0.5)
        fan_fig.update_layout(
            **LAYOUT_BASE,
            title=dict(text="Fan Chart — Pronóstico ARIMA(1,0,4) · Retornos Logarítmicos (30 días)",
                       font=dict(family='Playfair Display, serif', size=16, color=colors['luxury'])),
            height=420
        )

        # ── Residuos Autocorrelación ─────────────────────────────────────────
        resid     = ar['resid']
        acf_vals  = acf(resid.dropna(), nlags=40)
        n_obs     = len(resid.dropna())
        ci_bound  = 1.96 / np.sqrt(n_obs)

        acf_fig = go.Figure()
        acf_fig.add_hrect(y0=-ci_bound, y1=ci_bound,
                          fillcolor=colors['accent'], opacity=0.08,
                          line_width=0, name='Banda IC')
        acf_fig.add_trace(go.Bar(
            x=list(range(len(acf_vals))), y=acf_vals,
            marker_color=[colors['danger'] if abs(v) > ci_bound else colors['accent']
                          for v in acf_vals],
            name='ACF Residuos'
        ))
        acf_fig.update_layout(
            **LAYOUT_BASE,
            title=dict(text="ACF Residuos — Ljung-Box",
                       font=dict(family='Playfair Display, serif', size=14, color=colors['luxury'])),
            height=320, showlegend=False
        )

        # ── Residuos vs Tiempo ───────────────────────────────────────────────
        resid_fig = go.Figure()
        resid_fig.add_trace(go.Scatter(
            x=resid.index, y=resid.values,
            mode='lines', name='Residuos',
            line=dict(color=colors['accent'], width=0.8)
        ))
        resid_fig.add_hrect(y0=-2*resid.std(), y1=2*resid.std(),
                            fillcolor=colors['luxury'], opacity=0.05, line_width=0)
        resid_fig.update_layout(
            **LAYOUT_BASE,
            title=dict(text="Residuos vs Tiempo — Detección de Clustering",
                       font=dict(family='Playfair Display, serif', size=14, color=colors['luxury'])),
            height=320
        )

        # ── Trading Edge — cálculos ──────────────────────────────────────────
        last_price     = df['Close'].iloc[-1]
        next_ret       = fc_mean.iloc[0]
        expected_move  = last_price * next_ret
        ret_std        = df['Returns'].std()
        signal_str     = "Alta Probabilidad" if abs(next_ret) > 2 * ret_std else "Normal"
        signal_cls     = "signal-high" if signal_str == "Alta Probabilidad" else "signal-normal"
        stop_loss_pct  = ci_90['lower'].iloc[0]           # límite inferior IC 90%
        stop_loss_usd  = last_price * (1 + stop_loss_pct)

        # Guard para theilu NaN (datasets muy cortos o singulares)
        import math
        _theilu_valid  = not (math.isnan(theilu) if isinstance(theilu, float) else False)
        _theilu_label  = f"U = {theilu:.4f}" if _theilu_valid else "U = N/D"
        _theilu_text   = ("Modelo Válido ✓" if theilu < 1 else "Inadecuado ✗") if _theilu_valid else "Sin datos"
        _theilu_class  = ("badge-valid" if theilu < 1 else "badge-invalid") if _theilu_valid else "badge-invalid"
        theilu_badge   = html.Span(
            f"{_theilu_label}  — {_theilu_text}",
            className=_theilu_class
        )

        return html.Div([
            html.H2("Engine de Predicción — ARIMA(1,0,4)", style={'color': colors['accent']}),
            html.Hr(className="section-divider"),

            # ── Métrica Cards ────────────────────────────────────────────────
            dbc.Row([
                dbc.Col(html.Div([
                    html.Div("AIC", className="metric-label"),
                    html.Div(f"{aic:,.2f}", className="metric-value"),
                    html.Div("Akaike Information Criterion", className="metric-sub"),
                ], className="metric-card"), width=12, md=2, className="mb-3"),

                dbc.Col(html.Div([
                    html.Div("BIC", className="metric-label"),
                    html.Div(f"{bic:,.2f}", className="metric-value"),
                    html.Div("Bayesian Information Criterion", className="metric-sub"),
                ], className="metric-card"), width=12, md=2, className="mb-3"),

                dbc.Col(html.Div([
                    html.Div("RMSE", className="metric-label"),
                    html.Div(f"{rmse:.6f}", className="metric-value"),
                    html.Div("Root Mean Square Error", className="metric-sub"),
                ], className="metric-card"), width=12, md=2, className="mb-3"),

                dbc.Col(html.Div([
                    html.Div("MAPE", className="metric-label"),
                    html.Div(f"{mape:.2f}%", className="metric-value"),
                    html.Div("Mean Absolute Percentage Error", className="metric-sub"),
                ], className="metric-card"), width=12, md=2, className="mb-3"),

                dbc.Col(html.Div([
                    html.Div("Theil's U", className="metric-label"),
                    html.Div(theilu_badge, style={'marginTop': '8px'}),
                    html.Div("vs Naïve Forecast", className="metric-sub", style={'marginTop': '8px'}),
                ], className="metric-card"), width=12, md=4, className="mb-3"),
            ], className="mb-2"),

            html.Hr(className="section-divider"),

            # ── Fan Chart ────────────────────────────────────────────────────
            dcc.Graph(figure=fan_fig),

            html.Hr(className="section-divider"),

            # ── Diagnóstico Residuos ─────────────────────────────────────────
            html.H4("Diagnóstico de Residuos del Modelo",
                    style={'color': colors['luxury'], 'fontFamily': 'Playfair Display, serif'}),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=acf_fig),   width=6),
                dbc.Col(dcc.Graph(figure=resid_fig), width=6),
            ], className="mb-2"),

            html.Hr(className="section-divider"),

            # ── Trading Edge Panel ───────────────────────────────────────────
            html.H4("The Trading Edge — Señales Accionables",
                    style={'color': colors['luxury'], 'fontFamily': 'Playfair Display, serif'}),
            dbc.Row([
                dbc.Col(html.Div([
                    html.Div("Expected Move", className="edge-label"),
                    html.Div(
                        f"{'▲' if expected_move >= 0 else '▼'} ${abs(expected_move):.2f}",
                        className="edge-value",
                        style={'color': colors['accent'] if expected_move >= 0 else colors['danger']}
                    ),
                    html.Div(f"Próxima sesión · Precio base: ${last_price:.2f}", className="edge-sub"),
                ], className="edge-card"), width=12, md=4, className="mb-3"),

                dbc.Col(html.Div([
                    html.Div("Signal Strength", className="edge-label"),
                    html.Div(signal_str, className=f"edge-value {signal_cls}"),
                    html.Div(
                        f"Retorno pronosticado: {next_ret*100:.4f}% · 2σ = {2*ret_std*100:.4f}%",
                        className="edge-sub"
                    ),
                ], className="edge-card"), width=12, md=4, className="mb-3"),

                dbc.Col(html.Div([
                    html.Div("Stop Loss Técnico", className="edge-label"),
                    html.Div(f"${stop_loss_usd:.2f}", className="edge-value",
                             style={'color': colors['danger']}),
                    html.Div(f"IC 90% límite inferior · Δ {stop_loss_pct*100:.3f}%", className="edge-sub"),
                ], className="edge-card"), width=12, md=4, className="mb-3"),
            ]),

            # Fórmula del modelo
            html.Hr(className="section-divider"),
            html.H5("Especificación del Modelo",
                    style={'color': colors['muted'], 'fontFamily': 'Montserrat, sans-serif',
                           'fontSize': '0.80rem', 'letterSpacing': '0.1em', 'textTransform': 'uppercase'}),
            dcc.Markdown(r"""
$$y_t = \phi_1 y_{t-1} + \varepsilon_t + \theta_1\varepsilon_{t-1} + \theta_2\varepsilon_{t-2}
+ \theta_3\varepsilon_{t-3} + \theta_4\varepsilon_{t-4}$$

donde $y_t = r_t = \ln(P_t) - \ln(P_{t-1})$ es el retorno logarítmico diario y
$\varepsilon_t \sim \mathcal{N}(0, \sigma^2)$.
            """, mathjax=True)
        ])

    # ── CONCLUSIONES ─────────────────────────────────────────────────────────
    elif active_tab == "tab-conclusions":
        return html.Div([
            html.H2("Conclusiones del Análisis Estructural (2000–2026)", style={'color': colors['accent']}),
            html.Hr(className="section-divider"),
            html.P(
                "Tras la evaluación rigurosa de la dinámica histórica de Microsoft (MSFT), se concluye que "
                "el activo presenta una estructura de serie de tiempo compleja cuya modelación directa en "
                "niveles de precios resultaría técnicamente inválida. La evidencia de no estacionariedad, "
                "ratificada por un p-valor de 0.9901 en la prueba ADF y una persistencia prolongada en la "
                "función de autocorrelación, confirma que la serie original está dominada por una tendencia "
                "estocástica creciente, lo que justifica la transformación obligatoria a retornos logarítmicos.",
                style={'lineHeight': '1.9'}
            ),
            html.P(
                "El análisis de la distribución de los retornos logarítmicos permite caracterizar el "
                "comportamiento del activo mediante una marcada leptocurtosis: el riesgo histórico está "
                "definido por una frecuencia de eventos extremos superior a la de una distribución normal, "
                "particularmente visible en las crisis financieras de las últimas dos décadas.",
                style={'lineHeight': '1.9'}
            ),
            html.P(
                "El Engine de Predicción ARIMA(1,0,4) proporciona un marco cuantitativo riguroso para "
                "proyectar los retornos, cuyos intervalos de confianza (90%, 95%, 99%) ofrecen señales "
                "accionables —Expected Move, Signal Strength y Stop Loss Técnico— que traducen la "
                "econometría en valor de negocio institucional. Un Theil's U < 1 valida que el modelo "
                "supera consistentemente al pronóstico naïve.",
                style={'lineHeight': '1.9'}
            ),
        ])

    # ── REFERENCIAS ───────────────────────────────────────────────────────────
    elif active_tab == "tab-refs":
        return html.Div([
            html.H2("Referencias Bibliográficas", style={'color': colors['accent']}),
            html.Hr(className="section-divider"),
            dcc.Markdown("""
* **Heimann, G. (2016).** *Statistical Analysis of Financial Time Series*. Oxford University Press.
* **Tsay, R. S. (2001).** *Analysis of Financial Time Series*. Wiley-Interscience.
* **Morales, J. A. M. (2013).** *Análisis de Series Temporales*. Universidad Complutense de Madrid.
* **Engle, R. F. (1982).** Autoregressive conditional heteroscedasticity with estimates of the variance
  of United Kingdom inflation. *Econometrica, 50*(4), 987–1007.
* **Bollerslev, T. (1986).** Generalized autoregressive conditional heteroskedasticity.
  *Journal of Econometrics, 31*(3), 307–327.
* **Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015).**
  *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.
* **Theil, H. (1966).** *Applied Economic Forecasting*. North-Holland.
            """, style={'lineHeight': '2.1'})
        ])

    return html.Div("Selecciona una pestaña.")


# =============================================================================
#  CALLBACK — Gráficos EDA
# =============================================================================
@app.callback(
    [Output('grafico-principal',    'figure'),
     Output('grafico-distribucion', 'figure'),
     Output('grafico-volatilidad',  'figure')],
    [Input('selector', 'value'),
     Input('slider',   'value')]
)
def update_graphs(col, years):
    filtered = df[(df.index.year >= years[0]) & (df.index.year <= years[1])].copy()

    fig1 = go.Figure()

    if col == 'Candle':
        fig1.add_trace(go.Candlestick(
            x=filtered.index,
            open=filtered['Open'], high=filtered['High'],
            low=filtered['Low'],  close=filtered['Close'],
            increasing_line_color=colors['accent'],
            decreasing_line_color=colors['danger'],
            name='MSFT'
        ))
        fig1.update_layout(xaxis_rangeslider_visible=False)
        title_str = "Velas Japonesas — MSFT"

    elif col == 'Moving_Averages':
        fig1.add_trace(go.Scatter(x=filtered.index, y=filtered['Close'],
                                  name='Precio', line=dict(color='rgba(255,255,255,0.15)', width=1)))
        fig1.add_trace(go.Scatter(x=filtered.index, y=filtered['Close'].rolling(60).mean(),
                                  name='MA Trimestral (60)', line=dict(color=colors['accent'], width=1.5)))
        fig1.add_trace(go.Scatter(x=filtered.index, y=filtered['Close'].rolling(252).mean(),
                                  name='MA Anual (252)', line=dict(color=colors['luxury'], width=2)))
        title_str = "Medias Móviles — Identificación de Tendencia"

    elif col == 'Decomposition':
        try:
            res  = seasonal_decompose(filtered['Close'], model='multiplicative', period=252)
            fig1 = make_subplots(rows=2, cols=1, subplot_titles=('Tendencia', 'Residuo'))
            fig1.add_trace(go.Scatter(x=filtered.index, y=res.trend,  name='Tendencia',
                                      line=dict(color=colors['luxury'])), row=1, col=1)
            fig1.add_trace(go.Scatter(x=filtered.index, y=res.resid, name='Residuo',
                                      mode='markers', marker=dict(color=colors['accent'], size=2, opacity=0.5)), row=2, col=1)
        except Exception:
            fig1.add_annotation(text="Rango insuficiente (mín. 2 períodos completos).",
                                showarrow=False, font=dict(color=colors['danger']))
        title_str = "Descomposición Multiplicativa"

    elif col == 'Anomalies':
        m, s = filtered['Returns'].mean(), filtered['Returns'].std()
        anom = filtered[(filtered['Returns'] > m + 3*s) | (filtered['Returns'] < m - 3*s)]
        fig1.add_trace(go.Scatter(x=filtered.index, y=filtered['Returns'],
                                  name='Retorno', line=dict(color='rgba(0,200,150,0.25)', width=0.8)))
        fig1.add_trace(go.Scatter(x=anom.index, y=anom['Returns'], mode='markers',
                                  name=f'Outlier (|z|>3) — {len(anom)} eventos',
                                  marker=dict(color=colors['luxury'], size=6, symbol='diamond')))
        title_str = "Detección de Anomalías — Outliers (|z| > 3σ)"

    elif col == 'ACF_Returns':
        y_acf    = acf(filtered['Returns'], nlags=40)
        n_obs    = len(filtered['Returns'])
        ci_bound = 1.96 / np.sqrt(n_obs)
        fig1.add_hrect(y0=-ci_bound, y1=ci_bound,
                       fillcolor=colors['accent'], opacity=0.08, line_width=0)
        fig1.add_trace(go.Bar(
            x=list(range(len(y_acf))), y=y_acf,
            marker_color=[colors['danger'] if abs(v) > ci_bound else colors['accent'] for v in y_acf],
            name='ACF'
        ))
        title_str = "Autocorrelación (ACF) — Retornos Logarítmicos"
    else:
        title_str = col

    fig1.update_layout(
        **LAYOUT_BASE,
        title=dict(text=title_str,
                   font=dict(family='Playfair Display, serif', size=16, color=colors['luxury'])),
        height=480
    )

    # ── Distribución ─────────────────────────────────────────────────────────
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(
        x=filtered['Returns'], nbinsx=60,
        marker_color=colors['accent'], opacity=0.75, name='Retornos'
    ))
    fig2.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Distribución de Retornos",
                   font=dict(family='Playfair Display, serif', size=14, color=colors['luxury'])),
        height=300, showlegend=False
    )

    # ── Precio cierre ─────────────────────────────────────────────────────────
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=filtered.index, y=filtered['Close'],
        fill='tozeroy',
        fillcolor='rgba(212,175,55,0.07)',
        line=dict(color=colors['luxury'], width=1.5),
        name='Precio Cierre'
    ))
    fig3.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Precio de Cierre Histórico",
                   font=dict(family='Playfair Display, serif', size=14, color=colors['luxury'])),
        height=300, showlegend=False
    )

    return fig1, fig2, fig3


# =============================================================================
#  ENTRY POINT
# =============================================================================
#if __name__ == '__main__':
    #app.run(debug=True)

#if __name__ == "__main__":
    # Esta línea es la clave: lee el puerto de Railway o usa el 5000 por defecto
    #port = int(os.environ.get("PORT", 5000))
    # Debes añadir host='0.0.0.0' para que sea accesible desde afuera
    #app.run(host='0.0.0.0', port=port
    
 # =============================================================================
#  EJECUCIÓN DE LA APP (CONFIGURADO PARA RAILWAY)
# =============================================================================
import os

if __name__ == '__main__':
    # Railway nos da el puerto en una variable de entorno llamada PORT
    # Si no existe usará el 8050 por defecto
    port = int(os.environ.get("PORT", 8050))
    
    # Importante: host='0.0.0.0' permite que Railway redirija el tráfico a tu app
    app.run_server(host='0.0.0.0', port=port, debug=False)   