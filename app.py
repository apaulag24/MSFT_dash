import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

# Así debe quedar la línea en app.py:
df = pd.read_csv('data/MSFT.csv', index_col=0, parse_dates=True)
# 1. Configuración y Carga de Datos
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)
server = app.server # Necesario para Google Cloud

try:
    df = pd.read_csv('data/MSFT.csv', index_col=0, parse_dates=True)
    df['Returns'] = df['Close'].pct_change()
except:
    print("Error: No se encontró msft_data.csv. Ejecuta primero get_data.py")

# 2. Layout Principal
app.layout = dbc.Container([
    html.Div([
        html.H1("Dashboard de Análisis: Microsoft (MSFT)", className="display-4 text-primary"),
        html.P("Análisis de Series de Tiempo y Volatilidad (2000-2026)", className="lead"),
    ], className="text-center my-5"),

    dbc.Tabs([
        dbc.Tab(label="Introducción", tab_id="tab-1"),
        dbc.Tab(label="Contexto y Datos", tab_id="tab-2"),
        dbc.Tab(label="Metodología", tab_id="tab-6"),
        dbc.Tab(label="Resultados Interactivos", tab_id="tab-7"),
        dbc.Tab(label="Conclusiones", tab_id="tab-8"),
    ], id="tabs", active_tab="tab-7"),

    html.Div(id="content", className="p-4 border rounded bg-light mt-3")
], fluid=True)

# 3. Callbacks para Navegación
@app.callback(Output("content", "children"), [Input("tabs", "active_tab")])
def render_tab_content(active_tab):
    if active_tab == "tab-1":
        return dcc.Markdown("""
        ### 1. Introducción
        Este proyecto analiza el comportamiento histórico de las acciones de Microsoft (MSFT). 
        El objetivo es identificar patrones de crecimiento, estacionalidad y riesgos mediante un EDA riguroso.
        """)
    
    elif active_tab == "tab-2":
        stats = df['Close'].describe().reset_index()
        return html.Div([
            html.H4("Resumen Estadístico del Precio de Cierre"),
            dbc.Table.from_dataframe(stats, striped=True, bordered=True, hover=True)
        ])

    elif active_tab == "tab-7":
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.Label("Seleccione Visualización:"),
                    dcc.Dropdown(
                        id='selector',
                        options=[
                            {'label': 'Precio de Cierre', 'value': 'Close'},
                            {'label': 'Retornos Diarios', 'value': 'Returns'}
                        ], value='Close'
                    )
                ], width=6),
                dbc.Col([
                    html.Label("Rango de Años:"),
                    dcc.RangeSlider(id='slider', min=2000, max=2026, value=[2015, 2026],
                                    marks={i: str(i) for i in range(2000, 2027, 5)})
                ], width=6)
            ], className="mb-4"),
            dcc.Graph(id='main-graph')
        ])
    
    elif active_tab == "tab-8":
        return html.Div([
            html.H3("Conclusiones del Análisis"),
            dcc.Markdown("""
            * **Tendencia:** MSFT muestra un crecimiento exponencial post-2014.
            * **Estacionariedad:** El Test ADF confirmó que solo los retornos son aptos para modelos predictivos.
            * **Volatilidad:** Se observa 'Clustering' en periodos de crisis global.
            """)
        ])
    return html.P("Sección en desarrollo...")

# 4. Callback para Gráficos Interactivos
@app.callback(
    Output('main-graph', 'figure'),
    [Input('selector', 'value'), Input('slider', 'value')]
)
def update_graph(col, years):
    filtered = df[(df.index.year >= years[0]) & (df.index.year <= years[1])]
    fig = go.Figure()
    color = 'steelblue' if col == 'Close' else 'crimson'
    fig.add_trace(go.Scatter(x=filtered.index, y=filtered[col], line=dict(color=color)))
    fig.update_layout(title=f"Evolución de {col} ({years[0]}-{years[1]})", template="plotly_white")
    return fig

if __name__ == '__main__':
    app.run(debug=True)