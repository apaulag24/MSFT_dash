import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd

# 1. Configuración y Carga de Datos
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)
server = app.server 

try:
    df = pd.read_csv('data/MSFT.csv', index_col=0, parse_dates=True)
    df['Returns'] = df['Close'].pct_change()
except:
    print("Error: No se encontró el archivo de datos.")

# 2. Estilos Personalizados (Inyección de CSS)
colors = {
    'background': '#0A1F1C',       # Verde bosque oscuro
    'card_bg': '#122E2B',          # Verde oscuro medio
    'accent': '#00C896',           # Verde menta brillante
    'luxury': '#D4AF37',           # Dorado
    'text': '#F5F5F5'              # Gris humo
}

# 3. Layout Principal con Estilos Aplicados
app.layout = dbc.Container([
    # Inyectamos CSS directamente para cambiar el fondo de toda la página
    html.Style(f"""
        body {{ background-color: {colors['background']}; color: {colors['text']}; }}
        .card {{ background-color: {colors['card_bg']}; border: 1px solid {colors['accent']}; }}
        .nav-tabs .nav-link {{ color: {colors['text']}; }}
        .nav-tabs .nav-link.active {{ 
            background-color: {colors['accent']} !important; 
            color: {colors['background']} !important; 
            border-color: {colors['accent']};
        }}
        .dropdown .Select-control {{ background-color: {colors['card_bg']}; color: white; }}
    """),

    html.Div([
        html.H1("Dashboard de Análisis: Microsoft (MSFT)", style={'color': colors['accent']}, className="display-4"),
        html.P("Análisis de Series de Tiempo y Volatilidad (2000-2026)", className="lead"),
    ], className="text-center my-5"),

    dbc.Tabs([
        dbc.Tab(label="Introducción", tab_id="tab-1"),
        dbc.Tab(label="Contexto y Datos", tab_id="tab-2"),
        dbc.Tab(label="Metodología", tab_id="tab-6"),
        dbc.Tab(label="Resultados Interactivos", tab_id="tab-7"),
        dbc.Tab(label="Conclusiones", tab_id="tab-8"),
    ], id="tabs", active_tab="tab-7", className="mb-3"),

    # El contenedor de contenido ahora tiene el color de las "Tarjetas"
    html.Div(id="content", style={'backgroundColor': colors['card_bg'], 'color': colors['text']}, className="p-4 border rounded mt-3")
], fluid=True)

# 4. Callbacks para Navegación
@app.callback(Output("content", "children"), [Input("tabs", "active_tab")])
def render_tab_content(active_tab):
    if active_tab == "tab-1":
        return dcc.Markdown(f"""
        ### 1. Introducción
        Este proyecto analiza el comportamiento histórico de las acciones de Microsoft (MSFT). 
        El objetivo es identificar patrones de crecimiento, estacionalidad y riesgos mediante un EDA riguroso.
        """, style={'color': colors['text']})
    
    elif active_tab == "tab-2":
        stats = df['Close'].describe().reset_index()
        return html.Div([
            html.H4("Resumen Estadístico del Precio de Cierre", style={'color': colors['luxury']}),
            dbc.Table.from_dataframe(stats, striped=True, bordered=True, hover=True, dark=True)
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
                        ], value='Close',
                        style={'backgroundColor': colors['background'], 'color': '#000'} # Dropdown necesita fondo claro para leer texto usualmente
                    )
                ], width=6),
                dbc.Col([
                    html.Label("Rango de Años:"),
                    dcc.RangeSlider(id='slider', min=2000, max=2026, value=[2015, 2026],
                                    marks={i: {'label': str(i), 'style': {'color': colors['text']}} for i in range(2000, 2027, 5)})
                ], width=6)
            ], className="mb-4"),
            dcc.Graph(id='main-graph')
        ])
    
    elif active_tab == "tab-8":
        return html.Div([
            html.H3("Conclusiones del Análisis", style={'color': colors['luxury']}),
            dcc.Markdown(f"""
            * **Tendencia:** MSFT muestra un crecimiento exponencial post-2014.
            * **Estacionariedad:** El Test ADF confirmó que solo los retornos son aptos para modelos predictivos.
            * **Volatilidad:** Se observa 'Clustering' en periodos de crisis global.
            """, style={'color': colors['text']})
        ])
    return html.P("Sección en desarrollo...")

# 5. Callback para Gráficos Interactivos con Estilo Oscuro
@app.callback(
    Output('main-graph', 'figure'),
    [Input('selector', 'value'), Input('slider', 'value')]
)
def update_graph(col, years):
    filtered = df[(df.index.year >= years[0]) & (df.index.year <= years[1])]
    fig = go.Figure()
    
    # Usamos el Verde Menta para el precio y Dorado para los retornos
    line_color = colors['accent'] if col == 'Close' else colors['luxury']
    
    fig.add_trace(go.Scatter(x=filtered.index, y=filtered[col], line=dict(color=line_color, width=2)))
    
    fig.update_layout(
        title=f"Evolución de {col} ({years[0]}-{years[1]})",
        paper_bgcolor='rgba(0,0,0,0)', # Transparente para que tome el del contenedor
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=colors['text']),
        xaxis=dict(gridcolor='#223e3b'), # Líneas de división sutiles
        yaxis=dict(gridcolor='#223e3b'),
        template="plotly_dark"
    )
    return fig

if __name__ == '__main__':
    app.run(debug=True)