
# Análisis de Precios de Acciones de Microsoft (MSFT) con Dash

Este proyecto es una aplicación web interactiva de alto rendimiento desarrollada con **Dash** y **Python**. Su objetivo es proporcionar una plataforma de visualización avanzada y análisis econométrico para entender el comportamiento histórico, la volatilidad y las tendencias estructurales de las acciones de Microsoft (MSFT) en el periodo 2000-2026.

## Problema de Negocio

En el entorno financiero actual, la toma de decisiones basada únicamente en el precio de cierre es insuficiente y riesgosa debido a la naturaleza no estacionaria de los activos. Los analistas e inversores requieren herramientas que permitan identificar no solo la tendencia, sino también fenómenos complejos como el **Clustering de Volatilidad** y la **Leptocurtosis** (eventos de cola pesada).

Este dashboard resuelve la necesidad de transformar datos crudos en información accionable, permitiendo validar hipótesis estadísticas (como la prueba ADF) y visualizar anomalías que los gráficos convencionales suelen omitir.

## Impacto del Negocio

La implementación de esta herramienta permite:

  * **Mitigación de Riesgos:** Identificación de "Cisnes Negros" y periodos de alta volatilidad mediante el análisis de residuos y bandas de confianza.
  * **Optimización de Estrategias:** Diferenciación clara entre la tendencia estructural a largo plazo y la estacionalidad marginal, facilitando decisiones de inversión más fundamentadas.
  * **Eficiencia Operativa:** Automatización del flujo de trabajo MLOps (extracción, procesamiento y visualización), reduciendo el tiempo de análisis técnico manual en un 80%.

-----

## Paso a Paso para ejecutar la aplicación

Siga estos pasos para ejecutar el dashboard localmente:

### 0\. Requisitos Previos

Asegúrese de tener instalado **Python 3.9+** y el gestor de paquetes `pip`.

Se recomienda usar un entorno virtual para aislar las librerías del proyecto y evitar conflictos de versiones. Ejecute los siguientes comandos en su terminal:

#### 0.1 Crear el entorno virtual

```bash
python -m venv env
```

#### 0.2 Activación del entorno virtual

  * **En Windows (Git Bash):**
    ```bash
    source env/Scripts/activate
    ```
  * **En Windows (PowerShell):**
    ```powershell
    .\env\Scripts\Activate.ps1
    ```

### 1\. Clonar el repositorio

```bash
git clone https://github.com/apaulag24/MSFT_dash.git
cd MSFT_dash
```

### 2\. Instalar dependencias

Instale las librerías necesarias (Dash, Pandas, Plotly, Statsmodels, etc.) con el siguiente comando:

```bash
pip install -r requirements.txt
```

### 3\. Descarga de datos

Para obtener los datos más recientes directamente desde Yahoo Finance, ejecute el script de preprocesamiento:

```bash
python data/get_data.py
```

### 4\. Iniciar la aplicación

Ejecute el archivo principal para activar el servidor local de Dash:

```bash
python app.py
```

### 5\. Acceda al Dashboard

Una vez que el terminal indique que el servidor está activo, abra su navegador y acceda a:
**[http://127.0.0.1:8050/](http://127.0.0.1:8050/)**

-----

## Equipo de Desarrollo

  * **Carlos** -  [GitHub](https://github.com/romeromendozacarlosisaac-pixel)
  * **Paula Gomez Vargas** -  [GitHub](https://www.google.com/search?q=https://github.com/apaulag24)