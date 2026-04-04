# Análisis de Precios de Acciones de Microsoft (MSFT) con Dash

Este proyecto es una aplicación web interactiva desarrollada con **Dash** y **Python** para visualizar y analizar el comportamiento histórico de las acciones de Microsoft.

## Problema de Negocio
[Aquí explica por qué es importante analizar MSFT. Ejemplo: "Los inversores necesitan herramientas interactivas para visualizar la volatilidad y los retornos diarios de Microsoft en tiempo real para la toma de decisiones financieras."]

## Impacto del Negocio
[Aquí explica qué se logra con tu app. Ejemplo: "Este dashboard permite reducir la incertidumbre al visualizar tendencias históricas y calcular retornos, facilitando el análisis técnico para portafolios de inversión."]

## Cómo ejecutar la aplicación (Paso a Paso)

Sigue estos pasos para ejecutar el dashboard localmente:

### 0. Requisitos Previos
Asegúrate de tener instalado **Python 3.9+** y un gestor de paquetes como `pip`.

### 1. Clonar el repositorio
```bash
git clone  https://github.com/apaulag24/MSFT_dash.git
cd MSFT_dash
```

### 2. Instalar dependencias
Se recomienda usar un entorno virtual. Instale las librerías necesarias con:

```
pip install -r requirements.txt
```

### 3. Descarga de datos

Para obtener los datos mas recientes de Yahoo Finance, ejecuta el script de descarga:

```
python data/get_data.py
```

### 4. Iniciar la aplicación 

Ejecute el archivo principal para activar el servidor de dash:

```
python app.py
```

### 5. Acceda al Dashboard

Una vex el servidor se esté ejecuntando, abra el navegador y acceda a: 

```
http://127.0.0.1:8050/
```

Equipo de Desarrollo:

* Carlos - [linkedin]- [github]

* Paula Gomez Vargas - [linkedin]-[Git]
