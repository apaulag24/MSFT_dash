import yfinance as yf
import pandas as pd
import os

if not os.path.exists('data'):
    os.makedirs('data')

print("Descargando datos...")
df = yf.download('MSFT', start="2000-01-01")

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)

df.to_csv("data/MSFT.csv", index=True)
print("¡Completado! Archivo guardado en data/msft_data.csv")