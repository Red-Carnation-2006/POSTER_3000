import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 1. Cargar el dataset unificado
df = pd.read_csv('Demandarealnacionalc45f74b4-54d1-4812-9f78-e07dce7e5f6e.csv')

# 2. Filtrar por una única versión de liquidación (TX2) para evitar duplicados en septiembre
df_filtrado = df[df['Version'] == 'TXF'].copy()

# 3. Convertir FechaHora a datetime
df_filtrado['FechaHora'] = pd.to_datetime(df_filtrado['FechaHora'])

# 4. Agrupar la demanda total por hora
df_total = (
    df_filtrado.groupby('FechaHora')['Valor']
    .sum()
    .reset_index()
    .sort_values('FechaHora')
)

# 5. Configurar el gráfico con Seaborn
sns.set_theme(style='whitegrid')
fig, ax = plt.subplots(figsize=(13, 5))

sns.lineplot(
    data=df_total, x='FechaHora', y='Valor', color='#1f77b4', linewidth=1.5
)

# Configurar formato de fechas en el eje X para 31 días
ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))

plt.title(
    'Demanda Real Nacional por Hora (10 Sep - 10 Oct 2026 | Versión TX2)',
    fontsize=13,
    fontweight='bold',
)
plt.xlabel('Fecha', fontsize=11)
plt.ylabel('Valor Total (kWh)', fontsize=11)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# 2. Crear la figura para ubicar ambos gráficos
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# 3. Graficar la FAC (ACF)
plot_acf(
    df_total['Valor'],
    lags=48,
    ax=ax1,
    title='Función de Autocorrelación (FAC / ACF)',
)
ax1.set_xlabel('Rezagos / Lags (Horas)')
ax1.set_ylabel('Autocorrelación')

# 4. Graficar la FACP (PACF)
plot_pacf(
    df_total['Valor'],
    lags=48,
    ax=ax2,
    title='Función de Autocorrelación Parcial (FACP / PACF)',
    method='ywm',
)
ax2.set_xlabel('Rezagos / Lags (Horas)')
ax2.set_ylabel('Autocorrelación Parcial')

plt.tight_layout()
plt.show()