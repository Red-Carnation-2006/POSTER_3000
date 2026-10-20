import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 1. Cargar y preparar la serie de tiempo
df = pd.read_csv('Demandarealnacionalc45f74b4-54d1-4812-9f78-e07dce7e5f6e.csv')
df_tx2 = df[df['Version'] == 'TX2'].copy()
df_tx2['FechaHora'] = pd.to_datetime(df_tx2['FechaHora'])

df_total = (
    df_tx2.groupby('FechaHora')['Valor']
    .sum()
    .reset_index()
    .sort_values('FechaHora')
)
y = df_total.set_index('FechaHora')['Valor']
y_log = np.log(y)

# 2. Aplicar las diferenciaciones
diferenciaciones = {
    '1. Diferencia Ordinaria (d=1)': y.diff(1).dropna(),
    '2. Diferencia Estacional (D=1, s=24)': y.diff(24).dropna(),
    '3. Diferencia Doble (d=1, D=1, s=24)': y.diff(1).diff(24).dropna(),
    '4. Log + Dif. Estacional (ln(Y_t), D=1)': y_log.diff(24).dropna(),
    '5. Log + Dif. Doble (ln(Y_t), d=1, D=1)': y_log.diff(1).diff(24).dropna(),
}

# 3. Función para ejecutar y formatear el Test ADF
print('=== PRUEBA DICKEY-FULLER EN SERIES DIFERENCIADAS ===\n')

for nombre, serie_diff in diferenciaciones.items():
    resultado = adfuller(serie_diff)
    print(f'--- {nombre} ---')
    print(f'Estadístico ADF : {resultado[0]:.4f}')
    print(f'p-valor         : {resultado[1]:.4e}')
    print(f'Valor crítico 1%: {resultado[4]["1%"]:.4f}')
    print(f'Valor crítico 5%: {resultado[4]["5%"]:.4f}')

    if resultado[1] < 0.05:
        print('Conclusión      : Se rechaza H0 --> La serie ES ESTACIONARIA')
    else:
        print(
            'Conclusión      : No se rechaza H0 --> La serie NO es'
            ' estacionaria'
        )
    print('-' * 55)


#=========Graficación de Serie Transformada=============

# 2. Aplicar la diferencia estacional (D=1, s=24)
y_diff24 = y.diff(24).dropna()

sns.set_theme(style='whitegrid')

# =============================================================
# IMAGEN 1: Gráfico de la Serie Diferenciada
# =============================================================
plt.figure(figsize=(13, 5))
plt.plot(y_diff24.index, y_diff24.values, color='#2ca02c', linewidth=0.8)
plt.title(
    r'Serie con Diferencia Estacional ($D=1, s=24$: $\Delta_{24} Y_t = Y_t -'
    r' Y_{t-24}$)',
    fontsize=13,
    fontweight='bold',
)
plt.ylabel(r'$\Delta_{24} Y_t$ (kWh)', fontsize=11)
plt.xlabel('Fecha', fontsize=11)

plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('grafico_serie_diferenciada.png', dpi=300)
plt.show()  # Muestra la primera imagen de forma independiente

# =============================================================
# IMAGEN 2: Gráficos de Autocorrelación (FAC y FACP)
# =============================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Panel FAC
plot_acf(
    y_diff24,
    lags=48,
    ax=ax1,
    title='Función de Autocorrelación (FAC / ACF)',
    color='#1f77b4',
    vlines_kwargs={'colors': '#1f77b4'},
)
ax1.set_xlabel('Rezagos / Lags (Horas)', fontsize=10)
ax1.set_ylabel('Autocorrelación', fontsize=10)

# Panel FACP
plot_pacf(
    y_diff24,
    lags=48,
    ax=ax2,
    title='Función de Autocorrelación Parcial (FACP / PACF)',
    method='ywm',
    color='#2ca02c',
    vlines_kwargs={'colors': '#2ca02c'},
)
ax2.set_xlabel('Rezagos / Lags (Horas)', fontsize=10)
ax2.set_ylabel('Autocorrelación Parcial', fontsize=10)

plt.tight_layout()
plt.savefig('grafico_fac_facp_diferenciada.png', dpi=300)
plt.show()