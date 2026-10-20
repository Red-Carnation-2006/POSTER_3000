import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller

# 1. Cargar y preparar el dataset unificado
df = pd.read_csv('Demandarealnacionalc45f74b4-54d1-4812-9f78-e07dce7e5f6e.csv')

# Filtrar por versión TX2 para evitar registros duplicados
df_tx2 = df[df['Version'] == 'TX2'].copy()
df_tx2['FechaHora'] = pd.to_datetime(df_tx2['FechaHora'])

# Agrupar la demanda total por hora
df_total = (
    df_tx2.groupby('FechaHora')['Valor']
    .sum()
    .reset_index()
    .sort_values('FechaHora')
)
serie = df_total['Valor']

# -------------------------------------------------------------
# PRUEBA 1: Test de Dickey-Fuller Aumentado (ADF)
# -------------------------------------------------------------
print("=== PRUEBA DICKEY-FULLER AUMENTADA (ADF) ===")
resultado_adf = adfuller(serie)

print(f'Estadístico ADF: {resultado_adf[0]:.4f}')
print(f'p-valor: {resultado_adf[1]:.4f}')
print('Valores Críticos:')
for clave, valor in resultado_adf[4].items():
    print(f'   {clave}: {valor:.4f}')

# Interpretación automática ADF
if resultado_adf[1] < 0.05:
    print('--> Conclusión: Se rechaza H0. La serie ES ESTACIONARIA en media.')
else:
    print(
        '--> Conclusión: No se rechaza H0. La serie NO es estacionaria'
        ' (requiere diferenciación).'
    )

print('\n' + '=' * 45 + '\n')

# -------------------------------------------------------------
# PRUEBA 2: Test de Levene (Homocedasticidad diaria)
# -------------------------------------------------------------
print('=== TEST DE LEVENE (IGUALDAD DE VARIANZAS POR DÍA) ===')

# Agrupar las observaciones de 24 horas correspondientes a cada día
df_total['Fecha'] = df_total['FechaHora'].dt.date
grupos_diarios = [
    grupo['Valor'].values
    for _, grupo in df_total.groupby('Fecha')
    if len(grupo) == 24
]

estadistico_w, p_valor_levene = stats.levene(*grupos_diarios)

print(f'Estadístico W: {estadistico_w:.4f}')
print(f'p-valor: {p_valor_levene:.4e}')

# Interpretación automática Levene
if p_valor_levene < 0.05:
    print(
        '--> Conclusión: Se rechaza H0. La varianza NO ES CONSTANTE'
        ' (existe heterocedasticidad).'
    )
else:
    print('--> Conclusión: No se rechaza H0. Las varianzas son homogéneas.')