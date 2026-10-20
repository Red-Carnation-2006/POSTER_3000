import warnings
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Ocultar advertencias secundarias de optimización y convergencia
warnings.filterwarnings('ignore')

# 1. Cargar el dataset
df = pd.read_csv('Demandarealnacionalc45f74b4-54d1-4812-9f78-e07dce7e5f6e.csv')
df['FechaHora'] = pd.to_datetime(df['FechaHora'])

# 2. Filtrar por la versión de facturación (TX2 abarca el periodo completo de 744 horas)
#    Nota: Si deseas la versión más avanzada disponible por fecha (TXF > TXR > TX2),
#    puedes seleccionar TX2 o filtrar dinámicamente.
df_version = df[df['Version'] == 'TX2']

# Agrupar la demanda total por hora y fijar la frecuencia horaria ('H')
df_total = (
    df_version.groupby('FechaHora')['Valor']
    .sum()
    .reset_index()
    .sort_values('FechaHora')
)

# Establecer índice temporal con frecuencia horaria explícita
y = df_total.set_index('FechaHora')['Valor'].asfreq('H')

# 3. Aplicar la diferencia estacional (D=1, s=24)
y_diff24 = y.diff(24).dropna()

# 4. Definir los modelos AR y ARMA a evaluar
modelos = {
    'AR(1)': (1, 0, 0),
    'AR(2)': (2, 0, 0),
    'ARMA(2,1)': (2, 0, 1),
    'ARMA(2,2)': (2, 0, 2),
    'ARMA(3,2)': (3, 0, 2),
}

resultados = []

# 5. Estimación de cada modelo y extracción de métricas AIC / BIC
for nombre, orden in modelos.items():
    try:
        modelo = ARIMA(y_diff24, order=orden)
        modelo_ajustado = modelo.fit()

        resultados.append({
            'Modelo': nombre,
            'Orden (p,d,q)': orden,
            'AIC': modelo_ajustado.aic,
            'BIC': modelo_ajustado.bic,
            'Log-Likelihood': modelo_ajustado.llf,
        })
    except Exception as e:
        print(f'Error al estimar el modelo {nombre}: {e}')

# 6. Organizar y mostrar la tabla ordenada por AIC
df_comparacion = pd.DataFrame(resultados)
df_comparacion = df_comparacion.sort_values('AIC').reset_index(drop=True)

print('=== COMPARACIÓN DE CRITERIOS DE INFORMACIÓN (AIC / BIC) ===')
print('Serie con Diferenciación Estacional (D=1, s=24)\n')
print(
    df_comparacion.to_string(
        index=False,
        formatters={
            'AIC': '{:,.2f}'.format,
            'BIC': '{:,.2f}'.format,
            'Log-Likelihood': '{:,.2f}'.format,
        },
    )
)