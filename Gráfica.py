import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Cargar ambos datasets
df_anterior = pd.read_csv(
    'Demandarealnacional1abe0fa4-efa6-405c-89c3-623ddbd18800.csv'
)
df_nuevo = pd.read_csv(
    'Demandarealnacional51c92a73-e56c-4be4-ab44-b664f54bf72d.csv'
)

# 2. Unir (concatenar) los dos DataFrames
df_combinado = pd.concat([df_anterior, df_nuevo], ignore_index=True)

# 3. Eliminar posibles filas duplicadas (buena práctica)
df_combinado = df_combinado.drop_duplicates()

# 4. Convertir FechaHora a datetime
df_combinado['FechaHora'] = pd.to_datetime(df_combinado['FechaHora'])

# 5. Agrupar por hora y sumar la demanda total
df_total = df_combinado.groupby('FechaHora')['Valor'].sum().reset_index()

# 6. Ordenar cronológicamente por fecha
df_total = df_total.sort_values('FechaHora')

# 7. Graficar con Seaborn
sns.set_theme(style='whitegrid')
plt.figure(figsize=(13, 5))

sns.lineplot(data=df_total, x='FechaHora', y='Valor', marker='o', linewidth=1.5)

plt.title(
    'Demanda Real Nacional Total por Hora (5 Días - Mercado No Regulado)',
    fontsize=14,
)
plt.xlabel('Fecha y Hora', fontsize=12)
plt.ylabel('Valor Total (kWh)', fontsize=12)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()