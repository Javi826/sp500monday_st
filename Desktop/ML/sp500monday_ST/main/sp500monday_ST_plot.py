#Created on Sat Dec 23 14:39:08 2023

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import tri
import pandas as pd
import os


current_directory = os.getcwd()
carpeta_principal = 'outputs'
ruta_salir_main = os.path.dirname(current_directory)
carpeta_briefs = os.path.join(ruta_salir_main,carpeta_principal, 'briefs')



# Lee el DataFrame df_brief desde el archivo df_brief.xlsx
ruta_df_brief = os.path.join(carpeta_briefs, 'df_brief.xlsx')
df_brief = pd.read_excel(ruta_df_brief)


# Extrae las columnas necesarias del DataFrame df_brief
x_values = df_brief['bf_num_dias_retroceder']
y_values = df_brief['bf_max_dias_ejecucion_venta']
z_values = df_brief['bf_%renta']

# Imprime el valor máximo de z_values
max_z_value = z_values.max()
print(f"El valor máximo de z_values es: {max_z_value}")

# Crea una figura 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Triangula los puntos para crear una superficie
triang = tri.Triangulation(x_values, y_values)

# Crea la superficie 3D
surf = ax.plot_trisurf(triang, z_values, cmap='viridis', edgecolor='k', linewidth=0.2)

# Etiquetas de los ejes
ax.set_xlabel('n_compra')
ax.set_ylabel('m_venta')
ax.set_zlabel('%_Operación')

# Añade una barra de color para la superficie
fig.colorbar(surf, ax=ax, shrink=0.6, aspect=5)

# Título de la gráfica
plt.title('Gráfico 3D: f(x, y) = z')

fig.set_size_inches(4, 4)  # Ancho: 10 pulgadas, Alto: 8 pulgadas
plt.show()