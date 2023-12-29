# -*- coding: utf-8 -*-
#Created on Sat Dec 23 14:21:22 2023
import os
import pandas as pd
import numpy as np
import warnings
import time
import yfinance as yf
import sys
from modules.mod_dtset_clean import mod_dtset_clean

#TIMING
inicio_tiempo = time.time()

#IGNORE WARNINGS
warnings.filterwarnings("ignore")

#VISUALIZATION PRINTS
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

# PATHS 
path_base = r"C:\Users\jlaho\Desktop\Programas\Spyder\sp500monday_ST"
folder = "inputs\historicyh"
archivo = "sp500_datayh.csv"
path_absolut = os.path.join(path_base, folder, archivo)
current_directory = os.getcwd()
print("Ruta actual:", current_directory) 

# INDEX SYMBOL
symbol = "^GSPC"

# DATA DATE PERIOD
start_date = "1950-01-03"
end_date = "2023-12-01"

# YAHOO CALL
sp500_data = yf.download(symbol, start=start_date, end=end_date)

# DIRECTORY YAHOO CSV file
csv_folder = r"C:\Users\jlaho\Desktop\Programas\Spyder\sp500monday_ST\inputs\historicyh"

# Ensure the folder exists or create it if it doesn't
if not os.path.exists(csv_folder): os.makedirs(csv_folder)

# YAHOO FILE SAVING
csv_file_path = os.path.join(csv_folder, "sp500_datayh.csv")
sp500_data.to_csv(csv_file_path)

# YAHOO FILE READING
df_data = pd.read_csv(csv_file_path, header=None, skiprows=1, names=['date','open','high','low','close','adj_close','volume'])
print("Step 01 OK: reading file")

#DATA SET CLEANING
df_clean=mod_dtset_clean(df_data)

#sys.exit()

      
# DATA YAHOO CLEANING DOWNLOADED FOLDER
folder = "inputs\historicyh"
archivo = "df_data_cleanSPX.xlsx"
#
ruta_salir_main = os.path.dirname(current_directory)
ruta_absoluta = os.path.join(ruta_salir_main, folder, archivo)


# Especifica la carpeta principal
carpeta_principal = 'outputs'


# Verifica si la carpeta principal existe, y si no, la crea
if not os.path.exists(os.path.join(ruta_salir_main, carpeta_principal)):
    os.makedirs(os.path.join(ruta_salir_main, carpeta_principal))

# Carpetas dentro de la carpeta principal
carpeta_salidas = os.path.join(ruta_salir_main, carpeta_principal, 'salidas')
if not os.path.exists(carpeta_salidas):
    os.makedirs(carpeta_salidas)

# Carpetas dentro de la carpeta principal
carpeta_operaciones = os.path.join(ruta_salir_main, carpeta_principal, 'operaciones')
if not os.path.exists(carpeta_operaciones):
    os.makedirs(carpeta_operaciones)

# Carpetas dentro de la carpeta principal
carpeta_rentabilidades = os.path.join(ruta_salir_main, carpeta_principal, 'rentabilidades')
if not os.path.exists(carpeta_rentabilidades):
    os.makedirs(carpeta_rentabilidades)

# Carpetas dentro de la carpeta principal
carpeta_briefs = os.path.join(ruta_salir_main, carpeta_principal, 'briefs')
if not os.path.exists(carpeta_briefs):
    os.makedirs(carpeta_briefs)

# Verificar si se deben reiniciar los DataFrames al principio
reiniciar_dataframes = True  # Puedes ajustar esto según tus necesidades

# Reiniciar o definir el DataFrame principal
if 'dataframe' in locals() and reiniciar_dataframes:
    del dataframe  # Eliminar el DataFrame existente si existe
dataframe = pd.DataFrame()

# Definir variables
num_dias_retroceder_values = [1,2,3,4,5,6,7,8,9,10,11,12]
max_dias_ejecucion_venta_values = [1,2,3,4,5,6,7,8,9,10,11,12]
num_dias_retroceder_values = [1,2]
max_dias_ejecucion_venta_values = [1,2]
comision = 0
annus = 23
evalua_lunes = 5

df_brief_columns = ['bf_annus', 'bf_num_dias_retroceder', 'bf_max_dias_ejecucion_venta',
                    'bf_agrupa_operacion', 'bf_dias_ejecuta_venta', 'bf_%operacion','bf_%renta']


df_brief_data = []
#-----------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------
for num_dias_retroceder in num_dias_retroceder_values:
    for max_dias_ejecucion_venta in max_dias_ejecucion_venta_values:
        print(f"\nINICIO Iteración: {num_dias_retroceder} - {max_dias_ejecucion_venta}")
        
        # Lee el archivo excel en un DataFrame de pandas
        dataframe = pd.read_excel(ruta_absoluta)

        
        # Añade una nueva columna llamada 'iden' con numeración de cada registro
        dataframe['iden'] = range(1, len(dataframe) + 1)

        # Formatea la columna 'iden' para que tenga 5 dígitos
        dataframe['iden'] = dataframe['iden'].apply(lambda x: f'{x:05d}')


        # Especifica la condición para seleccionar las filas
        condicion_dia_lunes=dataframe['dia_semana_dia'] == 'Monday'

        # Agrega una nueva columna al DataFrame original para indicar si la condición se cumple
        dataframe['condicion_dia_lunes'] = condicion_dia_lunes

        # Agrega una nueva columna adicional que compara el valor de la columna 'ultimo_precio'
        # con los registros de 'ultimo_precio' en las filas posteriores hasta 'num_dias_retroceder'
        condicion_compra_dias_bajando = None  # Inicializamos con None para la primera iteración

        for i in range(1, num_dias_retroceder + 1):
            condicion_actual = dataframe['ultimo_precio'] <= dataframe['ultimo_precio'].shift(i)
            if  condicion_compra_dias_bajando is None:
                condicion_compra_dias_bajando = condicion_actual
            else:
                condicion_compra_dias_bajando &= condicion_actual

        dataframe['condicion_compra_dias_bajando'] = condicion_compra_dias_bajando

        # Asigna True a las filas donde ambas condiciones son verdaderas
        dataframe['iden_lunes_compra'] = condicion_compra_dias_bajando & condicion_dia_lunes

        dataframe['iden_lunes_compra'] = np.where(dataframe['iden_lunes_compra'],'Pot Lunes Compra','')

        # Agrega una nueva columna llamada 'cont_pot' con valor 1 si 'iden_lunes_compra' es igual a 'Pot Lunes Compra' y 0 de lo contrario
        dataframe['cont_pot'] = np.where(dataframe['iden_lunes_compra'] == 'Pot Lunes Compra', 1, 0)

        # Agrega una nueva columna llamada 'sum_cont_pot'
        dataframe['sum_cont_pot'] = 0

        # Itera sobre el DataFrame
        for i in range(len(dataframe)):
            if dataframe.at[i, 'cont_pot'] == 1:
                # Suma los valores de 'cont_pot' de los registros anteriores hasta un máximo de 'max_dias_ejecucion_venta'
                sum_cont_pot = dataframe.loc[max(0, i - max_dias_ejecucion_venta):i - 1, 'cont_pot'].sum()
                dataframe.at[i, 'sum_cont_pot'] = sum_cont_pot

        # Agrega una columna adicional 'contador_venta' con la lógica especificada
        dataframe['cont_venta'] = np.where(
            (dataframe['ultimo_precio'] > dataframe['maximo_dia'].shift(1)),
           1,
           0
        )

        # Agrega una nueva columna llamada 'sum_cont_venta'
        dataframe['sum_cont_venta'] = 0

        # Itera sobre el DataFrame
        for i in range(len(dataframe)):
            if dataframe.at[i, 'cont_pot'] == 1:
                # Suma los valores de 'cont_venta' de los registros anteriores
                # hasta un máximo de 'evalua_lunes' en lugar de 'max_dias_ejecucion_venta'
                sum_cont_venta = dataframe.loc[max(0, i - evalua_lunes):i, 'cont_venta'].sum()
                dataframe.at[i, 'sum_cont_venta'] = sum_cont_venta
                
        print("Paso 01 OK")    
#-----------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------
        # Agrega una nueva columna llamada 'ejecuta_lunes' con las condiciones dadas y secuencia numérica
        dataframe['ejecuta_lunes'] = '' 
        contador_secuencia = 1

        for i in range(len(dataframe)):
            if (dataframe.at[i, 'cont_pot'] == 1) and (dataframe.at[i, 'sum_cont_pot'] == 0):
                dataframe.at[i, 'ejecuta_lunes'] = f'Si Compra Lunes_1_{str(contador_secuencia).zfill(3)}'
                contador_secuencia += 1
            elif (dataframe.at[i, 'cont_pot'] == 1) and (dataframe.at[i, 'sum_cont_pot'] != 0) and (dataframe.at[i, 'sum_cont_venta'] != 0):
                dataframe.at[i, 'ejecuta_lunes'] = f'Si Compra Lunes_2_{str(contador_secuencia).zfill(3)}'
                contador_secuencia += 1
            elif (dataframe.at[i, 'cont_pot'] == 1) and (dataframe.at[i, 'sum_cont_pot'] != 0) and (dataframe.at[i, 'sum_cont_venta'] == 0):
                dataframe.at[i, 'ejecuta_lunes'] = 'No Compra Lunes'


        # Agrega una nueva columna llamada 'cont_ejecuta_lunes'
        dataframe['cont_ejecuta_lunes'] = np.where(dataframe['ejecuta_lunes'].str.startswith('Si Compra Lunes'), 1, 0)


        print("Paso 02 OK")
#-----------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------        
                # Inicializa el número de secuencia
        contador_secuencia = 1

        # Agrega una nueva columna llamada 'ejecuta_venta'
        dataframe['ejecuta_venta'] = ''

        # Itera sobre todos los índices
        for indice_ejecuta_lunes in dataframe[dataframe['cont_ejecuta_lunes'] == 1].index:
            # Inicializa el número de secuencia aquí para cada compra
            num_secuencia = int(dataframe['ejecuta_lunes'].iloc[indice_ejecuta_lunes].split('_')[-1])
            encontrado = False
            # Resto del código para la venta
            for i in range(1, min(max_dias_ejecucion_venta + 1, len(dataframe) - indice_ejecuta_lunes)):
                if dataframe['cont_venta'].iloc[indice_ejecuta_lunes + i] == 1:
                    # Actualiza el DataFrame para ese día
                    dataframe.at[indice_ejecuta_lunes + i, 'ejecuta_venta'] = f'venta_{str(num_secuencia).zfill(3)}_ejecutada_dia_{str(i).zfill(3)}'
                    encontrado = True
                    break

            # Si no se encontró venta dentro del límite, asigna un valor específico al día máximo
            if not encontrado:
                max_dia_ejecucion_venta = min(max_dias_ejecucion_venta, len(dataframe) - indice_ejecuta_lunes - 1)
                dataframe.at[indice_ejecuta_lunes + max_dia_ejecucion_venta, 'ejecuta_venta'] = f'venta_{str(num_secuencia).zfill(3)}_ejecutada_dia_{str(max_dias_ejecucion_venta).zfill(3)}'



        # Agrega una nueva columna llamada 'cont_ejecuta_lunes'
        dataframe['cont_ejecuta_venta'] = np.where(dataframe['ejecuta_venta'].str.startswith('venta_'), 1, 0)

        print("Paso 03 OK")
#-----------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------        
        # Agrega una nueva columna llamada 'carga_operacion'
        dataframe['carga_operacion'] = ''

        # Itera sobre todos los índices donde 'cont_ejecuta_lunes' es igual a 1
        for indice_ejecuta_lunes in dataframe[dataframe['cont_ejecuta_lunes'] == 1].index:
            # Obtiene el número de secuencia de la compra
            num_secuencia_compra = int(dataframe['ejecuta_lunes'].iloc[indice_ejecuta_lunes].split('_')[-1])

            # Asigna el valor correspondiente en la columna 'carga_operacion' para la compra
            dataframe.at[indice_ejecuta_lunes, 'carga_operacion'] = f'Operacion_C_{str(num_secuencia_compra).zfill(3)}'

        # Itera sobre todos los índices donde 'cont_ejecuta_venta' es igual a 1
        for indice_ejecuta_venta in dataframe[dataframe['cont_ejecuta_venta'] == 1].index:
            # Obtiene el número de secuencia de la venta
            num_secuencia_venta = int(dataframe['ejecuta_venta'].iloc[indice_ejecuta_venta].split('_')[1])

            # Obtiene el número de días que tarda en ejecutarse la venta
            dias_ejecucion_venta = int(dataframe['ejecuta_venta'].iloc[indice_ejecuta_venta].split('_')[-1])

            # Asigna el valor correspondiente en la columna 'carga_operacion' para la venta
            dataframe.at[indice_ejecuta_venta, 'carga_operacion'] = f'Operacion_V_{str(num_secuencia_venta).zfill(3)}_{str(dias_ejecucion_venta).zfill(3)}'

        print("Paso 04 OK")
#-----------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------        
                # Cambia el nombre de la nueva columna de 'ejecuta_operacion' a 'agrupa_operacion'
        dataframe['agrupa_operacion'] = ''

        # Itera sobre el DataFrame
        for i in range(len(dataframe) - 1):
            # Verifica si el registro actual en 'carga_operacion' comienza con 'Operacion_C_'
            if dataframe['carga_operacion'].iloc[i].startswith('Operacion_C_'):
                # Encuentra el siguiente índice donde 'carga_operacion' está informado
                siguiente_indice = i + 1
                while siguiente_indice < len(dataframe) and not dataframe['carga_operacion'].iloc[siguiente_indice]:
                    siguiente_indice += 1

                # Verifica si encontró un siguiente registro informado
                if siguiente_indice < len(dataframe):
                    # Obtén los últimos tres dígitos del siguiente registro en 'carga_operacion' separados por '_'
                    ultimos_tres_digitos = dataframe['carga_operacion'].iloc[siguiente_indice].split('_')[-1]

                    # Combina el registro actual con los últimos tres dígitos del siguiente registro en 'carga_operacion'
                    nuevo_valor = f"{dataframe['carga_operacion'].iloc[i]}_{ultimos_tres_digitos}"

                    # Asigna el nuevo valor a la columna 'agrupa_operacion'
                    dataframe.at[i, 'agrupa_operacion'] = nuevo_valor
                else:
                    # Si no hay un siguiente registro informado, el registro es el mismo
                    dataframe.at[i, 'agrupa_operacion'] = dataframe['carga_operacion'].iloc[i]
            elif dataframe['carga_operacion'].iloc[i].startswith('Operacion_V_'):
                # Para los registros que comienzan con 'Operacion_V_', el registro es el mismo
                dataframe.at[i, 'agrupa_operacion'] = dataframe['carga_operacion'].iloc[i]

        # El último registro de 'agrupa_operacion' será el mismo que el último registro de 'carga_operacion'
        dataframe.at[len(dataframe) - 1, 'agrupa_operacion'] = dataframe['carga_operacion'].iloc[len(dataframe) - 1]
      
        print("Paso 05 OK")
#-----------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------        
        # Especifica la ruta para guardar el nuevo archivo Excel con el nombre personalizado
        nombre_archivo_salida = f'df_salida_{annus}_{num_dias_retroceder:03d}_{max_dias_ejecucion_venta:03d}.xlsx'
        ruta_df_salida = os.path.join(carpeta_salidas, nombre_archivo_salida)

        # Guarda el DataFrame resultante en un nuevo archivo Excel con la hoja llamada "Master"
        dataframe.to_excel(ruta_df_salida, sheet_name='Master', index=False)

        #print(f"\nEl DataFrame se ha guardado en '{ruta_df_salida}' con la hoja 'Master'.")
        print("Paso 06 OK")
#-----------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------        
        # Define el nombre para el segundo archivo Excel con el DataFrame limpio
        nombre_archivo_salida_limpia = f'df_salida_limpia_{annus}_{num_dias_retroceder:03d}_{max_dias_ejecucion_venta:03d}.xlsx'
        ruta_df_salida_limpia = os.path.join(carpeta_salidas, nombre_archivo_salida_limpia)

        # Define las columnas seleccionadas y su orden
        columnas_seleccionadas = ['iden', 'fecha_formato', 'dia_semana_dia', 'ultimo_precio', 'apertura_dia', 'maximo_dia', 'ejecuta_lunes', 'ejecuta_venta', 'carga_operacion', 'agrupa_operacion']

        # Crea un nuevo DataFrame con las columnas seleccionadas
        dataframe_limpio = dataframe[columnas_seleccionadas]

        # Guarda el DataFrame limpio en un nuevo archivo Excel con el nombre personalizado
        dataframe_limpio.to_excel(ruta_df_salida_limpia, sheet_name='Master', index=False)

        
        print("Paso 07 OK")
#-----------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------        
        # Especifica la ruta para leer el archivo Excel con el nombre personalizado
        nombre_archivo_entrada = f'df_salida_{annus}_{num_dias_retroceder:03d}_{max_dias_ejecucion_venta:03d}.xlsx'
        ruta_df_salida = os.path.join(carpeta_salidas, nombre_archivo_entrada)

        # Lee el archivo DataFrameSalida.xlsx en un DataFrame de pandas
        df_salida = pd.read_excel(ruta_df_salida, sheet_name='Master')

        # Filtra solo los registros con información en la columna 'agrupa_operacion'
        df_operaciones = df_salida[df_salida['agrupa_operacion'].notnull()]

        # Especifica la ruta para leer el archivo Excel de operaciones con el nombre personalizado
        nombre_archivo_operaciones = f'df_operaciones_{annus}_{num_dias_retroceder:03d}_{max_dias_ejecucion_venta:03d}.xlsx'
        ruta_df_operaciones = os.path.join(carpeta_operaciones, nombre_archivo_operaciones)

        # Guarda los registros filtrados en un nuevo archivo Excel con la hoja llamada "Operaciones"
        df_operaciones[['iden', 'fecha_formato', 'ultimo_precio', 'agrupa_operacion']].to_excel(
            ruta_df_operaciones, sheet_name='Operaciones', index=False
        )

        
        print("Paso 08 OK")
#-----------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------        
        # Construye el nombre del archivo Excel basado en los parámetros
        nombre_archivo_entrada = f'df_operaciones_{annus}_{num_dias_retroceder:03d}_{max_dias_ejecucion_venta:03d}.xlsx'
        ruta_df_operaciones = os.path.join(carpeta_operaciones, nombre_archivo_entrada)

        # Lee el archivo Excel en un DataFrame de pandas
        df_operaciones = pd.read_excel(ruta_df_operaciones, sheet_name='Operaciones')

        # Filtra el DataFrame para operaciones que comienzan con 'Operacion_C_'
        df_rentabilidades = df_operaciones[df_operaciones['agrupa_operacion'].str.startswith('Operacion_C_')]
        df_rentabilidades['fecha_formato'] = df_operaciones['fecha_formato']
        df_rentabilidades['ultimo_precio_compra'] = df_operaciones['ultimo_precio']
        df_rentabilidades['ultimo_precio_venta'] = df_operaciones['ultimo_precio'].shift(-1)
        df_rentabilidades['dias_ejecuta_venta'] = df_rentabilidades['agrupa_operacion'].str[-3:]
        df_rentabilidades['dias_ejecuta_venta'] = df_rentabilidades['dias_ejecuta_venta'].astype(int)
        df_rentabilidades = df_rentabilidades[df_rentabilidades['agrupa_operacion'].str.startswith('Operacion_C_')]

        # Agrega una nueva columna 'delta_operacion'
        df_rentabilidades['delta_operacion'] = df_rentabilidades['ultimo_precio_venta'] - df_rentabilidades['ultimo_precio_compra']

        df_rentabilidades['%_operacion'] = ((df_rentabilidades['delta_operacion'] / df_rentabilidades['ultimo_precio_compra']) * 100).round(2)

 
        # Capital inicial
        capital_inicial = 10000
        df_rentabilidades['capital'] = 0.0  # Inicializa la columna 'capital'
        
        # Calcula la columna 'capital' usando la lógica descrita
        df_rentabilidades.at[df_rentabilidades.index[0], 'capital'] = capital_inicial * (1 + df_rentabilidades.at[df_rentabilidades.index[0], '%_operacion'] / 100)*(1-(comision/100))
        
        for i in range(1, len(df_rentabilidades)):
            df_rentabilidades.at[df_rentabilidades.index[i], 'capital'] = df_rentabilidades.at[df_rentabilidades.index[i - 1], 'capital'] * (1 + df_rentabilidades.at[df_rentabilidades.index[i], '%_operacion'] / 100)*(1-(comision/100))

        # Calcula la rentabilidad anualizada para cada operación en df_rentabilidades
        df_rentabilidades['rentabilidad_anualizada'] = (((df_rentabilidades['capital'] / capital_inicial) ** (1/annus) - 1) * 100).round(2)
        
        # Guarda los registros filtrados en un nuevo archivo Excel con la hoja llamada "Rentabilidades"
        ruta_df_rentabilidades = os.path.join(carpeta_rentabilidades, f'df_rentabilidades_{annus}_{num_dias_retroceder:03d}_{max_dias_ejecucion_venta:03d}.xlsx')
        df_rentabilidades.to_excel(ruta_df_rentabilidades, sheet_name='rentabilidades', index=False)

        
        print("Paso 09 OK")
#-----------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------- 
        # Lee el archivo Excel en un DataFrame de pandas
        df_operaciones = pd.read_excel(ruta_df_operaciones, sheet_name='Operaciones')
        
        # Filtra el DataFrame para operaciones que comienzan con 'Operacion_C_'
        df_rent_anio = df_operaciones[df_operaciones['agrupa_operacion'].str.startswith('Operacion_C_')]
        
        # Asigna los valores de las nuevas columnas
        df_rent_anio['fecha_formato'] = df_operaciones['fecha_formato']
        df_rent_anio['ultimo_precio_compra'] = df_operaciones['ultimo_precio']
        df_rent_anio['ultimo_precio_venta'] = df_operaciones['ultimo_precio'].shift(-1)
        df_rent_anio['dias_ejecuta_venta'] = df_rent_anio['agrupa_operacion'].str[-3:]
        df_rent_anio['dias_ejecuta_venta'] = df_rent_anio['dias_ejecuta_venta'].astype(int)
        df_rent_anio = df_rent_anio[df_rent_anio['agrupa_operacion'].str.startswith('Operacion_C_')]
        
        # Agrega una nueva columna 'delta_operacion'
        df_rent_anio['delta_operacion'] = df_rent_anio['ultimo_precio_venta'] - df_rent_anio['ultimo_precio_compra']
        
        df_rent_anio['%_operacion'] = ((df_rent_anio['delta_operacion'] / df_rent_anio['ultimo_precio_compra']) * 100).round(2)
        
        # Capital inicial
        capital_inicial = 10000
        
        # Calcula la columna 'capital' usando la lógica descrita, reiniciando en 10000 cada vez que cambia el año
        for i in range(len(df_rent_anio)):
            if i == 0 or df_rent_anio.at[df_rent_anio.index[i], 'fecha_formato'] != df_rent_anio.at[df_rent_anio.index[i - 1], 'fecha_formato']:
                # Utiliza el porcentaje de operación para la primera operación de cada año
                df_rent_anio.at[df_rent_anio.index[i], 'capital'] = capital_inicial * (1 + df_rent_anio.at[df_rent_anio.index[i], '%_operacion'] / 100)*(1-(comision/100))
            else:
                df_rent_anio.at[df_rent_anio.index[i], 'capital'] = df_rent_anio.at[df_rent_anio.index[i - 1], 'capital'] * (1 + df_rent_anio.at[df_rent_anio.index[i], '%_operacion'] / 100)*(1-(comision/100))
        
        # Calcula la rentabilidad anualizada para cada operación en df_rent_anio
        df_rent_anio['rentabilidad_anualizada'] = (((df_rent_anio['capital'] / capital_inicial) ** (1/1) - 1) * 100).round(2)
        
        # Guarda los resultados en un nuevo archivo Excel con la hoja llamada "Agrupado_Capital_10000"
        ruta_df_rent_anio = os.path.join(carpeta_rentabilidades, f'df_rent_anio_{annus}_{num_dias_retroceder:03d}_{max_dias_ejecucion_venta:03d}.xlsx')
        df_rent_anio.to_excel(ruta_df_rent_anio, sheet_name='Agrupado', index=False)
        
        print("Paso 10 OK")
#-----------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------- 

        # Agrupa df_rentabilidades por año
        df_rent_anio_grouped = df_rent_anio.groupby('fecha_formato')
        
        # Inicializa listas para cada columna del nuevo DataFrame
        fecha_formato_anio_list = []
        agrupa_operacion_anio_list = []
        dias_ejecuta_venta_anio_list = []
        porcentaje_operacion_anio_list = []
        capital_inicial_anio_list = []
        capital_final_anio_list = []
        rent_anualizada_anio_list = []
        
        # Itera sobre cada grupo
        for year, group in df_rent_anio_grouped:
            # Añade información a las listas
            fecha_formato_anio_list.append(year)
            agrupa_operacion_anio_list.append(len(group['agrupa_operacion']))
            dias_ejecuta_venta_anio_list.append(group['dias_ejecuta_venta'].mean())
            porcentaje_operacion_anio_list.append(group['%_operacion'].mean())
            capital_inicial_anio_list.append(capital_inicial)
            capital_final_anio_list.append(group['capital'].iloc[-1])
            rent_anualizada_anio_list.append(group['rentabilidad_anualizada'].iloc[-1])
        
        # Crea el nuevo DataFrame
        df_rent_anio_unic = pd.DataFrame({
            'fecha_formato_anio': fecha_formato_anio_list,
            'agrupa_operacion_anio': agrupa_operacion_anio_list,
            'dias_ejecuta_venta_anio': dias_ejecuta_venta_anio_list,
            '%_operacion_anio': porcentaje_operacion_anio_list,
            'capital_inicial_anio': capital_inicial_anio_list,
            'capital_final_anio': capital_final_anio_list,
            'rent_anualizada_anio': rent_anualizada_anio_list
        })
        print("Paso 11 OK")  

        # Especifica la ruta para guardar el nuevo archivo Excel
        ruta_df_rent_anio_unic = os.path.join(carpeta_rentabilidades, f'df_rent_anio_unic_{annus}_{num_dias_retroceder:03d}_{max_dias_ejecucion_venta:03d}.xlsx')
        
        # Guarda el DataFrame en el nuevo archivo Excel
        df_rent_anio_unic.to_excel(ruta_df_rent_anio_unic, sheet_name='Rent_Anio_Unic', index=False)

#-----------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------

        # Encuentra el último valor de la columna rentabilidad_anualizada
        ultimo_valor_rentabilidad_anualizada = df_rentabilidades['rentabilidad_anualizada'].iloc[-1]

        df_brief_data.append([
            annus,
            num_dias_retroceder,
            max_dias_ejecucion_venta,
            len(df_rentabilidades['agrupa_operacion'].unique()),
            df_rentabilidades['dias_ejecuta_venta'].mean(),
            df_rentabilidades['%_operacion'].mean(),
            ultimo_valor_rentabilidad_anualizada
        ])
    
        # Después de tus bucles, crea df_brief utilizando df_brief_data
        df_brief = pd.DataFrame(df_brief_data, columns=df_brief_columns)

        # Guarda df_brief en un archivo Excel
        ruta_df_brief = os.path.join(carpeta_briefs, 'df_brief.xlsx')
        df_brief.to_excel(ruta_df_brief, sheet_name='Brief', index=False)
        
        print("Paso 12 OK") 
        
        print(f"\nFINAL Iteración: {num_dias_retroceder} - {max_dias_ejecucion_venta}")
        # Guarda el tiempo de finalización
        fin_tiempo = time.time()
        
# Calcula el tiempo transcurrido
tiempo_transcurrido = round((fin_tiempo - inicio_tiempo)/60, 1)

print(f"El proceso tardó {tiempo_transcurrido} min en ejecutarse.")