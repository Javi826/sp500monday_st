# modules/mod_dtset_clean.py
import pandas as pd
import os
import time

def mod_dtset_clean(df_data):
    
    # restart dataframe jic
    restart_dataframes = True  
    # rstart or define main DataFrame 
    if 'df_data_clean' in locals() and restart_dataframes:del df_data_clean  # delete dataframe if exits
    
    print(f'START MODUL mod_dtset_clean')
     
    df_data_clean = df_data
    
    def day_week(df_data_clean):
        
        #df_data_clean = df_data.copy()
        # column with dates
        date_column = 'date' 
        # Asegurarse de que la columna de fechas tenga solo la parte de la fecha (sin la hora)
        df_data_clean[date_column] = pd.to_datetime(df_data_clean[date_column]).dt.strftime('%Y-%m-%d')
    
        # ensuring date_column with date format
        df_data_clean[date_column] = pd.to_datetime(df_data_clean[date_column])
    
        # Cambia la configuración regional para que los nombres de los días se muestren en español
        df_data_clean['day_week'] = df_data_clean[date_column].dt.strftime('%A')
        
        # add new column fecha_formato with the year from date_column
        df_data_clean['fecha_formato'] = df_data_clean[date_column].dt.year
        
        return df_data_clean
    

    # update df_date_clean
    df_data_clean = day_week(df_data_clean)
    print("Step 01 OK: day_week") 

   
    def sort_columns(df_data_clean):
        # Mapeo de columnas
        desired_column_mapping = {
            'date': 'fecha',
            'day_week': 'dia_semana_dia',
            'close': 'ultimo_precio',
            'open': 'apertura_dia',
            'high': 'maximo_dia',
            'low': 'minimo_dia',
            'fecha_formato': 'fecha_formato'  # Agregada la columna fecha_formato
        }
    
        # Renombra las columnas según el mapeo deseado
        df_data_clean = df_data_clean.rename(columns=desired_column_mapping)
    
        # Elimina las columnas que no están en el mapeo
        columns_to_keep = set(desired_column_mapping.values())
        columns_to_drop = set(df_data_clean.columns) - columns_to_keep
        df_data_clean = df_data_clean.drop(columns=columns_to_drop, errors='ignore')
    
        # Reordena las columnas según el orden deseado
        desired_column_order = ['fecha', 'fecha_formato', 'dia_semana_dia', 'ultimo_precio', 'apertura_dia', 'maximo_dia', 'minimo_dia']
        df_data_clean = df_data_clean[desired_column_order]
    
        return df_data_clean
    
    # udpdate df_data_clean
    df_data_clean = sort_columns(df_data_clean)
    
    print("Step 02 OK: sort_columns")
    
    def rounding_data(df_data_clean):

        columns_to_round = ['apertura_dia', 'maximo_dia', 'minimo_dia', 'ultimo_precio']
        
        # format float
        df_data_clean[columns_to_round] = df_data_clean[columns_to_round].astype(float)
        #df_data_clean['day_week'] = df_data_clean['day_week'].astype(int)
        
        #format rounding 
        for column in columns_to_round:
          if column in df_data_clean.columns:
              df_data_clean[column] = df_data_clean[column].round(2)
        
        #for column in columns_to_round:
            #if column in df_data_clean.columns:
                #df_data_clean[column] = df_data_clean[column].apply(lambda x: '{:.2f}'.format(x))
        
            
        return df_data_clean
    
    # udpdate df_data_clean
    df_data_clean = rounding_data(df_data_clean)
    
    print("Step 03 OK: rounding_data") 
  
  
       
    path_base = r"C:\Users\jlahoz\Desktop\tograms\sp500monday_ST"
    path_destination = "inputs\historicyh"
    path_save = os.path.join(path_base, path_destination, "df_data_cleanSPX.xlsx")
    # creating folder jic
    if not os.path.exists(os.path.join(path_base, path_destination)):
        os.makedirs(os.path.join(path_base, path_destination))
    
    df_data_clean.to_excel(path_save, index=False)

    print(f"DataFrame saved in: {path_save}")
    print(f'END MODUL mod_dtset_clean\n')
    
    return df_data_clean