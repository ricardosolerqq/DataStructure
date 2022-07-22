# Databricks notebook source
# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

import pickle
import os
import pandas as pd
import numpy as np
import time
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pyspark.sql.types import DateType

# COMMAND ----------

# MAGIC %md
# MAGIC # <font color='blue'>IA - Feature Selection</font>
# MAGIC 
# MAGIC # <font color='blue'>Ferramenta de Criação de Variáveis</font>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importando os pacotes Python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inserindo hiperparâmetros do Algoritmo

# COMMAND ----------

## Parâmetros do Algoritmo

#Variável chave-primaria
chave = 'DOCUMENT'

#Caminho da base de dados
#caminho_base = "/dbfs/mnt/ml-prd/ml-data/propensaodeal/credigy/mongoDB_database/data_base_to_score_2022-02-08/"
#list_base = os.listdir(caminho_base)
caminho_base = "/dbfs/mnt/ml-prd/ml-data/propensaodeal/credigy/trusted/output_to_score_model"
#caminho_trusted_dbfs=caminho_base+os.listdir(caminho_base)[0]
#caminho_base=caminho_trusted_dbfs+'/'
#caminho_trusted=str.replace(caminho_trusted_dbfs,'/dbfs','')

#Nome da Base de Dados
#N_Base = max(list_base)
#dt_max = N_Base.split('.')[0]

#Separador
separador_ = ";"

#Decimal
decimal_ = "."

#caminho_trusted = '/mnt/ml-prd/ml-data/propensaodeal/credigy/mongoDB_database/data_base_to_score_2022-02-08'
#caminho_trusted_dbfs = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/credigy/mongoDB_database/data_base_to_score_2022-02-08'

pickle_path = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/credigy/pickle_model/'

outputpath = 'mnt/ml-prd/ml-data/propensaodeal/credigy/output/'
outputpath_dbfs = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/credigy/output/'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importação da Base de Dados

# COMMAND ----------

# DBTITLE 1,Deletar na versão final
import glob

all_files = glob.glob(caminho_base + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, sep=separador_, decimal=decimal_)
    li.append(df)

base_dados = pd.concat(li, axis=0, ignore_index=True)
base_dados['DT_NASC'] = base_dados['DT_NASC'].str[:10]
base_dados['DT_CADASTRO'] = base_dados['DT_CADASTRO'].str[:10]

base_dados.dropna(subset = ['ID_PESSOA'], inplace = True)
base_dados.loc[base_dados['DT_NASC'].isin(['0074-09-03','0083-09-28'])] = None

# COMMAND ----------

#base_dados = pd.read_csv(caminho_base, sep=separador_, decimal=decimal_)
base_dados = base_dados[['DOCUMENT', 'ID_PESSOA', 'EST_CIVIL', 'GENERO', 'DT_NASC', 'DT_CADASTRO']]

base_dados.fillna(-3)

#string
base_dados['GENERO'] = base_dados['GENERO'].replace(np.nan, '-3')

#numericas
base_dados['DOCUMENT'] = base_dados['DOCUMENT'].replace(np.nan, '-3')
base_dados['ID_PESSOA'] = base_dados['ID_PESSOA'].replace(np.nan, '-3')
base_dados['EST_CIVIL'] = base_dados['EST_CIVIL'].replace(np.nan, '-3')


base_dados = base_dados.apply(pd.to_numeric, errors='ignore')

base_dados['DOCUMENT'] = base_dados['DOCUMENT'].astype(np.int64)
base_dados['ID_PESSOA'] = base_dados['ID_PESSOA'].astype(np.int64)
base_dados['EST_CIVIL'] = base_dados['EST_CIVIL'].astype(int)


base_dados['DT_NASC'] = pd.to_datetime(base_dados['DT_NASC']) ##Alterad por conta de datas de nascimento null
base_dados['DT_CADASTRO'] = pd.to_datetime(base_dados['DT_CADASTRO'])

base_dados['mob_DT_NASC'] = ((datetime.today()) - base_dados.DT_NASC)/np.timedelta64(1, 'M')
base_dados['mob_DT_CADASTRO'] = ((datetime.today()) - base_dados.DT_CADASTRO)/np.timedelta64(1, 'M')

del base_dados['DT_NASC']
del base_dados['DT_CADASTRO']

base_dados.drop_duplicates(keep=False, inplace=True)

print("shape da Base de Dados:",base_dados.shape)

#base_dados.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gerando variáveis Categóricas

# COMMAND ----------

base_dados['GENERO_gh30'] = np.where(base_dados['GENERO'] == 'F', 0,
np.where(base_dados['GENERO'] == 'M', 1,
np.where(base_dados['GENERO'] == 'U', 2,
0)))
base_dados['GENERO_gh31'] = np.where(base_dados['GENERO_gh30'] == 0, 0,
np.where(base_dados['GENERO_gh30'] == 1, 1,
np.where(base_dados['GENERO_gh30'] == 2, 2,
0)))
base_dados['GENERO_gh32'] = np.where(base_dados['GENERO_gh31'] == 0, 0,
np.where(base_dados['GENERO_gh31'] == 1, 1,
np.where(base_dados['GENERO_gh31'] == 2, 2,
0)))
base_dados['GENERO_gh33'] = np.where(base_dados['GENERO_gh32'] == 0, 0,
np.where(base_dados['GENERO_gh32'] == 1, 1,
np.where(base_dados['GENERO_gh32'] == 2, 2,
0)))
base_dados['GENERO_gh34'] = np.where(base_dados['GENERO_gh33'] == 0, 0,
np.where(base_dados['GENERO_gh33'] == 1, 1,
np.where(base_dados['GENERO_gh33'] == 2, 0,
0)))
base_dados['GENERO_gh35'] = np.where(base_dados['GENERO_gh34'] == 0, 0,
np.where(base_dados['GENERO_gh34'] == 1, 1,
0))
base_dados['GENERO_gh36'] = np.where(base_dados['GENERO_gh35'] == 0, 1,
np.where(base_dados['GENERO_gh35'] == 1, 0,
0))
base_dados['GENERO_gh37'] = np.where(base_dados['GENERO_gh36'] == 0, 0,
np.where(base_dados['GENERO_gh36'] == 1, 1,
0))
base_dados['GENERO_gh38'] = np.where(base_dados['GENERO_gh37'] == 0, 0,
np.where(base_dados['GENERO_gh37'] == 1, 1,
0))
                                     
                                     
                                     
                                     
                                     
                                     
base_dados['EST_CIVIL_gh30'] = np.where(base_dados['EST_CIVIL'] == 0, 0,
np.where(base_dados['EST_CIVIL'] == 1, 1,
np.where(base_dados['EST_CIVIL'] == 2, 2,
np.where(base_dados['EST_CIVIL'] == 3, 3,
np.where(base_dados['EST_CIVIL'] == 4, 4,
np.where(base_dados['EST_CIVIL'] == 5, 5,
np.where(base_dados['EST_CIVIL'] == 6, 6,
np.where(base_dados['EST_CIVIL'] == 7, 7,
0))))))))
base_dados['EST_CIVIL_gh31'] = np.where(base_dados['EST_CIVIL_gh30'] == 0, 0,
np.where(base_dados['EST_CIVIL_gh30'] == 1, 1,
np.where(base_dados['EST_CIVIL_gh30'] == 2, 2,
np.where(base_dados['EST_CIVIL_gh30'] == 3, 3,
np.where(base_dados['EST_CIVIL_gh30'] == 4, 3,
np.where(base_dados['EST_CIVIL_gh30'] == 5, 5,
np.where(base_dados['EST_CIVIL_gh30'] == 6, 5,
np.where(base_dados['EST_CIVIL_gh30'] == 7, 7,
0))))))))
base_dados['EST_CIVIL_gh32'] = np.where(base_dados['EST_CIVIL_gh31'] == 0, 0,
np.where(base_dados['EST_CIVIL_gh31'] == 1, 1,
np.where(base_dados['EST_CIVIL_gh31'] == 2, 2,
np.where(base_dados['EST_CIVIL_gh31'] == 3, 3,
np.where(base_dados['EST_CIVIL_gh31'] == 5, 4,
np.where(base_dados['EST_CIVIL_gh31'] == 7, 5,
0))))))
base_dados['EST_CIVIL_gh33'] = np.where(base_dados['EST_CIVIL_gh32'] == 0, 0,
np.where(base_dados['EST_CIVIL_gh32'] == 1, 1,
np.where(base_dados['EST_CIVIL_gh32'] == 2, 2,
np.where(base_dados['EST_CIVIL_gh32'] == 3, 3,
np.where(base_dados['EST_CIVIL_gh32'] == 4, 4,
np.where(base_dados['EST_CIVIL_gh32'] == 5, 5,
0))))))
base_dados['EST_CIVIL_gh34'] = np.where(base_dados['EST_CIVIL_gh33'] == 0, 0,
np.where(base_dados['EST_CIVIL_gh33'] == 1, 1,
np.where(base_dados['EST_CIVIL_gh33'] == 2, 2,
np.where(base_dados['EST_CIVIL_gh33'] == 3, 0,
np.where(base_dados['EST_CIVIL_gh33'] == 4, 4,
np.where(base_dados['EST_CIVIL_gh33'] == 5, 6,
0))))))
base_dados['EST_CIVIL_gh35'] = np.where(base_dados['EST_CIVIL_gh34'] == 0, 0,
np.where(base_dados['EST_CIVIL_gh34'] == 1, 1,
np.where(base_dados['EST_CIVIL_gh34'] == 2, 2,
np.where(base_dados['EST_CIVIL_gh34'] == 4, 3,
np.where(base_dados['EST_CIVIL_gh34'] == 6, 4,
0)))))
base_dados['EST_CIVIL_gh36'] = np.where(base_dados['EST_CIVIL_gh35'] == 0, 2,
np.where(base_dados['EST_CIVIL_gh35'] == 1, 1,
np.where(base_dados['EST_CIVIL_gh35'] == 2, 2,
np.where(base_dados['EST_CIVIL_gh35'] == 3, 2,
np.where(base_dados['EST_CIVIL_gh35'] == 4, 0,
0)))))
base_dados['EST_CIVIL_gh37'] = np.where(base_dados['EST_CIVIL_gh36'] == 0, 1,
np.where(base_dados['EST_CIVIL_gh36'] == 1, 1,
np.where(base_dados['EST_CIVIL_gh36'] == 2, 2,
0)))
base_dados['EST_CIVIL_gh38'] = np.where(base_dados['EST_CIVIL_gh37'] == 1, 0,
np.where(base_dados['EST_CIVIL_gh37'] == 2, 1,
0))
                                        
                                        
                                        
                                        
                                        
                                        
base_dados['mob_DT_CADASTRO_gh30'] = np.where(base_dados['mob_DT_CADASTRO'] == -8.159991819192804, 0,
np.where(base_dados['mob_DT_CADASTRO'] <= -7.174345296676941, 1,
np.where(base_dados['mob_DT_CADASTRO'] == -0.209109870898178, 2,
np.where(base_dados['mob_DT_CADASTRO'] == 0.28371339035975335, 3,
np.where(base_dados['mob_DT_CADASTRO'] == 0.31656827444361546, 4,
np.where(base_dados['mob_DT_CADASTRO'] == 0.8093915357015469, 5,
np.where(base_dados['mob_DT_CADASTRO'] == 0.9736659561208573, 6,
np.where(base_dados['mob_DT_CADASTRO'] == 1.0065208402047194, 7,
np.where(base_dados['mob_DT_CADASTRO'] == 1.4336343332949266, 8,
np.where(base_dados['mob_DT_CADASTRO'] == 1.4664892173787887, 9,
np.where(base_dados['mob_DT_CADASTRO'] == 5.507639959693826, 10,
np.where(base_dados['mob_DT_CADASTRO'] == 6.526141366293551, 11,
np.where(base_dados['mob_DT_CADASTRO'] == 7.478933004725551, 12,
np.where(base_dados['mob_DT_CADASTRO'] == 8.497434411325276, 13,
np.where(base_dados['mob_DT_CADASTRO'] == 10.633001876776312, 14,
np.where(base_dados['mob_DT_CADASTRO'] == 11.585793515208312, 15,
np.where(base_dados['mob_DT_CADASTRO'] == 12.604294921808037, 16,
np.where(base_dados['mob_DT_CADASTRO'] == 12.80142422631121, 17,
np.where(base_dados['mob_DT_CADASTRO'] == 13.294247487569141, 18,
np.where(base_dados['mob_DT_CADASTRO'] == 13.5899414443239, 19,
np.where(base_dados['mob_DT_CADASTRO'] == 13.622796328407762, 20,
np.where(base_dados['mob_DT_CADASTRO'] == 14.279894010085004, 21,
np.where(base_dados['mob_DT_CADASTRO'] == 16.645445664123073, 22,
np.where(base_dados['mob_DT_CADASTRO'] == 16.87542985271011, 23,
np.where(base_dados['mob_DT_CADASTRO'] == 16.941139620877834, 24,
np.where(base_dados['mob_DT_CADASTRO'] == 16.973994504961695, 25,
np.where(base_dados['mob_DT_CADASTRO'] == 17.00684938904556, 26,
np.where(base_dados['mob_DT_CADASTRO'] == 17.20397869354873, 27,
np.where(base_dados['mob_DT_CADASTRO'] == 17.33539822988418, 28,
np.where(base_dados['mob_DT_CADASTRO'] == 17.532527534387352, 29,
np.where(base_dados['mob_DT_CADASTRO'] == 17.631092186638938, 30,
np.where(base_dados['mob_DT_CADASTRO'] == 17.861076375225974, 31,
np.where(base_dados['mob_DT_CADASTRO'] == 19.04385220224501, 32,
np.where(base_dados['mob_DT_CADASTRO'] == 19.208126622664317, 33,
np.where(base_dados['mob_DT_CADASTRO'] == 19.503820579419077, 34,
np.where(base_dados['mob_DT_CADASTRO'] == 19.99664384067701, 35,
np.where(base_dados['mob_DT_CADASTRO'] == 21.540823392618528, 36,
np.where(base_dados['mob_DT_CADASTRO'] == 22.592179683302113, 37,
np.where(base_dados['mob_DT_CADASTRO'] == 24.596327612417703, 38,
np.where(base_dados['mob_DT_CADASTRO'] >= 26.403346237030117, 39,
0))))))))))))))))))))))))))))))))))))))))
base_dados['mob_DT_CADASTRO_gh31'] = np.where(base_dados['mob_DT_CADASTRO_gh30'] == 0, 0,
np.where(base_dados['mob_DT_CADASTRO_gh30'] == 1, 1,
np.where(base_dados['mob_DT_CADASTRO_gh30'] == 2, 1,
np.where(base_dados['mob_DT_CADASTRO_gh30'] == 3, 1,
np.where(base_dados['mob_DT_CADASTRO_gh30'] == 4, 1,
np.where(base_dados['mob_DT_CADASTRO_gh30'] == 5, 1,
np.where(base_dados['mob_DT_CADASTRO_gh30'] == 6, 1,
np.where(base_dados['mob_DT_CADASTRO_gh30'] == 7, 1,
np.where(base_dados['mob_DT_CADASTRO_gh30'] == 8, 1,
np.where(base_dados['mob_DT_CADASTRO_gh30'] == 9, 9,
np.where(base_dados['mob_DT_CADASTRO_gh30'] == 10, 10,
np.where(base_dados['mob_DT_CADASTRO_gh30'] == 11, 11,
np.where(base_dados['mob_DT_CADASTRO_gh30'] == 12, 12,
np.where(base_dados['mob_DT_CADASTRO_gh30'] == 13, 13,
np.where(base_dados['mob_DT_CADASTRO_gh30'] == 14, 14,
np.where(base_dados['mob_DT_CADASTRO_gh30'] == 15, 15,
np.where(base_dados['mob_DT_CADASTRO_gh30'] == 16, 15,
np.where(base_dados['mob_DT_CADASTRO_gh30'] == 17, 17,
np.where(base_dados['mob_DT_CADASTRO_gh30'] == 18, 17,
np.where(base_dados['mob_DT_CADASTRO_gh30'] == 19, 17,
np.where(base_dados['mob_DT_CADASTRO_gh30'] == 20, 17,
np.where(base_dados['mob_DT_CADASTRO_gh30'] == 21, 17,
np.where(base_dados['mob_DT_CADASTRO_gh30'] == 22, 17,
np.where(base_dados['mob_DT_CADASTRO_gh30'] == 23, 17,
np.where(base_dados['mob_DT_CADASTRO_gh30'] == 24, 17,
np.where(base_dados['mob_DT_CADASTRO_gh30'] == 25, 17,
np.where(base_dados['mob_DT_CADASTRO_gh30'] == 26, 26,
np.where(base_dados['mob_DT_CADASTRO_gh30'] == 27, 27,
np.where(base_dados['mob_DT_CADASTRO_gh30'] == 28, 28,
np.where(base_dados['mob_DT_CADASTRO_gh30'] == 29, 28,
np.where(base_dados['mob_DT_CADASTRO_gh30'] == 30, 28,
np.where(base_dados['mob_DT_CADASTRO_gh30'] == 31, 28,
np.where(base_dados['mob_DT_CADASTRO_gh30'] == 32, 28,
np.where(base_dados['mob_DT_CADASTRO_gh30'] == 33, 28,
np.where(base_dados['mob_DT_CADASTRO_gh30'] == 34, 34,
np.where(base_dados['mob_DT_CADASTRO_gh30'] == 35, 35,
np.where(base_dados['mob_DT_CADASTRO_gh30'] == 36, 35,
np.where(base_dados['mob_DT_CADASTRO_gh30'] == 37, 35,
np.where(base_dados['mob_DT_CADASTRO_gh30'] == 38, 35,
np.where(base_dados['mob_DT_CADASTRO_gh30'] == 39, 39,
0))))))))))))))))))))))))))))))))))))))))
base_dados['mob_DT_CADASTRO_gh32'] = np.where(base_dados['mob_DT_CADASTRO_gh31'] == 0, 0,
np.where(base_dados['mob_DT_CADASTRO_gh31'] == 1, 1,
np.where(base_dados['mob_DT_CADASTRO_gh31'] == 9, 2,
np.where(base_dados['mob_DT_CADASTRO_gh31'] == 10, 3,
np.where(base_dados['mob_DT_CADASTRO_gh31'] == 11, 4,
np.where(base_dados['mob_DT_CADASTRO_gh31'] == 12, 5,
np.where(base_dados['mob_DT_CADASTRO_gh31'] == 13, 6,
np.where(base_dados['mob_DT_CADASTRO_gh31'] == 14, 7,
np.where(base_dados['mob_DT_CADASTRO_gh31'] == 15, 8,
np.where(base_dados['mob_DT_CADASTRO_gh31'] == 17, 9,
np.where(base_dados['mob_DT_CADASTRO_gh31'] == 26, 10,
np.where(base_dados['mob_DT_CADASTRO_gh31'] == 27, 11,
np.where(base_dados['mob_DT_CADASTRO_gh31'] == 28, 12,
np.where(base_dados['mob_DT_CADASTRO_gh31'] == 34, 13,
np.where(base_dados['mob_DT_CADASTRO_gh31'] == 35, 14,
np.where(base_dados['mob_DT_CADASTRO_gh31'] == 39, 15,
0))))))))))))))))
base_dados['mob_DT_CADASTRO_gh33'] = np.where(base_dados['mob_DT_CADASTRO_gh32'] == 0, 0,
np.where(base_dados['mob_DT_CADASTRO_gh32'] == 1, 1,
np.where(base_dados['mob_DT_CADASTRO_gh32'] == 2, 2,
np.where(base_dados['mob_DT_CADASTRO_gh32'] == 3, 3,
np.where(base_dados['mob_DT_CADASTRO_gh32'] == 4, 4,
np.where(base_dados['mob_DT_CADASTRO_gh32'] == 5, 5,
np.where(base_dados['mob_DT_CADASTRO_gh32'] == 6, 6,
np.where(base_dados['mob_DT_CADASTRO_gh32'] == 7, 7,
np.where(base_dados['mob_DT_CADASTRO_gh32'] == 8, 8,
np.where(base_dados['mob_DT_CADASTRO_gh32'] == 9, 9,
np.where(base_dados['mob_DT_CADASTRO_gh32'] == 10, 10,
np.where(base_dados['mob_DT_CADASTRO_gh32'] == 11, 11,
np.where(base_dados['mob_DT_CADASTRO_gh32'] == 12, 12,
np.where(base_dados['mob_DT_CADASTRO_gh32'] == 13, 13,
np.where(base_dados['mob_DT_CADASTRO_gh32'] == 14, 14,
np.where(base_dados['mob_DT_CADASTRO_gh32'] == 15, 15,
0))))))))))))))))
base_dados['mob_DT_CADASTRO_gh34'] = np.where(base_dados['mob_DT_CADASTRO_gh33'] == 0, 15,
np.where(base_dados['mob_DT_CADASTRO_gh33'] == 1, 1,
np.where(base_dados['mob_DT_CADASTRO_gh33'] == 2, 8,
np.where(base_dados['mob_DT_CADASTRO_gh33'] == 3, 8,
np.where(base_dados['mob_DT_CADASTRO_gh33'] == 4, 1,
np.where(base_dados['mob_DT_CADASTRO_gh33'] == 5, 15,
np.where(base_dados['mob_DT_CADASTRO_gh33'] == 6, 1,
np.where(base_dados['mob_DT_CADASTRO_gh33'] == 7, 1,
np.where(base_dados['mob_DT_CADASTRO_gh33'] == 8, 8,
np.where(base_dados['mob_DT_CADASTRO_gh33'] == 9, 15,
np.where(base_dados['mob_DT_CADASTRO_gh33'] == 10, 15,
np.where(base_dados['mob_DT_CADASTRO_gh33'] == 11, 1,
np.where(base_dados['mob_DT_CADASTRO_gh33'] == 12, 15,
np.where(base_dados['mob_DT_CADASTRO_gh33'] == 13, 15,
np.where(base_dados['mob_DT_CADASTRO_gh33'] == 14, 15,
np.where(base_dados['mob_DT_CADASTRO_gh33'] == 15, 15,
0))))))))))))))))
base_dados['mob_DT_CADASTRO_gh35'] = np.where(base_dados['mob_DT_CADASTRO_gh34'] == 1, 0,
np.where(base_dados['mob_DT_CADASTRO_gh34'] == 8, 1,
np.where(base_dados['mob_DT_CADASTRO_gh34'] == 15, 2,
0)))
base_dados['mob_DT_CADASTRO_gh36'] = np.where(base_dados['mob_DT_CADASTRO_gh35'] == 0, 0,
np.where(base_dados['mob_DT_CADASTRO_gh35'] == 1, 1,
np.where(base_dados['mob_DT_CADASTRO_gh35'] == 2, 2,
0)))
base_dados['mob_DT_CADASTRO_gh37'] = np.where(base_dados['mob_DT_CADASTRO_gh36'] == 0, 0,
np.where(base_dados['mob_DT_CADASTRO_gh36'] == 1, 0,
np.where(base_dados['mob_DT_CADASTRO_gh36'] == 2, 2,
0)))
base_dados['mob_DT_CADASTRO_gh38'] = np.where(base_dados['mob_DT_CADASTRO_gh37'] == 0, 0,
np.where(base_dados['mob_DT_CADASTRO_gh37'] == 2, 1,
0))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gerando variáveis numéricas contínuas

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parte 1 de 2

# COMMAND ----------

base_dados['DOCUMENT__pe_10'] = np.where(base_dados['DOCUMENT'] <= 9539068681.0, 0.0,
np.where(np.bitwise_and(base_dados['DOCUMENT'] > 9539068681.0, base_dados['DOCUMENT'] <= 18693323857.0), 1.0,
np.where(np.bitwise_and(base_dados['DOCUMENT'] > 18693323857.0, base_dados['DOCUMENT'] <= 28990152810.0), 2.0,
np.where(np.bitwise_and(base_dados['DOCUMENT'] > 28990152810.0, base_dados['DOCUMENT'] <= 38523515453.0), 3.0,
np.where(np.bitwise_and(base_dados['DOCUMENT'] > 38523515453.0, base_dados['DOCUMENT'] <= 48287822149.0), 4.0,
np.where(np.bitwise_and(base_dados['DOCUMENT'] > 48287822149.0, base_dados['DOCUMENT'] <= 57989508087.0), 5.0,
np.where(np.bitwise_and(base_dados['DOCUMENT'] > 57989508087.0, base_dados['DOCUMENT'] <= 67660169491.0), 6.0,
np.where(np.bitwise_and(base_dados['DOCUMENT'] > 67660169491.0, base_dados['DOCUMENT'] <= 77307607115.0), 7.0,
np.where(np.bitwise_and(base_dados['DOCUMENT'] > 77307607115.0, base_dados['DOCUMENT'] <= 86829963400.0), 8.0,
np.where(np.bitwise_and(base_dados['DOCUMENT'] > 86829963400.0, base_dados['DOCUMENT'] <= 96533951653.0), 9.0,
 -2))))))))))
base_dados['DOCUMENT__pe_10_g_1_1'] = np.where(base_dados['DOCUMENT__pe_10'] == -2.0, 3,
np.where(base_dados['DOCUMENT__pe_10'] == 0.0, 1,
np.where(base_dados['DOCUMENT__pe_10'] == 1.0, 0,
np.where(base_dados['DOCUMENT__pe_10'] == 2.0, 1,
np.where(base_dados['DOCUMENT__pe_10'] == 3.0, 2,
np.where(base_dados['DOCUMENT__pe_10'] == 4.0, 3,
np.where(base_dados['DOCUMENT__pe_10'] == 5.0, 3,
np.where(base_dados['DOCUMENT__pe_10'] == 6.0, 3,
np.where(base_dados['DOCUMENT__pe_10'] == 7.0, 1,
np.where(base_dados['DOCUMENT__pe_10'] == 8.0, 2,
np.where(base_dados['DOCUMENT__pe_10'] == 9.0, 2,
 0)))))))))))
base_dados['DOCUMENT__pe_10_g_1_2'] = np.where(base_dados['DOCUMENT__pe_10_g_1_1'] == 0, 2,
np.where(base_dados['DOCUMENT__pe_10_g_1_1'] == 1, 0,
np.where(base_dados['DOCUMENT__pe_10_g_1_1'] == 2, 2,
np.where(base_dados['DOCUMENT__pe_10_g_1_1'] == 3, 0,
 0))))
base_dados['DOCUMENT__pe_10_g_1'] = np.where(base_dados['DOCUMENT__pe_10_g_1_2'] == 0, 0,
np.where(base_dados['DOCUMENT__pe_10_g_1_2'] == 2, 1,
 0))
                                             
                                             
                                             
                                             
                                             
base_dados['DOCUMENT__pe_5'] = np.where(base_dados['DOCUMENT'] <= 18693323857.0, 0.0,
np.where(np.bitwise_and(base_dados['DOCUMENT'] > 18693323857.0, base_dados['DOCUMENT'] <= 38523515453.0), 1.0,
np.where(np.bitwise_and(base_dados['DOCUMENT'] > 38523515453.0, base_dados['DOCUMENT'] <= 57989508087.0), 2.0,
np.where(np.bitwise_and(base_dados['DOCUMENT'] > 57989508087.0, base_dados['DOCUMENT'] <= 77307607115.0), 3.0,
np.where(np.bitwise_and(base_dados['DOCUMENT'] > 77307607115.0, base_dados['DOCUMENT'] <= 96533951653.0), 4.0,
 -2)))))
base_dados['DOCUMENT__pe_5_g_1_1'] = np.where(base_dados['DOCUMENT__pe_5'] == -2.0, 3,
np.where(base_dados['DOCUMENT__pe_5'] == 0.0, 0,
np.where(base_dados['DOCUMENT__pe_5'] == 1.0, 2,
np.where(base_dados['DOCUMENT__pe_5'] == 2.0, 3,
np.where(base_dados['DOCUMENT__pe_5'] == 3.0, 1,
np.where(base_dados['DOCUMENT__pe_5'] == 4.0, 2,
 0))))))
base_dados['DOCUMENT__pe_5_g_1_2'] = np.where(base_dados['DOCUMENT__pe_5_g_1_1'] == 0, 1,
np.where(base_dados['DOCUMENT__pe_5_g_1_1'] == 1, 0,
np.where(base_dados['DOCUMENT__pe_5_g_1_1'] == 2, 3,
np.where(base_dados['DOCUMENT__pe_5_g_1_1'] == 3, 1,
 0))))
base_dados['DOCUMENT__pe_5_g_1'] = np.where(base_dados['DOCUMENT__pe_5_g_1_2'] == 0, 0,
np.where(base_dados['DOCUMENT__pe_5_g_1_2'] == 1, 1,
np.where(base_dados['DOCUMENT__pe_5_g_1_2'] == 3, 2,
 0)))
         
         
         
         
         
base_dados['ID_PESSOA__pe_8'] = np.where(base_dados['ID_PESSOA'] <= 1582231.0, 0.0,
np.where(np.bitwise_and(base_dados['ID_PESSOA'] > 1582231.0, base_dados['ID_PESSOA'] <= 3186223.0), 1.0,
np.where(np.bitwise_and(base_dados['ID_PESSOA'] > 3186223.0, base_dados['ID_PESSOA'] <= 4741655.0), 2.0,
np.where(np.bitwise_and(base_dados['ID_PESSOA'] > 4741655.0, base_dados['ID_PESSOA'] <= 6393389.0), 3.0,
np.where(np.bitwise_and(base_dados['ID_PESSOA'] > 6393389.0, base_dados['ID_PESSOA'] <= 7971208.0), 4.0,
np.where(np.bitwise_and(base_dados['ID_PESSOA'] > 7971208.0, base_dados['ID_PESSOA'] <= 9589263.0), 5.0,
np.where(np.bitwise_and(base_dados['ID_PESSOA'] > 9589263.0, base_dados['ID_PESSOA'] <= 11193862.0), 6.0,
np.where(base_dados['ID_PESSOA'] > 11193862.0, 7.0,
 -2))))))))
base_dados['ID_PESSOA__pe_8_g_1_1'] = np.where(base_dados['ID_PESSOA__pe_8'] == -2.0, 1,
np.where(base_dados['ID_PESSOA__pe_8'] == 0.0, 3,
np.where(base_dados['ID_PESSOA__pe_8'] == 1.0, 3,
np.where(base_dados['ID_PESSOA__pe_8'] == 2.0, 3,
np.where(base_dados['ID_PESSOA__pe_8'] == 3.0, 3,
np.where(base_dados['ID_PESSOA__pe_8'] == 4.0, 3,
np.where(base_dados['ID_PESSOA__pe_8'] == 5.0, 2,
np.where(base_dados['ID_PESSOA__pe_8'] == 6.0, 0,
np.where(base_dados['ID_PESSOA__pe_8'] == 7.0, 2,
 0)))))))))
base_dados['ID_PESSOA__pe_8_g_1_2'] = np.where(base_dados['ID_PESSOA__pe_8_g_1_1'] == 0, 1,
np.where(base_dados['ID_PESSOA__pe_8_g_1_1'] == 1, 3,
np.where(base_dados['ID_PESSOA__pe_8_g_1_1'] == 2, 0,
np.where(base_dados['ID_PESSOA__pe_8_g_1_1'] == 3, 1,
 0))))
base_dados['ID_PESSOA__pe_8_g_1'] = np.where(base_dados['ID_PESSOA__pe_8_g_1_2'] == 0, 0,
np.where(base_dados['ID_PESSOA__pe_8_g_1_2'] == 1, 1,
np.where(base_dados['ID_PESSOA__pe_8_g_1_2'] == 3, 2,
 0)))
         
         
         
         
         
         
         
base_dados['ID_PESSOA__S'] = np.sin(base_dados['ID_PESSOA'])
np.where(base_dados['ID_PESSOA__S'] == 0, -1, base_dados['ID_PESSOA__S'])
base_dados['ID_PESSOA__S'] = base_dados['ID_PESSOA__S'].fillna(-2)
base_dados['ID_PESSOA__S__p_3'] = np.where(base_dados['ID_PESSOA__S'] <= -0.48823759861152705, 0.0,
np.where(np.bitwise_and(base_dados['ID_PESSOA__S'] > -0.48823759861152705, base_dados['ID_PESSOA__S'] <= 0.5199243247059501), 1.0,
np.where(base_dados['ID_PESSOA__S'] > 0.5199243247059501, 2.0,
 0)))
base_dados['ID_PESSOA__S__p_3_g_1_1'] = np.where(base_dados['ID_PESSOA__S__p_3'] == 0.0, 1,
np.where(base_dados['ID_PESSOA__S__p_3'] == 1.0, 0,
np.where(base_dados['ID_PESSOA__S__p_3'] == 2.0, 0,
 0)))
base_dados['ID_PESSOA__S__p_3_g_1_2'] = np.where(base_dados['ID_PESSOA__S__p_3_g_1_1'] == 0, 1,
np.where(base_dados['ID_PESSOA__S__p_3_g_1_1'] == 1, 0,
 0))
base_dados['ID_PESSOA__S__p_3_g_1'] = np.where(base_dados['ID_PESSOA__S__p_3_g_1_2'] == 0, 0,
np.where(base_dados['ID_PESSOA__S__p_3_g_1_2'] == 1, 1,
 0))
                                               
                                               
                                               
                                               
                                               
                                               
base_dados['mob_DT_NASC__R'] = np.sqrt(base_dados['mob_DT_NASC'])
np.where(base_dados['mob_DT_NASC__R'] == 0, -1, base_dados['mob_DT_NASC__R'])
base_dados['mob_DT_NASC__R'] = base_dados['mob_DT_NASC__R'].fillna(-2)
base_dados['mob_DT_NASC__R__p_8'] = np.where(base_dados['mob_DT_NASC__R'] <= 21.127102843953207, 0.0,
np.where(np.bitwise_and(base_dados['mob_DT_NASC__R'] > 21.127102843953207, base_dados['mob_DT_NASC__R'] <= 22.426846409057827), 1.0,
np.where(np.bitwise_and(base_dados['mob_DT_NASC__R'] > 22.426846409057827, base_dados['mob_DT_NASC__R'] <= 23.455828876507006), 2.0,
np.where(np.bitwise_and(base_dados['mob_DT_NASC__R'] > 23.455828876507006, base_dados['mob_DT_NASC__R'] <= 24.409920056414528), 3.0,
np.where(np.bitwise_and(base_dados['mob_DT_NASC__R'] > 24.409920056414528, base_dados['mob_DT_NASC__R'] <= 25.48199263105371), 4.0,
np.where(np.bitwise_and(base_dados['mob_DT_NASC__R'] > 25.48199263105371, base_dados['mob_DT_NASC__R'] <= 26.573258043192094), 5.0,
np.where(np.bitwise_and(base_dados['mob_DT_NASC__R'] > 26.573258043192094, base_dados['mob_DT_NASC__R'] <= 28.13001782038973), 6.0,
np.where(base_dados['mob_DT_NASC__R'] > 28.13001782038973, 7.0,
 0))))))))
base_dados['mob_DT_NASC__R__p_8_g_1_1'] = np.where(base_dados['mob_DT_NASC__R__p_8'] == 0.0, 1,
np.where(base_dados['mob_DT_NASC__R__p_8'] == 1.0, 3,
np.where(base_dados['mob_DT_NASC__R__p_8'] == 2.0, 4,
np.where(base_dados['mob_DT_NASC__R__p_8'] == 3.0, 5,
np.where(base_dados['mob_DT_NASC__R__p_8'] == 4.0, 0,
np.where(base_dados['mob_DT_NASC__R__p_8'] == 5.0, 2,
np.where(base_dados['mob_DT_NASC__R__p_8'] == 6.0, 5,
np.where(base_dados['mob_DT_NASC__R__p_8'] == 7.0, 4,
 0))))))))
base_dados['mob_DT_NASC__R__p_8_g_1_2'] = np.where(base_dados['mob_DT_NASC__R__p_8_g_1_1'] == 0, 1,
np.where(base_dados['mob_DT_NASC__R__p_8_g_1_1'] == 1, 5,
np.where(base_dados['mob_DT_NASC__R__p_8_g_1_1'] == 2, 1,
np.where(base_dados['mob_DT_NASC__R__p_8_g_1_1'] == 3, 4,
np.where(base_dados['mob_DT_NASC__R__p_8_g_1_1'] == 4, 1,
np.where(base_dados['mob_DT_NASC__R__p_8_g_1_1'] == 5, 0,
 0))))))
base_dados['mob_DT_NASC__R__p_8_g_1'] = np.where(base_dados['mob_DT_NASC__R__p_8_g_1_2'] == 0, 0,
np.where(base_dados['mob_DT_NASC__R__p_8_g_1_2'] == 1, 1,
np.where(base_dados['mob_DT_NASC__R__p_8_g_1_2'] == 4, 2,
np.where(base_dados['mob_DT_NASC__R__p_8_g_1_2'] == 5, 3,
 0))))
         
         
         
         
         
         
         
base_dados['mob_DT_NASC__L'] = np.log(base_dados['mob_DT_NASC'])
np.where(base_dados['mob_DT_NASC__L'] == 0, -1, base_dados['mob_DT_NASC__L'])
base_dados['mob_DT_NASC__L'] = base_dados['mob_DT_NASC__L'].fillna(-2)
base_dados['mob_DT_NASC__L__pu_10'] = np.where(base_dados['mob_DT_NASC__L'] <= -2.0, 0.0,
np.where(np.bitwise_and(base_dados['mob_DT_NASC__L'] > -2.0, base_dados['mob_DT_NASC__L'] <= 6.35935338687095), 8.0,
np.where(base_dados['mob_DT_NASC__L'] > 6.35935338687095, 9.0,
 0)))
base_dados['mob_DT_NASC__L__pu_10_g_1_1'] = np.where(base_dados['mob_DT_NASC__L__pu_10'] == 0.0, 1,
np.where(base_dados['mob_DT_NASC__L__pu_10'] == 8.0, 1,
np.where(base_dados['mob_DT_NASC__L__pu_10'] == 9.0, 0,
 0)))
base_dados['mob_DT_NASC__L__pu_10_g_1_2'] = np.where(base_dados['mob_DT_NASC__L__pu_10_g_1_1'] == 0, 0,
np.where(base_dados['mob_DT_NASC__L__pu_10_g_1_1'] == 1, 1,
 0))
base_dados['mob_DT_NASC__L__pu_10_g_1'] = np.where(base_dados['mob_DT_NASC__L__pu_10_g_1_2'] == 0, 0,
np.where(base_dados['mob_DT_NASC__L__pu_10_g_1_2'] == 1, 1,
 0))


# COMMAND ----------

# MAGIC %md
# MAGIC ### Parte 2 de 2

# COMMAND ----------

base_dados['DOCUMENT__pe_5_g_1_c1_10_1'] = np.where(np.bitwise_and(base_dados['DOCUMENT__pe_10_g_1'] == 0, base_dados['DOCUMENT__pe_5_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['DOCUMENT__pe_10_g_1'] == 0, base_dados['DOCUMENT__pe_5_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['DOCUMENT__pe_10_g_1'] == 0, base_dados['DOCUMENT__pe_5_g_1'] == 2), 2,
np.where(np.bitwise_and(base_dados['DOCUMENT__pe_10_g_1'] == 1, base_dados['DOCUMENT__pe_5_g_1'] == 0), 2,
np.where(np.bitwise_and(base_dados['DOCUMENT__pe_10_g_1'] == 1, base_dados['DOCUMENT__pe_5_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['DOCUMENT__pe_10_g_1'] == 1, base_dados['DOCUMENT__pe_5_g_1'] == 2), 3,
 0))))))
base_dados['DOCUMENT__pe_5_g_1_c1_10_2'] = np.where(base_dados['DOCUMENT__pe_5_g_1_c1_10_1'] == 0, 0,
np.where(base_dados['DOCUMENT__pe_5_g_1_c1_10_1'] == 1, 1,
np.where(base_dados['DOCUMENT__pe_5_g_1_c1_10_1'] == 2, 2,
np.where(base_dados['DOCUMENT__pe_5_g_1_c1_10_1'] == 3, 3,
0))))
base_dados['DOCUMENT__pe_5_g_1_c1_10'] = np.where(base_dados['DOCUMENT__pe_5_g_1_c1_10_2'] == 0, 0,
np.where(base_dados['DOCUMENT__pe_5_g_1_c1_10_2'] == 1, 1,
np.where(base_dados['DOCUMENT__pe_5_g_1_c1_10_2'] == 2, 2,
np.where(base_dados['DOCUMENT__pe_5_g_1_c1_10_2'] == 3, 3,
 0))))
         
         
         
         
                
base_dados['ID_PESSOA__pe_8_g_1_c1_6_1'] = np.where(np.bitwise_and(base_dados['ID_PESSOA__pe_8_g_1'] == 0, base_dados['ID_PESSOA__S__p_3_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['ID_PESSOA__pe_8_g_1'] == 0, base_dados['ID_PESSOA__S__p_3_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['ID_PESSOA__pe_8_g_1'] == 1, base_dados['ID_PESSOA__S__p_3_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['ID_PESSOA__pe_8_g_1'] == 1, base_dados['ID_PESSOA__S__p_3_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['ID_PESSOA__pe_8_g_1'] == 2, base_dados['ID_PESSOA__S__p_3_g_1'] == 0), 3,
np.where(np.bitwise_and(base_dados['ID_PESSOA__pe_8_g_1'] == 2, base_dados['ID_PESSOA__S__p_3_g_1'] == 1), 3,
 0))))))
base_dados['ID_PESSOA__pe_8_g_1_c1_6_2'] = np.where(base_dados['ID_PESSOA__pe_8_g_1_c1_6_1'] == 0, 0,
np.where(base_dados['ID_PESSOA__pe_8_g_1_c1_6_1'] == 1, 1,
np.where(base_dados['ID_PESSOA__pe_8_g_1_c1_6_1'] == 2, 2,
np.where(base_dados['ID_PESSOA__pe_8_g_1_c1_6_1'] == 3, 3,
0))))
base_dados['ID_PESSOA__pe_8_g_1_c1_6'] = np.where(base_dados['ID_PESSOA__pe_8_g_1_c1_6_2'] == 0, 0,
np.where(base_dados['ID_PESSOA__pe_8_g_1_c1_6_2'] == 1, 1,
np.where(base_dados['ID_PESSOA__pe_8_g_1_c1_6_2'] == 2, 2,
np.where(base_dados['ID_PESSOA__pe_8_g_1_c1_6_2'] == 3, 3,
 0))))
         
         
         
         
         
        
base_dados['mob_DT_NASC__L__pu_10_g_1_c1_20_1'] = np.where(np.bitwise_and(base_dados['mob_DT_NASC__R__p_8_g_1'] == 0, base_dados['mob_DT_NASC__L__pu_10_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['mob_DT_NASC__R__p_8_g_1'] == 0, base_dados['mob_DT_NASC__L__pu_10_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['mob_DT_NASC__R__p_8_g_1'] == 1, base_dados['mob_DT_NASC__L__pu_10_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['mob_DT_NASC__R__p_8_g_1'] == 1, base_dados['mob_DT_NASC__L__pu_10_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['mob_DT_NASC__R__p_8_g_1'] == 2, base_dados['mob_DT_NASC__L__pu_10_g_1'] == 0), 3,
np.where(np.bitwise_and(base_dados['mob_DT_NASC__R__p_8_g_1'] == 2, base_dados['mob_DT_NASC__L__pu_10_g_1'] == 1), 3,
np.where(np.bitwise_and(base_dados['mob_DT_NASC__R__p_8_g_1'] == 3, base_dados['mob_DT_NASC__L__pu_10_g_1'] == 0), 4,
np.where(np.bitwise_and(base_dados['mob_DT_NASC__R__p_8_g_1'] == 3, base_dados['mob_DT_NASC__L__pu_10_g_1'] == 1), 4,
 0))))))))
base_dados['mob_DT_NASC__L__pu_10_g_1_c1_20_2'] = np.where(base_dados['mob_DT_NASC__L__pu_10_g_1_c1_20_1'] == 0, 0,
np.where(base_dados['mob_DT_NASC__L__pu_10_g_1_c1_20_1'] == 1, 1,
np.where(base_dados['mob_DT_NASC__L__pu_10_g_1_c1_20_1'] == 2, 2,
np.where(base_dados['mob_DT_NASC__L__pu_10_g_1_c1_20_1'] == 3, 3,
np.where(base_dados['mob_DT_NASC__L__pu_10_g_1_c1_20_1'] == 4, 4,
0)))))
base_dados['mob_DT_NASC__L__pu_10_g_1_c1_20'] = np.where(base_dados['mob_DT_NASC__L__pu_10_g_1_c1_20_2'] == 0, 0,
np.where(base_dados['mob_DT_NASC__L__pu_10_g_1_c1_20_2'] == 1, 1,
np.where(base_dados['mob_DT_NASC__L__pu_10_g_1_c1_20_2'] == 2, 2,
np.where(base_dados['mob_DT_NASC__L__pu_10_g_1_c1_20_2'] == 3, 3,
np.where(base_dados['mob_DT_NASC__L__pu_10_g_1_c1_20_2'] == 4, 4,
 0)))))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Mantendo apenas as variáveis do modelo

# COMMAND ----------

import pickle
modelo=pickle.load(open(pickle_path + 'model_fit_Credigy_gbm.sav', 'rb'))

base_teste_c0 = base_dados[[chave,'DOCUMENT__pe_5_g_1_c1_10', 'ID_PESSOA__pe_8_g_1_c1_6', 'mob_DT_NASC__L__pu_10_g_1_c1_20', 'mob_DT_CADASTRO_gh38', 'GENERO_gh38', 'EST_CIVIL_gh38']]

var_fin_c0=['DOCUMENT__pe_5_g_1_c1_10', 'ID_PESSOA__pe_8_g_1_c1_6', 'mob_DT_NASC__L__pu_10_g_1_c1_20', 'mob_DT_CADASTRO_gh38', 'GENERO_gh38', 'EST_CIVIL_gh38']

#print(var_fin_c0)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Rodando Regressão Logística

# COMMAND ----------

# Datasets de treino e de teste
x_teste = base_teste_c0[var_fin_c0]
z_teste = base_teste_c0[chave]

# Previsões
valores_previstos = modelo.predict(x_teste)

probabilidades = modelo.predict_proba(x_teste)
data_prob = pd.DataFrame({'P_1': probabilidades[:, 1]})

z_teste1 = z_teste.reset_index(drop=True)
data_prob1 = data_prob.reset_index(drop=True)

x_teste2 = pd.concat([z_teste1,data_prob1], axis=1)

#x_teste2


# COMMAND ----------

# MAGIC %md
# MAGIC # Modelo de Grupo Homogêneo

# COMMAND ----------

x_teste2['P_1_T'] = np.tan(x_teste2['P_1'])
x_teste2['P_1_T'] = np.where(x_teste2['P_1'] == 0, -1, x_teste2['P_1_T'])
x_teste2['P_1_T'] = x_teste2['P_1_T'].fillna(-2)
x_teste2['P_1_T'] = x_teste2['P_1_T'].fillna(-2)

x_teste2['P_1_T_pe_7_g_1'] = np.where(np.bitwise_and(x_teste2['P_1_T'] > 0.1888456, x_teste2['P_1_T'] <= 0.374710142), 1,
    np.where(np.bitwise_and(x_teste2['P_1_T'] > 0.374710142, x_teste2['P_1_T'] <= 0.626138457), 2,0))

x_teste2['P_1_p_17_g_1'] = np.where(x_teste2['P_1'] <= 0.151761221, 0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.151761221, x_teste2['P_1'] <= 0.365550985), 1,2))

x_teste2['GH'] = np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 0, x_teste2['P_1_T_pe_7_g_1'] == 0), 0,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 0, x_teste2['P_1_T_pe_7_g_1'] == 1), 0,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 0, x_teste2['P_1_T_pe_7_g_1'] == 2), 0,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 1, x_teste2['P_1_T_pe_7_g_1'] == 0), 1,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 1, x_teste2['P_1_T_pe_7_g_1'] == 1), 2,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 1, x_teste2['P_1_T_pe_7_g_1'] == 2), 2,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 2, x_teste2['P_1_T_pe_7_g_1'] == 0), 3,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 2, x_teste2['P_1_T_pe_7_g_1'] == 1), 3,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 2, x_teste2['P_1_T_pe_7_g_1'] == 2), 3,0)))))))))

del x_teste2['P_1_T']
del x_teste2['P_1_T_pe_7_g_1']
del x_teste2['P_1_p_17_g_1']
#x_teste2

# COMMAND ----------

try:
  dbutils.fs.rm(outputpath, True)
except:
  pass
dbutils.fs.mkdirs(outputpath)

x_teste2.to_csv(open(os.path.join(outputpath_dbfs, 'pre_output:'+(str(datetime.today())[0:10]+'.csv')),'wb'))
os.path.join(outputpath_dbfs, 'pre_output:'+(str(datetime.today())[0:10]+'.csv'))

# COMMAND ----------

# DBTITLE 1,Excluindo a tabela dos outputs_to_score
dir_outputs_trusted='/mnt/ml-prd/ml-data/propensaodeal/credigy/trusted'
df_outputs_scores=spark.createDataFrame(dbutils.fs.ls(dir_outputs_trusted)).filter(F.col('path').contains('output_to_score')).withColumn("path", F.regexp_replace("path","dbfs:","")).collect()
for i in df_outputs_scores:
  aux_remove=dbutils.fs.ls(i.path)
  for j in aux_remove:
    dbutils.fs.rm(j.path, True)
  dbutils.fs.rm(i.path,True)