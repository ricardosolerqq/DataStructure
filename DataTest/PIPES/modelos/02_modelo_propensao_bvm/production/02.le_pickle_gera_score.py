# Databricks notebook source
import pickle
import os
import pandas as pd
import numpy as np
import time

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pyspark.sql.types import DateType

# COMMAND ----------

# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

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
chave = 'NUCPFCNPJ'


#Caminho da base de dados
caminho_base = "/dbfs/mnt/ml-prd/ml-data/propensaodeal/bvm/trusted/"
list_base = os.listdir(caminho_base)

#Nome da Base de Dados
N_Base = max(list_base)
dt_max = N_Base.split('.')[0]

#Separador
separador_ = ";"

#Decimal
decimal_ = "."

caminho_trusted = '/mnt/ml-prd/ml-data/propensaodeal/bvm/trusted'
caminho_trusted_dbfs = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/bvm/trusted'

pickle_path = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/bvm/pickle_model/'

outputpath = 'mnt/ml-prd/ml-data/propensaodeal/bvm/output/'
outputpath_dbfs = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/bvm/output/'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importação da Base de Dados

# COMMAND ----------

base_dados = pd.read_csv(caminho_base+N_Base, sep=separador_, decimal=decimal_)
base_dados = base_dados[['ATRASO', 'NUCPFCNPJ', 'NUCONTRATO', 'Telefone1', 'Telefone2', 'Telefone3', 'Telefone4', 'VLR_RISCO', 'Email1', 'CDPRODUTO', 'NUPARCELA']]


base_dados.fillna(-3)

#string
base_dados['Email1'] = base_dados['Email1'].replace(np.nan, '-3')
base_dados['P_Email1'] = np.where(base_dados['Email1'] != '-3',1,0)

#numericas
base_dados['NUCPFCNPJ'] = base_dados['NUCPFCNPJ'].replace(np.nan, '-3')
base_dados['NUCONTRATO'] = base_dados['NUCONTRATO'].replace(np.nan, '-3')
base_dados['CDPRODUTO'] = base_dados['CDPRODUTO'].replace(np.nan, '-3')
base_dados['NUPARCELA'] = base_dados['NUPARCELA'].replace(np.nan, '-3')
base_dados['VLR_RISCO'] = base_dados['VLR_RISCO'].replace(np.nan, '-3')
base_dados['ATRASO'] = base_dados['ATRASO'].replace(np.nan, '-3')
base_dados['Telefone1'] = base_dados['Telefone1'].replace(np.nan, '-3')
base_dados['Telefone2'] = base_dados['Telefone2'].replace(np.nan, '-3')
base_dados['Telefone3'] = base_dados['Telefone3'].replace(np.nan, '-3')
base_dados['Telefone4'] = base_dados['Telefone4'].replace(np.nan, '-3')

base_dados = base_dados.apply(pd.to_numeric, errors='ignore')

base_dados['NUCPFCNPJ'] = base_dados['NUCPFCNPJ'].astype(np.int64)
base_dados['NUCONTRATO'] = base_dados['NUCONTRATO'].astype(np.int64)
base_dados['CDPRODUTO'] = base_dados['CDPRODUTO'].astype(int)
base_dados['NUPARCELA'] = base_dados['NUPARCELA'].astype(int)
base_dados['VLR_RISCO'] = base_dados['VLR_RISCO'].astype(float)
base_dados['ATRASO'] = base_dados['ATRASO'].astype(np.int)
base_dados['Telefone1'] = base_dados['Telefone1'].astype(np.int64)
base_dados['Telefone2'] = base_dados['Telefone2'].astype(np.int64)
base_dados['Telefone3'] = base_dados['Telefone3'].astype(np.int64)
base_dados['Telefone4'] = base_dados['Telefone4'].astype(np.int64)

del base_dados['Email1']

base_dados.drop_duplicates(keep=False, inplace=True)

print("shape da Base de Dados:",base_dados.shape)


base_dados.head()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Gerando variáveis Categóricas

# COMMAND ----------

base_dados['CDPRODUTO_gh30'] = np.where(base_dados['CDPRODUTO'] == 10, 0,
np.where(base_dados['CDPRODUTO'] == 11, 1,
np.where(base_dados['CDPRODUTO'] == 12, 2,
np.where(base_dados['CDPRODUTO'] == 13, 3,
np.where(base_dados['CDPRODUTO'] == 14, 4,
0)))))
base_dados['CDPRODUTO_gh31'] = np.where(base_dados['CDPRODUTO_gh30'] == 0, 0,
np.where(base_dados['CDPRODUTO_gh30'] == 1, 1,
np.where(base_dados['CDPRODUTO_gh30'] == 2, 2,
np.where(base_dados['CDPRODUTO_gh30'] == 3, 3,
np.where(base_dados['CDPRODUTO_gh30'] == 4, 3,
0)))))
base_dados['CDPRODUTO_gh32'] = np.where(base_dados['CDPRODUTO_gh31'] == 0, 0,
np.where(base_dados['CDPRODUTO_gh31'] == 1, 1,
np.where(base_dados['CDPRODUTO_gh31'] == 2, 2,
np.where(base_dados['CDPRODUTO_gh31'] == 3, 3,
0))))
base_dados['CDPRODUTO_gh33'] = np.where(base_dados['CDPRODUTO_gh32'] == 0, 0,
np.where(base_dados['CDPRODUTO_gh32'] == 1, 1,
np.where(base_dados['CDPRODUTO_gh32'] == 2, 2,
np.where(base_dados['CDPRODUTO_gh32'] == 3, 3,
0))))
base_dados['CDPRODUTO_gh34'] = np.where(base_dados['CDPRODUTO_gh33'] == 0, 0,
np.where(base_dados['CDPRODUTO_gh33'] == 1, 1,
np.where(base_dados['CDPRODUTO_gh33'] == 2, 2,
np.where(base_dados['CDPRODUTO_gh33'] == 3, 4,
0))))
base_dados['CDPRODUTO_gh35'] = np.where(base_dados['CDPRODUTO_gh34'] == 0, 0,
np.where(base_dados['CDPRODUTO_gh34'] == 1, 1,
np.where(base_dados['CDPRODUTO_gh34'] == 2, 2,
np.where(base_dados['CDPRODUTO_gh34'] == 4, 3,
0))))
base_dados['CDPRODUTO_gh36'] = np.where(base_dados['CDPRODUTO_gh35'] == 0, 1,
np.where(base_dados['CDPRODUTO_gh35'] == 1, 3,
np.where(base_dados['CDPRODUTO_gh35'] == 2, 1,
np.where(base_dados['CDPRODUTO_gh35'] == 3, 0,
0))))
base_dados['CDPRODUTO_gh37'] = np.where(base_dados['CDPRODUTO_gh36'] == 0, 1,
np.where(base_dados['CDPRODUTO_gh36'] == 1, 1,
np.where(base_dados['CDPRODUTO_gh36'] == 3, 2,
0)))
base_dados['CDPRODUTO_gh38'] = np.where(base_dados['CDPRODUTO_gh37'] == 1, 0,
np.where(base_dados['CDPRODUTO_gh37'] == 2, 1,
0))
                                        
                                        
                                        
                                        
                                        
                                        
                                        
                                        
base_dados['P_Email1_gh30'] = np.where(base_dados['P_Email1'] == 0, 0,
np.where(base_dados['P_Email1'] == 1, 1,
0))
base_dados['P_Email1_gh31'] = np.where(base_dados['P_Email1_gh30'] == 0, 0,
np.where(base_dados['P_Email1_gh30'] == 1, 1,
0))
base_dados['P_Email1_gh32'] = np.where(base_dados['P_Email1_gh31'] == 0, 0,
np.where(base_dados['P_Email1_gh31'] == 1, 1,
0))
base_dados['P_Email1_gh33'] = np.where(base_dados['P_Email1_gh32'] == 0, 0,
np.where(base_dados['P_Email1_gh32'] == 1, 1,
0))
base_dados['P_Email1_gh34'] = np.where(base_dados['P_Email1_gh33'] == 0, 0,
np.where(base_dados['P_Email1_gh33'] == 1, 1,
0))
base_dados['P_Email1_gh35'] = np.where(base_dados['P_Email1_gh34'] == 0, 0,
np.where(base_dados['P_Email1_gh34'] == 1, 1,
0))
base_dados['P_Email1_gh36'] = np.where(base_dados['P_Email1_gh35'] == 0, 0,
np.where(base_dados['P_Email1_gh35'] == 1, 1,
0))
base_dados['P_Email1_gh37'] = np.where(base_dados['P_Email1_gh36'] == 0, 0,
np.where(base_dados['P_Email1_gh36'] == 1, 1,
0))
base_dados['P_Email1_gh38'] = np.where(base_dados['P_Email1_gh37'] == 0, 0,
np.where(base_dados['P_Email1_gh37'] == 1, 1,
0))






base_dados['NUPARCELA_gh60'] = np.where(base_dados['NUPARCELA'] == 1, 0,
np.where(base_dados['NUPARCELA'] == 2, 1,
np.where(base_dados['NUPARCELA'] == 3, 2,
np.where(base_dados['NUPARCELA'] == 4, 3,
np.where(base_dados['NUPARCELA'] == 5, 4,
np.where(base_dados['NUPARCELA'] == 6, 5,
np.where(base_dados['NUPARCELA'] == 7, 6,
np.where(base_dados['NUPARCELA'] == 8, 7,
np.where(base_dados['NUPARCELA'] == 9, 8,
np.where(base_dados['NUPARCELA'] == 10, 9,
np.where(base_dados['NUPARCELA'] == 11, 10,
np.where(base_dados['NUPARCELA'] == 12, 11,
np.where(base_dados['NUPARCELA'] == 13, 12,
np.where(base_dados['NUPARCELA'] == 14, 13,
np.where(base_dados['NUPARCELA'] == 15, 14,
np.where(base_dados['NUPARCELA'] == 16, 15,
np.where(base_dados['NUPARCELA'] == 17, 16,
np.where(base_dados['NUPARCELA'] == 18, 17,
np.where(base_dados['NUPARCELA'] == 19, 18,
np.where(base_dados['NUPARCELA'] == 20, 19,
np.where(base_dados['NUPARCELA'] == 21, 20,
np.where(base_dados['NUPARCELA'] == 22, 21,
np.where(base_dados['NUPARCELA'] == 23, 22,
np.where(base_dados['NUPARCELA'] == 24, 23,
np.where(base_dados['NUPARCELA'] == 25, 24,
np.where(base_dados['NUPARCELA'] == 26, 25,
np.where(base_dados['NUPARCELA'] == 27, 26,
np.where(base_dados['NUPARCELA'] == 28, 27,
np.where(base_dados['NUPARCELA'] == 29, 28,
np.where(base_dados['NUPARCELA'] == 30, 29,
np.where(base_dados['NUPARCELA'] == 31, 30,
np.where(base_dados['NUPARCELA'] == 32, 31,
np.where(base_dados['NUPARCELA'] == 33, 32,
np.where(base_dados['NUPARCELA'] == 34, 33,
np.where(base_dados['NUPARCELA'] == 35, 34,
np.where(base_dados['NUPARCELA'] == 36, 35,
np.where(base_dados['NUPARCELA'] == 37, 36,
np.where(base_dados['NUPARCELA'] == 38, 37,
np.where(base_dados['NUPARCELA'] == 39, 38,
np.where(base_dados['NUPARCELA'] == 40, 39,
np.where(base_dados['NUPARCELA'] == 41, 40,
np.where(base_dados['NUPARCELA'] == 42, 41,
np.where(base_dados['NUPARCELA'] == 43, 42,
np.where(base_dados['NUPARCELA'] == 44, 43,
np.where(base_dados['NUPARCELA'] == 46, 44,
np.where(base_dados['NUPARCELA'] == 47, 45,
np.where(base_dados['NUPARCELA'] == 48, 46,
np.where(base_dados['NUPARCELA'] == 53, 47,
np.where(base_dados['NUPARCELA'] == 59, 48,
0)))))))))))))))))))))))))))))))))))))))))))))))))

base_dados['NUPARCELA_gh61'] = np.where(base_dados['NUPARCELA_gh60'] == 0, 0,
np.where(base_dados['NUPARCELA_gh60'] == 1, 2,
np.where(base_dados['NUPARCELA_gh60'] == 2, 4,
np.where(base_dados['NUPARCELA_gh60'] == 3, 3,
np.where(base_dados['NUPARCELA_gh60'] == 4, 1,
np.where(base_dados['NUPARCELA_gh60'] == 5, 5,
np.where(base_dados['NUPARCELA_gh60'] == 6, 7,
np.where(base_dados['NUPARCELA_gh60'] == 7, -5,
np.where(base_dados['NUPARCELA_gh60'] == 8, 10,
np.where(base_dados['NUPARCELA_gh60'] == 9, -5,
np.where(base_dados['NUPARCELA_gh60'] == 10, 11,
np.where(base_dados['NUPARCELA_gh60'] == 11, -5,
np.where(base_dados['NUPARCELA_gh60'] == 12, -5,
np.where(base_dados['NUPARCELA_gh60'] == 13, -5,
np.where(base_dados['NUPARCELA_gh60'] == 14, 8,
np.where(base_dados['NUPARCELA_gh60'] == 15, 6,
np.where(base_dados['NUPARCELA_gh60'] == 16, 9,
np.where(base_dados['NUPARCELA_gh60'] == 17, -5,
np.where(base_dados['NUPARCELA_gh60'] == 18, -5,
np.where(base_dados['NUPARCELA_gh60'] == 19, -5,
np.where(base_dados['NUPARCELA_gh60'] == 20, -5,
np.where(base_dados['NUPARCELA_gh60'] == 21, -5,
np.where(base_dados['NUPARCELA_gh60'] == 22, -5,
np.where(base_dados['NUPARCELA_gh60'] == 23, -5,
np.where(base_dados['NUPARCELA_gh60'] == 24, -5,
np.where(base_dados['NUPARCELA_gh60'] == 25, -5,
np.where(base_dados['NUPARCELA_gh60'] == 26, -5,
np.where(base_dados['NUPARCELA_gh60'] == 27, -5,
np.where(base_dados['NUPARCELA_gh60'] == 28, -5,
np.where(base_dados['NUPARCELA_gh60'] == 29, -5,
np.where(base_dados['NUPARCELA_gh60'] == 30, -5,
np.where(base_dados['NUPARCELA_gh60'] == 31, -5,
np.where(base_dados['NUPARCELA_gh60'] == 32, -5,
np.where(base_dados['NUPARCELA_gh60'] == 33, -5,
np.where(base_dados['NUPARCELA_gh60'] == 34, -5,
np.where(base_dados['NUPARCELA_gh60'] == 35, -5,
np.where(base_dados['NUPARCELA_gh60'] == 36, -5,
np.where(base_dados['NUPARCELA_gh60'] == 37, -5,
np.where(base_dados['NUPARCELA_gh60'] == 38, -5,
np.where(base_dados['NUPARCELA_gh60'] == 39, -5,
np.where(base_dados['NUPARCELA_gh60'] == 40, -5,
np.where(base_dados['NUPARCELA_gh60'] == 41, -5,
np.where(base_dados['NUPARCELA_gh60'] == 42, -5,
np.where(base_dados['NUPARCELA_gh60'] == 43, -5,
np.where(base_dados['NUPARCELA_gh60'] == 44, -5,
np.where(base_dados['NUPARCELA_gh60'] == 45, -5,
np.where(base_dados['NUPARCELA_gh60'] == 46, -5,
np.where(base_dados['NUPARCELA_gh60'] == 47, -5,
np.where(base_dados['NUPARCELA_gh60'] == 48, -5,
0)))))))))))))))))))))))))))))))))))))))))))))))))

base_dados['NUPARCELA_gh62'] = np.where(base_dados['NUPARCELA_gh61'] == -5, 0,
np.where(base_dados['NUPARCELA_gh61'] == 0, 1,
np.where(base_dados['NUPARCELA_gh61'] == 1, 2,
np.where(base_dados['NUPARCELA_gh61'] == 2, 3,
np.where(base_dados['NUPARCELA_gh61'] == 3, 4,
np.where(base_dados['NUPARCELA_gh61'] == 4, 5,
np.where(base_dados['NUPARCELA_gh61'] == 5, 6,
np.where(base_dados['NUPARCELA_gh61'] == 6, 7,
np.where(base_dados['NUPARCELA_gh61'] == 7, 8,
np.where(base_dados['NUPARCELA_gh61'] == 8, 9,
np.where(base_dados['NUPARCELA_gh61'] == 9, 10,
np.where(base_dados['NUPARCELA_gh61'] == 10, 11,
np.where(base_dados['NUPARCELA_gh61'] == 11, 12,
0)))))))))))))

base_dados['NUPARCELA_gh63'] = np.where(base_dados['NUPARCELA_gh62'] == 0, 5,
np.where(base_dados['NUPARCELA_gh62'] == 1, 8,
np.where(base_dados['NUPARCELA_gh62'] == 2, 5,
np.where(base_dados['NUPARCELA_gh62'] == 3, 8,
np.where(base_dados['NUPARCELA_gh62'] == 4, 8,
np.where(base_dados['NUPARCELA_gh62'] == 5, 5,
np.where(base_dados['NUPARCELA_gh62'] == 6, 11,
np.where(base_dados['NUPARCELA_gh62'] == 7, 0,
np.where(base_dados['NUPARCELA_gh62'] == 8, 1,
np.where(base_dados['NUPARCELA_gh62'] == 9, 12,
np.where(base_dados['NUPARCELA_gh62'] == 10, 1,
np.where(base_dados['NUPARCELA_gh62'] == 11, 1,
np.where(base_dados['NUPARCELA_gh62'] == 12, 1,
0)))))))))))))

base_dados['NUPARCELA_gh64'] = np.where(base_dados['NUPARCELA_gh63'] == 0, 0,
np.where(base_dados['NUPARCELA_gh63'] == 1, 1,
np.where(base_dados['NUPARCELA_gh63'] == 5, 2,
np.where(base_dados['NUPARCELA_gh63'] == 8, 3,
np.where(base_dados['NUPARCELA_gh63'] == 11, 4,
np.where(base_dados['NUPARCELA_gh63'] == 12, 5,
0))))))

base_dados['NUPARCELA_gh65'] = np.where(base_dados['NUPARCELA_gh64'] == 0, 0,
np.where(base_dados['NUPARCELA_gh64'] == 1, 1,
np.where(base_dados['NUPARCELA_gh64'] == 2, 2,
np.where(base_dados['NUPARCELA_gh64'] == 3, 3,
np.where(base_dados['NUPARCELA_gh64'] == 4, 4,
np.where(base_dados['NUPARCELA_gh64'] == 5, 5,
0))))))

base_dados['NUPARCELA_gh66'] = np.where(base_dados['NUPARCELA_gh65'] == 0, 0,
np.where(base_dados['NUPARCELA_gh65'] == 1, 0,
np.where(base_dados['NUPARCELA_gh65'] == 2, 2,
np.where(base_dados['NUPARCELA_gh65'] == 3, 3,
np.where(base_dados['NUPARCELA_gh65'] == 4, 3,
np.where(base_dados['NUPARCELA_gh65'] == 5, 3,
0))))))

base_dados['NUPARCELA_gh67'] = np.where(base_dados['NUPARCELA_gh66'] == 0, 0,
np.where(base_dados['NUPARCELA_gh66'] == 2, 1,
np.where(base_dados['NUPARCELA_gh66'] == 3, 2,
0)))

base_dados['NUPARCELA_gh68'] = np.where(base_dados['NUPARCELA_gh67'] == 0, 0,
np.where(base_dados['NUPARCELA_gh67'] == 1, 1,
np.where(base_dados['NUPARCELA_gh67'] == 2, 2,
0)))

base_dados['NUPARCELA_gh69'] = np.where(base_dados['NUPARCELA_gh68'] == 0, 0,
np.where(base_dados['NUPARCELA_gh68'] == 1, 1,
np.where(base_dados['NUPARCELA_gh68'] == 2, 2,
0)))

base_dados['NUPARCELA_gh70'] = np.where(base_dados['NUPARCELA_gh69'] == 0, 1,
np.where(base_dados['NUPARCELA_gh69'] == 1, 1,
np.where(base_dados['NUPARCELA_gh69'] == 2, 2,
0)))

base_dados['NUPARCELA_gh71'] = np.where(base_dados['NUPARCELA_gh70'] == 1, 0,
np.where(base_dados['NUPARCELA_gh70'] == 2, 1,
0))



# COMMAND ----------

# MAGIC %md
# MAGIC ## Gerando variáveis numéricas contínuas

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parte 1 de 2

# COMMAND ----------

base_dados['NUCPFCNPJ__pk_7'] = np.where(base_dados['NUCPFCNPJ'] <= 9016041605.0, 0.0,
np.where(np.bitwise_and(base_dados['NUCPFCNPJ'] > 9016041605.0, base_dados['NUCPFCNPJ'] <= 23337165842.0), 1.0,
np.where(np.bitwise_and(base_dados['NUCPFCNPJ'] > 23337165842.0, base_dados['NUCPFCNPJ'] <= 38626179120.0), 2.0,
np.where(np.bitwise_and(base_dados['NUCPFCNPJ'] > 38626179120.0, base_dados['NUCPFCNPJ'] <= 53487176734.0), 3.0,
np.where(np.bitwise_and(base_dados['NUCPFCNPJ'] > 53487176734.0, base_dados['NUCPFCNPJ'] <= 70852081987.0), 4.0,
np.where(np.bitwise_and(base_dados['NUCPFCNPJ'] > 70852081987.0, base_dados['NUCPFCNPJ'] <= 85361046787.0), 5.0,
np.where(base_dados['NUCPFCNPJ'] > 85361046787.0, 6.0,
 0)))))))

base_dados['NUCPFCNPJ__pk_7_g_1_1'] = np.where(base_dados['NUCPFCNPJ__pk_7'] == 0.0, 0,
np.where(base_dados['NUCPFCNPJ__pk_7'] == 1.0, 1,
np.where(base_dados['NUCPFCNPJ__pk_7'] == 2.0, 2,
np.where(base_dados['NUCPFCNPJ__pk_7'] == 3.0, 2,
np.where(base_dados['NUCPFCNPJ__pk_7'] == 4.0, 2,
np.where(base_dados['NUCPFCNPJ__pk_7'] == 5.0, 3,
np.where(base_dados['NUCPFCNPJ__pk_7'] == 6.0, 3,
 0)))))))
         
base_dados['NUCPFCNPJ__pk_7_g_1_2'] = np.where(base_dados['NUCPFCNPJ__pk_7_g_1_1'] == 0, 3,
np.where(base_dados['NUCPFCNPJ__pk_7_g_1_1'] == 1, 0,
np.where(base_dados['NUCPFCNPJ__pk_7_g_1_1'] == 2, 2,
np.where(base_dados['NUCPFCNPJ__pk_7_g_1_1'] == 3, 1,
 0))))
base_dados['NUCPFCNPJ__pk_7_g_1'] = np.where(base_dados['NUCPFCNPJ__pk_7_g_1_2'] == 0, 0,
np.where(base_dados['NUCPFCNPJ__pk_7_g_1_2'] == 1, 1,
np.where(base_dados['NUCPFCNPJ__pk_7_g_1_2'] == 2, 2,
np.where(base_dados['NUCPFCNPJ__pk_7_g_1_2'] == 3, 3,
 0))))





base_dados['NUCPFCNPJ__S'] = np.sin(base_dados['NUCPFCNPJ'])
np.where(base_dados['NUCPFCNPJ__S'] == 0, -1, base_dados['NUCPFCNPJ__S'])
base_dados['NUCPFCNPJ__S'] = base_dados['NUCPFCNPJ__S'].fillna(-2)
base_dados['NUCPFCNPJ__S__p_6'] = np.where(base_dados['NUCPFCNPJ__S'] <= -0.8807325808850475, 0.0,
np.where(np.bitwise_and(base_dados['NUCPFCNPJ__S'] > -0.8807325808850475, base_dados['NUCPFCNPJ__S'] <= -0.5407915613305772), 1.0,
np.where(np.bitwise_and(base_dados['NUCPFCNPJ__S'] > -0.5407915613305772, base_dados['NUCPFCNPJ__S'] <= -0.06049334686066287), 2.0,
np.where(np.bitwise_and(base_dados['NUCPFCNPJ__S'] > -0.06049334686066287, base_dados['NUCPFCNPJ__S'] <= 0.36834702075723075), 3.0,
np.where(np.bitwise_and(base_dados['NUCPFCNPJ__S'] > 0.36834702075723075, base_dados['NUCPFCNPJ__S'] <= 0.847029902662262), 4.0,
np.where(base_dados['NUCPFCNPJ__S'] > 0.847029902662262, 5.0,
 0))))))

base_dados['NUCPFCNPJ__S__p_6_g_1_1'] = np.where(base_dados['NUCPFCNPJ__S__p_6'] == 0.0, 1,
np.where(base_dados['NUCPFCNPJ__S__p_6'] == 1.0, 2,
np.where(base_dados['NUCPFCNPJ__S__p_6'] == 2.0, 0,
np.where(base_dados['NUCPFCNPJ__S__p_6'] == 3.0, 0,
np.where(base_dados['NUCPFCNPJ__S__p_6'] == 4.0, 2,
np.where(base_dados['NUCPFCNPJ__S__p_6'] == 5.0, 1,
 0))))))

base_dados['NUCPFCNPJ__S__p_6_g_1_2'] = np.where(base_dados['NUCPFCNPJ__S__p_6_g_1_1'] == 0, 0,
np.where(base_dados['NUCPFCNPJ__S__p_6_g_1_1'] == 1, 2,
np.where(base_dados['NUCPFCNPJ__S__p_6_g_1_1'] == 2, 0,
 0)))

base_dados['NUCPFCNPJ__S__p_6_g_1'] = np.where(base_dados['NUCPFCNPJ__S__p_6_g_1_2'] == 0, 0,
np.where(base_dados['NUCPFCNPJ__S__p_6_g_1_2'] == 2, 1,
 0))







base_dados['NUCONTRATO__pe_15'] = np.where(np.bitwise_and(base_dados['NUCONTRATO'] >= -3.0, base_dados['NUCONTRATO'] <= 10274000030254.0), 11.0,
np.where(np.bitwise_and(base_dados['NUCONTRATO'] > 10274000030254.0, base_dados['NUCONTRATO'] <= 11032000086557.0), 12.0,
np.where(np.bitwise_and(base_dados['NUCONTRATO'] > 11032000086557.0, base_dados['NUCONTRATO'] <= 12145000087064.0), 13.0,
np.where(base_dados['NUCONTRATO'] > 12145000087064.0, 14.0,
 -2))))

base_dados['NUCONTRATO__pe_15_g_1_1'] = np.where(base_dados['NUCONTRATO__pe_15'] == -2.0, 0,
np.where(base_dados['NUCONTRATO__pe_15'] == 11.0, 2,
np.where(base_dados['NUCONTRATO__pe_15'] == 12.0, 1,
np.where(base_dados['NUCONTRATO__pe_15'] == 13.0, 1,
np.where(base_dados['NUCONTRATO__pe_15'] == 14.0, 2,
 0)))))

base_dados['NUCONTRATO__pe_15_g_1_2'] = np.where(base_dados['NUCONTRATO__pe_15_g_1_1'] == 0, 0,
np.where(base_dados['NUCONTRATO__pe_15_g_1_1'] == 1, 2,
np.where(base_dados['NUCONTRATO__pe_15_g_1_1'] == 2, 1,
 0)))

base_dados['NUCONTRATO__pe_15_g_1'] = np.where(base_dados['NUCONTRATO__pe_15_g_1_2'] == 0, 0,
np.where(base_dados['NUCONTRATO__pe_15_g_1_2'] == 1, 1,
np.where(base_dados['NUCONTRATO__pe_15_g_1_2'] == 2, 2,
 0)))







base_dados['NUCONTRATO__C'] = np.cos(base_dados['NUCONTRATO'])
np.where(base_dados['NUCONTRATO__C'] == 0, -1, base_dados['NUCONTRATO__C'])
base_dados['NUCONTRATO__C'] = base_dados['NUCONTRATO__C'].fillna(-2)
base_dados['NUCONTRATO__C__p_8'] = np.where(base_dados['NUCONTRATO__C'] <= -0.9901738175353013, 0.0,
np.where(np.bitwise_and(base_dados['NUCONTRATO__C'] > -0.9901738175353013, base_dados['NUCONTRATO__C'] <= -0.8026907224863737), 1.0,
np.where(np.bitwise_and(base_dados['NUCONTRATO__C'] > -0.8026907224863737, base_dados['NUCONTRATO__C'] <= -0.44692200420336786), 2.0,
np.where(np.bitwise_and(base_dados['NUCONTRATO__C'] > -0.44692200420336786, base_dados['NUCONTRATO__C'] <= 0.01928047465214817), 3.0,
np.where(np.bitwise_and(base_dados['NUCONTRATO__C'] > 0.01928047465214817, base_dados['NUCONTRATO__C'] <= 0.5600393935578594), 4.0,
np.where(np.bitwise_and(base_dados['NUCONTRATO__C'] > 0.5600393935578594, base_dados['NUCONTRATO__C'] <= 0.8688827586246344), 5.0,
np.where(base_dados['NUCONTRATO__C'] > 0.8688827586246344, 6.0,
 0)))))))

base_dados['NUCONTRATO__C__p_8_g_1_1'] = np.where(base_dados['NUCONTRATO__C__p_8'] == 0.0, 1,
np.where(base_dados['NUCONTRATO__C__p_8'] == 1.0, 0,
np.where(base_dados['NUCONTRATO__C__p_8'] == 2.0, 1,
np.where(base_dados['NUCONTRATO__C__p_8'] == 3.0, 1,
np.where(base_dados['NUCONTRATO__C__p_8'] == 4.0, 1,
np.where(base_dados['NUCONTRATO__C__p_8'] == 5.0, 1,
np.where(base_dados['NUCONTRATO__C__p_8'] == 6.0, 1,
 0)))))))

base_dados['NUCONTRATO__C__p_8_g_1_2'] = np.where(base_dados['NUCONTRATO__C__p_8_g_1_1'] == 0, 0,
np.where(base_dados['NUCONTRATO__C__p_8_g_1_1'] == 1, 1,
 0))

base_dados['NUCONTRATO__C__p_8_g_1'] = np.where(base_dados['NUCONTRATO__C__p_8_g_1_2'] == 0, 0,
np.where(base_dados['NUCONTRATO__C__p_8_g_1_2'] == 1, 1,
 0))






base_dados['VLR_RISCO__R'] = np.sqrt(base_dados['VLR_RISCO'])
np.where(base_dados['VLR_RISCO__R'] == 0, -1, base_dados['VLR_RISCO__R'])
base_dados['VLR_RISCO__R'] = base_dados['VLR_RISCO__R'].fillna(-2)
base_dados['VLR_RISCO__R__pu_10'] = np.where(base_dados['VLR_RISCO__R'] <= 37.69403135776273, 0.0,
np.where(np.bitwise_and(base_dados['VLR_RISCO__R'] > 37.69403135776273, base_dados['VLR_RISCO__R'] <= 71.48524323243225), 1.0,
np.where(np.bitwise_and(base_dados['VLR_RISCO__R'] > 71.48524323243225, base_dados['VLR_RISCO__R'] <= 105.20979041895293), 2.0,
np.where(np.bitwise_and(base_dados['VLR_RISCO__R'] > 105.20979041895293, base_dados['VLR_RISCO__R'] <= 138.80201727640704), 3.0,
np.where(np.bitwise_and(base_dados['VLR_RISCO__R'] > 138.80201727640704, base_dados['VLR_RISCO__R'] <= 172.5), 4.0,
np.where(np.bitwise_and(base_dados['VLR_RISCO__R'] > 172.5, base_dados['VLR_RISCO__R'] <= 205.951450589696), 5.0,
np.where(np.bitwise_and(base_dados['VLR_RISCO__R'] > 205.951450589696, base_dados['VLR_RISCO__R'] <= 240.89831879861677), 6.0,
np.where(np.bitwise_and(base_dados['VLR_RISCO__R'] > 240.89831879861677, base_dados['VLR_RISCO__R'] <= 261.1857576515228), 7.0,
np.where(np.bitwise_and(base_dados['VLR_RISCO__R'] > 261.1857576515228, base_dados['VLR_RISCO__R'] <= 301.4247169692625), 8.0,
np.where(base_dados['VLR_RISCO__R'] > 301.4247169692625, 9.0,
 0))))))))))

base_dados['VLR_RISCO__R__pu_10_g_1_1'] = np.where(base_dados['VLR_RISCO__R__pu_10'] == 0.0, 0,
np.where(base_dados['VLR_RISCO__R__pu_10'] == 1.0, 2,
np.where(base_dados['VLR_RISCO__R__pu_10'] == 2.0, 1,
np.where(base_dados['VLR_RISCO__R__pu_10'] == 3.0, 1,
np.where(base_dados['VLR_RISCO__R__pu_10'] == 4.0, 2,
np.where(base_dados['VLR_RISCO__R__pu_10'] == 5.0, 2,
np.where(base_dados['VLR_RISCO__R__pu_10'] == 6.0, 2,
np.where(base_dados['VLR_RISCO__R__pu_10'] == 7.0, 2,
np.where(base_dados['VLR_RISCO__R__pu_10'] == 8.0, 2,
np.where(base_dados['VLR_RISCO__R__pu_10'] == 9.0, 2,
 0))))))))))

base_dados['VLR_RISCO__R__pu_10_g_1_2'] = np.where(base_dados['VLR_RISCO__R__pu_10_g_1_1'] == 0, 2,
np.where(base_dados['VLR_RISCO__R__pu_10_g_1_1'] == 1, 0,
np.where(base_dados['VLR_RISCO__R__pu_10_g_1_1'] == 2, 1,
 0)))

base_dados['VLR_RISCO__R__pu_10_g_1'] = np.where(base_dados['VLR_RISCO__R__pu_10_g_1_2'] == 0, 0,
np.where(base_dados['VLR_RISCO__R__pu_10_g_1_2'] == 1, 1,
np.where(base_dados['VLR_RISCO__R__pu_10_g_1_2'] == 2, 2,
 0)))






base_dados['VLR_RISCO__S'] = np.sin(base_dados['VLR_RISCO'])
np.where(base_dados['VLR_RISCO__S'] == 0, -1, base_dados['VLR_RISCO__S'])
base_dados['VLR_RISCO__S'] = base_dados['VLR_RISCO__S'].fillna(-2)
base_dados['VLR_RISCO__S__p_5'] = np.where(base_dados['VLR_RISCO__S'] <= -0.8518511078588149, 0.0,
np.where(np.bitwise_and(base_dados['VLR_RISCO__S'] > -0.8518511078588149, base_dados['VLR_RISCO__S'] <= -0.34188883060111874), 1.0,
np.where(np.bitwise_and(base_dados['VLR_RISCO__S'] > -0.34188883060111874, base_dados['VLR_RISCO__S'] <= 0.279098933649483), 2.0,
np.where(np.bitwise_and(base_dados['VLR_RISCO__S'] > 0.279098933649483, base_dados['VLR_RISCO__S'] <= 0.7680303971607655), 3.0,
np.where(base_dados['VLR_RISCO__S'] > 0.7680303971607655, 4.0,
 0)))))

base_dados['VLR_RISCO__S__p_5_g_1_1'] = np.where(base_dados['VLR_RISCO__S__p_5'] == 0.0, 1,
np.where(base_dados['VLR_RISCO__S__p_5'] == 1.0, 0,
np.where(base_dados['VLR_RISCO__S__p_5'] == 2.0, 1,
np.where(base_dados['VLR_RISCO__S__p_5'] == 3.0, 1,
np.where(base_dados['VLR_RISCO__S__p_5'] == 4.0, 0,
 0)))))

base_dados['VLR_RISCO__S__p_5_g_1_2'] = np.where(base_dados['VLR_RISCO__S__p_5_g_1_1'] == 0, 0,
np.where(base_dados['VLR_RISCO__S__p_5_g_1_1'] == 1, 1,
 0))

base_dados['VLR_RISCO__S__p_5_g_1'] = np.where(base_dados['VLR_RISCO__S__p_5_g_1_2'] == 0, 0,
np.where(base_dados['VLR_RISCO__S__p_5_g_1_2'] == 1, 1,
 0))








base_dados['ATRASO__pk_4'] = np.where(base_dados['ATRASO'] <= 1045.0, 0.0,
np.where(np.bitwise_and(base_dados['ATRASO'] > 1045.0, base_dados['ATRASO'] <= 1852.0), 1.0,
np.where(np.bitwise_and(base_dados['ATRASO'] > 1852.0, base_dados['ATRASO'] <= 2736.0), 2.0,
np.where(base_dados['ATRASO'] > 2736.0, 3.0,
 0))))

base_dados['ATRASO__pk_4_g_1_1'] = np.where(base_dados['ATRASO__pk_4'] == 0.0, 1,
np.where(base_dados['ATRASO__pk_4'] == 1.0, 3,
np.where(base_dados['ATRASO__pk_4'] == 2.0, 0,
np.where(base_dados['ATRASO__pk_4'] == 3.0, 2,
 0))))

base_dados['ATRASO__pk_4_g_1_2'] = np.where(base_dados['ATRASO__pk_4_g_1_1'] == 0, 0,
np.where(base_dados['ATRASO__pk_4_g_1_1'] == 1, 3,
np.where(base_dados['ATRASO__pk_4_g_1_1'] == 2, 1,
np.where(base_dados['ATRASO__pk_4_g_1_1'] == 3, 2,
 0))))

base_dados['ATRASO__pk_4_g_1'] = np.where(base_dados['ATRASO__pk_4_g_1_2'] == 0, 0,
np.where(base_dados['ATRASO__pk_4_g_1_2'] == 1, 1,
np.where(base_dados['ATRASO__pk_4_g_1_2'] == 2, 2,
np.where(base_dados['ATRASO__pk_4_g_1_2'] == 3, 3,
 0))))





base_dados['ATRASO__pe_4'] = np.where(base_dados['ATRASO'] <= 763.0, 0.0,
np.where(np.bitwise_and(base_dados['ATRASO'] > 763.0, base_dados['ATRASO'] <= 1548.0), 1.0,
np.where(np.bitwise_and(base_dados['ATRASO'] > 1548.0, base_dados['ATRASO'] <= 2327.0), 2.0,
np.where(np.bitwise_and(base_dados['ATRASO'] > 2327.0, base_dados['ATRASO'] <= 3103.0), 3.0,
 -2))))

base_dados['ATRASO__pe_4_g_1_1'] = np.where(base_dados['ATRASO__pe_4'] == -2.0, 1,
np.where(base_dados['ATRASO__pe_4'] == 0.0, 0,
np.where(base_dados['ATRASO__pe_4'] == 1.0, 0,
np.where(base_dados['ATRASO__pe_4'] == 2.0, 0,
np.where(base_dados['ATRASO__pe_4'] == 3.0, 1,
 0)))))

base_dados['ATRASO__pe_4_g_1_2'] = np.where(base_dados['ATRASO__pe_4_g_1_1'] == 0, 1,
np.where(base_dados['ATRASO__pe_4_g_1_1'] == 1, 0,
 0))
base_dados['ATRASO__pe_4_g_1'] = np.where(base_dados['ATRASO__pe_4_g_1_2'] == 0, 0,
np.where(base_dados['ATRASO__pe_4_g_1_2'] == 1, 1,
 0))






base_dados['Telefone1__L'] = np.log(base_dados['Telefone1'])
np.where(base_dados['Telefone1__L'] == 0, -1, base_dados['Telefone1__L'])
base_dados['Telefone1__L'] = base_dados['Telefone1__L'].fillna(-2)
base_dados['Telefone1__L__pk_10'] = np.where(base_dados['Telefone1__L'] <= -2.0, 0.0,
np.where(np.bitwise_and(base_dados['Telefone1__L'] > -2.0, base_dados['Telefone1__L'] <= 21.501760000911116), 1.0,
np.where(np.bitwise_and(base_dados['Telefone1__L'] > 21.501760000911116, base_dados['Telefone1__L'] <= 22.319484232934645), 2.0,
np.where(np.bitwise_and(base_dados['Telefone1__L'] > 22.319484232934645, base_dados['Telefone1__L'] <= 22.715648120308508), 3.0,
np.where(np.bitwise_and(base_dados['Telefone1__L'] > 22.715648120308508, base_dados['Telefone1__L'] <= 22.831673800348003), 4.0,
np.where(np.bitwise_and(base_dados['Telefone1__L'] > 22.831673800348003, base_dados['Telefone1__L'] <= 22.867417663449093), 5.0,
np.where(np.bitwise_and(base_dados['Telefone1__L'] > 22.867417663449093, base_dados['Telefone1__L'] <= 23.009404833735886), 6.0,
np.where(np.bitwise_and(base_dados['Telefone1__L'] > 23.009404833735886, base_dados['Telefone1__L'] <= 23.556328406872073), 7.0,
np.where(np.bitwise_and(base_dados['Telefone1__L'] > 23.556328406872073, base_dados['Telefone1__L'] <= 24.38682629374202), 8.0,
np.where(base_dados['Telefone1__L'] > 24.38682629374202, 9.0,
 0))))))))))

base_dados['Telefone1__L__pk_10_g_1_1'] = np.where(base_dados['Telefone1__L__pk_10'] == 0.0, 1,
np.where(base_dados['Telefone1__L__pk_10'] == 1.0, 2,
np.where(base_dados['Telefone1__L__pk_10'] == 2.0, 2,
np.where(base_dados['Telefone1__L__pk_10'] == 3.0, 2,
np.where(base_dados['Telefone1__L__pk_10'] == 4.0, 2,
np.where(base_dados['Telefone1__L__pk_10'] == 5.0, 2,
np.where(base_dados['Telefone1__L__pk_10'] == 6.0, 1,
np.where(base_dados['Telefone1__L__pk_10'] == 7.0, 1,
np.where(base_dados['Telefone1__L__pk_10'] == 8.0, 0,
np.where(base_dados['Telefone1__L__pk_10'] == 9.0, 0,
 0))))))))))

base_dados['Telefone1__L__pk_10_g_1_2'] = np.where(base_dados['Telefone1__L__pk_10_g_1_1'] == 0, 1,
np.where(base_dados['Telefone1__L__pk_10_g_1_1'] == 1, 2,
np.where(base_dados['Telefone1__L__pk_10_g_1_1'] == 2, 0,
 0)))

base_dados['Telefone1__L__pk_10_g_1'] = np.where(base_dados['Telefone1__L__pk_10_g_1_2'] == 0, 0,
np.where(base_dados['Telefone1__L__pk_10_g_1_2'] == 1, 1,
np.where(base_dados['Telefone1__L__pk_10_g_1_2'] == 2, 2,
 0)))






base_dados['Telefone1__T'] = np.tan(base_dados['Telefone1'])
np.where(base_dados['Telefone1__T'] == 0, -1, base_dados['Telefone1__T'])
base_dados['Telefone1__T'] = base_dados['Telefone1__T'].fillna(-2)
base_dados['Telefone1__T__p_7'] = np.where(base_dados['Telefone1__T'] <= -1.6048144992115452, 0.0,
np.where(np.bitwise_and(base_dados['Telefone1__T'] > -1.6048144992115452, base_dados['Telefone1__T'] <= -0.5740537113434184), 1.0,
np.where(np.bitwise_and(base_dados['Telefone1__T'] > -0.5740537113434184, base_dados['Telefone1__T'] <= 0.04962117265283505), 2.0,
np.where(np.bitwise_and(base_dados['Telefone1__T'] > 0.04962117265283505, base_dados['Telefone1__T'] <= 0.12646105627083848), 3.0,
np.where(np.bitwise_and(base_dados['Telefone1__T'] > 0.12646105627083848, base_dados['Telefone1__T'] <= 0.4618174697179093), 4.0,
np.where(np.bitwise_and(base_dados['Telefone1__T'] > 0.4618174697179093, base_dados['Telefone1__T'] <= 1.5834980478003287), 5.0,
np.where(base_dados['Telefone1__T'] > 1.5834980478003287, 6.0,
 0)))))))

base_dados['Telefone1__T__p_7_g_1_1'] = np.where(base_dados['Telefone1__T__p_7'] == 0.0, 2,
np.where(base_dados['Telefone1__T__p_7'] == 1.0, 0,
np.where(base_dados['Telefone1__T__p_7'] == 2.0, 0,
np.where(base_dados['Telefone1__T__p_7'] == 3.0, 2,
np.where(base_dados['Telefone1__T__p_7'] == 4.0, 1,
np.where(base_dados['Telefone1__T__p_7'] == 5.0, 0,
np.where(base_dados['Telefone1__T__p_7'] == 6.0, 0,
 0)))))))
base_dados['Telefone1__T__p_7_g_1_2'] = np.where(base_dados['Telefone1__T__p_7_g_1_1'] == 0, 1,
np.where(base_dados['Telefone1__T__p_7_g_1_1'] == 1, 2,
np.where(base_dados['Telefone1__T__p_7_g_1_1'] == 2, 0,
 0)))
base_dados['Telefone1__T__p_7_g_1'] = np.where(base_dados['Telefone1__T__p_7_g_1_2'] == 0, 0,
np.where(base_dados['Telefone1__T__p_7_g_1_2'] == 1, 1,
np.where(base_dados['Telefone1__T__p_7_g_1_2'] == 2, 2,
 0)))








base_dados['Telefone2__pk_25'] = np.where(base_dados['Telefone2'] <= -3.0, 0.0,
np.where(np.bitwise_and(base_dados['Telefone2'] > -3.0, base_dados['Telefone2'] <= 3532145808.0), 1.0,
np.where(np.bitwise_and(base_dados['Telefone2'] > 3532145808.0, base_dados['Telefone2'] <= 7736285724.0), 2.0,
np.where(np.bitwise_and(base_dados['Telefone2'] > 7736285724.0, base_dados['Telefone2'] <= 11998083547.0), 3.0,
np.where(np.bitwise_and(base_dados['Telefone2'] > 11998083547.0, base_dados['Telefone2'] <= 15998570398.0), 4.0,
np.where(np.bitwise_and(base_dados['Telefone2'] > 15998570398.0, base_dados['Telefone2'] <= 18997499629.0), 5.0,
np.where(np.bitwise_and(base_dados['Telefone2'] > 18997499629.0, base_dados['Telefone2'] <= 22999768739.0), 6.0,
np.where(np.bitwise_and(base_dados['Telefone2'] > 22999768739.0, base_dados['Telefone2'] <= 27998707426.0), 7.0,
np.where(np.bitwise_and(base_dados['Telefone2'] > 27998707426.0, base_dados['Telefone2'] <= 32999190258.0), 8.0,
np.where(np.bitwise_and(base_dados['Telefone2'] > 32999190258.0, base_dados['Telefone2'] <= 35997056126.0), 9.0,
np.where(np.bitwise_and(base_dados['Telefone2'] > 35997056126.0, base_dados['Telefone2'] <= 37999990935.0), 10.0,
np.where(np.bitwise_and(base_dados['Telefone2'] > 37999990935.0, base_dados['Telefone2'] <= 43998623294.0), 11.0,
np.where(np.bitwise_and(base_dados['Telefone2'] > 43998623294.0, base_dados['Telefone2'] <= 48988465471.0), 12.0,
np.where(np.bitwise_and(base_dados['Telefone2'] > 48988465471.0, base_dados['Telefone2'] <= 51999126773.0), 13.0,
np.where(np.bitwise_and(base_dados['Telefone2'] > 51999126773.0, base_dados['Telefone2'] <= 54996252058.0), 14.0,
np.where(np.bitwise_and(base_dados['Telefone2'] > 54996252058.0, base_dados['Telefone2'] <= 62981548225.0), 15.0,
np.where(np.bitwise_and(base_dados['Telefone2'] > 62981548225.0, base_dados['Telefone2'] <= 67998809345.0), 16.0,
np.where(np.bitwise_and(base_dados['Telefone2'] > 67998809345.0, base_dados['Telefone2'] <= 71999888172.0), 17.0,
np.where(np.bitwise_and(base_dados['Telefone2'] > 71999888172.0, base_dados['Telefone2'] <= 75988255632.0), 18.0,
np.where(np.bitwise_and(base_dados['Telefone2'] > 75988255632.0, base_dados['Telefone2'] <= 79996890267.0), 19.0,
np.where(np.bitwise_and(base_dados['Telefone2'] > 79996890267.0, base_dados['Telefone2'] <= 83988197014.0), 20.0,
np.where(np.bitwise_and(base_dados['Telefone2'] > 83988197014.0, base_dados['Telefone2'] <= 87988714179.0), 21.0,
np.where(np.bitwise_and(base_dados['Telefone2'] > 87988714179.0, base_dados['Telefone2'] <= 91999076136.0), 22.0,
np.where(np.bitwise_and(base_dados['Telefone2'] > 91999076136.0, base_dados['Telefone2'] <= 95991700221.0), 23.0,
np.where(base_dados['Telefone2'] > 95991700221.0, 24.0,
 0)))))))))))))))))))))))))

base_dados['Telefone2__pk_25_g_1_1'] = np.where(base_dados['Telefone2__pk_25'] == 0.0, 2,
np.where(base_dados['Telefone2__pk_25'] == 1.0, 1,
np.where(base_dados['Telefone2__pk_25'] == 2.0, 0,
np.where(base_dados['Telefone2__pk_25'] == 3.0, 1,
np.where(base_dados['Telefone2__pk_25'] == 4.0, 3,
np.where(base_dados['Telefone2__pk_25'] == 5.0, 3,
np.where(base_dados['Telefone2__pk_25'] == 6.0, 3,
np.where(base_dados['Telefone2__pk_25'] == 7.0, 3,
np.where(base_dados['Telefone2__pk_25'] == 8.0, 2,
np.where(base_dados['Telefone2__pk_25'] == 9.0, 3,
np.where(base_dados['Telefone2__pk_25'] == 10.0, 3,
np.where(base_dados['Telefone2__pk_25'] == 11.0, 2,
np.where(base_dados['Telefone2__pk_25'] == 12.0, 1,
np.where(base_dados['Telefone2__pk_25'] == 13.0, 2,
np.where(base_dados['Telefone2__pk_25'] == 14.0, 3,
np.where(base_dados['Telefone2__pk_25'] == 15.0, 2,
np.where(base_dados['Telefone2__pk_25'] == 16.0, 3,
np.where(base_dados['Telefone2__pk_25'] == 17.0, 2,
np.where(base_dados['Telefone2__pk_25'] == 18.0, 3,
np.where(base_dados['Telefone2__pk_25'] == 19.0, 3,
np.where(base_dados['Telefone2__pk_25'] == 20.0, 3,
np.where(base_dados['Telefone2__pk_25'] == 21.0, 2,
np.where(base_dados['Telefone2__pk_25'] == 22.0, 2,
np.where(base_dados['Telefone2__pk_25'] == 23.0, 3,
np.where(base_dados['Telefone2__pk_25'] == 24.0, 3,
 0)))))))))))))))))))))))))

base_dados['Telefone2__pk_25_g_1_2'] = np.where(base_dados['Telefone2__pk_25_g_1_1'] == 0, 0,
np.where(base_dados['Telefone2__pk_25_g_1_1'] == 1, 1,
np.where(base_dados['Telefone2__pk_25_g_1_1'] == 2, 3,
np.where(base_dados['Telefone2__pk_25_g_1_1'] == 3, 1,
 0))))

base_dados['Telefone2__pk_25_g_1'] = np.where(base_dados['Telefone2__pk_25_g_1_2'] == 0, 0,
np.where(base_dados['Telefone2__pk_25_g_1_2'] == 1, 1,
np.where(base_dados['Telefone2__pk_25_g_1_2'] == 3, 2,
 0)))








base_dados['Telefone2__S'] = np.sin(base_dados['Telefone2'])
np.where(base_dados['Telefone2__S'] == 0, -1, base_dados['Telefone2__S'])
base_dados['Telefone2__S'] = base_dados['Telefone2__S'].fillna(-2)
base_dados['Telefone2__S__pe_10'] = np.where(np.bitwise_and(base_dados['Telefone2__S'] >= -0.999880195686407, base_dados['Telefone2__S'] <= 0.1975998827217302), 0.0,
np.where(np.bitwise_and(base_dados['Telefone2__S'] > 0.1975998827217302, base_dados['Telefone2__S'] <= 0.3943589729376371), 1.0,
np.where(np.bitwise_and(base_dados['Telefone2__S'] > 0.3943589729376371, base_dados['Telefone2__S'] <= 0.5963475086967533), 2.0,
np.where(np.bitwise_and(base_dados['Telefone2__S'] > 0.5963475086967533, base_dados['Telefone2__S'] <= 0.7949156160994627), 3.0,
np.where(np.bitwise_and(base_dados['Telefone2__S'] > 0.7949156160994627, base_dados['Telefone2__S'] <= 0.9948177203220375), 4.0,
np.where(base_dados['Telefone2__S'] > 0.9948177203220375, 5.0,
 -2))))))

base_dados['Telefone2__S__pe_10_g_1_1'] = np.where(base_dados['Telefone2__S__pe_10'] == -2.0, 1,
np.where(base_dados['Telefone2__S__pe_10'] == 0.0, 1,
np.where(base_dados['Telefone2__S__pe_10'] == 1.0, 1,
np.where(base_dados['Telefone2__S__pe_10'] == 2.0, 1,
np.where(base_dados['Telefone2__S__pe_10'] == 3.0, 1,
np.where(base_dados['Telefone2__S__pe_10'] == 4.0, 0,
np.where(base_dados['Telefone2__S__pe_10'] == 5.0, 1,
 0)))))))
base_dados['Telefone2__S__pe_10_g_1_2'] = np.where(base_dados['Telefone2__S__pe_10_g_1_1'] == 0, 0,
np.where(base_dados['Telefone2__S__pe_10_g_1_1'] == 1, 1,
 0))
base_dados['Telefone2__S__pe_10_g_1'] = np.where(base_dados['Telefone2__S__pe_10_g_1_2'] == 0, 0,
np.where(base_dados['Telefone2__S__pe_10_g_1_2'] == 1, 1,
 0))






base_dados['Telefone3__p_17'] = np.where(base_dados['Telefone3'] <= -3.0, 0.0,
np.where(np.bitwise_and(base_dados['Telefone3'] > -3.0, base_dados['Telefone3'] <= 2122559855.0), 1.0,
np.where(np.bitwise_and(base_dados['Telefone3'] > 2122559855.0, base_dados['Telefone3'] <= 4234226163.0), 2.0,
np.where(np.bitwise_and(base_dados['Telefone3'] > 4234226163.0, base_dados['Telefone3'] <= 8132054551.0), 3.0,
np.where(np.bitwise_and(base_dados['Telefone3'] > 8132054551.0, base_dados['Telefone3'] <= 15996664769.0), 4.0,
np.where(np.bitwise_and(base_dados['Telefone3'] > 15996664769.0, base_dados['Telefone3'] <= 41996386948.0), 5.0,
np.where(np.bitwise_and(base_dados['Telefone3'] > 41996386948.0, base_dados['Telefone3'] <= 74988499256.0), 6.0,
np.where(base_dados['Telefone3'] > 74988499256.0, 7.0,
 0))))))))

base_dados['Telefone3__p_17_g_1_1'] = np.where(base_dados['Telefone3__p_17'] == 0.0, 0,
np.where(base_dados['Telefone3__p_17'] == 1.0, 1,
np.where(base_dados['Telefone3__p_17'] == 2.0, 1,
np.where(base_dados['Telefone3__p_17'] == 3.0, 1,
np.where(base_dados['Telefone3__p_17'] == 4.0, 1,
np.where(base_dados['Telefone3__p_17'] == 5.0, 1,
np.where(base_dados['Telefone3__p_17'] == 6.0, 1,
np.where(base_dados['Telefone3__p_17'] == 7.0, 1,
 0))))))))

base_dados['Telefone3__p_17_g_1_2'] = np.where(base_dados['Telefone3__p_17_g_1_1'] == 0, 1,
np.where(base_dados['Telefone3__p_17_g_1_1'] == 1, 0,
 0))

base_dados['Telefone3__p_17_g_1'] = np.where(base_dados['Telefone3__p_17_g_1_2'] == 0, 0,
np.where(base_dados['Telefone3__p_17_g_1_2'] == 1, 1,
 0))







base_dados['Telefone3__T'] = np.tan(base_dados['Telefone3'])
np.where(base_dados['Telefone3__T'] == 0, -1, base_dados['Telefone3__T'])
base_dados['Telefone3__T'] = base_dados['Telefone3__T'].fillna(-2)
base_dados['Telefone3__T__pk_40'] = np.where(base_dados['Telefone3__T'] <= -71.52342075903928, 0.0,
np.where(np.bitwise_and(base_dados['Telefone3__T'] > -71.52342075903928, base_dados['Telefone3__T'] <= -38.10319208680386), 1.0,
np.where(np.bitwise_and(base_dados['Telefone3__T'] > -38.10319208680386, base_dados['Telefone3__T'] <= -13.812443032388849), 2.0,
np.where(np.bitwise_and(base_dados['Telefone3__T'] > -13.812443032388849, base_dados['Telefone3__T'] <= -8.276208980223824), 3.0,
np.where(np.bitwise_and(base_dados['Telefone3__T'] > -8.276208980223824, base_dados['Telefone3__T'] <= -5.51757174311117), 4.0,
np.where(np.bitwise_and(base_dados['Telefone3__T'] > -5.51757174311117, base_dados['Telefone3__T'] <= -3.790647944644318), 5.0,
np.where(np.bitwise_and(base_dados['Telefone3__T'] > -3.790647944644318, base_dados['Telefone3__T'] <= -2.803547636784186), 6.0,
np.where(np.bitwise_and(base_dados['Telefone3__T'] > -2.803547636784186, base_dados['Telefone3__T'] <= -1.9917542642051573), 7.0,
np.where(np.bitwise_and(base_dados['Telefone3__T'] > -1.9917542642051573, base_dados['Telefone3__T'] <= -1.7335520971186664), 8.0,
np.where(np.bitwise_and(base_dados['Telefone3__T'] > -1.7335520971186664, base_dados['Telefone3__T'] <= -1.6703432662271647), 9.0,
np.where(np.bitwise_and(base_dados['Telefone3__T'] > -1.6703432662271647, base_dados['Telefone3__T'] <= -1.581208951744476), 10.0,
np.where(np.bitwise_and(base_dados['Telefone3__T'] > -1.581208951744476, base_dados['Telefone3__T'] <= -1.2018218673824594), 11.0,
np.where(np.bitwise_and(base_dados['Telefone3__T'] > -1.2018218673824594, base_dados['Telefone3__T'] <= -0.7425930529451497), 12.0,
np.where(np.bitwise_and(base_dados['Telefone3__T'] > -0.7425930529451497, base_dados['Telefone3__T'] <= -0.15355569790120438), 13.0,
np.where(np.bitwise_and(base_dados['Telefone3__T'] > -0.15355569790120438, base_dados['Telefone3__T'] <= 0.34831122992692043), 14.0,
np.where(np.bitwise_and(base_dados['Telefone3__T'] > 0.34831122992692043, base_dados['Telefone3__T'] <= 0.7272447068522524), 15.0,
np.where(np.bitwise_and(base_dados['Telefone3__T'] > 0.7272447068522524, base_dados['Telefone3__T'] <= 0.9965709438654337), 16.0,
np.where(np.bitwise_and(base_dados['Telefone3__T'] > 0.9965709438654337, base_dados['Telefone3__T'] <= 1.2330956514519251), 17.0,
np.where(np.bitwise_and(base_dados['Telefone3__T'] > 1.2330956514519251, base_dados['Telefone3__T'] <= 1.342140641245824), 18.0,
np.where(np.bitwise_and(base_dados['Telefone3__T'] > 1.342140641245824, base_dados['Telefone3__T'] <= 1.4447825855134684), 19.0,
np.where(np.bitwise_and(base_dados['Telefone3__T'] > 1.4447825855134684, base_dados['Telefone3__T'] <= 1.4680043376160934), 20.0,
np.where(np.bitwise_and(base_dados['Telefone3__T'] > 1.4680043376160934, base_dados['Telefone3__T'] <= 1.5238669947489107), 21.0,
np.where(np.bitwise_and(base_dados['Telefone3__T'] > 1.5238669947489107, base_dados['Telefone3__T'] <= 1.617933150055185), 22.0,
np.where(np.bitwise_and(base_dados['Telefone3__T'] > 1.617933150055185, base_dados['Telefone3__T'] <= 1.638420809536175), 23.0,
np.where(np.bitwise_and(base_dados['Telefone3__T'] > 1.638420809536175, base_dados['Telefone3__T'] <= 1.718161831443783), 24.0,
np.where(np.bitwise_and(base_dados['Telefone3__T'] > 1.718161831443783, base_dados['Telefone3__T'] <= 2.1118146149810535), 25.0,
np.where(np.bitwise_and(base_dados['Telefone3__T'] > 2.1118146149810535, base_dados['Telefone3__T'] <= 2.8531423986067033), 26.0,
np.where(np.bitwise_and(base_dados['Telefone3__T'] > 2.8531423986067033, base_dados['Telefone3__T'] <= 3.6465878194346284), 27.0,
np.where(np.bitwise_and(base_dados['Telefone3__T'] > 3.6465878194346284, base_dados['Telefone3__T'] <= 4.530914493883977), 28.0,
np.where(np.bitwise_and(base_dados['Telefone3__T'] > 4.530914493883977, base_dados['Telefone3__T'] <= 5.585737097428541), 29.0,
np.where(np.bitwise_and(base_dados['Telefone3__T'] > 5.585737097428541, base_dados['Telefone3__T'] <= 6.592078822103194), 30.0,
np.where(np.bitwise_and(base_dados['Telefone3__T'] > 6.592078822103194, base_dados['Telefone3__T'] <= 7.73819059438191), 31.0,
np.where(np.bitwise_and(base_dados['Telefone3__T'] > 7.73819059438191, base_dados['Telefone3__T'] <= 9.155604925638393), 32.0,
np.where(np.bitwise_and(base_dados['Telefone3__T'] > 9.155604925638393, base_dados['Telefone3__T'] <= 13.910042835764706), 33.0,
np.where(np.bitwise_and(base_dados['Telefone3__T'] > 13.910042835764706, base_dados['Telefone3__T'] <= 16.93785216688903), 34.0,
np.where(np.bitwise_and(base_dados['Telefone3__T'] > 16.93785216688903, base_dados['Telefone3__T'] <= 17.669036417174258), 35.0,
np.where(np.bitwise_and(base_dados['Telefone3__T'] > 17.669036417174258, base_dados['Telefone3__T'] <= 24.816162139081683), 36.0,
np.where(np.bitwise_and(base_dados['Telefone3__T'] > 24.816162139081683, base_dados['Telefone3__T'] <= 34.42012221407448), 37.0,
np.where(np.bitwise_and(base_dados['Telefone3__T'] > 34.42012221407448, base_dados['Telefone3__T'] <= 42.053842339957654), 38.0,
np.where(base_dados['Telefone3__T'] > 42.053842339957654, 39.0,
 0))))))))))))))))))))))))))))))))))))))))
base_dados['Telefone3__T__pk_40_g_1_1'] = np.where(base_dados['Telefone3__T__pk_40'] == 0.0, 3,
np.where(base_dados['Telefone3__T__pk_40'] == 1.0, 2,
np.where(base_dados['Telefone3__T__pk_40'] == 2.0, 3,
np.where(base_dados['Telefone3__T__pk_40'] == 3.0, 3,
np.where(base_dados['Telefone3__T__pk_40'] == 4.0, 3,
np.where(base_dados['Telefone3__T__pk_40'] == 5.0, 3,
np.where(base_dados['Telefone3__T__pk_40'] == 6.0, 2,
np.where(base_dados['Telefone3__T__pk_40'] == 7.0, 2,
np.where(base_dados['Telefone3__T__pk_40'] == 8.0, 3,
np.where(base_dados['Telefone3__T__pk_40'] == 9.0, 3,
np.where(base_dados['Telefone3__T__pk_40'] == 10.0, 3,
np.where(base_dados['Telefone3__T__pk_40'] == 11.0, 2,
np.where(base_dados['Telefone3__T__pk_40'] == 12.0, 2,
np.where(base_dados['Telefone3__T__pk_40'] == 13.0, 1,
np.where(base_dados['Telefone3__T__pk_40'] == 14.0, 0,
np.where(base_dados['Telefone3__T__pk_40'] == 15.0, 1,
np.where(base_dados['Telefone3__T__pk_40'] == 16.0, 3,
np.where(base_dados['Telefone3__T__pk_40'] == 17.0, 2,
np.where(base_dados['Telefone3__T__pk_40'] == 18.0, 3,
np.where(base_dados['Telefone3__T__pk_40'] == 19.0, 3,
np.where(base_dados['Telefone3__T__pk_40'] == 20.0, 3,
np.where(base_dados['Telefone3__T__pk_40'] == 21.0, 3,
np.where(base_dados['Telefone3__T__pk_40'] == 22.0, 3,
np.where(base_dados['Telefone3__T__pk_40'] == 23.0, 3,
np.where(base_dados['Telefone3__T__pk_40'] == 24.0, 3,
np.where(base_dados['Telefone3__T__pk_40'] == 25.0, 2,
np.where(base_dados['Telefone3__T__pk_40'] == 26.0, 3,
np.where(base_dados['Telefone3__T__pk_40'] == 27.0, 3,
np.where(base_dados['Telefone3__T__pk_40'] == 28.0, 2,
np.where(base_dados['Telefone3__T__pk_40'] == 29.0, 2,
np.where(base_dados['Telefone3__T__pk_40'] == 30.0, 3,
np.where(base_dados['Telefone3__T__pk_40'] == 31.0, 3,
np.where(base_dados['Telefone3__T__pk_40'] == 32.0, 3,
np.where(base_dados['Telefone3__T__pk_40'] == 33.0, 2,
np.where(base_dados['Telefone3__T__pk_40'] == 34.0, 3,
np.where(base_dados['Telefone3__T__pk_40'] == 35.0, 3,
np.where(base_dados['Telefone3__T__pk_40'] == 36.0, 3,
np.where(base_dados['Telefone3__T__pk_40'] == 37.0, 3,
np.where(base_dados['Telefone3__T__pk_40'] == 38.0, 3,
np.where(base_dados['Telefone3__T__pk_40'] == 39.0, 3,
 0))))))))))))))))))))))))))))))))))))))))
base_dados['Telefone3__T__pk_40_g_1_2'] = np.where(base_dados['Telefone3__T__pk_40_g_1_1'] == 0, 2,
np.where(base_dados['Telefone3__T__pk_40_g_1_1'] == 1, 1,
np.where(base_dados['Telefone3__T__pk_40_g_1_1'] == 2, 3,
np.where(base_dados['Telefone3__T__pk_40_g_1_1'] == 3, 0,
 0))))
base_dados['Telefone3__T__pk_40_g_1'] = np.where(base_dados['Telefone3__T__pk_40_g_1_2'] == 0, 0,
np.where(base_dados['Telefone3__T__pk_40_g_1_2'] == 1, 1,
np.where(base_dados['Telefone3__T__pk_40_g_1_2'] == 2, 2,
np.where(base_dados['Telefone3__T__pk_40_g_1_2'] == 3, 3,
 0))))






base_dados['Telefone4__T'] = np.tan(base_dados['Telefone4'])
np.where(base_dados['Telefone4__T'] == 0, -1, base_dados['Telefone4__T'])
base_dados['Telefone4__T'] = base_dados['Telefone4__T'].fillna(-2)
base_dados['Telefone4__T__pk_25'] = np.where(base_dados['Telefone4__T'] <= -201.95895659945893, 0.0,
np.where(np.bitwise_and(base_dados['Telefone4__T'] > -201.95895659945893, base_dados['Telefone4__T'] <= -19.225876222764942), 1.0,
np.where(np.bitwise_and(base_dados['Telefone4__T'] > -19.225876222764942, base_dados['Telefone4__T'] <= -12.656067120489563), 2.0,
np.where(np.bitwise_and(base_dados['Telefone4__T'] > -12.656067120489563, base_dados['Telefone4__T'] <= -10.880543210029312), 3.0,
np.where(np.bitwise_and(base_dados['Telefone4__T'] > -10.880543210029312, base_dados['Telefone4__T'] <= -8.310152231241563), 4.0,
np.where(np.bitwise_and(base_dados['Telefone4__T'] > -8.310152231241563, base_dados['Telefone4__T'] <= -7.600988719127814), 5.0,
np.where(np.bitwise_and(base_dados['Telefone4__T'] > -7.600988719127814, base_dados['Telefone4__T'] <= -4.594237921982458), 6.0,
np.where(np.bitwise_and(base_dados['Telefone4__T'] > -4.594237921982458, base_dados['Telefone4__T'] <= -1.780729858359239), 7.0,
np.where(np.bitwise_and(base_dados['Telefone4__T'] > -1.780729858359239, base_dados['Telefone4__T'] <= -0.41376496585439554), 8.0,
np.where(np.bitwise_and(base_dados['Telefone4__T'] > -0.41376496585439554, base_dados['Telefone4__T'] <= 0.4067159899350505), 9.0,
np.where(np.bitwise_and(base_dados['Telefone4__T'] > 0.4067159899350505, base_dados['Telefone4__T'] <= 1.579296184246304), 10.0,
np.where(np.bitwise_and(base_dados['Telefone4__T'] > 1.579296184246304, base_dados['Telefone4__T'] <= 2.5449512987807883), 11.0,
np.where(np.bitwise_and(base_dados['Telefone4__T'] > 2.5449512987807883, base_dados['Telefone4__T'] <= 3.20285553041078), 12.0,
np.where(np.bitwise_and(base_dados['Telefone4__T'] > 3.20285553041078, base_dados['Telefone4__T'] <= 3.537737310525665), 13.0,
np.where(np.bitwise_and(base_dados['Telefone4__T'] > 3.537737310525665, base_dados['Telefone4__T'] <= 3.6396339905002772), 14.0,
np.where(np.bitwise_and(base_dados['Telefone4__T'] > 3.6396339905002772, base_dados['Telefone4__T'] <= 3.721790838508448), 15.0,
np.where(np.bitwise_and(base_dados['Telefone4__T'] > 3.721790838508448, base_dados['Telefone4__T'] <= 3.9395827435182684), 16.0,
np.where(np.bitwise_and(base_dados['Telefone4__T'] > 3.9395827435182684, base_dados['Telefone4__T'] <= 4.230690667811291), 17.0,
np.where(np.bitwise_and(base_dados['Telefone4__T'] > 4.230690667811291, base_dados['Telefone4__T'] <= 4.5330056039968), 18.0,
np.where(np.bitwise_and(base_dados['Telefone4__T'] > 4.5330056039968, base_dados['Telefone4__T'] <= 4.829838102597066), 19.0,
np.where(np.bitwise_and(base_dados['Telefone4__T'] > 4.829838102597066, base_dados['Telefone4__T'] <= 6.232011086488218), 20.0,
np.where(np.bitwise_and(base_dados['Telefone4__T'] > 6.232011086488218, base_dados['Telefone4__T'] <= 7.371907330344513), 21.0,
np.where(np.bitwise_and(base_dados['Telefone4__T'] > 7.371907330344513, base_dados['Telefone4__T'] <= 15.945836729906265), 22.0,
np.where(np.bitwise_and(base_dados['Telefone4__T'] > 15.945836729906265, base_dados['Telefone4__T'] <= 20.666009146296126), 23.0,
np.where(base_dados['Telefone4__T'] > 20.666009146296126, 24.0,
 0)))))))))))))))))))))))))
base_dados['Telefone4__T__pk_25_g_1_1'] = np.where(base_dados['Telefone4__T__pk_25'] == 0.0, 1,
np.where(base_dados['Telefone4__T__pk_25'] == 1.0, 1,
np.where(base_dados['Telefone4__T__pk_25'] == 2.0, 1,
np.where(base_dados['Telefone4__T__pk_25'] == 3.0, 1,
np.where(base_dados['Telefone4__T__pk_25'] == 4.0, 1,
np.where(base_dados['Telefone4__T__pk_25'] == 5.0, 1,
np.where(base_dados['Telefone4__T__pk_25'] == 6.0, 1,
np.where(base_dados['Telefone4__T__pk_25'] == 7.0, 0,
np.where(base_dados['Telefone4__T__pk_25'] == 8.0, 0,
np.where(base_dados['Telefone4__T__pk_25'] == 9.0, 0,
np.where(base_dados['Telefone4__T__pk_25'] == 10.0, 0,
np.where(base_dados['Telefone4__T__pk_25'] == 11.0, 1,
np.where(base_dados['Telefone4__T__pk_25'] == 12.0, 1,
np.where(base_dados['Telefone4__T__pk_25'] == 13.0, 1,
np.where(base_dados['Telefone4__T__pk_25'] == 14.0, 1,
np.where(base_dados['Telefone4__T__pk_25'] == 15.0, 1,
np.where(base_dados['Telefone4__T__pk_25'] == 16.0, 1,
np.where(base_dados['Telefone4__T__pk_25'] == 17.0, 1,
np.where(base_dados['Telefone4__T__pk_25'] == 18.0, 1,
np.where(base_dados['Telefone4__T__pk_25'] == 19.0, 1,
np.where(base_dados['Telefone4__T__pk_25'] == 20.0, 1,
np.where(base_dados['Telefone4__T__pk_25'] == 21.0, 1,
np.where(base_dados['Telefone4__T__pk_25'] == 22.0, 1,
np.where(base_dados['Telefone4__T__pk_25'] == 23.0, 1,
np.where(base_dados['Telefone4__T__pk_25'] == 24.0, 1,
 0)))))))))))))))))))))))))
base_dados['Telefone4__T__pk_25_g_1_2'] = np.where(base_dados['Telefone4__T__pk_25_g_1_1'] == 0, 0,
np.where(base_dados['Telefone4__T__pk_25_g_1_1'] == 1, 1,
 0))
base_dados['Telefone4__T__pk_25_g_1'] = np.where(base_dados['Telefone4__T__pk_25_g_1_2'] == 0, 0,
np.where(base_dados['Telefone4__T__pk_25_g_1_2'] == 1, 1,
 0))







base_dados['Telefone4__T'] = np.tan(base_dados['Telefone4'])
np.where(base_dados['Telefone4__T'] == 0, -1, base_dados['Telefone4__T'])
base_dados['Telefone4__T'] = base_dados['Telefone4__T'].fillna(-2)
base_dados['Telefone4__T__pe_13'] = np.where(np.bitwise_and(base_dados['Telefone4__T'] >= -201.95895659945893, base_dados['Telefone4__T'] <= 0.9135214625149808), 0.0,
np.where(np.bitwise_and(base_dados['Telefone4__T'] > 0.9135214625149808, base_dados['Telefone4__T'] <= 1.579296184246304), 1.0,
np.where(np.bitwise_and(base_dados['Telefone4__T'] > 1.579296184246304, base_dados['Telefone4__T'] <= 2.7106299741687327), 2.0,
np.where(np.bitwise_and(base_dados['Telefone4__T'] > 2.7106299741687327, base_dados['Telefone4__T'] <= 3.6396339905002772), 3.0,
np.where(np.bitwise_and(base_dados['Telefone4__T'] > 3.6396339905002772, base_dados['Telefone4__T'] <= 4.5330056039968), 4.0,
np.where(np.bitwise_and(base_dados['Telefone4__T'] > 4.5330056039968, base_dados['Telefone4__T'] <= 4.829838102597066), 5.0,
np.where(np.bitwise_and(base_dados['Telefone4__T'] > 4.829838102597066, base_dados['Telefone4__T'] <= 6.232011086488218), 6.0,
np.where(base_dados['Telefone4__T'] > 6.232011086488218, 8.0,
 -2))))))))
base_dados['Telefone4__T__pe_13_g_1_1'] = np.where(base_dados['Telefone4__T__pe_13'] == -2.0, 1,
np.where(base_dados['Telefone4__T__pe_13'] == 0.0, 0,
np.where(base_dados['Telefone4__T__pe_13'] == 1.0, 1,
np.where(base_dados['Telefone4__T__pe_13'] == 2.0, 2,
np.where(base_dados['Telefone4__T__pe_13'] == 3.0, 2,
np.where(base_dados['Telefone4__T__pe_13'] == 4.0, 2,
np.where(base_dados['Telefone4__T__pe_13'] == 5.0, 2,
np.where(base_dados['Telefone4__T__pe_13'] == 6.0, 2,
np.where(base_dados['Telefone4__T__pe_13'] == 8.0, 2,
 0)))))))))
base_dados['Telefone4__T__pe_13_g_1_2'] = np.where(base_dados['Telefone4__T__pe_13_g_1_1'] == 0, 1,
np.where(base_dados['Telefone4__T__pe_13_g_1_1'] == 1, 0,
np.where(base_dados['Telefone4__T__pe_13_g_1_1'] == 2, 2,
 0)))
base_dados['Telefone4__T__pe_13_g_1'] = np.where(base_dados['Telefone4__T__pe_13_g_1_2'] == 0, 0,
np.where(base_dados['Telefone4__T__pe_13_g_1_2'] == 1, 1,
np.where(base_dados['Telefone4__T__pe_13_g_1_2'] == 2, 2,
 0)))


# COMMAND ----------

# MAGIC %md
# MAGIC ### Parte 2 de 2

# COMMAND ----------

base_dados['NUCPFCNPJ__pk_7_g_1_c1_11_1'] = np.where(np.bitwise_and(base_dados['NUCPFCNPJ__pk_7_g_1'] == 0, base_dados['NUCPFCNPJ__S__p_6_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['NUCPFCNPJ__pk_7_g_1'] == 0, base_dados['NUCPFCNPJ__S__p_6_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['NUCPFCNPJ__pk_7_g_1'] == 1, base_dados['NUCPFCNPJ__S__p_6_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['NUCPFCNPJ__pk_7_g_1'] == 1, base_dados['NUCPFCNPJ__S__p_6_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['NUCPFCNPJ__pk_7_g_1'] == 2, base_dados['NUCPFCNPJ__S__p_6_g_1'] == 0), 2,
np.where(np.bitwise_and(base_dados['NUCPFCNPJ__pk_7_g_1'] == 2, base_dados['NUCPFCNPJ__S__p_6_g_1'] == 1), 3,
np.where(np.bitwise_and(base_dados['NUCPFCNPJ__pk_7_g_1'] == 3, base_dados['NUCPFCNPJ__S__p_6_g_1'] == 0), 3,
np.where(np.bitwise_and(base_dados['NUCPFCNPJ__pk_7_g_1'] == 3, base_dados['NUCPFCNPJ__S__p_6_g_1'] == 1), 4,
 0))))))))

base_dados['NUCPFCNPJ__pk_7_g_1_c1_11_2'] = np.where(base_dados['NUCPFCNPJ__pk_7_g_1_c1_11_1'] == 0, 0,
np.where(base_dados['NUCPFCNPJ__pk_7_g_1_c1_11_1'] == 1, 1,
np.where(base_dados['NUCPFCNPJ__pk_7_g_1_c1_11_1'] == 2, 2,
np.where(base_dados['NUCPFCNPJ__pk_7_g_1_c1_11_1'] == 3, 3,
np.where(base_dados['NUCPFCNPJ__pk_7_g_1_c1_11_1'] == 4, 4,
0)))))

base_dados['NUCPFCNPJ__pk_7_g_1_c1_11'] = np.where(base_dados['NUCPFCNPJ__pk_7_g_1_c1_11_2'] == 0, 0,
np.where(base_dados['NUCPFCNPJ__pk_7_g_1_c1_11_2'] == 1, 1,
np.where(base_dados['NUCPFCNPJ__pk_7_g_1_c1_11_2'] == 2, 2,
np.where(base_dados['NUCPFCNPJ__pk_7_g_1_c1_11_2'] == 3, 3,
np.where(base_dados['NUCPFCNPJ__pk_7_g_1_c1_11_2'] == 4, 4,
 0)))))






base_dados['NUCONTRATO__C__p_8_g_1_c1_40_1'] = np.where(np.bitwise_and(base_dados['NUCONTRATO__pe_15_g_1'] == 0, base_dados['NUCONTRATO__C__p_8_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['NUCONTRATO__pe_15_g_1'] == 0, base_dados['NUCONTRATO__C__p_8_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['NUCONTRATO__pe_15_g_1'] == 1, base_dados['NUCONTRATO__C__p_8_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['NUCONTRATO__pe_15_g_1'] == 1, base_dados['NUCONTRATO__C__p_8_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['NUCONTRATO__pe_15_g_1'] == 2, base_dados['NUCONTRATO__C__p_8_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['NUCONTRATO__pe_15_g_1'] == 2, base_dados['NUCONTRATO__C__p_8_g_1'] == 1), 3,
 0))))))

base_dados['NUCONTRATO__C__p_8_g_1_c1_40_2'] = np.where(base_dados['NUCONTRATO__C__p_8_g_1_c1_40_1'] == 0, 0,
np.where(base_dados['NUCONTRATO__C__p_8_g_1_c1_40_1'] == 1, 3,
np.where(base_dados['NUCONTRATO__C__p_8_g_1_c1_40_1'] == 2, 1,
np.where(base_dados['NUCONTRATO__C__p_8_g_1_c1_40_1'] == 3, 2,
0))))

base_dados['NUCONTRATO__C__p_8_g_1_c1_40'] = np.where(base_dados['NUCONTRATO__C__p_8_g_1_c1_40_2'] == 0, 0,
np.where(base_dados['NUCONTRATO__C__p_8_g_1_c1_40_2'] == 1, 1,
np.where(base_dados['NUCONTRATO__C__p_8_g_1_c1_40_2'] == 2, 2,
np.where(base_dados['NUCONTRATO__C__p_8_g_1_c1_40_2'] == 3, 3,
 0))))







base_dados['VLR_RISCO__R__pu_10_g_1_c1_29_1'] = np.where(np.bitwise_and(base_dados['VLR_RISCO__R__pu_10_g_1'] == 0, base_dados['VLR_RISCO__S__p_5_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['VLR_RISCO__R__pu_10_g_1'] == 0, base_dados['VLR_RISCO__S__p_5_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['VLR_RISCO__R__pu_10_g_1'] == 1, base_dados['VLR_RISCO__S__p_5_g_1'] == 0), 2,
np.where(np.bitwise_and(base_dados['VLR_RISCO__R__pu_10_g_1'] == 1, base_dados['VLR_RISCO__S__p_5_g_1'] == 1), 3,
np.where(np.bitwise_and(base_dados['VLR_RISCO__R__pu_10_g_1'] == 2, base_dados['VLR_RISCO__S__p_5_g_1'] == 0), 3,
np.where(np.bitwise_and(base_dados['VLR_RISCO__R__pu_10_g_1'] == 2, base_dados['VLR_RISCO__S__p_5_g_1'] == 1), 4,
 0))))))

base_dados['VLR_RISCO__R__pu_10_g_1_c1_29_2'] = np.where(base_dados['VLR_RISCO__R__pu_10_g_1_c1_29_1'] == 0, 0,
np.where(base_dados['VLR_RISCO__R__pu_10_g_1_c1_29_1'] == 1, 2,
np.where(base_dados['VLR_RISCO__R__pu_10_g_1_c1_29_1'] == 2, 1,
np.where(base_dados['VLR_RISCO__R__pu_10_g_1_c1_29_1'] == 3, 3,
np.where(base_dados['VLR_RISCO__R__pu_10_g_1_c1_29_1'] == 4, 4,
0)))))

base_dados['VLR_RISCO__R__pu_10_g_1_c1_29'] = np.where(base_dados['VLR_RISCO__R__pu_10_g_1_c1_29_2'] == 0, 0,
np.where(base_dados['VLR_RISCO__R__pu_10_g_1_c1_29_2'] == 1, 1,
np.where(base_dados['VLR_RISCO__R__pu_10_g_1_c1_29_2'] == 2, 2,
np.where(base_dados['VLR_RISCO__R__pu_10_g_1_c1_29_2'] == 3, 3,
np.where(base_dados['VLR_RISCO__R__pu_10_g_1_c1_29_2'] == 4, 4,
 0)))))






base_dados['ATRASO__pe_4_g_1_c1_17_1'] = np.where(np.bitwise_and(base_dados['ATRASO__pk_4_g_1'] == 0, base_dados['ATRASO__pe_4_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['ATRASO__pk_4_g_1'] == 0, base_dados['ATRASO__pe_4_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['ATRASO__pk_4_g_1'] == 1, base_dados['ATRASO__pe_4_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['ATRASO__pk_4_g_1'] == 1, base_dados['ATRASO__pe_4_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['ATRASO__pk_4_g_1'] == 2, base_dados['ATRASO__pe_4_g_1'] == 0), 3,
np.where(np.bitwise_and(base_dados['ATRASO__pk_4_g_1'] == 2, base_dados['ATRASO__pe_4_g_1'] == 1), 3,
np.where(np.bitwise_and(base_dados['ATRASO__pk_4_g_1'] == 3, base_dados['ATRASO__pe_4_g_1'] == 0), 4,
np.where(np.bitwise_and(base_dados['ATRASO__pk_4_g_1'] == 3, base_dados['ATRASO__pe_4_g_1'] == 1), 4,
 0))))))))
base_dados['ATRASO__pe_4_g_1_c1_17_2'] = np.where(base_dados['ATRASO__pe_4_g_1_c1_17_1'] == 0, 1,
np.where(base_dados['ATRASO__pe_4_g_1_c1_17_1'] == 1, 2,
np.where(base_dados['ATRASO__pe_4_g_1_c1_17_1'] == 2, 0,
np.where(base_dados['ATRASO__pe_4_g_1_c1_17_1'] == 3, 3,
np.where(base_dados['ATRASO__pe_4_g_1_c1_17_1'] == 4, 4,
0)))))
base_dados['ATRASO__pe_4_g_1_c1_17'] = np.where(base_dados['ATRASO__pe_4_g_1_c1_17_2'] == 0, 0,
np.where(base_dados['ATRASO__pe_4_g_1_c1_17_2'] == 1, 1,
np.where(base_dados['ATRASO__pe_4_g_1_c1_17_2'] == 2, 2,
np.where(base_dados['ATRASO__pe_4_g_1_c1_17_2'] == 3, 3,
np.where(base_dados['ATRASO__pe_4_g_1_c1_17_2'] == 4, 4,
 0)))))






base_dados['Telefone1__L__pk_10_g_1_c1_35_1'] = np.where(np.bitwise_and(base_dados['Telefone1__L__pk_10_g_1'] == 0, base_dados['Telefone1__T__p_7_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['Telefone1__L__pk_10_g_1'] == 0, base_dados['Telefone1__T__p_7_g_1'] == 1), 0,
np.where(np.bitwise_and(base_dados['Telefone1__L__pk_10_g_1'] == 0, base_dados['Telefone1__T__p_7_g_1'] == 2), 0,
np.where(np.bitwise_and(base_dados['Telefone1__L__pk_10_g_1'] == 1, base_dados['Telefone1__T__p_7_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['Telefone1__L__pk_10_g_1'] == 1, base_dados['Telefone1__T__p_7_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['Telefone1__L__pk_10_g_1'] == 1, base_dados['Telefone1__T__p_7_g_1'] == 2), 2,
np.where(np.bitwise_and(base_dados['Telefone1__L__pk_10_g_1'] == 2, base_dados['Telefone1__T__p_7_g_1'] == 0), 2,
np.where(np.bitwise_and(base_dados['Telefone1__L__pk_10_g_1'] == 2, base_dados['Telefone1__T__p_7_g_1'] == 1), 3,
np.where(np.bitwise_and(base_dados['Telefone1__L__pk_10_g_1'] == 2, base_dados['Telefone1__T__p_7_g_1'] == 2), 4,
 0)))))))))

base_dados['Telefone1__L__pk_10_g_1_c1_35_2'] = np.where(base_dados['Telefone1__L__pk_10_g_1_c1_35_1'] == 0, 0,
np.where(base_dados['Telefone1__L__pk_10_g_1_c1_35_1'] == 1, 1,
np.where(base_dados['Telefone1__L__pk_10_g_1_c1_35_1'] == 2, 3,
np.where(base_dados['Telefone1__L__pk_10_g_1_c1_35_1'] == 3, 2,
np.where(base_dados['Telefone1__L__pk_10_g_1_c1_35_1'] == 4, 4,
0)))))

base_dados['Telefone1__L__pk_10_g_1_c1_35'] = np.where(base_dados['Telefone1__L__pk_10_g_1_c1_35_2'] == 0, 0,
np.where(base_dados['Telefone1__L__pk_10_g_1_c1_35_2'] == 1, 1,
np.where(base_dados['Telefone1__L__pk_10_g_1_c1_35_2'] == 2, 2,
np.where(base_dados['Telefone1__L__pk_10_g_1_c1_35_2'] == 3, 3,
np.where(base_dados['Telefone1__L__pk_10_g_1_c1_35_2'] == 4, 4,
 0)))))






base_dados['Telefone2__S__pe_10_g_1_c1_29_1'] = np.where(np.bitwise_and(base_dados['Telefone2__pk_25_g_1'] == 0, base_dados['Telefone2__S__pe_10_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['Telefone2__pk_25_g_1'] == 0, base_dados['Telefone2__S__pe_10_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['Telefone2__pk_25_g_1'] == 1, base_dados['Telefone2__S__pe_10_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['Telefone2__pk_25_g_1'] == 1, base_dados['Telefone2__S__pe_10_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['Telefone2__pk_25_g_1'] == 2, base_dados['Telefone2__S__pe_10_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['Telefone2__pk_25_g_1'] == 2, base_dados['Telefone2__S__pe_10_g_1'] == 1), 3,
 0))))))

base_dados['Telefone2__S__pe_10_g_1_c1_29_2'] = np.where(base_dados['Telefone2__S__pe_10_g_1_c1_29_1'] == 0, 0,
np.where(base_dados['Telefone2__S__pe_10_g_1_c1_29_1'] == 1, 1,
np.where(base_dados['Telefone2__S__pe_10_g_1_c1_29_1'] == 2, 2,
np.where(base_dados['Telefone2__S__pe_10_g_1_c1_29_1'] == 3, 3,
0))))

base_dados['Telefone2__S__pe_10_g_1_c1_29'] = np.where(base_dados['Telefone2__S__pe_10_g_1_c1_29_2'] == 0, 0,
np.where(base_dados['Telefone2__S__pe_10_g_1_c1_29_2'] == 1, 1,
np.where(base_dados['Telefone2__S__pe_10_g_1_c1_29_2'] == 2, 2,
np.where(base_dados['Telefone2__S__pe_10_g_1_c1_29_2'] == 3, 3,
 0))))







base_dados['Telefone3__T__pk_40_g_1_c1_7_1'] = np.where(np.bitwise_and(base_dados['Telefone3__p_17_g_1'] == 0, base_dados['Telefone3__T__pk_40_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['Telefone3__p_17_g_1'] == 0, base_dados['Telefone3__T__pk_40_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['Telefone3__p_17_g_1'] == 0, base_dados['Telefone3__T__pk_40_g_1'] == 2), 1,
np.where(np.bitwise_and(base_dados['Telefone3__p_17_g_1'] == 0, base_dados['Telefone3__T__pk_40_g_1'] == 3), 3,
np.where(np.bitwise_and(base_dados['Telefone3__p_17_g_1'] == 1, base_dados['Telefone3__T__pk_40_g_1'] == 0), 2,
np.where(np.bitwise_and(base_dados['Telefone3__p_17_g_1'] == 1, base_dados['Telefone3__T__pk_40_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['Telefone3__p_17_g_1'] == 1, base_dados['Telefone3__T__pk_40_g_1'] == 2), 2,
np.where(np.bitwise_and(base_dados['Telefone3__p_17_g_1'] == 1, base_dados['Telefone3__T__pk_40_g_1'] == 3), 3,
 0))))))))
base_dados['Telefone3__T__pk_40_g_1_c1_7_2'] = np.where(base_dados['Telefone3__T__pk_40_g_1_c1_7_1'] == 0, 0,
np.where(base_dados['Telefone3__T__pk_40_g_1_c1_7_1'] == 1, 1,
np.where(base_dados['Telefone3__T__pk_40_g_1_c1_7_1'] == 2, 2,
np.where(base_dados['Telefone3__T__pk_40_g_1_c1_7_1'] == 3, 3,
0))))
base_dados['Telefone3__T__pk_40_g_1_c1_7'] = np.where(base_dados['Telefone3__T__pk_40_g_1_c1_7_2'] == 0, 0,
np.where(base_dados['Telefone3__T__pk_40_g_1_c1_7_2'] == 1, 1,
np.where(base_dados['Telefone3__T__pk_40_g_1_c1_7_2'] == 2, 2,
np.where(base_dados['Telefone3__T__pk_40_g_1_c1_7_2'] == 3, 3,
 0))))





base_dados['Telefone4__T__pe_13_g_1_c1_3_1'] = np.where(np.bitwise_and(base_dados['Telefone4__T__pk_25_g_1'] == 0, base_dados['Telefone4__T__pe_13_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['Telefone4__T__pk_25_g_1'] == 0, base_dados['Telefone4__T__pe_13_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['Telefone4__T__pk_25_g_1'] == 0, base_dados['Telefone4__T__pe_13_g_1'] == 2), 1,
np.where(np.bitwise_and(base_dados['Telefone4__T__pk_25_g_1'] == 1, base_dados['Telefone4__T__pe_13_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['Telefone4__T__pk_25_g_1'] == 1, base_dados['Telefone4__T__pe_13_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['Telefone4__T__pk_25_g_1'] == 1, base_dados['Telefone4__T__pe_13_g_1'] == 2), 1,
 0))))))
base_dados['Telefone4__T__pe_13_g_1_c1_3_2'] = np.where(base_dados['Telefone4__T__pe_13_g_1_c1_3_1'] == 0, 0,
np.where(base_dados['Telefone4__T__pe_13_g_1_c1_3_1'] == 1, 1,
0))
base_dados['Telefone4__T__pe_13_g_1_c1_3'] = np.where(base_dados['Telefone4__T__pe_13_g_1_c1_3_2'] == 0, 0,
np.where(base_dados['Telefone4__T__pe_13_g_1_c1_3_2'] == 1, 1,
 0))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Mantendo apenas as variáveis do modelo

# COMMAND ----------

import pickle
modelo=pickle.load(open(pickle_path + 'model_fit_bvm_r_forest.sav', 'rb'))

base_teste_c0 = base_dados[[chave,'ATRASO__pe_4_g_1_c1_17', 'NUCONTRATO__C__p_8_g_1_c1_40', 'NUCPFCNPJ__pk_7_g_1_c1_11', 'Telefone1__L__pk_10_g_1_c1_35', 'Telefone2__S__pe_10_g_1_c1_29', 'Telefone3__T__pk_40_g_1_c1_7', 'Telefone4__T__pe_13_g_1_c1_3', 'VLR_RISCO__R__pu_10_g_1_c1_29', 'P_Email1_gh38', 'CDPRODUTO_gh38', 'NUPARCELA_gh71']]

var_fin_c0=['ATRASO__pe_4_g_1_c1_17', 'NUCONTRATO__C__p_8_g_1_c1_40', 'NUCPFCNPJ__pk_7_g_1_c1_11', 'Telefone1__L__pk_10_g_1_c1_35', 'Telefone2__S__pe_10_g_1_c1_29', 'Telefone3__T__pk_40_g_1_c1_7', 'Telefone4__T__pe_13_g_1_c1_3', 'VLR_RISCO__R__pu_10_g_1_c1_29', 'P_Email1_gh38', 'CDPRODUTO_gh38', 'NUPARCELA_gh71']

print(var_fin_c0)


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

x_teste2


# COMMAND ----------

# MAGIC %md
# MAGIC # Modelo de Grupo Homogêneo

# COMMAND ----------

x_teste2['P_1_R'] = np.sqrt(x_teste2['P_1'])
x_teste2['P_1_R'] = np.where(x_teste2['P_1'] == 0, -1, x_teste2['P_1_R'])
x_teste2['P_1_R'] = x_teste2['P_1_R'].fillna(-2)
x_teste2['P_1_R'] = x_teste2['P_1_R'].fillna(-2)

x_teste2['P_1_R_p_10_g_1'] = np.where(x_teste2['P_1_R'] <= 0.295522137, 0,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.295522137, x_teste2['P_1_R'] <= 0.486483984), 1,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.486483984, x_teste2['P_1_R'] <= 0.685079071), 2,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.685079071, x_teste2['P_1_R'] <= 0.799478997), 3,4))))

x_teste2['P_1_p_40_g_1'] = np.where(x_teste2['P_1'] <= 0.197833333, 0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.197833333, x_teste2['P_1'] <= 0.249166667), 1,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.249166667, x_teste2['P_1'] <= 0.3275), 2,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.3275, x_teste2['P_1'] <= 0.639166667), 3,4))))

x_teste2['GH'] = np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 0, x_teste2['P_1_R_p_10_g_1'] == 0), 0,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 0, x_teste2['P_1_R_p_10_g_1'] == 1), 1,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 0, x_teste2['P_1_R_p_10_g_1'] == 2), 1,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 0, x_teste2['P_1_R_p_10_g_1'] == 3), 1,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 0, x_teste2['P_1_R_p_10_g_1'] == 4), 1,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 1, x_teste2['P_1_R_p_10_g_1'] == 0), 1,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 1, x_teste2['P_1_R_p_10_g_1'] == 1), 1,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 1, x_teste2['P_1_R_p_10_g_1'] == 2), 1,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 1, x_teste2['P_1_R_p_10_g_1'] == 3), 1,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 1, x_teste2['P_1_R_p_10_g_1'] == 4), 2,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 2, x_teste2['P_1_R_p_10_g_1'] == 0), 2,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 2, x_teste2['P_1_R_p_10_g_1'] == 1), 2,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 2, x_teste2['P_1_R_p_10_g_1'] == 2), 2,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 2, x_teste2['P_1_R_p_10_g_1'] == 3), 2,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 2, x_teste2['P_1_R_p_10_g_1'] == 4), 3,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 3, x_teste2['P_1_R_p_10_g_1'] == 0), 3,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 3, x_teste2['P_1_R_p_10_g_1'] == 1), 3,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 3, x_teste2['P_1_R_p_10_g_1'] == 2), 3,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 3, x_teste2['P_1_R_p_10_g_1'] == 3), 4,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 3, x_teste2['P_1_R_p_10_g_1'] == 4), 4,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 4, x_teste2['P_1_R_p_10_g_1'] == 0), 4,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 4, x_teste2['P_1_R_p_10_g_1'] == 1), 4,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 4, x_teste2['P_1_R_p_10_g_1'] == 2), 4,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 4, x_teste2['P_1_R_p_10_g_1'] == 3), 5,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 4, x_teste2['P_1_R_p_10_g_1'] == 4), 5,0)))))))))))))))))))))))))

del x_teste2['P_1_R']
del x_teste2['P_1_R_p_10_g_1']
del x_teste2['P_1_p_40_g_1']
x_teste2

# COMMAND ----------

x_teste2.groupby(['GH'])['P_1'].count()

# COMMAND ----------

try:
  dbutils.fs.rm(outputpath, True)
except:
  pass
dbutils.fs.mkdirs(outputpath)

x_teste2.to_csv(open(os.path.join(outputpath_dbfs, 'pre_output:'+N_Base),'wb'))
os.path.join(outputpath_dbfs, 'pre_output:'+N_Base)