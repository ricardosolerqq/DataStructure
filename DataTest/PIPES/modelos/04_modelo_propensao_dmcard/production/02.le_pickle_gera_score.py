# Databricks notebook source
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

# Imports
import os
import pickle
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

%matplotlib inline

# COMMAND ----------

try:
  dbutils.widgets.remove('ARQUIVO_ESCOLHIDO')
except:
  pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inserindo hiperparâmetros do Algoritmo

# COMMAND ----------

blob_account_source_prd = "qqprd"
blob_account_source_ml = "qqdatastoragemain"
blob_container_source_prd = "qq-integrator"
blob_container_source_ml = "ml-prd"

prefix = "etl/dmcard/processed"

mount_blob_storage_key(dbutils,blob_account_source_prd,blob_account_source_prd,'/mnt/qq-integrator')
mount_blob_storage_key(dbutils,blob_account_source_ml,blob_account_source_ml,'/mnt/ml-prd')

caminho_trusted = '/mnt/ml-prd/ml-data/propensaodeal/dmcard/trusted'
caminho_trusted_dbfs = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/dmcard/trusted'

pickle_path = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/dmcard/pickle_model/'

outputpath = 'mnt/ml-prd/ml-data/propensaodeal/dmcard/output/'
outputpath_dbfs = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/dmcard/output/'

# COMMAND ----------

dbutils.widgets.dropdown('ARQUIVO_ESCOLHIDO', max(os.listdir(caminho_trusted_dbfs)), os.listdir(caminho_trusted_dbfs))
arquivo_escolhido = dbutils.widgets.get('ARQUIVO_ESCOLHIDO')
arquivo_escolhido = arquivo_escolhido

caminho_arquivo_escolhido = os.path.join(caminho_trusted_dbfs, arquivo_escolhido)
caminho_arquivo_escolhido

# COMMAND ----------

## Parâmetros do Algoritmo

#Variável chave-primaria
chave = 'DOC'

#Variável resposta ou target
#target = 'VARIAVEL_RESPOSTA'

#Lista com a variável Tempo
var_tmp = 'SEXO'

#Nome da Base de Dados
N_Base = arquivo_escolhido

#Caminho da base de dados
caminho_base = caminho_arquivo_escolhido

#Separador
separador_ = ";"

#Decimal
decimal_ = ","


# COMMAND ----------

# MAGIC %md
# MAGIC ## Importação da Base de Dados

# COMMAND ----------

#carregar o arquivo em formato tabela
base_dados = pd.read_parquet(os.path.join(caminho_base,N_Base+'.PARQUET'))
#base_dados[target] = base_dados[target].map({True:1,False:0},na_action=None)
print("shape da Base de Dados:",base_dados.shape)

#list_var = [chave,var_tmp,target, 'DIAS_ATRASO','CODIGO','DDD1','LIMITE','COD_EMPRESA','DDD5','COD_FILA','DDD4','UF']
list_var = [chave,var_tmp,'DIAS_ATRASO','CODIGO','DDD1','LIMITE','COD_EMPRESA','DDD5','COD_FILA','DDD4','UF']

base_dados = base_dados[list_var]
base_dados['DIAS_ATRASO'] = base_dados['DIAS_ATRASO'].replace(np.nan, 0)
base_dados['DIAS_ATRASO'] = base_dados['DIAS_ATRASO'].astype(int)

base_dados['CODIGO'] = base_dados['CODIGO'].replace(np.nan, 0)
base_dados['CODIGO'] = base_dados['CODIGO'].astype(int)

base_dados['DDD1'] = base_dados['DDD1'].replace(np.nan, 0)
base_dados['DDD1'] = base_dados['DDD1'].astype(int)

base_dados['LIMITE'] = base_dados['LIMITE'].replace(np.nan, 0)
base_dados['LIMITE'] = base_dados['LIMITE'].str.replace(',','.').astype(float)

base_dados['COD_EMPRESA'] = base_dados['COD_EMPRESA'].replace(np.nan, 0)
base_dados['COD_EMPRESA'] = base_dados['COD_EMPRESA'].astype(int)

base_dados['DDD5'] = base_dados['DDD5'].replace(np.nan, 0)
base_dados['DDD5'] = base_dados['DDD5'].astype(int)

base_dados['COD_FILA'] = base_dados['COD_FILA'].replace(np.nan, 0)
base_dados['COD_FILA'] = base_dados['COD_FILA'].astype(int)

base_dados['DDD4'] = base_dados['DDD4'].replace(np.nan, 0)
base_dados['DDD4'] = base_dados['DDD4'].astype(int)

base_dados['UF'] = base_dados['UF'].replace(np.nan, '-3')

base_dados.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gerando variáveis Categóricas

# COMMAND ----------


base_dados['UF_gh30'] = np.where(base_dados['UF'] == '-3', 0,
    np.where(base_dados['UF'] == 'AL', 1,
    np.where(base_dados['UF'] == 'AM', 2,
    np.where(base_dados['UF'] == 'BA', 3,
    np.where(base_dados['UF'] == 'CE', 4,
    np.where(base_dados['UF'] == 'ES', 5,
    np.where(base_dados['UF'] == 'GO', 6,
    np.where(base_dados['UF'] == 'MA', 7,
    np.where(base_dados['UF'] == 'MG', 8,
    np.where(base_dados['UF'] == 'MS', 9,
    np.where(base_dados['UF'] == 'MT', 10,
    np.where(base_dados['UF'] == 'PA', 11,
    np.where(base_dados['UF'] == 'PB', 12,
    np.where(base_dados['UF'] == 'PE', 13,
    np.where(base_dados['UF'] == 'PI', 14,
    np.where(base_dados['UF'] == 'PR', 15,
    np.where(base_dados['UF'] == 'RJ', 16,
    np.where(base_dados['UF'] == 'RN', 17,
    np.where(base_dados['UF'] == 'RO', 18,
    np.where(base_dados['UF'] == 'RR', 19,
    np.where(base_dados['UF'] == 'RS', 20,
    np.where(base_dados['UF'] == 'SC', 21,
    np.where(base_dados['UF'] == 'SE', 22,
    np.where(base_dados['UF'] == 'SP', 23,
    np.where(base_dados['UF'] == 'TO', 24,
    np.where(base_dados['UF'] == 'XX', 25,
    0))))))))))))))))))))))))))

base_dados['UF_gh31'] = np.where(base_dados['UF_gh30'] == 0, 0,
    np.where(base_dados['UF_gh30'] == 1, 1,
    np.where(base_dados['UF_gh30'] == 2, 2,
    np.where(base_dados['UF_gh30'] == 3, 3,
    np.where(base_dados['UF_gh30'] == 4, 3,
    np.where(base_dados['UF_gh30'] == 5, 5,
    np.where(base_dados['UF_gh30'] == 6, 5,
    np.where(base_dados['UF_gh30'] == 7, 7,
    np.where(base_dados['UF_gh30'] == 8, 8,
    np.where(base_dados['UF_gh30'] == 9, 9,
    np.where(base_dados['UF_gh30'] == 10, 10,
    np.where(base_dados['UF_gh30'] == 11, 10,
    np.where(base_dados['UF_gh30'] == 12, 12,
    np.where(base_dados['UF_gh30'] == 13, 13,
    np.where(base_dados['UF_gh30'] == 14, 14,
    np.where(base_dados['UF_gh30'] == 15, 15,
    np.where(base_dados['UF_gh30'] == 16, 16,
    np.where(base_dados['UF_gh30'] == 17, 17,
    np.where(base_dados['UF_gh30'] == 18, 18,
    np.where(base_dados['UF_gh30'] == 19, 18,
    np.where(base_dados['UF_gh30'] == 20, 20,
    np.where(base_dados['UF_gh30'] == 21, 21,
    np.where(base_dados['UF_gh30'] == 22, 22,
    np.where(base_dados['UF_gh30'] == 23, 23,
    np.where(base_dados['UF_gh30'] == 24, 24,
    np.where(base_dados['UF_gh30'] == 25, 25,
    0))))))))))))))))))))))))))

base_dados['UF_gh32'] = np.where(base_dados['UF_gh31'] == 0, 0,
    np.where(base_dados['UF_gh31'] == 1, 1,
    np.where(base_dados['UF_gh31'] == 2, 2,
    np.where(base_dados['UF_gh31'] == 3, 3,
    np.where(base_dados['UF_gh31'] == 5, 4,
    np.where(base_dados['UF_gh31'] == 7, 5,
    np.where(base_dados['UF_gh31'] == 8, 6,
    np.where(base_dados['UF_gh31'] == 9, 7,
    np.where(base_dados['UF_gh31'] == 10, 8,
    np.where(base_dados['UF_gh31'] == 12, 9,
    np.where(base_dados['UF_gh31'] == 13, 10,
    np.where(base_dados['UF_gh31'] == 14, 11,
    np.where(base_dados['UF_gh31'] == 15, 12,
    np.where(base_dados['UF_gh31'] == 16, 13,
    np.where(base_dados['UF_gh31'] == 17, 14,
    np.where(base_dados['UF_gh31'] == 18, 15,
    np.where(base_dados['UF_gh31'] == 20, 16,
    np.where(base_dados['UF_gh31'] == 21, 17,
    np.where(base_dados['UF_gh31'] == 22, 18,
    np.where(base_dados['UF_gh31'] == 23, 19,
    np.where(base_dados['UF_gh31'] == 24, 20,
    np.where(base_dados['UF_gh31'] == 25, 21,
    0))))))))))))))))))))))

base_dados['UF_gh33'] = np.where(base_dados['UF_gh32'] == 0, 0,
    np.where(base_dados['UF_gh32'] == 1, 1,
    np.where(base_dados['UF_gh32'] == 2, 2,
    np.where(base_dados['UF_gh32'] == 3, 3,
    np.where(base_dados['UF_gh32'] == 4, 4,
    np.where(base_dados['UF_gh32'] == 5, 5,
    np.where(base_dados['UF_gh32'] == 6, 6,
    np.where(base_dados['UF_gh32'] == 7, 7,
    np.where(base_dados['UF_gh32'] == 8, 8,
    np.where(base_dados['UF_gh32'] == 9, 9,
    np.where(base_dados['UF_gh32'] == 10, 10,
    np.where(base_dados['UF_gh32'] == 11, 11,
    np.where(base_dados['UF_gh32'] == 12, 12,
    np.where(base_dados['UF_gh32'] == 13, 13,
    np.where(base_dados['UF_gh32'] == 14, 14,
    np.where(base_dados['UF_gh32'] == 15, 15,
    np.where(base_dados['UF_gh32'] == 16, 16,
    np.where(base_dados['UF_gh32'] == 17, 17,
    np.where(base_dados['UF_gh32'] == 18, 18,
    np.where(base_dados['UF_gh32'] == 19, 19,
    np.where(base_dados['UF_gh32'] == 20, 20,
    np.where(base_dados['UF_gh32'] == 21, 21,
    0))))))))))))))))))))))

base_dados['UF_gh34'] = np.where(base_dados['UF_gh33'] == 0, 21,
    np.where(base_dados['UF_gh33'] == 1, 2,
    np.where(base_dados['UF_gh33'] == 2, 2,
    np.where(base_dados['UF_gh33'] == 3, 6,
    np.where(base_dados['UF_gh33'] == 4, 2,
    np.where(base_dados['UF_gh33'] == 5, 2,
    np.where(base_dados['UF_gh33'] == 6, 6,
    np.where(base_dados['UF_gh33'] == 7, 2,
    np.where(base_dados['UF_gh33'] == 8, 6,
    np.where(base_dados['UF_gh33'] == 9, 2,
    np.where(base_dados['UF_gh33'] == 10, 19,
    np.where(base_dados['UF_gh33'] == 11, 2,
    np.where(base_dados['UF_gh33'] == 12, 6,
    np.where(base_dados['UF_gh33'] == 13, 6,
    np.where(base_dados['UF_gh33'] == 14, 6,
    np.where(base_dados['UF_gh33'] == 15, 2,
    np.where(base_dados['UF_gh33'] == 16, 6,
    np.where(base_dados['UF_gh33'] == 17, 6,
    np.where(base_dados['UF_gh33'] == 18, 2,
    np.where(base_dados['UF_gh33'] == 19, 19,
    np.where(base_dados['UF_gh33'] == 20, 2,
    np.where(base_dados['UF_gh33'] == 21, 21,
    2))))))))))))))))))))))

base_dados['UF_gh35'] = np.where(base_dados['UF_gh34'] == 2, 0,
    np.where(base_dados['UF_gh34'] == 6, 1,
    np.where(base_dados['UF_gh34'] == 19, 2,
    np.where(base_dados['UF_gh34'] == 21, 3,
    0))))

base_dados['UF_gh36'] = np.where(base_dados['UF_gh35'] == 0, 0,
    np.where(base_dados['UF_gh35'] == 1, 2,
    np.where(base_dados['UF_gh35'] == 2, 1,
    np.where(base_dados['UF_gh35'] == 3, 3,
    0))))

base_dados['UF_gh37'] = np.where(base_dados['UF_gh36'] == 0, 1,
    np.where(base_dados['UF_gh36'] == 1, 1,
    np.where(base_dados['UF_gh36'] == 2, 2,
    np.where(base_dados['UF_gh36'] == 3, 3,
    1))))

base_dados['UF_gh38'] = np.where(base_dados['UF_gh37'] == 1, 0,
    np.where(base_dados['UF_gh37'] == 2, 1,
    np.where(base_dados['UF_gh37'] == 3, 2,
    0)))
         
         
              
base_dados['DDD5_gh30'] = np.where(base_dados['DDD5'] == 0, 0,
    np.where(base_dados['DDD5'] == 1, 1,
    np.where(base_dados['DDD5'] == 3, 2,
    np.where(base_dados['DDD5'] == 4, 3,
    np.where(base_dados['DDD5'] == 11, 4,
    np.where(base_dados['DDD5'] == 12, 5,
    np.where(base_dados['DDD5'] == 13, 6,
    np.where(base_dados['DDD5'] == 14, 7,
    np.where(base_dados['DDD5'] == 15, 8,
    np.where(base_dados['DDD5'] == 16, 9,
    np.where(base_dados['DDD5'] == 17, 10,
    np.where(base_dados['DDD5'] == 18, 11,
    np.where(base_dados['DDD5'] == 19, 12,
    np.where(base_dados['DDD5'] == 21, 13,
    np.where(base_dados['DDD5'] == 22, 14,
    np.where(base_dados['DDD5'] == 24, 15,
    np.where(base_dados['DDD5'] == 27, 16,
    np.where(base_dados['DDD5'] == 31, 17,
    np.where(base_dados['DDD5'] == 33, 18,
    np.where(base_dados['DDD5'] == 34, 19,
    np.where(base_dados['DDD5'] == 35, 20,
    np.where(base_dados['DDD5'] == 38, 21,
    np.where(base_dados['DDD5'] == 41, 22,
    np.where(base_dados['DDD5'] == 42, 23,
    np.where(base_dados['DDD5'] == 43, 24,
    np.where(base_dados['DDD5'] == 47, 25,
    np.where(base_dados['DDD5'] == 48, 26,
    np.where(base_dados['DDD5'] == 51, 27,
    np.where(base_dados['DDD5'] == 53, 28,
    np.where(base_dados['DDD5'] == 55, 29,
    np.where(base_dados['DDD5'] == 65, 30,
    np.where(base_dados['DDD5'] == 66, 31,
    np.where(base_dados['DDD5'] == 67, 32,
    np.where(base_dados['DDD5'] == 69, 33,
    np.where(base_dados['DDD5'] == 71, 34,
    np.where(base_dados['DDD5'] == 73, 35,
    np.where(base_dados['DDD5'] == 75, 36,
    np.where(base_dados['DDD5'] == 79, 37,
    np.where(base_dados['DDD5'] == 84, 38,
    np.where(base_dados['DDD5'] == 91, 39,
    0))))))))))))))))))))))))))))))))))))))))
             
base_dados['DDD5_gh31'] = np.where(base_dados['DDD5_gh30'] == 0, 0,
    np.where(base_dados['DDD5_gh30'] == 1, 1,
    np.where(base_dados['DDD5_gh30'] == 2, 1,
    np.where(base_dados['DDD5_gh30'] == 3, 1,
    np.where(base_dados['DDD5_gh30'] == 4, 4,
    np.where(base_dados['DDD5_gh30'] == 5, 5,
    np.where(base_dados['DDD5_gh30'] == 6, 6,
    np.where(base_dados['DDD5_gh30'] == 7, 6,
    np.where(base_dados['DDD5_gh30'] == 8, 8,
    np.where(base_dados['DDD5_gh30'] == 9, 9,
    np.where(base_dados['DDD5_gh30'] == 10, 10,
    np.where(base_dados['DDD5_gh30'] == 11, 11,
    np.where(base_dados['DDD5_gh30'] == 12, 12,
    np.where(base_dados['DDD5_gh30'] == 13, 13,
    np.where(base_dados['DDD5_gh30'] == 14, 14,
    np.where(base_dados['DDD5_gh30'] == 15, 15,
    np.where(base_dados['DDD5_gh30'] == 16, 16,
    np.where(base_dados['DDD5_gh30'] == 17, 17,
    np.where(base_dados['DDD5_gh30'] == 18, 18,
    np.where(base_dados['DDD5_gh30'] == 19, 18,
    np.where(base_dados['DDD5_gh30'] == 20, 20,
    np.where(base_dados['DDD5_gh30'] == 21, 21,
    np.where(base_dados['DDD5_gh30'] == 22, 21,
    np.where(base_dados['DDD5_gh30'] == 23, 23,
    np.where(base_dados['DDD5_gh30'] == 24, 24,
    np.where(base_dados['DDD5_gh30'] == 25, 25,
    np.where(base_dados['DDD5_gh30'] == 26, 26,
    np.where(base_dados['DDD5_gh30'] == 27, 26,
    np.where(base_dados['DDD5_gh30'] == 28, 28,
    np.where(base_dados['DDD5_gh30'] == 29, 29,
    np.where(base_dados['DDD5_gh30'] == 30, 30,
    np.where(base_dados['DDD5_gh30'] == 31, 31,
    np.where(base_dados['DDD5_gh30'] == 32, 31,
    np.where(base_dados['DDD5_gh30'] == 33, 33,
    np.where(base_dados['DDD5_gh30'] == 34, 34,
    np.where(base_dados['DDD5_gh30'] == 35, 34,
    np.where(base_dados['DDD5_gh30'] == 36, 36,
    np.where(base_dados['DDD5_gh30'] == 37, 37,
    np.where(base_dados['DDD5_gh30'] == 38, 37,
    np.where(base_dados['DDD5_gh30'] == 39, 37,
    0))))))))))))))))))))))))))))))))))))))))

base_dados['DDD5_gh32'] = np.where(base_dados['DDD5_gh31'] == 0, 0,
    np.where(base_dados['DDD5_gh31'] == 1, 1,
    np.where(base_dados['DDD5_gh31'] == 4, 2,
    np.where(base_dados['DDD5_gh31'] == 5, 3,
    np.where(base_dados['DDD5_gh31'] == 6, 4,
    np.where(base_dados['DDD5_gh31'] == 8, 5,
    np.where(base_dados['DDD5_gh31'] == 9, 6,
    np.where(base_dados['DDD5_gh31'] == 10, 7,
    np.where(base_dados['DDD5_gh31'] == 11, 8,
    np.where(base_dados['DDD5_gh31'] == 12, 9,
    np.where(base_dados['DDD5_gh31'] == 13, 10,
    np.where(base_dados['DDD5_gh31'] == 14, 11,
    np.where(base_dados['DDD5_gh31'] == 15, 12,
    np.where(base_dados['DDD5_gh31'] == 16, 13,
    np.where(base_dados['DDD5_gh31'] == 17, 14,
    np.where(base_dados['DDD5_gh31'] == 18, 15,
    np.where(base_dados['DDD5_gh31'] == 20, 16,
    np.where(base_dados['DDD5_gh31'] == 21, 17,
    np.where(base_dados['DDD5_gh31'] == 23, 18,
    np.where(base_dados['DDD5_gh31'] == 24, 19,
    np.where(base_dados['DDD5_gh31'] == 25, 20,
    np.where(base_dados['DDD5_gh31'] == 26, 21,
    np.where(base_dados['DDD5_gh31'] == 28, 22,
    np.where(base_dados['DDD5_gh31'] == 29, 23,
    np.where(base_dados['DDD5_gh31'] == 30, 24,
    np.where(base_dados['DDD5_gh31'] == 31, 25,
    np.where(base_dados['DDD5_gh31'] == 33, 26,
    np.where(base_dados['DDD5_gh31'] == 34, 27,
    np.where(base_dados['DDD5_gh31'] == 36, 28,
    np.where(base_dados['DDD5_gh31'] == 37, 29,
    0))))))))))))))))))))))))))))))

base_dados['DDD5_gh33'] = np.where(base_dados['DDD5_gh32'] == 0, 0,
    np.where(base_dados['DDD5_gh32'] == 1, 1,
    np.where(base_dados['DDD5_gh32'] == 2, 2,
    np.where(base_dados['DDD5_gh32'] == 3, 3,
    np.where(base_dados['DDD5_gh32'] == 4, 4,
    np.where(base_dados['DDD5_gh32'] == 5, 5,
    np.where(base_dados['DDD5_gh32'] == 6, 6,
    np.where(base_dados['DDD5_gh32'] == 7, 7,
    np.where(base_dados['DDD5_gh32'] == 8, 8,
    np.where(base_dados['DDD5_gh32'] == 9, 9,
    np.where(base_dados['DDD5_gh32'] == 10, 10,
    np.where(base_dados['DDD5_gh32'] == 11, 11,
    np.where(base_dados['DDD5_gh32'] == 12, 12,
    np.where(base_dados['DDD5_gh32'] == 13, 13,
    np.where(base_dados['DDD5_gh32'] == 14, 14,
    np.where(base_dados['DDD5_gh32'] == 15, 15,
    np.where(base_dados['DDD5_gh32'] == 16, 16,
    np.where(base_dados['DDD5_gh32'] == 17, 17,
    np.where(base_dados['DDD5_gh32'] == 18, 18,
    np.where(base_dados['DDD5_gh32'] == 19, 19,
    np.where(base_dados['DDD5_gh32'] == 20, 20,
    np.where(base_dados['DDD5_gh32'] == 21, 21,
    np.where(base_dados['DDD5_gh32'] == 22, 22,
    np.where(base_dados['DDD5_gh32'] == 23, 23,
    np.where(base_dados['DDD5_gh32'] == 24, 24,
    np.where(base_dados['DDD5_gh32'] == 25, 25,
    np.where(base_dados['DDD5_gh32'] == 26, 26,
    np.where(base_dados['DDD5_gh32'] == 27, 27,
    np.where(base_dados['DDD5_gh32'] == 28, 28,
    np.where(base_dados['DDD5_gh32'] == 29, 29,
    0))))))))))))))))))))))))))))))
             
base_dados['DDD5_gh34'] = np.where(base_dados['DDD5_gh33'] == 0, 0,
    np.where(base_dados['DDD5_gh33'] == 1, 29,
    np.where(base_dados['DDD5_gh33'] == 2, 2,
    np.where(base_dados['DDD5_gh33'] == 3, 2,
    np.where(base_dados['DDD5_gh33'] == 4, 29,
    np.where(base_dados['DDD5_gh33'] == 5, 2,
    np.where(base_dados['DDD5_gh33'] == 6, 29,
    np.where(base_dados['DDD5_gh33'] == 7, 2,
    np.where(base_dados['DDD5_gh33'] == 8, 29,
    np.where(base_dados['DDD5_gh33'] == 9, 29,
    np.where(base_dados['DDD5_gh33'] == 10, 2,
    np.where(base_dados['DDD5_gh33'] == 11, 29,
    np.where(base_dados['DDD5_gh33'] == 12, 2,
    np.where(base_dados['DDD5_gh33'] == 13, 29,
    np.where(base_dados['DDD5_gh33'] == 14, 0,
    np.where(base_dados['DDD5_gh33'] == 15, 29,
    np.where(base_dados['DDD5_gh33'] == 16, 29,
    np.where(base_dados['DDD5_gh33'] == 17, 29,
    np.where(base_dados['DDD5_gh33'] == 18, 29,
    np.where(base_dados['DDD5_gh33'] == 19, 29,
    np.where(base_dados['DDD5_gh33'] == 20, 0,
    np.where(base_dados['DDD5_gh33'] == 21, 0,
    np.where(base_dados['DDD5_gh33'] == 22, 29,
    np.where(base_dados['DDD5_gh33'] == 23, 2,
    np.where(base_dados['DDD5_gh33'] == 24, 2,
    np.where(base_dados['DDD5_gh33'] == 25, 29,
    np.where(base_dados['DDD5_gh33'] == 26, 0,
    np.where(base_dados['DDD5_gh33'] == 27, 29,
    np.where(base_dados['DDD5_gh33'] == 28, 0,
    np.where(base_dados['DDD5_gh33'] == 29, 29,
    0))))))))))))))))))))))))))))))
         
base_dados['DDD5_gh35'] = np.where(base_dados['DDD5_gh34'] == 0, 0,
    np.where(base_dados['DDD5_gh34'] == 2, 1,
    np.where(base_dados['DDD5_gh34'] == 29, 2,
    0)))
         
base_dados['DDD5_gh36'] = np.where(base_dados['DDD5_gh35'] == 0, 2,
    np.where(base_dados['DDD5_gh35'] == 1, 1,
    np.where(base_dados['DDD5_gh35'] == 2, 0,
    0)))

base_dados['DDD5_gh37'] = np.where(base_dados['DDD5_gh36'] == 0, 0,
    np.where(base_dados['DDD5_gh36'] == 1, 0,
    np.where(base_dados['DDD5_gh36'] == 2, 2,
    0)))

base_dados['DDD5_gh38'] = np.where(base_dados['DDD5_gh37'] == 0, 0,
    np.where(base_dados['DDD5_gh37'] == 2, 1,
    0))
                                   
                                   
                                   
                                   
                                   
                                   
                                   
base_dados['DDD4_gh30'] = np.where(base_dados['DDD4'] == 0, 0,
    np.where(base_dados['DDD4'] == 3, 1,
    np.where(base_dados['DDD4'] == 11, 2,
    np.where(base_dados['DDD4'] == 12, 3,
    np.where(base_dados['DDD4'] == 13, 4,
    np.where(base_dados['DDD4'] == 14, 5,
    np.where(base_dados['DDD4'] == 15, 6,
    np.where(base_dados['DDD4'] == 16, 7,
    np.where(base_dados['DDD4'] == 17, 8,
    np.where(base_dados['DDD4'] == 18, 9,
    np.where(base_dados['DDD4'] == 19, 10,
    np.where(base_dados['DDD4'] == 21, 11,
    np.where(base_dados['DDD4'] == 22, 12,
    np.where(base_dados['DDD4'] == 24, 13,
    np.where(base_dados['DDD4'] == 27, 14,
    np.where(base_dados['DDD4'] == 28, 15,
    np.where(base_dados['DDD4'] == 31, 16,
    np.where(base_dados['DDD4'] == 32, 17,
    np.where(base_dados['DDD4'] == 33, 18,
    np.where(base_dados['DDD4'] == 34, 19,
    np.where(base_dados['DDD4'] == 35, 20,
    np.where(base_dados['DDD4'] == 36, 21,
    np.where(base_dados['DDD4'] == 37, 22,
    np.where(base_dados['DDD4'] == 38, 23,
    np.where(base_dados['DDD4'] == 41, 24,
    np.where(base_dados['DDD4'] == 42, 25,
    np.where(base_dados['DDD4'] == 47, 26,
    np.where(base_dados['DDD4'] == 48, 27,
    np.where(base_dados['DDD4'] == 49, 28,
    np.where(base_dados['DDD4'] == 51, 29,
    np.where(base_dados['DDD4'] == 53, 30,
    np.where(base_dados['DDD4'] == 54, 31,
    np.where(base_dados['DDD4'] == 55, 32,
    np.where(base_dados['DDD4'] == 61, 33,
    np.where(base_dados['DDD4'] == 63, 34,
    np.where(base_dados['DDD4'] == 65, 35,
    np.where(base_dados['DDD4'] == 67, 36,
    np.where(base_dados['DDD4'] == 69, 37,
    np.where(base_dados['DDD4'] == 71, 38,
    np.where(base_dados['DDD4'] == 73, 39,
    np.where(base_dados['DDD4'] == 75, 40,
    np.where(base_dados['DDD4'] == 77, 41,
    np.where(base_dados['DDD4'] == 79, 42,
    np.where(base_dados['DDD4'] == 82, 43,
    np.where(base_dados['DDD4'] == 84, 44,
    np.where(base_dados['DDD4'] == 86, 45,
    np.where(base_dados['DDD4'] == 87, 46,
    np.where(base_dados['DDD4'] == 91, 47,
    np.where(base_dados['DDD4'] == 95, 48,
    np.where(base_dados['DDD4'] == 98, 49,
    0))))))))))))))))))))))))))))))))))))))))))))))))))

base_dados['DDD4_gh31'] = np.where(base_dados['DDD4_gh30'] == 0, 0,
    np.where(base_dados['DDD4_gh30'] == 1, 1,
    np.where(base_dados['DDD4_gh30'] == 2, 2,
    np.where(base_dados['DDD4_gh30'] == 3, 2,
    np.where(base_dados['DDD4_gh30'] == 4, 2,
    np.where(base_dados['DDD4_gh30'] == 5, 5,
    np.where(base_dados['DDD4_gh30'] == 6, 6,
    np.where(base_dados['DDD4_gh30'] == 7, 6,
    np.where(base_dados['DDD4_gh30'] == 8, 8,
    np.where(base_dados['DDD4_gh30'] == 9, 9,
    np.where(base_dados['DDD4_gh30'] == 10, 10,
    np.where(base_dados['DDD4_gh30'] == 11, 11,
    np.where(base_dados['DDD4_gh30'] == 12, 12,
    np.where(base_dados['DDD4_gh30'] == 13, 13,
    np.where(base_dados['DDD4_gh30'] == 14, 14,
    np.where(base_dados['DDD4_gh30'] == 15, 15,
    np.where(base_dados['DDD4_gh30'] == 16, 16,
    np.where(base_dados['DDD4_gh30'] == 17, 17,
    np.where(base_dados['DDD4_gh30'] == 18, 17,
    np.where(base_dados['DDD4_gh30'] == 19, 19,
    np.where(base_dados['DDD4_gh30'] == 20, 20,
    np.where(base_dados['DDD4_gh30'] == 21, 21,
    np.where(base_dados['DDD4_gh30'] == 22, 22,
    np.where(base_dados['DDD4_gh30'] == 23, 23,
    np.where(base_dados['DDD4_gh30'] == 24, 24,
    np.where(base_dados['DDD4_gh30'] == 25, 25,
    np.where(base_dados['DDD4_gh30'] == 26, 26,
    np.where(base_dados['DDD4_gh30'] == 27, 27,
    np.where(base_dados['DDD4_gh30'] == 28, 28,
    np.where(base_dados['DDD4_gh30'] == 29, 29,
    np.where(base_dados['DDD4_gh30'] == 30, 30,
    np.where(base_dados['DDD4_gh30'] == 31, 31,
    np.where(base_dados['DDD4_gh30'] == 32, 32,
    np.where(base_dados['DDD4_gh30'] == 33, 32,
    np.where(base_dados['DDD4_gh30'] == 34, 34,
    np.where(base_dados['DDD4_gh30'] == 35, 35,
    np.where(base_dados['DDD4_gh30'] == 36, 35,
    np.where(base_dados['DDD4_gh30'] == 37, 37,
    np.where(base_dados['DDD4_gh30'] == 38, 38,
    np.where(base_dados['DDD4_gh30'] == 39, 39,
    np.where(base_dados['DDD4_gh30'] == 40, 40,
    np.where(base_dados['DDD4_gh30'] == 41, 41,
    np.where(base_dados['DDD4_gh30'] == 42, 41,
    np.where(base_dados['DDD4_gh30'] == 43, 43,
    np.where(base_dados['DDD4_gh30'] == 44, 44,
    np.where(base_dados['DDD4_gh30'] == 45, 45,
    np.where(base_dados['DDD4_gh30'] == 46, 46,
    np.where(base_dados['DDD4_gh30'] == 47, 47,
    np.where(base_dados['DDD4_gh30'] == 48, 48,
    np.where(base_dados['DDD4_gh30'] == 49, 49,
    0))))))))))))))))))))))))))))))))))))))))))))))))))

base_dados['DDD4_gh32'] = np.where(base_dados['DDD4_gh31'] == 0, 0,
    np.where(base_dados['DDD4_gh31'] == 1, 1,
    np.where(base_dados['DDD4_gh31'] == 2, 2,
    np.where(base_dados['DDD4_gh31'] == 5, 3,
    np.where(base_dados['DDD4_gh31'] == 6, 4,
    np.where(base_dados['DDD4_gh31'] == 8, 5,
    np.where(base_dados['DDD4_gh31'] == 9, 6,
    np.where(base_dados['DDD4_gh31'] == 10, 7,
    np.where(base_dados['DDD4_gh31'] == 11, 8,
    np.where(base_dados['DDD4_gh31'] == 12, 9,
    np.where(base_dados['DDD4_gh31'] == 13, 10,
    np.where(base_dados['DDD4_gh31'] == 14, 11,
    np.where(base_dados['DDD4_gh31'] == 15, 12,
    np.where(base_dados['DDD4_gh31'] == 16, 13,
    np.where(base_dados['DDD4_gh31'] == 17, 14,
    np.where(base_dados['DDD4_gh31'] == 19, 15,
    np.where(base_dados['DDD4_gh31'] == 20, 16,
    np.where(base_dados['DDD4_gh31'] == 21, 17,
    np.where(base_dados['DDD4_gh31'] == 22, 18,
    np.where(base_dados['DDD4_gh31'] == 23, 19,
    np.where(base_dados['DDD4_gh31'] == 24, 20,
    np.where(base_dados['DDD4_gh31'] == 25, 21,
    np.where(base_dados['DDD4_gh31'] == 26, 22,
    np.where(base_dados['DDD4_gh31'] == 27, 23,
    np.where(base_dados['DDD4_gh31'] == 28, 24,
    np.where(base_dados['DDD4_gh31'] == 29, 25,
    np.where(base_dados['DDD4_gh31'] == 30, 26,
    np.where(base_dados['DDD4_gh31'] == 31, 27,
    np.where(base_dados['DDD4_gh31'] == 32, 28,
    np.where(base_dados['DDD4_gh31'] == 34, 29,
    np.where(base_dados['DDD4_gh31'] == 35, 30,
    np.where(base_dados['DDD4_gh31'] == 37, 31,
    np.where(base_dados['DDD4_gh31'] == 38, 32,
    np.where(base_dados['DDD4_gh31'] == 39, 33,
    np.where(base_dados['DDD4_gh31'] == 40, 34,
    np.where(base_dados['DDD4_gh31'] == 41, 35,
    np.where(base_dados['DDD4_gh31'] == 43, 36,
    np.where(base_dados['DDD4_gh31'] == 44, 37,
    np.where(base_dados['DDD4_gh31'] == 45, 38,
    np.where(base_dados['DDD4_gh31'] == 46, 39,
    np.where(base_dados['DDD4_gh31'] == 47, 40,
    np.where(base_dados['DDD4_gh31'] == 48, 41,
    np.where(base_dados['DDD4_gh31'] == 49, 42,
    0)))))))))))))))))))))))))))))))))))))))))))

base_dados['DDD4_gh33'] = np.where(base_dados['DDD4_gh32'] == 0, 0,
    np.where(base_dados['DDD4_gh32'] == 1, 1,
    np.where(base_dados['DDD4_gh32'] == 2, 2,
    np.where(base_dados['DDD4_gh32'] == 3, 3,
    np.where(base_dados['DDD4_gh32'] == 4, 4,
    np.where(base_dados['DDD4_gh32'] == 5, 5,
    np.where(base_dados['DDD4_gh32'] == 6, 6,
    np.where(base_dados['DDD4_gh32'] == 7, 7,
    np.where(base_dados['DDD4_gh32'] == 8, 8,
    np.where(base_dados['DDD4_gh32'] == 9, 9,
    np.where(base_dados['DDD4_gh32'] == 10, 10,
    np.where(base_dados['DDD4_gh32'] == 11, 11,
    np.where(base_dados['DDD4_gh32'] == 12, 12,
    np.where(base_dados['DDD4_gh32'] == 13, 13,
    np.where(base_dados['DDD4_gh32'] == 14, 14,
    np.where(base_dados['DDD4_gh32'] == 15, 15,
    np.where(base_dados['DDD4_gh32'] == 16, 16,
    np.where(base_dados['DDD4_gh32'] == 17, 17,
    np.where(base_dados['DDD4_gh32'] == 18, 18,
    np.where(base_dados['DDD4_gh32'] == 19, 19,
    np.where(base_dados['DDD4_gh32'] == 20, 20,
    np.where(base_dados['DDD4_gh32'] == 21, 21,
    np.where(base_dados['DDD4_gh32'] == 22, 22,
    np.where(base_dados['DDD4_gh32'] == 23, 23,
    np.where(base_dados['DDD4_gh32'] == 24, 24,
    np.where(base_dados['DDD4_gh32'] == 25, 25,
    np.where(base_dados['DDD4_gh32'] == 26, 26,
    np.where(base_dados['DDD4_gh32'] == 27, 27,
    np.where(base_dados['DDD4_gh32'] == 28, 28,
    np.where(base_dados['DDD4_gh32'] == 29, 29,
    np.where(base_dados['DDD4_gh32'] == 30, 30,
    np.where(base_dados['DDD4_gh32'] == 31, 31,
    np.where(base_dados['DDD4_gh32'] == 32, 32,
    np.where(base_dados['DDD4_gh32'] == 33, 33,
    np.where(base_dados['DDD4_gh32'] == 34, 34,
    np.where(base_dados['DDD4_gh32'] == 35, 35,
    np.where(base_dados['DDD4_gh32'] == 36, 36,
    np.where(base_dados['DDD4_gh32'] == 37, 37,
    np.where(base_dados['DDD4_gh32'] == 38, 38,
    np.where(base_dados['DDD4_gh32'] == 39, 39,
    np.where(base_dados['DDD4_gh32'] == 40, 40,
    np.where(base_dados['DDD4_gh32'] == 41, 41,
    np.where(base_dados['DDD4_gh32'] == 42, 42,
    0)))))))))))))))))))))))))))))))))))))))))))
             
base_dados['DDD4_gh34'] = np.where(base_dados['DDD4_gh33'] == 0, 0,
    np.where(base_dados['DDD4_gh33'] == 1, 0,
    np.where(base_dados['DDD4_gh33'] == 2, 2,
    np.where(base_dados['DDD4_gh33'] == 3, 0,
    np.where(base_dados['DDD4_gh33'] == 4, 2,
    np.where(base_dados['DDD4_gh33'] == 5, 33,
    np.where(base_dados['DDD4_gh33'] == 6, 33,
    np.where(base_dados['DDD4_gh33'] == 7, 33,
    np.where(base_dados['DDD4_gh33'] == 8, 0,
    np.where(base_dados['DDD4_gh33'] == 9, 33,
    np.where(base_dados['DDD4_gh33'] == 10, 0,
    np.where(base_dados['DDD4_gh33'] == 11, 0,
    np.where(base_dados['DDD4_gh33'] == 12, 0,
    np.where(base_dados['DDD4_gh33'] == 13, 33,
    np.where(base_dados['DDD4_gh33'] == 14, 0,
    np.where(base_dados['DDD4_gh33'] == 15, 33,
    np.where(base_dados['DDD4_gh33'] == 16, 33,
    np.where(base_dados['DDD4_gh33'] == 17, 33,
    np.where(base_dados['DDD4_gh33'] == 18, 0,
    np.where(base_dados['DDD4_gh33'] == 19, 2,
    np.where(base_dados['DDD4_gh33'] == 20, 33,
    np.where(base_dados['DDD4_gh33'] == 21, 33,
    np.where(base_dados['DDD4_gh33'] == 22, 0,
    np.where(base_dados['DDD4_gh33'] == 23, 0,
    np.where(base_dados['DDD4_gh33'] == 24, 33,
    np.where(base_dados['DDD4_gh33'] == 25, 2,
    np.where(base_dados['DDD4_gh33'] == 26, 0,
    np.where(base_dados['DDD4_gh33'] == 27, 33,
    np.where(base_dados['DDD4_gh33'] == 28, 0,
    np.where(base_dados['DDD4_gh33'] == 29, 0,
    np.where(base_dados['DDD4_gh33'] == 30, 2,
    np.where(base_dados['DDD4_gh33'] == 31, 0,
    np.where(base_dados['DDD4_gh33'] == 32, 0,
    np.where(base_dados['DDD4_gh33'] == 33, 33,
    np.where(base_dados['DDD4_gh33'] == 34, 0,
    np.where(base_dados['DDD4_gh33'] == 35, 33,
    np.where(base_dados['DDD4_gh33'] == 36, 0,
    np.where(base_dados['DDD4_gh33'] == 37, 33,
    np.where(base_dados['DDD4_gh33'] == 38, 0,
    np.where(base_dados['DDD4_gh33'] == 39, 33,
    np.where(base_dados['DDD4_gh33'] == 40, 0,
    np.where(base_dados['DDD4_gh33'] == 41, 33,
    np.where(base_dados['DDD4_gh33'] == 42, 0,
    0)))))))))))))))))))))))))))))))))))))))))))
             
base_dados['DDD4_gh35'] = np.where(base_dados['DDD4_gh34'] == 0, 0,
    np.where(base_dados['DDD4_gh34'] == 2, 1,
    np.where(base_dados['DDD4_gh34'] == 33, 2,
    0)))

base_dados['DDD4_gh36'] = np.where(base_dados['DDD4_gh35'] == 0, 2,
    np.where(base_dados['DDD4_gh35'] == 1, 1,
    np.where(base_dados['DDD4_gh35'] == 2, 0,
    0)))

base_dados['DDD4_gh37'] = np.where(base_dados['DDD4_gh36'] == 0, 1,
    np.where(base_dados['DDD4_gh36'] == 1, 1,
    np.where(base_dados['DDD4_gh36'] == 2, 2,
    0)))

base_dados['DDD4_gh38'] = np.where(base_dados['DDD4_gh37'] == 1, 0,
    np.where(base_dados['DDD4_gh37'] == 2, 1,
    0))
                                   
                                   
                                   
                                   
                                   
                                   
base_dados['COD_FILA_gh30'] = np.where(base_dados['COD_FILA'] == 332, 0,
    np.where(base_dados['COD_FILA'] == 334, 1,
    np.where(base_dados['COD_FILA'] == 339, 2,
    0)))

base_dados['COD_FILA_gh31'] = np.where(base_dados['COD_FILA_gh30'] == 0, 0,
    np.where(base_dados['COD_FILA_gh30'] == 1, 0,
    np.where(base_dados['COD_FILA_gh30'] == 2, 2,
    0)))

base_dados['COD_FILA_gh32'] = np.where(base_dados['COD_FILA_gh31'] == 0, 0,
    np.where(base_dados['COD_FILA_gh31'] == 2, 1,
    0))

base_dados['COD_FILA_gh33'] = np.where(base_dados['COD_FILA_gh32'] == 0, 0,
    np.where(base_dados['COD_FILA_gh32'] == 1, 1,
    0))

base_dados['COD_FILA_gh34'] = np.where(base_dados['COD_FILA_gh33'] == 0, 0,
    np.where(base_dados['COD_FILA_gh33'] == 1, 1,
    0))

base_dados['COD_FILA_gh35'] = np.where(base_dados['COD_FILA_gh34'] == 0, 0,
    np.where(base_dados['COD_FILA_gh34'] == 1, 1,
    0))

base_dados['COD_FILA_gh36'] = np.where(base_dados['COD_FILA_gh35'] == 0, 1,
    np.where(base_dados['COD_FILA_gh35'] == 1, 0,
    0))

base_dados['COD_FILA_gh37'] = np.where(base_dados['COD_FILA_gh36'] == 0, 0,
    np.where(base_dados['COD_FILA_gh36'] == 1, 1,
    0))

base_dados['COD_FILA_gh38'] = np.where(base_dados['COD_FILA_gh37'] == 0, 0,
    np.where(base_dados['COD_FILA_gh37'] == 1, 1,
    0))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Gerando variáveis numéricas contínuas

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parte 1 de 2

# COMMAND ----------


base_dados['DIAS_ATRASO__pe_7'] = np.where(base_dados['DIAS_ATRASO'] <= 800.0, 0.0,
    np.where(np.bitwise_and(base_dados['DIAS_ATRASO'] > 800.0, base_dados['DIAS_ATRASO'] <= 1597.0), 1.0,
    np.where(np.bitwise_and(base_dados['DIAS_ATRASO'] > 1597.0, base_dados['DIAS_ATRASO'] <= 2400.0), 2.0,
    np.where(np.bitwise_and(base_dados['DIAS_ATRASO'] > 2400.0, base_dados['DIAS_ATRASO'] <= 3201.0), 3.0,
    np.where(np.bitwise_and(base_dados['DIAS_ATRASO'] > 3201.0, base_dados['DIAS_ATRASO'] <= 3998.0), 4.0,
    np.where(np.bitwise_and(base_dados['DIAS_ATRASO'] > 3998.0, base_dados['DIAS_ATRASO'] <= 4800.0), 5.0,
    np.where(base_dados['DIAS_ATRASO'] > 4800.0, 6.0,
     -2)))))))

base_dados['DIAS_ATRASO__pe_7_g_1_1'] = np.where(base_dados['DIAS_ATRASO__pe_7'] == -2.0, 2,
    np.where(base_dados['DIAS_ATRASO__pe_7'] == 0.0, 0,
    np.where(base_dados['DIAS_ATRASO__pe_7'] == 1.0, 1,
    np.where(base_dados['DIAS_ATRASO__pe_7'] == 2.0, 2,
    np.where(base_dados['DIAS_ATRASO__pe_7'] == 3.0, 1,
    np.where(base_dados['DIAS_ATRASO__pe_7'] == 4.0, 2,
    np.where(base_dados['DIAS_ATRASO__pe_7'] == 5.0, 2,
    np.where(base_dados['DIAS_ATRASO__pe_7'] == 6.0, 2,
     0))))))))
         
base_dados['DIAS_ATRASO__pe_7_g_1_2'] = np.where(base_dados['DIAS_ATRASO__pe_7_g_1_1'] == 0, 2,
    np.where(base_dados['DIAS_ATRASO__pe_7_g_1_1'] == 1, 1,
    np.where(base_dados['DIAS_ATRASO__pe_7_g_1_1'] == 2, 0,
     0)))
         
base_dados['DIAS_ATRASO__pe_7_g_1'] = np.where(base_dados['DIAS_ATRASO__pe_7_g_1_2'] == 0, 0,
    np.where(base_dados['DIAS_ATRASO__pe_7_g_1_2'] == 1, 1,
    np.where(base_dados['DIAS_ATRASO__pe_7_g_1_2'] == 2, 2,
     0)))
         
         
         
         
         
base_dados['DIAS_ATRASO__L'] = np.log(base_dados['DIAS_ATRASO'])
np.where(base_dados['DIAS_ATRASO__L'] == 0, -1, base_dados['DIAS_ATRASO__L'])
base_dados['DIAS_ATRASO__L'] = base_dados['DIAS_ATRASO__L'].fillna(0)
base_dados['DIAS_ATRASO__L__p_5'] = np.where(base_dados['DIAS_ATRASO__L'] <= 5.075173815233827, 0.0,
    np.where(np.bitwise_and(base_dados['DIAS_ATRASO__L'] > 5.075173815233827, base_dados['DIAS_ATRASO__L'] <= 5.988961416889864), 1.0,
    np.where(np.bitwise_and(base_dados['DIAS_ATRASO__L'] > 5.988961416889864, base_dados['DIAS_ATRASO__L'] <= 7.029087564149662), 2.0,
    np.where(np.bitwise_and(base_dados['DIAS_ATRASO__L'] > 7.029087564149662, base_dados['DIAS_ATRASO__L'] <= 7.805474625270857), 3.0,
    np.where(base_dados['DIAS_ATRASO__L'] > 7.805474625270857, 4.0,
     0)))))
         
base_dados['DIAS_ATRASO__L__p_5_g_1_1'] = np.where(base_dados['DIAS_ATRASO__L__p_5'] == 0, 1,
    np.where(base_dados['DIAS_ATRASO__L__p_5'] == 1, 0,
    np.where(base_dados['DIAS_ATRASO__L__p_5'] == 2, 2,
    np.where(base_dados['DIAS_ATRASO__L__p_5'] == 3, 2,
    np.where(base_dados['DIAS_ATRASO__L__p_5'] == 4, 3,
     0)))))
         
base_dados['DIAS_ATRASO__L__p_5_g_1_2'] = np.where(base_dados['DIAS_ATRASO__L__p_5_g_1_1'] == 0, 2,
    np.where(base_dados['DIAS_ATRASO__L__p_5_g_1_1'] == 1, 3,
    np.where(base_dados['DIAS_ATRASO__L__p_5_g_1_1'] == 2, 1,
    np.where(base_dados['DIAS_ATRASO__L__p_5_g_1_1'] == 3, 0,
     0))))
         
base_dados['DIAS_ATRASO__L__p_5_g_1'] = np.where(base_dados['DIAS_ATRASO__L__p_5_g_1_2'] == 0, 0,
    np.where(base_dados['DIAS_ATRASO__L__p_5_g_1_2'] == 1, 1,
    np.where(base_dados['DIAS_ATRASO__L__p_5_g_1_2'] == 2, 2,
    np.where(base_dados['DIAS_ATRASO__L__p_5_g_1_2'] == 3, 3,
     0))))
                                                  
                                                  
                                                  
                                                  
                                                  
                                                  
base_dados['CODIGO__pe_5'] = np.where(base_dados['CODIGO'] <= 944544.0, 0.0,
    np.where(np.bitwise_and(base_dados['CODIGO'] > 944544.0, base_dados['CODIGO'] <= 1904329.0), 1.0,
    np.where(np.bitwise_and(base_dados['CODIGO'] > 1904329.0, base_dados['CODIGO'] <= 2856414.0), 2.0,
    np.where(np.bitwise_and(base_dados['CODIGO'] > 2856414.0, base_dados['CODIGO'] <= 3808893.0), 3.0,
    np.where(base_dados['CODIGO'] > 3808893.0, 4.0,
     -2)))))
         
base_dados['CODIGO__pe_5_g_1_1'] = np.where(base_dados['CODIGO__pe_5'] == -2.0, 3,
    np.where(base_dados['CODIGO__pe_5'] == 0.0, 0,
    np.where(base_dados['CODIGO__pe_5'] == 1.0, 1,
    np.where(base_dados['CODIGO__pe_5'] == 2.0, 2,
    np.where(base_dados['CODIGO__pe_5'] == 3.0, 2,
    np.where(base_dados['CODIGO__pe_5'] == 4.0, 3,
     0))))))
         
base_dados['CODIGO__pe_5_g_1_2'] = np.where(base_dados['CODIGO__pe_5_g_1_1'] == 0, 0,
    np.where(base_dados['CODIGO__pe_5_g_1_1'] == 1, 1,
    np.where(base_dados['CODIGO__pe_5_g_1_1'] == 2, 2,
    np.where(base_dados['CODIGO__pe_5_g_1_1'] == 3, 3,
     0))))
         
base_dados['CODIGO__pe_5_g_1'] = np.where(base_dados['CODIGO__pe_5_g_1_2'] == 0, 0,
    np.where(base_dados['CODIGO__pe_5_g_1_2'] == 1, 1,
    np.where(base_dados['CODIGO__pe_5_g_1_2'] == 2, 2,
    np.where(base_dados['CODIGO__pe_5_g_1_2'] == 3, 3,
     0))))
         
         
         
         
         
         
base_dados['CODIGO__L'] = np.log(base_dados['CODIGO'])
np.where(base_dados['CODIGO__L'] == 0, -1, base_dados['CODIGO__L'])
base_dados['CODIGO__L'] = base_dados['CODIGO__L'].fillna(0)
base_dados['CODIGO__L__p_5'] = np.where(base_dados['CODIGO__L'] <= 14.236012707615734, 0.0,
    np.where(np.bitwise_and(base_dados['CODIGO__L'] > 14.236012707615734, base_dados['CODIGO__L'] <= 14.679971430473001), 1.0,
    np.where(np.bitwise_and(base_dados['CODIGO__L'] > 14.679971430473001, base_dados['CODIGO__L'] <= 14.958208948154466), 2.0,
    np.where(np.bitwise_and(base_dados['CODIGO__L'] > 14.958208948154466, base_dados['CODIGO__L'] <= 15.213850083942608), 3.0,
    np.where(base_dados['CODIGO__L'] > 15.213850083942608, 4.0,
     0)))))
         
base_dados['CODIGO__L__p_5_g_1_1'] = np.where(base_dados['CODIGO__L__p_5'] == 0, 0,
    np.where(base_dados['CODIGO__L__p_5'] == 1, 0,
    np.where(base_dados['CODIGO__L__p_5'] == 2, 1,
    np.where(base_dados['CODIGO__L__p_5'] == 3, 1,
    np.where(base_dados['CODIGO__L__p_5'] == 4, 2,
     0)))))
         
base_dados['CODIGO__L__p_5_g_1_2'] = np.where(base_dados['CODIGO__L__p_5_g_1_1'] == 0, 0,
    np.where(base_dados['CODIGO__L__p_5_g_1_1'] == 1, 1,
    np.where(base_dados['CODIGO__L__p_5_g_1_1'] == 2, 2,
     0)))
         
base_dados['CODIGO__L__p_5_g_1'] = np.where(base_dados['CODIGO__L__p_5_g_1_2'] == 0, 0,
    np.where(base_dados['CODIGO__L__p_5_g_1_2'] == 1, 1,
    np.where(base_dados['CODIGO__L__p_5_g_1_2'] == 2, 2,
     0)))
         
         
         
         
         
         
base_dados['DDD1__L'] = np.log(base_dados['DDD1'])
np.where(base_dados['DDD1__L'] == 0, -1, base_dados['DDD1__L'])
base_dados['DDD1__L'] = base_dados['DDD1__L'].fillna(-2)
base_dados['DDD1__L__pe_15'] = np.where(np.bitwise_and(base_dados['DDD1__L'] >= -1.0, base_dados['DDD1__L'] <= 2.5649493574615367), 6.0,
    np.where(np.bitwise_and(base_dados['DDD1__L'] > 2.5649493574615367, base_dados['DDD1__L'] <= 2.8903717578961645), 7.0,
    np.where(np.bitwise_and(base_dados['DDD1__L'] > 2.8903717578961645, base_dados['DDD1__L'] <= 3.295836866004329), 8.0,
    np.where(np.bitwise_and(base_dados['DDD1__L'] > 3.295836866004329, base_dados['DDD1__L'] <= 3.6375861597263857), 9.0,
    np.where(np.bitwise_and(base_dados['DDD1__L'] > 3.6375861597263857, base_dados['DDD1__L'] <= 4.007333185232471), 10.0,
    np.where(np.bitwise_and(base_dados['DDD1__L'] > 4.007333185232471, base_dados['DDD1__L'] <= 4.406719247264253), 11.0,
    np.where(base_dados['DDD1__L'] > 4.406719247264253, 12.0,
     -2)))))))
         
base_dados['DDD1__L__pe_15_g_1_1'] = np.where(base_dados['DDD1__L__pe_15'] == -2.0, 1,
    np.where(base_dados['DDD1__L__pe_15'] == 6.0, 0,
    np.where(base_dados['DDD1__L__pe_15'] == 7.0, 0,
    np.where(base_dados['DDD1__L__pe_15'] == 8.0, 1,
    np.where(base_dados['DDD1__L__pe_15'] == 9.0, 0,
    np.where(base_dados['DDD1__L__pe_15'] == 10.0, 2,
    np.where(base_dados['DDD1__L__pe_15'] == 11.0, 0,
    np.where(base_dados['DDD1__L__pe_15'] == 12.0, 2,
     0))))))))
         
base_dados['DDD1__L__pe_15_g_1_2'] = np.where(base_dados['DDD1__L__pe_15_g_1_1'] == 0, 1,
    np.where(base_dados['DDD1__L__pe_15_g_1_1'] == 1, 0,
    np.where(base_dados['DDD1__L__pe_15_g_1_1'] == 2, 2,
     0)))
         
base_dados['DDD1__L__pe_15_g_1'] = np.where(base_dados['DDD1__L__pe_15_g_1_2'] == 0, 0,
    np.where(base_dados['DDD1__L__pe_15_g_1_2'] == 1, 1,
    np.where(base_dados['DDD1__L__pe_15_g_1_2'] == 2, 2,
     0)))
         
         
         
         
         
base_dados['DDD1__S'] = np.sin(base_dados['DDD1'])
np.where(base_dados['DDD1__S'] == 0, -1, base_dados['DDD1__S'])
base_dados['DDD1__S'] = base_dados['DDD1__S'].fillna(-2)
base_dados['DDD1__S__p_10'] = np.where(np.bitwise_and(base_dados['DDD1__S'] >= -1.0, base_dados['DDD1__S'] <= 0.14987720966295234), 7.0,
    np.where(np.bitwise_and(base_dados['DDD1__S'] > 0.14987720966295234, base_dados['DDD1__S'] <= 0.6702291758433747), 8.0,
    np.where(base_dados['DDD1__S'] > 0.6702291758433747, 9.0,
     -2)))
         
base_dados['DDD1__S__p_10_g_1_1'] = np.where(base_dados['DDD1__S__p_10'] == -2.0, 0,
    np.where(base_dados['DDD1__S__p_10'] == 7.0, 1,
    np.where(base_dados['DDD1__S__p_10'] == 8.0, 1,
    np.where(base_dados['DDD1__S__p_10'] == 9.0, 1,
     0))))
         
base_dados['DDD1__S__p_10_g_1_2'] = np.where(base_dados['DDD1__S__p_10_g_1_1'] == 0, 0,
    np.where(base_dados['DDD1__S__p_10_g_1_1'] == 1, 1,
     0))
                                             
base_dados['DDD1__S__p_10_g_1'] = np.where(base_dados['DDD1__S__p_10_g_1_2'] == 0, 0,
    np.where(base_dados['DDD1__S__p_10_g_1_2'] == 1, 1,
     0))
         
         
         
         
         
         
         
base_dados['LIMITE__pe_10'] = np.where(base_dados['LIMITE'] == 0 , -1.0,
    np.where(np.bitwise_and(base_dados['LIMITE'] > 0.0, base_dados['LIMITE'] <= 220.0), 0.0,
    np.where(np.bitwise_and(base_dados['LIMITE'] > 220.0, base_dados['LIMITE'] <= 470.0), 1.0,
    np.where(np.bitwise_and(base_dados['LIMITE'] > 470.0, base_dados['LIMITE'] <= 705.0), 2.0,
    np.where(np.bitwise_and(base_dados['LIMITE'] > 705.0, base_dados['LIMITE'] <= 940.0), 3.0,
    np.where(np.bitwise_and(base_dados['LIMITE'] > 940.0, base_dados['LIMITE'] <= 1170.0), 4.0,
    np.where(np.bitwise_and(base_dados['LIMITE'] > 1170.0, base_dados['LIMITE'] <= 1410.0), 5.0,
    np.where(np.bitwise_and(base_dados['LIMITE'] > 1410.0, base_dados['LIMITE'] <= 1640.0), 6.0,
    np.where(np.bitwise_and(base_dados['LIMITE'] > 1640.0, base_dados['LIMITE'] <= 1870.0), 7.0,
    np.where(np.bitwise_and(base_dados['LIMITE'] > 1870.0, base_dados['LIMITE'] <= 2110.0), 8.0,
    np.where(base_dados['LIMITE'] > 2110.0, 9.0,
     -2)))))))))))
         
base_dados['LIMITE__pe_10_g_1_1'] = np.where(base_dados['LIMITE__pe_10'] == -2.0, 1,
    np.where(base_dados['LIMITE__pe_10'] == -1.0, 1,
    np.where(base_dados['LIMITE__pe_10'] == 0.0, 1,
    np.where(base_dados['LIMITE__pe_10'] == 1.0, 0,
    np.where(base_dados['LIMITE__pe_10'] == 2.0, 0,
    np.where(base_dados['LIMITE__pe_10'] == 3.0, 1,
    np.where(base_dados['LIMITE__pe_10'] == 4.0, 1,
    np.where(base_dados['LIMITE__pe_10'] == 5.0, 1,
    np.where(base_dados['LIMITE__pe_10'] == 6.0, 1,
    np.where(base_dados['LIMITE__pe_10'] == 7.0, 1,
    np.where(base_dados['LIMITE__pe_10'] == 8.0, 1,
    np.where(base_dados['LIMITE__pe_10'] == 9.0, 1,
     0))))))))))))
         
base_dados['LIMITE__pe_10_g_1_2'] = np.where(base_dados['LIMITE__pe_10_g_1_1'] == 0, 1,
    np.where(base_dados['LIMITE__pe_10_g_1_1'] == 1, 0,
     0))
                                             
base_dados['LIMITE__pe_10_g_1'] = np.where(base_dados['LIMITE__pe_10_g_1_2'] == 0, 0,
    np.where(base_dados['LIMITE__pe_10_g_1_2'] == 1, 1,
     0))
                                           
                                           
                                           
                                           
                                           
                                           
                                           
base_dados['LIMITE__C'] = np.cos(base_dados['LIMITE'])
np.where(base_dados['LIMITE__C'] == 0, -1, base_dados['LIMITE__C'])
base_dados['LIMITE__C'] = base_dados['LIMITE__C'].fillna(0)
base_dados['LIMITE__C__p_10'] = np.where(np.bitwise_and(base_dados['LIMITE__C'] >= -1.0, base_dados['LIMITE__C'] <= 0.287975485911901), 7.0,
    np.where(np.bitwise_and(base_dados['LIMITE__C'] > 0.287975485911901, base_dados['LIMITE__C'] <= 0.7597467172716508), 8.0,
    np.where(base_dados['LIMITE__C'] > 0.7597467172716508, 9.0,
     -2)))
         
base_dados['LIMITE__C__p_10_g_1_1'] = np.where(base_dados['LIMITE__C__p_10'] == -2.0, 0,
    np.where(base_dados['LIMITE__C__p_10'] == 7.0, 1,
    np.where(base_dados['LIMITE__C__p_10'] == 8.0, 1,
    np.where(base_dados['LIMITE__C__p_10'] == 9.0, 1,
     0))))
         
base_dados['LIMITE__C__p_10_g_1_2'] = np.where(base_dados['LIMITE__C__p_10_g_1_1'] == 0, 1,
    np.where(base_dados['LIMITE__C__p_10_g_1_1'] == 1, 0,
     0))
                                               
base_dados['LIMITE__C__p_10_g_1'] = np.where(base_dados['LIMITE__C__p_10_g_1_2'] == 0, 0,
    np.where(base_dados['LIMITE__C__p_10_g_1_2'] == 1, 1,
     0))
                                             
                                             
                                             
                                             
                                             
                                             
                                             
                                             
base_dados['COD_EMPRESA__p_6'] = np.where(base_dados['COD_EMPRESA'] <= 87, 0,
    np.where(np.bitwise_and(base_dados['COD_EMPRESA'] > 87, base_dados['COD_EMPRESA'] <= 161), 1,
    np.where(np.bitwise_and(base_dados['COD_EMPRESA'] > 161, base_dados['COD_EMPRESA'] <= 176), 2,
    np.where(np.bitwise_and(base_dados['COD_EMPRESA'] > 176, base_dados['COD_EMPRESA'] <= 258), 3,
    np.where(np.bitwise_and(base_dados['COD_EMPRESA'] > 258, base_dados['COD_EMPRESA'] <= 287), 4,
    np.where(base_dados['COD_EMPRESA'] > 287, 5,
     0))))))
         
base_dados['COD_EMPRESA__p_6_g_1_1'] = np.where(base_dados['COD_EMPRESA__p_6'] == 0, 0,
    np.where(base_dados['COD_EMPRESA__p_6'] == 1, 1,
    np.where(base_dados['COD_EMPRESA__p_6'] == 2, 2,
    np.where(base_dados['COD_EMPRESA__p_6'] == 3, 2,
    np.where(base_dados['COD_EMPRESA__p_6'] == 4, 2,
    np.where(base_dados['COD_EMPRESA__p_6'] == 5, 3,
     0))))))
         
base_dados['COD_EMPRESA__p_6_g_1_2'] = np.where(base_dados['COD_EMPRESA__p_6_g_1_1'] == 0, 1,
    np.where(base_dados['COD_EMPRESA__p_6_g_1_1'] == 1, 0,
    np.where(base_dados['COD_EMPRESA__p_6_g_1_1'] == 2, 2,
    np.where(base_dados['COD_EMPRESA__p_6_g_1_1'] == 3, 3,
     0))))
         
base_dados['COD_EMPRESA__p_6_g_1'] = np.where(base_dados['COD_EMPRESA__p_6_g_1_2'] == 0, 0,
    np.where(base_dados['COD_EMPRESA__p_6_g_1_2'] == 1, 1,
    np.where(base_dados['COD_EMPRESA__p_6_g_1_2'] == 2, 2,
    np.where(base_dados['COD_EMPRESA__p_6_g_1_2'] == 3, 3,
     0))))
         
         
         
         
         
         
         
base_dados['COD_EMPRESA__S'] = np.sin(base_dados['COD_EMPRESA'])
np.where(base_dados['COD_EMPRESA__S'] == 0, -1, base_dados['COD_EMPRESA__S'])
base_dados['COD_EMPRESA__S'] = base_dados['COD_EMPRESA__S'].fillna(0)
base_dados['COD_EMPRESA__S__p_34'] = np.where(np.bitwise_and(base_dados['COD_EMPRESA__S'] >= -0.9999903395061709, base_dados['COD_EMPRESA__S'] <= 0.03539830273366068), 19.0,
    np.where(np.bitwise_and(base_dados['COD_EMPRESA__S'] > 0.03539830273366068, base_dados['COD_EMPRESA__S'] <= 0.07075223608034517), 20.0,
    np.where(np.bitwise_and(base_dados['COD_EMPRESA__S'] > 0.07075223608034517, base_dados['COD_EMPRESA__S'] <= 0.16732598101183924), 21.0,
    np.where(np.bitwise_and(base_dados['COD_EMPRESA__S'] > 0.16732598101183924, base_dados['COD_EMPRESA__S'] <= 0.21945466799406363), 22.0,
    np.where(np.bitwise_and(base_dados['COD_EMPRESA__S'] > 0.21945466799406363, base_dados['COD_EMPRESA__S'] <= 0.346621180094276), 23.0,
    np.where(np.bitwise_and(base_dados['COD_EMPRESA__S'] > 0.346621180094276, base_dados['COD_EMPRESA__S'] <= 0.4121184852417566), 24.0,
    np.where(np.bitwise_and(base_dados['COD_EMPRESA__S'] > 0.4121184852417566, base_dados['COD_EMPRESA__S'] <= 0.49873928180328125), 25.0,
    np.where(np.bitwise_and(base_dados['COD_EMPRESA__S'] > 0.49873928180328125, base_dados['COD_EMPRESA__S'] <= 0.5140043136735694), 26.0,
    np.where(np.bitwise_and(base_dados['COD_EMPRESA__S'] > 0.5140043136735694, base_dados['COD_EMPRESA__S'] <= 0.6702291758433747), 27.0,
    np.where(np.bitwise_and(base_dados['COD_EMPRESA__S'] > 0.6702291758433747, base_dados['COD_EMPRESA__S'] <= 0.8939966636005579), 28.0,
    np.where(np.bitwise_and(base_dados['COD_EMPRESA__S'] > 0.8939966636005579, base_dados['COD_EMPRESA__S'] <= 0.9200142254959646), 29.0,
    np.where(np.bitwise_and(base_dados['COD_EMPRESA__S'] > 0.9200142254959646, base_dados['COD_EMPRESA__S'] <= 0.9510639681125854), 30.0,
    np.where(np.bitwise_and(base_dados['COD_EMPRESA__S'] > 0.9510639681125854, base_dados['COD_EMPRESA__S'] <= 0.9683644611001854), 31.0,
    np.where(base_dados['COD_EMPRESA__S'] > 0.9683644611001854, 32.0,
     -2))))))))))))))
         
base_dados['COD_EMPRESA__S__p_34_g_1_1'] = np.where(base_dados['COD_EMPRESA__S__p_34'] == -2.0, 0,
    np.where(base_dados['COD_EMPRESA__S__p_34'] == 19.0, 1,
    np.where(base_dados['COD_EMPRESA__S__p_34'] == 20.0, 2,
    np.where(base_dados['COD_EMPRESA__S__p_34'] == 21.0, 2,
    np.where(base_dados['COD_EMPRESA__S__p_34'] == 22.0, 2,
    np.where(base_dados['COD_EMPRESA__S__p_34'] == 23.0, 1,
    np.where(base_dados['COD_EMPRESA__S__p_34'] == 24.0, 1,
    np.where(base_dados['COD_EMPRESA__S__p_34'] == 25.0, 1,
    np.where(base_dados['COD_EMPRESA__S__p_34'] == 26.0, 1,
    np.where(base_dados['COD_EMPRESA__S__p_34'] == 27.0, 1,
    np.where(base_dados['COD_EMPRESA__S__p_34'] == 28.0, 1,
    np.where(base_dados['COD_EMPRESA__S__p_34'] == 29.0, 0,
    np.where(base_dados['COD_EMPRESA__S__p_34'] == 30.0, 2,
    np.where(base_dados['COD_EMPRESA__S__p_34'] == 31.0, 2,
    np.where(base_dados['COD_EMPRESA__S__p_34'] == 32.0, 2,
     0)))))))))))))))
         
base_dados['COD_EMPRESA__S__p_34_g_1_2'] = np.where(base_dados['COD_EMPRESA__S__p_34_g_1_1'] == 0, 1,
    np.where(base_dados['COD_EMPRESA__S__p_34_g_1_1'] == 1, 0,
    np.where(base_dados['COD_EMPRESA__S__p_34_g_1_1'] == 2, 1,
     0)))
         
base_dados['COD_EMPRESA__S__p_34_g_1'] = np.where(base_dados['COD_EMPRESA__S__p_34_g_1_2'] == 0, 0,
    np.where(base_dados['COD_EMPRESA__S__p_34_g_1_2'] == 1, 1,
     0))

         

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parte 2 de 2

# COMMAND ----------


base_dados['DIAS_ATRASO__L__p_5_g_1_c1_23_1'] = np.where(np.bitwise_and(base_dados['DIAS_ATRASO__pe_7_g_1'] == 0, base_dados['DIAS_ATRASO__L__p_5_g_1'] == 0), 0,
    np.where(np.bitwise_and(base_dados['DIAS_ATRASO__pe_7_g_1'] == 0, base_dados['DIAS_ATRASO__L__p_5_g_1'] == 1), 1,
    np.where(np.bitwise_and(base_dados['DIAS_ATRASO__pe_7_g_1'] == 1, base_dados['DIAS_ATRASO__L__p_5_g_1'] == 0), 1,
    np.where(np.bitwise_and(base_dados['DIAS_ATRASO__pe_7_g_1'] == 1, base_dados['DIAS_ATRASO__L__p_5_g_1'] == 1), 2,
    np.where(np.bitwise_and(base_dados['DIAS_ATRASO__pe_7_g_1'] == 2, base_dados['DIAS_ATRASO__L__p_5_g_1'] == 1), 3,
    np.where(np.bitwise_and(base_dados['DIAS_ATRASO__pe_7_g_1'] == 2, base_dados['DIAS_ATRASO__L__p_5_g_1'] == 2), 4,
    np.where(np.bitwise_and(base_dados['DIAS_ATRASO__pe_7_g_1'] == 2, base_dados['DIAS_ATRASO__L__p_5_g_1'] == 3), 5,
     0)))))))

base_dados['DIAS_ATRASO__L__p_5_g_1_c1_23_2'] = np.where(base_dados['DIAS_ATRASO__L__p_5_g_1_c1_23_1'] == 0, 0,
    np.where(base_dados['DIAS_ATRASO__L__p_5_g_1_c1_23_1'] == 1, 1,
    np.where(base_dados['DIAS_ATRASO__L__p_5_g_1_c1_23_1'] == 2, 2,
    np.where(base_dados['DIAS_ATRASO__L__p_5_g_1_c1_23_1'] == 3, 3,
    np.where(base_dados['DIAS_ATRASO__L__p_5_g_1_c1_23_1'] == 4, 4,
    np.where(base_dados['DIAS_ATRASO__L__p_5_g_1_c1_23_1'] == 5, 5,
    0))))))
base_dados['DIAS_ATRASO__L__p_5_g_1_c1_23'] = np.where(base_dados['DIAS_ATRASO__L__p_5_g_1_c1_23_2'] == 0, 0,
    np.where(base_dados['DIAS_ATRASO__L__p_5_g_1_c1_23_2'] == 1, 1,
    np.where(base_dados['DIAS_ATRASO__L__p_5_g_1_c1_23_2'] == 2, 2,
    np.where(base_dados['DIAS_ATRASO__L__p_5_g_1_c1_23_2'] == 3, 3,
    np.where(base_dados['DIAS_ATRASO__L__p_5_g_1_c1_23_2'] == 4, 4,
    np.where(base_dados['DIAS_ATRASO__L__p_5_g_1_c1_23_2'] == 5, 5,
     0))))))
         
         
         
         
base_dados['CODIGO__L__p_5_g_1_c1_21_1'] = np.where(np.bitwise_and(base_dados['CODIGO__pe_5_g_1'] == 0, base_dados['CODIGO__L__p_5_g_1'] == 0), 0,
    np.where(np.bitwise_and(base_dados['CODIGO__pe_5_g_1'] == 1, base_dados['CODIGO__L__p_5_g_1'] == 0), 1,
    np.where(np.bitwise_and(base_dados['CODIGO__pe_5_g_1'] == 2, base_dados['CODIGO__L__p_5_g_1'] == 0), 2,
    np.where(np.bitwise_and(base_dados['CODIGO__pe_5_g_1'] == 2, base_dados['CODIGO__L__p_5_g_1'] == 1), 3,
    np.where(np.bitwise_and(base_dados['CODIGO__pe_5_g_1'] == 3, base_dados['CODIGO__L__p_5_g_1'] == 1), 3,
    np.where(np.bitwise_and(base_dados['CODIGO__pe_5_g_1'] == 3, base_dados['CODIGO__L__p_5_g_1'] == 2), 4,
     0))))))
             
base_dados['CODIGO__L__p_5_g_1_c1_21_2'] = np.where(base_dados['CODIGO__L__p_5_g_1_c1_21_1'] == 0, 0,
    np.where(base_dados['CODIGO__L__p_5_g_1_c1_21_1'] == 1, 1,
    np.where(base_dados['CODIGO__L__p_5_g_1_c1_21_1'] == 2, 2,
    np.where(base_dados['CODIGO__L__p_5_g_1_c1_21_1'] == 3, 3,
    np.where(base_dados['CODIGO__L__p_5_g_1_c1_21_1'] == 4, 4,
    0)))))
             
base_dados['CODIGO__L__p_5_g_1_c1_21'] = np.where(base_dados['CODIGO__L__p_5_g_1_c1_21_2'] == 0, 0,
    np.where(base_dados['CODIGO__L__p_5_g_1_c1_21_2'] == 1, 1,
    np.where(base_dados['CODIGO__L__p_5_g_1_c1_21_2'] == 2, 2,
    np.where(base_dados['CODIGO__L__p_5_g_1_c1_21_2'] == 3, 3,
    np.where(base_dados['CODIGO__L__p_5_g_1_c1_21_2'] == 4, 4,
     0)))))
         
         
         
         
base_dados['DDD1__S__p_10_g_1_c1_3_1'] = np.where(np.bitwise_and(base_dados['DDD1__L__pe_15_g_1'] == 0, base_dados['DDD1__S__p_10_g_1'] == 0), 0,
    np.where(np.bitwise_and(base_dados['DDD1__L__pe_15_g_1'] == 0, base_dados['DDD1__S__p_10_g_1'] == 1), 2,
    np.where(np.bitwise_and(base_dados['DDD1__L__pe_15_g_1'] == 1, base_dados['DDD1__S__p_10_g_1'] == 0), 1,
    np.where(np.bitwise_and(base_dados['DDD1__L__pe_15_g_1'] == 1, base_dados['DDD1__S__p_10_g_1'] == 1), 2,
    np.where(np.bitwise_and(base_dados['DDD1__L__pe_15_g_1'] == 2, base_dados['DDD1__S__p_10_g_1'] == 0), 2,
    np.where(np.bitwise_and(base_dados['DDD1__L__pe_15_g_1'] == 2, base_dados['DDD1__S__p_10_g_1'] == 1), 2,
     0))))))
             
base_dados['DDD1__S__p_10_g_1_c1_3_2'] = np.where(base_dados['DDD1__S__p_10_g_1_c1_3_1'] == 0, 0,
    np.where(base_dados['DDD1__S__p_10_g_1_c1_3_1'] == 1, 1,
    np.where(base_dados['DDD1__S__p_10_g_1_c1_3_1'] == 2, 2,
    0)))

base_dados['DDD1__S__p_10_g_1_c1_3'] = np.where(base_dados['DDD1__S__p_10_g_1_c1_3_2'] == 0, 0,
    np.where(base_dados['DDD1__S__p_10_g_1_c1_3_2'] == 1, 1,
    np.where(base_dados['DDD1__S__p_10_g_1_c1_3_2'] == 2, 2,
     0)))
         
         
         
         
base_dados['LIMITE__C__p_10_g_1_c1_7_1'] = np.where(np.bitwise_and(base_dados['LIMITE__pe_10_g_1'] == 0, base_dados['LIMITE__C__p_10_g_1'] == 0), 0,
    np.where(np.bitwise_and(base_dados['LIMITE__pe_10_g_1'] == 0, base_dados['LIMITE__C__p_10_g_1'] == 1), 2,
    np.where(np.bitwise_and(base_dados['LIMITE__pe_10_g_1'] == 1, base_dados['LIMITE__C__p_10_g_1'] == 0), 1,
    np.where(np.bitwise_and(base_dados['LIMITE__pe_10_g_1'] == 1, base_dados['LIMITE__C__p_10_g_1'] == 1), 3,
     0))))
         
base_dados['LIMITE__C__p_10_g_1_c1_7_2'] = np.where(base_dados['LIMITE__C__p_10_g_1_c1_7_1'] == 0, 1,
    np.where(base_dados['LIMITE__C__p_10_g_1_c1_7_1'] == 1, 0,
    np.where(base_dados['LIMITE__C__p_10_g_1_c1_7_1'] == 2, 1,
    np.where(base_dados['LIMITE__C__p_10_g_1_c1_7_1'] == 3, 3,
    0))))
         
base_dados['LIMITE__C__p_10_g_1_c1_7'] = np.where(base_dados['LIMITE__C__p_10_g_1_c1_7_2'] == 0, 0,
    np.where(base_dados['LIMITE__C__p_10_g_1_c1_7_2'] == 1, 1,
    np.where(base_dados['LIMITE__C__p_10_g_1_c1_7_2'] == 3, 2,
     0)))
         
         
         
         
         
base_dados['COD_EMPRESA__S__p_34_g_1_c1_13_1'] = np.where(np.bitwise_and(base_dados['COD_EMPRESA__p_6_g_1'] == 0, base_dados['COD_EMPRESA__S__p_34_g_1'] == 0), 0,
    np.where(np.bitwise_and(base_dados['COD_EMPRESA__p_6_g_1'] == 0, base_dados['COD_EMPRESA__S__p_34_g_1'] == 1), 2,
    np.where(np.bitwise_and(base_dados['COD_EMPRESA__p_6_g_1'] == 1, base_dados['COD_EMPRESA__S__p_34_g_1'] == 0), 0,
    np.where(np.bitwise_and(base_dados['COD_EMPRESA__p_6_g_1'] == 1, base_dados['COD_EMPRESA__S__p_34_g_1'] == 1), 3,
    np.where(np.bitwise_and(base_dados['COD_EMPRESA__p_6_g_1'] == 2, base_dados['COD_EMPRESA__S__p_34_g_1'] == 0), 1,
    np.where(np.bitwise_and(base_dados['COD_EMPRESA__p_6_g_1'] == 2, base_dados['COD_EMPRESA__S__p_34_g_1'] == 1), 4,
    np.where(np.bitwise_and(base_dados['COD_EMPRESA__p_6_g_1'] == 3, base_dados['COD_EMPRESA__S__p_34_g_1'] == 0), 1,
    np.where(np.bitwise_and(base_dados['COD_EMPRESA__p_6_g_1'] == 3, base_dados['COD_EMPRESA__S__p_34_g_1'] == 1), 5,
     0))))))))
         
base_dados['COD_EMPRESA__S__p_34_g_1_c1_13_2'] = np.where(base_dados['COD_EMPRESA__S__p_34_g_1_c1_13_1'] == 0, 0,
    np.where(base_dados['COD_EMPRESA__S__p_34_g_1_c1_13_1'] == 1, 3,
    np.where(base_dados['COD_EMPRESA__S__p_34_g_1_c1_13_1'] == 2, 1,
    np.where(base_dados['COD_EMPRESA__S__p_34_g_1_c1_13_1'] == 3, 2,
    np.where(base_dados['COD_EMPRESA__S__p_34_g_1_c1_13_1'] == 4, 3,
    np.where(base_dados['COD_EMPRESA__S__p_34_g_1_c1_13_1'] == 5, 5,
    0))))))
         
base_dados['COD_EMPRESA__S__p_34_g_1_c1_13'] = np.where(base_dados['COD_EMPRESA__S__p_34_g_1_c1_13_2'] == 0, 0,
    np.where(base_dados['COD_EMPRESA__S__p_34_g_1_c1_13_2'] == 1, 1,
    np.where(base_dados['COD_EMPRESA__S__p_34_g_1_c1_13_2'] == 2, 2,
    np.where(base_dados['COD_EMPRESA__S__p_34_g_1_c1_13_2'] == 3, 3,
    np.where(base_dados['COD_EMPRESA__S__p_34_g_1_c1_13_2'] == 5, 4,
     0)))))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Mantendo apenas as variáveis do modelo

# COMMAND ----------

varvar=[]
varvar= [chave,var_tmp,'DIAS_ATRASO__L__p_5_g_1_c1_23','DDD5_gh38','LIMITE__C__p_10_g_1_c1_7','COD_FILA_gh38','DDD4_gh38','COD_EMPRESA__S__p_34_g_1_c1_13','CODIGO__L__p_5_g_1_c1_21','UF_gh38','DDD1__S__p_10_g_1_c1_3']
base_teste_c0 = base_dados[varvar]
base_teste_c0


# COMMAND ----------

# DBTITLE 1,load pickle
modelo = pickle.load(open(os.path.join(pickle_path,'dmcard_v1.sav'),'rb'))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rodando Regressão Logística

# COMMAND ----------

base_treino_c0 = base_dados
var_fin_c0=list(base_teste_c0.columns)
#var_fin_c0.remove(target)
var_fin_c0.remove(var_tmp)
var_fin_c0.remove(chave)

# COMMAND ----------

# Datasets de treino e de teste
x_teste = base_teste_c0[var_fin_c0]
z_teste = base_teste_c0[chave]

probabilidades = modelo.predict_proba(x_teste)
data_prob = pd.DataFrame({'P_0': probabilidades[:, 0], 'P_1': probabilidades[:, 1]})

z_teste1 = z_teste.reset_index(drop=True)
x_teste1 = x_teste.reset_index(drop=True)
data_prob1 = data_prob.reset_index(drop=True)


x_teste2 = pd.concat([z_teste1,x_teste1, data_prob1], axis=1)

x_teste2


# COMMAND ----------

# MAGIC %md
# MAGIC ## Avaliando os dados do modelo

# COMMAND ----------

# MAGIC %md
# MAGIC # Modelo de Grupo Homogêneo

# COMMAND ----------


x_teste2['P_1_pe_15_g_1'] = np.where(x_teste2['P_1'] <= 0.075194141, 0.0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.075194141, x_teste2['P_1'] <= 0.136352034), 1.0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.136352034, x_teste2['P_1'] <= 0.248982659), 2.0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.248982659, x_teste2['P_1'] <= 0.441555879), 3.0,
    np.where(x_teste2['P_1'] > 0.441555879,4,0)))))

x_teste2['P_1_pe_20_g_1'] = np.where(x_teste2['P_1'] <= 0.053474711, 0.0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.053474711, x_teste2['P_1'] <= 0.160298586), 1.0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.160298586, x_teste2['P_1'] <= 0.213997612), 2.0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.213997612, x_teste2['P_1'] <= 0.293729986), 3.0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.293729986, x_teste2['P_1'] <= 0.441555879), 4.0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.441555879, x_teste2['P_1'] <= 0.534158296), 5.0,
    np.where(x_teste2['P_1'] > 0.534158296, 6.0,0)))))))

x_teste2['GH'] = np.where(np.bitwise_and(x_teste2['P_1_pe_15_g_1'] == 0, x_teste2['P_1_pe_20_g_1'] == 0), 0,
    np.where(np.bitwise_and(x_teste2['P_1_pe_15_g_1'] == 0, x_teste2['P_1_pe_20_g_1'] == 1), 1,
    np.where(np.bitwise_and(x_teste2['P_1_pe_15_g_1'] == 1, x_teste2['P_1_pe_20_g_1'] == 1), 2,
    np.where(np.bitwise_and(x_teste2['P_1_pe_15_g_1'] == 2, x_teste2['P_1_pe_20_g_1'] == 1), 2,
    np.where(np.bitwise_and(x_teste2['P_1_pe_15_g_1'] == 2, x_teste2['P_1_pe_20_g_1'] == 2), 3,
    np.where(np.bitwise_and(x_teste2['P_1_pe_15_g_1'] == 2, x_teste2['P_1_pe_20_g_1'] == 3), 3,
    np.where(np.bitwise_and(x_teste2['P_1_pe_15_g_1'] == 3, x_teste2['P_1_pe_20_g_1'] == 3), 4,
    np.where(np.bitwise_and(x_teste2['P_1_pe_15_g_1'] == 3, x_teste2['P_1_pe_20_g_1'] == 4), 4,
    np.where(np.bitwise_and(x_teste2['P_1_pe_15_g_1'] == 3, x_teste2['P_1_pe_20_g_1'] == 5), 5,
    np.where(np.bitwise_and(x_teste2['P_1_pe_15_g_1'] == 4, x_teste2['P_1_pe_20_g_1'] == 4), 5,
    np.where(np.bitwise_and(x_teste2['P_1_pe_15_g_1'] == 4, x_teste2['P_1_pe_20_g_1'] == 5), 5,
    np.where(np.bitwise_and(x_teste2['P_1_pe_15_g_1'] == 4, x_teste2['P_1_pe_20_g_1'] == 6), 5,
             0))))))))))))

del x_teste2['P_1_pe_15_g_1']
del x_teste2['P_1_pe_20_g_1']

x_teste2


# COMMAND ----------

try:
  dbutils.fs.rm(outputpath, True)
except:
  pass
dbutils.fs.mkdirs(outputpath)

x_teste2.to_csv(open(os.path.join(outputpath_dbfs, 'pre_output:'+arquivo_escolhido+'.csv'),'wb'))
os.path.join(outputpath_dbfs, 'pre_output:'+arquivo_escolhido+'.csv')