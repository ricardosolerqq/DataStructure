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
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.preprocessing import KBinsDiscretizer
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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

caminho_trusted = '/mnt/ml-prd/ml-data/propensaodeal/credsystem/trusted'
caminho_trusted_dbfs = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/credsystem/trusted'

pickle_path = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/credsystem/pickle_model/'

outputpath = 'mnt/ml-prd/ml-data/propensaodeal/credsystem/output/'
outputpath_dbfs = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/credsystem/output/'

# COMMAND ----------

## Parâmetros do Algoritmo

#Variável chave-primaria
chave = 'CPF'

#Caminho da base de dados
caminho_base = "/dbfs/mnt/ml-prd/ml-data/propensaodeal/credsystem/trusted/"
list_base = os.listdir(caminho_base)

#Nome da Base de Dados
N_Base = max(list_base)
dt_max = N_Base.split('_')[2]
dt_max = dt_max.split('.')[0]
nm_base = "trustedFile_credsystem"

#Separador
separador_ = ";"

#Decimal
decimal_ = "."

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importação da Base de Dados

# COMMAND ----------

base_dados = pd.read_csv(caminho_base+N_Base, sep=separador_, decimal=decimal_)
base_dados = base_dados[[chave,'CARTOES','CONTRATO','PRIMEIRO_VENCIMENTO']]

base_dados.fillna(-3)

#string
base_dados['CARTOES'] = base_dados['CARTOES'].replace(np.nan, '-3')

base_dados = base_dados.apply(pd.to_numeric, errors='ignore')

base_dados['CONTRATO'] = base_dados['CONTRATO'].astype(np.int64)

base_dados['PRIMEIRO_VENCIMENTO'] = pd.to_datetime(base_dados['PRIMEIRO_VENCIMENTO'])
base_dados['MOB_PRIMEIRO_VENCIMENTO'] = ((datetime.today()) - base_dados.PRIMEIRO_VENCIMENTO)/np.timedelta64(1, 'M')


del base_dados['PRIMEIRO_VENCIMENTO']


base_dados.drop_duplicates(keep=False, inplace=True)

print("shape da Base de Dados:",base_dados.shape)

base_dados.head()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Gerando variáveis Categóricas

# COMMAND ----------

base_dados['CARTOES_gh30'] = np.where(base_dados['CARTOES'] == '-3', 0,
np.where(base_dados['CARTOES'] == '3B', 1,
np.where(base_dados['CARTOES'] == 'ALEGRIA', 2,
np.where(base_dados['CARTOES'] == 'AMERICAN SHOES', 3,
np.where(base_dados['CARTOES'] == 'CAEDU', 4,
np.where(base_dados['CARTOES'] == 'CALCADOS ITAPUA', 5,
np.where(base_dados['CARTOES'] == 'CARTAO MAIS', 6,
np.where(base_dados['CARTOES'] == 'CASA E VIDEO', 7,
np.where(base_dados['CARTOES'] == 'CATTAN', 8,
np.where(base_dados['CARTOES'] == 'COSTAZUL ALIMENTOS', 9,
np.where(base_dados['CARTOES'] == 'DEL FIORI', 10,
np.where(base_dados['CARTOES'] == 'DEMANOS/DIMANOS', 11,
np.where(base_dados['CARTOES'] == 'DENY SPORTS', 12,
np.where(base_dados['CARTOES'] == 'DESKONCARD', 13,
np.where(base_dados['CARTOES'] == 'DI GASPI', 14,
np.where(base_dados['CARTOES'] == 'DI SANTINNI', 15,
np.where(base_dados['CARTOES'] == 'DIBS', 16,
np.where(base_dados['CARTOES'] == 'EDMAIS', 17,
np.where(base_dados['CARTOES'] == 'ELMO', 18,
np.where(base_dados['CARTOES'] == 'EMANUELLE PE', 19,
np.where(base_dados['CARTOES'] == 'EMMANUELLE SE', 20,
np.where(base_dados['CARTOES'] == 'EMPORIO ALEX', 21,
np.where(base_dados['CARTOES'] == 'ESKALA', 22,
np.where(base_dados['CARTOES'] == 'FREE CENTER CALCADOS', 23,
np.where(base_dados['CARTOES'] == 'GRIPPON SP', 24,
np.where(base_dados['CARTOES'] == 'HUMANITARIAN', 25,
np.where(base_dados['CARTOES'] == 'KALLAN CARD', 26,
np.where(base_dados['CARTOES'] == 'KATUXA CALCADOS', 27,
np.where(base_dados['CARTOES'] == 'KHELF', 28,
np.where(base_dados['CARTOES'] == 'LEADER', 29,
np.where(base_dados['CARTOES'] == 'LOJAO DO BRAS', 30,
np.where(base_dados['CARTOES'] == 'LOJAS MEL', 31,
np.where(base_dados['CARTOES'] == 'MAGAZINE DA ECONOMIA', 32,
np.where(base_dados['CARTOES'] == 'MAIS MEGA LOJA', 33,
np.where(base_dados['CARTOES'] == 'MUITO+', 34,
np.where(base_dados['CARTOES'] == 'NITEL', 35,
np.where(base_dados['CARTOES'] == 'NOVO ATACAREJO', 36,
np.where(base_dados['CARTOES'] == 'NOVO MUNDO', 37,
np.where(base_dados['CARTOES'] == 'OBJETIVA', 38,
np.where(base_dados['CARTOES'] == 'PAQUETA CALCADOS', 39,
np.where(base_dados['CARTOES'] == 'PASSARELA MODAS', 40,
np.where(base_dados['CARTOES'] == 'POLYELLE', 41,
np.where(base_dados['CARTOES'] == 'PONTAL CARD', 42,
np.where(base_dados['CARTOES'] == 'PONTO MIX', 43,
np.where(base_dados['CARTOES'] == 'PRINCESA SUPERMERCADO', 44,
np.where(base_dados['CARTOES'] == 'REDE LITORAL', 45,
np.where(base_dados['CARTOES'] == 'SAPATELLA', 46,
np.where(base_dados['CARTOES'] == 'SHOEBIZ', 47,
np.where(base_dados['CARTOES'] == 'SK SUPER E SUPER DIA-DIA', 48,
np.where(base_dados['CARTOES'] == 'SKG', 49,
np.where(base_dados['CARTOES'] == 'SONHO DOS PES', 50,
np.where(base_dados['CARTOES'] == 'SUP. EPA E BRASIL ATACAREJO', 51,
np.where(base_dados['CARTOES'] == 'SUPER DO POVO SUPERMERCADOS', 52,
np.where(base_dados['CARTOES'] == 'SUPERMERCADO FONSECA', 53,
np.where(base_dados['CARTOES'] == 'SUPERMERCADOS CAMPEAO', 54,
np.where(base_dados['CARTOES'] == 'SUPERPRIX', 55,
np.where(base_dados['CARTOES'] == 'TACO', 56,
np.where(base_dados['CARTOES'] == 'TENNISBAR', 57,
np.where(base_dados['CARTOES'] == 'TESOURA DE OURO', 58,
np.where(base_dados['CARTOES'] == 'TORRA TORRA', 59,
np.where(base_dados['CARTOES'] == 'ZINZANE', 60,
np.where(base_dados['CARTOES'] == 'ZUKEN VIP', 61,
0))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
base_dados['CARTOES_gh31'] = np.where(base_dados['CARTOES_gh30'] == 0, 0,
np.where(base_dados['CARTOES_gh30'] == 1, 1,
np.where(base_dados['CARTOES_gh30'] == 2, 1,
np.where(base_dados['CARTOES_gh30'] == 3, 3,
np.where(base_dados['CARTOES_gh30'] == 4, 4,
np.where(base_dados['CARTOES_gh30'] == 5, 4,
np.where(base_dados['CARTOES_gh30'] == 6, 6,
np.where(base_dados['CARTOES_gh30'] == 7, 7,
np.where(base_dados['CARTOES_gh30'] == 8, 8,
np.where(base_dados['CARTOES_gh30'] == 9, 9,
np.where(base_dados['CARTOES_gh30'] == 10, 10,
np.where(base_dados['CARTOES_gh30'] == 11, 11,
np.where(base_dados['CARTOES_gh30'] == 12, 12,
np.where(base_dados['CARTOES_gh30'] == 13, 13,
np.where(base_dados['CARTOES_gh30'] == 14, 14,
np.where(base_dados['CARTOES_gh30'] == 15, 15,
np.where(base_dados['CARTOES_gh30'] == 16, 16,
np.where(base_dados['CARTOES_gh30'] == 17, 17,
np.where(base_dados['CARTOES_gh30'] == 18, 18,
np.where(base_dados['CARTOES_gh30'] == 19, 19,
np.where(base_dados['CARTOES_gh30'] == 20, 20,
np.where(base_dados['CARTOES_gh30'] == 21, 21,
np.where(base_dados['CARTOES_gh30'] == 22, 22,
np.where(base_dados['CARTOES_gh30'] == 23, 22,
np.where(base_dados['CARTOES_gh30'] == 24, 24,
np.where(base_dados['CARTOES_gh30'] == 25, 25,
np.where(base_dados['CARTOES_gh30'] == 26, 26,
np.where(base_dados['CARTOES_gh30'] == 27, 27,
np.where(base_dados['CARTOES_gh30'] == 28, 28,
np.where(base_dados['CARTOES_gh30'] == 29, 29,
np.where(base_dados['CARTOES_gh30'] == 30, 29,
np.where(base_dados['CARTOES_gh30'] == 31, 31,
np.where(base_dados['CARTOES_gh30'] == 32, 32,
np.where(base_dados['CARTOES_gh30'] == 33, 33,
np.where(base_dados['CARTOES_gh30'] == 34, 34,
np.where(base_dados['CARTOES_gh30'] == 35, 35,
np.where(base_dados['CARTOES_gh30'] == 36, 36,
np.where(base_dados['CARTOES_gh30'] == 37, 37,
np.where(base_dados['CARTOES_gh30'] == 38, 38,
np.where(base_dados['CARTOES_gh30'] == 39, 39,
np.where(base_dados['CARTOES_gh30'] == 40, 40,
np.where(base_dados['CARTOES_gh30'] == 41, 41,
np.where(base_dados['CARTOES_gh30'] == 42, 42,
np.where(base_dados['CARTOES_gh30'] == 43, 43,
np.where(base_dados['CARTOES_gh30'] == 44, 44,
np.where(base_dados['CARTOES_gh30'] == 45, 45,
np.where(base_dados['CARTOES_gh30'] == 46, 46,
np.where(base_dados['CARTOES_gh30'] == 47, 47,
np.where(base_dados['CARTOES_gh30'] == 48, 48,
np.where(base_dados['CARTOES_gh30'] == 49, 49,
np.where(base_dados['CARTOES_gh30'] == 50, 50,
np.where(base_dados['CARTOES_gh30'] == 51, 51,
np.where(base_dados['CARTOES_gh30'] == 52, 51,
np.where(base_dados['CARTOES_gh30'] == 53, 51,
np.where(base_dados['CARTOES_gh30'] == 54, 54,
np.where(base_dados['CARTOES_gh30'] == 55, 55,
np.where(base_dados['CARTOES_gh30'] == 56, 56,
np.where(base_dados['CARTOES_gh30'] == 57, 57,
np.where(base_dados['CARTOES_gh30'] == 58, 58,
np.where(base_dados['CARTOES_gh30'] == 59, 59,
np.where(base_dados['CARTOES_gh30'] == 60, 60,
np.where(base_dados['CARTOES_gh30'] == 61, 61,
0))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
base_dados['CARTOES_gh32'] = np.where(base_dados['CARTOES_gh31'] == 0, 0,
np.where(base_dados['CARTOES_gh31'] == 1, 1,
np.where(base_dados['CARTOES_gh31'] == 3, 2,
np.where(base_dados['CARTOES_gh31'] == 4, 3,
np.where(base_dados['CARTOES_gh31'] == 6, 4,
np.where(base_dados['CARTOES_gh31'] == 7, 5,
np.where(base_dados['CARTOES_gh31'] == 8, 6,
np.where(base_dados['CARTOES_gh31'] == 9, 7,
np.where(base_dados['CARTOES_gh31'] == 10, 8,
np.where(base_dados['CARTOES_gh31'] == 11, 9,
np.where(base_dados['CARTOES_gh31'] == 12, 10,
np.where(base_dados['CARTOES_gh31'] == 13, 11,
np.where(base_dados['CARTOES_gh31'] == 14, 12,
np.where(base_dados['CARTOES_gh31'] == 15, 13,
np.where(base_dados['CARTOES_gh31'] == 16, 14,
np.where(base_dados['CARTOES_gh31'] == 17, 15,
np.where(base_dados['CARTOES_gh31'] == 18, 16,
np.where(base_dados['CARTOES_gh31'] == 19, 17,
np.where(base_dados['CARTOES_gh31'] == 20, 18,
np.where(base_dados['CARTOES_gh31'] == 21, 19,
np.where(base_dados['CARTOES_gh31'] == 22, 20,
np.where(base_dados['CARTOES_gh31'] == 24, 21,
np.where(base_dados['CARTOES_gh31'] == 25, 22,
np.where(base_dados['CARTOES_gh31'] == 26, 23,
np.where(base_dados['CARTOES_gh31'] == 27, 24,
np.where(base_dados['CARTOES_gh31'] == 28, 25,
np.where(base_dados['CARTOES_gh31'] == 29, 26,
np.where(base_dados['CARTOES_gh31'] == 31, 27,
np.where(base_dados['CARTOES_gh31'] == 32, 28,
np.where(base_dados['CARTOES_gh31'] == 33, 29,
np.where(base_dados['CARTOES_gh31'] == 34, 30,
np.where(base_dados['CARTOES_gh31'] == 35, 31,
np.where(base_dados['CARTOES_gh31'] == 36, 32,
np.where(base_dados['CARTOES_gh31'] == 37, 33,
np.where(base_dados['CARTOES_gh31'] == 38, 34,
np.where(base_dados['CARTOES_gh31'] == 39, 35,
np.where(base_dados['CARTOES_gh31'] == 40, 36,
np.where(base_dados['CARTOES_gh31'] == 41, 37,
np.where(base_dados['CARTOES_gh31'] == 42, 38,
np.where(base_dados['CARTOES_gh31'] == 43, 39,
np.where(base_dados['CARTOES_gh31'] == 44, 40,
np.where(base_dados['CARTOES_gh31'] == 45, 41,
np.where(base_dados['CARTOES_gh31'] == 46, 42,
np.where(base_dados['CARTOES_gh31'] == 47, 43,
np.where(base_dados['CARTOES_gh31'] == 48, 44,
np.where(base_dados['CARTOES_gh31'] == 49, 45,
np.where(base_dados['CARTOES_gh31'] == 50, 46,
np.where(base_dados['CARTOES_gh31'] == 51, 47,
np.where(base_dados['CARTOES_gh31'] == 54, 48,
np.where(base_dados['CARTOES_gh31'] == 55, 49,
np.where(base_dados['CARTOES_gh31'] == 56, 50,
np.where(base_dados['CARTOES_gh31'] == 57, 51,
np.where(base_dados['CARTOES_gh31'] == 58, 52,
np.where(base_dados['CARTOES_gh31'] == 59, 53,
np.where(base_dados['CARTOES_gh31'] == 60, 54,
np.where(base_dados['CARTOES_gh31'] == 61, 55,
0))))))))))))))))))))))))))))))))))))))))))))))))))))))))
base_dados['CARTOES_gh33'] = np.where(base_dados['CARTOES_gh32'] == 0, 0,
np.where(base_dados['CARTOES_gh32'] == 1, 1,
np.where(base_dados['CARTOES_gh32'] == 2, 2,
np.where(base_dados['CARTOES_gh32'] == 3, 3,
np.where(base_dados['CARTOES_gh32'] == 4, 4,
np.where(base_dados['CARTOES_gh32'] == 5, 5,
np.where(base_dados['CARTOES_gh32'] == 6, 6,
np.where(base_dados['CARTOES_gh32'] == 7, 7,
np.where(base_dados['CARTOES_gh32'] == 8, 8,
np.where(base_dados['CARTOES_gh32'] == 9, 9,
np.where(base_dados['CARTOES_gh32'] == 10, 10,
np.where(base_dados['CARTOES_gh32'] == 11, 11,
np.where(base_dados['CARTOES_gh32'] == 12, 12,
np.where(base_dados['CARTOES_gh32'] == 13, 13,
np.where(base_dados['CARTOES_gh32'] == 14, 14,
np.where(base_dados['CARTOES_gh32'] == 15, 15,
np.where(base_dados['CARTOES_gh32'] == 16, 16,
np.where(base_dados['CARTOES_gh32'] == 17, 17,
np.where(base_dados['CARTOES_gh32'] == 18, 18,
np.where(base_dados['CARTOES_gh32'] == 19, 19,
np.where(base_dados['CARTOES_gh32'] == 20, 20,
np.where(base_dados['CARTOES_gh32'] == 21, 21,
np.where(base_dados['CARTOES_gh32'] == 22, 22,
np.where(base_dados['CARTOES_gh32'] == 23, 23,
np.where(base_dados['CARTOES_gh32'] == 24, 24,
np.where(base_dados['CARTOES_gh32'] == 25, 25,
np.where(base_dados['CARTOES_gh32'] == 26, 26,
np.where(base_dados['CARTOES_gh32'] == 27, 27,
np.where(base_dados['CARTOES_gh32'] == 28, 28,
np.where(base_dados['CARTOES_gh32'] == 29, 29,
np.where(base_dados['CARTOES_gh32'] == 30, 30,
np.where(base_dados['CARTOES_gh32'] == 31, 31,
np.where(base_dados['CARTOES_gh32'] == 32, 32,
np.where(base_dados['CARTOES_gh32'] == 33, 33,
np.where(base_dados['CARTOES_gh32'] == 34, 34,
np.where(base_dados['CARTOES_gh32'] == 35, 35,
np.where(base_dados['CARTOES_gh32'] == 36, 36,
np.where(base_dados['CARTOES_gh32'] == 37, 37,
np.where(base_dados['CARTOES_gh32'] == 38, 38,
np.where(base_dados['CARTOES_gh32'] == 39, 39,
np.where(base_dados['CARTOES_gh32'] == 40, 40,
np.where(base_dados['CARTOES_gh32'] == 41, 41,
np.where(base_dados['CARTOES_gh32'] == 42, 42,
np.where(base_dados['CARTOES_gh32'] == 43, 43,
np.where(base_dados['CARTOES_gh32'] == 44, 44,
np.where(base_dados['CARTOES_gh32'] == 45, 45,
np.where(base_dados['CARTOES_gh32'] == 46, 46,
np.where(base_dados['CARTOES_gh32'] == 47, 47,
np.where(base_dados['CARTOES_gh32'] == 48, 48,
np.where(base_dados['CARTOES_gh32'] == 49, 49,
np.where(base_dados['CARTOES_gh32'] == 50, 50,
np.where(base_dados['CARTOES_gh32'] == 51, 51,
np.where(base_dados['CARTOES_gh32'] == 52, 52,
np.where(base_dados['CARTOES_gh32'] == 53, 53,
np.where(base_dados['CARTOES_gh32'] == 54, 54,
np.where(base_dados['CARTOES_gh32'] == 55, 55,
0))))))))))))))))))))))))))))))))))))))))))))))))))))))))
base_dados['CARTOES_gh34'] = np.where(base_dados['CARTOES_gh33'] == 0, 55,
np.where(base_dados['CARTOES_gh33'] == 1, 55,
np.where(base_dados['CARTOES_gh33'] == 2, 55,
np.where(base_dados['CARTOES_gh33'] == 3, 55,
np.where(base_dados['CARTOES_gh33'] == 4, 4,
np.where(base_dados['CARTOES_gh33'] == 5, 4,
np.where(base_dados['CARTOES_gh33'] == 6, 55,
np.where(base_dados['CARTOES_gh33'] == 7, 4,
np.where(base_dados['CARTOES_gh33'] == 8, 55,
np.where(base_dados['CARTOES_gh33'] == 9, 4,
np.where(base_dados['CARTOES_gh33'] == 10, 55,
np.where(base_dados['CARTOES_gh33'] == 11, 4,
np.where(base_dados['CARTOES_gh33'] == 12, 55,
np.where(base_dados['CARTOES_gh33'] == 13, 4,
np.where(base_dados['CARTOES_gh33'] == 14, 55,
np.where(base_dados['CARTOES_gh33'] == 15, 4,
np.where(base_dados['CARTOES_gh33'] == 16, 55,
np.where(base_dados['CARTOES_gh33'] == 17, 55,
np.where(base_dados['CARTOES_gh33'] == 18, 4,
np.where(base_dados['CARTOES_gh33'] == 19, 55,
np.where(base_dados['CARTOES_gh33'] == 20, 4,
np.where(base_dados['CARTOES_gh33'] == 21, 55,
np.where(base_dados['CARTOES_gh33'] == 22, 4,
np.where(base_dados['CARTOES_gh33'] == 23, 55,
np.where(base_dados['CARTOES_gh33'] == 24, 55,
np.where(base_dados['CARTOES_gh33'] == 25, 55,
np.where(base_dados['CARTOES_gh33'] == 26, 4,
np.where(base_dados['CARTOES_gh33'] == 27, 4,
np.where(base_dados['CARTOES_gh33'] == 28, 55,
np.where(base_dados['CARTOES_gh33'] == 29, 55,
np.where(base_dados['CARTOES_gh33'] == 30, 55,
np.where(base_dados['CARTOES_gh33'] == 31, 55,
np.where(base_dados['CARTOES_gh33'] == 32, 55,
np.where(base_dados['CARTOES_gh33'] == 33, 55,
np.where(base_dados['CARTOES_gh33'] == 34, 4,
np.where(base_dados['CARTOES_gh33'] == 35, 55,
np.where(base_dados['CARTOES_gh33'] == 36, 4,
np.where(base_dados['CARTOES_gh33'] == 37, 55,
np.where(base_dados['CARTOES_gh33'] == 38, 55,
np.where(base_dados['CARTOES_gh33'] == 39, 4,
np.where(base_dados['CARTOES_gh33'] == 40, 55,
np.where(base_dados['CARTOES_gh33'] == 41, 55,
np.where(base_dados['CARTOES_gh33'] == 42, 4,
np.where(base_dados['CARTOES_gh33'] == 43, 55,
np.where(base_dados['CARTOES_gh33'] == 44, 4,
np.where(base_dados['CARTOES_gh33'] == 45, 4,
np.where(base_dados['CARTOES_gh33'] == 46, 4,
np.where(base_dados['CARTOES_gh33'] == 47, 4,
np.where(base_dados['CARTOES_gh33'] == 48, 55,
np.where(base_dados['CARTOES_gh33'] == 49, 4,
np.where(base_dados['CARTOES_gh33'] == 50, 4,
np.where(base_dados['CARTOES_gh33'] == 51, 55,
np.where(base_dados['CARTOES_gh33'] == 52, 4,
np.where(base_dados['CARTOES_gh33'] == 53, 55,
np.where(base_dados['CARTOES_gh33'] == 54, 4,
np.where(base_dados['CARTOES_gh33'] == 55, 55,
0))))))))))))))))))))))))))))))))))))))))))))))))))))))))
base_dados['CARTOES_gh35'] = np.where(base_dados['CARTOES_gh34'] == 4, 0,
np.where(base_dados['CARTOES_gh34'] == 55, 1,
0))
base_dados['CARTOES_gh36'] = np.where(base_dados['CARTOES_gh35'] == 0, 1,
np.where(base_dados['CARTOES_gh35'] == 1, 0,
0))
base_dados['CARTOES_gh37'] = np.where(base_dados['CARTOES_gh36'] == 0, 0,
np.where(base_dados['CARTOES_gh36'] == 1, 1,
0))
base_dados['CARTOES_gh38'] = np.where(base_dados['CARTOES_gh37'] == 0, 0,
np.where(base_dados['CARTOES_gh37'] == 1, 1,
0))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gerando variáveis numéricas contínuas

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parte 1 de 2

# COMMAND ----------

base_dados['CONTRATO__L'] = np.log(base_dados['CONTRATO'])
np.where(base_dados['CONTRATO__L'] == 0, -1, base_dados['CONTRATO__L'])
base_dados['CONTRATO__L'] = base_dados['CONTRATO__L'].fillna(-2)
base_dados['CONTRATO__L__p_7'] = np.where(base_dados['CONTRATO__L'] <= 17.873761443421273, 0.0,
np.where(np.bitwise_and(base_dados['CONTRATO__L'] > 17.873761443421273, base_dados['CONTRATO__L'] <= 18.231208413206332), 1.0,
np.where(np.bitwise_and(base_dados['CONTRATO__L'] > 18.231208413206332, base_dados['CONTRATO__L'] <= 18.479405840459428), 2.0,
np.where(np.bitwise_and(base_dados['CONTRATO__L'] > 18.479405840459428, base_dados['CONTRATO__L'] <= 18.696574159940397), 3.0,
np.where(np.bitwise_and(base_dados['CONTRATO__L'] > 18.696574159940397, base_dados['CONTRATO__L'] <= 18.82754221003214), 4.0,
np.where(np.bitwise_and(base_dados['CONTRATO__L'] > 18.82754221003214, base_dados['CONTRATO__L'] <= 18.930022429226053), 5.0,
np.where(base_dados['CONTRATO__L'] > 18.930022429226053, 6.0,
 0)))))))
base_dados['CONTRATO__L__p_7_g_1_1'] = np.where(base_dados['CONTRATO__L__p_7'] == 0.0, 2,
np.where(base_dados['CONTRATO__L__p_7'] == 1.0, 4,
np.where(base_dados['CONTRATO__L__p_7'] == 2.0, 2,
np.where(base_dados['CONTRATO__L__p_7'] == 3.0, 3,
np.where(base_dados['CONTRATO__L__p_7'] == 4.0, 0,
np.where(base_dados['CONTRATO__L__p_7'] == 5.0, 3,
np.where(base_dados['CONTRATO__L__p_7'] == 6.0, 1,
 0)))))))
base_dados['CONTRATO__L__p_7_g_1_2'] = np.where(base_dados['CONTRATO__L__p_7_g_1_1'] == 0, 1,
np.where(base_dados['CONTRATO__L__p_7_g_1_1'] == 1, 4,
np.where(base_dados['CONTRATO__L__p_7_g_1_1'] == 2, 1,
np.where(base_dados['CONTRATO__L__p_7_g_1_1'] == 3, 3,
np.where(base_dados['CONTRATO__L__p_7_g_1_1'] == 4, 0,
 0)))))
base_dados['CONTRATO__L__p_7_g_1'] = np.where(base_dados['CONTRATO__L__p_7_g_1_2'] == 0, 0,
np.where(base_dados['CONTRATO__L__p_7_g_1_2'] == 1, 1,
np.where(base_dados['CONTRATO__L__p_7_g_1_2'] == 3, 2,
np.where(base_dados['CONTRATO__L__p_7_g_1_2'] == 4, 3,
 0))))
         
         
         
         
         
         
         
base_dados['CONTRATO__L'] = np.log(base_dados['CONTRATO'])
np.where(base_dados['CONTRATO__L'] == 0, -1, base_dados['CONTRATO__L'])
base_dados['CONTRATO__L'] = base_dados['CONTRATO__L'].fillna(-2)
base_dados['CONTRATO__L__p_5'] = np.where(base_dados['CONTRATO__L'] <= 18.038637031023892, 0.0,
np.where(np.bitwise_and(base_dados['CONTRATO__L'] > 18.038637031023892, base_dados['CONTRATO__L'] <= 18.439481379226084), 1.0,
np.where(np.bitwise_and(base_dados['CONTRATO__L'] > 18.439481379226084, base_dados['CONTRATO__L'] <= 18.723270120841427), 2.0,
np.where(np.bitwise_and(base_dados['CONTRATO__L'] > 18.723270120841427, base_dados['CONTRATO__L'] <= 18.89387661008392), 3.0,
np.where(base_dados['CONTRATO__L'] > 18.89387661008392, 4.0,
 0)))))
base_dados['CONTRATO__L__p_5_g_1_1'] = np.where(base_dados['CONTRATO__L__p_5'] == 0.0, 2,
np.where(base_dados['CONTRATO__L__p_5'] == 1.0, 2,
np.where(base_dados['CONTRATO__L__p_5'] == 2.0, 3,
np.where(base_dados['CONTRATO__L__p_5'] == 3.0, 0,
np.where(base_dados['CONTRATO__L__p_5'] == 4.0, 1,
 0)))))
base_dados['CONTRATO__L__p_5_g_1_2'] = np.where(base_dados['CONTRATO__L__p_5_g_1_1'] == 0, 1,
np.where(base_dados['CONTRATO__L__p_5_g_1_1'] == 1, 3,
np.where(base_dados['CONTRATO__L__p_5_g_1_1'] == 2, 0,
np.where(base_dados['CONTRATO__L__p_5_g_1_1'] == 3, 1,
 0))))
base_dados['CONTRATO__L__p_5_g_1'] = np.where(base_dados['CONTRATO__L__p_5_g_1_2'] == 0, 0,
np.where(base_dados['CONTRATO__L__p_5_g_1_2'] == 1, 1,
np.where(base_dados['CONTRATO__L__p_5_g_1_2'] == 3, 2,
 0)))
         
         
         
         
         
         
         
base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13'] = np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO'] <= 12.74832796512515, 0.0,
np.where(np.bitwise_and(base_dados['MOB_PRIMEIRO_VENCIMENTO'] > 12.74832796512515, base_dados['MOB_PRIMEIRO_VENCIMENTO'] <= 25.49602298966364), 1.0,
np.where(np.bitwise_and(base_dados['MOB_PRIMEIRO_VENCIMENTO'] > 25.49602298966364, base_dados['MOB_PRIMEIRO_VENCIMENTO'] <= 38.04658870969896), 2.0,
np.where(np.bitwise_and(base_dados['MOB_PRIMEIRO_VENCIMENTO'] > 38.04658870969896, base_dados['MOB_PRIMEIRO_VENCIMENTO'] <= 51.08997769099221), 3.0,
np.where(np.bitwise_and(base_dados['MOB_PRIMEIRO_VENCIMENTO'] > 51.08997769099221, base_dados['MOB_PRIMEIRO_VENCIMENTO'] <= 63.87052759961456), 4.0,
np.where(np.bitwise_and(base_dados['MOB_PRIMEIRO_VENCIMENTO'] > 63.87052759961456, base_dados['MOB_PRIMEIRO_VENCIMENTO'] <= 76.58536774006919), 5.0,
np.where(np.bitwise_and(base_dados['MOB_PRIMEIRO_VENCIMENTO'] > 76.58536774006919, base_dados['MOB_PRIMEIRO_VENCIMENTO'] <= 89.36591764869155), 6.0,
np.where(np.bitwise_and(base_dados['MOB_PRIMEIRO_VENCIMENTO'] > 89.36591764869155, base_dados['MOB_PRIMEIRO_VENCIMENTO'] <= 102.11361267323004), 7.0,
np.where(np.bitwise_and(base_dados['MOB_PRIMEIRO_VENCIMENTO'] > 102.11361267323004, base_dados['MOB_PRIMEIRO_VENCIMENTO'] <= 114.8941625818524), 8.0,
np.where(np.bitwise_and(base_dados['MOB_PRIMEIRO_VENCIMENTO'] > 114.8941625818524, base_dados['MOB_PRIMEIRO_VENCIMENTO'] <= 127.60900272230703), 9.0,
np.where(np.bitwise_and(base_dados['MOB_PRIMEIRO_VENCIMENTO'] > 127.60900272230703, base_dados['MOB_PRIMEIRO_VENCIMENTO'] <= 140.52097216726483), 10.0,
np.where(np.bitwise_and(base_dados['MOB_PRIMEIRO_VENCIMENTO'] > 140.52097216726483, base_dados['MOB_PRIMEIRO_VENCIMENTO'] <= 153.23581230771944), 11.0,
np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO'] > 153.23581230771944, 12.0,
 -2)))))))))))))
base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13_g_1_1'] = np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13'] == -2.0, 4,
np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13'] == 0.0, 1,
np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13'] == 1.0, 0,
np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13'] == 2.0, 2,
np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13'] == 3.0, 2,
np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13'] == 4.0, 2,
np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13'] == 5.0, 3,
np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13'] == 6.0, 3,
np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13'] == 7.0, 4,
np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13'] == 8.0, 4,
np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13'] == 9.0, 4,
np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13'] == 10.0, 4,
np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13'] == 11.0, 2,
np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13'] == 12.0, 4,
 0))))))))))))))
base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13_g_1_2'] = np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13_g_1_1'] == 0, 3,
np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13_g_1_1'] == 1, 4,
np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13_g_1_1'] == 2, 2,
np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13_g_1_1'] == 3, 1,
np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13_g_1_1'] == 4, 0,
 0)))))
base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13_g_1'] = np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13_g_1_2'] == 0, 0,
np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13_g_1_2'] == 1, 1,
np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13_g_1_2'] == 2, 2,
np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13_g_1_2'] == 3, 3,
np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13_g_1_2'] == 4, 4,
 0)))))
         
         
         
         
         
         
         
base_dados['MOB_PRIMEIRO_VENCIMENTO__L'] = np.log(base_dados['MOB_PRIMEIRO_VENCIMENTO'])
np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__L'] == 0, -1, base_dados['MOB_PRIMEIRO_VENCIMENTO__L'])
base_dados['MOB_PRIMEIRO_VENCIMENTO__L'] = base_dados['MOB_PRIMEIRO_VENCIMENTO__L'].fillna(-2)
base_dados['MOB_PRIMEIRO_VENCIMENTO__L__pk_8'] = np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__L'] <= 0.8472963077606552, 0.0,
np.where(np.bitwise_and(base_dados['MOB_PRIMEIRO_VENCIMENTO__L'] > 0.8472963077606552, base_dados['MOB_PRIMEIRO_VENCIMENTO__L'] <= 1.6596393458191854), 1.0,
np.where(np.bitwise_and(base_dados['MOB_PRIMEIRO_VENCIMENTO__L'] > 1.6596393458191854, base_dados['MOB_PRIMEIRO_VENCIMENTO__L'] <= 2.317748753619186), 2.0,
np.where(np.bitwise_and(base_dados['MOB_PRIMEIRO_VENCIMENTO__L'] > 2.317748753619186, base_dados['MOB_PRIMEIRO_VENCIMENTO__L'] <= 2.9141003781704566), 3.0,
np.where(np.bitwise_and(base_dados['MOB_PRIMEIRO_VENCIMENTO__L'] > 2.9141003781704566, base_dados['MOB_PRIMEIRO_VENCIMENTO__L'] <= 3.5245862515010673), 4.0,
np.where(np.bitwise_and(base_dados['MOB_PRIMEIRO_VENCIMENTO__L'] > 3.5245862515010673, base_dados['MOB_PRIMEIRO_VENCIMENTO__L'] <= 4.067600384322291), 5.0,
np.where(np.bitwise_and(base_dados['MOB_PRIMEIRO_VENCIMENTO__L'] > 4.067600384322291, base_dados['MOB_PRIMEIRO_VENCIMENTO__L'] <= 4.6180098098127615), 6.0,
np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__L'] > 4.6180098098127615, 7.0,
 0))))))))
base_dados['MOB_PRIMEIRO_VENCIMENTO__L__pk_8_g_1_1'] = np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__L__pk_8'] == 0.0, 4,
np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__L__pk_8'] == 1.0, 1,
np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__L__pk_8'] == 2.0, 3,
np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__L__pk_8'] == 3.0, 0,
np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__L__pk_8'] == 4.0, 4,
np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__L__pk_8'] == 5.0, 2,
np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__L__pk_8'] == 6.0, 4,
np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__L__pk_8'] == 7.0, 4,
 0))))))))
base_dados['MOB_PRIMEIRO_VENCIMENTO__L__pk_8_g_1_2'] = np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__L__pk_8_g_1_1'] == 0, 2,
np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__L__pk_8_g_1_1'] == 1, 3,
np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__L__pk_8_g_1_1'] == 2, 1,
np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__L__pk_8_g_1_1'] == 3, 3,
np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__L__pk_8_g_1_1'] == 4, 0,
 0)))))
base_dados['MOB_PRIMEIRO_VENCIMENTO__L__pk_8_g_1'] = np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__L__pk_8_g_1_2'] == 0, 0,
np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__L__pk_8_g_1_2'] == 1, 1,
np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__L__pk_8_g_1_2'] == 2, 2,
np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__L__pk_8_g_1_2'] == 3, 3,
 0))))


# COMMAND ----------

# MAGIC %md
# MAGIC ### Parte 2 de 2

# COMMAND ----------

base_dados['CONTRATO__L__p_5_g_1_c1_5_1'] = np.where(np.bitwise_and(base_dados['CONTRATO__L__p_7_g_1'] == 0, base_dados['CONTRATO__L__p_5_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['CONTRATO__L__p_7_g_1'] == 0, base_dados['CONTRATO__L__p_5_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['CONTRATO__L__p_7_g_1'] == 0, base_dados['CONTRATO__L__p_5_g_1'] == 2), 1,
np.where(np.bitwise_and(base_dados['CONTRATO__L__p_7_g_1'] == 0, base_dados['CONTRATO__L__p_5_g_1'] == 3), 2,
np.where(np.bitwise_and(base_dados['CONTRATO__L__p_7_g_1'] == 1, base_dados['CONTRATO__L__p_5_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['CONTRATO__L__p_7_g_1'] == 1, base_dados['CONTRATO__L__p_5_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['CONTRATO__L__p_7_g_1'] == 1, base_dados['CONTRATO__L__p_5_g_1'] == 2), 2,
np.where(np.bitwise_and(base_dados['CONTRATO__L__p_7_g_1'] == 1, base_dados['CONTRATO__L__p_5_g_1'] == 3), 2,
np.where(np.bitwise_and(base_dados['CONTRATO__L__p_7_g_1'] == 2, base_dados['CONTRATO__L__p_5_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['CONTRATO__L__p_7_g_1'] == 2, base_dados['CONTRATO__L__p_5_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['CONTRATO__L__p_7_g_1'] == 2, base_dados['CONTRATO__L__p_5_g_1'] == 2), 3,
np.where(np.bitwise_and(base_dados['CONTRATO__L__p_7_g_1'] == 2, base_dados['CONTRATO__L__p_5_g_1'] == 3), 3,
np.where(np.bitwise_and(base_dados['CONTRATO__L__p_7_g_1'] == 3, base_dados['CONTRATO__L__p_5_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['CONTRATO__L__p_7_g_1'] == 3, base_dados['CONTRATO__L__p_5_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['CONTRATO__L__p_7_g_1'] == 3, base_dados['CONTRATO__L__p_5_g_1'] == 2), 3,
np.where(np.bitwise_and(base_dados['CONTRATO__L__p_7_g_1'] == 3, base_dados['CONTRATO__L__p_5_g_1'] == 3), 3,
 0))))))))))))))))
base_dados['CONTRATO__L__p_5_g_1_c1_5_2'] = np.where(base_dados['CONTRATO__L__p_5_g_1_c1_5_1'] == 0, 0,
np.where(base_dados['CONTRATO__L__p_5_g_1_c1_5_1'] == 1, 1,
np.where(base_dados['CONTRATO__L__p_5_g_1_c1_5_1'] == 2, 2,
np.where(base_dados['CONTRATO__L__p_5_g_1_c1_5_1'] == 3, 3,
0))))
base_dados['CONTRATO__L__p_5_g_1_c1_5'] = np.where(base_dados['CONTRATO__L__p_5_g_1_c1_5_2'] == 0, 0,
np.where(base_dados['CONTRATO__L__p_5_g_1_c1_5_2'] == 1, 1,
np.where(base_dados['CONTRATO__L__p_5_g_1_c1_5_2'] == 2, 2,
np.where(base_dados['CONTRATO__L__p_5_g_1_c1_5_2'] == 3, 3,
 0))))
         
         
         
         
         
         
         
base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13_g_1_c1_17_1'] = np.where(np.bitwise_and(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13_g_1'] == 0, base_dados['MOB_PRIMEIRO_VENCIMENTO__L__pk_8_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13_g_1'] == 1, base_dados['MOB_PRIMEIRO_VENCIMENTO__L__pk_8_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13_g_1'] == 2, base_dados['MOB_PRIMEIRO_VENCIMENTO__L__pk_8_g_1'] == 0), 2,
np.where(np.bitwise_and(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13_g_1'] == 3, base_dados['MOB_PRIMEIRO_VENCIMENTO__L__pk_8_g_1'] == 0), 4,
np.where(np.bitwise_and(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13_g_1'] == 4, base_dados['MOB_PRIMEIRO_VENCIMENTO__L__pk_8_g_1'] == 0), 5,
np.where(np.bitwise_and(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13_g_1'] == 0, base_dados['MOB_PRIMEIRO_VENCIMENTO__L__pk_8_g_1'] == 1), 0,
np.where(np.bitwise_and(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13_g_1'] == 1, base_dados['MOB_PRIMEIRO_VENCIMENTO__L__pk_8_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13_g_1'] == 2, base_dados['MOB_PRIMEIRO_VENCIMENTO__L__pk_8_g_1'] == 1), 3,
np.where(np.bitwise_and(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13_g_1'] == 3, base_dados['MOB_PRIMEIRO_VENCIMENTO__L__pk_8_g_1'] == 1), 3,
np.where(np.bitwise_and(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13_g_1'] == 4, base_dados['MOB_PRIMEIRO_VENCIMENTO__L__pk_8_g_1'] == 1), 5,
np.where(np.bitwise_and(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13_g_1'] == 0, base_dados['MOB_PRIMEIRO_VENCIMENTO__L__pk_8_g_1'] == 2), 0,
np.where(np.bitwise_and(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13_g_1'] == 1, base_dados['MOB_PRIMEIRO_VENCIMENTO__L__pk_8_g_1'] == 2), 1,
np.where(np.bitwise_and(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13_g_1'] == 2, base_dados['MOB_PRIMEIRO_VENCIMENTO__L__pk_8_g_1'] == 2), 3,
np.where(np.bitwise_and(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13_g_1'] == 3, base_dados['MOB_PRIMEIRO_VENCIMENTO__L__pk_8_g_1'] == 2), 4,
np.where(np.bitwise_and(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13_g_1'] == 4, base_dados['MOB_PRIMEIRO_VENCIMENTO__L__pk_8_g_1'] == 2), 5,
np.where(np.bitwise_and(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13_g_1'] == 0, base_dados['MOB_PRIMEIRO_VENCIMENTO__L__pk_8_g_1'] == 3), 1,
np.where(np.bitwise_and(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13_g_1'] == 1, base_dados['MOB_PRIMEIRO_VENCIMENTO__L__pk_8_g_1'] == 3), 2,
np.where(np.bitwise_and(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13_g_1'] == 2, base_dados['MOB_PRIMEIRO_VENCIMENTO__L__pk_8_g_1'] == 3), 4,
np.where(np.bitwise_and(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13_g_1'] == 3, base_dados['MOB_PRIMEIRO_VENCIMENTO__L__pk_8_g_1'] == 3), 4,
np.where(np.bitwise_and(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13_g_1'] == 4, base_dados['MOB_PRIMEIRO_VENCIMENTO__L__pk_8_g_1'] == 3), 5,  
0))))))))))))))))))))
base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13_g_1_c1_17_2'] = np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13_g_1_c1_17_1'] == 0, 0,
np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13_g_1_c1_17_1'] == 1, 1,
np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13_g_1_c1_17_1'] == 2, 2,
np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13_g_1_c1_17_1'] == 3, 3,
np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13_g_1_c1_17_1'] == 4, 4,
np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13_g_1_c1_17_1'] == 5, 5,
0))))))
base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13_g_1_c1_17'] = np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13_g_1_c1_17_2'] == 0, 0,
np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13_g_1_c1_17_2'] == 1, 1,
np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13_g_1_c1_17_2'] == 2, 2,
np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13_g_1_c1_17_2'] == 3, 3,
np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13_g_1_c1_17_2'] == 4, 4,
np.where(base_dados['MOB_PRIMEIRO_VENCIMENTO__pe_13_g_1_c1_17_2'] == 5, 5,
 0))))))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Mantendo apenas as variáveis do modelo

# COMMAND ----------

import pickle
modelo=pickle.load(open(pickle_path + 'model_fit_credsystem.sav', 'rb'))

base_teste_c0 = base_dados[[chave,'CARTOES_gh38','CONTRATO__L__p_5_g_1_c1_5','MOB_PRIMEIRO_VENCIMENTO__pe_13_g_1_c1_17']]

var_fin_c0=['CARTOES_gh38','CONTRATO__L__p_5_g_1_c1_5','MOB_PRIMEIRO_VENCIMENTO__pe_13_g_1_c1_17']

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

    
x_teste2['P_1_p_17_g_1'] = np.where(x_teste2['P_1'] <= 0.110856942, 0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.110856942, x_teste2['P_1'] <= 0.190063757), 1,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.190063757, x_teste2['P_1'] <= 0.230756082), 2,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.230756082, x_teste2['P_1'] <= 0.327856523), 3,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.327856523, x_teste2['P_1'] <= 0.502214347), 4,5)))))

x_teste2['P_1_p_40_g_1'] = np.where(x_teste2['P_1'] <= 0.168560128, 0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.168560128, x_teste2['P_1'] <= 0.295438847), 1,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.295438847, x_teste2['P_1'] <= 0.46570069), 2,3)))

x_teste2['GH'] = np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 0, x_teste2['P_1_p_40_g_1'] == 0), 0,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 0, x_teste2['P_1_p_40_g_1'] == 1), 0,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 0, x_teste2['P_1_p_40_g_1'] == 2), 1,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 0, x_teste2['P_1_p_40_g_1'] == 3), 1,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 1, x_teste2['P_1_p_40_g_1'] == 0), 1,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 1, x_teste2['P_1_p_40_g_1'] == 1), 2,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 1, x_teste2['P_1_p_40_g_1'] == 2), 2,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 1, x_teste2['P_1_p_40_g_1'] == 3), 3,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 2, x_teste2['P_1_p_40_g_1'] == 0), 2,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 2, x_teste2['P_1_p_40_g_1'] == 1), 2,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 2, x_teste2['P_1_p_40_g_1'] == 2), 3,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 2, x_teste2['P_1_p_40_g_1'] == 3), 3,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 3, x_teste2['P_1_p_40_g_1'] == 0), 2,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 3, x_teste2['P_1_p_40_g_1'] == 1), 3,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 3, x_teste2['P_1_p_40_g_1'] == 2), 4,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 3, x_teste2['P_1_p_40_g_1'] == 3), 4,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 4, x_teste2['P_1_p_40_g_1'] == 0), 3,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 4, x_teste2['P_1_p_40_g_1'] == 1), 3,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 4, x_teste2['P_1_p_40_g_1'] == 2), 4,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 4, x_teste2['P_1_p_40_g_1'] == 3), 5,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 5, x_teste2['P_1_p_40_g_1'] == 0), 3,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 5, x_teste2['P_1_p_40_g_1'] == 1), 4,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 5, x_teste2['P_1_p_40_g_1'] == 2), 5,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 5, x_teste2['P_1_p_40_g_1'] == 3), 5,0))))))))))))))))))))))))

x_teste2

# COMMAND ----------

x_teste2.groupby(['GH'])['P_1'].count()

# COMMAND ----------

try:
  dbutils.fs.rm(outputpath, True)
except:
  pass
dbutils.fs.mkdirs(outputpath)

x_teste2.to_csv(open(os.path.join(outputpath_dbfs, 'pre_output:' + nm_base.replace('-','') + '_' + dt_max + '.csv'),'wb'))
os.path.join(outputpath_dbfs, 'pre_output:' + nm_base.replace('-','') + '_' + dt_max + '.csv')