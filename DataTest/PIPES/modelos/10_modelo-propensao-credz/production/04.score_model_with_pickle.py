# Databricks notebook source
# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

try:
  dbutils.widgets.remove('MODELO_ESCOLHIDO')
except:
  pass
try:
  dbutils.widgets.remove('ARQUIVO_ESCOLHIDO')
except:
  pass

# COMMAND ----------

picklepath = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/credz/pickle_model'
trustedpath = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/credz/trusted'
outputpath = '/mnt/ml-prd/ml-data/propensaodeal/credz/output'
outputpath_dbfs = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/credz/output'

# COMMAND ----------

trustedpath_listdir = []
trustedpath_listdir_temp = os.listdir(trustedpath)

for item in trustedpath_listdir_temp:
  if '-' in item:
    trustedpath_listdir.append(item)

# COMMAND ----------

dbutils.widgets.combobox('MODELO_ESCOLHIDO', max(os.listdir(picklepath)), os.listdir(picklepath))
dbutils.widgets.combobox('ARQUIVO_ESCOLHIDO', max(trustedpath_listdir), trustedpath_listdir)

data_arquivo = dbutils.widgets.get('ARQUIVO_ESCOLHIDO')
data_modelo = dbutils.widgets.get('MODELO_ESCOLHIDO')

for file in os.listdir(os.path.join(trustedpath, data_arquivo)):
  pass
file = os.path.join(trustedpath, data_arquivo, file)

for model in os.listdir(os.path.join(picklepath, data_modelo)):
  pass
model = os.path.join(picklepath, data_modelo, model)

print (file, '\n',model)

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
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inserindo hiperparâmetros do Algoritmo

# COMMAND ----------

## Parâmetros do Algoritmo

#Variável chave-primaria
chave = 'DOCUMENTO_PESSOA'


# COMMAND ----------

# MAGIC %md
# MAGIC ## Importação da Base de Dados

# COMMAND ----------

#carregar o arquivo em formato tabela
base_dados = pd.read_parquet(file)
base_dados = base_dados[['TIPO_TELEFONE_3','ID_DIVIDA','DETALHES_CONTRATOS_STATUS_ACORDO','DETALHES_CONTRATOS_CLASSE','DOCUMENTO_PESSOA','DETALHES_CLIENTES_SCORE_CARGA','DETALHES_DIVIDAS_VALOR_JUROS','DETALHES_CLIENTES_VALOR_FATURA','DETALHES_CONTRATOS_BLOQUEIO2','TIPO_TELEFONE_4']]

#string
base_dados['DETALHES_CONTRATOS_BLOQUEIO2'] = base_dados['DETALHES_CONTRATOS_BLOQUEIO2'].replace(np.nan, '-3')
base_dados['DETALHES_CONTRATOS_CLASSE'] = base_dados['DETALHES_CONTRATOS_CLASSE'].replace(np.nan, '-3')
base_dados['DETALHES_CONTRATOS_STATUS_ACORDO'] = base_dados['DETALHES_CONTRATOS_STATUS_ACORDO'].replace(np.nan, '-3')
base_dados['TIPO_TELEFONE_3'] = base_dados['TIPO_TELEFONE_3'].replace(np.nan, '-3')
base_dados['TIPO_TELEFONE_4'] = base_dados['TIPO_TELEFONE_4'].replace(np.nan, '-3')


#numericas
base_dados['DOCUMENTO_PESSOA'] = base_dados['DOCUMENTO_PESSOA'].replace(np.nan, '-3')
base_dados['ID_DIVIDA'] = base_dados['ID_DIVIDA'].replace(np.nan, '-3')
base_dados['DETALHES_CLIENTES_SCORE_CARGA'] = base_dados['DETALHES_CLIENTES_SCORE_CARGA'].replace(np.nan, '-3')
base_dados['DETALHES_CLIENTES_VALOR_FATURA'] = base_dados['DETALHES_CLIENTES_VALOR_FATURA'].replace(np.nan, '-3')
base_dados['DETALHES_DIVIDAS_VALOR_JUROS'] = base_dados['DETALHES_DIVIDAS_VALOR_JUROS'].replace(np.nan, '-3')


base_dados = base_dados.apply(pd.to_numeric, errors='ignore')

base_dados['DOCUMENTO_PESSOA'] = base_dados['DOCUMENTO_PESSOA'].astype(np.int64)
base_dados['ID_DIVIDA'] = base_dados['ID_DIVIDA'].astype(np.int64)
base_dados['DETALHES_CLIENTES_SCORE_CARGA'] = base_dados['DETALHES_CLIENTES_SCORE_CARGA'].astype(np.int64)
base_dados['DETALHES_CLIENTES_VALOR_FATURA'] = base_dados['DETALHES_CLIENTES_VALOR_FATURA'].astype(float)
base_dados['DETALHES_DIVIDAS_VALOR_JUROS'] = base_dados['DETALHES_DIVIDAS_VALOR_JUROS'].astype(float)

base_dados.drop_duplicates(keep='first', inplace=True)

print("shape da Base de Dados:",base_dados.shape)

base_dados.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gerando variáveis Categóricas

# COMMAND ----------


base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh30'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2'] == '1-06DIAS', 0,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2'] == 'ATIVO', 1,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2'] == 'COB> 07D', 2,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2'] == 'CRELI', 3,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2'] == 'EXCESSO', 4,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2'] == 'H', 5,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2'] == 'J', 6,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2'] == 'Q', 7,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2'] == 'QUEBRAAC', 8,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2'] == 'SPC/SER', 9,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2'] == 'outros', 10,
0)))))))))))
base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh31'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh30'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh30'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh30'] == 2, 2,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh30'] == 3, 3,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh30'] == 4, 4,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh30'] == 5, 5,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh30'] == 6, 6,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh30'] == 7, 7,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh30'] == 8, 8,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh30'] == 9, 9,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh30'] == 10, 10,
0)))))))))))
base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh32'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh31'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh31'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh31'] == 2, 2,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh31'] == 3, 3,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh31'] == 4, 4,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh31'] == 5, 5,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh31'] == 6, 6,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh31'] == 7, 7,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh31'] == 8, 8,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh31'] == 9, 9,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh31'] == 10, 10,
0)))))))))))
base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh33'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh32'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh32'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh32'] == 2, 2,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh32'] == 3, 3,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh32'] == 4, 4,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh32'] == 5, 5,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh32'] == 6, 6,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh32'] == 7, 7,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh32'] == 8, 8,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh32'] == 9, 9,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh32'] == 10, 10,
0)))))))))))
base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh34'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh33'] == 0, 7,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh33'] == 1, 4,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh33'] == 2, 10,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh33'] == 3, 4,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh33'] == 4, 4,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh33'] == 5, 7,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh33'] == 6, 6,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh33'] == 7, 7,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh33'] == 8, 4,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh33'] == 9, 7,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh33'] == 10, 10,
0)))))))))))
base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh35'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh34'] == 4, 0,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh34'] == 6, 1,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh34'] == 7, 2,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh34'] == 10, 3,
0))))
base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh36'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh35'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh35'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh35'] == 2, 3,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh35'] == 3, 1,
0))))
base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh37'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh36'] == 0, 1,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh36'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh36'] == 3, 2,
0)))
base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh38'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh37'] == 1, 0,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh37'] == 2, 1,
0))
         
         
         
         
         
         
base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh30'] = np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO'] == 'A', 0,
np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO'] == 'G', 1,
np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO'] == 'P', 2,
np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO'] == 'Q', 3,
np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO'] == 'T', 4,
np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO'] == 'outros', 5,
0))))))
base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh31'] = np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh30'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh30'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh30'] == 2, 2,
np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh30'] == 3, 3,
np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh30'] == 4, 4,
np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh30'] == 5, 5,
0))))))
base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh32'] = np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh31'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh31'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh31'] == 2, 2,
np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh31'] == 3, 3,
np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh31'] == 4, 4,
np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh31'] == 5, 5,
0))))))
base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh33'] = np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh32'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh32'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh32'] == 2, 2,
np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh32'] == 3, 3,
np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh32'] == 4, 4,
np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh32'] == 5, 5,
0))))))
base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh34'] = np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh33'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh33'] == 1, 2,
np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh33'] == 2, 0,
np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh33'] == 3, 5,
np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh33'] == 4, 5,
np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh33'] == 5, 5,
0))))))
base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh35'] = np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh34'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh34'] == 2, 1,
np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh34'] == 5, 2,
0)))
base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh36'] = np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh35'] == 0, 2,
np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh35'] == 1, 0,
np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh35'] == 2, 1,
0)))
base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh37'] = np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh36'] == 0, 1,
np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh36'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh36'] == 2, 2,
0)))
base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh38'] = np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh37'] == 1, 0,
np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh37'] == 2, 1,
0))
                                                               
                                                               
                                                               
                                                               
                                                               
                                                               
base_dados['ID_DIVIDA_gh30'] = np.where(base_dados['ID_DIVIDA'] == 4329580000000000, 0,
np.where(base_dados['ID_DIVIDA'] == 4329590000000000, 1,
np.where(base_dados['ID_DIVIDA'] == 6367600000000000, 2,
np.where(base_dados['ID_DIVIDA'] == 6367610000000000, 3,
0))))
base_dados['ID_DIVIDA_gh31'] = np.where(base_dados['ID_DIVIDA_gh30'] == 0, 0,
np.where(base_dados['ID_DIVIDA_gh30'] == 1, 1,
np.where(base_dados['ID_DIVIDA_gh30'] == 2, 2,
np.where(base_dados['ID_DIVIDA_gh30'] == 3, 3,
0))))
base_dados['ID_DIVIDA_gh32'] = np.where(base_dados['ID_DIVIDA_gh31'] == 0, 0,
np.where(base_dados['ID_DIVIDA_gh31'] == 1, 1,
np.where(base_dados['ID_DIVIDA_gh31'] == 2, 2,
np.where(base_dados['ID_DIVIDA_gh31'] == 3, 3,
0))))
base_dados['ID_DIVIDA_gh33'] = np.where(base_dados['ID_DIVIDA_gh32'] == 0, 0,
np.where(base_dados['ID_DIVIDA_gh32'] == 1, 1,
np.where(base_dados['ID_DIVIDA_gh32'] == 2, 2,
np.where(base_dados['ID_DIVIDA_gh32'] == 3, 3,
0))))
base_dados['ID_DIVIDA_gh34'] = np.where(base_dados['ID_DIVIDA_gh33'] == 0, 0,
np.where(base_dados['ID_DIVIDA_gh33'] == 1, 1,
np.where(base_dados['ID_DIVIDA_gh33'] == 2, 2,
np.where(base_dados['ID_DIVIDA_gh33'] == 3, 3,
0))))
base_dados['ID_DIVIDA_gh35'] = np.where(base_dados['ID_DIVIDA_gh34'] == 0, 0,
np.where(base_dados['ID_DIVIDA_gh34'] == 1, 1,
np.where(base_dados['ID_DIVIDA_gh34'] == 2, 2,
np.where(base_dados['ID_DIVIDA_gh34'] == 3, 3,
0))))
base_dados['ID_DIVIDA_gh36'] = np.where(base_dados['ID_DIVIDA_gh35'] == 0, 3,
np.where(base_dados['ID_DIVIDA_gh35'] == 1, 2,
np.where(base_dados['ID_DIVIDA_gh35'] == 2, 0,
np.where(base_dados['ID_DIVIDA_gh35'] == 3, 1,
0))))
base_dados['ID_DIVIDA_gh37'] = np.where(base_dados['ID_DIVIDA_gh36'] == 0, 0,
np.where(base_dados['ID_DIVIDA_gh36'] == 1, 0,
np.where(base_dados['ID_DIVIDA_gh36'] == 2, 0,
np.where(base_dados['ID_DIVIDA_gh36'] == 3, 3,
0))))
base_dados['ID_DIVIDA_gh38'] = np.where(base_dados['ID_DIVIDA_gh37'] == 0, 0,
np.where(base_dados['ID_DIVIDA_gh37'] == 3, 1,
0))
                                              
                                              
                                              
                                              
                                              
                                              
base_dados['TIPO_TELEFONE_3_gh30'] = np.where(base_dados['TIPO_TELEFONE_3'] == 'False', 0,
np.where(base_dados['TIPO_TELEFONE_3'] == 'True', 1,
0))                                              
base_dados['TIPO_TELEFONE_3_gh31'] = np.where(base_dados['TIPO_TELEFONE_3_gh30'] == 0, 0,
np.where(base_dados['TIPO_TELEFONE_3_gh30'] == 1, 1,
0))
base_dados['TIPO_TELEFONE_3_gh32'] = np.where(base_dados['TIPO_TELEFONE_3_gh31'] == 0, 0,
np.where(base_dados['TIPO_TELEFONE_3_gh31'] == 1, 1,
0))
base_dados['TIPO_TELEFONE_3_gh33'] = np.where(base_dados['TIPO_TELEFONE_3_gh32'] == 0, 0,
np.where(base_dados['TIPO_TELEFONE_3_gh32'] == 1, 1,
0))
base_dados['TIPO_TELEFONE_3_gh34'] = np.where(base_dados['TIPO_TELEFONE_3_gh33'] == 0, 0,
np.where(base_dados['TIPO_TELEFONE_3_gh33'] == 1, 1,
0))
base_dados['TIPO_TELEFONE_3_gh35'] = np.where(base_dados['TIPO_TELEFONE_3_gh34'] == 0, 0,
np.where(base_dados['TIPO_TELEFONE_3_gh34'] == 1, 1,
0))
base_dados['TIPO_TELEFONE_3_gh36'] = np.where(base_dados['TIPO_TELEFONE_3_gh35'] == 0, 1,
np.where(base_dados['TIPO_TELEFONE_3_gh35'] == 1, 0,
0))
base_dados['TIPO_TELEFONE_3_gh37'] = np.where(base_dados['TIPO_TELEFONE_3_gh36'] == 0, 0,
np.where(base_dados['TIPO_TELEFONE_3_gh36'] == 1, 1,
0))
base_dados['TIPO_TELEFONE_3_gh38'] = np.where(base_dados['TIPO_TELEFONE_3_gh37'] == 0, 0,
np.where(base_dados['TIPO_TELEFONE_3_gh37'] == 1, 1,
0))
                                              
                                              
                                              
                                              
                                              
                                              
base_dados['TIPO_TELEFONE_4_gh30'] = np.where(base_dados['TIPO_TELEFONE_4'] == 'False', 0,
np.where(base_dados['TIPO_TELEFONE_4'] == 'True', 1,
0))
base_dados['TIPO_TELEFONE_4_gh31'] = np.where(base_dados['TIPO_TELEFONE_4_gh30'] == 0, 0,
np.where(base_dados['TIPO_TELEFONE_4_gh30'] == 1, 1,
0))
base_dados['TIPO_TELEFONE_4_gh32'] = np.where(base_dados['TIPO_TELEFONE_4_gh31'] == 0, 0,
np.where(base_dados['TIPO_TELEFONE_4_gh31'] == 1, 1,
0))
base_dados['TIPO_TELEFONE_4_gh33'] = np.where(base_dados['TIPO_TELEFONE_4_gh32'] == 0, 0,
np.where(base_dados['TIPO_TELEFONE_4_gh32'] == 1, 1,
0))
base_dados['TIPO_TELEFONE_4_gh34'] = np.where(base_dados['TIPO_TELEFONE_4_gh33'] == 0, 0,
np.where(base_dados['TIPO_TELEFONE_4_gh33'] == 1, 1,
0))
base_dados['TIPO_TELEFONE_4_gh35'] = np.where(base_dados['TIPO_TELEFONE_4_gh34'] == 0, 0,
np.where(base_dados['TIPO_TELEFONE_4_gh34'] == 1, 1,
0))
base_dados['TIPO_TELEFONE_4_gh36'] = np.where(base_dados['TIPO_TELEFONE_4_gh35'] == 0, 1,
np.where(base_dados['TIPO_TELEFONE_4_gh35'] == 1, 0,
0))
base_dados['TIPO_TELEFONE_4_gh37'] = np.where(base_dados['TIPO_TELEFONE_4_gh36'] == 0, 0,
np.where(base_dados['TIPO_TELEFONE_4_gh36'] == 1, 1,
0))
base_dados['TIPO_TELEFONE_4_gh38'] = np.where(base_dados['TIPO_TELEFONE_4_gh37'] == 0, 0,
np.where(base_dados['TIPO_TELEFONE_4_gh37'] == 1, 1,
0))






base_dados['DETALHES_CONTRATOS_CLASSE_gh60'] = np.where(base_dados['DETALHES_CONTRATOS_CLASSE'] == '121', 0,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE'] == '21', 1,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE'] == '701', 2,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE'] == '903', 3,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE'] == '904', 4,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE'] == 'ACORDOS QUEBRADOS SLD DEVEDOR', 5,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE'] == 'BLOQUEIO X', 6,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE'] == 'C.L ACIMA 62 DIAS AC 20.00 SR', 7,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE'] == 'EXCESSO DE LIMITE', 8,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE'] == 'RENEGOCIACAO EM ATRASO', 9,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE'] == 'outros', 10,
0)))))))))))
base_dados['DETALHES_CONTRATOS_CLASSE_gh61'] = np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh60'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh60'] == 1, -5,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh60'] == 2, -5,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh60'] == 3, -5,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh60'] == 4, 1,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh60'] == 5, 3,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh60'] == 6, -5,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh60'] == 7, 2,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh60'] == 8, -5,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh60'] == 9, -5,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh60'] == 10, -5,
0)))))))))))
base_dados['DETALHES_CONTRATOS_CLASSE_gh62'] = np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh61'] == -5, 0,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh61'] == 0, 1,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh61'] == 1, 2,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh61'] == 2, 3,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh61'] == 3, 4,
0)))))
base_dados['DETALHES_CONTRATOS_CLASSE_gh63'] = np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh62'] == 0, 3,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh62'] == 1, 2,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh62'] == 2, 4,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh62'] == 3, 0,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh62'] == 4, 1,
0)))))
base_dados['DETALHES_CONTRATOS_CLASSE_gh64'] = np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh63'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh63'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh63'] == 2, 2,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh63'] == 3, 3,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh63'] == 4, 4,
0)))))
base_dados['DETALHES_CONTRATOS_CLASSE_gh65'] = np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh64'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh64'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh64'] == 2, 2,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh64'] == 3, 3,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh64'] == 4, 4,
0)))))
base_dados['DETALHES_CONTRATOS_CLASSE_gh66'] = np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh65'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh65'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh65'] == 2, 2,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh65'] == 3, 3,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh65'] == 4, 4,
0)))))
base_dados['DETALHES_CONTRATOS_CLASSE_gh67'] = np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh66'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh66'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh66'] == 2, 2,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh66'] == 3, 3,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh66'] == 4, 4,
0)))))
base_dados['DETALHES_CONTRATOS_CLASSE_gh68'] = np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh67'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh67'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh67'] == 2, 2,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh67'] == 3, 3,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh67'] == 4, 4,
0)))))
base_dados['DETALHES_CONTRATOS_CLASSE_gh69'] = np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh68'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh68'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh68'] == 2, 2,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh68'] == 3, 3,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh68'] == 4, 4,
0)))))
base_dados['DETALHES_CONTRATOS_CLASSE_gh70'] = np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh69'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh69'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh69'] == 2, 2,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh69'] == 3, 3,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh69'] == 4, 4,
0)))))
base_dados['DETALHES_CONTRATOS_CLASSE_gh71'] = np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh70'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh70'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh70'] == 2, 2,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh70'] == 3, 3,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh70'] == 4, 4,
0)))))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Gerando variáveis numéricas contínuas

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parte 1 de 3

# COMMAND ----------

base_dados['DOCUMENTO_PESSOA__L'] = np.log(base_dados['DOCUMENTO_PESSOA'])
np.where(base_dados['DOCUMENTO_PESSOA__L'] == 0, -1, base_dados['DOCUMENTO_PESSOA__L'])
base_dados['DOCUMENTO_PESSOA__L'] = base_dados['DOCUMENTO_PESSOA__L'].fillna(-2)
base_dados['DOCUMENTO_PESSOA__L__p_7'] = np.where(base_dados['DOCUMENTO_PESSOA__L'] <= 22.381372155090062, 0.0,
np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__L'] > 22.381372155090062, base_dados['DOCUMENTO_PESSOA__L'] <= 23.109601910601373), 1.0,
np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__L'] > 23.109601910601373, base_dados['DOCUMENTO_PESSOA__L'] <= 23.71514308226829), 2.0,
np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__L'] > 23.71514308226829, base_dados['DOCUMENTO_PESSOA__L'] <= 24.177366619744948), 3.0,
np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__L'] > 24.177366619744948, base_dados['DOCUMENTO_PESSOA__L'] <= 24.385087658920906), 4.0,
np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__L'] > 24.385087658920906, base_dados['DOCUMENTO_PESSOA__L'] <= 24.59552490963863), 5.0,
np.where(base_dados['DOCUMENTO_PESSOA__L'] > 24.59552490963863, 6.0,
 0)))))))
base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_1'] = np.where(base_dados['DOCUMENTO_PESSOA__L__p_7'] == 0.0, 1,
np.where(base_dados['DOCUMENTO_PESSOA__L__p_7'] == 1.0, 1,
np.where(base_dados['DOCUMENTO_PESSOA__L__p_7'] == 2.0, 1,
np.where(base_dados['DOCUMENTO_PESSOA__L__p_7'] == 3.0, 1,
np.where(base_dados['DOCUMENTO_PESSOA__L__p_7'] == 4.0, 0,
np.where(base_dados['DOCUMENTO_PESSOA__L__p_7'] == 5.0, 0,
np.where(base_dados['DOCUMENTO_PESSOA__L__p_7'] == 6.0, 1,
 0)))))))
base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_2'] = np.where(base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_1'] == 0, 0,
np.where(base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_1'] == 1, 1,
 0))
base_dados['DOCUMENTO_PESSOA__L__p_7_g_1'] = np.where(base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_2'] == 0, 0,
np.where(base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_2'] == 1, 1,
 0))
                                                      
                                                      
                                                      
                                                      
                                                      
                                                      
                                                      
base_dados['DOCUMENTO_PESSOA__L'] = np.log(base_dados['DOCUMENTO_PESSOA'])
np.where(base_dados['DOCUMENTO_PESSOA__L'] == 0, -1, base_dados['DOCUMENTO_PESSOA__L'])
base_dados['DOCUMENTO_PESSOA__L'] = base_dados['DOCUMENTO_PESSOA__L'].fillna(-2)
base_dados['DOCUMENTO_PESSOA__L__pu_15'] = np.where(base_dados['DOCUMENTO_PESSOA__L'] <= 15.720562848760999, 0.0,
np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__L'] > 15.720562848760999, base_dados['DOCUMENTO_PESSOA__L'] <= 16.419777221546976), 1.0,
np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__L'] > 16.419777221546976, base_dados['DOCUMENTO_PESSOA__L'] <= 16.899969249484), 2.0,
np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__L'] > 16.899969249484, base_dados['DOCUMENTO_PESSOA__L'] <= 17.79535292792268), 3.0,
np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__L'] > 17.79535292792268, base_dados['DOCUMENTO_PESSOA__L'] <= 18.51233110751913), 4.0,
np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__L'] > 18.51233110751913, base_dados['DOCUMENTO_PESSOA__L'] <= 19.26546130431107), 5.0,
np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__L'] > 19.26546130431107, base_dados['DOCUMENTO_PESSOA__L'] <= 19.942991609434596), 6.0,
np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__L'] > 19.942991609434596, base_dados['DOCUMENTO_PESSOA__L'] <= 20.640620683931026), 7.0,
np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__L'] > 20.640620683931026, base_dados['DOCUMENTO_PESSOA__L'] <= 21.310964008488664), 8.0,
np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__L'] > 21.310964008488664, base_dados['DOCUMENTO_PESSOA__L'] <= 21.98304289992455), 9.0,
np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__L'] > 21.98304289992455, base_dados['DOCUMENTO_PESSOA__L'] <= 22.652314259898965), 10.0,
np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__L'] > 22.652314259898965, base_dados['DOCUMENTO_PESSOA__L'] <= 23.320231308218624), 11.0,
np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__L'] > 23.320231308218624, base_dados['DOCUMENTO_PESSOA__L'] <= 23.989153096487023), 12.0,
np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__L'] > 23.989153096487023, base_dados['DOCUMENTO_PESSOA__L'] <= 24.6588459585914), 13.0,
np.where(base_dados['DOCUMENTO_PESSOA__L'] > 24.6588459585914, 14.0,
 0)))))))))))))))
base_dados['DOCUMENTO_PESSOA__L__pu_15_g_1_1'] = np.where(base_dados['DOCUMENTO_PESSOA__L__pu_15'] == 0.0, 1,
np.where(base_dados['DOCUMENTO_PESSOA__L__pu_15'] == 1.0, 1,
np.where(base_dados['DOCUMENTO_PESSOA__L__pu_15'] == 2.0, 1,
np.where(base_dados['DOCUMENTO_PESSOA__L__pu_15'] == 3.0, 1,
np.where(base_dados['DOCUMENTO_PESSOA__L__pu_15'] == 4.0, 1,
np.where(base_dados['DOCUMENTO_PESSOA__L__pu_15'] == 5.0, 1,
np.where(base_dados['DOCUMENTO_PESSOA__L__pu_15'] == 6.0, 1,
np.where(base_dados['DOCUMENTO_PESSOA__L__pu_15'] == 7.0, 1,
np.where(base_dados['DOCUMENTO_PESSOA__L__pu_15'] == 8.0, 1,
np.where(base_dados['DOCUMENTO_PESSOA__L__pu_15'] == 9.0, 1,
np.where(base_dados['DOCUMENTO_PESSOA__L__pu_15'] == 10.0, 1,
np.where(base_dados['DOCUMENTO_PESSOA__L__pu_15'] == 11.0, 1,
np.where(base_dados['DOCUMENTO_PESSOA__L__pu_15'] == 12.0, 1,
np.where(base_dados['DOCUMENTO_PESSOA__L__pu_15'] == 13.0, 0,
np.where(base_dados['DOCUMENTO_PESSOA__L__pu_15'] == 14.0, 1,
 0)))))))))))))))
base_dados['DOCUMENTO_PESSOA__L__pu_15_g_1_2'] = np.where(base_dados['DOCUMENTO_PESSOA__L__pu_15_g_1_1'] == 0, 0,
np.where(base_dados['DOCUMENTO_PESSOA__L__pu_15_g_1_1'] == 1, 1,
 0))
base_dados['DOCUMENTO_PESSOA__L__pu_15_g_1'] = np.where(base_dados['DOCUMENTO_PESSOA__L__pu_15_g_1_2'] == 0, 0,
np.where(base_dados['DOCUMENTO_PESSOA__L__pu_15_g_1_2'] == 1, 1,
 0))
         
         
         
         
         
         
         
base_dados['DETALHES_CLIENTES_SCORE_CARGA__p_5'] = np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA'] == 0 , -1.0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_SCORE_CARGA'] > 0.0, base_dados['DETALHES_CLIENTES_SCORE_CARGA'] <= 98.0), 0.0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_SCORE_CARGA'] > 98.0, base_dados['DETALHES_CLIENTES_SCORE_CARGA'] <= 214.0), 1.0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_SCORE_CARGA'] > 214.0, base_dados['DETALHES_CLIENTES_SCORE_CARGA'] <= 317.0), 2.0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_SCORE_CARGA'] > 317.0, base_dados['DETALHES_CLIENTES_SCORE_CARGA'] <= 433.0), 3.0,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA'] > 433.0, 4.0,
 0))))))
base_dados['DETALHES_CLIENTES_SCORE_CARGA__p_5_g_1_1'] = np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__p_5'] == -1.0, 1,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__p_5'] == 0.0, 0,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__p_5'] == 1.0, 0,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__p_5'] == 2.0, 0,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__p_5'] == 3.0, 1,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__p_5'] == 4.0, 1,
 0))))))
base_dados['DETALHES_CLIENTES_SCORE_CARGA__p_5_g_1_2'] = np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__p_5_g_1_1'] == 0, 1,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__p_5_g_1_1'] == 1, 0,
 0))
base_dados['DETALHES_CLIENTES_SCORE_CARGA__p_5_g_1'] = np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__p_5_g_1_2'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__p_5_g_1_2'] == 1, 1,
 0))
                                                                
                                                                
                                                                
                                                                
                                                                
                                                                
                                                                
base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] = np.log(base_dados['DETALHES_CLIENTES_SCORE_CARGA'])
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] == 0, -1, base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'])
base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] = base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'].fillna(-2)
base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13'] = np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] == 0 , -1.0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] > 0.0, base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] <= 1.0986122886681098), 1.0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] > 1.0986122886681098, base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] <= 1.6094379124341003), 2.0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] > 1.6094379124341003, base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] <= 2.1972245773362196), 3.0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] > 2.1972245773362196, base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] <= 2.833213344056216), 4.0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] > 2.833213344056216, base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] <= 3.4011973816621555), 5.0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] > 3.4011973816621555, base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] <= 3.9889840465642745), 6.0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] > 3.9889840465642745, base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] <= 4.564348191467836), 7.0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] > 4.564348191467836, base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] <= 5.135798437050262), 8.0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] > 5.135798437050262, base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] <= 5.707110264748875), 9.0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] > 5.707110264748875, base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] <= 6.278521424165844), 10.0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] > 6.278521424165844, base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] <= 6.839476438228843), 11.0,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] > 6.839476438228843, 12.0,
 -2)))))))))))))
base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_1'] = np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13'] == -2.0, 2,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13'] == -1.0, 2,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13'] == 1.0, 2,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13'] == 2.0, 1,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13'] == 3.0, 2,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13'] == 4.0, 0,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13'] == 5.0, 0,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13'] == 6.0, 1,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13'] == 7.0, 2,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13'] == 8.0, 0,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13'] == 9.0, 0,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13'] == 10.0, 1,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13'] == 11.0, 2,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13'] == 12.0, 2,
 0))))))))))))))
base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_2'] = np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_1'] == 0, 1,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_1'] == 1, 0,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_1'] == 2, 1,
 0)))
base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1'] = np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_2'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_2'] == 1, 1,
 0))
                                                                     
                                                                     
                                                                     
                                                                     
                                                                     
                                                                     
                                                                     
                                                                     
base_dados['DETALHES_CLIENTES_VALOR_FATURA__R'] = np.sqrt(base_dados['DETALHES_CLIENTES_VALOR_FATURA'])
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R'] == 0, -1, base_dados['DETALHES_CLIENTES_VALOR_FATURA__R'])
base_dados['DETALHES_CLIENTES_VALOR_FATURA__R'] = base_dados['DETALHES_CLIENTES_VALOR_FATURA__R'].fillna(-2)
base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R'] <= 26.31425469208657, 0.0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R'] > 26.31425469208657, base_dados['DETALHES_CLIENTES_VALOR_FATURA__R'] <= 37.54210969031975), 1.0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R'] > 37.54210969031975, base_dados['DETALHES_CLIENTES_VALOR_FATURA__R'] <= 49.663266102824934), 2.0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R'] > 49.663266102824934, base_dados['DETALHES_CLIENTES_VALOR_FATURA__R'] <= 69.18106677408205), 3.0,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R'] > 69.18106677408205, 4.0,
 0)))))
base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_1'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5'] == 0.0, 2,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5'] == 1.0, 0,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5'] == 2.0, 0,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5'] == 3.0, 1,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5'] == 4.0, 2,
 0)))))
base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_2'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_1'] == 0, 1,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_1'] == 1, 0,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_1'] == 2, 1,
 0)))
base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_2'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_2'] == 1, 1,
 0))
                                                                     
                                                                     
                                                                     
                                                                     
                                                                     
                                                                     
                                                                     
                                                                     
base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] = np.tan(base_dados['DETALHES_CLIENTES_VALOR_FATURA'])
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] == 0, -1, base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'])
base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] = base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'].fillna(-2)
base_dados['DETALHES_CLIENTES_VALOR_FATURA__T__pk_34'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] <= -3458.227209386098, 0.0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] > -3458.227209386098, base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] <= -788.7952247406655), 1.0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] > -788.7952247406655, base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] <= -548.9883164955077), 2.0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] > -548.9883164955077, base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] <= -338.56473991927754), 3.0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] > -338.56473991927754, base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] <= -141.69600726545013), 4.0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] > -141.69600726545013, base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] <= -50.656640677440166), 5.0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] > -50.656640677440166, base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] <= -10.834696298958422), 6.0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] > -10.834696298958422, base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] <= -0.3186086054636096), 7.0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] > -0.3186086054636096, base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] <= 3.7406178677331163), 8.0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] > 3.7406178677331163, base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] <= 10.395783588085726), 9.0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] > 10.395783588085726, base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] <= 17.807270240190984), 10.0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] > 17.807270240190984, base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] <= 23.834596106409194), 11.0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] > 23.834596106409194, base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] <= 28.80471629136568), 12.0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] > 28.80471629136568, base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] <= 31.701530575665), 13.0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] > 31.701530575665, base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] <= 32.72049189877904), 14.0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] > 32.72049189877904, base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] <= 34.521061760300675), 15.0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] > 34.521061760300675, base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] <= 35.22836109378995), 16.0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] > 35.22836109378995, base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] <= 35.39940395545131), 17.0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] > 35.39940395545131, base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] <= 36.08631384161725), 18.0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] > 36.08631384161725, base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] <= 37.647886787701424), 19.0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] > 37.647886787701424, base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] <= 38.17618654512842), 20.0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] > 38.17618654512842, base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] <= 38.55979978746642), 21.0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] > 38.55979978746642, base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] <= 41.07977340501236), 22.0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] > 41.07977340501236, base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] <= 42.03661604602068), 23.0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] > 42.03661604602068, base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] <= 42.250892304070405), 24.0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] > 42.250892304070405, base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] <= 46.53228430545966), 25.0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] > 46.53228430545966, base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] <= 49.873577926259266), 26.0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] > 49.873577926259266, base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] <= 63.5099118051609), 27.0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] > 63.5099118051609, base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] <= 96.87603286912503), 28.0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] > 96.87603286912503, base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] <= 130.30584031267466), 29.0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] > 130.30584031267466, base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] <= 229.0713822444694), 30.0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] > 229.0713822444694, base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] <= 348.9177223005156), 31.0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] > 348.9177223005156, base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] <= 815.1747969209366), 32.0,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T'] > 815.1747969209366, 33.0,
 0))))))))))))))))))))))))))))))))))
base_dados['DETALHES_CLIENTES_VALOR_FATURA__T__pk_34_g_1_1'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T__pk_34'] == 0.0, 2,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T__pk_34'] == 1.0, 1,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T__pk_34'] == 2.0, 2,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T__pk_34'] == 3.0, 2,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T__pk_34'] == 4.0, 0,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T__pk_34'] == 5.0, 2,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T__pk_34'] == 6.0, 0,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T__pk_34'] == 7.0, 0,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T__pk_34'] == 8.0, 0,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T__pk_34'] == 9.0, 0,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T__pk_34'] == 10.0, 0,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T__pk_34'] == 11.0, 2,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T__pk_34'] == 12.0, 0,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T__pk_34'] == 13.0, 0,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T__pk_34'] == 14.0, 2,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T__pk_34'] == 15.0, 2,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T__pk_34'] == 16.0, 2,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T__pk_34'] == 17.0, 2,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T__pk_34'] == 18.0, 2,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T__pk_34'] == 19.0, 2,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T__pk_34'] == 20.0, 2,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T__pk_34'] == 21.0, 2,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T__pk_34'] == 22.0, 2,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T__pk_34'] == 23.0, 2,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T__pk_34'] == 24.0, 2,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T__pk_34'] == 25.0, 0,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T__pk_34'] == 26.0, 2,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T__pk_34'] == 27.0, 2,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T__pk_34'] == 28.0, 2,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T__pk_34'] == 29.0, 2,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T__pk_34'] == 30.0, 2,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T__pk_34'] == 31.0, 2,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T__pk_34'] == 32.0, 2,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T__pk_34'] == 33.0, 2,
 0))))))))))))))))))))))))))))))))))
base_dados['DETALHES_CLIENTES_VALOR_FATURA__T__pk_34_g_1_2'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T__pk_34_g_1_1'] == 0, 1,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T__pk_34_g_1_1'] == 1, 2,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T__pk_34_g_1_1'] == 2, 0,
 0)))
base_dados['DETALHES_CLIENTES_VALOR_FATURA__T__pk_34_g_1'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T__pk_34_g_1_2'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T__pk_34_g_1_2'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__T__pk_34_g_1_2'] == 2, 2,
 0)))
         
         
         
         
         
         
         
base_dados['DETALHES_DIVIDAS_VALOR_JUROS__R'] = np.sqrt(base_dados['DETALHES_DIVIDAS_VALOR_JUROS'])
np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__R'] == 0, -1, base_dados['DETALHES_DIVIDAS_VALOR_JUROS__R'])
base_dados['DETALHES_DIVIDAS_VALOR_JUROS__R'] = base_dados['DETALHES_DIVIDAS_VALOR_JUROS__R'].fillna(-2)
base_dados['DETALHES_DIVIDAS_VALOR_JUROS__R__pu_3'] = np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__R'] <= 19.341664871463365, 0.0,
np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__R'] > 19.341664871463365, base_dados['DETALHES_DIVIDAS_VALOR_JUROS__R'] <= 39.10843898700126), 1.0,
np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__R'] > 39.10843898700126, 2.0,
 0)))
base_dados['DETALHES_DIVIDAS_VALOR_JUROS__R__pu_3_g_1_1'] = np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__R__pu_3'] == 0.0, 0,
np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__R__pu_3'] == 1.0, 1,
np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__R__pu_3'] == 2.0, 1,
 0)))
base_dados['DETALHES_DIVIDAS_VALOR_JUROS__R__pu_3_g_1_2'] = np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__R__pu_3_g_1_1'] == 0, 1,
np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__R__pu_3_g_1_1'] == 1, 0,
 0))
base_dados['DETALHES_DIVIDAS_VALOR_JUROS__R__pu_3_g_1'] = np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__R__pu_3_g_1_2'] == 0, 0,
np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__R__pu_3_g_1_2'] == 1, 1,
 0))
                                                                   
                                                                   
                                                                   
                                                                   
                                                                   
                                                                   
base_dados['DETALHES_DIVIDAS_VALOR_JUROS__S'] = np.sin(base_dados['DETALHES_DIVIDAS_VALOR_JUROS'])
np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__S'] == 0, -1, base_dados['DETALHES_DIVIDAS_VALOR_JUROS__S'])
base_dados['DETALHES_DIVIDAS_VALOR_JUROS__S'] = base_dados['DETALHES_DIVIDAS_VALOR_JUROS__S'].fillna(-2)
base_dados['DETALHES_DIVIDAS_VALOR_JUROS__S__p_17'] = np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__S'] <= -0.9997859814915458, 0.0,
np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__S'] > -0.9997859814915458, base_dados['DETALHES_DIVIDAS_VALOR_JUROS__S'] <= -0.9199121403212497), 1.0,
np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__S'] > -0.9199121403212497, base_dados['DETALHES_DIVIDAS_VALOR_JUROS__S'] <= -0.7216850939753164), 2.0,
np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__S'] > -0.7216850939753164, base_dados['DETALHES_DIVIDAS_VALOR_JUROS__S'] <= -0.4469373876612218), 3.0,
np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__S'] > -0.4469373876612218, base_dados['DETALHES_DIVIDAS_VALOR_JUROS__S'] <= -0.16161534181868528), 4.0,
np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__S'] > -0.16161534181868528, base_dados['DETALHES_DIVIDAS_VALOR_JUROS__S'] <= 0.203140422885245), 5.0,
np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__S'] > 0.203140422885245, base_dados['DETALHES_DIVIDAS_VALOR_JUROS__S'] <= 0.5003445783714376), 6.0,
np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__S'] > 0.5003445783714376, base_dados['DETALHES_DIVIDAS_VALOR_JUROS__S'] <= 0.7683529246060552), 7.0,
np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__S'] > 0.7683529246060552, base_dados['DETALHES_DIVIDAS_VALOR_JUROS__S'] <= 0.9373455246227418), 8.0,
np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__S'] > 0.9373455246227418, 9.0,
 0))))))))))
base_dados['DETALHES_DIVIDAS_VALOR_JUROS__S__p_17_g_1_1'] = np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__S__p_17'] == 0.0, 0,
np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__S__p_17'] == 1.0, 1,
np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__S__p_17'] == 2.0, 2,
np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__S__p_17'] == 3.0, 1,
np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__S__p_17'] == 4.0, 1,
np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__S__p_17'] == 5.0, 1,
np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__S__p_17'] == 6.0, 2,
np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__S__p_17'] == 7.0, 1,
np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__S__p_17'] == 8.0, 1,
np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__S__p_17'] == 9.0, 2,
 0))))))))))
base_dados['DETALHES_DIVIDAS_VALOR_JUROS__S__p_17_g_1_2'] = np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__S__p_17_g_1_1'] == 0, 2,
np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__S__p_17_g_1_1'] == 1, 0,
np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__S__p_17_g_1_1'] == 2, 1,
 0)))
base_dados['DETALHES_DIVIDAS_VALOR_JUROS__S__p_17_g_1'] = np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__S__p_17_g_1_2'] == 0, 0,
np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__S__p_17_g_1_2'] == 1, 1,
np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__S__p_17_g_1_2'] == 2, 2,
 0)))



# COMMAND ----------

# MAGIC %md
# MAGIC ### Parte 2 de 3

# COMMAND ----------

base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_4_1'] = np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__L__p_7_g_1'] == 0, base_dados['DOCUMENTO_PESSOA__L__pu_15_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__L__p_7_g_1'] == 0, base_dados['DOCUMENTO_PESSOA__L__pu_15_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__L__p_7_g_1'] == 1, base_dados['DOCUMENTO_PESSOA__L__pu_15_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__L__p_7_g_1'] == 1, base_dados['DOCUMENTO_PESSOA__L__pu_15_g_1'] == 1), 1,
 0))))
base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_4_2'] = np.where(base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_4_1'] == 0, 0,
np.where(base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_4_1'] == 1, 1,
0))
base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_4'] = np.where(base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_4_2'] == 0, 0,
np.where(base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_4_2'] == 1, 1,
 0))
         
         
         
         
         
base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_9_1'] = np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_SCORE_CARGA__p_5_g_1'] == 0, base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_SCORE_CARGA__p_5_g_1'] == 0, base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_SCORE_CARGA__p_5_g_1'] == 1, base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_SCORE_CARGA__p_5_g_1'] == 1, base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1'] == 1), 1,
 0))))
base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_9_2'] = np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_9_1'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_9_1'] == 1, 1,
0))
base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_9'] = np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_9_2'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_9_2'] == 1, 1,
 0))
                                                                          
                                                                          
                                                                          
                                                                          
                                                                          
base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_5_1'] = np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1'] == 0, base_dados['DETALHES_CLIENTES_VALOR_FATURA__T__pk_34_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1'] == 0, base_dados['DETALHES_CLIENTES_VALOR_FATURA__T__pk_34_g_1'] == 1), 0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1'] == 0, base_dados['DETALHES_CLIENTES_VALOR_FATURA__T__pk_34_g_1'] == 2), 1,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1'] == 1, base_dados['DETALHES_CLIENTES_VALOR_FATURA__T__pk_34_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1'] == 1, base_dados['DETALHES_CLIENTES_VALOR_FATURA__T__pk_34_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1'] == 1, base_dados['DETALHES_CLIENTES_VALOR_FATURA__T__pk_34_g_1'] == 2), 1,
 0))))))
base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_5_2'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_5_1'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_5_1'] == 1, 1,
0))
base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_5'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_5_2'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_5_2'] == 1, 1,
 0))
         
         
         
         
         
         
base_dados['DETALHES_DIVIDAS_VALOR_JUROS__R__pu_3_g_1_c1_43_1'] = np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__R__pu_3_g_1'] == 0, base_dados['DETALHES_DIVIDAS_VALOR_JUROS__S__p_17_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__R__pu_3_g_1'] == 0, base_dados['DETALHES_DIVIDAS_VALOR_JUROS__S__p_17_g_1'] == 1), 0,
np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__R__pu_3_g_1'] == 0, base_dados['DETALHES_DIVIDAS_VALOR_JUROS__S__p_17_g_1'] == 2), 0,
np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__R__pu_3_g_1'] == 1, base_dados['DETALHES_DIVIDAS_VALOR_JUROS__S__p_17_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__R__pu_3_g_1'] == 1, base_dados['DETALHES_DIVIDAS_VALOR_JUROS__S__p_17_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__R__pu_3_g_1'] == 1, base_dados['DETALHES_DIVIDAS_VALOR_JUROS__S__p_17_g_1'] == 2), 3,
 0))))))
base_dados['DETALHES_DIVIDAS_VALOR_JUROS__R__pu_3_g_1_c1_43_2'] = np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__R__pu_3_g_1_c1_43_1'] == 0, 0,
np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__R__pu_3_g_1_c1_43_1'] == 1, 1,
np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__R__pu_3_g_1_c1_43_1'] == 2, 2,
np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__R__pu_3_g_1_c1_43_1'] == 3, 3,
0))))
base_dados['DETALHES_DIVIDAS_VALOR_JUROS__R__pu_3_g_1_c1_43'] = np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__R__pu_3_g_1_c1_43_2'] == 0, 0,
np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__R__pu_3_g_1_c1_43_2'] == 1, 1,
np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__R__pu_3_g_1_c1_43_2'] == 2, 2,
np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__R__pu_3_g_1_c1_43_2'] == 3, 3,
 0))))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gerando variáveis compostas gerais

# COMMAND ----------

base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_1'] = np.where(np.bitwise_and(base_dados['ID_DIVIDA_gh38'] == 0, base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_9'] == 0), 0,
np.where(np.bitwise_and(base_dados['ID_DIVIDA_gh38'] == 0, base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_9'] == 1), 2,
np.where(np.bitwise_and(base_dados['ID_DIVIDA_gh38'] == 1, base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_9'] == 0), 1,
np.where(np.bitwise_and(base_dados['ID_DIVIDA_gh38'] == 1, base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_9'] == 1), 3,
 0))))
base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_2'] = np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_1'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_1'] == 1, 3,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_1'] == 2, 1,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_1'] == 3, 2,
 0))))
base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_3'] = np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_2'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_2'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_2'] == 2, 2,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_2'] == 3, 3,
 0))))
base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_4'] = np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_3'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_3'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_3'] == 2, 2,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_3'] == 3, 3,
 0))))
base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_5'] = np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_4'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_4'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_4'] == 2, 2,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_4'] == 3, 3,
 0))))
base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_6'] = np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_5'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_5'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_5'] == 2, 2,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_5'] == 3, 3,
 0))))
base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_7'] = np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_6'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_6'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_6'] == 2, 2,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_6'] == 3, 3,
 0))))
base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_8'] = np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_7'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_7'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_7'] == 2, 2,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_7'] == 3, 3,
 0))))
base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_9'] = np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_8'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_8'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_8'] == 2, 2,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_8'] == 3, 3,
 0))))
base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_10'] = np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_9'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_9'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_9'] == 2, 2,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_9'] == 3, 3,
 0))))
base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_11'] = np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_10'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_10'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_10'] == 2, 2,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_10'] == 3, 3,
 0))))
base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_12'] = np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_11'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_11'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_11'] == 2, 2,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_11'] == 3, 2,
 0))))
base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_13'] = np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_12'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_12'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_12'] == 2, 2,
 0)))
base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_14'] = np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_13'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_13'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_13'] == 2, 2,
 0)))
base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29'] = np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_14'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_14'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29_14'] == 2, 2,
 0)))
         
         
         
         
         
         
         
base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_1'] = np.where(np.bitwise_and(base_dados['TIPO_TELEFONE_4_gh38'] == 0, base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_5'] == 0), 0,
np.where(np.bitwise_and(base_dados['TIPO_TELEFONE_4_gh38'] == 0, base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_5'] == 1), 2,
np.where(np.bitwise_and(base_dados['TIPO_TELEFONE_4_gh38'] == 1, base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_5'] == 0), 1,
np.where(np.bitwise_and(base_dados['TIPO_TELEFONE_4_gh38'] == 1, base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_5'] == 1), 3,
 0))))
base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_2'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_1'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_1'] == 1, 2,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_1'] == 2, 1,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_1'] == 3, 3,
 0))))
base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_3'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_2'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_2'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_2'] == 2, 2,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_2'] == 3, 3,
 0))))
base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_4'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_3'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_3'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_3'] == 2, 2,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_3'] == 3, 3,
 0))))
base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_5'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_4'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_4'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_4'] == 2, 2,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_4'] == 3, 3,
 0))))
base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_6'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_5'] == 0, 1,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_5'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_5'] == 2, 2,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_5'] == 3, 3,
 0))))
base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_7'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_6'] == 1, 0,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_6'] == 2, 1,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_6'] == 3, 2,
 0)))
base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_8'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_7'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_7'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_7'] == 2, 2,
 0)))
base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_9'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_8'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_8'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_8'] == 2, 2,
 0)))
base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_10'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_9'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_9'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_9'] == 2, 2,
 0)))
base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_11'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_10'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_10'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_10'] == 2, 2,
 0)))
base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_12'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_11'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_11'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_11'] == 2, 2,
 0)))
base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_13'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_12'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_12'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_12'] == 2, 2,
 0)))
base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_14'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_13'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_13'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_13'] == 2, 2,
 0)))
base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_14'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_14'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29_14'] == 2, 2,
 0)))
         
         
         
         
         
         
         
base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_1'] = np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh38'] == 0, base_dados['DETALHES_DIVIDAS_VALOR_JUROS__R__pu_3_g_1_c1_43'] == 0), 0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh38'] == 0, base_dados['DETALHES_DIVIDAS_VALOR_JUROS__R__pu_3_g_1_c1_43'] == 1), 1,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh38'] == 0, base_dados['DETALHES_DIVIDAS_VALOR_JUROS__R__pu_3_g_1_c1_43'] == 2), 2,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh38'] == 0, base_dados['DETALHES_DIVIDAS_VALOR_JUROS__R__pu_3_g_1_c1_43'] == 3), 3,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh38'] == 1, base_dados['DETALHES_DIVIDAS_VALOR_JUROS__R__pu_3_g_1_c1_43'] == 0), 4,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh38'] == 1, base_dados['DETALHES_DIVIDAS_VALOR_JUROS__R__pu_3_g_1_c1_43'] == 1), 5,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh38'] == 1, base_dados['DETALHES_DIVIDAS_VALOR_JUROS__R__pu_3_g_1_c1_43'] == 2), 6,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh38'] == 1, base_dados['DETALHES_DIVIDAS_VALOR_JUROS__R__pu_3_g_1_c1_43'] == 3), 7,
 0))))))))
base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_2'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_1'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_1'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_1'] == 2, 2,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_1'] == 3, 2,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_1'] == 4, 2,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_1'] == 5, 2,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_1'] == 6, 2,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_1'] == 7, 7,
 0))))))))
base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_3'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_2'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_2'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_2'] == 2, 2,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_2'] == 7, 3,
 0))))
base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_4'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_3'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_3'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_3'] == 2, 2,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_3'] == 3, 3,
 0))))
base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_5'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_4'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_4'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_4'] == 2, 2,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_4'] == 3, 3,
 0))))
base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_6'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_5'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_5'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_5'] == 2, 2,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_5'] == 3, 3,
 0))))
base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_7'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_6'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_6'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_6'] == 2, 2,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_6'] == 3, 3,
 0))))
base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_8'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_7'] == 0, 1,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_7'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_7'] == 2, 2,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_7'] == 3, 3,
 0))))
base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_9'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_8'] == 1, 0,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_8'] == 2, 1,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_8'] == 3, 2,
 0)))
base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_10'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_9'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_9'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_9'] == 2, 2,
 0)))
base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_11'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_10'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_10'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_10'] == 2, 2,
 0)))
base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_12'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_11'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_11'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_11'] == 2, 2,
 0)))
base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_13'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_12'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_12'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_12'] == 2, 2,
 0)))
base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_14'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_13'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_13'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_13'] == 2, 2,
 0)))
base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_14'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_14'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29_14'] == 2, 2,
 0)))
         
         
         
         
         
         
         
base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_1'] = np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_CLASSE_gh71'] == 0, base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh38'] == 0), 0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_CLASSE_gh71'] == 0, base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh38'] == 1), 1,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_CLASSE_gh71'] == 1, base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh38'] == 0), 2,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_CLASSE_gh71'] == 1, base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh38'] == 1), 3,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_CLASSE_gh71'] == 2, base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh38'] == 0), 4,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_CLASSE_gh71'] == 2, base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh38'] == 1), 5,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_CLASSE_gh71'] == 3, base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh38'] == 0), 6,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_CLASSE_gh71'] == 3, base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh38'] == 1), 7,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_CLASSE_gh71'] == 4, base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh38'] == 0), 8,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_CLASSE_gh71'] == 4, base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh38'] == 1), 9,
 0))))))))))
base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_2'] = np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_1'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_1'] == 1, 2,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_1'] == 2, 1,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_1'] == 3, 2,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_1'] == 4, 4,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_1'] == 5, 6,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_1'] == 6, 5,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_1'] == 7, 8,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_1'] == 8, 6,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_1'] == 9, 8,
 0))))))))))
base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_3'] = np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_2'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_2'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_2'] == 2, 2,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_2'] == 4, 3,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_2'] == 5, 4,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_2'] == 6, 5,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_2'] == 8, 6,
 0)))))))
base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_4'] = np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_3'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_3'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_3'] == 2, 2,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_3'] == 3, 3,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_3'] == 4, 4,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_3'] == 5, 5,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_3'] == 6, 6,
 0)))))))
base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_5'] = np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_4'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_4'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_4'] == 2, 2,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_4'] == 3, 3,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_4'] == 4, 4,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_4'] == 5, 5,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_4'] == 6, 6,
 0)))))))
base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_6'] = np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_5'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_5'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_5'] == 2, 1,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_5'] == 3, 3,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_5'] == 4, 4,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_5'] == 5, 5,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_5'] == 6, 6,
 0)))))))
base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_7'] = np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_6'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_6'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_6'] == 3, 2,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_6'] == 4, 3,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_6'] == 5, 4,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_6'] == 6, 5,
 0))))))
base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_8'] = np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_7'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_7'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_7'] == 2, 2,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_7'] == 3, 3,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_7'] == 4, 4,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_7'] == 5, 5,
 0))))))
base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_9'] = np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_8'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_8'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_8'] == 2, 2,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_8'] == 3, 3,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_8'] == 4, 4,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_8'] == 5, 5,
 0))))))
base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_10'] = np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_9'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_9'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_9'] == 2, 2,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_9'] == 3, 3,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_9'] == 4, 4,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_9'] == 5, 5,
 0))))))
base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_11'] = np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_10'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_10'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_10'] == 2, 2,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_10'] == 3, 3,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_10'] == 4, 4,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_10'] == 5, 5,
 0))))))
base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_12'] = np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_11'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_11'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_11'] == 2, 2,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_11'] == 3, 2,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_11'] == 4, 4,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_11'] == 5, 5,
 0))))))
base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_13'] = np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_12'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_12'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_12'] == 2, 2,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_12'] == 4, 3,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_12'] == 5, 4,
 0)))))
base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_14'] = np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_13'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_13'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_13'] == 2, 2,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_13'] == 3, 3,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_13'] == 4, 4,
 0)))))
base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29'] = np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_14'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_14'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_14'] == 2, 2,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_14'] == 3, 3,
np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh717_gh29_14'] == 4, 4,
 0)))))
         
         
         
         
         
         
base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_48_gh29_1'] = np.where(np.bitwise_and(base_dados['TIPO_TELEFONE_3_gh38'] == 0, base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_4'] == 0), 0,
np.where(np.bitwise_and(base_dados['TIPO_TELEFONE_3_gh38'] == 0, base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_4'] == 1), 2,
np.where(np.bitwise_and(base_dados['TIPO_TELEFONE_3_gh38'] == 1, base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_4'] == 0), 1,
np.where(np.bitwise_and(base_dados['TIPO_TELEFONE_3_gh38'] == 1, base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_4'] == 1), 3,
 0))))
base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_48_gh29_2'] = np.where(base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_48_gh29_1'] == 0, 0,
np.where(base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_48_gh29_1'] == 1, 1,
np.where(base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_48_gh29_1'] == 2, 1,
np.where(base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_48_gh29_1'] == 3, 3,
 0))))
base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_48_gh29_3'] = np.where(base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_48_gh29_2'] == 0, 0,
np.where(base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_48_gh29_2'] == 1, 1,
np.where(base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_48_gh29_2'] == 3, 2,
 0)))
base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_48_gh29_4'] = np.where(base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_48_gh29_3'] == 0, 0,
np.where(base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_48_gh29_3'] == 1, 1,
np.where(base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_48_gh29_3'] == 2, 2,
 0)))
base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_48_gh29_5'] = np.where(base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_48_gh29_4'] == 0, 0,
np.where(base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_48_gh29_4'] == 1, 1,
np.where(base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_48_gh29_4'] == 2, 2,
 0)))
base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_48_gh29_6'] = np.where(base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_48_gh29_5'] == 0, 1,
np.where(base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_48_gh29_5'] == 1, 1,
np.where(base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_48_gh29_5'] == 2, 2,
 0)))
base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_48_gh29_7'] = np.where(base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_48_gh29_6'] == 1, 0,
np.where(base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_48_gh29_6'] == 2, 1,
 0))
base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_48_gh29_8'] = np.where(base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_48_gh29_7'] == 0, 0,
np.where(base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_48_gh29_7'] == 1, 1,
 0))
base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_48_gh29_9'] = np.where(base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_48_gh29_8'] == 0, 0,
np.where(base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_48_gh29_8'] == 1, 1,
 0))
base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_48_gh29_10'] = np.where(base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_48_gh29_9'] == 0, 0,
np.where(base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_48_gh29_9'] == 1, 1,
 0))
base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_48_gh29_11'] = np.where(base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_48_gh29_10'] == 0, 0,
np.where(base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_48_gh29_10'] == 1, 1,
 0))
base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_48_gh29_12'] = np.where(base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_48_gh29_11'] == 0, 0,
np.where(base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_48_gh29_11'] == 1, 1,
 0))
base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_48_gh29_13'] = np.where(base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_48_gh29_12'] == 0, 0,
np.where(base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_48_gh29_12'] == 1, 1,
 0))
base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_48_gh29_14'] = np.where(base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_48_gh29_13'] == 0, 0,
np.where(base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_48_gh29_13'] == 1, 1,
 0))
base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_48_gh29'] = np.where(base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_48_gh29_14'] == 0, 0,
np.where(base_dados['DOCUMENTO_PESSOA__L__p_7_g_1_c1_48_gh29_14'] == 1, 1,
 0))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Mantendo apenas as variáveis do modelo

# COMMAND ----------

import pickle
modelo=pickle.load(open(model,'rb'))

base_teste_c0 = base_dados[[chave,'DOCUMENTO_PESSOA__L__p_7_g_1_c1_48_gh29', 'DETALHES_CONTRATOS_CLASSE_gh717_gh29','DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29','DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29','DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29']]

var_fin_c0=['DOCUMENTO_PESSOA__L__p_7_g_1_c1_48_gh29', 'DETALHES_CONTRATOS_CLASSE_gh717_gh29','DETALHES_CONTRATOS_BLOQUEIO2_gh383_gh29','DETALHES_CLIENTES_VALOR_FATURA__R__pk_5_g_1_c1_55_gh29','DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_c1_94_gh29']

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


x_teste2['P_1_L'] = np.log(x_teste2['P_1'])
x_teste2['P_1_L'] = np.where(x_teste2['P_1'] == 0, -1, x_teste2['P_1_L'])
x_teste2['P_1_L'] = np.where(x_teste2['P_1'] == np.nan, -2, x_teste2['P_1_L'])
x_teste2['P_1_L'] = x_teste2['P_1_L'].fillna(-2)


x_teste2['P_1_p_40_g_1'] = np.where(x_teste2['P_1'] <= 0.077393865, 0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.077393865, x_teste2['P_1'] <= 0.120501646), 1,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.120501646, x_teste2['P_1'] <= 0.187714758), 2,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.187714758, x_teste2['P_1'] <= 0.270517013), 3,4))))

x_teste2['P_1_L_pu_25_g_1'] = np.where(x_teste2['P_1_L'] <= -1.78250155, 0,
    np.where(np.bitwise_and(x_teste2['P_1_L'] > -1.78250155, x_teste2['P_1_L'] <= -1.37186481), 1,
    np.where(np.bitwise_and(x_teste2['P_1_L'] > -1.37186481, x_teste2['P_1_L'] <= -1.231926227), 2,3)))

x_teste2['GH'] = np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 0, x_teste2['P_1_L_pu_25_g_1'] == 0), 0,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 0, x_teste2['P_1_L_pu_25_g_1'] == 1), 0,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 0, x_teste2['P_1_L_pu_25_g_1'] == 2), 0,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 0, x_teste2['P_1_L_pu_25_g_1'] == 3), 1,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 1, x_teste2['P_1_L_pu_25_g_1'] == 0), 1,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 1, x_teste2['P_1_L_pu_25_g_1'] == 1), 1,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 1, x_teste2['P_1_L_pu_25_g_1'] == 2), 1,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 1, x_teste2['P_1_L_pu_25_g_1'] == 3), 2,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 2, x_teste2['P_1_L_pu_25_g_1'] == 0), 2,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 2, x_teste2['P_1_L_pu_25_g_1'] == 1), 2,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 2, x_teste2['P_1_L_pu_25_g_1'] == 2), 2,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 2, x_teste2['P_1_L_pu_25_g_1'] == 3), 3,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 3, x_teste2['P_1_L_pu_25_g_1'] == 0), 3,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 3, x_teste2['P_1_L_pu_25_g_1'] == 1), 3,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 3, x_teste2['P_1_L_pu_25_g_1'] == 2), 4,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 3, x_teste2['P_1_L_pu_25_g_1'] == 3), 4,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 4, x_teste2['P_1_L_pu_25_g_1'] == 0), 4,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 4, x_teste2['P_1_L_pu_25_g_1'] == 1), 4,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 4, x_teste2['P_1_L_pu_25_g_1'] == 2), 4,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 4, x_teste2['P_1_L_pu_25_g_1'] == 3), 5,
             2))))))))))))))))))))

del x_teste2['P_1_L']
del x_teste2['P_1_p_40_g_1']
del x_teste2['P_1_L_pu_25_g_1']

x_teste2


# COMMAND ----------

x_teste2.groupby(['GH'])['P_1'].count()

# COMMAND ----------

try:
  dbutils.fs.rm(outputpath, True)
except:
  pass
dbutils.fs.mkdirs(outputpath)

x_teste2.to_csv(open(os.path.join(outputpath_dbfs, 'pre_output:'+data_arquivo+'.csv'),'wb'))
os.path.join(outputpath_dbfs, 'pre_output:'+data_arquivo+'.csv')