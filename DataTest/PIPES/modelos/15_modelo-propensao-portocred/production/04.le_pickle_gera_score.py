# Databricks notebook source
# MAGIC %md
# MAGIC # <font color='blue'>IA - Feature Selection</font>
# MAGIC 
# MAGIC # <font color='blue'>Ferramenta de Criação de Variáveis</font>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importando os pacotes Python

# COMMAND ----------

# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

# Imports
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline

# COMMAND ----------

blob_account_source_ml = "qqdatastoragemain"
blob_container_source_ml = "ml-prd"

mount_blob_storage_key(dbutils,blob_account_source_ml,blob_account_source_ml,'/mnt/ml-prd')

readpath_stage = '/mnt/ml-prd/ml-data/propensaodeal/portocred/stage'
readpath_stage_dbfs = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/portocred/stage'
writepath_trusted = '/mnt/ml-prd/ml-data/propensaodeal/portocred/trusted'
list_readpath = os.listdir(readpath_stage_dbfs)

writepath_sample = '/mnt/ml-prd/ml-data/propensaodeal/portocred/sample'

spark.conf.set('spark.sql.sources.partitionOverwriteMode', 'dynamic')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inserindo hiperparâmetros do Algoritmo

# COMMAND ----------

## Parâmetros do Algoritmo

#Variável chave-primaria
chave = 'ID_PESSOA:ID_DIVIDA'

#Nome da Base de Dados
N_Base = "trustedFile_portocred.csv"
nm_base = "trustedFile_portocred"

#Caminho da base de dados
caminho_base = "/dbfs/mnt/ml-prd/ml-data/propensaodeal/portocred/trusted/"
list_base = os.listdir(caminho_base)
dt_max = max(list_base)
caminho_base = caminho_base+dt_max+"/"

caminho_base_pickle = "/dbfs/mnt/ml-prd/ml-data/propensaodeal/portocred/pickle_model/"
outputpath_dbfs = "/dbfs/mnt/ml-prd/ml-data/propensaodeal/portocred/output"
outputpath = "/mnt/ml-prd/ml-data/propensaodeal/portocred/output"

#Separador
separador_ = ";"

#Decimal
decimal_ = "."

print(caminho_base)
print(caminho_base_pickle)
print(nm_base)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importação da Base de Dados

# COMMAND ----------

base_dados = pd.read_csv(caminho_base+N_Base, sep=separador_, decimal=decimal_)
base_dados['ID_PESSOA:ID_DIVIDA']=base_dados['ID_PESSOA:ID_DIVIDA']+'_'+base_dados['DOCUMENTO_PESSOA'].astype(str)
base_dados = base_dados[[chave,'TIPO_EMAIL','PLANO','DETALHES_CONTRATOS_SALDO_ABERTO','DETALHES_CLIENTES_CONTA_REF_BANCARIA3','DETALHES_CONTRATOS_EMISSAO','ID_CONTRATO','TIPO_ENDERECO','DETALHES_CONTRATOS_TOTAL_PAGO']]

base_dados.fillna(-3)

#string
base_dados['TIPO_ENDERECO'] = base_dados['TIPO_ENDERECO'].replace(np.nan, '-3')
base_dados['TIPO_EMAIL'] = base_dados['TIPO_EMAIL'].replace(np.nan, '-3')

#numericas
base_dados['PLANO'] = base_dados['PLANO'].replace(np.nan, '-3')
base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3'] = base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3'].replace(np.nan, '-3')

base_dados['ID_CONTRATO'] = base_dados['ID_CONTRATO'].replace(np.nan, '-3')
base_dados['DETALHES_CONTRATOS_TOTAL_PAGO'] = base_dados['DETALHES_CONTRATOS_TOTAL_PAGO'].replace(np.nan, '-3')
base_dados['DETALHES_CONTRATOS_SALDO_ABERTO'] = base_dados['DETALHES_CONTRATOS_SALDO_ABERTO'].replace(np.nan, '-3')

base_dados = base_dados.apply(pd.to_numeric, errors='ignore')

base_dados['PLANO'] = base_dados['PLANO'].astype(np.int64)
base_dados['ID_CONTRATO'] = base_dados['ID_CONTRATO'].astype(np.int64)
base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3'] = base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3'].astype(np.int64)
base_dados['DETALHES_CONTRATOS_TOTAL_PAGO'] = base_dados['DETALHES_CONTRATOS_TOTAL_PAGO'].astype(np.int64)
base_dados['DETALHES_CONTRATOS_SALDO_ABERTO'] = base_dados['DETALHES_CONTRATOS_SALDO_ABERTO'].astype(np.int64)

base_dados['DETALHES_CONTRATOS_EMISSAO'] = pd.to_datetime(base_dados['DETALHES_CONTRATOS_EMISSAO'])

base_dados['mob_contrato'] = ((datetime.today()) - base_dados.DETALHES_CONTRATOS_EMISSAO)/np.timedelta64(1, 'M')

base_dados.drop_duplicates(keep='first', inplace=True)

del base_dados['DETALHES_CONTRATOS_EMISSAO']

print("shape da Base de Dados:",base_dados.shape)

base_dados.head()



# COMMAND ----------

# MAGIC %md
# MAGIC ## Gerando variáveis Categóricas

# COMMAND ----------


base_dados['TIPO_ENDERECO_gh30'] = np.where(base_dados['TIPO_ENDERECO'] == '-3', 0,
np.where(base_dados['TIPO_ENDERECO'] == '[0]', 1,
np.where(base_dados['TIPO_ENDERECO'] == '[1, 0]', 2,
np.where(base_dados['TIPO_ENDERECO'] == '[1, 2, 0]', 3,
np.where(base_dados['TIPO_ENDERECO'] == '[1]', 4,
np.where(base_dados['TIPO_ENDERECO'] == '[3, 0]', 5,
np.where(base_dados['TIPO_ENDERECO'] == '[3]', 6,
0)))))))

base_dados['TIPO_ENDERECO_gh31'] = np.where(base_dados['TIPO_ENDERECO_gh30'] == 0, 0,
np.where(base_dados['TIPO_ENDERECO_gh30'] == 1, 1,
np.where(base_dados['TIPO_ENDERECO_gh30'] == 2, 2,
np.where(base_dados['TIPO_ENDERECO_gh30'] == 3, 3,
np.where(base_dados['TIPO_ENDERECO_gh30'] == 4, 3,
np.where(base_dados['TIPO_ENDERECO_gh30'] == 5, 3,
np.where(base_dados['TIPO_ENDERECO_gh30'] == 6, 3,
0)))))))

base_dados['TIPO_ENDERECO_gh32'] = np.where(base_dados['TIPO_ENDERECO_gh31'] == 0, 0,
np.where(base_dados['TIPO_ENDERECO_gh31'] == 1, 1,
np.where(base_dados['TIPO_ENDERECO_gh31'] == 2, 2,
np.where(base_dados['TIPO_ENDERECO_gh31'] == 3, 3,
0))))

base_dados['TIPO_ENDERECO_gh33'] = np.where(base_dados['TIPO_ENDERECO_gh32'] == 0, 0,
np.where(base_dados['TIPO_ENDERECO_gh32'] == 1, 1,
np.where(base_dados['TIPO_ENDERECO_gh32'] == 2, 2,
np.where(base_dados['TIPO_ENDERECO_gh32'] == 3, 3,
0))))

base_dados['TIPO_ENDERECO_gh34'] = np.where(base_dados['TIPO_ENDERECO_gh33'] == 0, 1,
np.where(base_dados['TIPO_ENDERECO_gh33'] == 1, 1,
np.where(base_dados['TIPO_ENDERECO_gh33'] == 2, 2,
np.where(base_dados['TIPO_ENDERECO_gh33'] == 3, 4,
0))))

base_dados['TIPO_ENDERECO_gh35'] = np.where(base_dados['TIPO_ENDERECO_gh34'] == 1, 0,
np.where(base_dados['TIPO_ENDERECO_gh34'] == 2, 1,
np.where(base_dados['TIPO_ENDERECO_gh34'] == 4, 2,
0)))

base_dados['TIPO_ENDERECO_gh36'] = np.where(base_dados['TIPO_ENDERECO_gh35'] == 0, 1,
np.where(base_dados['TIPO_ENDERECO_gh35'] == 1, 2,
np.where(base_dados['TIPO_ENDERECO_gh35'] == 2, 0,
0)))

base_dados['TIPO_ENDERECO_gh37'] = np.where(base_dados['TIPO_ENDERECO_gh36'] == 0, 1,
np.where(base_dados['TIPO_ENDERECO_gh36'] == 1, 1,
np.where(base_dados['TIPO_ENDERECO_gh36'] == 2, 2,
0)))

base_dados['TIPO_ENDERECO_gh38'] = np.where(base_dados['TIPO_ENDERECO_gh37'] == 1, 0,
np.where(base_dados['TIPO_ENDERECO_gh37'] == 2, 1,
0))

                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
base_dados['TIPO_EMAIL_gh30'] = np.where(base_dados['TIPO_EMAIL'] == '-3', 0,
np.where(base_dados['TIPO_EMAIL'] == '[0]', 1,
np.where(base_dados['TIPO_EMAIL'] == '[1, 0]', 2,
np.where(base_dados['TIPO_EMAIL'] == '[1]', 3,
np.where(base_dados['TIPO_EMAIL'] == '[2, 0]', 4,
np.where(base_dados['TIPO_EMAIL'] == '[2]', 5,
np.where(base_dados['TIPO_EMAIL'] == '[3, 0]', 6,
np.where(base_dados['TIPO_EMAIL'] == '[3, 2, 0]', 7,
np.where(base_dados['TIPO_EMAIL'] == '[3]', 8,
0)))))))))

base_dados['TIPO_EMAIL_gh31'] = np.where(base_dados['TIPO_EMAIL_gh30'] == 0, 0,
np.where(base_dados['TIPO_EMAIL_gh30'] == 1, 1,
np.where(base_dados['TIPO_EMAIL_gh30'] == 2, 2,
np.where(base_dados['TIPO_EMAIL_gh30'] == 3, 3,
np.where(base_dados['TIPO_EMAIL_gh30'] == 4, 4,
np.where(base_dados['TIPO_EMAIL_gh30'] == 5, 5,
np.where(base_dados['TIPO_EMAIL_gh30'] == 6, 6,
np.where(base_dados['TIPO_EMAIL_gh30'] == 7, 7,
np.where(base_dados['TIPO_EMAIL_gh30'] == 8, 8,
0)))))))))

base_dados['TIPO_EMAIL_gh32'] = np.where(base_dados['TIPO_EMAIL_gh31'] == 0, 0,
np.where(base_dados['TIPO_EMAIL_gh31'] == 1, 1,
np.where(base_dados['TIPO_EMAIL_gh31'] == 2, 2,
np.where(base_dados['TIPO_EMAIL_gh31'] == 3, 3,
np.where(base_dados['TIPO_EMAIL_gh31'] == 4, 4,
np.where(base_dados['TIPO_EMAIL_gh31'] == 5, 5,
np.where(base_dados['TIPO_EMAIL_gh31'] == 6, 6,
np.where(base_dados['TIPO_EMAIL_gh31'] == 7, 7,
np.where(base_dados['TIPO_EMAIL_gh31'] == 8, 8,
0)))))))))

base_dados['TIPO_EMAIL_gh33'] = np.where(base_dados['TIPO_EMAIL_gh32'] == 0, 0,
np.where(base_dados['TIPO_EMAIL_gh32'] == 1, 1,
np.where(base_dados['TIPO_EMAIL_gh32'] == 2, 2,
np.where(base_dados['TIPO_EMAIL_gh32'] == 3, 3,
np.where(base_dados['TIPO_EMAIL_gh32'] == 4, 4,
np.where(base_dados['TIPO_EMAIL_gh32'] == 5, 5,
np.where(base_dados['TIPO_EMAIL_gh32'] == 6, 6,
np.where(base_dados['TIPO_EMAIL_gh32'] == 7, 7,
np.where(base_dados['TIPO_EMAIL_gh32'] == 8, 8,
0)))))))))

base_dados['TIPO_EMAIL_gh34'] = np.where(base_dados['TIPO_EMAIL_gh33'] == 0, 0,
np.where(base_dados['TIPO_EMAIL_gh33'] == 1, 1,
np.where(base_dados['TIPO_EMAIL_gh33'] == 2, 2,
np.where(base_dados['TIPO_EMAIL_gh33'] == 3, 0,
np.where(base_dados['TIPO_EMAIL_gh33'] == 4, 1,
np.where(base_dados['TIPO_EMAIL_gh33'] == 5, 0,
np.where(base_dados['TIPO_EMAIL_gh33'] == 6, 1,
np.where(base_dados['TIPO_EMAIL_gh33'] == 7, 2,
np.where(base_dados['TIPO_EMAIL_gh33'] == 8, 0,
0)))))))))

base_dados['TIPO_EMAIL_gh35'] = np.where(base_dados['TIPO_EMAIL_gh34'] == 0, 0,
np.where(base_dados['TIPO_EMAIL_gh34'] == 1, 1,
np.where(base_dados['TIPO_EMAIL_gh34'] == 2, 2,
0)))

base_dados['TIPO_EMAIL_gh36'] = np.where(base_dados['TIPO_EMAIL_gh35'] == 0, 1,
np.where(base_dados['TIPO_EMAIL_gh35'] == 1, 2,
np.where(base_dados['TIPO_EMAIL_gh35'] == 2, 0,
0)))

base_dados['TIPO_EMAIL_gh37'] = np.where(base_dados['TIPO_EMAIL_gh36'] == 0, 1,
np.where(base_dados['TIPO_EMAIL_gh36'] == 1, 1,
np.where(base_dados['TIPO_EMAIL_gh36'] == 2, 2,
0)))

base_dados['TIPO_EMAIL_gh38'] = np.where(base_dados['TIPO_EMAIL_gh37'] == 1, 0,
np.where(base_dados['TIPO_EMAIL_gh37'] == 2, 1,
0))
                                                                         
                                                                         
                                                                         
                                                                         
                                                                         
                                                                         
                                                                         
                                                                         
base_dados['PLANO_gh30'] = np.where(base_dados['PLANO'] == 1, 0,
np.where(base_dados['PLANO'] == 2, 1,
np.where(base_dados['PLANO'] == 3, 2,
np.where(base_dados['PLANO'] == 4, 3,
np.where(base_dados['PLANO'] == 5, 4,
np.where(base_dados['PLANO'] == 6, 5,
np.where(base_dados['PLANO'] == 7, 6,
np.where(base_dados['PLANO'] == 8, 7,
np.where(base_dados['PLANO'] == 9, 8,
np.where(base_dados['PLANO'] == 10, 9,
np.where(base_dados['PLANO'] == 11, 10,
np.where(base_dados['PLANO'] == 12, 11,
np.where(base_dados['PLANO'] == 13, 12,
np.where(base_dados['PLANO'] == 14, 13,
np.where(base_dados['PLANO'] == 15, 14,
np.where(base_dados['PLANO'] == 16, 15,
np.where(base_dados['PLANO'] == 17, 16,
np.where(base_dados['PLANO'] == 18, 17,
np.where(base_dados['PLANO'] == 19, 18,
np.where(base_dados['PLANO'] == 20, 19,
np.where(base_dados['PLANO'] == 21, 20,
np.where(base_dados['PLANO'] == 22, 21,
np.where(base_dados['PLANO'] == 23, 22,
np.where(base_dados['PLANO'] == 24, 23,
np.where(base_dados['PLANO'] == 25, 24,
np.where(base_dados['PLANO'] == 26, 25,
np.where(base_dados['PLANO'] == 28, 26,
np.where(base_dados['PLANO'] == 30, 27,
np.where(base_dados['PLANO'] == 36, 28,
np.where(base_dados['PLANO'] == 40, 29,
np.where(base_dados['PLANO'] == 42, 30,
np.where(base_dados['PLANO'] == 43, 31,
np.where(base_dados['PLANO'] == 44, 32,
np.where(base_dados['PLANO'] == 46, 33,
np.where(base_dados['PLANO'] == 48, 34,
np.where(base_dados['PLANO'] == 49, 35,
np.where(base_dados['PLANO'] == 60, 36,
np.where(base_dados['PLANO'] == 72, 37,
np.where(base_dados['PLANO'] == 84, 38,
np.where(base_dados['PLANO'] == 96, 39,
0))))))))))))))))))))))))))))))))))))))))

base_dados['PLANO_gh31'] = np.where(base_dados['PLANO_gh30'] == 0, 0,
np.where(base_dados['PLANO_gh30'] == 1, 0,
np.where(base_dados['PLANO_gh30'] == 2, 0,
np.where(base_dados['PLANO_gh30'] == 3, 0,
np.where(base_dados['PLANO_gh30'] == 4, 4,
np.where(base_dados['PLANO_gh30'] == 5, 5,
np.where(base_dados['PLANO_gh30'] == 6, 6,
np.where(base_dados['PLANO_gh30'] == 7, 7,
np.where(base_dados['PLANO_gh30'] == 8, 8,
np.where(base_dados['PLANO_gh30'] == 9, 9,
np.where(base_dados['PLANO_gh30'] == 10, 10,
np.where(base_dados['PLANO_gh30'] == 11, 10,
np.where(base_dados['PLANO_gh30'] == 12, 12,
np.where(base_dados['PLANO_gh30'] == 13, 13,
np.where(base_dados['PLANO_gh30'] == 14, 14,
np.where(base_dados['PLANO_gh30'] == 15, 15,
np.where(base_dados['PLANO_gh30'] == 16, 16,
np.where(base_dados['PLANO_gh30'] == 17, 17,
np.where(base_dados['PLANO_gh30'] == 18, 18,
np.where(base_dados['PLANO_gh30'] == 19, 19,
np.where(base_dados['PLANO_gh30'] == 20, 20,
np.where(base_dados['PLANO_gh30'] == 21, 21,
np.where(base_dados['PLANO_gh30'] == 22, 22,
np.where(base_dados['PLANO_gh30'] == 23, 23,
np.where(base_dados['PLANO_gh30'] == 24, 24,
np.where(base_dados['PLANO_gh30'] == 25, 25,
np.where(base_dados['PLANO_gh30'] == 26, 26,
np.where(base_dados['PLANO_gh30'] == 27, 27,
np.where(base_dados['PLANO_gh30'] == 28, 27,
np.where(base_dados['PLANO_gh30'] == 29, 29,
np.where(base_dados['PLANO_gh30'] == 30, 29,
np.where(base_dados['PLANO_gh30'] == 31, 31,
np.where(base_dados['PLANO_gh30'] == 32, 31,
np.where(base_dados['PLANO_gh30'] == 33, 33,
np.where(base_dados['PLANO_gh30'] == 34, 34,
np.where(base_dados['PLANO_gh30'] == 35, 35,
np.where(base_dados['PLANO_gh30'] == 36, 36,
np.where(base_dados['PLANO_gh30'] == 37, 37,
np.where(base_dados['PLANO_gh30'] == 38, 38,
np.where(base_dados['PLANO_gh30'] == 39, 39,
0))))))))))))))))))))))))))))))))))))))))

base_dados['PLANO_gh32'] = np.where(base_dados['PLANO_gh31'] == 0, 0,
np.where(base_dados['PLANO_gh31'] == 4, 1,
np.where(base_dados['PLANO_gh31'] == 5, 2,
np.where(base_dados['PLANO_gh31'] == 6, 3,
np.where(base_dados['PLANO_gh31'] == 7, 4,
np.where(base_dados['PLANO_gh31'] == 8, 5,
np.where(base_dados['PLANO_gh31'] == 9, 6,
np.where(base_dados['PLANO_gh31'] == 10, 7,
np.where(base_dados['PLANO_gh31'] == 12, 8,
np.where(base_dados['PLANO_gh31'] == 13, 9,
np.where(base_dados['PLANO_gh31'] == 14, 10,
np.where(base_dados['PLANO_gh31'] == 15, 11,
np.where(base_dados['PLANO_gh31'] == 16, 12,
np.where(base_dados['PLANO_gh31'] == 17, 13,
np.where(base_dados['PLANO_gh31'] == 18, 14,
np.where(base_dados['PLANO_gh31'] == 19, 15,
np.where(base_dados['PLANO_gh31'] == 20, 16,
np.where(base_dados['PLANO_gh31'] == 21, 17,
np.where(base_dados['PLANO_gh31'] == 22, 18,
np.where(base_dados['PLANO_gh31'] == 23, 19,
np.where(base_dados['PLANO_gh31'] == 24, 20,
np.where(base_dados['PLANO_gh31'] == 25, 21,
np.where(base_dados['PLANO_gh31'] == 26, 22,
np.where(base_dados['PLANO_gh31'] == 27, 23,
np.where(base_dados['PLANO_gh31'] == 29, 24,
np.where(base_dados['PLANO_gh31'] == 31, 25,
np.where(base_dados['PLANO_gh31'] == 33, 26,
np.where(base_dados['PLANO_gh31'] == 34, 27,
np.where(base_dados['PLANO_gh31'] == 35, 28,
np.where(base_dados['PLANO_gh31'] == 36, 29,
np.where(base_dados['PLANO_gh31'] == 37, 30,
np.where(base_dados['PLANO_gh31'] == 38, 31,
np.where(base_dados['PLANO_gh31'] == 39, 32,
0)))))))))))))))))))))))))))))))))

base_dados['PLANO_gh33'] = np.where(base_dados['PLANO_gh32'] == 0, 0,
np.where(base_dados['PLANO_gh32'] == 1, 1,
np.where(base_dados['PLANO_gh32'] == 2, 2,
np.where(base_dados['PLANO_gh32'] == 3, 3,
np.where(base_dados['PLANO_gh32'] == 4, 4,
np.where(base_dados['PLANO_gh32'] == 5, 5,
np.where(base_dados['PLANO_gh32'] == 6, 6,
np.where(base_dados['PLANO_gh32'] == 7, 7,
np.where(base_dados['PLANO_gh32'] == 8, 8,
np.where(base_dados['PLANO_gh32'] == 9, 9,
np.where(base_dados['PLANO_gh32'] == 10, 10,
np.where(base_dados['PLANO_gh32'] == 11, 11,
np.where(base_dados['PLANO_gh32'] == 12, 12,
np.where(base_dados['PLANO_gh32'] == 13, 13,
np.where(base_dados['PLANO_gh32'] == 14, 14,
np.where(base_dados['PLANO_gh32'] == 15, 15,
np.where(base_dados['PLANO_gh32'] == 16, 16,
np.where(base_dados['PLANO_gh32'] == 17, 17,
np.where(base_dados['PLANO_gh32'] == 18, 18,
np.where(base_dados['PLANO_gh32'] == 19, 19,
np.where(base_dados['PLANO_gh32'] == 20, 20,
np.where(base_dados['PLANO_gh32'] == 21, 21,
np.where(base_dados['PLANO_gh32'] == 22, 22,
np.where(base_dados['PLANO_gh32'] == 23, 23,
np.where(base_dados['PLANO_gh32'] == 24, 24,
np.where(base_dados['PLANO_gh32'] == 25, 25,
np.where(base_dados['PLANO_gh32'] == 26, 26,
np.where(base_dados['PLANO_gh32'] == 27, 27,
np.where(base_dados['PLANO_gh32'] == 28, 28,
np.where(base_dados['PLANO_gh32'] == 29, 29,
np.where(base_dados['PLANO_gh32'] == 30, 30,
np.where(base_dados['PLANO_gh32'] == 31, 31,
np.where(base_dados['PLANO_gh32'] == 32, 32,
0)))))))))))))))))))))))))))))))))

base_dados['PLANO_gh34'] = np.where(base_dados['PLANO_gh33'] == 0, 0,
np.where(base_dados['PLANO_gh33'] == 1, 13,
np.where(base_dados['PLANO_gh33'] == 2, 27,
np.where(base_dados['PLANO_gh33'] == 3, 7,
np.where(base_dados['PLANO_gh33'] == 4, 6,
np.where(base_dados['PLANO_gh33'] == 5, 7,
np.where(base_dados['PLANO_gh33'] == 6, 6,
np.where(base_dados['PLANO_gh33'] == 7, 7,
np.where(base_dados['PLANO_gh33'] == 8, 13,
np.where(base_dados['PLANO_gh33'] == 9, 27,
np.where(base_dados['PLANO_gh33'] == 10, 10,
np.where(base_dados['PLANO_gh33'] == 11, 11,
np.where(base_dados['PLANO_gh33'] == 12, 32,
np.where(base_dados['PLANO_gh33'] == 13, 13,
np.where(base_dados['PLANO_gh33'] == 14, 0,
np.where(base_dados['PLANO_gh33'] == 15, 27,
np.where(base_dados['PLANO_gh33'] == 16, 19,
np.where(base_dados['PLANO_gh33'] == 17, 19,
np.where(base_dados['PLANO_gh33'] == 18, 19,
np.where(base_dados['PLANO_gh33'] == 19, 19,
np.where(base_dados['PLANO_gh33'] == 20, 27,
np.where(base_dados['PLANO_gh33'] == 21, 19,
np.where(base_dados['PLANO_gh33'] == 22, 19,
np.where(base_dados['PLANO_gh33'] == 23, 23,
np.where(base_dados['PLANO_gh33'] == 24, 19,
np.where(base_dados['PLANO_gh33'] == 25, 0,
np.where(base_dados['PLANO_gh33'] == 26, 19,
np.where(base_dados['PLANO_gh33'] == 27, 27,
np.where(base_dados['PLANO_gh33'] == 28, 0,
np.where(base_dados['PLANO_gh33'] == 29, 32,
np.where(base_dados['PLANO_gh33'] == 30, 32,
np.where(base_dados['PLANO_gh33'] == 31, 0,
np.where(base_dados['PLANO_gh33'] == 32, 32,
0)))))))))))))))))))))))))))))))))

base_dados['PLANO_gh35'] = np.where(base_dados['PLANO_gh34'] == 0, 0,
np.where(base_dados['PLANO_gh34'] == 6, 1,
np.where(base_dados['PLANO_gh34'] == 7, 2,
np.where(base_dados['PLANO_gh34'] == 10, 3,
np.where(base_dados['PLANO_gh34'] == 11, 4,
np.where(base_dados['PLANO_gh34'] == 13, 5,
np.where(base_dados['PLANO_gh34'] == 19, 6,
np.where(base_dados['PLANO_gh34'] == 23, 7,
np.where(base_dados['PLANO_gh34'] == 27, 8,
np.where(base_dados['PLANO_gh34'] == 32, 9,
0))))))))))

base_dados['PLANO_gh36'] = np.where(base_dados['PLANO_gh35'] == 0, 0,
np.where(base_dados['PLANO_gh35'] == 1, 3,
np.where(base_dados['PLANO_gh35'] == 2, 5,
np.where(base_dados['PLANO_gh35'] == 3, 7,
np.where(base_dados['PLANO_gh35'] == 4, 2,
np.where(base_dados['PLANO_gh35'] == 5, 7,
np.where(base_dados['PLANO_gh35'] == 6, 9,
np.where(base_dados['PLANO_gh35'] == 7, 5,
np.where(base_dados['PLANO_gh35'] == 8, 4,
np.where(base_dados['PLANO_gh35'] == 9, 1,
0))))))))))

base_dados['PLANO_gh37'] = np.where(base_dados['PLANO_gh36'] == 0, 1,
np.where(base_dados['PLANO_gh36'] == 1, 1,
np.where(base_dados['PLANO_gh36'] == 2, 1,
np.where(base_dados['PLANO_gh36'] == 3, 1,
np.where(base_dados['PLANO_gh36'] == 4, 1,
np.where(base_dados['PLANO_gh36'] == 5, 5,
np.where(base_dados['PLANO_gh36'] == 7, 6,
np.where(base_dados['PLANO_gh36'] == 9, 7,
0))))))))

base_dados['PLANO_gh38'] = np.where(base_dados['PLANO_gh37'] == 1, 0,
np.where(base_dados['PLANO_gh37'] == 5, 1,
np.where(base_dados['PLANO_gh37'] == 6, 2,
np.where(base_dados['PLANO_gh37'] == 7, 3,
0))))
         
         
         
         
         
         
         
         
         
base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh30'] = np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3'] == -3, 0,
np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3'] == 2, 1,
np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3'] == 182992, 2,
0)))

base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh31'] = np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh30'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh30'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh30'] == 2, 2,
0)))

base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh32'] = np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh31'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh31'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh31'] == 2, 2,
0)))

base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh33'] = np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh32'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh32'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh32'] == 2, 2,
0)))

base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh34'] = np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh33'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh33'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh33'] == 2, 1,
0)))

base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh35'] = np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh34'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh34'] == 1, 1,
0))

base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh36'] = np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh35'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh35'] == 1, 1,
0))

base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh37'] = np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh36'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh36'] == 1, 1,
0))

base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh38'] = np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh37'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh37'] == 1, 1,
0))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gerando variáveis numéricas contínuas

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parte 1 de 2

# COMMAND ----------

       
base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__p_7'] = np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO'] <= 753.0, 0.0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO'] > 753.0, base_dados['DETALHES_CONTRATOS_SALDO_ABERTO'] <= 1264.0), 1.0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO'] > 1264.0, base_dados['DETALHES_CONTRATOS_SALDO_ABERTO'] <= 1880.0), 2.0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO'] > 1880.0, base_dados['DETALHES_CONTRATOS_SALDO_ABERTO'] <= 2654.0), 3.0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO'] > 2654.0, base_dados['DETALHES_CONTRATOS_SALDO_ABERTO'] <= 4302.0), 4.0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO'] > 4302.0, base_dados['DETALHES_CONTRATOS_SALDO_ABERTO'] <= 7249.0), 5.0,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO'] > 7249.0, 6.0,
 0)))))))

base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__p_7_g_1_1'] = np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__p_7'] == 0.0, 0,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__p_7'] == 1.0, 1,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__p_7'] == 2.0, 0,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__p_7'] == 3.0, 1,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__p_7'] == 4.0, 1,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__p_7'] == 5.0, 1,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__p_7'] == 6.0, 1,
 0)))))))

base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__p_7_g_1_2'] = np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__p_7_g_1_1'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__p_7_g_1_1'] == 1, 1,
 0))

base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__p_7_g_1'] = np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__p_7_g_1_2'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__p_7_g_1_2'] == 1, 1,
 0))
                                                                  
                                                                  
                                                                  
                                                                  
                                                                  
                                                                  
                                                                  
                                                                  
base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L'] = np.log(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO'])
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L'] == 0, -1, base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L'])
base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L'] = base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L'].fillna(-2)
base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__pu_3'] = np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L'] <= 6.3578422665081, 0.0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L'] > 6.3578422665081, base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L'] <= 10.144078017454058), 1.0,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L'] > 10.144078017454058, 2.0,
 0)))

base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__pu_3_g_1_1'] = np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__pu_3'] == 0.0, 0,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__pu_3'] == 1.0, 1,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__pu_3'] == 2.0, 1,
 0)))

base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__pu_3_g_1_2'] = np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__pu_3_g_1_1'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__pu_3_g_1_1'] == 1, 1,
 0))

base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__pu_3_g_1'] = np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__pu_3_g_1_2'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__pu_3_g_1_2'] == 1, 1,
 0))
                                                                      
                                                                      
                                                                      
                                                                      
                                                                      
                                                                      
                                                                      
                                                                      
                                                                      
                                                                      
base_dados['mob_contrato__pk_15'] = np.where(base_dados['mob_contrato'] <= 15.11745757685343, 0.0,
np.where(np.bitwise_and(base_dados['mob_contrato'] > 15.11745757685343, base_dados['mob_contrato'] <= 25.236761874682955), 1.0,
np.where(np.bitwise_and(base_dados['mob_contrato'] > 25.236761874682955, base_dados['mob_contrato'] <= 34.896097795338406), 2.0,
np.where(np.bitwise_and(base_dados['mob_contrato'] > 34.896097795338406, base_dados['mob_contrato'] <= 46.822420717780346), 3.0,
np.where(np.bitwise_and(base_dados['mob_contrato'] > 46.822420717780346, base_dados['mob_contrato'] <= 60.85145622158946), 4.0,
np.where(np.bitwise_and(base_dados['mob_contrato'] > 60.85145622158946, base_dados['mob_contrato'] <= 76.687510350011), 5.0,
np.where(np.bitwise_and(base_dados['mob_contrato'] > 76.687510350011, base_dados['mob_contrato'] <= 91.04509469465873), 6.0,
np.where(np.bitwise_and(base_dados['mob_contrato'] > 91.04509469465873, base_dados['mob_contrato'] <= 106.71687440266093), 7.0,
np.where(np.bitwise_and(base_dados['mob_contrato'] > 106.71687440266093, base_dados['mob_contrato'] <= 121.8301210812375), 8.0,
np.where(np.bitwise_and(base_dados['mob_contrato'] > 121.8301210812375, base_dados['mob_contrato'] <= 135.46489797604025), 9.0,
np.where(np.bitwise_and(base_dados['mob_contrato'] > 135.46489797604025, base_dados['mob_contrato'] <= 150.34816046602978), 10.0,
np.where(np.bitwise_and(base_dados['mob_contrato'] > 150.34816046602978, base_dados['mob_contrato'] <= 161.48596617045902), 11.0,
np.where(np.bitwise_and(base_dados['mob_contrato'] > 161.48596617045902, base_dados['mob_contrato'] <= 177.09203611029352), 12.0,
np.where(np.bitwise_and(base_dados['mob_contrato'] > 177.09203611029352, base_dados['mob_contrato'] <= 194.60368932699203), 13.0,
np.where(base_dados['mob_contrato'] > 194.60368932699203, 14.0,
 0)))))))))))))))

base_dados['mob_contrato__pk_15_g_1_1'] = np.where(base_dados['mob_contrato__pk_15'] == 0.0, 0,
np.where(base_dados['mob_contrato__pk_15'] == 1.0, 0,
np.where(base_dados['mob_contrato__pk_15'] == 2.0, 1,
np.where(base_dados['mob_contrato__pk_15'] == 3.0, 0,
np.where(base_dados['mob_contrato__pk_15'] == 4.0, 1,
np.where(base_dados['mob_contrato__pk_15'] == 5.0, 1,
np.where(base_dados['mob_contrato__pk_15'] == 6.0, 1,
np.where(base_dados['mob_contrato__pk_15'] == 7.0, 1,
np.where(base_dados['mob_contrato__pk_15'] == 8.0, 1,
np.where(base_dados['mob_contrato__pk_15'] == 9.0, 1,
np.where(base_dados['mob_contrato__pk_15'] == 10.0, 1,
np.where(base_dados['mob_contrato__pk_15'] == 11.0, 1,
np.where(base_dados['mob_contrato__pk_15'] == 12.0, 1,
np.where(base_dados['mob_contrato__pk_15'] == 13.0, 1,
np.where(base_dados['mob_contrato__pk_15'] == 14.0, 1,
 0)))))))))))))))

base_dados['mob_contrato__pk_15_g_1_2'] = np.where(base_dados['mob_contrato__pk_15_g_1_1'] == 0, 1,
np.where(base_dados['mob_contrato__pk_15_g_1_1'] == 1, 0,
 0))

base_dados['mob_contrato__pk_15_g_1'] = np.where(base_dados['mob_contrato__pk_15_g_1_2'] == 0, 0,
np.where(base_dados['mob_contrato__pk_15_g_1_2'] == 1, 1,
 0))
                                                 
                                                 
                                                 
                                                 
                                                 
                                                 
                                                 
                                                 
                                                 
                                                 
base_dados['mob_contrato__L'] = np.log(base_dados['mob_contrato'])
np.where(base_dados['mob_contrato__L'] == 0, -1, base_dados['mob_contrato__L'])
base_dados['mob_contrato__L'] = base_dados['mob_contrato__L'].fillna(-2)
base_dados['mob_contrato__L__pu_10'] = np.where(base_dados['mob_contrato__L'] <= 1.0631373266094342, 0.0,
np.where(np.bitwise_and(base_dados['mob_contrato__L'] > 1.0631373266094342, base_dados['mob_contrato__L'] <= 1.5340135936210655), 1.0,
np.where(np.bitwise_and(base_dados['mob_contrato__L'] > 1.5340135936210655, base_dados['mob_contrato__L'] <= 2.018626659649308), 2.0,
np.where(np.bitwise_and(base_dados['mob_contrato__L'] > 2.018626659649308, base_dados['mob_contrato__L'] <= 2.4927762895450822), 3.0,
np.where(np.bitwise_and(base_dados['mob_contrato__L'] > 2.4927762895450822, base_dados['mob_contrato__L'] <= 2.968068242309635), 4.0,
np.where(np.bitwise_and(base_dados['mob_contrato__L'] > 2.968068242309635, base_dados['mob_contrato__L'] <= 3.44409451535733), 5.0,
np.where(np.bitwise_and(base_dados['mob_contrato__L'] > 3.44409451535733, base_dados['mob_contrato__L'] <= 3.918758139700632), 6.0,
np.where(np.bitwise_and(base_dados['mob_contrato__L'] > 3.918758139700632, base_dados['mob_contrato__L'] <= 4.391907295532006), 7.0,
np.where(np.bitwise_and(base_dados['mob_contrato__L'] > 4.391907295532006, base_dados['mob_contrato__L'] <= 4.86913406671491), 8.0,
np.where(base_dados['mob_contrato__L'] > 4.86913406671491, 9.0,
 0))))))))))

base_dados['mob_contrato__L__pu_10_g_1_1'] = np.where(base_dados['mob_contrato__L__pu_10'] == 0.0, 4,
np.where(base_dados['mob_contrato__L__pu_10'] == 1.0, 4,
np.where(base_dados['mob_contrato__L__pu_10'] == 2.0, 4,
np.where(base_dados['mob_contrato__L__pu_10'] == 3.0, 4,
np.where(base_dados['mob_contrato__L__pu_10'] == 4.0, 1,
np.where(base_dados['mob_contrato__L__pu_10'] == 5.0, 2,
np.where(base_dados['mob_contrato__L__pu_10'] == 6.0, 0,
np.where(base_dados['mob_contrato__L__pu_10'] == 7.0, 3,
np.where(base_dados['mob_contrato__L__pu_10'] == 8.0, 5,
np.where(base_dados['mob_contrato__L__pu_10'] == 9.0, 5,
 0))))))))))

base_dados['mob_contrato__L__pu_10_g_1_2'] = np.where(base_dados['mob_contrato__L__pu_10_g_1_1'] == 0, 2,
np.where(base_dados['mob_contrato__L__pu_10_g_1_1'] == 1, 4,
np.where(base_dados['mob_contrato__L__pu_10_g_1_1'] == 2, 3,
np.where(base_dados['mob_contrato__L__pu_10_g_1_1'] == 3, 1,
np.where(base_dados['mob_contrato__L__pu_10_g_1_1'] == 4, 4,
np.where(base_dados['mob_contrato__L__pu_10_g_1_1'] == 5, 0,
 0))))))

base_dados['mob_contrato__L__pu_10_g_1'] = np.where(base_dados['mob_contrato__L__pu_10_g_1_2'] == 0, 0,
np.where(base_dados['mob_contrato__L__pu_10_g_1_2'] == 1, 1,
np.where(base_dados['mob_contrato__L__pu_10_g_1_2'] == 2, 2,
np.where(base_dados['mob_contrato__L__pu_10_g_1_2'] == 3, 3,
np.where(base_dados['mob_contrato__L__pu_10_g_1_2'] == 4, 4,
 0)))))
         
         
         
         
         
               
         
         
base_dados['ID_CONTRATO__pu_6'] = np.where(base_dados['ID_CONTRATO'] <= 147044.0, 0.0,
np.where(np.bitwise_and(base_dados['ID_CONTRATO'] > 147044.0, base_dados['ID_CONTRATO'] <= 224120.0), 1.0,
np.where(np.bitwise_and(base_dados['ID_CONTRATO'] > 224120.0, base_dados['ID_CONTRATO'] <= 300050.0), 2.0,
np.where(np.bitwise_and(base_dados['ID_CONTRATO'] > 300050.0, base_dados['ID_CONTRATO'] <= 376344.0), 3.0,
np.where(np.bitwise_and(base_dados['ID_CONTRATO'] > 376344.0, base_dados['ID_CONTRATO'] <= 452645.0), 4.0,
np.where(base_dados['ID_CONTRATO'] > 452645.0, 5.0,
 0))))))
base_dados['ID_CONTRATO__pu_6_g_1_1'] = np.where(base_dados['ID_CONTRATO__pu_6'] == 0.0, 0,
np.where(base_dados['ID_CONTRATO__pu_6'] == 1.0, 4,
np.where(base_dados['ID_CONTRATO__pu_6'] == 2.0, 3,
np.where(base_dados['ID_CONTRATO__pu_6'] == 3.0, 2,
np.where(base_dados['ID_CONTRATO__pu_6'] == 4.0, 4,
np.where(base_dados['ID_CONTRATO__pu_6'] == 5.0, 1,
 0))))))
base_dados['ID_CONTRATO__pu_6_g_1_2'] = np.where(base_dados['ID_CONTRATO__pu_6_g_1_1'] == 0, 0,
np.where(base_dados['ID_CONTRATO__pu_6_g_1_1'] == 1, 4,
np.where(base_dados['ID_CONTRATO__pu_6_g_1_1'] == 2, 2,
np.where(base_dados['ID_CONTRATO__pu_6_g_1_1'] == 3, 1,
np.where(base_dados['ID_CONTRATO__pu_6_g_1_1'] == 4, 3,
 0)))))
base_dados['ID_CONTRATO__pu_6_g_1'] = np.where(base_dados['ID_CONTRATO__pu_6_g_1_2'] == 0, 0,
np.where(base_dados['ID_CONTRATO__pu_6_g_1_2'] == 1, 1,
np.where(base_dados['ID_CONTRATO__pu_6_g_1_2'] == 2, 2,
np.where(base_dados['ID_CONTRATO__pu_6_g_1_2'] == 3, 3,
np.where(base_dados['ID_CONTRATO__pu_6_g_1_2'] == 4, 4,
 0)))))
         
         
         
         
         
         
         
         
         
         
base_dados['ID_CONTRATO__pk_7'] = np.where(base_dados['ID_CONTRATO'] <= 130407.0, 0.0,
np.where(np.bitwise_and(base_dados['ID_CONTRATO'] > 130407.0, base_dados['ID_CONTRATO'] <= 199464.0), 1.0,
np.where(np.bitwise_and(base_dados['ID_CONTRATO'] > 199464.0, base_dados['ID_CONTRATO'] <= 264786.0), 2.0,
np.where(np.bitwise_and(base_dados['ID_CONTRATO'] > 264786.0, base_dados['ID_CONTRATO'] <= 329595.0), 3.0,
np.where(np.bitwise_and(base_dados['ID_CONTRATO'] > 329595.0, base_dados['ID_CONTRATO'] <= 391816.0), 4.0,
np.where(np.bitwise_and(base_dados['ID_CONTRATO'] > 391816.0, base_dados['ID_CONTRATO'] <= 451322.0), 5.0,
np.where(base_dados['ID_CONTRATO'] > 451322.0, 6.0,
 0)))))))
base_dados['ID_CONTRATO__pk_7_g_1_1'] = np.where(base_dados['ID_CONTRATO__pk_7'] == 0.0, 0,
np.where(base_dados['ID_CONTRATO__pk_7'] == 1.0, 5,
np.where(base_dados['ID_CONTRATO__pk_7'] == 2.0, 3,
np.where(base_dados['ID_CONTRATO__pk_7'] == 3.0, 1,
np.where(base_dados['ID_CONTRATO__pk_7'] == 4.0, 2,
np.where(base_dados['ID_CONTRATO__pk_7'] == 5.0, 4,
np.where(base_dados['ID_CONTRATO__pk_7'] == 6.0, 1,
 0)))))))
base_dados['ID_CONTRATO__pk_7_g_1_2'] = np.where(base_dados['ID_CONTRATO__pk_7_g_1_1'] == 0, 0,
np.where(base_dados['ID_CONTRATO__pk_7_g_1_1'] == 1, 4,
np.where(base_dados['ID_CONTRATO__pk_7_g_1_1'] == 2, 2,
np.where(base_dados['ID_CONTRATO__pk_7_g_1_1'] == 3, 1,
np.where(base_dados['ID_CONTRATO__pk_7_g_1_1'] == 4, 4,
np.where(base_dados['ID_CONTRATO__pk_7_g_1_1'] == 5, 2,
 0))))))
base_dados['ID_CONTRATO__pk_7_g_1'] = np.where(base_dados['ID_CONTRATO__pk_7_g_1_2'] == 0, 0,
np.where(base_dados['ID_CONTRATO__pk_7_g_1_2'] == 1, 1,
np.where(base_dados['ID_CONTRATO__pk_7_g_1_2'] == 2, 2,
np.where(base_dados['ID_CONTRATO__pk_7_g_1_2'] == 4, 3,
 0))))
         
         
         
         
         
         
         
         
         
         
base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C'] = np.cos(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO'])
np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C'] == 0, -1, base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C'])
base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C'] = base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C'].fillna(-2)
base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C__pk_3'] = np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C'] <= -0.5402769400239504, 0.0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C'] > -0.5402769400239504, base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C'] <= 0.3089940590981371), 1.0,
np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C'] > 0.3089940590981371, 2.0,
 0)))
base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C__pk_3_g_1_1'] = np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C__pk_3'] == 0.0, 0,
np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C__pk_3'] == 1.0, 1,
np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C__pk_3'] == 2.0, 0,
 0)))
base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C__pk_3_g_1_2'] = np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C__pk_3_g_1_1'] == 0, 1,
np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C__pk_3_g_1_1'] == 1, 0,
 0))
base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C__pk_3_g_1'] = np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C__pk_3_g_1_2'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C__pk_3_g_1_2'] == 1, 1,
 0))
                                                                    
                                                                    
                                                                    
                                                                    
                                                                    
                                                                    
                                                                    
base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T'] = np.tan(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO'])
np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T'] == 0, -1, base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T'])
base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T'] = base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T'].fillna(-2)
base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_13'] = np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T'] >= -215.6680166481949, base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T'] <= 2.0408598529742332), 0.0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T'] > 2.0408598529742332, base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T'] <= 4.028840379766699), 1.0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T'] > 4.028840379766699, base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T'] <= 6.076049448296407), 2.0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T'] > 6.076049448296407, base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T'] <= 8.335103501827563), 3.0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T'] > 8.335103501827563, base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T'] <= 9.79298026353578), 4.0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T'] > 9.79298026353578, base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T'] <= 11.91999619729968), 5.0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T'] > 11.91999619729968, base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T'] <= 15.39897670346363), 7.0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T'] > 15.39897670346363, base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T'] <= 20.60154909340478), 9.0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T'] > 20.60154909340478, base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T'] <= 25.13061277103135), 11.0,
np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T'] > 25.13061277103135, 12.0,
 -2))))))))))

base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_13_g_1_1'] = np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_13'] == -2.0, 1,
np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_13'] == 0.0, 0,
np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_13'] == 1.0, 1,
np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_13'] == 2.0, 1,
np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_13'] == 3.0, 1,
np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_13'] == 4.0, 1,
np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_13'] == 5.0, 1,
np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_13'] == 7.0, 1,
np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_13'] == 9.0, 1,
np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_13'] == 11.0, 1,
np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_13'] == 12.0, 1,
 0)))))))))))
base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_13_g_1_2'] = np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_13_g_1_1'] == 0, 1,
np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_13_g_1_1'] == 1, 0,
 0))
base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_13_g_1'] = np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_13_g_1_2'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_13_g_1_2'] == 1, 1,
 0))


# COMMAND ----------

# MAGIC %md
# MAGIC ### Parte 2 de 2

# COMMAND ----------

      
base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__pu_3_g_1_c1_16_1'] = np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__p_7_g_1'] == 0, base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__pu_3_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__p_7_g_1'] == 0, base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__pu_3_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__p_7_g_1'] == 1, base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__pu_3_g_1'] == 0), 2,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__p_7_g_1'] == 1, base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__pu_3_g_1'] == 1), 2,
 0))))
base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__pu_3_g_1_c1_16_2'] = np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__pu_3_g_1_c1_16_1'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__pu_3_g_1_c1_16_1'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__pu_3_g_1_c1_16_1'] == 2, 2,
0)))
base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__pu_3_g_1_c1_16'] = np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__pu_3_g_1_c1_16_2'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__pu_3_g_1_c1_16_2'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__pu_3_g_1_c1_16_2'] == 2, 2,
 0)))
         
         
         
         
               
base_dados['mob_contrato__pk_15_g_1_c1_20_1'] = np.where(np.bitwise_and(base_dados['mob_contrato__pk_15_g_1'] == 0, base_dados['mob_contrato__L__pu_10_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['mob_contrato__pk_15_g_1'] == 0, base_dados['mob_contrato__L__pu_10_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['mob_contrato__pk_15_g_1'] == 0, base_dados['mob_contrato__L__pu_10_g_1'] == 2), 2,
np.where(np.bitwise_and(base_dados['mob_contrato__pk_15_g_1'] == 0, base_dados['mob_contrato__L__pu_10_g_1'] == 3), 2,
np.where(np.bitwise_and(base_dados['mob_contrato__pk_15_g_1'] == 0, base_dados['mob_contrato__L__pu_10_g_1'] == 4), 2,
np.where(np.bitwise_and(base_dados['mob_contrato__pk_15_g_1'] == 1, base_dados['mob_contrato__L__pu_10_g_1'] == 0), 2,
np.where(np.bitwise_and(base_dados['mob_contrato__pk_15_g_1'] == 1, base_dados['mob_contrato__L__pu_10_g_1'] == 1), 3,
np.where(np.bitwise_and(base_dados['mob_contrato__pk_15_g_1'] == 1, base_dados['mob_contrato__L__pu_10_g_1'] == 2), 3,
np.where(np.bitwise_and(base_dados['mob_contrato__pk_15_g_1'] == 1, base_dados['mob_contrato__L__pu_10_g_1'] == 3), 4,
np.where(np.bitwise_and(base_dados['mob_contrato__pk_15_g_1'] == 1, base_dados['mob_contrato__L__pu_10_g_1'] == 4), 5,
 0))))))))))
base_dados['mob_contrato__pk_15_g_1_c1_20_2'] = np.where(base_dados['mob_contrato__pk_15_g_1_c1_20_1'] == 0, 0,
np.where(base_dados['mob_contrato__pk_15_g_1_c1_20_1'] == 1, 1,
np.where(base_dados['mob_contrato__pk_15_g_1_c1_20_1'] == 2, 3,
np.where(base_dados['mob_contrato__pk_15_g_1_c1_20_1'] == 3, 2,
np.where(base_dados['mob_contrato__pk_15_g_1_c1_20_1'] == 4, 4,
np.where(base_dados['mob_contrato__pk_15_g_1_c1_20_1'] == 5, 5,
0))))))
base_dados['mob_contrato__pk_15_g_1_c1_20'] = np.where(base_dados['mob_contrato__pk_15_g_1_c1_20_2'] == 0, 0,
np.where(base_dados['mob_contrato__pk_15_g_1_c1_20_2'] == 1, 1,
np.where(base_dados['mob_contrato__pk_15_g_1_c1_20_2'] == 2, 2,
np.where(base_dados['mob_contrato__pk_15_g_1_c1_20_2'] == 3, 3,
np.where(base_dados['mob_contrato__pk_15_g_1_c1_20_2'] == 4, 4,
np.where(base_dados['mob_contrato__pk_15_g_1_c1_20_2'] == 5, 5,
 0))))))
         
         
         
         
         
         
         
base_dados['ID_CONTRATO__pk_7_g_1_c1_24_1'] = np.where(np.bitwise_and(base_dados['ID_CONTRATO__pu_6_g_1'] == 0, base_dados['ID_CONTRATO__pk_7_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['ID_CONTRATO__pu_6_g_1'] == 0, base_dados['ID_CONTRATO__pk_7_g_1'] == 1), 0,
np.where(np.bitwise_and(base_dados['ID_CONTRATO__pu_6_g_1'] == 0, base_dados['ID_CONTRATO__pk_7_g_1'] == 2), 1,
np.where(np.bitwise_and(base_dados['ID_CONTRATO__pu_6_g_1'] == 0, base_dados['ID_CONTRATO__pk_7_g_1'] == 3), 1,
np.where(np.bitwise_and(base_dados['ID_CONTRATO__pu_6_g_1'] == 1, base_dados['ID_CONTRATO__pk_7_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['ID_CONTRATO__pu_6_g_1'] == 1, base_dados['ID_CONTRATO__pk_7_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['ID_CONTRATO__pu_6_g_1'] == 1, base_dados['ID_CONTRATO__pk_7_g_1'] == 2), 2,
np.where(np.bitwise_and(base_dados['ID_CONTRATO__pu_6_g_1'] == 1, base_dados['ID_CONTRATO__pk_7_g_1'] == 3), 3,
np.where(np.bitwise_and(base_dados['ID_CONTRATO__pu_6_g_1'] == 2, base_dados['ID_CONTRATO__pk_7_g_1'] == 0), 2,
np.where(np.bitwise_and(base_dados['ID_CONTRATO__pu_6_g_1'] == 2, base_dados['ID_CONTRATO__pk_7_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['ID_CONTRATO__pu_6_g_1'] == 2, base_dados['ID_CONTRATO__pk_7_g_1'] == 2), 2,
np.where(np.bitwise_and(base_dados['ID_CONTRATO__pu_6_g_1'] == 2, base_dados['ID_CONTRATO__pk_7_g_1'] == 3), 3,
np.where(np.bitwise_and(base_dados['ID_CONTRATO__pu_6_g_1'] == 3, base_dados['ID_CONTRATO__pk_7_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['ID_CONTRATO__pu_6_g_1'] == 3, base_dados['ID_CONTRATO__pk_7_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['ID_CONTRATO__pu_6_g_1'] == 3, base_dados['ID_CONTRATO__pk_7_g_1'] == 2), 2,
np.where(np.bitwise_and(base_dados['ID_CONTRATO__pu_6_g_1'] == 3, base_dados['ID_CONTRATO__pk_7_g_1'] == 3), 4,
np.where(np.bitwise_and(base_dados['ID_CONTRATO__pu_6_g_1'] == 4, base_dados['ID_CONTRATO__pk_7_g_1'] == 0), 3,
np.where(np.bitwise_and(base_dados['ID_CONTRATO__pu_6_g_1'] == 4, base_dados['ID_CONTRATO__pk_7_g_1'] == 1), 4,
np.where(np.bitwise_and(base_dados['ID_CONTRATO__pu_6_g_1'] == 4, base_dados['ID_CONTRATO__pk_7_g_1'] == 2), 5,
np.where(np.bitwise_and(base_dados['ID_CONTRATO__pu_6_g_1'] == 4, base_dados['ID_CONTRATO__pk_7_g_1'] == 3), 5,
 0))))))))))))))))))))
base_dados['ID_CONTRATO__pk_7_g_1_c1_24_2'] = np.where(base_dados['ID_CONTRATO__pk_7_g_1_c1_24_1'] == 0, 0,
np.where(base_dados['ID_CONTRATO__pk_7_g_1_c1_24_1'] == 1, 1,
np.where(base_dados['ID_CONTRATO__pk_7_g_1_c1_24_1'] == 2, 3,
np.where(base_dados['ID_CONTRATO__pk_7_g_1_c1_24_1'] == 3, 2,
np.where(base_dados['ID_CONTRATO__pk_7_g_1_c1_24_1'] == 4, 4,
np.where(base_dados['ID_CONTRATO__pk_7_g_1_c1_24_1'] == 5, 5,
0))))))
base_dados['ID_CONTRATO__pk_7_g_1_c1_24'] = np.where(base_dados['ID_CONTRATO__pk_7_g_1_c1_24_2'] == 0, 0,
np.where(base_dados['ID_CONTRATO__pk_7_g_1_c1_24_2'] == 1, 1,
np.where(base_dados['ID_CONTRATO__pk_7_g_1_c1_24_2'] == 2, 2,
np.where(base_dados['ID_CONTRATO__pk_7_g_1_c1_24_2'] == 3, 3,
np.where(base_dados['ID_CONTRATO__pk_7_g_1_c1_24_2'] == 4, 4,
np.where(base_dados['ID_CONTRATO__pk_7_g_1_c1_24_2'] == 5, 5,
 0)))))) 
         
         
         
         
         
         
         
base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_13_g_1_c1_3_1'] = np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C__pk_3_g_1'] == 0, base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_13_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C__pk_3_g_1'] == 0, base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_13_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C__pk_3_g_1'] == 1, base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_13_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C__pk_3_g_1'] == 1, base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_13_g_1'] == 1), 2,
 0))))
base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_13_g_1_c1_3_2'] = np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_13_g_1_c1_3_1'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_13_g_1_c1_3_1'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_13_g_1_c1_3_1'] == 2, 2,
0)))
base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_13_g_1_c1_3'] = np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_13_g_1_c1_3_2'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_13_g_1_c1_3_2'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_13_g_1_c1_3_2'] == 2, 2,
 0)))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Mantendo apenas as variáveis do modelo

# COMMAND ----------

import pickle
modelo=pickle.load(open(caminho_base_pickle + 'model_fit_portocred_v4.sav', 'rb'))

base_teste_c0 = base_dados[[chave,'TIPO_EMAIL_gh38','PLANO_gh38','DETALHES_CONTRATOS_SALDO_ABERTO__L__pu_3_g_1_c1_16','DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh38','mob_contrato__pk_15_g_1_c1_20','ID_CONTRATO__pk_7_g_1_c1_24','TIPO_ENDERECO_gh38','DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_13_g_1_c1_3']]

var_fin_c0=['TIPO_EMAIL_gh38','PLANO_gh38','DETALHES_CONTRATOS_SALDO_ABERTO__L__pu_3_g_1_c1_16','DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh38','mob_contrato__pk_15_g_1_c1_20','ID_CONTRATO__pk_7_g_1_c1_24','TIPO_ENDERECO_gh38','DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_13_g_1_c1_3']

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

x_teste2['P_1_C'] = np.cos(x_teste2['P_1'])
x_teste2['P_1_C'] = np.where(x_teste2['P_1_C'] == 0, -1, x_teste2['P_1_C'])
x_teste2['P_1_C'] = np.where(x_teste2['P_1_C'] == np.nan, -2, x_teste2['P_1_C'])
x_teste2['P_1_C'] = x_teste2['P_1_C'].fillna(-2)
 
x_teste2['P_1_R'] = np.sqrt(x_teste2['P_1'])
x_teste2['P_1_R'] = np.where(x_teste2['P_1_R'] == 0, -1, x_teste2['P_1_R'])
x_teste2['P_1_R'] = np.where(x_teste2['P_1_R'] == np.nan, -2, x_teste2['P_1_R'])
x_teste2['P_1_R'] = x_teste2['P_1_R'].fillna(-2)

x_teste2['P_1_C_pu_7_g_1'] = np.where(x_teste2['P_1_C'] < 0.848565941, 3,
    np.where(np.bitwise_and(x_teste2['P_1_C'] > 0.848565941, x_teste2['P_1_C'] <= 0.924341184), 2,
    np.where(np.bitwise_and(x_teste2['P_1_C'] > 0.924341184, x_teste2['P_1_C'] <= 0.962085043), 1,0)))

x_teste2['P_1_R_pu_7_g_1'] = np.where(x_teste2['P_1_R'] < 0.226964321, 0,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.226964321, x_teste2['P_1_R'] <= 0.33172857), 1,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.33172857, x_teste2['P_1_R'] <= 0.43896328), 2,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.43896328, x_teste2['P_1_R'] <= 0.650394787), 3,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.650394787, x_teste2['P_1_R'] <= 0.75473771), 4,
             5)))))

x_teste2['GH'] = np.where(np.bitwise_and(x_teste2['P_1_R_pu_7_g_1'] == 0, x_teste2['P_1_C_pu_7_g_1'] == 0), 0,
    np.where(np.bitwise_and(x_teste2['P_1_R_pu_7_g_1'] == 0, x_teste2['P_1_C_pu_7_g_1'] == 1), 0,
    np.where(np.bitwise_and(x_teste2['P_1_R_pu_7_g_1'] == 0, x_teste2['P_1_C_pu_7_g_1'] == 2), 0,
    np.where(np.bitwise_and(x_teste2['P_1_R_pu_7_g_1'] == 0, x_teste2['P_1_C_pu_7_g_1'] == 3), 1,
    np.where(np.bitwise_and(x_teste2['P_1_R_pu_7_g_1'] == 1, x_teste2['P_1_C_pu_7_g_1'] == 0), 1,
    np.where(np.bitwise_and(x_teste2['P_1_R_pu_7_g_1'] == 1, x_teste2['P_1_C_pu_7_g_1'] == 1), 1,
    np.where(np.bitwise_and(x_teste2['P_1_R_pu_7_g_1'] == 1, x_teste2['P_1_C_pu_7_g_1'] == 2), 1,
    np.where(np.bitwise_and(x_teste2['P_1_R_pu_7_g_1'] == 1, x_teste2['P_1_C_pu_7_g_1'] == 3), 2,
    np.where(np.bitwise_and(x_teste2['P_1_R_pu_7_g_1'] == 2, x_teste2['P_1_C_pu_7_g_1'] == 0), 2,
    np.where(np.bitwise_and(x_teste2['P_1_R_pu_7_g_1'] == 2, x_teste2['P_1_C_pu_7_g_1'] == 1), 2,
    np.where(np.bitwise_and(x_teste2['P_1_R_pu_7_g_1'] == 2, x_teste2['P_1_C_pu_7_g_1'] == 2), 2,
    np.where(np.bitwise_and(x_teste2['P_1_R_pu_7_g_1'] == 2, x_teste2['P_1_C_pu_7_g_1'] == 3), 3,
    np.where(np.bitwise_and(x_teste2['P_1_R_pu_7_g_1'] == 3, x_teste2['P_1_C_pu_7_g_1'] == 0), 4,
    np.where(np.bitwise_and(x_teste2['P_1_R_pu_7_g_1'] == 3, x_teste2['P_1_C_pu_7_g_1'] == 1), 3,
    np.where(np.bitwise_and(x_teste2['P_1_R_pu_7_g_1'] == 3, x_teste2['P_1_C_pu_7_g_1'] == 2), 5,
    np.where(np.bitwise_and(x_teste2['P_1_R_pu_7_g_1'] == 3, x_teste2['P_1_C_pu_7_g_1'] == 3), 5,
    np.where(np.bitwise_and(x_teste2['P_1_R_pu_7_g_1'] == 4, x_teste2['P_1_C_pu_7_g_1'] == 0), 5,
    np.where(np.bitwise_and(x_teste2['P_1_R_pu_7_g_1'] == 4, x_teste2['P_1_C_pu_7_g_1'] == 1), 5,
    np.where(np.bitwise_and(x_teste2['P_1_R_pu_7_g_1'] == 4, x_teste2['P_1_C_pu_7_g_1'] == 2), 5,
    np.where(np.bitwise_and(x_teste2['P_1_R_pu_7_g_1'] == 4, x_teste2['P_1_C_pu_7_g_1'] == 3), 6,
    np.where(np.bitwise_and(x_teste2['P_1_R_pu_7_g_1'] == 5, x_teste2['P_1_C_pu_7_g_1'] == 0), 5,
    np.where(np.bitwise_and(x_teste2['P_1_R_pu_7_g_1'] == 5, x_teste2['P_1_C_pu_7_g_1'] == 1), 5,
    np.where(np.bitwise_and(x_teste2['P_1_R_pu_7_g_1'] == 5, x_teste2['P_1_C_pu_7_g_1'] == 2), 6,
    np.where(np.bitwise_and(x_teste2['P_1_R_pu_7_g_1'] == 5, x_teste2['P_1_C_pu_7_g_1'] == 3), 6,
             0))))))))))))))))))))))))

del x_teste2['P_1_C']
del x_teste2['P_1_R']
del x_teste2['P_1_R_pu_7_g_1']
del x_teste2['P_1_C_pu_7_g_1']
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