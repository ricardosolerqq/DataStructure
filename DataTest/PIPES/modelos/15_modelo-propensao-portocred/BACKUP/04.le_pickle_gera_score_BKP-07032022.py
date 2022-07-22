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
import os
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
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
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
chave = 'DOCUMENTO_PESSOA'

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

base_teste = pd.read_csv('/dbfs/mnt/ml-prd/ml-data/propensaodeal/portocred/trusted/2022-02-20/trustedFile_portocred.csv', sep=';', decimal='.')
base_teste

# COMMAND ----------

base_dados = pd.read_csv(caminho_base+N_Base, sep=separador_, decimal=decimal_)
base_dados

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importação da Base de Dados

# COMMAND ----------

#carregar o arquivo em formato tabela
base_dados = pd.read_csv(caminho_base+N_Base, sep=separador_, decimal=decimal_)
base_dados = base_dados[[chave,'DETALHES_CONTRATOS_TAXA_MES','TIPO_EMAIL','PLANO','TIPO_ENDERECO','DETALHES_CLIENTES_CONTA_REF_BANCARIA3','ID_CONTRATO','ID_PRODUTO','DETALHES_CONTRATOS_SALDO_ABERTO','DETALHES_CONTRATOS_TOTAL_PAGO']]

base_dados.fillna(-3)

#string
base_dados['TIPO_ENDERECO'] = base_dados['TIPO_ENDERECO'].replace(np.nan, '-3')
base_dados['TIPO_EMAIL'] = base_dados['TIPO_EMAIL'].replace(np.nan, '-3')

#numericas
base_dados['PLANO'] = base_dados['PLANO'].replace(np.nan, '-3')
base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3'] = base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3'].replace(np.nan, '-3')
base_dados['DETALHES_CONTRATOS_TAXA_MES'] = base_dados['DETALHES_CONTRATOS_TAXA_MES'].replace(np.nan, '-3')

base_dados['ID_CONTRATO'] = base_dados['ID_CONTRATO'].replace(np.nan, '-3')
base_dados['DETALHES_CONTRATOS_TOTAL_PAGO'] = base_dados['DETALHES_CONTRATOS_TOTAL_PAGO'].replace(np.nan, '-3')
base_dados['DETALHES_CONTRATOS_SALDO_ABERTO'] = base_dados['DETALHES_CONTRATOS_SALDO_ABERTO'].replace(np.nan, '-3')
base_dados['ID_PRODUTO'] = base_dados['ID_PRODUTO'].replace(np.nan, '-3')

base_dados = base_dados.apply(pd.to_numeric, errors='ignore')

base_dados['PLANO'] = base_dados['PLANO'].astype(np.int64)
base_dados['ID_CONTRATO'] = base_dados['ID_CONTRATO'].astype(np.int64)
base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3'] = base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3'].astype(np.int64)
base_dados['DETALHES_CONTRATOS_TOTAL_PAGO'] = base_dados['DETALHES_CONTRATOS_TOTAL_PAGO'].astype(np.int64)
base_dados['DETALHES_CONTRATOS_SALDO_ABERTO'] = base_dados['DETALHES_CONTRATOS_SALDO_ABERTO'].astype(np.int64)
base_dados['DETALHES_CONTRATOS_TAXA_MES'] = base_dados['DETALHES_CONTRATOS_TAXA_MES'].astype(np.int64)
base_dados['ID_PRODUTO'] = base_dados['ID_PRODUTO'].astype(np.int64)

base_dados.drop_duplicates(keep='first', inplace=True)

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
                                         
                                         
                                         
                                         
                                         
                                         
base_dados['DETALHES_CONTRATOS_TAXA_MES_gh30'] = np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES'] == 1, 0,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES'] == 2, 1,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES'] == 3, 2,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES'] == 4, 3,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES'] == 5, 4,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES'] == 6, 5,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES'] == 7, 6,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES'] == 8, 7,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES'] == 9, 8,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES'] == 10, 9,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES'] == 11, 10,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES'] == 12, 11,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES'] == 13, 12,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES'] == 14, 13,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES'] == 15, 14,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES'] == 16, 15,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES'] == 17, 16,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES'] == 18, 17,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES'] == 19, 18,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES'] == 21, 19,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES'] == 22, 20,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES'] == 23, 21,
0))))))))))))))))))))))

base_dados['DETALHES_CONTRATOS_TAXA_MES_gh31'] = np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh30'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh30'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh30'] == 2, 1,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh30'] == 3, 3,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh30'] == 4, 4,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh30'] == 5, 4,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh30'] == 6, 6,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh30'] == 7, 7,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh30'] == 8, 7,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh30'] == 9, 9,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh30'] == 10, 10,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh30'] == 11, 10,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh30'] == 12, 10,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh30'] == 13, 13,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh30'] == 14, 14,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh30'] == 15, 15,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh30'] == 16, 15,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh30'] == 17, 15,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh30'] == 18, 18,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh30'] == 19, 18,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh30'] == 20, 18,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh30'] == 21, 18,
0))))))))))))))))))))))

base_dados['DETALHES_CONTRATOS_TAXA_MES_gh32'] = np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh31'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh31'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh31'] == 3, 2,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh31'] == 4, 3,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh31'] == 6, 4,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh31'] == 7, 5,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh31'] == 9, 6,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh31'] == 10, 7,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh31'] == 13, 8,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh31'] == 14, 9,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh31'] == 15, 10,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh31'] == 18, 11,
0))))))))))))

base_dados['DETALHES_CONTRATOS_TAXA_MES_gh33'] = np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh32'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh32'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh32'] == 2, 2,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh32'] == 3, 3,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh32'] == 4, 4,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh32'] == 5, 5,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh32'] == 6, 6,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh32'] == 7, 7,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh32'] == 8, 8,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh32'] == 9, 9,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh32'] == 10, 10,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh32'] == 11, 11,
0))))))))))))

base_dados['DETALHES_CONTRATOS_TAXA_MES_gh34'] = np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh33'] == 0, 10,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh33'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh33'] == 2, 2,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh33'] == 3, 3,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh33'] == 4, 10,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh33'] == 5, 10,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh33'] == 6, 10,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh33'] == 7, 7,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh33'] == 8, 8,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh33'] == 9, 2,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh33'] == 10, 10,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh33'] == 11, 12,
0))))))))))))

base_dados['DETALHES_CONTRATOS_TAXA_MES_gh35'] = np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh34'] == 1, 0,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh34'] == 2, 1,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh34'] == 3, 2,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh34'] == 7, 3,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh34'] == 8, 4,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh34'] == 10, 5,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh34'] == 12, 6,
0)))))))

base_dados['DETALHES_CONTRATOS_TAXA_MES_gh36'] = np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh35'] == 0, 4,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh35'] == 1, 3,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh35'] == 2, 1,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh35'] == 3, 4,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh35'] == 4, 1,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh35'] == 5, 6,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh35'] == 6, 0,
0)))))))

base_dados['DETALHES_CONTRATOS_TAXA_MES_gh37'] = np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh36'] == 0, 1,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh36'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh36'] == 3, 2,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh36'] == 4, 3,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh36'] == 6, 4,
0)))))

base_dados['DETALHES_CONTRATOS_TAXA_MES_gh38'] = np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh37'] == 1, 0,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh37'] == 2, 1,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh37'] == 3, 2,
np.where(base_dados['DETALHES_CONTRATOS_TAXA_MES_gh37'] == 4, 3,
0))))
                                                                        
                                                                        
                                                                        
                                                                        
                                                                        
                                                                        
                                                                        
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

         
base_dados['ID_CONTRATO__pu_8'] = np.where(base_dados['ID_CONTRATO'] <= 127361.0, 0.0,
np.where(np.bitwise_and(base_dados['ID_CONTRATO'] > 127361.0, base_dados['ID_CONTRATO'] <= 184876.0), 1.0,
np.where(np.bitwise_and(base_dados['ID_CONTRATO'] > 184876.0, base_dados['ID_CONTRATO'] <= 243114.0), 2.0,
np.where(np.bitwise_and(base_dados['ID_CONTRATO'] > 243114.0, base_dados['ID_CONTRATO'] <= 300050.0), 3.0,
np.where(np.bitwise_and(base_dados['ID_CONTRATO'] > 300050.0, base_dados['ID_CONTRATO'] <= 357492.0), 4.0,
np.where(np.bitwise_and(base_dados['ID_CONTRATO'] > 357492.0, base_dados['ID_CONTRATO'] <= 414623.0), 5.0,
np.where(np.bitwise_and(base_dados['ID_CONTRATO'] > 414623.0, base_dados['ID_CONTRATO'] <= 471675.0), 6.0,
np.where(base_dados['ID_CONTRATO'] > 471675.0, 7.0,
 0))))))))

base_dados['ID_CONTRATO__pu_8_g_1_1'] = np.where(base_dados['ID_CONTRATO__pu_8'] == 0.0, 0,
np.where(base_dados['ID_CONTRATO__pu_8'] == 1.0, 5,
np.where(base_dados['ID_CONTRATO__pu_8'] == 2.0, 5,
np.where(base_dados['ID_CONTRATO__pu_8'] == 3.0, 3,
np.where(base_dados['ID_CONTRATO__pu_8'] == 4.0, 2,
np.where(base_dados['ID_CONTRATO__pu_8'] == 5.0, 4,
np.where(base_dados['ID_CONTRATO__pu_8'] == 6.0, 3,
np.where(base_dados['ID_CONTRATO__pu_8'] == 7.0, 1,
 0))))))))
base_dados['ID_CONTRATO__pu_8_g_1_2'] = np.where(base_dados['ID_CONTRATO__pu_8_g_1_1'] == 0, 0,
np.where(base_dados['ID_CONTRATO__pu_8_g_1_1'] == 1, 5,
np.where(base_dados['ID_CONTRATO__pu_8_g_1_1'] == 2, 2,
np.where(base_dados['ID_CONTRATO__pu_8_g_1_1'] == 3, 4,
np.where(base_dados['ID_CONTRATO__pu_8_g_1_1'] == 4, 2,
np.where(base_dados['ID_CONTRATO__pu_8_g_1_1'] == 5, 1,
 0))))))
base_dados['ID_CONTRATO__pu_8_g_1'] = np.where(base_dados['ID_CONTRATO__pu_8_g_1_2'] == 0, 0,
np.where(base_dados['ID_CONTRATO__pu_8_g_1_2'] == 1, 1,
np.where(base_dados['ID_CONTRATO__pu_8_g_1_2'] == 2, 2,
np.where(base_dados['ID_CONTRATO__pu_8_g_1_2'] == 4, 3,
np.where(base_dados['ID_CONTRATO__pu_8_g_1_2'] == 5, 4,
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
         
         
         
         
         
base_dados['ID_PRODUTO__C'] = np.cos(base_dados['ID_PRODUTO'])
np.where(base_dados['ID_PRODUTO__C'] == 0, -1, base_dados['ID_PRODUTO__C'])
base_dados['ID_PRODUTO__C'] = base_dados['ID_PRODUTO__C'].fillna(-2)
base_dados['ID_PRODUTO__C__p_4'] = np.where(base_dados['ID_PRODUTO__C'] <= -0.6400980212258834, 0.0,
np.where(np.bitwise_and(base_dados['ID_PRODUTO__C'] > -0.6400980212258834, base_dados['ID_PRODUTO__C'] <= -0.07519620904973118), 1.0,
np.where(np.bitwise_and(base_dados['ID_PRODUTO__C'] > -0.07519620904973118, base_dados['ID_PRODUTO__C'] <= 0.6195679368857272), 2.0,
np.where(base_dados['ID_PRODUTO__C'] > 0.6195679368857272, 3.0,
 0))))
base_dados['ID_PRODUTO__C__p_4_g_1_1'] = np.where(base_dados['ID_PRODUTO__C__p_4'] == 0.0, 2,
np.where(base_dados['ID_PRODUTO__C__p_4'] == 1.0, 0,
np.where(base_dados['ID_PRODUTO__C__p_4'] == 2.0, 1,
np.where(base_dados['ID_PRODUTO__C__p_4'] == 3.0, 0,
 0))))
base_dados['ID_PRODUTO__C__p_4_g_1_2'] = np.where(base_dados['ID_PRODUTO__C__p_4_g_1_1'] == 0, 1,
np.where(base_dados['ID_PRODUTO__C__p_4_g_1_1'] == 1, 0,
np.where(base_dados['ID_PRODUTO__C__p_4_g_1_1'] == 2, 2,
 0)))
base_dados['ID_PRODUTO__C__p_4_g_1'] = np.where(base_dados['ID_PRODUTO__C__p_4_g_1_2'] == 0, 0,
np.where(base_dados['ID_PRODUTO__C__p_4_g_1_2'] == 1, 1,
np.where(base_dados['ID_PRODUTO__C__p_4_g_1_2'] == 2, 2,
 0)))
         
         
         
         
         
base_dados['ID_PRODUTO__T'] = np.tan(base_dados['ID_PRODUTO'])
np.where(base_dados['ID_PRODUTO__T'] == 0, -1, base_dados['ID_PRODUTO__T'])
base_dados['ID_PRODUTO__T'] = base_dados['ID_PRODUTO__T'].fillna(-2)
base_dados['ID_PRODUTO__T__pu_20'] = np.where(base_dados['ID_PRODUTO__T'] <= -32.20586121597425, 0.0,
np.where(np.bitwise_and(base_dados['ID_PRODUTO__T'] > -32.20586121597425, base_dados['ID_PRODUTO__T'] <= -17.352653686610658), 6.0,
np.where(np.bitwise_and(base_dados['ID_PRODUTO__T'] > -17.352653686610658, base_dados['ID_PRODUTO__T'] <= -15.034506078798417), 7.0,
np.where(np.bitwise_and(base_dados['ID_PRODUTO__T'] > -15.034506078798417, base_dados['ID_PRODUTO__T'] <= -13.260891336219183), 8.0,
np.where(np.bitwise_and(base_dados['ID_PRODUTO__T'] > -13.260891336219183, base_dados['ID_PRODUTO__T'] <= -8.99870998872415), 9.0,
np.where(np.bitwise_and(base_dados['ID_PRODUTO__T'] > -8.99870998872415, base_dados['ID_PRODUTO__T'] <= -4.026763652295865), 11.0,
np.where(np.bitwise_and(base_dados['ID_PRODUTO__T'] > -4.026763652295865, base_dados['ID_PRODUTO__T'] <= -1.498191709749359), 12.0,
np.where(np.bitwise_and(base_dados['ID_PRODUTO__T'] > -1.498191709749359, base_dados['ID_PRODUTO__T'] <= 0.7821577354975264), 13.0,
np.where(np.bitwise_and(base_dados['ID_PRODUTO__T'] > 0.7821577354975264, base_dados['ID_PRODUTO__T'] <= 2.9879847010804625), 14.0,
np.where(np.bitwise_and(base_dados['ID_PRODUTO__T'] > 2.9879847010804625, base_dados['ID_PRODUTO__T'] <= 9.795902188167448), 17.0,
np.where(np.bitwise_and(base_dados['ID_PRODUTO__T'] > 9.795902188167448, base_dados['ID_PRODUTO__T'] <= 11.872737300627563), 18.0,
np.where(base_dados['ID_PRODUTO__T'] > 11.872737300627563, 19.0,
 0))))))))))))
base_dados['ID_PRODUTO__T__pu_20_g_1_1'] = np.where(base_dados['ID_PRODUTO__T__pu_20'] == 0.0, 1,
np.where(base_dados['ID_PRODUTO__T__pu_20'] == 6.0, 1,
np.where(base_dados['ID_PRODUTO__T__pu_20'] == 7.0, 1,
np.where(base_dados['ID_PRODUTO__T__pu_20'] == 8.0, 1,
np.where(base_dados['ID_PRODUTO__T__pu_20'] == 9.0, 1,
np.where(base_dados['ID_PRODUTO__T__pu_20'] == 11.0, 0,
np.where(base_dados['ID_PRODUTO__T__pu_20'] == 12.0, 0,
np.where(base_dados['ID_PRODUTO__T__pu_20'] == 13.0, 0,
np.where(base_dados['ID_PRODUTO__T__pu_20'] == 14.0, 1,
np.where(base_dados['ID_PRODUTO__T__pu_20'] == 17.0, 0,
np.where(base_dados['ID_PRODUTO__T__pu_20'] == 18.0, 0,
np.where(base_dados['ID_PRODUTO__T__pu_20'] == 19.0, 1,
 0))))))))))))
base_dados['ID_PRODUTO__T__pu_20_g_1_2'] = np.where(base_dados['ID_PRODUTO__T__pu_20_g_1_1'] == 0, 1,
np.where(base_dados['ID_PRODUTO__T__pu_20_g_1_1'] == 1, 0,
 0))
base_dados['ID_PRODUTO__T__pu_20_g_1'] = np.where(base_dados['ID_PRODUTO__T__pu_20_g_1_2'] == 0, 0,
np.where(base_dados['ID_PRODUTO__T__pu_20_g_1_2'] == 1, 1,
 0))
         
         
         
         
         
base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L'] = np.log(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO'])
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L'] == 0, -1, base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L'])
base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L'] = base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L'].fillna(-2)
base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__p_7'] = np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L'] <= 6.6240652277998935, 0.0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L'] > 6.6240652277998935, base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L'] <= 7.142036574706803), 1.0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L'] > 7.142036574706803, base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L'] <= 7.539027055823995), 2.0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L'] > 7.539027055823995, base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L'] <= 7.8838232148921525), 3.0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L'] > 7.8838232148921525, base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L'] <= 8.366835309827675), 4.0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L'] > 8.366835309827675, base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L'] <= 8.888618807300878), 5.0,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L'] > 8.888618807300878, 6.0,
 0)))))))
base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__p_7_g_1_1'] = np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__p_7'] == 0.0, 0,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__p_7'] == 1.0, 1,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__p_7'] == 2.0, 0,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__p_7'] == 3.0, 1,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__p_7'] == 4.0, 1,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__p_7'] == 5.0, 1,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__p_7'] == 6.0, 1,
 0)))))))
base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__p_7_g_1_2'] = np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__p_7_g_1_1'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__p_7_g_1_1'] == 1, 1,
 0))
base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__p_7_g_1'] = np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__p_7_g_1_2'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__p_7_g_1_2'] == 1, 1,
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
                                                   
                                                   
                                                   
                                                   
                                                   
base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C'] = np.cos(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO'])
np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C'] == 0, -1, base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C'])
base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C'] = base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C'].fillna(-2)
base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C__p_7'] = np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C'] <= -0.9912546013069977, 0.0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C'] > -0.9912546013069977, base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C'] <= -0.9899924966004454), 1.0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C'] > -0.9899924966004454, base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C'] <= -0.4480736161291701), 2.0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C'] > -0.4480736161291701, base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C'] <= 0.5638737005763481), 3.0,
np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C'] > 0.5638737005763481, 4.0,
 0)))))
base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C__p_7_g_1_1'] = np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C__p_7'] == 0.0, 0,
np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C__p_7'] == 1.0, 0,
np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C__p_7'] == 2.0, 0,
np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C__p_7'] == 3.0, 1,
np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C__p_7'] == 4.0, 0,
 0)))))
base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C__p_7_g_1_2'] = np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C__p_7_g_1_1'] == 0, 1,
np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C__p_7_g_1_1'] == 1, 0,
 0))
base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C__p_7_g_1'] = np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C__p_7_g_1_2'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C__p_7_g_1_2'] == 1, 1,
 0))
                                                                   
                                                                   
                                                                   
                                                                   
                                                                   
base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T'] = np.tan(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO'])
np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T'] == 0, -1, base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T'])
base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T'] = base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T'].fillna(-2)
base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_6'] = np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T'] <= 4.365198093460923, 0.0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T'] > 4.365198093460923, base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T'] <= 9.013560982203286), 1.0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T'] > 9.013560982203286, base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T'] <= 11.91999619729968), 2.0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T'] > 11.91999619729968, base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T'] <= 15.39897670346363), 3.0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T'] > 15.39897670346363, base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T'] <= 20.60154909340478), 4.0,
np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T'] > 20.60154909340478, 5.0,
 -2))))))
base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_6_g_1_1'] = np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_6'] == -2.0, 1,
np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_6'] == 0.0, 0,
np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_6'] == 1.0, 1,
np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_6'] == 2.0, 1,
np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_6'] == 3.0, 1,
np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_6'] == 4.0, 1,
np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_6'] == 5.0, 1,
 0)))))))
base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_6_g_1_2'] = np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_6_g_1_1'] == 0, 1,
np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_6_g_1_1'] == 1, 0,
 0))
base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_6_g_1'] = np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_6_g_1_2'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_6_g_1_2'] == 1, 1,
 0))


# COMMAND ----------

# MAGIC %md
# MAGIC ### Parte 2 de 2

# COMMAND ----------

base_dados['ID_CONTRATO__pk_7_g_1_c1_22_1'] = np.where(np.bitwise_and(base_dados['ID_CONTRATO__pu_8_g_1'] == 0, base_dados['ID_CONTRATO__pk_7_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['ID_CONTRATO__pu_8_g_1'] == 0, base_dados['ID_CONTRATO__pk_7_g_1'] == 1), 0,
np.where(np.bitwise_and(base_dados['ID_CONTRATO__pu_8_g_1'] == 0, base_dados['ID_CONTRATO__pk_7_g_1'] == 2), 1,
np.where(np.bitwise_and(base_dados['ID_CONTRATO__pu_8_g_1'] == 0, base_dados['ID_CONTRATO__pk_7_g_1'] == 3), 2,
np.where(np.bitwise_and(base_dados['ID_CONTRATO__pu_8_g_1'] == 1, base_dados['ID_CONTRATO__pk_7_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['ID_CONTRATO__pu_8_g_1'] == 1, base_dados['ID_CONTRATO__pk_7_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['ID_CONTRATO__pu_8_g_1'] == 1, base_dados['ID_CONTRATO__pk_7_g_1'] == 2), 2,
np.where(np.bitwise_and(base_dados['ID_CONTRATO__pu_8_g_1'] == 1, base_dados['ID_CONTRATO__pk_7_g_1'] == 3), 2,
np.where(np.bitwise_and(base_dados['ID_CONTRATO__pu_8_g_1'] == 2, base_dados['ID_CONTRATO__pk_7_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['ID_CONTRATO__pu_8_g_1'] == 2, base_dados['ID_CONTRATO__pk_7_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['ID_CONTRATO__pu_8_g_1'] == 2, base_dados['ID_CONTRATO__pk_7_g_1'] == 2), 2,
np.where(np.bitwise_and(base_dados['ID_CONTRATO__pu_8_g_1'] == 2, base_dados['ID_CONTRATO__pk_7_g_1'] == 3), 3,
np.where(np.bitwise_and(base_dados['ID_CONTRATO__pu_8_g_1'] == 3, base_dados['ID_CONTRATO__pk_7_g_1'] == 0), 2,
np.where(np.bitwise_and(base_dados['ID_CONTRATO__pu_8_g_1'] == 3, base_dados['ID_CONTRATO__pk_7_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['ID_CONTRATO__pu_8_g_1'] == 3, base_dados['ID_CONTRATO__pk_7_g_1'] == 2), 3,
np.where(np.bitwise_and(base_dados['ID_CONTRATO__pu_8_g_1'] == 3, base_dados['ID_CONTRATO__pk_7_g_1'] == 3), 4,
np.where(np.bitwise_and(base_dados['ID_CONTRATO__pu_8_g_1'] == 4, base_dados['ID_CONTRATO__pk_7_g_1'] == 0), 4,
np.where(np.bitwise_and(base_dados['ID_CONTRATO__pu_8_g_1'] == 4, base_dados['ID_CONTRATO__pk_7_g_1'] == 1), 4,
np.where(np.bitwise_and(base_dados['ID_CONTRATO__pu_8_g_1'] == 4, base_dados['ID_CONTRATO__pk_7_g_1'] == 2), 5,
np.where(np.bitwise_and(base_dados['ID_CONTRATO__pu_8_g_1'] == 4, base_dados['ID_CONTRATO__pk_7_g_1'] == 3), 5,
 0))))))))))))))))))))
base_dados['ID_CONTRATO__pk_7_g_1_c1_22_2'] = np.where(base_dados['ID_CONTRATO__pk_7_g_1_c1_22_1'] == 0, 0,
np.where(base_dados['ID_CONTRATO__pk_7_g_1_c1_22_1'] == 1, 1,
np.where(base_dados['ID_CONTRATO__pk_7_g_1_c1_22_1'] == 2, 3,
np.where(base_dados['ID_CONTRATO__pk_7_g_1_c1_22_1'] == 3, 2,
np.where(base_dados['ID_CONTRATO__pk_7_g_1_c1_22_1'] == 4, 4,
np.where(base_dados['ID_CONTRATO__pk_7_g_1_c1_22_1'] == 5, 5,
0))))))
base_dados['ID_CONTRATO__pk_7_g_1_c1_22'] = np.where(base_dados['ID_CONTRATO__pk_7_g_1_c1_22_2'] == 0, 0,
np.where(base_dados['ID_CONTRATO__pk_7_g_1_c1_22_2'] == 1, 1,
np.where(base_dados['ID_CONTRATO__pk_7_g_1_c1_22_2'] == 2, 2,
np.where(base_dados['ID_CONTRATO__pk_7_g_1_c1_22_2'] == 3, 3,
np.where(base_dados['ID_CONTRATO__pk_7_g_1_c1_22_2'] == 4, 4,
np.where(base_dados['ID_CONTRATO__pk_7_g_1_c1_22_2'] == 5, 5,
 0))))))
         
         
 
         
         
         
base_dados['ID_PRODUTO__C__p_4_g_1_c1_25_1'] = np.where(np.bitwise_and(base_dados['ID_PRODUTO__C__p_4_g_1'] == 0, base_dados['ID_PRODUTO__T__pu_20_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['ID_PRODUTO__C__p_4_g_1'] == 0, base_dados['ID_PRODUTO__T__pu_20_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['ID_PRODUTO__C__p_4_g_1'] == 1, base_dados['ID_PRODUTO__T__pu_20_g_1'] == 0), 2,
np.where(np.bitwise_and(base_dados['ID_PRODUTO__C__p_4_g_1'] == 1, base_dados['ID_PRODUTO__T__pu_20_g_1'] == 1), 3,
np.where(np.bitwise_and(base_dados['ID_PRODUTO__C__p_4_g_1'] == 2, base_dados['ID_PRODUTO__T__pu_20_g_1'] == 0), 3,
np.where(np.bitwise_and(base_dados['ID_PRODUTO__C__p_4_g_1'] == 2, base_dados['ID_PRODUTO__T__pu_20_g_1'] == 1), 3,
 0))))))
base_dados['ID_PRODUTO__C__p_4_g_1_c1_25_2'] = np.where(base_dados['ID_PRODUTO__C__p_4_g_1_c1_25_1'] == 0, 0,
np.where(base_dados['ID_PRODUTO__C__p_4_g_1_c1_25_1'] == 1, 2,
np.where(base_dados['ID_PRODUTO__C__p_4_g_1_c1_25_1'] == 2, 1,
np.where(base_dados['ID_PRODUTO__C__p_4_g_1_c1_25_1'] == 3, 3,
0))))
base_dados['ID_PRODUTO__C__p_4_g_1_c1_25'] = np.where(base_dados['ID_PRODUTO__C__p_4_g_1_c1_25_2'] == 0, 0,
np.where(base_dados['ID_PRODUTO__C__p_4_g_1_c1_25_2'] == 1, 1,
np.where(base_dados['ID_PRODUTO__C__p_4_g_1_c1_25_2'] == 2, 2,
np.where(base_dados['ID_PRODUTO__C__p_4_g_1_c1_25_2'] == 3, 3,
 0))))
         
         
         
         
              
base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__p_7_g_1_c1_5_1'] = np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__p_7_g_1'] == 0, base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__pu_3_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__p_7_g_1'] == 0, base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__pu_3_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__p_7_g_1'] == 1, base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__pu_3_g_1'] == 0), 2,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__p_7_g_1'] == 1, base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__pu_3_g_1'] == 1), 2,
 0))))
base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__p_7_g_1_c1_5_2'] = np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__p_7_g_1_c1_5_1'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__p_7_g_1_c1_5_1'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__p_7_g_1_c1_5_1'] == 2, 2,
0)))
base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__p_7_g_1_c1_5'] = np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__p_7_g_1_c1_5_2'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__p_7_g_1_c1_5_2'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__p_7_g_1_c1_5_2'] == 2, 2,
 0)))
         
         
         
         
         
        
base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C__p_7_g_1_c1_7_1'] = np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C__p_7_g_1'] == 0, base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_6_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C__p_7_g_1'] == 0, base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_6_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C__p_7_g_1'] == 1, base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_6_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C__p_7_g_1'] == 1, base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__T__pe_6_g_1'] == 1), 2,
 0))))
base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C__p_7_g_1_c1_7_2'] = np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C__p_7_g_1_c1_7_1'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C__p_7_g_1_c1_7_1'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C__p_7_g_1_c1_7_1'] == 2, 2,
0)))
base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C__p_7_g_1_c1_7'] = np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C__p_7_g_1_c1_7_2'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C__p_7_g_1_c1_7_2'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_TOTAL_PAGO__C__p_7_g_1_c1_7_2'] == 2, 2,
 0)))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Mantendo apenas as variáveis do modelo

# COMMAND ----------

import pickle
modelo=pickle.load(open(caminho_base_pickle + 'model_fit_portocred_v2.sav', 'rb'))

base_teste_c0 = base_dados[[chave,'TIPO_EMAIL_gh38','PLANO_gh38','DETALHES_CONTRATOS_SALDO_ABERTO__L__p_7_g_1_c1_5','DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh38','ID_PRODUTO__C__p_4_g_1_c1_25','TIPO_ENDERECO_gh38','DETALHES_CONTRATOS_TAXA_MES_gh38','ID_CONTRATO__pk_7_g_1_c1_22','DETALHES_CONTRATOS_TOTAL_PAGO__C__p_7_g_1_c1_7']]

var_fin_c0=['TIPO_EMAIL_gh38','PLANO_gh38','DETALHES_CONTRATOS_SALDO_ABERTO__L__p_7_g_1_c1_5','DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh38','ID_PRODUTO__C__p_4_g_1_c1_25','TIPO_ENDERECO_gh38','DETALHES_CONTRATOS_TAXA_MES_gh38','ID_CONTRATO__pk_7_g_1_c1_22','DETALHES_CONTRATOS_TOTAL_PAGO__C__p_7_g_1_c1_7']

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
    
x_teste2['GH'] = np.where(x_teste2['P_1_C'] > 0.995397145, 0.0,
    np.where(np.bitwise_and(x_teste2['P_1_C'] > 0.976949901, x_teste2['P_1_C'] <= 0.995397145), 1,
    np.where(np.bitwise_and(x_teste2['P_1_C'] > 0.897663829, x_teste2['P_1_C'] <= 0.976949901), 3,
    np.where(np.bitwise_and(x_teste2['P_1_C'] > 0.854382632, x_teste2['P_1_C'] <= 0.897663829), 2,4))))

del x_teste2['P_1_C']
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