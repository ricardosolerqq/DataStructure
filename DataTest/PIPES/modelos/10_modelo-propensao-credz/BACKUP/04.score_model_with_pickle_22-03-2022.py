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

blob_account_source_ml = "qqdatastoragemain"
blob_container_source_ml = "ml-prd"

blob_account_source_lake = "saqueroquitar"
blob_container_source_lake = "trusted"

mount_blob_storage_key(dbutils,blob_account_source_ml,blob_container_source_ml,'/mnt/ml-prd')
mount_blob_storage_key(dbutils,blob_account_source_lake,blob_container_source_lake,'/mnt/ml-prd')

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

trustedpath_listdir

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
import os
import datetime
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
import pickle

%matplotlib inline

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inserindo hiperparâmetros do Algoritmo

# COMMAND ----------

file

# COMMAND ----------

## Parâmetros do Algoritmo

#Variável chave-primaria
chave = 'DOCUMENTO:ID_DIVIDA'

#Variável resposta ou target
target = 'VARIAVEL_RESPOSTA'

#Lista com a variável Tempo
var_tmp = 'DISC_TMP'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importação da Base de Dados

# COMMAND ----------

#carregar o arquivo em formato tabela
base_dados = pd.read_parquet(file)
print("shape da Base de Dados:",base_dados.shape)

base_dados.head()

# COMMAND ----------

base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC'] = base_dados['DETALHES_CONTRATOS_BLOQUEIO1___DESC']

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gerando variáveis Categóricas

# COMMAND ----------

base_dados['DETALHES_CONTRATOS_CLASSE_gh30'] = np.where(base_dados['DETALHES_CONTRATOS_CLASSE'] == '121', 0,
    np.where(base_dados['DETALHES_CONTRATOS_CLASSE'] == '21', 1,
        np.where(base_dados['DETALHES_CONTRATOS_CLASSE'] == '701', 2,
            np.where(base_dados['DETALHES_CONTRATOS_CLASSE'] == '904', 3,
                np.where(base_dados['DETALHES_CONTRATOS_CLASSE'] == 'ACORDOS QUEBRADOS SLD DEVEDOR', 4,
                    np.where(base_dados['DETALHES_CONTRATOS_CLASSE'] == 'ACORDOS VIGENTES', 5,
                        np.where(base_dados['DETALHES_CONTRATOS_CLASSE'] == 'BLOQUEIO X', 6,
                            np.where(base_dados['DETALHES_CONTRATOS_CLASSE'] == 'C.L ACIMA 62 DIAS AC 20.00 SR', 7,
                                np.where(base_dados['DETALHES_CONTRATOS_CLASSE'] == 'EXCESSO DE LIMITE', 8,
                                    np.where(base_dados['DETALHES_CONTRATOS_CLASSE'] == 'RENEGOCIACAO EM ATRASO', 9,
                                        np.where(base_dados['DETALHES_CONTRATOS_CLASSE'] == 'outros', 10,
                                            0)))))))))))

base_dados['DETALHES_CONTRATOS_CLASSE_gh31'] = np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh30'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh30'] == 1, 1,
        np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh30'] == 2, 2,
            np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh30'] == 3, 3,
                np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh30'] == 4, 4,
                    np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh30'] == 5, 5,
                        np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh30'] == 6, 6,
                            np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh30'] == 7, 7,
                                np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh30'] == 8, 8,
                                    np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh30'] == 9, 9,
                                        np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh30'] == 10, 10,
                                            0)))))))))))
base_dados['DETALHES_CONTRATOS_CLASSE_gh32'] = np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh31'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh31'] == 1, 1,
        np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh31'] == 2, 2,
            np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh31'] == 3, 3,
                np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh31'] == 4, 4,
                    np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh31'] == 5, 5,
                        np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh31'] == 6, 6,
                            np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh31'] == 7, 7,
                                np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh31'] == 8, 8,
                                    np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh31'] == 9, 9,
                                        np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh31'] == 10, 10,
                                            0)))))))))))
base_dados['DETALHES_CONTRATOS_CLASSE_gh33'] = np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh32'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh32'] == 1, 1,
        np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh32'] == 2, 2,
            np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh32'] == 3, 3,
                np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh32'] == 4, 4,
                    np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh32'] == 5, 5,
                        np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh32'] == 6, 6,
                            np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh32'] == 7, 7,
                                np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh32'] == 8, 8,
                                    np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh32'] == 9, 9,
                                        np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh32'] == 10, 10,
                                            0)))))))))))
base_dados['DETALHES_CONTRATOS_CLASSE_gh34'] = np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh33'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh33'] == 1, 1,
        np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh33'] == 2, 1,
            np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh33'] == 3, 3,
                np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh33'] == 4, 4,
                    np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh33'] == 5, 10,
                        np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh33'] == 6, 10,
                            np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh33'] == 7, 7,
                                np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh33'] == 8, 3,
                                    np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh33'] == 9, 7,
                                        np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh33'] == 10, 10,
                                            0)))))))))))
base_dados['DETALHES_CONTRATOS_CLASSE_gh35'] = np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh34'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh34'] == 1, 1,
        np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh34'] == 3, 2,
            np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh34'] == 4, 3,
                np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh34'] == 7, 4,
                    np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh34'] == 10, 5,
                        0))))))
base_dados['DETALHES_CONTRATOS_CLASSE_gh36'] = np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh35'] == 0, 3,
    np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh35'] == 1, 1,
        np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh35'] == 2, 3,
            np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh35'] == 3, 2,
                np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh35'] == 4, 0,
                    np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh35'] == 5, 5,
                        0))))))
base_dados['DETALHES_CONTRATOS_CLASSE_gh37'] = np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh36'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh36'] == 1, 0,
        np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh36'] == 2, 0,
            np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh36'] == 3, 3,
                np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh36'] == 5, 3,
                    0)))))
base_dados['DETALHES_CONTRATOS_CLASSE_gh38'] = np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh37'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_CLASSE_gh37'] == 3, 1,
        0))
                                                        
                                                        
                                                        
base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh30'] = np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO'] == '1', 0,
    np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO'] == '108', 1,
        np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO'] == '112', 2,
            np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO'] == '130', 3,
                np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO'] == '136', 4,
                    np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO'] == '151', 5,
                        np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO'] == '17', 6,
                            np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO'] == '26', 7,
                                np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO'] == '33', 8,
                                    np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO'] == '8', 9,
                                        np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO'] == 'outros', 10,
                                            0)))))))))))
base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh31'] = np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh30'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh30'] == 1, 1,
        np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh30'] == 2, 1,
            np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh30'] == 3, 3,
                np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh30'] == 4, 4,
                    np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh30'] == 5, 5,
                        np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh30'] == 6, 6,
                            np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh30'] == 7, 7,
                                np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh30'] == 8, 8,
                                    np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh30'] == 9, 9,
                                        np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh30'] == 10, 10,
                                            0)))))))))))
base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh32'] = np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh31'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh31'] == 1, 1,
        np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh31'] == 3, 2,
            np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh31'] == 4, 3,
                np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh31'] == 5, 4,
                    np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh31'] == 6, 5,
                        np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh31'] == 7, 6,
                            np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh31'] == 8, 7,
                                np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh31'] == 9, 8,
                                    np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh31'] == 10, 9,
                                        0))))))))))
base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh33'] = np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh32'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh32'] == 1, 1,
        np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh32'] == 2, 2,
            np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh32'] == 3, 3,
                np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh32'] == 4, 4,
                    np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh32'] == 5, 5,
                        np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh32'] == 6, 6,
                            np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh32'] == 7, 7,
                                np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh32'] == 8, 8,
                                    np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh32'] == 9, 9,
                                        0))))))))))
base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh34'] = np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh33'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh33'] == 1, 1,
        np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh33'] == 2, 2,
            np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh33'] == 3, 9,
                np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh33'] == 4, 4,
                    np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh33'] == 5, 5,
                        np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh33'] == 6, 0,
                            np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh33'] == 7, 0,
                                np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh33'] == 8, 5,
                                    np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh33'] == 9, 9,
                                        0))))))))))
base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh35'] = np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh34'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh34'] == 1, 1,
        np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh34'] == 2, 2,
            np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh34'] == 4, 3,
                np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh34'] == 5, 4,
                    np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh34'] == 9, 5,
                        0))))))
base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh36'] = np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh35'] == 0, 1,
    np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh35'] == 1, 4,
        np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh35'] == 2, 2,
            np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh35'] == 3, 4,
                np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh35'] == 4, 0,
                    np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh35'] == 5, 2,
                        0))))))
base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh37'] = np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh36'] == 0, 1,
    np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh36'] == 1, 1,
        np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh36'] == 2, 2,
            np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh36'] == 4, 3,
                0))))
base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh38'] = np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh37'] == 1, 0,
    np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh37'] == 2, 1,
        np.where(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh37'] == 3, 2,
            0)))
             
             
             
base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh30'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC'] == 'ATIVO', 0,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC'] == 'CANCELAM', 1,
        np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC'] == 'CRELI', 2,
            np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC'] == 'EXCESSO', 3,
                np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC'] == 'RENEGOC', 4,
                    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC'] == 'outros', 5,
                        0))))))
base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh31'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh30'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh30'] == 1, 1,
        np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh30'] == 2, 2,
            np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh30'] == 3, 3,
                np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh30'] == 4, 4,
                    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh30'] == 5, 5,
                        0))))))
base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh32'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh31'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh31'] == 1, 1,
        np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh31'] == 2, 2,
            np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh31'] == 3, 3,
                np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh31'] == 4, 4,
                    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh31'] == 5, 5,
                        0))))))
base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh33'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh32'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh32'] == 1, 1,
        np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh32'] == 2, 2,
            np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh32'] == 3, 3,
                np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh32'] == 4, 4,
                    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh32'] == 5, 5,
                        0))))))
base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh34'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh33'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh33'] == 1, 1,
        np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh33'] == 2, 2,
            np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh33'] == 3, 3,
                np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh33'] == 4, 1,
                    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh33'] == 5, 5,
                        0))))))
base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh35'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh34'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh34'] == 1, 1,
        np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh34'] == 2, 2,
            np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh34'] == 3, 3,
                np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh34'] == 5, 4,
                    0)))))
base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh36'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh35'] == 0, 2,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh35'] == 1, 0,
        np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh35'] == 2, 1,
            np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh35'] == 3, 3,
                np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh35'] == 4, 3,
                    0)))))
base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh37'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh36'] == 0, 1,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh36'] == 1, 1,
        np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh36'] == 2, 2,
            np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh36'] == 3, 3,
                0))))
base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh38'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh37'] == 1, 0,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh37'] == 2, 1,
        np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh37'] == 3, 2,
            0)))
             
             
             
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
    np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh33'] == 1, 1,
        np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh33'] == 2, 0,
            np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh33'] == 3, 1,
                np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh33'] == 4, 0,
                    np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh33'] == 5, 5,
                        0))))))
base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh35'] = np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh34'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh34'] == 1, 1,
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
                                                               
                                                               
                                                               
base_dados['TIPO_TELEFONE_0_gh30'] = np.where(base_dados['TIPO_TELEFONE_0'] == False, 0,
    np.where(base_dados['TIPO_TELEFONE_0'] == True, 1,
        0))
base_dados['TIPO_TELEFONE_0_gh31'] = np.where(base_dados['TIPO_TELEFONE_0_gh30'] == 0, 0,
    np.where(base_dados['TIPO_TELEFONE_0_gh30'] == 1, 1,
        0))
base_dados['TIPO_TELEFONE_0_gh32'] = np.where(base_dados['TIPO_TELEFONE_0_gh31'] == 0, 0,
    np.where(base_dados['TIPO_TELEFONE_0_gh31'] == 1, 1,
        0))
base_dados['TIPO_TELEFONE_0_gh33'] = np.where(base_dados['TIPO_TELEFONE_0_gh32'] == 0, 0,
    np.where(base_dados['TIPO_TELEFONE_0_gh32'] == 1, 1,
        0))
base_dados['TIPO_TELEFONE_0_gh34'] = np.where(base_dados['TIPO_TELEFONE_0_gh33'] == 0, 0,
    np.where(base_dados['TIPO_TELEFONE_0_gh33'] == 1, 1,
        0))
base_dados['TIPO_TELEFONE_0_gh35'] = np.where(base_dados['TIPO_TELEFONE_0_gh34'] == 0, 0,
    np.where(base_dados['TIPO_TELEFONE_0_gh34'] == 1, 1,
        0))
base_dados['TIPO_TELEFONE_0_gh36'] = np.where(base_dados['TIPO_TELEFONE_0_gh35'] == 0, 1,
    np.where(base_dados['TIPO_TELEFONE_0_gh35'] == 1, 0,
        0))
base_dados['TIPO_TELEFONE_0_gh37'] = np.where(base_dados['TIPO_TELEFONE_0_gh36'] == 0, 0,
    np.where(base_dados['TIPO_TELEFONE_0_gh36'] == 1, 1,
        0))
base_dados['TIPO_TELEFONE_0_gh38'] = np.where(base_dados['TIPO_TELEFONE_0_gh37'] == 0, 0,
    np.where(base_dados['TIPO_TELEFONE_0_gh37'] == 1, 1,
        0))
                                              
         
                                              
                                              
base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh30'] = np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA'] == -8, 0,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA'] == -4, 1,
        np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA'] == 0, 2,
            np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA'] == 3, 3,
                np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA'] == 8, 4,
                    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA'] == 13, 5,
                        np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA'] == 18, 6,
                            np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA'] == 23, 7,
                                np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA'] == 775, 8,
                                    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA'] == 780, 9,
                                        np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA'] == 785, 10,
                                            np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA'] == 789, 11,
                                                np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA'] == 792, 12,
                                                    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA'] == 795, 13,
                                                        np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA'] == 800, 14,
                                                            np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA'] == 805, 15,
                                                                np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA'] == 807, 16,
                                                                    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA'] == 892, 17,
                                                                        np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA'] == 897, 18,
                                                                            np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA'] == 899, 19,
                                                                                np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA'] == 902, 20,
                                                                                    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA'] == 907, 21,
                                                                                        np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA'] == 912, 22,
                                                                                            np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA'] == 982, 23,
                                                                                                np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA'] == 987, 24,
                                                                                                    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA'] == 989, 25,
                                                                                                        np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA'] == 992, 26,
                                                                                                            np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA'] == 997, 27,
                                                                                                                np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA'] == 1001, 28,
                                                                                                                    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA'] == 1004, 29,
                                                                                                                        np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA'] == 1007, 30,
                                                                                                                            0)))))))))))))))))))))))))))))))
base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh31'] = np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh30'] == 0, 0,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh30'] == 1, 0,
        np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh30'] == 2, 2,
            np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh30'] == 3, 3,
                np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh30'] == 4, 4,
                    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh30'] == 5, 5,
                        np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh30'] == 6, 6,
                            np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh30'] == 7, 7,
                                np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh30'] == 8, 8,
                                    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh30'] == 9, 8,
                                        np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh30'] == 10, 8,
                                            np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh30'] == 11, 8,
                                                np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh30'] == 12, 8,
                                                    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh30'] == 13, 8,
                                                        np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh30'] == 14, 8,
                                                            np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh30'] == 15, 8,
                                                                np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh30'] == 16, 8,
                                                                    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh30'] == 17, 8,
                                                                        np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh30'] == 18, 8,
                                                                            np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh30'] == 19, 8,
                                                                                np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh30'] == 20, 8,
                                                                                    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh30'] == 21, 8,
                                                                                        np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh30'] == 22, 8,
                                                                                            np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh30'] == 23, 23,
                                                                                                np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh30'] == 24, 24,
                                                                                                    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh30'] == 25, 25,
                                                                                                        np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh30'] == 26, 26,
                                                                                                            np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh30'] == 27, 27,
                                                                                                                np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh30'] == 28, 28,
                                                                                                                    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh30'] == 29, 29,
                                                                                                                        np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh30'] == 30, 29,
                                                                                                                            0)))))))))))))))))))))))))))))))
base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh32'] = np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh31'] == 0, 0,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh31'] == 2, 1,
        np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh31'] == 3, 2,
            np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh31'] == 4, 3,
                np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh31'] == 5, 4,
                    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh31'] == 6, 5,
                        np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh31'] == 7, 6,
                            np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh31'] == 8, 7,
                                np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh31'] == 23, 8,
                                    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh31'] == 24, 9,
                                        np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh31'] == 25, 10,
                                            np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh31'] == 26, 11,
                                                np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh31'] == 27, 12,
                                                    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh31'] == 28, 13,
                                                        np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh31'] == 29, 14,
                                                            0)))))))))))))))
base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh33'] = np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh32'] == 0, 0,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh32'] == 1, 1,
        np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh32'] == 2, 2,
            np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh32'] == 3, 3,
                np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh32'] == 4, 4,
                    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh32'] == 5, 5,
                        np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh32'] == 6, 6,
                            np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh32'] == 7, 7,
                                np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh32'] == 8, 8,
                                    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh32'] == 9, 9,
                                        np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh32'] == 10, 10,
                                            np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh32'] == 11, 11,
                                                np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh32'] == 12, 12,
                                                    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh32'] == 13, 13,
                                                        np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh32'] == 14, 14,
                                                            0)))))))))))))))
base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh34'] = np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh33'] == 0, 0,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh33'] == 1, 3,
        np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh33'] == 2, 2,
            np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh33'] == 3, 3,
                np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh33'] == 4, 4,
                    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh33'] == 5, 5,
                        np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh33'] == 6, 5,
                            np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh33'] == 7, 7,
                                np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh33'] == 8, 11,
                                    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh33'] == 9, 9,
                                        np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh33'] == 10, 11,
                                            np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh33'] == 11, 11,
                                                np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh33'] == 12, 11,
                                                    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh33'] == 13, 11,
                                                        np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh33'] == 14, 14,
                                                            0)))))))))))))))
base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh35'] = np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh34'] == 0, 0,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh34'] == 2, 1,
        np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh34'] == 3, 2,
            np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh34'] == 4, 3,
                np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh34'] == 5, 4,
                    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh34'] == 7, 5,
                        np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh34'] == 9, 6,
                            np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh34'] == 11, 7,
                                np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh34'] == 14, 8,
                                    0)))))))))
base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh36'] = np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh35'] == 0, 5,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh35'] == 1, 4,
        np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh35'] == 2, 8,
            np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh35'] == 3, 5,
                np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh35'] == 4, 7,
                    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh35'] == 5, 0,
                        np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh35'] == 6, 2,
                            np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh35'] == 7, 3,
                                np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh35'] == 8, 0,
                                    0)))))))))
base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh37'] = np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh36'] == 0, 0,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh36'] == 2, 0,
        np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh36'] == 3, 2,
            np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh36'] == 4, 3,
                np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh36'] == 5, 4,
                    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh36'] == 7, 5,
                        np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh36'] == 8, 5,
                            0)))))))
base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh38'] = np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh37'] == 0, 0,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh37'] == 2, 1,
        np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh37'] == 3, 2,
            np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh37'] == 4, 3,
                np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh37'] == 5, 4,
                    0)))))
                     
                     
                     
                     
base_dados['TIPO_TELEFONE_4_gh30'] = np.where(base_dados['TIPO_TELEFONE_4'] == False, 0,
    np.where(base_dados['TIPO_TELEFONE_4'] == True, 1,
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
                                              
                                              
                                              
                                              
base_dados['TIPO_EMAIL_0_gh30'] = np.where(base_dados['TIPO_EMAIL_0'] == False, 0,
    np.where(base_dados['TIPO_EMAIL_0'] == True, 1,
        0))
base_dados['TIPO_EMAIL_0_gh31'] = np.where(base_dados['TIPO_EMAIL_0_gh30'] == 0, 0,
    np.where(base_dados['TIPO_EMAIL_0_gh30'] == 1, 1,
        0))
base_dados['TIPO_EMAIL_0_gh32'] = np.where(base_dados['TIPO_EMAIL_0_gh31'] == 0, 0,
    np.where(base_dados['TIPO_EMAIL_0_gh31'] == 1, 1,
        0))
base_dados['TIPO_EMAIL_0_gh33'] = np.where(base_dados['TIPO_EMAIL_0_gh32'] == 0, 0,
    np.where(base_dados['TIPO_EMAIL_0_gh32'] == 1, 1,
        0))
base_dados['TIPO_EMAIL_0_gh34'] = np.where(base_dados['TIPO_EMAIL_0_gh33'] == 0, 0,
    np.where(base_dados['TIPO_EMAIL_0_gh33'] == 1, 1,
        0))
base_dados['TIPO_EMAIL_0_gh35'] = np.where(base_dados['TIPO_EMAIL_0_gh34'] == 0, 0,
    np.where(base_dados['TIPO_EMAIL_0_gh34'] == 1, 1,
        0))
base_dados['TIPO_EMAIL_0_gh36'] = np.where(base_dados['TIPO_EMAIL_0_gh35'] == 0, 0,
    np.where(base_dados['TIPO_EMAIL_0_gh35'] == 1, 1,
        0))
base_dados['TIPO_EMAIL_0_gh37'] = np.where(base_dados['TIPO_EMAIL_0_gh36'] == 0, 0,
    np.where(base_dados['TIPO_EMAIL_0_gh36'] == 1, 1,
        0))
base_dados['TIPO_EMAIL_0_gh38'] = np.where(base_dados['TIPO_EMAIL_0_gh37'] == 0, 0,
    np.where(base_dados['TIPO_EMAIL_0_gh37'] == 1, 1,
        0))
                                           
                                           
                                           
                                           
base_dados['TIPO_TELEFONE_1_gh30'] = np.where(base_dados['TIPO_TELEFONE_1'] == False, 0,
    np.where(base_dados['TIPO_TELEFONE_1'] == True, 1,
        0))
base_dados['TIPO_TELEFONE_1_gh31'] = np.where(base_dados['TIPO_TELEFONE_1_gh30'] == 0, 0,
    np.where(base_dados['TIPO_TELEFONE_1_gh30'] == 1, 1,
        0))
base_dados['TIPO_TELEFONE_1_gh32'] = np.where(base_dados['TIPO_TELEFONE_1_gh31'] == 0, 0,
    np.where(base_dados['TIPO_TELEFONE_1_gh31'] == 1, 1,
        0))
base_dados['TIPO_TELEFONE_1_gh33'] = np.where(base_dados['TIPO_TELEFONE_1_gh32'] == 0, 0,
    np.where(base_dados['TIPO_TELEFONE_1_gh32'] == 1, 1,
        0))
base_dados['TIPO_TELEFONE_1_gh34'] = np.where(base_dados['TIPO_TELEFONE_1_gh33'] == 0, 0,
    np.where(base_dados['TIPO_TELEFONE_1_gh33'] == 1, 1,
        0))
base_dados['TIPO_TELEFONE_1_gh35'] = np.where(base_dados['TIPO_TELEFONE_1_gh34'] == 0, 0,
    np.where(base_dados['TIPO_TELEFONE_1_gh34'] == 1, 1,
        0))
base_dados['TIPO_TELEFONE_1_gh36'] = np.where(base_dados['TIPO_TELEFONE_1_gh35'] == 0, 1,
    np.where(base_dados['TIPO_TELEFONE_1_gh35'] == 1, 0,
        0))
base_dados['TIPO_TELEFONE_1_gh37'] = np.where(base_dados['TIPO_TELEFONE_1_gh36'] == 0, 0,
    np.where(base_dados['TIPO_TELEFONE_1_gh36'] == 1, 1,
        0))
base_dados['TIPO_TELEFONE_1_gh38'] = np.where(base_dados['TIPO_TELEFONE_1_gh37'] == 0, 0,
    np.where(base_dados['TIPO_TELEFONE_1_gh37'] == 1, 1,
        0))

base_dados['DETALHES_CONTRATOS_HISTORICO_FPD_gh30'] = np.where(base_dados['DETALHES_CONTRATOS_HISTORICO_FPD'] == '121', 0,
    np.where(base_dados['DETALHES_CONTRATOS_HISTORICO_FPD'] == '701', 1,
    np.where(base_dados['DETALHES_CONTRATOS_HISTORICO_FPD'] == '901', 2,
    np.where(base_dados['DETALHES_CONTRATOS_HISTORICO_FPD'] == '903', 3,
    np.where(base_dados['DETALHES_CONTRATOS_HISTORICO_FPD'] == '904', 4,
    np.where(base_dados['DETALHES_CONTRATOS_HISTORICO_FPD'] == 'outros', 5,
    0))))))

base_dados['DETALHES_CONTRATOS_HISTORICO_FPD_gh31'] = np.where(base_dados['DETALHES_CONTRATOS_HISTORICO_FPD_gh30'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_HISTORICO_FPD_gh30'] == 1, 1,
    np.where(base_dados['DETALHES_CONTRATOS_HISTORICO_FPD_gh30'] == 2, 2,
    np.where(base_dados['DETALHES_CONTRATOS_HISTORICO_FPD_gh30'] == 3, 2,
    np.where(base_dados['DETALHES_CONTRATOS_HISTORICO_FPD_gh30'] == 4, 4,
    np.where(base_dados['DETALHES_CONTRATOS_HISTORICO_FPD_gh30'] == 5, 5,
    0))))))

base_dados['DETALHES_CONTRATOS_HISTORICO_FPD_gh32'] = np.where(base_dados['DETALHES_CONTRATOS_HISTORICO_FPD_gh31'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_HISTORICO_FPD_gh31'] == 1, 1,
    np.where(base_dados['DETALHES_CONTRATOS_HISTORICO_FPD_gh31'] == 2, 2,
    np.where(base_dados['DETALHES_CONTRATOS_HISTORICO_FPD_gh31'] == 4, 3,
    np.where(base_dados['DETALHES_CONTRATOS_HISTORICO_FPD_gh31'] == 5, 4,
    0)))))

base_dados['DETALHES_CONTRATOS_HISTORICO_FPD_gh33'] = np.where(base_dados['DETALHES_CONTRATOS_HISTORICO_FPD_gh32'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_HISTORICO_FPD_gh32'] == 1, 1,
    np.where(base_dados['DETALHES_CONTRATOS_HISTORICO_FPD_gh32'] == 2, 2,
    np.where(base_dados['DETALHES_CONTRATOS_HISTORICO_FPD_gh32'] == 3, 3,
    np.where(base_dados['DETALHES_CONTRATOS_HISTORICO_FPD_gh32'] == 4, 4,
    0)))))

base_dados['DETALHES_CONTRATOS_HISTORICO_FPD_gh34'] = np.where(base_dados['DETALHES_CONTRATOS_HISTORICO_FPD_gh33'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_HISTORICO_FPD_gh33'] == 1, 4,
    np.where(base_dados['DETALHES_CONTRATOS_HISTORICO_FPD_gh33'] == 2, 3,
    np.where(base_dados['DETALHES_CONTRATOS_HISTORICO_FPD_gh33'] == 3, 3,
    np.where(base_dados['DETALHES_CONTRATOS_HISTORICO_FPD_gh33'] == 4, 4,
    0)))))

base_dados['DETALHES_CONTRATOS_HISTORICO_FPD_gh35'] = np.where(base_dados['DETALHES_CONTRATOS_HISTORICO_FPD_gh34'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_HISTORICO_FPD_gh34'] == 3, 1,
    np.where(base_dados['DETALHES_CONTRATOS_HISTORICO_FPD_gh34'] == 4, 2,
    0)))

base_dados['DETALHES_CONTRATOS_HISTORICO_FPD_gh36'] = np.where(base_dados['DETALHES_CONTRATOS_HISTORICO_FPD_gh35'] == 0, 1,
    np.where(base_dados['DETALHES_CONTRATOS_HISTORICO_FPD_gh35'] == 1, 2,
    np.where(base_dados['DETALHES_CONTRATOS_HISTORICO_FPD_gh35'] == 2, 0,
    0)))

base_dados['DETALHES_CONTRATOS_HISTORICO_FPD_gh37'] = np.where(base_dados['DETALHES_CONTRATOS_HISTORICO_FPD_gh36'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_HISTORICO_FPD_gh36'] == 1, 1,
    np.where(base_dados['DETALHES_CONTRATOS_HISTORICO_FPD_gh36'] == 2, 2,
    0)))

base_dados['DETALHES_CONTRATOS_HISTORICO_FPD_gh38'] = np.where(base_dados['DETALHES_CONTRATOS_HISTORICO_FPD_gh37'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_HISTORICO_FPD_gh37'] == 1, 1,
    np.where(base_dados['DETALHES_CONTRATOS_HISTORICO_FPD_gh37'] == 2, 2,
    0)))

         
         
         
base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh30'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2'] == '1-06DIAS', 0,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2'] == 'ANUID NP', 1,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2'] == 'ATIVO', 2,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2'] == 'COB> 07D', 3,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2'] == 'CRELI', 4,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2'] == 'EXCESSO', 5,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2'] == 'J', 6,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2'] == 'Q', 7,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2'] == 'QUEBRAAC', 8,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2'] == 'SPC/SER', 9,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2'] == 'outros', 10,
    0)))))))))))

base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh31'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh30'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh30'] == 1, 0,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh30'] == 2, 2,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh30'] == 3, 3,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh30'] == 4, 4,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh30'] == 5, 5,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh30'] == 6, 5,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh30'] == 7, 5,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh30'] == 8, 8,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh30'] == 9, 9,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh30'] == 10, 9,
    0)))))))))))

base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh32'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh31'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh31'] == 2, 1,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh31'] == 3, 2,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh31'] == 4, 3,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh31'] == 5, 4,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh31'] == 8, 5,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh31'] == 9, 6,
    0)))))))

base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh33'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh32'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh32'] == 1, 1,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh32'] == 2, 2,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh32'] == 3, 3,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh32'] == 4, 4,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh32'] == 5, 5,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh32'] == 6, 6,
    0)))))))

base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh34'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh33'] == 0, 6,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh33'] == 1, 3,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh33'] == 2, 6,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh33'] == 3, 3,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh33'] == 4, 4,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh33'] == 5, 5,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh33'] == 6, 6,
    0)))))))

base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh35'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh34'] == 3, 0,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh34'] == 4, 1,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh34'] == 5, 2,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh34'] == 6, 3,
    0))))

base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh36'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh35'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh35'] == 1, 2,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh35'] == 2, 0,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh35'] == 3, 3,
    0))))

base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh37'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh36'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh36'] == 2, 1,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh36'] == 3, 2,
    0)))

base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh38'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh37'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh37'] == 1, 1,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO2_gh37'] == 2, 2,
    0)))
  
         
base_dados['TIPO_TELEFONE_3_gh30'] = np.where(base_dados['TIPO_TELEFONE_3'] == False, 0,
    np.where(base_dados['TIPO_TELEFONE_3'] == True, 1,
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
         

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gerando variáveis numéricas contínuas

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parte 1 de 3

# COMMAND ----------

base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] = np.log(base_dados['DETALHES_CLIENTES_SCORE_CARGA'])
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] == 0, -1, base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'])
base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] = base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'].fillna(-2)

base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13'] = np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] == 0 , -1.0,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] > 0.0, base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] <= 1.0986122886681098), 1.0,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] > 1.0986122886681098, base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] <= 1.6094379124341003), 2.0,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] > 1.6094379124341003, base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] <= 2.1972245773362196), 3.0,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] > 2.1972245773362196, base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] <= 2.833213344056216), 4.0,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] > 2.833213344056216, base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] <= 3.4339872044851463), 5.0,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] > 3.4339872044851463, base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] <= 3.970291913552122), 6.0,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] > 3.970291913552122, base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] <= 4.564348191467836), 7.0,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] > 4.564348191467836, base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] <= 5.147494476813453), 8.0,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] > 5.147494476813453, base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] <= 5.723585101952381), 9.0,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] > 5.723585101952381, base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] <= 6.2878585601617845), 10.0,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] > 6.2878585601617845, base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] <= 6.733401891837359), 11.0,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L'] > 6.733401891837359, 12.0,-1)))))))))))))

base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_1'] = np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13'] == -2.0, 2,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13'] == -1.0, 2,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13'] == 1.0, 2,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13'] == 2.0, 2,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13'] == 3.0, 2,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13'] == 4.0, 2,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13'] == 5.0, 2,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13'] == 6.0, 2,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13'] == 7.0, 1,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13'] == 8.0, 1,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13'] == 9.0, 1,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13'] == 10.0, 0,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13'] == 11.0, 1,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13'] == 12.0, 2,-2))))))))))))))

base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_2'] = np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_1'] == 0, 0,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_1'] == 1, 2,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_1'] == 2, 0,-2)))

base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1'] = np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_2'] == 0, 0,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1_2'] == 2, 1,-2))




base_dados['DETALHES_CLIENTES_SCORE_CARGA__C'] = np.cos(base_dados['DETALHES_CLIENTES_SCORE_CARGA'])
np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C'] == 0, -1, base_dados['DETALHES_CLIENTES_SCORE_CARGA__C'])
base_dados['DETALHES_CLIENTES_SCORE_CARGA__C'] = base_dados['DETALHES_CLIENTES_SCORE_CARGA__C'].fillna(-2)

base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13'] = np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C'] <= 0.22377033018717848, 6.0,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C'] > 0.22377033018717848, base_dados['DETALHES_CLIENTES_SCORE_CARGA__C'] <= 0.4322051285150291), 7.0,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C'] > 0.4322051285150291, base_dados['DETALHES_CLIENTES_SCORE_CARGA__C'] <= 0.6669605223036468), 8.0,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C'] > 0.6669605223036468, base_dados['DETALHES_CLIENTES_SCORE_CARGA__C'] <= 0.8390879278598296), 9.0,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C'] > 0.8390879278598296, base_dados['DETALHES_CLIENTES_SCORE_CARGA__C'] <= 0.9379947521194415), 10.0,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C'] > 0.9379947521194415, 11.0,6))))))

base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_1'] = np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13'] == -2.0, 0,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13'] == 6.0, 1,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13'] == 7.0, 1,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13'] == 8.0, 0,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13'] == 9.0, 1,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13'] == 10.0, 2,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13'] == 11.0, 2,-2)))))))

base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_2'] = np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_1'] == 0, 1,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_1'] == 1, 0,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_1'] == 2, 1,-2)))

base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1'] = np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_2'] == 0, 0,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_2'] == 1, 1,-2))




base_dados['DOCUMENTO_PESSOA__L'] = np.log(base_dados['DOCUMENTO_PESSOA'])
np.where(base_dados['DOCUMENTO_PESSOA__L'] == 0, -1, base_dados['DOCUMENTO_PESSOA__L'])
base_dados['DOCUMENTO_PESSOA__L'] = base_dados['DOCUMENTO_PESSOA__L'].fillna(-2)

base_dados['DOCUMENTO_PESSOA__L__p_13'] = np.where(base_dados['DOCUMENTO_PESSOA__L'] <= 22.04589020817958, 0.0,
    np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__L'] > 22.04589020817958, base_dados['DOCUMENTO_PESSOA__L'] <= 22.57431734763722), 1.0,
    np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__L'] > 22.57431734763722, base_dados['DOCUMENTO_PESSOA__L'] <= 22.91321723505565), 2.0,
    np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__L'] > 22.91321723505565, base_dados['DOCUMENTO_PESSOA__L'] <= 23.239250020232852), 3.0,
    np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__L'] > 23.239250020232852, base_dados['DOCUMENTO_PESSOA__L'] <= 23.51263115924523), 4.0,
    np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__L'] > 23.51263115924523, base_dados['DOCUMENTO_PESSOA__L'] <= 23.845776336872394), 5.0,
    np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__L'] > 23.845776336872394, base_dados['DOCUMENTO_PESSOA__L'] <= 24.345751738495387), 8.0,
    np.where(base_dados['DOCUMENTO_PESSOA__L'] > 24.345751738495387, 11.0,0))))))))

base_dados['DOCUMENTO_PESSOA__L__p_13_g_1_1'] = np.where(base_dados['DOCUMENTO_PESSOA__L__p_13'] == -2.0, 3,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_13'] == 0.0, 1,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_13'] == 1.0, 4,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_13'] == 2.0, 3,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_13'] == 3.0, 0,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_13'] == 4.0, 4,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_13'] == 5.0, 1,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_13'] == 8.0, 0,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_13'] == 11.0, 2,-2)))))))))

base_dados['DOCUMENTO_PESSOA__L__p_13_g_1_2'] = np.where(base_dados['DOCUMENTO_PESSOA__L__p_13_g_1_1'] == 0, 1,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_13_g_1_1'] == 1, 3,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_13_g_1_1'] == 2, 0,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_13_g_1_1'] == 3, 4,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_13_g_1_1'] == 4, 1,-2)))))

base_dados['DOCUMENTO_PESSOA__L__p_13_g_1'] = np.where(base_dados['DOCUMENTO_PESSOA__L__p_13_g_1_2'] == 0, 0,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_13_g_1_2'] == 1, 1,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_13_g_1_2'] == 3, 2,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_13_g_1_2'] == 4, 3,-2))))




base_dados['DOCUMENTO_PESSOA__L'] = np.log(base_dados['DOCUMENTO_PESSOA'])
np.where(base_dados['DOCUMENTO_PESSOA__L'] == 0, -1, base_dados['DOCUMENTO_PESSOA__L'])
base_dados['DOCUMENTO_PESSOA__L'] = base_dados['DOCUMENTO_PESSOA__L'].fillna(-2)

base_dados['DOCUMENTO_PESSOA__L__p_3'] = np.where(base_dados['DOCUMENTO_PESSOA__L'] <= 23.268805323575926, 0.0,
    np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__L'] > 23.268805323575926, base_dados['DOCUMENTO_PESSOA__L'] <= 24.251504534514545), 1.0,
    np.where(base_dados['DOCUMENTO_PESSOA__L'] > 24.251504534514545, 2.0,0)))

base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_1'] = np.where(base_dados['DOCUMENTO_PESSOA__L__p_3'] == 0, 0,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3'] == 1, 1,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3'] == 2, 1,-2)))

base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_2'] = np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_1'] == 0, 1,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_1'] == 1, 0,-2))

base_dados['DOCUMENTO_PESSOA__L__p_3_g_1'] = np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_2'] == 0, 0,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_2'] == 1, 1,-2))




base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__pe_15'] = np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO'] == 0 , -1.0,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO'] > 0.0, base_dados['DETALHES_DIVIDAS_TAXA_SERVICO'] <= 8.0), 0.0,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO'] > 8.0, base_dados['DETALHES_DIVIDAS_TAXA_SERVICO'] <= 16.46), 1.0,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO'] > 16.46, base_dados['DETALHES_DIVIDAS_TAXA_SERVICO'] <= 24.63), 2.0,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO'] > 24.63, base_dados['DETALHES_DIVIDAS_TAXA_SERVICO'] <= 33.24), 3.0,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO'] > 33.24, base_dados['DETALHES_DIVIDAS_TAXA_SERVICO'] <= 41.69), 4.0,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO'] > 41.69, base_dados['DETALHES_DIVIDAS_TAXA_SERVICO'] <= 50.03), 5.0,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO'] > 50.03, base_dados['DETALHES_DIVIDAS_TAXA_SERVICO'] <= 58.38), 6.0,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO'] > 58.38, base_dados['DETALHES_DIVIDAS_TAXA_SERVICO'] <= 66.73), 7.0,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO'] > 66.73, base_dados['DETALHES_DIVIDAS_TAXA_SERVICO'] <= 74.99), 8.0,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO'] > 74.99, base_dados['DETALHES_DIVIDAS_TAXA_SERVICO'] <= 83.04), 9.0,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO'] > 83.04, base_dados['DETALHES_DIVIDAS_TAXA_SERVICO'] <= 91.22), 10.0,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO'] > 91.22, base_dados['DETALHES_DIVIDAS_TAXA_SERVICO'] <= 99.99), 11.0,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO'] > 99.99, base_dados['DETALHES_DIVIDAS_TAXA_SERVICO'] <= 108.28), 12.0,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO'] > 108.28, base_dados['DETALHES_DIVIDAS_TAXA_SERVICO'] <= 115.27), 13.0,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO'] > 115.27, 14.0,-1))))))))))))))))

base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__pe_15_g_1_1'] = np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__pe_15'] == -2.0, 5,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__pe_15'] == -1.0, 5,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__pe_15'] == 0.0, 4,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__pe_15'] == 1.0, 5,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__pe_15'] == 2.0, 5,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__pe_15'] == 3.0, 4,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__pe_15'] == 4.0, 3,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__pe_15'] == 5.0, 0,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__pe_15'] == 6.0, 2,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__pe_15'] == 7.0, 1,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__pe_15'] == 8.0, 4,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__pe_15'] == 9.0, 3,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__pe_15'] == 10.0, 4,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__pe_15'] == 11.0, 5,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__pe_15'] == 12.0, 4,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__pe_15'] == 13.0, 4,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__pe_15'] == 14.0, 4,-2)))))))))))))))))

base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__pe_15_g_1_2'] = np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__pe_15_g_1_1'] == 0, 2,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__pe_15_g_1_1'] == 1, 4,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__pe_15_g_1_1'] == 2, 2,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__pe_15_g_1_1'] == 3, 1,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__pe_15_g_1_1'] == 4, 4,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__pe_15_g_1_1'] == 5, 0,-2))))))

base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__pe_15_g_1'] = np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__pe_15_g_1_2'] == 0, 0,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__pe_15_g_1_2'] == 1, 1,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__pe_15_g_1_2'] == 2, 2,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__pe_15_g_1_2'] == 4, 3,-2))))





base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L'] = np.log(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO'])
np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L'] == 0, -1, base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L'])
base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L'] = base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L'].fillna(-2)

base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15'] = np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L'] >= -1.0, base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L'] <= 3.1945831322991562), 0.0,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L'] > 3.1945831322991562, base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L'] <= 3.5360196074696004), 1.0,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L'] > 3.5360196074696004, base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L'] <= 3.6985822295753215), 2.0,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L'] > 3.6985822295753215, base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L'] <= 3.805995600782616), 3.0,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L'] > 3.805995600782616, base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L'] <= 3.9895394478559743), 9.0,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L'] > 3.9895394478559743, base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L'] <= 4.040767961318708), 10.0,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L'] > 4.040767961318708, base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L'] <= 4.1121844803504315), 11.0,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L'] > 4.1121844803504315, base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L'] <= 4.196600262926448), 12.0,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L'] > 4.196600262926448, base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L'] <= 4.317354771313298), 13.0,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L'] > 4.317354771313298, 14.0,0))))))))))

base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1_1'] = np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15'] == -2.0, 3,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15'] == 0.0, 3,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15'] == 1.0, 3,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15'] == 2.0, 3,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15'] == 3.0, 0,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15'] == 9.0, 1,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15'] == 10.0, 3,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15'] == 11.0, 1,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15'] == 12.0, 2,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15'] == 13.0, 2,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15'] == 14.0, 1,-2)))))))))))

base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1_2'] = np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1_1'] == 0, 0,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1_1'] == 1, 2,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1_1'] == 2, 3,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1_1'] == 3, 1,-2))))

base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1'] = np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1_2'] == 0, 0,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1_2'] == 1, 1,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1_2'] == 2, 2,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1_2'] == 3, 3,-2))))




base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L'] = np.log(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO'])
np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L'] == 0, -1, base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L'])
base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L'] = base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L'].fillna(-2)

base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25'] = np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L'] >= -1.0, base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L'] <= 5.265535877023657), 0.0,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L'] > 5.265535877023657, base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L'] <= 5.709764546333749), 1.0,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L'] > 5.709764546333749, base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L'] <= 6.012345024687337), 2.0,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L'] > 6.012345024687337, base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L'] <= 6.336949617277547), 3.0,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L'] > 6.336949617277547, base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L'] <= 6.503749106110217), 4.0,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L'] > 6.503749106110217, base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L'] <= 6.607258109483513), 5.0,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L'] > 6.607258109483513, base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L'] <= 6.7093897024685), 6.0,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L'] > 6.7093897024685, base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L'] <= 6.782328101173997), 7.0,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L'] > 6.782328101173997, base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L'] <= 6.981953247086102), 10.0,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L'] > 6.981953247086102, base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L'] <= 7.204171596619904), 13.0,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L'] > 7.204171596619904, base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L'] <= 7.47178417197192), 17.0,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L'] > 7.47178417197192, base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L'] <= 7.64278309350581), 19.0,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L'] > 7.64278309350581, base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L'] <= 7.757029756003505), 20.0,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L'] > 7.757029756003505, base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L'] <= 7.910832695937919), 21.0,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L'] > 7.910832695937919, base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L'] <= 8.1240113343499), 22.0,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L'] > 8.1240113343499, base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L'] <= 8.404955750485453), 23.0,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L'] > 8.404955750485453, 24.0,0)))))))))))))))))

base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_1'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25'] == -2.0, 0,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25'] == 0.0, 0,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25'] == 1.0, 0,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25'] == 2.0, 0,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25'] == 3.0, 0,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25'] == 4.0, 1,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25'] == 5.0, 0,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25'] == 6.0, 1,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25'] == 7.0, 1,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25'] == 10.0, 0,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25'] == 13.0, 0,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25'] == 17.0, 0,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25'] == 19.0, 1,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25'] == 20.0, 1,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25'] == 21.0, 1,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25'] == 22.0, 0,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25'] == 23.0, 1,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25'] == 24.0, 0,-2))))))))))))))))))

base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_2'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_1'] == 0, 1,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_1'] == 1, 0,-2))

base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_2'] == 0, 0,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_2'] == 1, 1,-2))




base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__S'] = np.sin(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO'])
np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__S'] == 0, -1, base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__S'])
base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__S'] = base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__S'].fillna(-2)

base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__S__p_3'] = np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__S'] >= -1.0, base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__S'] <= 0.4761192476593861), 1.0,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__S'] > 0.4761192476593861, 2.0,1))

base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__S__p_3_g_1_1'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__S__p_3'] == -2.0, 2,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__S__p_3'] == 1.0, 0,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__S__p_3'] == 2.0, 1,-2)))

base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__S__p_3_g_1_2'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__S__p_3_g_1_1'] == 0, 2,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__S__p_3_g_1_1'] == 1, 0,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__S__p_3_g_1_1'] == 2, 1,-2)))

base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__S__p_3_g_1'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__S__p_3_g_1_2'] == 0, 0,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__S__p_3_g_1_2'] == 1, 1,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__S__p_3_g_1_2'] == 2, 2,-2)))





base_dados['AGING__p_7'] = np.where(base_dados['AGING'] <= 115, 0,
    np.where(np.bitwise_and(base_dados['AGING'] > 115, base_dados['AGING'] <= 171), 1,
    np.where(np.bitwise_and(base_dados['AGING'] > 171, base_dados['AGING'] <= 225), 2,
    np.where(np.bitwise_and(base_dados['AGING'] > 225, base_dados['AGING'] <= 357), 3,
    np.where(np.bitwise_and(base_dados['AGING'] > 357, base_dados['AGING'] <= 1311), 4,
    np.where(np.bitwise_and(base_dados['AGING'] > 1311, base_dados['AGING'] <= 1810), 5,
    np.where(base_dados['AGING'] > 1810, 6,0)))))))

base_dados['AGING__p_7_g_1_1'] = np.where(base_dados['AGING__p_7'] == 0, 0,
    np.where(base_dados['AGING__p_7'] == 1, 2,
    np.where(base_dados['AGING__p_7'] == 2, 1,
    np.where(base_dados['AGING__p_7'] == 3, 1,
    np.where(base_dados['AGING__p_7'] == 4, 3,
    np.where(base_dados['AGING__p_7'] == 5, 3,
    np.where(base_dados['AGING__p_7'] == 6, 4,-2)))))))

base_dados['AGING__p_7_g_1_2'] = np.where(base_dados['AGING__p_7_g_1_1'] == 0, 4,
    np.where(base_dados['AGING__p_7_g_1_1'] == 1, 2,
    np.where(base_dados['AGING__p_7_g_1_1'] == 2, 3,
    np.where(base_dados['AGING__p_7_g_1_1'] == 3, 1,
    np.where(base_dados['AGING__p_7_g_1_1'] == 4, 0,-2)))))

base_dados['AGING__p_7_g_1'] = np.where(base_dados['AGING__p_7_g_1_2'] == 0, 0,
    np.where(base_dados['AGING__p_7_g_1_2'] == 1, 1,
    np.where(base_dados['AGING__p_7_g_1_2'] == 2, 2,
    np.where(base_dados['AGING__p_7_g_1_2'] == 3, 3,
    np.where(base_dados['AGING__p_7_g_1_2'] == 4, 4,-2)))))



base_dados['AGING__pe_20'] = np.where(base_dados['AGING'] <= 130.0, 0.0,
    np.where(np.bitwise_and(base_dados['AGING'] > 130.0, base_dados['AGING'] <= 266.0), 1.0,
    np.where(np.bitwise_and(base_dados['AGING'] > 266.0, base_dados['AGING'] <= 399.0), 2.0,
    np.where(np.bitwise_and(base_dados['AGING'] > 399.0, base_dados['AGING'] <= 505.0), 3.0,
    np.where(np.bitwise_and(base_dados['AGING'] > 505.0, base_dados['AGING'] <= 662.0), 4.0,
    np.where(np.bitwise_and(base_dados['AGING'] > 662.0, base_dados['AGING'] <= 734.0), 5.0,
    np.where(np.bitwise_and(base_dados['AGING'] > 734.0, base_dados['AGING'] <= 1073.0), 7.0,
    np.where(np.bitwise_and(base_dados['AGING'] > 1073.0, base_dados['AGING'] <= 1206.0), 8.0,
    np.where(np.bitwise_and(base_dados['AGING'] > 1206.0, base_dados['AGING'] <= 1342.0), 9.0,
    np.where(np.bitwise_and(base_dados['AGING'] > 1342.0, base_dados['AGING'] <= 1479.0), 10.0,
    np.where(np.bitwise_and(base_dados['AGING'] > 1479.0, base_dados['AGING'] <= 1614.0), 11.0,
    np.where(np.bitwise_and(base_dados['AGING'] > 1614.0, base_dados['AGING'] <= 1747.0), 12.0,
    np.where(np.bitwise_and(base_dados['AGING'] > 1747.0, base_dados['AGING'] <= 1870.0), 13.0,
    np.where(np.bitwise_and(base_dados['AGING'] > 1870.0, base_dados['AGING'] <= 2016.0), 14.0,
    np.where(np.bitwise_and(base_dados['AGING'] > 2016.0, base_dados['AGING'] <= 2146.0), 15.0,
    np.where(np.bitwise_and(base_dados['AGING'] > 2146.0, base_dados['AGING'] <= 2256.0), 16.0,
    np.where(np.bitwise_and(base_dados['AGING'] > 2256.0, base_dados['AGING'] <= 2407.0), 17.0,
    np.where(np.bitwise_and(base_dados['AGING'] > 2407.0, base_dados['AGING'] <= 2539.0), 18.0,
    np.where(base_dados['AGING'] > 2539.0, 19.0,0)))))))))))))))))))

base_dados['AGING__pe_20_g_1_1'] = np.where(base_dados['AGING__pe_20'] == -2.0, 4,
    np.where(base_dados['AGING__pe_20'] == 0.0, 0,
    np.where(base_dados['AGING__pe_20'] == 1.0, 1,
    np.where(base_dados['AGING__pe_20'] == 2.0, 2,
    np.where(base_dados['AGING__pe_20'] == 3.0, 2,
    np.where(base_dados['AGING__pe_20'] == 4.0, 1,
    np.where(base_dados['AGING__pe_20'] == 5.0, 3,
    np.where(base_dados['AGING__pe_20'] == 7.0, 4,
    np.where(base_dados['AGING__pe_20'] == 8.0, 4,
    np.where(base_dados['AGING__pe_20'] == 9.0, 3,
    np.where(base_dados['AGING__pe_20'] == 10.0, 4,
    np.where(base_dados['AGING__pe_20'] == 11.0, 2,
    np.where(base_dados['AGING__pe_20'] == 12.0, 3,
    np.where(base_dados['AGING__pe_20'] == 13.0, 4,
    np.where(base_dados['AGING__pe_20'] == 14.0, 4,
    np.where(base_dados['AGING__pe_20'] == 15.0, 4,
    np.where(base_dados['AGING__pe_20'] == 16.0, 3,
    np.where(base_dados['AGING__pe_20'] == 17.0, 3,
    np.where(base_dados['AGING__pe_20'] == 18.0, 4,
    np.where(base_dados['AGING__pe_20'] == 19.0, 4,-2))))))))))))))))))))

base_dados['AGING__pe_20_g_1_2'] = np.where(base_dados['AGING__pe_20_g_1_1'] == 0, 4,
    np.where(base_dados['AGING__pe_20_g_1_1'] == 1, 3,
    np.where(base_dados['AGING__pe_20_g_1_1'] == 2, 2,
    np.where(base_dados['AGING__pe_20_g_1_1'] == 3, 1,
    np.where(base_dados['AGING__pe_20_g_1_1'] == 4, 0,-2)))))

base_dados['AGING__pe_20_g_1'] = np.where(base_dados['AGING__pe_20_g_1_2'] == 0, 0,
    np.where(base_dados['AGING__pe_20_g_1_2'] == 1, 1,
    np.where(base_dados['AGING__pe_20_g_1_2'] == 2, 2,
    np.where(base_dados['AGING__pe_20_g_1_2'] == 3, 3,
    np.where(base_dados['AGING__pe_20_g_1_2'] == 4, 4,-2)))))





base_dados['DETALHES_DIVIDAS_VALOR_JUROS__pe_3'] = np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS'] == 0 , -1.0,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_VALOR_JUROS'] > 0.0, base_dados['DETALHES_DIVIDAS_VALOR_JUROS'] <= 381.18), 0.0,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_VALOR_JUROS'] > 381.18, base_dados['DETALHES_DIVIDAS_VALOR_JUROS'] <= 752.42), 1.0,
    np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS'] > 752.42, 2.0,-1))))

base_dados['DETALHES_DIVIDAS_VALOR_JUROS__pe_3_g_1_1'] = np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__pe_3'] == -2.0, 1,
    np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__pe_3'] == -1.0, 0,
    np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__pe_3'] == 0.0, 0,
    np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__pe_3'] == 1.0, 1,
    np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__pe_3'] == 2.0, 0,-2)))))

base_dados['DETALHES_DIVIDAS_VALOR_JUROS__pe_3_g_1_2'] = np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__pe_3_g_1_1'] == 0, 1,
    np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__pe_3_g_1_1'] == 1, 0,-2))

base_dados['DETALHES_DIVIDAS_VALOR_JUROS__pe_3_g_1'] = np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__pe_3_g_1_2'] == 0, 0,
    np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__pe_3_g_1_2'] == 1, 1,-2))





base_dados['DETALHES_DIVIDAS_VALOR_JUROS__L'] = np.log(base_dados['DETALHES_DIVIDAS_VALOR_JUROS'])
np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__L'] == 0, -1, base_dados['DETALHES_DIVIDAS_VALOR_JUROS__L'])
base_dados['DETALHES_DIVIDAS_VALOR_JUROS__L'] = base_dados['DETALHES_DIVIDAS_VALOR_JUROS__L'].fillna(-2)

base_dados['DETALHES_DIVIDAS_VALOR_JUROS__L__pe_3'] = np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__L'] >= -1.0, base_dados['DETALHES_DIVIDAS_VALOR_JUROS__L'] <= 2.5817308344235403), 0.0,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__L'] > 2.5817308344235403, base_dados['DETALHES_DIVIDAS_VALOR_JUROS__L'] <= 5.3063847376257955), 1.0,
    np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__L'] > 5.3063847376257955, 2.0,0)))

base_dados['DETALHES_DIVIDAS_VALOR_JUROS__L__pe_3_g_1_1'] = np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__L__pe_3'] == -2.0, 1,
    np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__L__pe_3'] == 0.0, 2,
    np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__L__pe_3'] == 1.0, 0,
    np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__L__pe_3'] == 2.0, 2,-2))))

base_dados['DETALHES_DIVIDAS_VALOR_JUROS__L__pe_3_g_1_2'] = np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__L__pe_3_g_1_1'] == 0, 1,
    np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__L__pe_3_g_1_1'] == 1, 2,
    np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__L__pe_3_g_1_1'] == 2, 0,-2)))

base_dados['DETALHES_DIVIDAS_VALOR_JUROS__L__pe_3_g_1'] = np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__L__pe_3_g_1_2'] == 0, 0,
    np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__L__pe_3_g_1_2'] == 1, 1,
    np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__L__pe_3_g_1_2'] == 2, 2,-2)))




base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__pe_17'] = np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO'] == 0 , -1.0,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO'] > 0.0, base_dados['DETALHES_DIVIDAS_TAXA_ATRASO'] <= 4.6), 0.0,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO'] > 4.6, base_dados['DETALHES_DIVIDAS_TAXA_ATRASO'] <= 9.16), 1.0,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO'] > 9.16, base_dados['DETALHES_DIVIDAS_TAXA_ATRASO'] <= 13.79), 2.0,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO'] > 13.79, base_dados['DETALHES_DIVIDAS_TAXA_ATRASO'] <= 18.35), 3.0,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO'] > 18.35, base_dados['DETALHES_DIVIDAS_TAXA_ATRASO'] <= 22.98), 4.0,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO'] > 22.98, base_dados['DETALHES_DIVIDAS_TAXA_ATRASO'] <= 27.41), 5.0,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO'] > 27.41, base_dados['DETALHES_DIVIDAS_TAXA_ATRASO'] <= 32.16), 6.0,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO'] > 32.16, base_dados['DETALHES_DIVIDAS_TAXA_ATRASO'] <= 36.62), 7.0,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO'] > 36.62, base_dados['DETALHES_DIVIDAS_TAXA_ATRASO'] <= 40.99), 8.0,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO'] > 40.99, base_dados['DETALHES_DIVIDAS_TAXA_ATRASO'] <= 44.11), 9.0,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO'] > 44.11, base_dados['DETALHES_DIVIDAS_TAXA_ATRASO'] <= 49.85), 10.0,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO'] > 49.85, base_dados['DETALHES_DIVIDAS_TAXA_ATRASO'] <= 54.19), 11.0,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO'] > 54.19, base_dados['DETALHES_DIVIDAS_TAXA_ATRASO'] <= 59.32), 12.0,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO'] > 59.32, base_dados['DETALHES_DIVIDAS_TAXA_ATRASO'] <= 62.53), 13.0,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO'] > 62.53, base_dados['DETALHES_DIVIDAS_TAXA_ATRASO'] <= 66.7), 14.0,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO'] > 66.7, base_dados['DETALHES_DIVIDAS_TAXA_ATRASO'] <= 72.69), 15.0,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO'] > 72.69, 16.0,-1))))))))))))))))))

base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__pe_17_g_1_1'] = np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__pe_17'] == -2.0, 3,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__pe_17'] == -1.0, 3,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__pe_17'] == 0.0, 0,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__pe_17'] == 1.0, 2,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__pe_17'] == 2.0, 0,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__pe_17'] == 3.0, 1,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__pe_17'] == 4.0, 2,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__pe_17'] == 5.0, 2,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__pe_17'] == 6.0, 3,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__pe_17'] == 7.0, 3,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__pe_17'] == 8.0, 3,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__pe_17'] == 9.0, 3,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__pe_17'] == 10.0, 3,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__pe_17'] == 11.0, 3,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__pe_17'] == 12.0, 2,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__pe_17'] == 13.0, 3,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__pe_17'] == 14.0, 3,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__pe_17'] == 15.0, 3,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__pe_17'] == 16.0, 3,-2)))))))))))))))))))

base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__pe_17_g_1_2'] = np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__pe_17_g_1_1'] == 0, 1,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__pe_17_g_1_1'] == 1, 3,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__pe_17_g_1_1'] == 2, 0,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__pe_17_g_1_1'] == 3, 1,-2))))

base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__pe_17_g_1'] = np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__pe_17_g_1_2'] == 0, 0,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__pe_17_g_1_2'] == 1, 1,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__pe_17_g_1_2'] == 3, 2,-2)))




base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T'] = np.tan(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO'])
np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T'] == 0, -1, base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T'])
base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T'] = base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T'].fillna(-2)

base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3'] = np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T'] <= 0.5185882000695138, 1.0,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T'] > 0.5185882000695138, 2.0,1))

base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_1'] = np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3'] == -2.0, 1,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3'] == 1.0, 1,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3'] == 2.0, 0,-2)))

base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_2'] = np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_1'] == 0, 0,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_1'] == 1, 1,-2))

base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1'] = np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_2'] == 0, 0,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_2'] == 1, 1,-2))



base_dados['DETALHES_CLIENTES_VALOR_FATURA__L'] = np.log(base_dados['DETALHES_CLIENTES_VALOR_FATURA'])
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__L'] == 0, -1, base_dados['DETALHES_CLIENTES_VALOR_FATURA__L'])
base_dados['DETALHES_CLIENTES_VALOR_FATURA__L'] = base_dados['DETALHES_CLIENTES_VALOR_FATURA__L'].fillna(-2)

base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__L'] <= 5.5032562132074, 0.0,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__L'] > 5.5032562132074, base_dados['DETALHES_CLIENTES_VALOR_FATURA__L'] <= 5.837030493800918), 1.0,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__L'] > 5.837030493800918, base_dados['DETALHES_CLIENTES_VALOR_FATURA__L'] <= 6.136495194146357), 2.0,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__L'] > 6.136495194146357, base_dados['DETALHES_CLIENTES_VALOR_FATURA__L'] <= 6.345004582351227), 3.0,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__L'] > 6.345004582351227, base_dados['DETALHES_CLIENTES_VALOR_FATURA__L'] <= 6.492058000307724), 4.0,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__L'] > 6.492058000307724, base_dados['DETALHES_CLIENTES_VALOR_FATURA__L'] <= 6.580458815896264), 5.0,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__L'] > 6.580458815896264, base_dados['DETALHES_CLIENTES_VALOR_FATURA__L'] <= 6.651739818142582), 6.0,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__L'] > 6.651739818142582, base_dados['DETALHES_CLIENTES_VALOR_FATURA__L'] <= 7.202370759711681), 17.0,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__L'] > 7.202370759711681, base_dados['DETALHES_CLIENTES_VALOR_FATURA__L'] <= 7.7457903720901555), 27.0,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__L'] > 7.7457903720901555, base_dados['DETALHES_CLIENTES_VALOR_FATURA__L'] <= 7.867450268463577), 28.0,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__L'] > 7.867450268463577, base_dados['DETALHES_CLIENTES_VALOR_FATURA__L'] <= 7.971006330462178), 29.0,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__L'] > 7.971006330462178, base_dados['DETALHES_CLIENTES_VALOR_FATURA__L'] <= 8.136930809670726), 30.0,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__L'] > 8.136930809670726, base_dados['DETALHES_CLIENTES_VALOR_FATURA__L'] <= 8.336095685166656), 31.0,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__L'] > 8.336095685166656, base_dados['DETALHES_CLIENTES_VALOR_FATURA__L'] <= 8.605826585557853), 32.0,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__L'] > 8.605826585557853, 33.0,0)))))))))))))))

base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34_g_1_1'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34'] == 0, 0,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34'] == 1, 0,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34'] == 2, 0,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34'] == 3, 0,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34'] == 4, 0,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34'] == 5, 1,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34'] == 6, 0,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34'] == 17, 0,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34'] == 27, 0,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34'] == 28, 1,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34'] == 29, 1,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34'] == 30, 0,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34'] == 31, 1,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34'] == 32, 1,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34'] == 33, 0,-2)))))))))))))))

base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34_g_1_2'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34_g_1_1'] == 0, 1,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34_g_1_1'] == 1, 0,-2))

base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34_g_1'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34_g_1_2'] == 0, 0,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34_g_1_2'] == 1, 1,-2))




base_dados['DETALHES_CLIENTES_VALOR_FATURA__S'] = np.sin(base_dados['DETALHES_CLIENTES_VALOR_FATURA'])
np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__S'] == 0, -1, base_dados['DETALHES_CLIENTES_VALOR_FATURA__S'])
base_dados['DETALHES_CLIENTES_VALOR_FATURA__S'] = base_dados['DETALHES_CLIENTES_VALOR_FATURA__S'].fillna(-2)

base_dados['DETALHES_CLIENTES_VALOR_FATURA__S__p_3'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__S'] <= 0.4761192476593861, 1.0,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__S'] > 0.4761192476593861, 2.0,1))

base_dados['DETALHES_CLIENTES_VALOR_FATURA__S__p_3_g_1_1'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__S__p_3'] == -2.0, 0,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__S__p_3'] == 1.0, 2,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__S__p_3'] == 2.0, 1,-2)))

base_dados['DETALHES_CLIENTES_VALOR_FATURA__S__p_3_g_1_2'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__S__p_3_g_1_1'] == 0, 1,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__S__p_3_g_1_1'] == 1, 0,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__S__p_3_g_1_1'] == 2, 2,-2)))

base_dados['DETALHES_CLIENTES_VALOR_FATURA__S__p_3_g_1'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__S__p_3_g_1_2'] == 0, 0,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__S__p_3_g_1_2'] == 1, 1,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__S__p_3_g_1_2'] == 2, 2,-2)))




base_dados['IDADE_PESSOA__p_7'] = np.where(base_dados['IDADE_PESSOA'] <= 27.0, 0.0,
    np.where(np.bitwise_and(base_dados['IDADE_PESSOA'] > 27.0, base_dados['IDADE_PESSOA'] <= 32.0), 1.0,
    np.where(np.bitwise_and(base_dados['IDADE_PESSOA'] > 32.0, base_dados['IDADE_PESSOA'] <= 37.0), 2.0,
    np.where(np.bitwise_and(base_dados['IDADE_PESSOA'] > 37.0, base_dados['IDADE_PESSOA'] <= 42.0), 3.0,
    np.where(np.bitwise_and(base_dados['IDADE_PESSOA'] > 42.0, base_dados['IDADE_PESSOA'] <= 48.0), 4.0,
    np.where(np.bitwise_and(base_dados['IDADE_PESSOA'] > 48.0, base_dados['IDADE_PESSOA'] <= 58.0), 5.0,
    np.where(base_dados['IDADE_PESSOA'] > 58.0, 6.0,0)))))))

base_dados['IDADE_PESSOA__p_7_g_1_1'] = np.where(base_dados['IDADE_PESSOA__p_7'] == 0, 0,
    np.where(base_dados['IDADE_PESSOA__p_7'] == 1, 1,
    np.where(base_dados['IDADE_PESSOA__p_7'] == 2, 0,
    np.where(base_dados['IDADE_PESSOA__p_7'] == 3, 1,
    np.where(base_dados['IDADE_PESSOA__p_7'] == 4, 0,
    np.where(base_dados['IDADE_PESSOA__p_7'] == 5, 0,
    np.where(base_dados['IDADE_PESSOA__p_7'] == 6, 1,-2)))))))

base_dados['IDADE_PESSOA__p_7_g_1_2'] = np.where(base_dados['IDADE_PESSOA__p_7_g_1_1'] == 0, 1,
    np.where(base_dados['IDADE_PESSOA__p_7_g_1_1'] == 1, 0,-2))

base_dados['IDADE_PESSOA__p_7_g_1'] = np.where(base_dados['IDADE_PESSOA__p_7_g_1_2'] == 0, 0,
    np.where(base_dados['IDADE_PESSOA__p_7_g_1_2'] == 1, 1,-2))





base_dados['IDADE_PESSOA__pe_4'] = np.where(base_dados['IDADE_PESSOA'] <= 27.0, 1.0,
    np.where(np.bitwise_and(base_dados['IDADE_PESSOA'] > 27.0, base_dados['IDADE_PESSOA'] <= 41.0), 2.0,
    np.where(base_dados['IDADE_PESSOA'] > 41.0, 3.0,1)))

base_dados['IDADE_PESSOA__pe_4_g_1_1'] = np.where(base_dados['IDADE_PESSOA__pe_4'] == -2.0, 1,
    np.where(base_dados['IDADE_PESSOA__pe_4'] == 1.0, 1,
    np.where(base_dados['IDADE_PESSOA__pe_4'] == 2.0, 0,
    np.where(base_dados['IDADE_PESSOA__pe_4'] == 3.0, 0,-2))))

base_dados['IDADE_PESSOA__pe_4_g_1_2'] = np.where(base_dados['IDADE_PESSOA__pe_4_g_1_1'] == 0, 0,
    np.where(base_dados['IDADE_PESSOA__pe_4_g_1_1'] == 1, 1,-2))

base_dados['IDADE_PESSOA__pe_4_g_1'] = np.where(base_dados['IDADE_PESSOA__pe_4_g_1_2'] == 0, 0,
    np.where(base_dados['IDADE_PESSOA__pe_4_g_1_2'] == 1, 1,-2))


# COMMAND ----------

# MAGIC %md
# MAGIC ### Parte 2 de 3

# COMMAND ----------


base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_3_1'] = np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1'] == 0, base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1'] == 0), 0,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1'] == 0, base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1'] == 1), 1,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1'] == 1, base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1'] == 0), 1,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_SCORE_CARGA__L__pe_13_g_1'] == 1, base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1'] == 1), 2,0))))

base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_3_2'] = np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_3_1'] == 0, 0,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_3_1'] == 1, 1,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_3_1'] == 2, 2,0)))

base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_3'] = np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_3_2'] == 0, 0,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_3_2'] == 1, 1,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_3_2'] == 2, 2,0)))





base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_3_1'] = np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__L__p_13_g_1'] == 0, base_dados['DOCUMENTO_PESSOA__L__p_3_g_1'] == 0), 0,
    np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__L__p_13_g_1'] == 1, base_dados['DOCUMENTO_PESSOA__L__p_3_g_1'] == 0), 1,
    np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__L__p_13_g_1'] == 1, base_dados['DOCUMENTO_PESSOA__L__p_3_g_1'] == 1), 3,
    np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__L__p_13_g_1'] == 2, base_dados['DOCUMENTO_PESSOA__L__p_3_g_1'] == 0), 2,
    np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__L__p_13_g_1'] == 2, base_dados['DOCUMENTO_PESSOA__L__p_3_g_1'] == 1), 4,
    np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__L__p_13_g_1'] == 3, base_dados['DOCUMENTO_PESSOA__L__p_3_g_1'] == 0), 2,
    np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__L__p_13_g_1'] == 3, base_dados['DOCUMENTO_PESSOA__L__p_3_g_1'] == 1), 4,0)))))))

base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_3_2'] = np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_3_1'] == 0, 0,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_3_1'] == 1, 1,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_3_1'] == 2, 3,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_3_1'] == 3, 2,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_3_1'] == 4, 4,0)))))

base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_3'] = np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_3_2'] == 0, 0,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_3_2'] == 1, 1,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_3_2'] == 2, 2,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_3_2'] == 3, 3,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_3_2'] == 4, 4,0)))))





base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1_c1_26_1'] = np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__pe_15_g_1'] == 0, base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1'] == 1), 1,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__pe_15_g_1'] == 0, base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1'] == 2), 2,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__pe_15_g_1'] == 1, base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1'] == 0), 0,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__pe_15_g_1'] == 1, base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1'] == 1), 1,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__pe_15_g_1'] == 1, base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1'] == 2), 2,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__pe_15_g_1'] == 2, base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1'] == 0), 0,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__pe_15_g_1'] == 2, base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1'] == 1), 2,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__pe_15_g_1'] == 2, base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1'] == 2), 3,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__pe_15_g_1'] == 3, base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1'] == 1), 2,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__pe_15_g_1'] == 3, base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1'] == 2), 4,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__pe_15_g_1'] == 3, base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1'] == 3), 4,0)))))))))))

base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1_c1_26_2'] = np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1_c1_26_1'] == 0, 1,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1_c1_26_1'] == 1, 0,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1_c1_26_1'] == 2, 2,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1_c1_26_1'] == 3, 3,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1_c1_26_1'] == 4, 4,0)))))

base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1_c1_26'] = np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1_c1_26_2'] == 0, 0,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1_c1_26_2'] == 1, 1,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1_c1_26_2'] == 2, 2,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1_c1_26_2'] == 3, 3,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1_c1_26_2'] == 4, 4,0)))))




base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_20_1'] = np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1'] == 0, base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__S__p_3_g_1'] == 0), 0,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1'] == 0, base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__S__p_3_g_1'] == 1), 1,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1'] == 0, base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__S__p_3_g_1'] == 2), 1,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1'] == 1, base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__S__p_3_g_1'] == 0), 2,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1'] == 1, base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__S__p_3_g_1'] == 1), 3,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1'] == 1, base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__S__p_3_g_1'] == 2), 4,0))))))

base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_20_2'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_20_1'] == 0, 0,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_20_1'] == 1, 1,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_20_1'] == 2, 2,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_20_1'] == 3, 3,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_20_1'] == 4, 4,0)))))

base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_20'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_20_2'] == 0, 0,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_20_2'] == 1, 1,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_20_2'] == 2, 2,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_20_2'] == 3, 3,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_20_2'] == 4, 4,0)))))





base_dados['AGING__p_7_g_1_c1_17_1'] = np.where(np.bitwise_and(base_dados['AGING__p_7_g_1'] == 0, base_dados['AGING__pe_20_g_1'] == 0), 0,
    np.where(np.bitwise_and(base_dados['AGING__p_7_g_1'] == 0, base_dados['AGING__pe_20_g_1'] == 1), 0,
    np.where(np.bitwise_and(base_dados['AGING__p_7_g_1'] == 1, base_dados['AGING__pe_20_g_1'] == 0), 1,
    np.where(np.bitwise_and(base_dados['AGING__p_7_g_1'] == 1, base_dados['AGING__pe_20_g_1'] == 1), 2,
    np.where(np.bitwise_and(base_dados['AGING__p_7_g_1'] == 1, base_dados['AGING__pe_20_g_1'] == 2), 3,
    np.where(np.bitwise_and(base_dados['AGING__p_7_g_1'] == 1, base_dados['AGING__pe_20_g_1'] == 3), 3,
    np.where(np.bitwise_and(base_dados['AGING__p_7_g_1'] == 2, base_dados['AGING__pe_20_g_1'] == 2), 3,
    np.where(np.bitwise_and(base_dados['AGING__p_7_g_1'] == 2, base_dados['AGING__pe_20_g_1'] == 3), 3,
    np.where(np.bitwise_and(base_dados['AGING__p_7_g_1'] == 3, base_dados['AGING__pe_20_g_1'] == 3), 4,
    np.where(np.bitwise_and(base_dados['AGING__p_7_g_1'] == 3, base_dados['AGING__pe_20_g_1'] == 4), 4,
    np.where(np.bitwise_and(base_dados['AGING__p_7_g_1'] == 4, base_dados['AGING__pe_20_g_1'] == 4), 5,0)))))))))))

base_dados['AGING__p_7_g_1_c1_17_2'] = np.where(base_dados['AGING__p_7_g_1_c1_17_1'] == 0, 1,
    np.where(base_dados['AGING__p_7_g_1_c1_17_1'] == 1, 0,
    np.where(base_dados['AGING__p_7_g_1_c1_17_1'] == 2, 2,
    np.where(base_dados['AGING__p_7_g_1_c1_17_1'] == 3, 3,
    np.where(base_dados['AGING__p_7_g_1_c1_17_1'] == 4, 4,
    np.where(base_dados['AGING__p_7_g_1_c1_17_1'] == 5, 5,0))))))

base_dados['AGING__p_7_g_1_c1_17'] = np.where(base_dados['AGING__p_7_g_1_c1_17_2'] == 0, 0,
    np.where(base_dados['AGING__p_7_g_1_c1_17_2'] == 1, 1,
    np.where(base_dados['AGING__p_7_g_1_c1_17_2'] == 2, 2,
    np.where(base_dados['AGING__p_7_g_1_c1_17_2'] == 3, 3,
    np.where(base_dados['AGING__p_7_g_1_c1_17_2'] == 4, 4,
    np.where(base_dados['AGING__p_7_g_1_c1_17_2'] == 5, 5,0))))))





base_dados['DETALHES_DIVIDAS_VALOR_JUROS__pe_3_g_1_c1_3_1'] = np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__pe_3_g_1'] == 0, base_dados['DETALHES_DIVIDAS_VALOR_JUROS__L__pe_3_g_1'] == 0), 0,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__pe_3_g_1'] == 0, base_dados['DETALHES_DIVIDAS_VALOR_JUROS__L__pe_3_g_1'] == 2), 0,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__pe_3_g_1'] == 1, base_dados['DETALHES_DIVIDAS_VALOR_JUROS__L__pe_3_g_1'] == 0), 1,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__pe_3_g_1'] == 1, base_dados['DETALHES_DIVIDAS_VALOR_JUROS__L__pe_3_g_1'] == 1), 2,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__pe_3_g_1'] == 1, base_dados['DETALHES_DIVIDAS_VALOR_JUROS__L__pe_3_g_1'] == 2), 3,0)))))

base_dados['DETALHES_DIVIDAS_VALOR_JUROS__pe_3_g_1_c1_3_2'] = np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__pe_3_g_1_c1_3_1'] == 0, 0,
    np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__pe_3_g_1_c1_3_1'] == 1, 1,
    np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__pe_3_g_1_c1_3_1'] == 2, 2,
    np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__pe_3_g_1_c1_3_1'] == 3, 3,0))))

base_dados['DETALHES_DIVIDAS_VALOR_JUROS__pe_3_g_1_c1_3'] = np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__pe_3_g_1_c1_3_2'] == 0, 0,
    np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__pe_3_g_1_c1_3_2'] == 1, 1,
    np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__pe_3_g_1_c1_3_2'] == 2, 2,
    np.where(base_dados['DETALHES_DIVIDAS_VALOR_JUROS__pe_3_g_1_c1_3_2'] == 3, 3,0))))





base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_18_1'] = np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__pe_17_g_1'] == 0, base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1'] == 0), 0,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__pe_17_g_1'] == 0, base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1'] == 1), 1,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__pe_17_g_1'] == 1, base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1'] == 0), 1,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__pe_17_g_1'] == 1, base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1'] == 1), 2,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__pe_17_g_1'] == 2, base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1'] == 0), 1,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__pe_17_g_1'] == 2, base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1'] == 1), 3,0))))))

base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_18_2'] = np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_18_1'] == 0, 0,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_18_1'] == 1, 1,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_18_1'] == 2, 2,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_18_1'] == 3, 3,0))))

base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_18'] = np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_18_2'] == 0, 0,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_18_2'] == 1, 1,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_18_2'] == 2, 2,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_18_2'] == 3, 3,0))))





base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34_g_1_c1_7_1'] = np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34_g_1'] == 0, base_dados['DETALHES_CLIENTES_VALOR_FATURA__S__p_3_g_1'] == 0), 0,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34_g_1'] == 0, base_dados['DETALHES_CLIENTES_VALOR_FATURA__S__p_3_g_1'] == 1), 0,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34_g_1'] == 0, base_dados['DETALHES_CLIENTES_VALOR_FATURA__S__p_3_g_1'] == 2), 0,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34_g_1'] == 1, base_dados['DETALHES_CLIENTES_VALOR_FATURA__S__p_3_g_1'] == 0), 1,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34_g_1'] == 1, base_dados['DETALHES_CLIENTES_VALOR_FATURA__S__p_3_g_1'] == 1), 2,
    np.where(np.bitwise_and(base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34_g_1'] == 1, base_dados['DETALHES_CLIENTES_VALOR_FATURA__S__p_3_g_1'] == 2), 3,0))))))

base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34_g_1_c1_7_2'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34_g_1_c1_7_1'] == 0, 0,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34_g_1_c1_7_1'] == 1, 1,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34_g_1_c1_7_1'] == 2, 2,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34_g_1_c1_7_1'] == 3, 3,0))))

base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34_g_1_c1_7'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34_g_1_c1_7_2'] == 0, 0,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34_g_1_c1_7_2'] == 1, 1,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34_g_1_c1_7_2'] == 2, 2,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34_g_1_c1_7_2'] == 3, 3,0))))





base_dados['IDADE_PESSOA__p_7_g_1_c1_3_1'] = np.where(np.bitwise_and(base_dados['IDADE_PESSOA__p_7_g_1'] == 0, base_dados['IDADE_PESSOA__pe_4_g_1'] == 0), 0,
    np.where(np.bitwise_and(base_dados['IDADE_PESSOA__p_7_g_1'] == 0, base_dados['IDADE_PESSOA__pe_4_g_1'] == 1), 0,
    np.where(np.bitwise_and(base_dados['IDADE_PESSOA__p_7_g_1'] == 1, base_dados['IDADE_PESSOA__pe_4_g_1'] == 0), 1,
    np.where(np.bitwise_and(base_dados['IDADE_PESSOA__p_7_g_1'] == 1, base_dados['IDADE_PESSOA__pe_4_g_1'] == 1), 2,0))))

base_dados['IDADE_PESSOA__p_7_g_1_c1_3_2'] = np.where(base_dados['IDADE_PESSOA__p_7_g_1_c1_3_1'] == 0, 0,
    np.where(base_dados['IDADE_PESSOA__p_7_g_1_c1_3_1'] == 1, 1,
    np.where(base_dados['IDADE_PESSOA__p_7_g_1_c1_3_1'] == 2, 2,0)))

base_dados['IDADE_PESSOA__p_7_g_1_c1_3'] = np.where(base_dados['IDADE_PESSOA__p_7_g_1_c1_3_2'] == 0, 0,
    np.where(base_dados['IDADE_PESSOA__p_7_g_1_c1_3_2'] == 1, 1,
    np.where(base_dados['IDADE_PESSOA__p_7_g_1_c1_3_2'] == 2, 2,0)))


# COMMAND ----------

# MAGIC %md
# MAGIC ### Parte 2 de 3

# COMMAND ----------

base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_1'] = np.where(np.bitwise_and(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh38'] == 0, base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34_g_1_c1_7'] == 0), 0,
    np.where(np.bitwise_and(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh38'] == 0, base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34_g_1_c1_7'] == 1), 1,
    np.where(np.bitwise_and(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh38'] == 0, base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34_g_1_c1_7'] == 2), 2,
    np.where(np.bitwise_and(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh38'] == 0, base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34_g_1_c1_7'] == 3), 3,
    np.where(np.bitwise_and(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh38'] == 1, base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34_g_1_c1_7'] == 0), 4,
    np.where(np.bitwise_and(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh38'] == 1, base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34_g_1_c1_7'] == 1), 5,
    np.where(np.bitwise_and(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh38'] == 1, base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34_g_1_c1_7'] == 2), 6,
    np.where(np.bitwise_and(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh38'] == 1, base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34_g_1_c1_7'] == 3), 7,
    np.where(np.bitwise_and(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh38'] == 2, base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34_g_1_c1_7'] == 0), 8,
    np.where(np.bitwise_and(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh38'] == 2, base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34_g_1_c1_7'] == 1), 9,
    np.where(np.bitwise_and(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh38'] == 2, base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34_g_1_c1_7'] == 2), 10,
    np.where(np.bitwise_and(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh38'] == 2, base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34_g_1_c1_7'] == 3), 11,
    np.where(np.bitwise_and(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh38'] == 3, base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34_g_1_c1_7'] == 0), 12,
    np.where(np.bitwise_and(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh38'] == 3, base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34_g_1_c1_7'] == 1), 13,
    np.where(np.bitwise_and(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh38'] == 3, base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34_g_1_c1_7'] == 2), 14,
    np.where(np.bitwise_and(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh38'] == 3, base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34_g_1_c1_7'] == 3), 15,
    np.where(np.bitwise_and(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh38'] == 4, base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34_g_1_c1_7'] == 0), 16,
    np.where(np.bitwise_and(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh38'] == 4, base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34_g_1_c1_7'] == 1), 17,
    np.where(np.bitwise_and(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh38'] == 4, base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34_g_1_c1_7'] == 2), 18,
    np.where(np.bitwise_and(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh38'] == 4, base_dados['DETALHES_CLIENTES_VALOR_FATURA__L__p_34_g_1_c1_7'] == 3), 19,
     0))))))))))))))))))))

base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_2'] = np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_1'] == 0, 0,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_1'] == 1, 0,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_1'] == 2, 0,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_1'] == 3, 5,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_1'] == 4, 8,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_1'] == 5, 5,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_1'] == 6, 5,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_1'] == 7, 0,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_1'] == 8, 0,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_1'] == 9, 9,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_1'] == 10, 13,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_1'] == 11, 14,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_1'] == 12, 9,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_1'] == 13, 9,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_1'] == 14, 14,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_1'] == 15, 17,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_1'] == 16, 9,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_1'] == 17, 14,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_1'] == 18, 17,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_1'] == 19, 19,
     0))))))))))))))))))))

base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_3'] = np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_2'] == 0, 0,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_2'] == 5, 1,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_2'] == 8, 2,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_2'] == 9, 3,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_2'] == 13, 4,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_2'] == 14, 5,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_2'] == 17, 6,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_2'] == 19, 7,
     0))))))))

base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_4'] = np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_3'] == 0, 0,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_3'] == 1, 1,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_3'] == 2, 2,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_3'] == 3, 2,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_3'] == 4, 4,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_3'] == 5, 4,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_3'] == 6, 6,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_3'] == 7, 6,
     0))))))))

base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_5'] = np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_4'] == 0, 0,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_4'] == 1, 1,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_4'] == 2, 2,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_4'] == 4, 3,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_4'] == 6, 4,
     0)))))

base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_6'] = np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_5'] == 0, 0,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_5'] == 1, 1,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_5'] == 2, 2,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_5'] == 3, 3,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_5'] == 4, 4,
     0)))))

base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_7'] = np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_6'] == 0, 0,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_6'] == 1, 1,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_6'] == 2, 2,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_6'] == 3, 3,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_6'] == 4, 4,
     0)))))

base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_8'] = np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_7'] == 0, 0,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_7'] == 1, 1,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_7'] == 2, 2,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_7'] == 3, 3,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_7'] == 4, 4,
     0)))))

base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_9'] = np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_8'] == 0, 0,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_8'] == 1, 1,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_8'] == 2, 2,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_8'] == 3, 3,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_8'] == 4, 4,
     0)))))

base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_10'] = np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_9'] == 0, 0,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_9'] == 1, 1,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_9'] == 2, 2,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_9'] == 3, 3,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_9'] == 4, 4,
     0)))))

base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_11'] = np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_10'] == 0, 0,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_10'] == 1, 1,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_10'] == 2, 2,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_10'] == 3, 3,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_10'] == 4, 4,
     0)))))

base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_12'] = np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_11'] == 0, 0,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_11'] == 1, 1,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_11'] == 2, 2,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_11'] == 3, 3,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_11'] == 4, 4,
     0)))))

base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_13'] = np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_12'] == 0, 0,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_12'] == 1, 1,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_12'] == 2, 2,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_12'] == 3, 3,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_12'] == 4, 4,
     0)))))

base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_14'] = np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_13'] == 0, 0,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_13'] == 1, 1,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_13'] == 2, 2,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_13'] == 3, 3,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_13'] == 4, 4,
     0)))))

base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29'] = np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_14'] == 0, 0,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_14'] == 1, 1,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_14'] == 2, 2,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_14'] == 3, 3,
    np.where(base_dados['ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29_14'] == 4, 4,
     0)))))






base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_1'] = np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh38'] == 0, base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_3'] == 0), 0,
    np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh38'] == 0, base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_3'] == 1), 3,
    np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh38'] == 0, base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_3'] == 2), 6,
    np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh38'] == 1, base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_3'] == 0), 1,
    np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh38'] == 1, base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_3'] == 1), 4,
    np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh38'] == 1, base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_3'] == 2), 7,
    np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh38'] == 2, base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_3'] == 0), 2,
    np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh38'] == 2, base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_3'] == 1), 5,
    np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_CODIGO_LOGO_gh38'] == 2, base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_3'] == 2), 8,
     0)))))))))

base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_2'] = np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_1'] == 0, 1,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_1'] == 1, 3,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_1'] == 2, 6,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_1'] == 3, 2,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_1'] == 4, 4,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_1'] == 5, 6,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_1'] == 6, 0,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_1'] == 7, 5,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_1'] == 8, 6,
     0)))))))))

base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_3'] = np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_2'] == 0, 0,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_2'] == 1, 1,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_2'] == 2, 2,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_2'] == 3, 3,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_2'] == 4, 4,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_2'] == 5, 5,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_2'] == 6, 6,
     0)))))))

base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_4'] = np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_3'] == 0, 0,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_3'] == 1, 1,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_3'] == 2, 2,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_3'] == 3, 3,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_3'] == 4, 4,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_3'] == 5, 5,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_3'] == 6, 6,
     0)))))))

base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_5'] = np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_4'] == 0, 0,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_4'] == 1, 1,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_4'] == 2, 2,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_4'] == 3, 3,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_4'] == 4, 4,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_4'] == 5, 5,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_4'] == 6, 6,
     0)))))))

base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_6'] = np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_5'] == 0, 0,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_5'] == 1, 0,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_5'] == 2, 2,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_5'] == 3, 2,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_5'] == 4, 4,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_5'] == 5, 5,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_5'] == 6, 6,
     0)))))))

base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_7'] = np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_6'] == 0, 0,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_6'] == 2, 1,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_6'] == 4, 2,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_6'] == 5, 3,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_6'] == 6, 4,
     0)))))

base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_8'] = np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_7'] == 0, 1,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_7'] == 1, 1,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_7'] == 2, 2,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_7'] == 3, 3,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_7'] == 4, 4,
     0)))))

base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_9'] = np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_8'] == 1, 0,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_8'] == 2, 1,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_8'] == 3, 2,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_8'] == 4, 3,
     0))))

base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_10'] = np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_9'] == 0, 0,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_9'] == 1, 1,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_9'] == 2, 2,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_9'] == 3, 3,
     0))))

base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_11'] = np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_10'] == 0, 0,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_10'] == 1, 1,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_10'] == 2, 2,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_10'] == 3, 3,
     0))))

base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_12'] = np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_11'] == 0, 0,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_11'] == 1, 1,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_11'] == 2, 2,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_11'] == 3, 3,
     0))))

base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_13'] = np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_12'] == 0, 0,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_12'] == 1, 1,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_12'] == 2, 2,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_12'] == 3, 3,
     0))))

base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_14'] = np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_13'] == 0, 0,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_13'] == 1, 1,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_13'] == 2, 2,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_13'] == 3, 3,
     0))))

base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29'] = np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_14'] == 0, 0,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_14'] == 1, 1,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_14'] == 2, 2,
    np.where(base_dados['DETALHES_CLIENTES_SCORE_CARGA__C__p_13_g_1_c1_38_gh29_14'] == 3, 3,
     0))))





base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_1'] = np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_CLASSE_gh38'] == 0, base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_20'] == 0), 0,
    np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_CLASSE_gh38'] == 0, base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_20'] == 1), 2,
    np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_CLASSE_gh38'] == 0, base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_20'] == 2), 4,
    np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_CLASSE_gh38'] == 0, base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_20'] == 3), 6,
    np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_CLASSE_gh38'] == 0, base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_20'] == 4), 8,
    np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_CLASSE_gh38'] == 1, base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_20'] == 0), 1,
    np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_CLASSE_gh38'] == 1, base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_20'] == 1), 3,
    np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_CLASSE_gh38'] == 1, base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_20'] == 2), 5,
    np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_CLASSE_gh38'] == 1, base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_20'] == 3), 7,
    np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_CLASSE_gh38'] == 1, base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_20'] == 4), 9,
     0))))))))))

base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_2'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_1'] == 0, 3,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_1'] == 1, 5,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_1'] == 2, 4,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_1'] == 3, 6,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_1'] == 4, 0,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_1'] == 5, 7,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_1'] == 6, 2,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_1'] == 7, 8,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_1'] == 8, 0,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_1'] == 9, 9,
     0))))))))))

base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_3'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_2'] == 0, 0,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_2'] == 2, 1,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_2'] == 3, 2,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_2'] == 4, 3,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_2'] == 5, 4,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_2'] == 6, 5,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_2'] == 7, 6,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_2'] == 8, 7,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_2'] == 9, 8,
     0)))))))))

base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_4'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_3'] == 0, 0,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_3'] == 1, 1,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_3'] == 2, 2,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_3'] == 3, 3,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_3'] == 4, 4,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_3'] == 5, 5,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_3'] == 6, 5,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_3'] == 7, 7,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_3'] == 8, 7,
     0)))))))))

base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_5'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_4'] == 0, 0,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_4'] == 1, 1,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_4'] == 2, 2,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_4'] == 3, 3,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_4'] == 4, 4,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_4'] == 5, 5,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_4'] == 7, 6,
     0)))))))

base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_6'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_5'] == 0, 0,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_5'] == 1, 1,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_5'] == 2, 3,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_5'] == 3, 3,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_5'] == 4, 4,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_5'] == 5, 5,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_5'] == 6, 6,
     0)))))))

base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_7'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_6'] == 0, 0,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_6'] == 1, 1,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_6'] == 3, 2,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_6'] == 4, 3,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_6'] == 5, 4,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_6'] == 6, 5,
     0))))))

base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_8'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_7'] == 0, 1,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_7'] == 1, 1,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_7'] == 2, 1,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_7'] == 3, 1,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_7'] == 4, 4,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_7'] == 5, 5,
     0))))))

base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_9'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_8'] == 1, 0,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_8'] == 4, 1,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_8'] == 5, 2,
     0)))

base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_10'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_9'] == 0, 0,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_9'] == 1, 1,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_9'] == 2, 2,
     0)))

base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_11'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_10'] == 0, 0,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_10'] == 1, 1,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_10'] == 2, 2,
     0)))

base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_12'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_11'] == 0, 0,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_11'] == 1, 1,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_11'] == 2, 2,
     0)))

base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_13'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_12'] == 0, 0,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_12'] == 1, 1,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_12'] == 2, 2,
     0)))

base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_14'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_13'] == 0, 0,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_13'] == 1, 1,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_13'] == 2, 2,
     0)))

base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29'] = np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_14'] == 0, 0,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_14'] == 1, 1,
    np.where(base_dados['DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO__L__p_25_g_1_c1_209_gh29_14'] == 2, 2,
     0)))





base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_1'] = np.where(np.bitwise_and(base_dados['TIPO_EMAIL_0_gh38'] == 0, base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh38'] == 0), 0,
    np.where(np.bitwise_and(base_dados['TIPO_EMAIL_0_gh38'] == 0, base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh38'] == 1), 2,
    np.where(np.bitwise_and(base_dados['TIPO_EMAIL_0_gh38'] == 0, base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh38'] == 2), 4,
    np.where(np.bitwise_and(base_dados['TIPO_EMAIL_0_gh38'] == 1, base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh38'] == 0), 1,
    np.where(np.bitwise_and(base_dados['TIPO_EMAIL_0_gh38'] == 1, base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh38'] == 1), 3,
    np.where(np.bitwise_and(base_dados['TIPO_EMAIL_0_gh38'] == 1, base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh38'] == 2), 5,
     0))))))

base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_2'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_1'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_1'] == 1, 2,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_1'] == 2, 1,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_1'] == 3, 3,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_1'] == 4, 3,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_1'] == 5, 3,
     0))))))

base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_3'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_2'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_2'] == 1, 1,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_2'] == 2, 2,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_2'] == 3, 3,
     0))))

base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_4'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_3'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_3'] == 1, 1,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_3'] == 2, 2,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_3'] == 3, 3,
     0))))

base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_5'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_4'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_4'] == 1, 1,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_4'] == 2, 2,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_4'] == 3, 3,
     0))))

base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_6'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_5'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_5'] == 1, 1,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_5'] == 2, 2,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_5'] == 3, 3,
     0))))

base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_7'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_6'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_6'] == 1, 1,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_6'] == 2, 2,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_6'] == 3, 3,
     0))))

base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_8'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_7'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_7'] == 1, 1,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_7'] == 2, 3,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_7'] == 3, 3,
     0))))

base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_9'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_8'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_8'] == 1, 1,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_8'] == 3, 2,
     0)))

base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_10'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_9'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_9'] == 1, 1,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_9'] == 2, 2,
     0)))

base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_11'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_10'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_10'] == 1, 1,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_10'] == 2, 2,
     0)))

base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_12'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_11'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_11'] == 1, 1,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_11'] == 2, 2,
     0)))

base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_13'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_12'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_12'] == 1, 1,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_12'] == 2, 2,
     0)))

base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_14'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_13'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_13'] == 1, 1,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_13'] == 2, 2,
     0)))

base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29'] = np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_14'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_14'] == 1, 1,
    np.where(base_dados['DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29_14'] == 2, 2,
     0)))







base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_1'] = np.where(np.bitwise_and(base_dados['TIPO_TELEFONE_4_gh38'] == 0, base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh38'] == 0), 0,
    np.where(np.bitwise_and(base_dados['TIPO_TELEFONE_4_gh38'] == 0, base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh38'] == 1), 2,
    np.where(np.bitwise_and(base_dados['TIPO_TELEFONE_4_gh38'] == 1, base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh38'] == 0), 1,
    np.where(np.bitwise_and(base_dados['TIPO_TELEFONE_4_gh38'] == 1, base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh38'] == 1), 3,
     0))))

base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_2'] = np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_1'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_1'] == 1, 2,
    np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_1'] == 2, 1,
    np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_1'] == 3, 3,
     0))))

base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_3'] = np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_2'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_2'] == 1, 1,
    np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_2'] == 2, 2,
    np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_2'] == 3, 3,
     0))))

base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_4'] = np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_3'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_3'] == 1, 1,
    np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_3'] == 2, 2,
    np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_3'] == 3, 3,
     0))))

base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_5'] = np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_4'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_4'] == 1, 1,
    np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_4'] == 2, 2,
    np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_4'] == 3, 3,
     0))))

base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_6'] = np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_5'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_5'] == 1, 1,
    np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_5'] == 2, 2,
    np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_5'] == 3, 3,
     0))))

base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_7'] = np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_6'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_6'] == 1, 1,
    np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_6'] == 2, 2,
    np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_6'] == 3, 3,
     0))))

base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_8'] = np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_7'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_7'] == 1, 2,
    np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_7'] == 2, 2,
    np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_7'] == 3, 3,
     0))))

base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_9'] = np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_8'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_8'] == 2, 1,
    np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_8'] == 3, 2,
     0)))

base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_10'] = np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_9'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_9'] == 1, 1,
    np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_9'] == 2, 2,
     0)))

base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_11'] = np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_10'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_10'] == 1, 1,
    np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_10'] == 2, 2,
     0)))

base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_12'] = np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_11'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_11'] == 1, 1,
    np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_11'] == 2, 2,
     0)))

base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_13'] = np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_12'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_12'] == 1, 1,
    np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_12'] == 2, 2,
     0)))

base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_14'] = np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_13'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_13'] == 1, 1,
    np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_13'] == 2, 2,
     0)))

base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29'] = np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_14'] == 0, 0,
    np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_14'] == 1, 1,
    np.where(base_dados['DETALHES_CONTRATOS_STATUS_ACORDO_gh3814_gh29_14'] == 2, 2,
     0)))




base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_1'] = np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1_c1_26'] == 0, base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_18'] == 0), 0,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1_c1_26'] == 0, base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_18'] == 1), 5,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1_c1_26'] == 0, base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_18'] == 2), 10,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1_c1_26'] == 0, base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_18'] == 3), 15,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1_c1_26'] == 1, base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_18'] == 0), 1,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1_c1_26'] == 1, base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_18'] == 1), 6,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1_c1_26'] == 1, base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_18'] == 2), 11,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1_c1_26'] == 1, base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_18'] == 3), 16,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1_c1_26'] == 2, base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_18'] == 0), 2,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1_c1_26'] == 2, base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_18'] == 1), 7,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1_c1_26'] == 2, base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_18'] == 2), 12,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1_c1_26'] == 2, base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_18'] == 3), 17,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1_c1_26'] == 3, base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_18'] == 0), 3,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1_c1_26'] == 3, base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_18'] == 1), 8,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1_c1_26'] == 3, base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_18'] == 2), 13,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1_c1_26'] == 3, base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_18'] == 3), 18,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1_c1_26'] == 4, base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_18'] == 0), 4,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1_c1_26'] == 4, base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_18'] == 1), 9,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1_c1_26'] == 4, base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_18'] == 2), 14,
    np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_TAXA_SERVICO__L__p_15_g_1_c1_26'] == 4, base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_18'] == 3), 19,
     0))))))))))))))))))))

base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_2'] = np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_1'] == 0, 1,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_1'] == 1, 0,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_1'] == 2, 8,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_1'] == 3, 8,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_1'] == 4, 8,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_1'] == 5, 1,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_1'] == 6, 5,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_1'] == 7, 6,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_1'] == 8, 12,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_1'] == 9, 17,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_1'] == 10, 1,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_1'] == 11, 12,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_1'] == 12, 12,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_1'] == 13, 15,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_1'] == 14, 17,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_1'] == 15, 6,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_1'] == 16, 1,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_1'] == 17, 8,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_1'] == 18, 15,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_1'] == 19, 19,
     0))))))))))))))))))))

base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_3'] = np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_2'] == 0, 0,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_2'] == 1, 1,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_2'] == 5, 2,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_2'] == 6, 3,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_2'] == 8, 4,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_2'] == 12, 5,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_2'] == 15, 6,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_2'] == 17, 7,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_2'] == 19, 8,
     0)))))))))

base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_4'] = np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_3'] == 0, 0,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_3'] == 1, 1,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_3'] == 2, 2,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_3'] == 3, 3,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_3'] == 4, 4,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_3'] == 5, 5,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_3'] == 6, 6,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_3'] == 7, 6,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_3'] == 8, 8,
     0)))))))))

base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_5'] = np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_4'] == 0, 0,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_4'] == 1, 1,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_4'] == 2, 2,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_4'] == 3, 3,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_4'] == 4, 4,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_4'] == 5, 5,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_4'] == 6, 6,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_4'] == 8, 7,
     0))))))))

base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_6'] = np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_5'] == 0, 1,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_5'] == 1, 1,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_5'] == 2, 2,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_5'] == 3, 3,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_5'] == 4, 4,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_5'] == 5, 5,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_5'] == 6, 6,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_5'] == 7, 6,
     0))))))))

base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_7'] = np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_6'] == 1, 0,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_6'] == 2, 1,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_6'] == 3, 2,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_6'] == 4, 3,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_6'] == 5, 4,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_6'] == 6, 5,
     0))))))

base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_8'] = np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_7'] == 0, 0,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_7'] == 1, 0,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_7'] == 2, 0,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_7'] == 3, 4,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_7'] == 4, 4,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_7'] == 5, 5,
     0))))))

base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_9'] = np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_8'] == 0, 0,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_8'] == 4, 1,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_8'] == 5, 2,
     0)))

base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_10'] = np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_9'] == 0, 0,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_9'] == 1, 1,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_9'] == 2, 2,
     0)))

base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_11'] = np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_10'] == 0, 0,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_10'] == 1, 1,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_10'] == 2, 2,
     0)))

base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_12'] = np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_11'] == 0, 0,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_11'] == 1, 1,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_11'] == 2, 2,
     0)))

base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_13'] = np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_12'] == 0, 0,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_12'] == 1, 1,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_12'] == 2, 2,
     0)))

base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_14'] = np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_13'] == 0, 0,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_13'] == 1, 1,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_13'] == 2, 2,
     0)))

base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29'] = np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_14'] == 0, 0,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_14'] == 1, 1,
    np.where(base_dados['DETALHES_DIVIDAS_TAXA_ATRASO__T__p_3_g_1_c1_184_gh29_14'] == 2, 2,
     0)))





base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_1'] = np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_3'] == 0, base_dados['IDADE_PESSOA__p_7_g_1_c1_3'] == 0), 0,
    np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_3'] == 0, base_dados['IDADE_PESSOA__p_7_g_1_c1_3'] == 1), 1,
    np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_3'] == 0, base_dados['IDADE_PESSOA__p_7_g_1_c1_3'] == 2), 2,
    np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_3'] == 1, base_dados['IDADE_PESSOA__p_7_g_1_c1_3'] == 0), 3,
    np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_3'] == 1, base_dados['IDADE_PESSOA__p_7_g_1_c1_3'] == 1), 4,
    np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_3'] == 1, base_dados['IDADE_PESSOA__p_7_g_1_c1_3'] == 2), 5,
    np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_3'] == 2, base_dados['IDADE_PESSOA__p_7_g_1_c1_3'] == 0), 6,
    np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_3'] == 2, base_dados['IDADE_PESSOA__p_7_g_1_c1_3'] == 1), 7,
    np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_3'] == 2, base_dados['IDADE_PESSOA__p_7_g_1_c1_3'] == 2), 8,
    np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_3'] == 3, base_dados['IDADE_PESSOA__p_7_g_1_c1_3'] == 0), 9,
    np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_3'] == 3, base_dados['IDADE_PESSOA__p_7_g_1_c1_3'] == 1), 10,
    np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_3'] == 3, base_dados['IDADE_PESSOA__p_7_g_1_c1_3'] == 2), 11,
    np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_3'] == 4, base_dados['IDADE_PESSOA__p_7_g_1_c1_3'] == 0), 12,
    np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_3'] == 4, base_dados['IDADE_PESSOA__p_7_g_1_c1_3'] == 1), 13,
    np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_3'] == 4, base_dados['IDADE_PESSOA__p_7_g_1_c1_3'] == 2), 14,
     0)))))))))))))))

base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_2'] = np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_1'] == 0, 0,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_1'] == 1, 1,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_1'] == 2, 4,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_1'] == 3, 1,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_1'] == 4, 1,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_1'] == 5, 9,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_1'] == 6, 4,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_1'] == 7, 4,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_1'] == 8, 9,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_1'] == 9, 4,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_1'] == 10, 8,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_1'] == 11, 14,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_1'] == 12, 9,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_1'] == 13, 12,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_1'] == 14, 12,
     0)))))))))))))))

base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_3'] = np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_2'] == 0, 0,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_2'] == 1, 1,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_2'] == 4, 2,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_2'] == 8, 3,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_2'] == 9, 4,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_2'] == 12, 5,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_2'] == 14, 6,
     0)))))))

base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_4'] = np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_3'] == 0, 0,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_3'] == 1, 1,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_3'] == 2, 2,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_3'] == 3, 3,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_3'] == 4, 3,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_3'] == 5, 5,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_3'] == 6, 5,
     0)))))))

base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_5'] = np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_4'] == 0, 0,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_4'] == 1, 1,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_4'] == 2, 2,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_4'] == 3, 3,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_4'] == 5, 4,
     0)))))

base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_6'] = np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_5'] == 0, 0,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_5'] == 1, 1,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_5'] == 2, 2,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_5'] == 3, 3,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_5'] == 4, 4,
     0)))))

base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_7'] = np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_6'] == 0, 0,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_6'] == 1, 1,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_6'] == 2, 2,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_6'] == 3, 3,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_6'] == 4, 4,
     0)))))

base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_8'] = np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_7'] == 0, 1,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_7'] == 1, 1,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_7'] == 2, 2,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_7'] == 3, 3,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_7'] == 4, 4,
     0)))))

base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_9'] = np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_8'] == 1, 0,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_8'] == 2, 1,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_8'] == 3, 2,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_8'] == 4, 3,
     0))))

base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_10'] = np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_9'] == 0, 0,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_9'] == 1, 1,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_9'] == 2, 2,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_9'] == 3, 3,
     0))))

base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_11'] = np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_10'] == 0, 0,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_10'] == 1, 1,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_10'] == 2, 2,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_10'] == 3, 3,
     0))))

base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_12'] = np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_11'] == 0, 0,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_11'] == 1, 1,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_11'] == 2, 2,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_11'] == 3, 3,
     0))))

base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_13'] = np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_12'] == 0, 0,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_12'] == 1, 1,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_12'] == 2, 2,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_12'] == 3, 3,
     0))))

base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_14'] = np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_13'] == 0, 0,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_13'] == 1, 1,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_13'] == 2, 2,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_13'] == 3, 3,
     0))))

base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29'] = np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_14'] == 0, 0,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_14'] == 1, 1,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_14'] == 2, 2,
    np.where(base_dados['DOCUMENTO_PESSOA__L__p_3_g_1_c1_36_gh29_14'] == 3, 3,
     0))))





base_dados['TIPO_TELEFONE_0_gh3814_gh29_1'] = np.where(np.bitwise_and(base_dados['TIPO_TELEFONE_0_gh38'] == 0, base_dados['TIPO_TELEFONE_1_gh38'] == 0), 0,
    np.where(np.bitwise_and(base_dados['TIPO_TELEFONE_0_gh38'] == 0, base_dados['TIPO_TELEFONE_1_gh38'] == 1), 1,
    np.where(np.bitwise_and(base_dados['TIPO_TELEFONE_0_gh38'] == 1, base_dados['TIPO_TELEFONE_1_gh38'] == 0), 2,
    np.where(np.bitwise_and(base_dados['TIPO_TELEFONE_0_gh38'] == 1, base_dados['TIPO_TELEFONE_1_gh38'] == 1), 3,
     0))))

base_dados['TIPO_TELEFONE_0_gh3814_gh29_2'] = np.where(base_dados['TIPO_TELEFONE_0_gh3814_gh29_1'] == 0, 0,
    np.where(base_dados['TIPO_TELEFONE_0_gh3814_gh29_1'] == 1, 1,
    np.where(base_dados['TIPO_TELEFONE_0_gh3814_gh29_1'] == 2, 1,
    np.where(base_dados['TIPO_TELEFONE_0_gh3814_gh29_1'] == 3, 3,
     0))))

base_dados['TIPO_TELEFONE_0_gh3814_gh29_3'] = np.where(base_dados['TIPO_TELEFONE_0_gh3814_gh29_2'] == 0, 0,
    np.where(base_dados['TIPO_TELEFONE_0_gh3814_gh29_2'] == 1, 1,
    np.where(base_dados['TIPO_TELEFONE_0_gh3814_gh29_2'] == 3, 2,
     0)))

base_dados['TIPO_TELEFONE_0_gh3814_gh29_4'] = np.where(base_dados['TIPO_TELEFONE_0_gh3814_gh29_3'] == 0, 0,
    np.where(base_dados['TIPO_TELEFONE_0_gh3814_gh29_3'] == 1, 1,
    np.where(base_dados['TIPO_TELEFONE_0_gh3814_gh29_3'] == 2, 2,
     0)))

base_dados['TIPO_TELEFONE_0_gh3814_gh29_5'] = np.where(base_dados['TIPO_TELEFONE_0_gh3814_gh29_4'] == 0, 0,
    np.where(base_dados['TIPO_TELEFONE_0_gh3814_gh29_4'] == 1, 1,
    np.where(base_dados['TIPO_TELEFONE_0_gh3814_gh29_4'] == 2, 2,
     0)))

base_dados['TIPO_TELEFONE_0_gh3814_gh29_6'] = np.where(base_dados['TIPO_TELEFONE_0_gh3814_gh29_5'] == 0, 0,
    np.where(base_dados['TIPO_TELEFONE_0_gh3814_gh29_5'] == 1, 1,
    np.where(base_dados['TIPO_TELEFONE_0_gh3814_gh29_5'] == 2, 2,
     0)))

base_dados['TIPO_TELEFONE_0_gh3814_gh29_7'] = np.where(base_dados['TIPO_TELEFONE_0_gh3814_gh29_6'] == 0, 0,
    np.where(base_dados['TIPO_TELEFONE_0_gh3814_gh29_6'] == 1, 1,
    np.where(base_dados['TIPO_TELEFONE_0_gh3814_gh29_6'] == 2, 2,
     0)))

base_dados['TIPO_TELEFONE_0_gh3814_gh29_8'] = np.where(base_dados['TIPO_TELEFONE_0_gh3814_gh29_7'] == 0, 0,
    np.where(base_dados['TIPO_TELEFONE_0_gh3814_gh29_7'] == 1, 1,
    np.where(base_dados['TIPO_TELEFONE_0_gh3814_gh29_7'] == 2, 2,
     0)))

base_dados['TIPO_TELEFONE_0_gh3814_gh29_9'] = np.where(base_dados['TIPO_TELEFONE_0_gh3814_gh29_8'] == 0, 0,
    np.where(base_dados['TIPO_TELEFONE_0_gh3814_gh29_8'] == 1, 1,
    np.where(base_dados['TIPO_TELEFONE_0_gh3814_gh29_8'] == 2, 2,
     0)))

base_dados['TIPO_TELEFONE_0_gh3814_gh29_10'] = np.where(base_dados['TIPO_TELEFONE_0_gh3814_gh29_9'] == 0, 0,
    np.where(base_dados['TIPO_TELEFONE_0_gh3814_gh29_9'] == 1, 1,
    np.where(base_dados['TIPO_TELEFONE_0_gh3814_gh29_9'] == 2, 2,
     0)))

base_dados['TIPO_TELEFONE_0_gh3814_gh29_11'] = np.where(base_dados['TIPO_TELEFONE_0_gh3814_gh29_10'] == 0, 0,
    np.where(base_dados['TIPO_TELEFONE_0_gh3814_gh29_10'] == 1, 1,
    np.where(base_dados['TIPO_TELEFONE_0_gh3814_gh29_10'] == 2, 2,
     0)))

base_dados['TIPO_TELEFONE_0_gh3814_gh29_12'] = np.where(base_dados['TIPO_TELEFONE_0_gh3814_gh29_11'] == 0, 0,
    np.where(base_dados['TIPO_TELEFONE_0_gh3814_gh29_11'] == 1, 1,
    np.where(base_dados['TIPO_TELEFONE_0_gh3814_gh29_11'] == 2, 2,
     0)))

base_dados['TIPO_TELEFONE_0_gh3814_gh29_13'] = np.where(base_dados['TIPO_TELEFONE_0_gh3814_gh29_12'] == 0, 0,
    np.where(base_dados['TIPO_TELEFONE_0_gh3814_gh29_12'] == 1, 1,
    np.where(base_dados['TIPO_TELEFONE_0_gh3814_gh29_12'] == 2, 2,
     0)))

base_dados['TIPO_TELEFONE_0_gh3814_gh29_14'] = np.where(base_dados['TIPO_TELEFONE_0_gh3814_gh29_13'] == 0, 0,
    np.where(base_dados['TIPO_TELEFONE_0_gh3814_gh29_13'] == 1, 1,
    np.where(base_dados['TIPO_TELEFONE_0_gh3814_gh29_13'] == 2, 2,
     0)))

base_dados['TIPO_TELEFONE_0_gh3814_gh29'] = np.where(base_dados['TIPO_TELEFONE_0_gh3814_gh29_14'] == 0, 0,
    np.where(base_dados['TIPO_TELEFONE_0_gh3814_gh29_14'] == 1, 1,
    np.where(base_dados['TIPO_TELEFONE_0_gh3814_gh29_14'] == 2, 2,
     0)))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Mantendo apenas as variáveis do modelo

# COMMAND ----------

varvar=[]
varvar= [chave,'TIPO_TELEFONE_3_gh38','DETALHES_CONTRATOS_HISTORICO_FPD_gh38','AGING__p_7_g_1_c1_17','DETALHES_DIVIDAS_VALOR_JUROS__pe_3_g_1_c1_3','DETALHES_CONTRATOS_BLOQUEIO1_DESC_gh3810_gh29','ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA_gh382_gh29','DETALHES_CONTRATOS_BLOQUEIO2_gh38','TIPO_TELEFONE_0_gh3814_gh29']
base_teste_c0 = base_dados[varvar]
base_teste_c0


# COMMAND ----------

var_fin_c0=list(base_teste_c0.columns)
var_fin_c0.remove(chave)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rodando Regressão Logística

# COMMAND ----------

base_teste_c0.columns

# COMMAND ----------

modelo = pickle.load(open(model,'rb'))

x_teste = base_teste_c0[var_fin_c0]
z_teste = base_teste_c0[chave]

valores_previstos = modelo.predict(x_teste)

probabilidades = modelo.predict_proba(x_teste)
data_prob = pd.DataFrame({'P_1': probabilidades[:, 1]})

z_teste1 = z_teste.reset_index(drop=True)
x_teste1 = x_teste.reset_index(drop=True)
data_prob1 = data_prob.reset_index(drop=True)


x_teste2 = pd.concat([z_teste1, data_prob1], axis=1)

x_teste2

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gerando Grupos Homogêneos 

# COMMAND ----------

x_teste2['P_1_R'] = np.sqrt(x_teste2['P_1'])
np.where(x_teste2['P_1_R'] == 0, -1, x_teste2['P_1_R'])
x_teste2['P_1_R'] = x_teste2['P_1_R'].fillna(-2)

x_teste2['P_1_R_p_5_g_1'] = np.where(x_teste2['P_1_R'] <= 0.08762053, 0.0,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.08762053, x_teste2['P_1_R'] <= 0.21931506), 1.0,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.21931506, x_teste2['P_1_R'] <= 0.507355871), 2.0,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.507355871, x_teste2['P_1_R'] <= 0.634935474), 3.0,
    np.where(x_teste2['P_1_R'] > 0.634935474, 4.0,0)))))


x_teste2['P_1_R_pe_6_g_1'] = np.where(x_teste2['P_1_R'] <= 0.124244564, 0.0,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.124244564, x_teste2['P_1_R'] <= 0.372810666), 1.0,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.372810666, x_teste2['P_1_R'] <= 0.248659773), 2.0,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.248659773, x_teste2['P_1_R'] <= 0.498867554), 3.0,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.498867554, x_teste2['P_1_R'] <= 0.623337704), 4.0,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.623337704, x_teste2['P_1_R'] <= 0.748353398), 5.0,
    np.where(x_teste2['P_1_R'] > 0.748353398, 6.0,0)))))))
             

x_teste2['GH'] = np.where(np.bitwise_and(x_teste2['P_1_R_p_5_g_1'] == 0, x_teste2['P_1_R_pe_6_g_1'] == 0), 0,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_5_g_1'] == 1, x_teste2['P_1_R_pe_6_g_1'] == 0), 1,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_5_g_1'] == 1, x_teste2['P_1_R_pe_6_g_1'] == 2), 3,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_5_g_1'] == 2, x_teste2['P_1_R_pe_6_g_1'] == 1), 2,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_5_g_1'] == 2, x_teste2['P_1_R_pe_6_g_1'] == 2), 2,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_5_g_1'] == 2, x_teste2['P_1_R_pe_6_g_1'] == 3), 4,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_5_g_1'] == 2, x_teste2['P_1_R_pe_6_g_1'] == 4), 4,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_5_g_1'] == 3, x_teste2['P_1_R_pe_6_g_1'] == 4), 5,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_5_g_1'] == 3, x_teste2['P_1_R_pe_6_g_1'] == 5), 5,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_5_g_1'] == 4, x_teste2['P_1_R_pe_6_g_1'] == 5), 6,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_5_g_1'] == 4, x_teste2['P_1_R_pe_6_g_1'] == 6), 6,0)))))))))))             

del x_teste2['P_1_R']
del x_teste2['P_1_R_p_5_g_1']
del x_teste2['P_1_R_pe_6_g_1']

x_teste2

# COMMAND ----------

try:
  dbutils.fs.rm(outputpath, True)
except:
  pass
dbutils.fs.mkdirs(outputpath)

x_teste2.to_csv(open(os.path.join(outputpath_dbfs, 'pre_output:'+data_arquivo+'.csv'),'wb'))
os.path.join(outputpath_dbfs, 'pre_output:'+data_arquivo+'.csv')

# COMMAND ----------

x_teste2.shape