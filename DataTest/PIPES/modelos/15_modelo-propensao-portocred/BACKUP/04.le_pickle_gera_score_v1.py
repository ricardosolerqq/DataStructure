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
chave = 'DOCUMENTO:ID_DIVIDA'

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

#carregar o arquivo em formato tabela
base_dados = pd.read_csv(caminho_base+N_Base, sep=separador_, decimal=decimal_)
base_dados = base_dados[[chave,'DETALHES_CLIENTES_DADOS_SPC','TIPO_EMAIL','TIPO_ENDERECO','DETALHES_CONTRATOS_FAMILIA_PRODUTO','DETALHES_CLIENTES_BANCO_REF_BANCARIA','DETALHES_CLIENTES_NEGATIVADO_SERASA','DETALHES_CLIENTES_CONTA_REF_BANCARIA3','DETALHES_CLIENTES_NEGATIVADO_SPC','DETALHES_CONTRATOS_SALDO_ABERTO','DETALHES_DIVIDAS_PRAZO','NOME_PRODUTO','VALOR_DIVIDA','IDADE_PESSOA','DETALHES_CONTRATOS_SEGURO','DOCUMENTO_PESSOA']]


base_dados.fillna(-3)

#string
base_dados['TIPO_ENDERECO'] = base_dados['TIPO_ENDERECO'].replace(np.nan, '-3')
base_dados['TIPO_EMAIL'] = base_dados['TIPO_EMAIL'].replace(np.nan, '-3')
base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO'] = base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO'].replace(np.nan, '-3')

base_dados['STATUS_SPC'] = np.where((base_dados['DETALHES_CLIENTES_DADOS_SPC'].astype(str).str[:11]) == 'REABILITADO',1,
                            np.where((base_dados['DETALHES_CLIENTES_DADOS_SPC'].astype(str).str.contains(' | ')) == True,0,2))

#numericas
base_dados['NOME_PRODUTO'] = base_dados['NOME_PRODUTO'].replace(np.nan, '-3')
base_dados['VALOR_DIVIDA'] = base_dados['VALOR_DIVIDA'].replace(np.nan, '-3')
base_dados['IDADE_PESSOA'] = base_dados['IDADE_PESSOA'].replace(np.nan, '-3')
base_dados['DOCUMENTO_PESSOA'] = base_dados['DOCUMENTO_PESSOA'].replace(np.nan, '-3')

base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3'] = base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3'].replace(np.nan, '-3')
base_dados['DETALHES_CLIENTES_NEGATIVADO_SERASA'] = base_dados['DETALHES_CLIENTES_NEGATIVADO_SERASA'].replace(np.nan, '-3')
base_dados['DETALHES_CONTRATOS_SALDO_ABERTO'] = base_dados['DETALHES_CONTRATOS_SALDO_ABERTO'].replace(np.nan, '-3')
base_dados['DETALHES_CLIENTES_NEGATIVADO_SPC'] = base_dados['DETALHES_CLIENTES_NEGATIVADO_SPC'].replace(np.nan, '-3')
base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA'] = base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA'].replace(np.nan, '-3')
base_dados['DETALHES_DIVIDAS_PRAZO'] = base_dados['DETALHES_DIVIDAS_PRAZO'].replace(np.nan, '-3')
base_dados['DETALHES_CONTRATOS_SEGURO'] = base_dados['DETALHES_CONTRATOS_SEGURO'].replace(np.nan, '-3')

base_dados = base_dados.apply(pd.to_numeric, errors='ignore')

base_dados['VALOR_DIVIDA'] = base_dados['VALOR_DIVIDA'].astype(float)
base_dados['DOCUMENTO_PESSOA'] = base_dados['DOCUMENTO_PESSOA'].astype(np.int64)
base_dados['NOME_PRODUTO'] = base_dados['NOME_PRODUTO'].astype(np.int64)
base_dados['IDADE_PESSOA'] = base_dados['IDADE_PESSOA'].astype(np.int64)
base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3'] = base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3'].astype(np.int64)
base_dados['DETALHES_CLIENTES_NEGATIVADO_SERASA'] = base_dados['DETALHES_CLIENTES_NEGATIVADO_SERASA'].astype(np.int64)
base_dados['DETALHES_CONTRATOS_SALDO_ABERTO'] = base_dados['DETALHES_CONTRATOS_SALDO_ABERTO'].astype(np.int64)
base_dados['DETALHES_CLIENTES_NEGATIVADO_SPC'] = base_dados['DETALHES_CLIENTES_NEGATIVADO_SPC'].astype(np.int64)
base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA'] = base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA'].astype(np.int64)
base_dados['DETALHES_DIVIDAS_PRAZO'] = base_dados['DETALHES_DIVIDAS_PRAZO'].astype(np.int64)
base_dados['DETALHES_CONTRATOS_SEGURO'] = base_dados['DETALHES_CONTRATOS_SEGURO'].astype(np.int64)

del base_dados['DETALHES_CLIENTES_DADOS_SPC']

base_dados.drop_duplicates(keep=False, inplace=True)

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
np.where(base_dados['TIPO_EMAIL_gh33'] == 3, 1,
np.where(base_dados['TIPO_EMAIL_gh33'] == 4, 1,
np.where(base_dados['TIPO_EMAIL_gh33'] == 5, 0,
np.where(base_dados['TIPO_EMAIL_gh33'] == 6, 0,
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
                                         
                                         
                                         
                                         
                                         
                                         
                                         
base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh30'] = np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO'] == 'AUTO', 0,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO'] == 'CARNE', 1,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO'] == 'CDC TRADICIONAL', 2,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO'] == 'CDCNOVO', 3,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO'] == 'CHEQUE', 4,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO'] == 'CONSIGNADOPRIVADO', 5,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO'] == 'CP LOJA', 6,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO'] == 'DEBITO', 7,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO'] == 'EPFATURA', 8,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO'] == 'FOLHA', 9,
0))))))))))
base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh31'] = np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh30'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh30'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh30'] == 2, 2,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh30'] == 3, 3,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh30'] == 4, 4,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh30'] == 5, 5,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh30'] == 6, 6,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh30'] == 7, 7,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh30'] == 8, 8,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh30'] == 9, 9,
0))))))))))
base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh32'] = np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh31'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh31'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh31'] == 2, 2,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh31'] == 3, 3,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh31'] == 4, 4,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh31'] == 5, 5,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh31'] == 6, 6,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh31'] == 7, 7,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh31'] == 8, 8,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh31'] == 9, 9,
0))))))))))
base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh33'] = np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh32'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh32'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh32'] == 2, 2,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh32'] == 3, 3,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh32'] == 4, 4,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh32'] == 5, 5,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh32'] == 6, 6,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh32'] == 7, 7,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh32'] == 8, 8,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh32'] == 9, 9,
0))))))))))
base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh34'] = np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh33'] == 0, 5,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh33'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh33'] == 2, 2,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh33'] == 3, 2,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh33'] == 4, 4,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh33'] == 5, 5,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh33'] == 6, 6,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh33'] == 7, 7,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh33'] == 8, 6,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh33'] == 9, 9,
0))))))))))
base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh35'] = np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh34'] == 1, 0,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh34'] == 2, 1,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh34'] == 4, 2,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh34'] == 5, 3,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh34'] == 6, 4,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh34'] == 7, 5,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh34'] == 9, 6,
0)))))))
base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh36'] = np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh35'] == 0, 4,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh35'] == 1, 0,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh35'] == 2, 4,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh35'] == 3, 3,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh35'] == 4, 1,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh35'] == 5, 6,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh35'] == 6, 2,
0)))))))
base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh37'] = np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh36'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh36'] == 1, 0,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh36'] == 2, 2,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh36'] == 3, 3,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh36'] == 4, 4,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh36'] == 6, 4,
0))))))
base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh38'] = np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh37'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh37'] == 2, 1,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh37'] == 3, 2,
np.where(base_dados['DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh37'] == 4, 3,
0))))
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
base_dados['DETALHES_CLIENTES_NEGATIVADO_SPC_gh30'] = np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SPC'] == -3, 0,
np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SPC'] == 0, 1,
np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SPC'] == 1, 2,
0)))
base_dados['DETALHES_CLIENTES_NEGATIVADO_SPC_gh31'] = np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SPC_gh30'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SPC_gh30'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SPC_gh30'] == 2, 2,
0)))
base_dados['DETALHES_CLIENTES_NEGATIVADO_SPC_gh32'] = np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SPC_gh31'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SPC_gh31'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SPC_gh31'] == 2, 2,
0)))
base_dados['DETALHES_CLIENTES_NEGATIVADO_SPC_gh33'] = np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SPC_gh32'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SPC_gh32'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SPC_gh32'] == 2, 2,
0)))
base_dados['DETALHES_CLIENTES_NEGATIVADO_SPC_gh34'] = np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SPC_gh33'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SPC_gh33'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SPC_gh33'] == 2, 2,
0)))
base_dados['DETALHES_CLIENTES_NEGATIVADO_SPC_gh35'] = np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SPC_gh34'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SPC_gh34'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SPC_gh34'] == 2, 2,
0)))
base_dados['DETALHES_CLIENTES_NEGATIVADO_SPC_gh36'] = np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SPC_gh35'] == 0, 2,
np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SPC_gh35'] == 1, 0,
np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SPC_gh35'] == 2, 1,
0)))
base_dados['DETALHES_CLIENTES_NEGATIVADO_SPC_gh37'] = np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SPC_gh36'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SPC_gh36'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SPC_gh36'] == 2, 2,
0)))
base_dados['DETALHES_CLIENTES_NEGATIVADO_SPC_gh38'] = np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SPC_gh37'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SPC_gh37'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SPC_gh37'] == 2, 2,
0)))
         
         
         
         
         
         
base_dados['DETALHES_CLIENTES_NEGATIVADO_SERASA_gh30'] = np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SERASA'] == -3, 0,
np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SERASA'] == 0, 1,
np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SERASA'] == 1, 2,
0)))
base_dados['DETALHES_CLIENTES_NEGATIVADO_SERASA_gh31'] = np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SERASA_gh30'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SERASA_gh30'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SERASA_gh30'] == 2, 2,
0)))
base_dados['DETALHES_CLIENTES_NEGATIVADO_SERASA_gh32'] = np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SERASA_gh31'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SERASA_gh31'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SERASA_gh31'] == 2, 2,
0)))
base_dados['DETALHES_CLIENTES_NEGATIVADO_SERASA_gh33'] = np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SERASA_gh32'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SERASA_gh32'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SERASA_gh32'] == 2, 2,
0)))
base_dados['DETALHES_CLIENTES_NEGATIVADO_SERASA_gh34'] = np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SERASA_gh33'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SERASA_gh33'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SERASA_gh33'] == 2, 2,
0)))
base_dados['DETALHES_CLIENTES_NEGATIVADO_SERASA_gh35'] = np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SERASA_gh34'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SERASA_gh34'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SERASA_gh34'] == 2, 2,
0)))
base_dados['DETALHES_CLIENTES_NEGATIVADO_SERASA_gh36'] = np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SERASA_gh35'] == 0, 2,
np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SERASA_gh35'] == 1, 0,
np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SERASA_gh35'] == 2, 1,
0)))
base_dados['DETALHES_CLIENTES_NEGATIVADO_SERASA_gh37'] = np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SERASA_gh36'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SERASA_gh36'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SERASA_gh36'] == 2, 2,
0)))
base_dados['DETALHES_CLIENTES_NEGATIVADO_SERASA_gh38'] = np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SERASA_gh37'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SERASA_gh37'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_NEGATIVADO_SERASA_gh37'] == 2, 2,
0)))
         
         
         
         
         
         
         
base_dados['STATUS_SPC_gh30'] = np.where(base_dados['STATUS_SPC'] == 0, 0,
np.where(base_dados['STATUS_SPC'] == 1, 1,
np.where(base_dados['STATUS_SPC'] == 2, 2,
0)))
base_dados['STATUS_SPC_gh31'] = np.where(base_dados['STATUS_SPC_gh30'] == 0, 0,
np.where(base_dados['STATUS_SPC_gh30'] == 1, 1,
np.where(base_dados['STATUS_SPC_gh30'] == 2, 2,
0)))
base_dados['STATUS_SPC_gh32'] = np.where(base_dados['STATUS_SPC_gh31'] == 0, 0,
np.where(base_dados['STATUS_SPC_gh31'] == 1, 1,
np.where(base_dados['STATUS_SPC_gh31'] == 2, 2,
0)))
base_dados['STATUS_SPC_gh33'] = np.where(base_dados['STATUS_SPC_gh32'] == 0, 0,
np.where(base_dados['STATUS_SPC_gh32'] == 1, 1,
np.where(base_dados['STATUS_SPC_gh32'] == 2, 2,
0)))
base_dados['STATUS_SPC_gh34'] = np.where(base_dados['STATUS_SPC_gh33'] == 0, 0,
np.where(base_dados['STATUS_SPC_gh33'] == 1, 2,
np.where(base_dados['STATUS_SPC_gh33'] == 2, 2,
0)))
base_dados['STATUS_SPC_gh35'] = np.where(base_dados['STATUS_SPC_gh34'] == 0, 0,
np.where(base_dados['STATUS_SPC_gh34'] == 2, 1,
0))
base_dados['STATUS_SPC_gh36'] = np.where(base_dados['STATUS_SPC_gh35'] == 0, 0,
np.where(base_dados['STATUS_SPC_gh35'] == 1, 1,
0))
base_dados['STATUS_SPC_gh37'] = np.where(base_dados['STATUS_SPC_gh36'] == 0, 0,
np.where(base_dados['STATUS_SPC_gh36'] == 1, 1,
0))
base_dados['STATUS_SPC_gh38'] = np.where(base_dados['STATUS_SPC_gh37'] == 0, 0,
np.where(base_dados['STATUS_SPC_gh37'] == 1, 1,
0))
                                         
                                         
                                         
                                         
                                         
                                         
base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh30'] = np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA'] == -3, 0,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA'] == 3, 2,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA'] == 19, 3,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA'] == 21, 4,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA'] == 33, 5,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA'] == 36, 6,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA'] == 37, 7,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA'] == 41, 8,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA'] == 47, 9,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA'] == 70, 10,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA'] == 77, 11,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA'] == 85, 12,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA'] == 104, 13,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA'] == 121, 14,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA'] == 136, 15,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA'] == 230, 16,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA'] == 235, 17,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA'] == 237, 18,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA'] == 260, 19,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA'] == 335, 20,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA'] == 341, 21,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA'] == 353, 22,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA'] == 356, 23,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA'] == 389, 24,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA'] == 399, 25,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA'] == 409, 26,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA'] == 422, 27,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA'] == 745, 28,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA'] == 748, 29,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA'] == 756, 30,
0)))))))))))))))))))))))))))))))
base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh31'] = np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh30'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh30'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh30'] == 2, 2,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh30'] == 3, 3,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh30'] == 4, 4,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh30'] == 5, 5,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh30'] == 6, 6,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh30'] == 7, 6,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh30'] == 8, 8,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh30'] == 9, 9,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh30'] == 10, 10,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh30'] == 11, 10,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh30'] == 12, 12,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh30'] == 13, 13,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh30'] == 14, 14,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh30'] == 15, 14,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh30'] == 16, 16,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh30'] == 17, 16,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh30'] == 18, 18,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh30'] == 19, 19,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh30'] == 20, 20,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh30'] == 21, 21,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh30'] == 22, 22,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh30'] == 23, 22,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh30'] == 24, 22,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh30'] == 25, 22,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh30'] == 26, 22,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh30'] == 27, 27,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh30'] == 28, 28,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh30'] == 29, 29,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh30'] == 30, 30,
0)))))))))))))))))))))))))))))))
base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh32'] = np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh31'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh31'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh31'] == 2, 2,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh31'] == 3, 3,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh31'] == 4, 4,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh31'] == 5, 5,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh31'] == 6, 6,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh31'] == 8, 7,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh31'] == 9, 8,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh31'] == 10, 9,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh31'] == 12, 10,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh31'] == 13, 11,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh31'] == 14, 12,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh31'] == 16, 13,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh31'] == 18, 14,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh31'] == 19, 15,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh31'] == 20, 16,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh31'] == 21, 17,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh31'] == 22, 18,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh31'] == 27, 19,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh31'] == 28, 20,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh31'] == 29, 21,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh31'] == 30, 22,
0)))))))))))))))))))))))
base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh33'] = np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh32'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh32'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh32'] == 2, 2,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh32'] == 3, 3,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh32'] == 4, 4,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh32'] == 5, 5,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh32'] == 6, 6,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh32'] == 7, 7,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh32'] == 8, 8,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh32'] == 9, 9,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh32'] == 10, 10,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh32'] == 11, 11,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh32'] == 12, 12,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh32'] == 13, 13,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh32'] == 14, 14,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh32'] == 15, 15,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh32'] == 16, 16,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh32'] == 17, 17,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh32'] == 18, 18,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh32'] == 19, 19,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh32'] == 20, 20,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh32'] == 21, 21,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh32'] == 22, 22,
0)))))))))))))))))))))))
base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh34'] = np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh33'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh33'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh33'] == 2, 1,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh33'] == 3, 22,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh33'] == 4, 17,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh33'] == 5, 5,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh33'] == 6, 22,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh33'] == 7, 7,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh33'] == 8, 22,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh33'] == 9, 11,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh33'] == 10, 22,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh33'] == 11, 11,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh33'] == 12, 17,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh33'] == 13, 22,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh33'] == 14, 14,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh33'] == 15, 22,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh33'] == 16, 17,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh33'] == 17, 17,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh33'] == 18, 22,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh33'] == 19, 17,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh33'] == 20, 22,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh33'] == 21, 17,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh33'] == 22, 22,
0)))))))))))))))))))))))
base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh35'] = np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh34'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh34'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh34'] == 5, 2,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh34'] == 7, 3,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh34'] == 11, 4,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh34'] == 14, 5,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh34'] == 17, 6,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh34'] == 22, 7,
0))))))))
base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh36'] = np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh35'] == 0, 1,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh35'] == 1, 4,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh35'] == 2, 5,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh35'] == 3, 3,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh35'] == 4, 1,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh35'] == 5, 5,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh35'] == 6, 5,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh35'] == 7, 0,
0))))))))
base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh37'] = np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh36'] == 0, 1,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh36'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh36'] == 3, 2,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh36'] == 4, 3,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh36'] == 5, 4,
0)))))
base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh38'] = np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh37'] == 1, 0,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh37'] == 2, 1,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh37'] == 3, 2,
np.where(base_dados['DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh37'] == 4, 3,
0))))
         
         
         
         
         
         
                                                                         
                                                                         
                                                                         
base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh30'] = np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3'] == -3, 0,
np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3'] == 2, 1,
np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3'] == 13907, 2,
np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3'] == 46108, 3,
np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3'] == 182992, 4,
np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3'] == 610171542, 5,
0))))))
base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh31'] = np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh30'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh30'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh30'] == 2, 2,
np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh30'] == 3, 2,
np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh30'] == 4, 4,
np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh30'] == 5, 5,
0))))))
base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh32'] = np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh31'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh31'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh31'] == 2, 2,
np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh31'] == 4, 3,
np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh31'] == 5, 4,
0)))))
base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh33'] = np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh32'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh32'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh32'] == 2, 2,
np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh32'] == 3, 3,
np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh32'] == 4, 4,
0)))))
base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh34'] = np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh33'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh33'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh33'] == 2, 2,
np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh33'] == 3, 1,
np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh33'] == 4, 2,
0)))))
base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh35'] = np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh34'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh34'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh34'] == 2, 2,
0)))
base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh36'] = np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh35'] == 0, 1,
np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh35'] == 1, 2,
np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh35'] == 2, 0,
0)))
base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh37'] = np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh36'] == '0', 1,
np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh36'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh36'] == 2, 2,
0)))
base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh38'] = np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh37'] == '1', 0,
np.where(base_dados['DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh37'] == 2, 1,
0))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Gerando variáveis numéricas contínuas

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parte 1 de 2

# COMMAND ----------

         
base_dados['VALOR_DIVIDA__pe_10'] = np.where(base_dados['VALOR_DIVIDA'] <= 21624.0, 0.0,
np.where(np.bitwise_and(base_dados['VALOR_DIVIDA'] > 21624.0, base_dados['VALOR_DIVIDA'] <= 43075.0), 1.0,
np.where(np.bitwise_and(base_dados['VALOR_DIVIDA'] > 43075.0, base_dados['VALOR_DIVIDA'] <= 64848.0), 2.0,
np.where(np.bitwise_and(base_dados['VALOR_DIVIDA'] > 64848.0, base_dados['VALOR_DIVIDA'] <= 86521.0), 3.0,
np.where(np.bitwise_and(base_dados['VALOR_DIVIDA'] > 86521.0, base_dados['VALOR_DIVIDA'] <= 107809.0), 4.0,
np.where(np.bitwise_and(base_dados['VALOR_DIVIDA'] > 107809.0, base_dados['VALOR_DIVIDA'] <= 127849.0), 5.0,
np.where(np.bitwise_and(base_dados['VALOR_DIVIDA'] > 127849.0, base_dados['VALOR_DIVIDA'] <= 151400.0), 6.0,
np.where(np.bitwise_and(base_dados['VALOR_DIVIDA'] > 151400.0, base_dados['VALOR_DIVIDA'] <= 172257.0), 7.0,
np.where(np.bitwise_and(base_dados['VALOR_DIVIDA'] > 172257.0, base_dados['VALOR_DIVIDA'] <= 193973.0), 8.0,
np.where(base_dados['VALOR_DIVIDA'] > 193973.0, 9.0,
 -2))))))))))

base_dados['VALOR_DIVIDA__pe_10_g_1_1'] = np.where(base_dados['VALOR_DIVIDA__pe_10'] == -2.0, 2,
np.where(base_dados['VALOR_DIVIDA__pe_10'] == 0.0, 1,
np.where(base_dados['VALOR_DIVIDA__pe_10'] == 1.0, 0,
np.where(base_dados['VALOR_DIVIDA__pe_10'] == 2.0, 2,
np.where(base_dados['VALOR_DIVIDA__pe_10'] == 3.0, 0,
np.where(base_dados['VALOR_DIVIDA__pe_10'] == 4.0, 1,
np.where(base_dados['VALOR_DIVIDA__pe_10'] == 5.0, 0,
np.where(base_dados['VALOR_DIVIDA__pe_10'] == 6.0, 2,
np.where(base_dados['VALOR_DIVIDA__pe_10'] == 7.0, 0,
np.where(base_dados['VALOR_DIVIDA__pe_10'] == 8.0, 2,
np.where(base_dados['VALOR_DIVIDA__pe_10'] == 9.0, 2,
 0)))))))))))
base_dados['VALOR_DIVIDA__pe_10_g_1_2'] = np.where(base_dados['VALOR_DIVIDA__pe_10_g_1_1'] == 0, 1,
np.where(base_dados['VALOR_DIVIDA__pe_10_g_1_1'] == 1, 0,
np.where(base_dados['VALOR_DIVIDA__pe_10_g_1_1'] == 2, 1,
 0)))
base_dados['VALOR_DIVIDA__pe_10_g_1'] = np.where(base_dados['VALOR_DIVIDA__pe_10_g_1_2'] == 0, 0,
np.where(base_dados['VALOR_DIVIDA__pe_10_g_1_2'] == 1, 1,
 0))
                                                 
                                                 
                                                 
                                                 
                                                 
                                                 
                                                 
                                                 
base_dados['VALOR_DIVIDA__L'] = np.log(base_dados['VALOR_DIVIDA'])
np.where(base_dados['VALOR_DIVIDA__L'] == 0, -1, base_dados['VALOR_DIVIDA__L'])
base_dados['VALOR_DIVIDA__L'] = base_dados['VALOR_DIVIDA__L'].fillna(-2)
base_dados['VALOR_DIVIDA__L__p_20'] = np.where(base_dados['VALOR_DIVIDA__L'] == 0 , -1.0,
np.where(np.bitwise_and(base_dados['VALOR_DIVIDA__L'] > 0.0, base_dados['VALOR_DIVIDA__L'] <= 8.70549681988774), 0.0,
np.where(np.bitwise_and(base_dados['VALOR_DIVIDA__L'] > 8.70549681988774, base_dados['VALOR_DIVIDA__L'] <= 9.076580381796658), 1.0,
np.where(np.bitwise_and(base_dados['VALOR_DIVIDA__L'] > 9.076580381796658, base_dados['VALOR_DIVIDA__L'] <= 9.269457976459272), 2.0,
np.where(np.bitwise_and(base_dados['VALOR_DIVIDA__L'] > 9.269457976459272, base_dados['VALOR_DIVIDA__L'] <= 9.436439551116026), 3.0,
np.where(np.bitwise_and(base_dados['VALOR_DIVIDA__L'] > 9.436439551116026, base_dados['VALOR_DIVIDA__L'] <= 9.617404201448045), 4.0,
np.where(np.bitwise_and(base_dados['VALOR_DIVIDA__L'] > 9.617404201448045, base_dados['VALOR_DIVIDA__L'] <= 9.72913416539135), 5.0,
np.where(np.bitwise_and(base_dados['VALOR_DIVIDA__L'] > 9.72913416539135, base_dados['VALOR_DIVIDA__L'] <= 9.863446502998944), 6.0,
np.where(np.bitwise_and(base_dados['VALOR_DIVIDA__L'] > 9.863446502998944, base_dados['VALOR_DIVIDA__L'] <= 10.344963098167325), 11.0,
np.where(np.bitwise_and(base_dados['VALOR_DIVIDA__L'] > 10.344963098167325, base_dados['VALOR_DIVIDA__L'] <= 10.460729095610086), 12.0,
np.where(np.bitwise_and(base_dados['VALOR_DIVIDA__L'] > 10.460729095610086, base_dados['VALOR_DIVIDA__L'] <= 10.718166289816029), 14.0,
np.where(np.bitwise_and(base_dados['VALOR_DIVIDA__L'] > 10.718166289816029, base_dados['VALOR_DIVIDA__L'] <= 10.883166143790458), 15.0,
np.where(np.bitwise_and(base_dados['VALOR_DIVIDA__L'] > 10.883166143790458, base_dados['VALOR_DIVIDA__L'] <= 11.060195778394762), 16.0,
np.where(np.bitwise_and(base_dados['VALOR_DIVIDA__L'] > 11.060195778394762, base_dados['VALOR_DIVIDA__L'] <= 11.289781913656018), 17.0,
np.where(np.bitwise_and(base_dados['VALOR_DIVIDA__L'] > 11.289781913656018, base_dados['VALOR_DIVIDA__L'] <= 11.638905920891801), 18.0,
np.where(base_dados['VALOR_DIVIDA__L'] > 11.638905920891801, 19.0,
 0))))))))))))))))
base_dados['VALOR_DIVIDA__L__p_20_g_1_1'] = np.where(base_dados['VALOR_DIVIDA__L__p_20'] == -1.0, 3,
np.where(base_dados['VALOR_DIVIDA__L__p_20'] == 0.0, 3,
np.where(base_dados['VALOR_DIVIDA__L__p_20'] == 1.0, 3,
np.where(base_dados['VALOR_DIVIDA__L__p_20'] == 2.0, 2,
np.where(base_dados['VALOR_DIVIDA__L__p_20'] == 3.0, 1,
np.where(base_dados['VALOR_DIVIDA__L__p_20'] == 4.0, 1,
np.where(base_dados['VALOR_DIVIDA__L__p_20'] == 5.0, 2,
np.where(base_dados['VALOR_DIVIDA__L__p_20'] == 6.0, 2,
np.where(base_dados['VALOR_DIVIDA__L__p_20'] == 11.0, 0,
np.where(base_dados['VALOR_DIVIDA__L__p_20'] == 12.0, 0,
np.where(base_dados['VALOR_DIVIDA__L__p_20'] == 14.0, 1,
np.where(base_dados['VALOR_DIVIDA__L__p_20'] == 15.0, 2,
np.where(base_dados['VALOR_DIVIDA__L__p_20'] == 16.0, 2,
np.where(base_dados['VALOR_DIVIDA__L__p_20'] == 17.0, 0,
np.where(base_dados['VALOR_DIVIDA__L__p_20'] == 18.0, 1,
np.where(base_dados['VALOR_DIVIDA__L__p_20'] == 19.0, 1,
 0))))))))))))))))
base_dados['VALOR_DIVIDA__L__p_20_g_1_2'] = np.where(base_dados['VALOR_DIVIDA__L__p_20_g_1_1'] == 0, 1,
np.where(base_dados['VALOR_DIVIDA__L__p_20_g_1_1'] == 1, 3,
np.where(base_dados['VALOR_DIVIDA__L__p_20_g_1_1'] == 2, 1,
np.where(base_dados['VALOR_DIVIDA__L__p_20_g_1_1'] == 3, 0,
 0))))
base_dados['VALOR_DIVIDA__L__p_20_g_1'] = np.where(base_dados['VALOR_DIVIDA__L__p_20_g_1_2'] == 0, 0,
np.where(base_dados['VALOR_DIVIDA__L__p_20_g_1_2'] == 1, 1,
np.where(base_dados['VALOR_DIVIDA__L__p_20_g_1_2'] == 3, 2,
 0)))
         
         
         
         
         
        
base_dados['NOME_PRODUTO__pe_3'] = np.where(base_dados['NOME_PRODUTO'] <= 104.0, 0.0,
np.where(np.bitwise_and(base_dados['NOME_PRODUTO'] > 104.0, base_dados['NOME_PRODUTO'] <= 403.0), 1.0,
np.where(base_dados['NOME_PRODUTO'] > 403.0, 2.0,
 -2)))
base_dados['NOME_PRODUTO__pe_3_g_1_1'] = np.where(base_dados['NOME_PRODUTO__pe_3'] == -2.0, 1,
np.where(base_dados['NOME_PRODUTO__pe_3'] == 0.0, 2,
np.where(base_dados['NOME_PRODUTO__pe_3'] == 1.0, 2,
np.where(base_dados['NOME_PRODUTO__pe_3'] == 2.0, 0,
 0))))
base_dados['NOME_PRODUTO__pe_3_g_1_2'] = np.where(base_dados['NOME_PRODUTO__pe_3_g_1_1'] == 0, 1,
np.where(base_dados['NOME_PRODUTO__pe_3_g_1_1'] == 1, 2,
np.where(base_dados['NOME_PRODUTO__pe_3_g_1_1'] == 2, 0,
 0)))
base_dados['NOME_PRODUTO__pe_3_g_1'] = np.where(base_dados['NOME_PRODUTO__pe_3_g_1_2'] == 0, 0,
np.where(base_dados['NOME_PRODUTO__pe_3_g_1_2'] == 1, 1,
np.where(base_dados['NOME_PRODUTO__pe_3_g_1_2'] == 2, 2,
 0)))
         
         
         
         
         
         
         
base_dados['NOME_PRODUTO__C'] = np.cos(base_dados['NOME_PRODUTO'])
np.where(base_dados['NOME_PRODUTO__C'] == 0, -1, base_dados['NOME_PRODUTO__C'])
base_dados['NOME_PRODUTO__C'] = base_dados['NOME_PRODUTO__C'].fillna(-2)
base_dados['NOME_PRODUTO__C__p_10'] = np.where(np.bitwise_and(base_dados['NOME_PRODUTO__C'] >= -0.9998449053985173, base_dados['NOME_PRODUTO__C'] <= 0.3089367201199639), 3.0,
np.where(np.bitwise_and(base_dados['NOME_PRODUTO__C'] > 0.3089367201199639, base_dados['NOME_PRODUTO__C'] <= 0.39990242950467825), 4.0,
np.where(np.bitwise_and(base_dados['NOME_PRODUTO__C'] > 0.39990242950467825, base_dados['NOME_PRODUTO__C'] <= 0.6670054417882312), 5.0,
np.where(np.bitwise_and(base_dados['NOME_PRODUTO__C'] > 0.6670054417882312, base_dados['NOME_PRODUTO__C'] <= 0.8623188722876839), 6.0,
np.where(np.bitwise_and(base_dados['NOME_PRODUTO__C'] > 0.8623188722876839, base_dados['NOME_PRODUTO__C'] <= 0.95768551238535), 7.0,
np.where(base_dados['NOME_PRODUTO__C'] > 0.95768551238535, 8.0,
 -2))))))
base_dados['NOME_PRODUTO__C__p_10_g_1_1'] = np.where(base_dados['NOME_PRODUTO__C__p_10'] == -2.0, 0,
np.where(base_dados['NOME_PRODUTO__C__p_10'] == 3.0, 4,
np.where(base_dados['NOME_PRODUTO__C__p_10'] == 4.0, 2,
np.where(base_dados['NOME_PRODUTO__C__p_10'] == 5.0, 4,
np.where(base_dados['NOME_PRODUTO__C__p_10'] == 6.0, 3,
np.where(base_dados['NOME_PRODUTO__C__p_10'] == 7.0, 1,
np.where(base_dados['NOME_PRODUTO__C__p_10'] == 8.0, 4,
 0)))))))
base_dados['NOME_PRODUTO__C__p_10_g_1_2'] = np.where(base_dados['NOME_PRODUTO__C__p_10_g_1_1'] == 0, 2,
np.where(base_dados['NOME_PRODUTO__C__p_10_g_1_1'] == 1, 1,
np.where(base_dados['NOME_PRODUTO__C__p_10_g_1_1'] == 2, 4,
np.where(base_dados['NOME_PRODUTO__C__p_10_g_1_1'] == 3, 0,
np.where(base_dados['NOME_PRODUTO__C__p_10_g_1_1'] == 4, 2,
 0)))))
base_dados['NOME_PRODUTO__C__p_10_g_1'] = np.where(base_dados['NOME_PRODUTO__C__p_10_g_1_2'] == 0, 0,
np.where(base_dados['NOME_PRODUTO__C__p_10_g_1_2'] == 1, 1,
np.where(base_dados['NOME_PRODUTO__C__p_10_g_1_2'] == 2, 2,
np.where(base_dados['NOME_PRODUTO__C__p_10_g_1_2'] == 4, 3,
 0))))
         
         
         
         
        
         
base_dados['IDADE_PESSOA__p_6'] = np.where(base_dados['IDADE_PESSOA'] <= 35, 0,
np.where(np.bitwise_and(base_dados['IDADE_PESSOA'] > 35, base_dados['IDADE_PESSOA'] <= 44), 1,
np.where(np.bitwise_and(base_dados['IDADE_PESSOA'] > 44, base_dados['IDADE_PESSOA'] <= 52), 2,
np.where(np.bitwise_and(base_dados['IDADE_PESSOA'] > 52, base_dados['IDADE_PESSOA'] <= 60), 3,
np.where(np.bitwise_and(base_dados['IDADE_PESSOA'] > 60, base_dados['IDADE_PESSOA'] <= 69), 4,
np.where(base_dados['IDADE_PESSOA'] > 69, 5,
 0))))))
base_dados['IDADE_PESSOA__p_6_g_1_1'] = np.where(base_dados['IDADE_PESSOA__p_6'] == 0, 0,
np.where(base_dados['IDADE_PESSOA__p_6'] == 1, 0,
np.where(base_dados['IDADE_PESSOA__p_6'] == 2, 0,
np.where(base_dados['IDADE_PESSOA__p_6'] == 3, 0,
np.where(base_dados['IDADE_PESSOA__p_6'] == 4, 0,
np.where(base_dados['IDADE_PESSOA__p_6'] == 5, 1,
 0))))))
base_dados['IDADE_PESSOA__p_6_g_1_2'] = np.where(base_dados['IDADE_PESSOA__p_6_g_1_1'] == 0, 1,
np.where(base_dados['IDADE_PESSOA__p_6_g_1_1'] == 1, 0,
 0))
base_dados['IDADE_PESSOA__p_6_g_1'] = np.where(base_dados['IDADE_PESSOA__p_6_g_1_2'] == 0, 0,
np.where(base_dados['IDADE_PESSOA__p_6_g_1_2'] == 1, 1,
 0))
                                               
                                               
                                               
                                               
                                               
base_dados['IDADE_PESSOA__pe_5'] = np.where(base_dados['IDADE_PESSOA'] <= 24.0, 1.0,
np.where(np.bitwise_and(base_dados['IDADE_PESSOA'] > 24.0, base_dados['IDADE_PESSOA'] <= 36.0), 2.0,
np.where(np.bitwise_and(base_dados['IDADE_PESSOA'] > 36.0, base_dados['IDADE_PESSOA'] <= 48.0), 3.0,
np.where(base_dados['IDADE_PESSOA'] > 48.0, 4.0,
 -2))))
base_dados['IDADE_PESSOA__pe_5_g_1_1'] = np.where(base_dados['IDADE_PESSOA__pe_5'] == -2.0, 1,
np.where(base_dados['IDADE_PESSOA__pe_5'] == 1.0, 1,
np.where(base_dados['IDADE_PESSOA__pe_5'] == 2.0, 0,
np.where(base_dados['IDADE_PESSOA__pe_5'] == 3.0, 0,
np.where(base_dados['IDADE_PESSOA__pe_5'] == 4.0, 0,
 0)))))
base_dados['IDADE_PESSOA__pe_5_g_1_2'] = np.where(base_dados['IDADE_PESSOA__pe_5_g_1_1'] == 0, 1,
np.where(base_dados['IDADE_PESSOA__pe_5_g_1_1'] == 1, 0,
 0))
base_dados['IDADE_PESSOA__pe_5_g_1'] = np.where(base_dados['IDADE_PESSOA__pe_5_g_1_2'] == 0, 0,
np.where(base_dados['IDADE_PESSOA__pe_5_g_1_2'] == 1, 1,
 0))
                                            
                                            
                                            
                                            
                                            
                                            
base_dados['DOCUMENTO_PESSOA__p_5'] = np.where(base_dados['DOCUMENTO_PESSOA'] <= 6057651898, 0,
np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA'] > 6057651898, base_dados['DOCUMENTO_PESSOA'] <= 20024434833), 1,
np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA'] > 20024434833, base_dados['DOCUMENTO_PESSOA'] <= 37682857072), 2,
np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA'] > 37682857072, base_dados['DOCUMENTO_PESSOA'] <= 64572277087), 3,
np.where(base_dados['DOCUMENTO_PESSOA'] > 64572277087, 4,
 0)))))
base_dados['DOCUMENTO_PESSOA__p_5_g_1_1'] = np.where(base_dados['DOCUMENTO_PESSOA__p_5'] == 0, 1,
np.where(base_dados['DOCUMENTO_PESSOA__p_5'] == 1, 2,
np.where(base_dados['DOCUMENTO_PESSOA__p_5'] == 2, 1,
np.where(base_dados['DOCUMENTO_PESSOA__p_5'] == 3, 0,
np.where(base_dados['DOCUMENTO_PESSOA__p_5'] == 4, 2,
 0)))))
base_dados['DOCUMENTO_PESSOA__p_5_g_1_2'] = np.where(base_dados['DOCUMENTO_PESSOA__p_5_g_1_1'] == 0, 0,
np.where(base_dados['DOCUMENTO_PESSOA__p_5_g_1_1'] == 1, 1,
np.where(base_dados['DOCUMENTO_PESSOA__p_5_g_1_1'] == 2, 2,
 0)))
base_dados['DOCUMENTO_PESSOA__p_5_g_1'] = np.where(base_dados['DOCUMENTO_PESSOA__p_5_g_1_2'] == 0, 0,
np.where(base_dados['DOCUMENTO_PESSOA__p_5_g_1_2'] == 1, 1,
np.where(base_dados['DOCUMENTO_PESSOA__p_5_g_1_2'] == 2, 2,
 0)))
         
         
         
         
                
base_dados['DOCUMENTO_PESSOA__pe_20'] = np.where(base_dados['DOCUMENTO_PESSOA'] <= 4927920623.0, 0.0,
np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA'] > 4927920623.0, base_dados['DOCUMENTO_PESSOA'] <= 9861530100.0), 1.0,
np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA'] > 9861530100.0, base_dados['DOCUMENTO_PESSOA'] <= 14779173817.0), 2.0,
np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA'] > 14779173817.0, base_dados['DOCUMENTO_PESSOA'] <= 19743750649.0), 3.0,
np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA'] > 19743750649.0, base_dados['DOCUMENTO_PESSOA'] <= 24430137668.0), 4.0,
np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA'] > 24430137668.0, base_dados['DOCUMENTO_PESSOA'] <= 29644089049.0), 5.0,
np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA'] > 29644089049.0, base_dados['DOCUMENTO_PESSOA'] <= 34501508809.0), 6.0,
np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA'] > 34501508809.0, base_dados['DOCUMENTO_PESSOA'] <= 39461785020.0), 7.0,
np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA'] > 39461785020.0, base_dados['DOCUMENTO_PESSOA'] <= 44421087572.0), 8.0,
np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA'] > 44421087572.0, base_dados['DOCUMENTO_PESSOA'] <= 49387561020.0), 9.0,
np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA'] > 49387561020.0, base_dados['DOCUMENTO_PESSOA'] <= 54351219572.0), 10.0,
np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA'] > 54351219572.0, base_dados['DOCUMENTO_PESSOA'] <= 59247487072.0), 11.0,
np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA'] > 59247487072.0, base_dados['DOCUMENTO_PESSOA'] <= 64191095404.0), 12.0,
np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA'] > 64191095404.0, base_dados['DOCUMENTO_PESSOA'] <= 69047979087.0), 13.0,
np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA'] > 69047979087.0, base_dados['DOCUMENTO_PESSOA'] <= 74078739504.0), 14.0,
np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA'] > 74078739504.0, base_dados['DOCUMENTO_PESSOA'] <= 78859654068.0), 15.0,
np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA'] > 78859654068.0, base_dados['DOCUMENTO_PESSOA'] <= 83921257620.0), 16.0,
np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA'] > 83921257620.0, base_dados['DOCUMENTO_PESSOA'] <= 88865800020.0), 17.0,
np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA'] > 88865800020.0, base_dados['DOCUMENTO_PESSOA'] <= 93864809053.0), 18.0,
np.where(base_dados['DOCUMENTO_PESSOA'] > 93864809053.0, 19.0,
 -2))))))))))))))))))))

base_dados['DOCUMENTO_PESSOA__pe_20_g_1_1'] = np.where(base_dados['DOCUMENTO_PESSOA__pe_20'] == -2.0, 2,
np.where(base_dados['DOCUMENTO_PESSOA__pe_20'] == 0.0, 1,
np.where(base_dados['DOCUMENTO_PESSOA__pe_20'] == 1.0, 0,
np.where(base_dados['DOCUMENTO_PESSOA__pe_20'] == 2.0, 2,
np.where(base_dados['DOCUMENTO_PESSOA__pe_20'] == 3.0, 2,
np.where(base_dados['DOCUMENTO_PESSOA__pe_20'] == 4.0, 2,
np.where(base_dados['DOCUMENTO_PESSOA__pe_20'] == 5.0, 1,
np.where(base_dados['DOCUMENTO_PESSOA__pe_20'] == 6.0, 0,
np.where(base_dados['DOCUMENTO_PESSOA__pe_20'] == 7.0, 3,
np.where(base_dados['DOCUMENTO_PESSOA__pe_20'] == 8.0, 1,
np.where(base_dados['DOCUMENTO_PESSOA__pe_20'] == 9.0, 1,
np.where(base_dados['DOCUMENTO_PESSOA__pe_20'] == 10.0, 0,
np.where(base_dados['DOCUMENTO_PESSOA__pe_20'] == 11.0, 3,
np.where(base_dados['DOCUMENTO_PESSOA__pe_20'] == 12.0, 3,
np.where(base_dados['DOCUMENTO_PESSOA__pe_20'] == 13.0, 1,
np.where(base_dados['DOCUMENTO_PESSOA__pe_20'] == 14.0, 1,
np.where(base_dados['DOCUMENTO_PESSOA__pe_20'] == 15.0, 2,
np.where(base_dados['DOCUMENTO_PESSOA__pe_20'] == 16.0, 3,
np.where(base_dados['DOCUMENTO_PESSOA__pe_20'] == 17.0, 2,
np.where(base_dados['DOCUMENTO_PESSOA__pe_20'] == 18.0, 2,
np.where(base_dados['DOCUMENTO_PESSOA__pe_20'] == 19.0, 3,
 0)))))))))))))))))))))
base_dados['DOCUMENTO_PESSOA__pe_20_g_1_2'] = np.where(base_dados['DOCUMENTO_PESSOA__pe_20_g_1_1'] == 0, 2,
np.where(base_dados['DOCUMENTO_PESSOA__pe_20_g_1_1'] == 1, 0,
np.where(base_dados['DOCUMENTO_PESSOA__pe_20_g_1_1'] == 2, 2,
np.where(base_dados['DOCUMENTO_PESSOA__pe_20_g_1_1'] == 3, 0,
 0))))
base_dados['DOCUMENTO_PESSOA__pe_20_g_1'] = np.where(base_dados['DOCUMENTO_PESSOA__pe_20_g_1_2'] == 0, 0,
np.where(base_dados['DOCUMENTO_PESSOA__pe_20_g_1_2'] == 2, 1,
 0))
         
         
         
         
         
         
         
base_dados['DETALHES_CONTRATOS_SEGURO__pe_10'] = np.where(base_dados['DETALHES_CONTRATOS_SEGURO'] == 0 , -1.0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_SEGURO'] > 0.0, base_dados['DETALHES_CONTRATOS_SEGURO'] <= 97.0), 0.0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_SEGURO'] > 97.0, base_dados['DETALHES_CONTRATOS_SEGURO'] <= 195.0), 1.0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_SEGURO'] > 195.0, base_dados['DETALHES_CONTRATOS_SEGURO'] <= 292.0), 2.0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_SEGURO'] > 292.0, base_dados['DETALHES_CONTRATOS_SEGURO'] <= 388.0), 3.0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_SEGURO'] > 388.0, base_dados['DETALHES_CONTRATOS_SEGURO'] <= 486.0), 4.0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_SEGURO'] > 486.0, base_dados['DETALHES_CONTRATOS_SEGURO'] <= 585.0), 5.0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_SEGURO'] > 585.0, base_dados['DETALHES_CONTRATOS_SEGURO'] <= 682.0), 6.0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_SEGURO'] > 682.0, base_dados['DETALHES_CONTRATOS_SEGURO'] <= 753.0), 7.0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_SEGURO'] > 753.0, base_dados['DETALHES_CONTRATOS_SEGURO'] <= 845.0), 8.0,
np.where(base_dados['DETALHES_CONTRATOS_SEGURO'] > 845.0, 9.0,
 -2)))))))))))
base_dados['DETALHES_CONTRATOS_SEGURO__pe_10_g_1_1'] = np.where(base_dados['DETALHES_CONTRATOS_SEGURO__pe_10'] == -2.0, 2,
np.where(base_dados['DETALHES_CONTRATOS_SEGURO__pe_10'] == -1.0, 0,
np.where(base_dados['DETALHES_CONTRATOS_SEGURO__pe_10'] == 0.0, 2,
np.where(base_dados['DETALHES_CONTRATOS_SEGURO__pe_10'] == 1.0, 1,
np.where(base_dados['DETALHES_CONTRATOS_SEGURO__pe_10'] == 2.0, 2,
np.where(base_dados['DETALHES_CONTRATOS_SEGURO__pe_10'] == 3.0, 2,
np.where(base_dados['DETALHES_CONTRATOS_SEGURO__pe_10'] == 4.0, 2,
np.where(base_dados['DETALHES_CONTRATOS_SEGURO__pe_10'] == 5.0, 1,
np.where(base_dados['DETALHES_CONTRATOS_SEGURO__pe_10'] == 6.0, 2,
np.where(base_dados['DETALHES_CONTRATOS_SEGURO__pe_10'] == 7.0, 2,
np.where(base_dados['DETALHES_CONTRATOS_SEGURO__pe_10'] == 8.0, 2,
np.where(base_dados['DETALHES_CONTRATOS_SEGURO__pe_10'] == 9.0, 2,
 0))))))))))))
base_dados['DETALHES_CONTRATOS_SEGURO__pe_10_g_1_2'] = np.where(base_dados['DETALHES_CONTRATOS_SEGURO__pe_10_g_1_1'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_SEGURO__pe_10_g_1_1'] == 1, 2,
np.where(base_dados['DETALHES_CONTRATOS_SEGURO__pe_10_g_1_1'] == 2, 0,
 0)))
base_dados['DETALHES_CONTRATOS_SEGURO__pe_10_g_1'] = np.where(base_dados['DETALHES_CONTRATOS_SEGURO__pe_10_g_1_2'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_SEGURO__pe_10_g_1_2'] == 2, 1,
 0))
                                                              
                                                              
                                                              
                                                              
                                                              
                                                              
base_dados['DETALHES_CONTRATOS_SEGURO__L'] = np.log(base_dados['DETALHES_CONTRATOS_SEGURO'])
np.where(base_dados['DETALHES_CONTRATOS_SEGURO__L'] == 0, -1, base_dados['DETALHES_CONTRATOS_SEGURO__L'])
base_dados['DETALHES_CONTRATOS_SEGURO__L'] = base_dados['DETALHES_CONTRATOS_SEGURO__L'].fillna(-2)
base_dados['DETALHES_CONTRATOS_SEGURO__L__p_4'] = np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_SEGURO__L'] >= -1.0, base_dados['DETALHES_CONTRATOS_SEGURO__L'] <= 4.30406509320417), 2.0,
np.where(base_dados['DETALHES_CONTRATOS_SEGURO__L'] > 4.30406509320417, 3.0,
 -2))
base_dados['DETALHES_CONTRATOS_SEGURO__L__p_4_g_1_1'] = np.where(base_dados['DETALHES_CONTRATOS_SEGURO__L__p_4'] == -2.0, 0,
np.where(base_dados['DETALHES_CONTRATOS_SEGURO__L__p_4'] == 2.0, 2,
np.where(base_dados['DETALHES_CONTRATOS_SEGURO__L__p_4'] == 3.0, 1,
 0)))
base_dados['DETALHES_CONTRATOS_SEGURO__L__p_4_g_1_2'] = np.where(base_dados['DETALHES_CONTRATOS_SEGURO__L__p_4_g_1_1'] == 0, 1,
np.where(base_dados['DETALHES_CONTRATOS_SEGURO__L__p_4_g_1_1'] == 1, 2,
np.where(base_dados['DETALHES_CONTRATOS_SEGURO__L__p_4_g_1_1'] == 2, 0,
 0)))
base_dados['DETALHES_CONTRATOS_SEGURO__L__p_4_g_1'] = np.where(base_dados['DETALHES_CONTRATOS_SEGURO__L__p_4_g_1_2'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_SEGURO__L__p_4_g_1_2'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_SEGURO__L__p_4_g_1_2'] == 2, 2,
 0)))
         
         
         
         
         
         
                
base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L'] = np.log(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO'])
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L'] == 0, -1, base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L'])
base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L'] = base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L'].fillna(-2)
base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__p_7'] = np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L'] <= 6.754604099487962, 0.0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L'] > 6.754604099487962, base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L'] <= 7.261225091971921), 1.0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L'] > 7.261225091971921, base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L'] <= 7.715123603632105), 2.0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L'] > 7.715123603632105, base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L'] <= 8.065579427282092), 3.0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L'] > 8.065579427282092, base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L'] <= 8.586719254064848), 4.0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L'] > 8.586719254064848, base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L'] <= 9.293393927111468), 5.0,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L'] > 9.293393927111468, 6.0,
 0)))))))
base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__p_7_g_1_1'] = np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__p_7'] == 0, 2,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__p_7'] == 1, 2,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__p_7'] == 2, 0,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__p_7'] == 3, 2,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__p_7'] == 4, 2,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__p_7'] == 5, 1,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__p_7'] == 6, 1,
 0)))))))
base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__p_7_g_1_2'] = np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__p_7_g_1_1'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__p_7_g_1_1'] == 1, 2,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__p_7_g_1_1'] == 2, 0,
 0)))
base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__p_7_g_1'] = np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__p_7_g_1_2'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__p_7_g_1_2'] == 2, 1,
 0))

                                                                     
                                                                     
                                                                     
                                                                     
                                                                     
base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__C'] = np.cos(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO'])
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__C'] == 0, -1, base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__C'])
base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__C'] = base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__C'].fillna(-2)
base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__C__p_13'] = np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__C'] >= -0.9999999959109308, base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__C'] <= 0.2149286330434905), 6.0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__C'] > 0.2149286330434905, base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__C'] <= 0.42423360271436256), 7.0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__C'] > 0.42423360271436256, base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__C'] <= 0.6129532197121991), 8.0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__C'] > 0.6129532197121991, base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__C'] <= 0.8240923484261706), 9.0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__C'] > 0.8240923484261706, base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__C'] <= 0.9189854930712886), 10.0,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__C'] > 0.9189854930712886, 11.0,
 -2))))))
base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__C__p_13_g_1_1'] = np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__C__p_13'] == -2.0, 0,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__C__p_13'] == 6.0, 1,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__C__p_13'] == 7.0, 0,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__C__p_13'] == 8.0, 0,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__C__p_13'] == 9.0, 1,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__C__p_13'] == 10.0, 0,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__C__p_13'] == 11.0, 1,
 0)))))))
base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__C__p_13_g_1_2'] = np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__C__p_13_g_1_1'] == 0, 1,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__C__p_13_g_1_1'] == 1, 0,
 0))
base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__C__p_13_g_1'] = np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__C__p_13_g_1_2'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__C__p_13_g_1_2'] == 1, 1,
 0))
                                                                      
                                                                      
                                                                      
                                                                      
                                                                      
                                                                      
base_dados['DETALHES_DIVIDAS_PRAZO__pe_13'] = np.where(base_dados['DETALHES_DIVIDAS_PRAZO'] <= 3.0, 0.0,
np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_PRAZO'] > 3.0, base_dados['DETALHES_DIVIDAS_PRAZO'] <= 7.0), 1.0,
np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_PRAZO'] > 7.0, base_dados['DETALHES_DIVIDAS_PRAZO'] <= 11.0), 2.0,
np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_PRAZO'] > 11.0, base_dados['DETALHES_DIVIDAS_PRAZO'] <= 15.0), 3.0,
np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_PRAZO'] > 15.0, base_dados['DETALHES_DIVIDAS_PRAZO'] <= 19.0), 4.0,
np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_PRAZO'] > 19.0, base_dados['DETALHES_DIVIDAS_PRAZO'] <= 22.0), 5.0,
np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_PRAZO'] > 22.0, base_dados['DETALHES_DIVIDAS_PRAZO'] <= 26.0), 6.0,
np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_PRAZO'] > 26.0, base_dados['DETALHES_DIVIDAS_PRAZO'] <= 30.0), 7.0,
np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_PRAZO'] > 30.0, base_dados['DETALHES_DIVIDAS_PRAZO'] <= 34.0), 8.0,
np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_PRAZO'] > 34.0, base_dados['DETALHES_DIVIDAS_PRAZO'] <= 38.0), 9.0,
np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_PRAZO'] > 38.0, base_dados['DETALHES_DIVIDAS_PRAZO'] <= 42.0), 10.0,
np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_PRAZO'] > 42.0, base_dados['DETALHES_DIVIDAS_PRAZO'] <= 45.0), 11.0,
np.where(base_dados['DETALHES_DIVIDAS_PRAZO'] > 45.0, 12.0,
 -2)))))))))))))
base_dados['DETALHES_DIVIDAS_PRAZO__pe_13_g_1_1'] = np.where(base_dados['DETALHES_DIVIDAS_PRAZO__pe_13'] == -2.0, 2,
np.where(base_dados['DETALHES_DIVIDAS_PRAZO__pe_13'] == 0.0, 0,
np.where(base_dados['DETALHES_DIVIDAS_PRAZO__pe_13'] == 1.0, 0,
np.where(base_dados['DETALHES_DIVIDAS_PRAZO__pe_13'] == 2.0, 1,
np.where(base_dados['DETALHES_DIVIDAS_PRAZO__pe_13'] == 3.0, 2,
np.where(base_dados['DETALHES_DIVIDAS_PRAZO__pe_13'] == 4.0, 2,
np.where(base_dados['DETALHES_DIVIDAS_PRAZO__pe_13'] == 5.0, 2,
np.where(base_dados['DETALHES_DIVIDAS_PRAZO__pe_13'] == 6.0, 2,
np.where(base_dados['DETALHES_DIVIDAS_PRAZO__pe_13'] == 7.0, 1,
np.where(base_dados['DETALHES_DIVIDAS_PRAZO__pe_13'] == 8.0, 0,
np.where(base_dados['DETALHES_DIVIDAS_PRAZO__pe_13'] == 9.0, 2,
np.where(base_dados['DETALHES_DIVIDAS_PRAZO__pe_13'] == 10.0, 2,
np.where(base_dados['DETALHES_DIVIDAS_PRAZO__pe_13'] == 11.0, 2,
np.where(base_dados['DETALHES_DIVIDAS_PRAZO__pe_13'] == 12.0, 2,
 0))))))))))))))
base_dados['DETALHES_DIVIDAS_PRAZO__pe_13_g_1_2'] = np.where(base_dados['DETALHES_DIVIDAS_PRAZO__pe_13_g_1_1'] == 0, 1,
np.where(base_dados['DETALHES_DIVIDAS_PRAZO__pe_13_g_1_1'] == 1, 2,
np.where(base_dados['DETALHES_DIVIDAS_PRAZO__pe_13_g_1_1'] == 2, 0,
 0)))
base_dados['DETALHES_DIVIDAS_PRAZO__pe_13_g_1'] = np.where(base_dados['DETALHES_DIVIDAS_PRAZO__pe_13_g_1_2'] == 0, 0,
np.where(base_dados['DETALHES_DIVIDAS_PRAZO__pe_13_g_1_2'] == 1, 1,
np.where(base_dados['DETALHES_DIVIDAS_PRAZO__pe_13_g_1_2'] == 2, 2,
 0)))
         
         
         
         
         
base_dados['DETALHES_DIVIDAS_PRAZO__C'] = np.cos(base_dados['DETALHES_DIVIDAS_PRAZO'])
np.where(base_dados['DETALHES_DIVIDAS_PRAZO__C'] == 0, -1, base_dados['DETALHES_DIVIDAS_PRAZO__C'])
base_dados['DETALHES_DIVIDAS_PRAZO__C'] = base_dados['DETALHES_DIVIDAS_PRAZO__C'].fillna(-2)
base_dados['DETALHES_DIVIDAS_PRAZO__C__p_3'] = np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_PRAZO__C'] >= -0.9999608263946371, base_dados['DETALHES_DIVIDAS_PRAZO__C'] <= 0.5403023058681398), 1.0,
np.where(base_dados['DETALHES_DIVIDAS_PRAZO__C'] > 0.5403023058681398, 2.0,
 -2))
base_dados['DETALHES_DIVIDAS_PRAZO__C__p_3_g_1_1'] = np.where(base_dados['DETALHES_DIVIDAS_PRAZO__C__p_3'] == -2.0, 0,
np.where(base_dados['DETALHES_DIVIDAS_PRAZO__C__p_3'] == 1.0, 0,
np.where(base_dados['DETALHES_DIVIDAS_PRAZO__C__p_3'] == 2.0, 1,
 0)))
base_dados['DETALHES_DIVIDAS_PRAZO__C__p_3_g_1_2'] = np.where(base_dados['DETALHES_DIVIDAS_PRAZO__C__p_3_g_1_1'] == 0, 1,
np.where(base_dados['DETALHES_DIVIDAS_PRAZO__C__p_3_g_1_1'] == 1, 0,
 0))
base_dados['DETALHES_DIVIDAS_PRAZO__C__p_3_g_1'] = np.where(base_dados['DETALHES_DIVIDAS_PRAZO__C__p_3_g_1_2'] == 0, 0,
np.where(base_dados['DETALHES_DIVIDAS_PRAZO__C__p_3_g_1_2'] == 1, 1,
 0))


# COMMAND ----------

# MAGIC %md
# MAGIC ### Parte 2 de 2

# COMMAND ----------


base_dados['VALOR_DIVIDA__L__p_20_g_1_c1_10_1'] = np.where(np.bitwise_and(base_dados['VALOR_DIVIDA__pe_10_g_1'] == 0, base_dados['VALOR_DIVIDA__L__p_20_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['VALOR_DIVIDA__pe_10_g_1'] == 0, base_dados['VALOR_DIVIDA__L__p_20_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['VALOR_DIVIDA__pe_10_g_1'] == 0, base_dados['VALOR_DIVIDA__L__p_20_g_1'] == 2), 2,
np.where(np.bitwise_and(base_dados['VALOR_DIVIDA__pe_10_g_1'] == 1, base_dados['VALOR_DIVIDA__L__p_20_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['VALOR_DIVIDA__pe_10_g_1'] == 1, base_dados['VALOR_DIVIDA__L__p_20_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['VALOR_DIVIDA__pe_10_g_1'] == 1, base_dados['VALOR_DIVIDA__L__p_20_g_1'] == 2), 3,
 0))))))
base_dados['VALOR_DIVIDA__L__p_20_g_1_c1_10_2'] = np.where(base_dados['VALOR_DIVIDA__L__p_20_g_1_c1_10_1'] == 0, 0,
np.where(base_dados['VALOR_DIVIDA__L__p_20_g_1_c1_10_1'] == 1, 1,
np.where(base_dados['VALOR_DIVIDA__L__p_20_g_1_c1_10_1'] == 2, 2,
np.where(base_dados['VALOR_DIVIDA__L__p_20_g_1_c1_10_1'] == 3, 3,
 0))))
base_dados['VALOR_DIVIDA__L__p_20_g_1_c1_10'] = np.where(base_dados['VALOR_DIVIDA__L__p_20_g_1_c1_10_2'] == 0, 0,
np.where(base_dados['VALOR_DIVIDA__L__p_20_g_1_c1_10_2'] == 1, 1,
np.where(base_dados['VALOR_DIVIDA__L__p_20_g_1_c1_10_2'] == 2, 2,
np.where(base_dados['VALOR_DIVIDA__L__p_20_g_1_c1_10_2'] == 3, 3,
 0))))
         
         
         
         
         
         
base_dados['NOME_PRODUTO__C__p_10_g_1_c1_26_1'] = np.where(np.bitwise_and(base_dados['NOME_PRODUTO__pe_3_g_1'] == 0, base_dados['NOME_PRODUTO__C__p_10_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['NOME_PRODUTO__pe_3_g_1'] == 0, base_dados['NOME_PRODUTO__C__p_10_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['NOME_PRODUTO__pe_3_g_1'] == 0, base_dados['NOME_PRODUTO__C__p_10_g_1'] == 2), 2,
np.where(np.bitwise_and(base_dados['NOME_PRODUTO__pe_3_g_1'] == 0, base_dados['NOME_PRODUTO__C__p_10_g_1'] == 3), 5,
np.where(np.bitwise_and(base_dados['NOME_PRODUTO__pe_3_g_1'] == 1, base_dados['NOME_PRODUTO__C__p_10_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['NOME_PRODUTO__pe_3_g_1'] == 1, base_dados['NOME_PRODUTO__C__p_10_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['NOME_PRODUTO__pe_3_g_1'] == 1, base_dados['NOME_PRODUTO__C__p_10_g_1'] == 2), 3,
np.where(np.bitwise_and(base_dados['NOME_PRODUTO__pe_3_g_1'] == 1, base_dados['NOME_PRODUTO__C__p_10_g_1'] == 3), 4,
np.where(np.bitwise_and(base_dados['NOME_PRODUTO__pe_3_g_1'] == 2, base_dados['NOME_PRODUTO__C__p_10_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['NOME_PRODUTO__pe_3_g_1'] == 2, base_dados['NOME_PRODUTO__C__p_10_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['NOME_PRODUTO__pe_3_g_1'] == 2, base_dados['NOME_PRODUTO__C__p_10_g_1'] == 2), 4,
np.where(np.bitwise_and(base_dados['NOME_PRODUTO__pe_3_g_1'] == 2, base_dados['NOME_PRODUTO__C__p_10_g_1'] == 3), 5,
 0))))))))))))
base_dados['NOME_PRODUTO__C__p_10_g_1_c1_26_2'] = np.where(base_dados['NOME_PRODUTO__C__p_10_g_1_c1_26_1'] == 0, 1,
np.where(base_dados['NOME_PRODUTO__C__p_10_g_1_c1_26_1'] == 1, 2,
np.where(base_dados['NOME_PRODUTO__C__p_10_g_1_c1_26_1'] == 2, 0,
np.where(base_dados['NOME_PRODUTO__C__p_10_g_1_c1_26_1'] == 3, 3,
np.where(base_dados['NOME_PRODUTO__C__p_10_g_1_c1_26_1'] == 4, 5,
np.where(base_dados['NOME_PRODUTO__C__p_10_g_1_c1_26_1'] == 5, 4,
 0))))))
base_dados['NOME_PRODUTO__C__p_10_g_1_c1_26'] = np.where(base_dados['NOME_PRODUTO__C__p_10_g_1_c1_26_2'] == 0, 0,
np.where(base_dados['NOME_PRODUTO__C__p_10_g_1_c1_26_2'] == 1, 1,
np.where(base_dados['NOME_PRODUTO__C__p_10_g_1_c1_26_2'] == 2, 2,
np.where(base_dados['NOME_PRODUTO__C__p_10_g_1_c1_26_2'] == 3, 3,
np.where(base_dados['NOME_PRODUTO__C__p_10_g_1_c1_26_2'] == 4, 4,
np.where(base_dados['NOME_PRODUTO__C__p_10_g_1_c1_26_2'] == 5, 5,
 0))))))
         
         
         
         
         
        
base_dados['IDADE_PESSOA__p_6_g_1_c1_6_1'] = np.where(np.bitwise_and(base_dados['IDADE_PESSOA__p_6_g_1'] == 0, base_dados['IDADE_PESSOA__pe_5_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['IDADE_PESSOA__p_6_g_1'] == 0, base_dados['IDADE_PESSOA__pe_5_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['IDADE_PESSOA__p_6_g_1'] == 1, base_dados['IDADE_PESSOA__pe_5_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['IDADE_PESSOA__p_6_g_1'] == 1, base_dados['IDADE_PESSOA__pe_5_g_1'] == 1), 2,
 0))))
base_dados['IDADE_PESSOA__p_6_g_1_c1_6_2'] = np.where(base_dados['IDADE_PESSOA__p_6_g_1_c1_6_1'] == 0, 0,
np.where(base_dados['IDADE_PESSOA__p_6_g_1_c1_6_1'] == 1, 1,
np.where(base_dados['IDADE_PESSOA__p_6_g_1_c1_6_1'] == 2, 2,
0)))
base_dados['IDADE_PESSOA__p_6_g_1_c1_6'] = np.where(base_dados['IDADE_PESSOA__p_6_g_1_c1_6_2'] == 0, 0,
np.where(base_dados['IDADE_PESSOA__p_6_g_1_c1_6_2'] == 1, 1,
np.where(base_dados['IDADE_PESSOA__p_6_g_1_c1_6_2'] == 2, 2,
 0)))
         
         
         
         
         
    
base_dados['DOCUMENTO_PESSOA__pe_20_g_1_c1_15_1'] = np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__p_5_g_1'] == 0, base_dados['DOCUMENTO_PESSOA__pe_20_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__p_5_g_1'] == 0, base_dados['DOCUMENTO_PESSOA__pe_20_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__p_5_g_1'] == 1, base_dados['DOCUMENTO_PESSOA__pe_20_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__p_5_g_1'] == 1, base_dados['DOCUMENTO_PESSOA__pe_20_g_1'] == 1), 3,
np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__p_5_g_1'] == 2, base_dados['DOCUMENTO_PESSOA__pe_20_g_1'] == 0), 2,
np.where(np.bitwise_and(base_dados['DOCUMENTO_PESSOA__p_5_g_1'] == 2, base_dados['DOCUMENTO_PESSOA__pe_20_g_1'] == 1), 3,
0))))))
base_dados['DOCUMENTO_PESSOA__pe_20_g_1_c1_15_2'] = np.where(base_dados['DOCUMENTO_PESSOA__pe_20_g_1_c1_15_1'] == 0, 0,
np.where(base_dados['DOCUMENTO_PESSOA__pe_20_g_1_c1_15_1'] == 1, 1,
np.where(base_dados['DOCUMENTO_PESSOA__pe_20_g_1_c1_15_1'] == 2, 2,
np.where(base_dados['DOCUMENTO_PESSOA__pe_20_g_1_c1_15_1'] == 3, 3,
0))))
base_dados['DOCUMENTO_PESSOA__pe_20_g_1_c1_15'] = np.where(base_dados['DOCUMENTO_PESSOA__pe_20_g_1_c1_15_2'] == 0, 0,
np.where(base_dados['DOCUMENTO_PESSOA__pe_20_g_1_c1_15_2'] == 1, 1,
np.where(base_dados['DOCUMENTO_PESSOA__pe_20_g_1_c1_15_2'] == 2, 2,
np.where(base_dados['DOCUMENTO_PESSOA__pe_20_g_1_c1_15_2'] == 3, 3,
0))))
         
         
         
         
         
        
base_dados['DETALHES_CONTRATOS_SEGURO__L__p_4_g_1_c1_10_1'] = np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_SEGURO__pe_10_g_1'] == 0, base_dados['DETALHES_CONTRATOS_SEGURO__L__p_4_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_SEGURO__pe_10_g_1'] == 0, base_dados['DETALHES_CONTRATOS_SEGURO__L__p_4_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_SEGURO__pe_10_g_1'] == 0, base_dados['DETALHES_CONTRATOS_SEGURO__L__p_4_g_1'] == 2), 2,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_SEGURO__pe_10_g_1'] == 1, base_dados['DETALHES_CONTRATOS_SEGURO__L__p_4_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_SEGURO__pe_10_g_1'] == 1, base_dados['DETALHES_CONTRATOS_SEGURO__L__p_4_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_SEGURO__pe_10_g_1'] == 1, base_dados['DETALHES_CONTRATOS_SEGURO__L__p_4_g_1'] == 2), 2,
 0))))))
base_dados['DETALHES_CONTRATOS_SEGURO__L__p_4_g_1_c1_10_2'] = np.where(base_dados['DETALHES_CONTRATOS_SEGURO__L__p_4_g_1_c1_10_1'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_SEGURO__L__p_4_g_1_c1_10_1'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_SEGURO__L__p_4_g_1_c1_10_1'] == 2, 2,
 0)))
base_dados['DETALHES_CONTRATOS_SEGURO__L__p_4_g_1_c1_10'] = np.where(base_dados['DETALHES_CONTRATOS_SEGURO__L__p_4_g_1_c1_10_2'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_SEGURO__L__p_4_g_1_c1_10_2'] == 1, 1,
np.where(base_dados['DETALHES_CONTRATOS_SEGURO__L__p_4_g_1_c1_10_2'] == 2, 2,
 0)))
         
         
         
         
               
base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__C__p_13_g_1_c1_3_1'] = np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__p_7_g_1'] == 0, base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__C__p_13_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__p_7_g_1'] == 0, base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__C__p_13_g_1'] == 1), 0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__p_7_g_1'] == 1, base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__C__p_13_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__L__p_7_g_1'] == 1, base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__C__p_13_g_1'] == 1), 1,
 0))))
base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__C__p_13_g_1_c1_3_2'] = np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__C__p_13_g_1_c1_3_1'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__C__p_13_g_1_c1_3_1'] == 1, 1,
 0))
base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__C__p_13_g_1_c1_3'] = np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__C__p_13_g_1_c1_3_2'] == 0, 0,
np.where(base_dados['DETALHES_CONTRATOS_SALDO_ABERTO__C__p_13_g_1_c1_3_2'] == 1, 1,
 0))
                                                                           
                                                                           
                                                                           
                                                                           
                                                                           
                                                                           
base_dados['DETALHES_DIVIDAS_PRAZO__C__p_3_g_1_c1_16_1'] = np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_PRAZO__pe_13_g_1'] == 0, base_dados['DETALHES_DIVIDAS_PRAZO__C__p_3_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_PRAZO__pe_13_g_1'] == 0, base_dados['DETALHES_DIVIDAS_PRAZO__C__p_3_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_PRAZO__pe_13_g_1'] == 1, base_dados['DETALHES_DIVIDAS_PRAZO__C__p_3_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_PRAZO__pe_13_g_1'] == 1, base_dados['DETALHES_DIVIDAS_PRAZO__C__p_3_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_PRAZO__pe_13_g_1'] == 2, base_dados['DETALHES_DIVIDAS_PRAZO__C__p_3_g_1'] == 1), 3,
np.where(np.bitwise_and(base_dados['DETALHES_DIVIDAS_PRAZO__pe_13_g_1'] == 2, base_dados['DETALHES_DIVIDAS_PRAZO__C__p_3_g_1'] == 2), 3,
 0))))))
base_dados['DETALHES_DIVIDAS_PRAZO__C__p_3_g_1_c1_16_2'] = np.where(base_dados['DETALHES_DIVIDAS_PRAZO__C__p_3_g_1_c1_16_1'] == 0, 1,
np.where(base_dados['DETALHES_DIVIDAS_PRAZO__C__p_3_g_1_c1_16_1'] == 1, 0,
np.where(base_dados['DETALHES_DIVIDAS_PRAZO__C__p_3_g_1_c1_16_1'] == 2, 2,
np.where(base_dados['DETALHES_DIVIDAS_PRAZO__C__p_3_g_1_c1_16_1'] == 3, 3,
0))))
base_dados['DETALHES_DIVIDAS_PRAZO__C__p_3_g_1_c1_16'] = np.where(base_dados['DETALHES_DIVIDAS_PRAZO__C__p_3_g_1_c1_16_2'] == 0, 0,
np.where(base_dados['DETALHES_DIVIDAS_PRAZO__C__p_3_g_1_c1_16_2'] == 1, 1,
np.where(base_dados['DETALHES_DIVIDAS_PRAZO__C__p_3_g_1_c1_16_2'] == 2, 2,
np.where(base_dados['DETALHES_DIVIDAS_PRAZO__C__p_3_g_1_c1_16_2'] == 3, 3,
 0))))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Mantendo apenas as variáveis do modelo

# COMMAND ----------

import pickle
modelo=pickle.load(open(caminho_base_pickle + 'model_fit_portocred.sav', 'rb'))

base_teste_c0 = base_dados[[chave,'TIPO_EMAIL_gh38','TIPO_ENDERECO_gh38','DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh38','DETALHES_CONTRATOS_SALDO_ABERTO__C__p_13_g_1_c1_3','DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh38','DETALHES_DIVIDAS_PRAZO__C__p_3_g_1_c1_16','NOME_PRODUTO__C__p_10_g_1_c1_26','DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh38','STATUS_SPC_gh38','DETALHES_CLIENTES_NEGATIVADO_SERASA_gh38','VALOR_DIVIDA__L__p_20_g_1_c1_10','IDADE_PESSOA__p_6_g_1_c1_6','DETALHES_CLIENTES_NEGATIVADO_SPC_gh38','DETALHES_CONTRATOS_SEGURO__L__p_4_g_1_c1_10','DOCUMENTO_PESSOA__pe_20_g_1_c1_15']]

var_fin_c0=['TIPO_EMAIL_gh38','TIPO_ENDERECO_gh38','DETALHES_CONTRATOS_FAMILIA_PRODUTO_gh38','DETALHES_CONTRATOS_SALDO_ABERTO__C__p_13_g_1_c1_3','DETALHES_CLIENTES_CONTA_REF_BANCARIA3_gh38','DETALHES_DIVIDAS_PRAZO__C__p_3_g_1_c1_16','NOME_PRODUTO__C__p_10_g_1_c1_26','DETALHES_CLIENTES_BANCO_REF_BANCARIA_gh38','STATUS_SPC_gh38','DETALHES_CLIENTES_NEGATIVADO_SERASA_gh38','VALOR_DIVIDA__L__p_20_g_1_c1_10','IDADE_PESSOA__p_6_g_1_c1_6','DETALHES_CLIENTES_NEGATIVADO_SPC_gh38','DETALHES_CONTRATOS_SEGURO__L__p_4_g_1_c1_10','DOCUMENTO_PESSOA__pe_20_g_1_c1_15']

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

x_teste2['GH'] = np.where(x_teste2['P_1'] <= 0.032497018, 0.0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.032497018, x_teste2['P_1'] <= 0.092815721), 1.0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.092815721, x_teste2['P_1'] <= 0.194385909), 2.0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.194385909, x_teste2['P_1'] <= 0.356899326), 3.0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.356899326, x_teste2['P_1'] <= 0.785430512), 4.0,
    np.where(x_teste2['P_1'] > 0.785430512,5,2))))))

x_teste2

# COMMAND ----------

try:
  dbutils.fs.rm(outputpath, True)
except:
  pass
dbutils.fs.mkdirs(outputpath)

x_teste2.to_csv(open(os.path.join(outputpath_dbfs, 'pre_output:' + nm_base.replace('-','') + '_' + dt_max + '.csv'),'wb'))
os.path.join(outputpath_dbfs, 'pre_output:' + nm_base.replace('-','') + '_' + dt_max + '.csv')