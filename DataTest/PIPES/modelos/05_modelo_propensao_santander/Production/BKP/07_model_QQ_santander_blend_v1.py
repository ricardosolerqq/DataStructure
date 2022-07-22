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
chave = 'Document'

caminho_base  = '/mnt/ml-prd/ml-data/propensaodeal/santander/trusted/'
caminho_base_dbfs = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/santander/trusted/'

list_base = os.listdir(caminho_base_dbfs)

#Nome da Base de Dados
N_Base = max(list_base)


#Separador
separador_ = ";"

#Decimal
decimal_ = "."


caminho_trusted = '/mnt/ml-prd/ml-data/propensaodeal/santander/trusted'
caminho_trusted_dbfs = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/santander/trusted'

pickle_path = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/santander/pickle_model/'

outputpath = 'mnt/ml-prd/ml-data/propensaodeal/santander/output/'
outputpath_dbfs = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/santander/output/'

N_Base

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importação da Base de Dados

# COMMAND ----------

base_dados = pd.read_csv(caminho_base_dbfs+N_Base, sep=separador_, decimal=decimal_)
base_dados = base_dados[[chave,'P_1_Acordo','P_1_Pgto']]

print("shape da Base de Dados:",base_dados.shape)

base_dados.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Mantendo apenas as variáveis do modelo

# COMMAND ----------

import pickle
modelo=pickle.load(open(pickle_path + 'model_fit_santander_blend_r_forest.sav', 'rb'))

base_teste_c0 = base_dados[[chave,'P_1_Acordo','P_1_Pgto']]

var_fin_c0=['P_1_Acordo','P_1_Pgto']

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


x_teste2['P_1_p_40_g_1'] = np.where(x_teste2['P_1'] <= 0.008571429, 0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.008571429, x_teste2['P_1'] <= 0.058666667), 1,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.058666667, x_teste2['P_1'] <= 0.208571429), 2,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.208571429, x_teste2['P_1'] <= 0.627309524), 3,4))))

x_teste2['P_1_p_17_g_1'] = np.where(x_teste2['P_1'] <= 0.008571429, 0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.008571429, x_teste2['P_1'] <= 0.029166667), 1,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.029166667, x_teste2['P_1'] <= 0.098333333), 2,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.098333333, x_teste2['P_1'] <= 0.517833333), 3,4))))

x_teste2['GH'] = np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 0, x_teste2['P_1_p_17_g_1'] == 0), 0,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 0, x_teste2['P_1_p_17_g_1'] == 1), 0,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 0, x_teste2['P_1_p_17_g_1'] == 2), 0,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 0, x_teste2['P_1_p_17_g_1'] == 3), 1,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 0, x_teste2['P_1_p_17_g_1'] == 4), 1,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 1, x_teste2['P_1_p_17_g_1'] == 0), 1,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 1, x_teste2['P_1_p_17_g_1'] == 1), 1,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 1, x_teste2['P_1_p_17_g_1'] == 2), 2,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 1, x_teste2['P_1_p_17_g_1'] == 3), 2,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 1, x_teste2['P_1_p_17_g_1'] == 4), 2,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 2, x_teste2['P_1_p_17_g_1'] == 0), 2,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 2, x_teste2['P_1_p_17_g_1'] == 1), 2,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 2, x_teste2['P_1_p_17_g_1'] == 2), 2,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 2, x_teste2['P_1_p_17_g_1'] == 3), 3,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 2, x_teste2['P_1_p_17_g_1'] == 4), 3,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 3, x_teste2['P_1_p_17_g_1'] == 0), 3,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 3, x_teste2['P_1_p_17_g_1'] == 1), 3,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 3, x_teste2['P_1_p_17_g_1'] == 2), 4,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 3, x_teste2['P_1_p_17_g_1'] == 3), 4,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 3, x_teste2['P_1_p_17_g_1'] == 4), 4,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 4, x_teste2['P_1_p_17_g_1'] == 0), 4,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 4, x_teste2['P_1_p_17_g_1'] == 1), 4,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 4, x_teste2['P_1_p_17_g_1'] == 2), 4,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 4, x_teste2['P_1_p_17_g_1'] == 3), 5,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 4, x_teste2['P_1_p_17_g_1'] == 4), 5,
             2)))))))))))))))))))))))))

del x_teste2['P_1_p_40_g_1']
del x_teste2['P_1_p_17_g_1']

x_teste2


# COMMAND ----------

x_teste2.groupby(['GH'])['P_1'].count()

# COMMAND ----------

createdAt = datetime.today().date()

output = spark.createDataFrame(x_teste2)
output = output.withColumn('Document', F.lpad(F.col("Document"),11,'0'))
output = output.groupBy(F.col('Document')).agg(F.max(F.col('GH')), F.max(F.col('P_1')), F.avg(F.col('P_1')))
output = output.withColumn('Provider', F.lit('qq_santander_propensity_blend_v1'))
output = output.withColumn('Date', F.lit(createdAt))
output = output.withColumn('CreatedAt', F.lit(createdAt))
output = changeColumnNames(output, ['Document','Score','ScoreValue','ScoreAvg','Provider','Date','CreatedAt'])

# COMMAND ----------

display(output)

# COMMAND ----------

output.coalesce(1).write.mode('overwrite').options(header='True', delimiter=';').csv(outputpath+'/tmp')

for files in dbutils.fs.ls(outputpath+'/tmp'):
  if files.name.split('.')[-1] == 'csv':
    dbutils.fs.cp(files.path, outputpath+'/santander_model_blend_to_production_'+str(createdAt)+'.csv')
    dbutils.fs.rm(outputpath+'/tmp', recurse=True)
    dbutils.fs.rm(caminho_base+N_Base, recurse=True)