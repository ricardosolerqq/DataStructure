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

mount_blob_storage_key(dbutils,blob_account_source_ml,blob_account_source_ml,'/mnt/ml-prd')

trustedpath = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/credz/trusted'
picklepath = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/credz/pickle_model'

spark.conf.set('spark.sql.sources.partitionOverwriteMode', 'dynamic')

# COMMAND ----------

# DBTITLE 1,cria widgets
dbutils.widgets.combobox('MODELO_ESCOLHIDO', max(os.listdir(picklepath)), os.listdir(picklepath))
dbutils.widgets.combobox('ARQUIVO_ESCOLHIDO', max(os.listdir(trustedpath)), os.listdir(trustedpath))

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

import numpy as np
import pandas as pd
import datetime
from sklearn.pipeline import Pipeline
from sklearn import set_config # visualização do pipe
set_config(display='diagram') # configuração da visualização do pipe
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
import pickle
df = pd.read_parquet(file)
from sklearn import *

# COMMAND ----------

# DBTITLE 1,set index
df = df.drop(columns=['DOCUMENTO_PESSOA', 'ID_DIVIDA', 'id_deals'])
df.set_index('DOCUMENTO:ID_DIVIDA')

# COMMAND ----------

#pre-processamento
imputer_preprocess = SimpleImputer(missing_values = np.NaN, strategy='mean')
standard_scaler_preprocess = StandardScaler()
oneHot_preprocess = OneHotEncoder(sparse=False, handle_unknown='ignore')

pipe_numerical = Pipeline([('SimpleImputer', imputer_preprocess), ('StandardScaler', standard_scaler_preprocess)])
pipe_categorical = Pipeline([('OneHot',oneHot_preprocess)])

preprocess = ColumnTransformer([('PipeNumeric', pipe_numerical, numeric_features), 
                                ('PipeCategorical', pipe_categorical, categorical_features)])
preprocess

# COMMAND ----------

# DBTITLE 1,preprocess df para x
for col in categorical_features:
  try:
    pipe_categorical.fit_transform(df[[col]])
  except Exception as e:
    print (col, e)

# COMMAND ----------

# DBTITLE 1,usando pickle para carregar modelo
modelo_escolhido = pickle.load(open(model,'rb'))