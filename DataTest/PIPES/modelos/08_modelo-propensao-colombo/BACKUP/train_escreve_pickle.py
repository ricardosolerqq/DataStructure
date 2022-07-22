# Databricks notebook source
# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

import os
import datetime

# COMMAND ----------

blob_account_source_ml = "qqdatastoragemain"
blob_container_source_ml = "ml-prd"

blob_account_source_lake = "saqueroquitar"
blob_container_source_lake = "trusted"

mount_blob_storage_key(dbutils,blob_account_source_ml,blob_container_source_ml,'/mnt/ml-prd')
mount_blob_storage_key(dbutils,blob_account_source_lake,blob_container_source_lake,'/mnt/ml-prd')

readpath_trusted = '/mnt/ml-prd/ml-data/propensaodeal/credz/trusted'
readpath_trusted_dbfs = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/credz/trusted'

list_readpath = os.listdir(readpath_trusted_dbfs)

# COMMAND ----------

dbutils.widgets.combobox('ARQUIVO ESCOLHIDO', max(list_readpath), list_readpath)
data_escolhida = dbutils.widgets.get('ARQUIVO ESCOLHIDO')
parquetfilepath = '/mnt/ml-prd/ml-data/propensaodeal/credz/trusted/variavel_resposta'+'/'+data_escolhida+'/'
outputpickle_dbfs = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/credz/pickle_model/'+data_escolhida+'/'
outputpickle = '/mnt/ml-prd/ml-data/propensaodeal/credz/pickle_model/'+data_escolhida+'/'
print(outputpickle)

# COMMAND ----------

file = parquetfilepath+"trustedFile_credz_FULL_variavel_resposta.PARQUET"
file

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
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve,precision_recall_fscore_support
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import imblearn # imbalanced learning library
from imblearn.over_sampling import SMOTE
import pickle



# COMMAND ----------

# DBTITLE 1,criando dataframe
df = pd.read_parquet('/dbfs/'+file, engine='pyarrow')
data_arquivo = datetime.date(2021,4,17)

# COMMAND ----------

# DBTITLE 1,definindo index
df = df.drop(columns=['DOCUMENTO_PESSOA', 'ID_DIVIDA', 'id_deals'])
df.set_index('DOCUMENTO:ID_DIVIDA')

# COMMAND ----------

numeric_features = ['IDADE_PESSOA',
 'VALOR_DIVIDA',
 'AGING',
 'ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA',
 'DETALHES_CLIENTES_SCORE_CARGA',
 'DETALHES_CLIENTES_VALOR_FATURA',
 'DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO',
 'DETALHES_CLIENTES_RENDA_CONSIDERADA',
 'DETALHES_CLIENTES_COLLECT_SCORE',
 'DETALHES_CLIENTES_LIMITE_APROVADO',
 'DETALHES_CONTRATOS_SALDO_ATUAL_ALDIV',
 'DETALHES_CONTRATOS_VALOR_PRINCIPAL',
 'DETALHES_DIVIDAS_VALOR_JUROS',
 'DETALHES_DIVIDAS_TAXA_SERVICO',
 'DETALHES_DIVIDAS_TAXA_ATRASO',
 'DETALHES_DIVIDAS_VALOR_JUROS_DIARIO',
 'DETALHES_DIVIDAS_TAXA_SEGURO']

categorical_features = ['ID_CEDENTE',
 'ID_CEDENTE',
 'NOME_PRODUTO',
 'DETALHES_CONTRATOS_BLOQUEIO2',
 'DETALHES_CONTRATOS_CLASSE',
 'DETALHES_CONTRATOS_CODIGO_LOGO',
 'DETALHES_CONTRATOS_HISTORICO_FPD',
 'DETALHES_CONTRATOS_BLOQUEIO2___DESC',
 'DETALHES_CONTRATOS_BLOQUEIO1___DESC',
 'DETALHES_CONTRATOS_CLASSE___DESC',
 'DETALHES_CONTRATOS_STATUS_ACORDO',
 'TIPO_EMAIL_0',
 'TIPO_EMAIL_1',
 'TIPO_EMAIL_2',
 'TIPO_EMAIL_3',
 'TIPO_TELEFONE_0',
 'TIPO_TELEFONE_1',
 'TIPO_TELEFONE_2',
 'TIPO_TELEFONE_3',
 'TIPO_TELEFONE_4',
 'TIPO_ENDERECO_0',
 'TIPO_ENDERECO_1',
 'TIPO_ENDERECO_2',
 'TIPO_ENDERECO_3']

# COMMAND ----------

imputer_preprocess = SimpleImputer(strategy='mean')
standard_scaler_preprocess = StandardScaler()
oneHot_preprocess = OneHotEncoder(sparse=False, handle_unknown='ignore')

pipe_numerical = Pipeline([('SimpleImputer', imputer_preprocess), ('StandardScaler', standard_scaler_preprocess)])
pipe_categorical = Pipeline([('OneHot',oneHot_preprocess)])

preprocess = ColumnTransformer([('PipeNumeric', pipe_numerical, numeric_features), 
                                ('PipeCategorical', pipe_categorical, categorical_features)])
preprocess

# COMMAND ----------

RANDOM_STATE = 42
x = df.drop(columns = 'VARIAVEL_RESPOSTA')
y = df.loc[:,'VARIAVEL_RESPOSTA', ].to_numpy()

x_pretransform = preprocess.fit_transform(x)

x_pretransform_train, x_pretransform_test, y_pretransform_train, y_pretransform_test = train_test_split(x_pretransform,y, test_size=0.2, random_state = RANDOM_STATE)

# COMMAND ----------

oversample = SMOTE()
x_oversample, y_oversample = oversample.fit_resample(x_pretransform_train, y_pretransform_train)

# COMMAND ----------

#AdaBoostClassifier
logistic_regression = LogisticRegression(random_state = RANDOM_STATE)
#pipe_adaboost_classifier = Pipeline([('preprocess', preprocess), ('adaBoots', adaboost_classifier)])

logistic_regression.fit(x_oversample, y_oversample)


# COMMAND ----------

dbutils.fs.mkdirs(outputpickle)

# COMMAND ----------

pickle.dump(logistic_regression, open(outputpickle_dbfs+'saved_model', 'wb'))