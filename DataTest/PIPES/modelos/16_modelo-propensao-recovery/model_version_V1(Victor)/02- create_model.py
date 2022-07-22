# Databricks notebook source
# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

# DBTITLE 1,Carregando a base e ajustando os diretórios
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn import set_config
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve,precision_recall_fscore_support
from sklearn.model_selection import cross_val_score, KFold
import os
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from scipy.stats import ks_2samp
import pickle

def transform_time(X):
    """Convert Datetime objects to seconds for numerical/quantitative parsing"""
    df = pd.DataFrame(X)
    return df.apply(lambda x: pd.to_datetime(x).apply(lambda x: x.timestamp()))

file = '/dbfs/mnt/qq-data-studies/Recovery_model/Tabela completa formatada para execução do modelo.csv'
dir_orig='/dbfs/mnt/qq-data-studies/Recovery_model'
Base_recov = pd.read_csv(file,sep=";",index_col='Chave')
#Base_recov

# COMMAND ----------

Base_recov.loc[Base_recov['Deals_30']==True,:]

# COMMAND ----------

# DBTITLE 1,Formatando as variáveis do modelo
###Retirando variáveis descritivas e as que não podem ser extraídas na validação##
Base_recov=Base_recov.drop(columns=['IdContatoSIR','CPF','Nome_Cliente / Empresa','Numero_Contrato','VlDividaAtualizado','SubTipo Produto'])


###Colocando os valores de data no formato correto##
Base_recov['Data de referência']=pd.to_datetime(Base_recov['Data de referência'],format='%Y-%m-%d %H:%M:%S')
Base_recov['Data_Mora']=pd.to_datetime(Base_recov['Data_Mora'],format='%Y-%m-%d %H:%M:%S')

                                                      
Base_recov['Data de referência']=transform_time(Base_recov['Data de referência'])
Base_recov['Data_Mora']=transform_time(Base_recov['Data_Mora'])    

# COMMAND ----------

# DBTITLE 1,Pré-processamento das variáveis
TEST_SIZE = 0.2
RANDOM_STATE = 42
#N_SPLITS = 3

impute = SimpleImputer(strategy='mean')
scaler = MinMaxScaler()
ohe = OneHotEncoder(handle_unknown='ignore')

numeric_feat = ['VlSOP','Aging','Data de referência','Data_Mora']
pipe_numeric_transf = Pipeline([('SimpleImputer', impute),
                               ('MinMaxScaler', scaler)])

categ_feat = ['Class_Carteira','Class_Produto','Class_Portfolio']
pipe_categ_feat = Pipeline([('OneHotEncoder', ohe)])

preprocessor = ColumnTransformer([('Pipe_Numeric', pipe_numeric_transf, numeric_feat),
                                 ('Pipe_Categoric', pipe_categ_feat, categ_feat)],
                                 remainder='passthrough')

# COMMAND ----------

# DBTITLE 1,Executando o modelo campeão
###Criando o modelo de regressão logística##
model_logistic = LogisticRegression(random_state=RANDOM_STATE)

# COMMAND ----------

# DBTITLE 1,Separando as variáveis de Treino e Teste além do desbalanceamento da base
##Separando input e output
X = Base_recov.drop(columns='Deals_30')
y = Base_recov.loc[:, 'Deals_30']

###pré-processando todas as variáveis###
x_pretransform=preprocessor.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(x_pretransform, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
smt = SMOTE(random_state=RANDOM_STATE)
X_train_SMOTE, y_train_SMOTE = smt.fit_resample(X_train, y_train)

# COMMAND ----------

# DBTITLE 1,Ajustando e salvando o modelo
###Fit###
model_logistic.fit(X_train_SMOTE,y_train_SMOTE)
##Save###
pickle.dump(model_logistic,open(dir_orig+'/model_fit_recovery_complete.sav','wb'))