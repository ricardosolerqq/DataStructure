# Databricks notebook source
# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

mount_blob_storage_key(dbutils,"qqdatastoragemain","ml-prd",'/mnt/ml-prd')

# COMMAND ----------

# DBTITLE 1,Carregando os pacotes
import pandas as pd
import sys 
import os
from datetime import datetime,timedelta
#import pymongo
from itertools import chain
import difflib
import numpy as np
#from sklearn.compose import ColumnTransformer
#from sklearn.pipeline import Pipeline
#from sklearn.impute import SimpleImputer
#from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import pickle
import ssl
import statsmodels.api as sm
import seaborn as sns

# COMMAND ----------

# DBTITLE 1,Carregando arquivo
dir_database=dir_save="/dbfs/mnt/ml-prd/ml-data/propensaodeal/recovery/processed/model_V2 (Elisvaldo)/formated_base"
x_teste2=pd.read_csv(dir_database+'/recovery_model_v2_formated_database_'+(datetime.today()-timedelta(hours=3)).strftime("%d%m%Y")+'.csv',sep=';',usecols=['Chave','CPF__pe_4_g_1_c1_15', 'VlDividaAtualizado__L__p_6_g_1_c1_14', 'Class_Carteira_gh38', 'Aging__L__p_25_g_1_c1_14', 'IdContatoSIR__L__p_8_g_1_c1_14'])
z_teste=x_teste2[['Chave']]
x_teste2=x_teste2.drop('Chave',axis=1)

# COMMAND ----------

# DBTITLE 1,Carregando o modelo
###Ajustando o diretório do modelo##
dir_orig="/dbfs/mnt/ml-prd/ml-data/propensaodeal/recovery/pickle_models/models_V2 (Elisvaldo)"

modelo=pickle.load(open(dir_orig+'/model_fit_V2_recovery.sav', 'rb'))

# COMMAND ----------

# DBTITLE 1,Classificando as bases
probabilidades = modelo.predict_proba(x_teste2)
data_prob = pd.DataFrame({'P_1': probabilidades[:, 1]})

z_teste1 = z_teste.reset_index(drop=True)
#x_teste1 = x_teste.reset_index(drop=True)
data_prob1 = data_prob.reset_index(drop=True)


x_teste2 = pd.concat([z_teste1, data_prob1], axis=1)
del z_teste1,data_prob1

# COMMAND ----------

# DBTITLE 1,Montando o grupo de escores
x_teste2['P_1_R'] = np.sqrt(x_teste2['P_1'])
np.where(x_teste2['P_1_R'] == 0, -1, x_teste2['P_1_R'])
x_teste2['P_1_R'] = x_teste2['P_1_R'].fillna(-2)

x_teste2['P_1_pe_20_g_1'] = np.where(x_teste2['P_1'] <= 0.039482547, 0.0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.039482547, x_teste2['P_1'] <= 0.078855436), 1.0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.078855436, x_teste2['P_1'] <= 0.23682533), 2.0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.23682533, x_teste2['P_1'] <= 0.354488676), 3.0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.354488676, x_teste2['P_1'] <= 0.434850909), 4.0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.434850909, x_teste2['P_1'] <= 0.517960421), 5.0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.517960421, x_teste2['P_1'] <= 0.591608), 6.0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.591608, x_teste2['P_1'] <= 0.792068002), 7.0,
    np.where(x_teste2['P_1'] > 0.792068002, 8.0,0)))))))))

x_teste2['P_1_R_p_5_g_1'] = np.where(x_teste2['P_1_R'] <= 0.210253883, 0.0,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.210253883, x_teste2['P_1_R'] <= 0.282139416), 1.0,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.282139416, x_teste2['P_1_R'] <= 0.362194322), 2.0,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.362194322, x_teste2['P_1_R'] <= 0.485356497), 3.0,
    np.where(x_teste2['P_1_R'] > 0.485356497, 4.0,0)))))

x_teste2['GH'] = np.where(np.bitwise_and(x_teste2['P_1_pe_20_g_1'] == 0, x_teste2['P_1_R_p_5_g_1'] == 0), 1,
    np.where(np.bitwise_and(x_teste2['P_1_pe_20_g_1'] == 1, x_teste2['P_1_R_p_5_g_1'] == 0), 0,
    np.where(np.bitwise_and(x_teste2['P_1_pe_20_g_1'] == 1, x_teste2['P_1_R_p_5_g_1'] == 1), 2,
    np.where(np.bitwise_and(x_teste2['P_1_pe_20_g_1'] == 2, x_teste2['P_1_R_p_5_g_1'] == 1), 3,
    np.where(np.bitwise_and(x_teste2['P_1_pe_20_g_1'] == 2, x_teste2['P_1_R_p_5_g_1'] == 2), 3,
    np.where(np.bitwise_and(x_teste2['P_1_pe_20_g_1'] == 2, x_teste2['P_1_R_p_5_g_1'] == 3), 3,
    np.where(np.bitwise_and(x_teste2['P_1_pe_20_g_1'] == 2, x_teste2['P_1_R_p_5_g_1'] == 4), 5,
    np.where(np.bitwise_and(x_teste2['P_1_pe_20_g_1'] == 3, x_teste2['P_1_R_p_5_g_1'] == 2), 5,
    np.where(np.bitwise_and(x_teste2['P_1_pe_20_g_1'] == 3, x_teste2['P_1_R_p_5_g_1'] == 3), 4,
    np.where(np.bitwise_and(x_teste2['P_1_pe_20_g_1'] == 3, x_teste2['P_1_R_p_5_g_1'] == 4), 4,
    np.where(np.bitwise_and(x_teste2['P_1_pe_20_g_1'] == 4, x_teste2['P_1_R_p_5_g_1'] == 3), 5,
    np.where(np.bitwise_and(x_teste2['P_1_pe_20_g_1'] == 4, x_teste2['P_1_R_p_5_g_1'] == 4), 5,
    np.where(np.bitwise_and(x_teste2['P_1_pe_20_g_1'] == 5, x_teste2['P_1_R_p_5_g_1'] == 4), 6,
    np.where(np.bitwise_and(x_teste2['P_1_pe_20_g_1'] == 6, x_teste2['P_1_R_p_5_g_1'] == 4), 6,
    np.where(np.bitwise_and(x_teste2['P_1_pe_20_g_1'] == 7, x_teste2['P_1_R_p_5_g_1'] == 4), 7,
    np.where(np.bitwise_and(x_teste2['P_1_pe_20_g_1'] == 8, x_teste2['P_1_R_p_5_g_1'] == 4), 8,
    np.where(np.bitwise_and(x_teste2['P_1_pe_20_g_1'] == 9, x_teste2['P_1_R_p_5_g_1'] == 4), 8,
             0)))))))))))))))))             

del x_teste2['P_1_R']
del x_teste2['P_1_pe_20_g_1']
del x_teste2['P_1_R_p_5_g_1']

# COMMAND ----------

# DBTITLE 1,Selecionando o CPF da chave
x_teste2['CPF']=x_teste2['Chave'].str.split(pat=":",n=2,expand=True)[0]

# COMMAND ----------

# DBTITLE 1,Agrupando
x_teste2=x_teste2.groupby('CPF',as_index=False).agg(Pred_prob = ('P_1','max'),
                                                                  GH=('GH','max'),
                                                                  ScoreAvg=('P_1',np.mean))

# COMMAND ----------

pd.DataFrame({'Escores' : x_teste2['GH'].value_counts()}).sort_index()

# COMMAND ----------

# DBTITLE 1,Colocando no formato para produção
data_hj=datetime.today()-timedelta(hours=3)
##Layout das colunas##
layout_columns=['Document','Score','ScoreValue','ScoreAvg','Provider','Date','CreatedAt']

##Formatando as tabelas##
x_teste2['Pred_prob']=x_teste2['Pred_prob']*100
x_teste2['ScoreAvg']=x_teste2['ScoreAvg']*100
x_teste2['Provider']='qq_recovery_propensity_v2'
x_teste2['Date']=data_hj.strftime("%Y-%m-%d")
x_teste2['CreatedAt']=x_teste2['Date']
x_teste2=x_teste2.rename(columns={'CPF' : 'Document', 'Pred_prob' : 'ScoreValue','GH' : 'Score'})
#x_teste2=x_teste2.drop('Pred',axis=1)
x_teste2=x_teste2[layout_columns]

# COMMAND ----------

# DBTITLE 1,Salvando backup e colocando em produção
dir_prod='/dbfs/mnt/ml-prd/ml-data/propensaodeal/santander/output'
x_teste2.to_csv(dir_prod+'/recovery_model_v2_to_production_'+(datetime.today()-timedelta(hours=3)).strftime("%d%m%Y")+'.csv',index=False,sep=";")

dir_backup="/dbfs/mnt/ml-prd/ml-data/propensaodeal/recovery/processed/model_V2 (Elisvaldo)/backup_score_base"
x_teste2.to_csv(dir_backup+'/recovery_model_v2_to_production_'+(datetime.today()-timedelta(hours=3)).strftime("%d%m%Y")+'.csv',index=False,sep=";")