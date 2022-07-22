# Databricks notebook source
# DBTITLE 1,Carregando as funções padrão
# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

# DBTITLE 1,montando qq-data-studies no cluster
mount_blob_storage_key(dbutils,'qqdatastoragemain','qq-data-studies','/mnt/qq-data-studies')

# COMMAND ----------

# DBTITLE 1,Selecionando os IDs ativos
import pandas as pd
import sys 
import os
from datetime import datetime,timedelta
#import pymongo
from itertools import chain
import difflib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import pickle
import ssl

###Carregando a base###
col_person = getPyMongoCollection('col_person')

###Ajustando as datas##
data_inicio=datetime.strptime('01/01/2005 03:00:00',"%d/%m/%Y %H:%M:%S")
data_fim=datetime.strptime('01/01/2045 03:00:00',"%d/%m/%Y %H:%M:%S")

###Extraindo os IDs ativos###
query=col_person.find({'debts':{"$elemMatch":
                                  {
                                    'creditor':'recovery',
                                    'status': 'active',
                                    'updatedAt':{'$gte': data_inicio, '$lte': data_fim}
                                    }},'documentType' : 'cpf'},
                                   {
                                     '_id' : 1
                                   })
Ids=pd.DataFrame(list(query))['_id'].to_list()
del data_inicio,data_fim
Ids
#Tab_model=pd.DataFrame(list(col_person.aggregate(pipeline=query_Ids,allowDiskUse=True,batchSize=100000)))
#del col_person

# COMMAND ----------

# DBTITLE 1,Selecionando as variáveis da tabela iterativamente
partes=list(range(0,len(Ids),10000))
if(partes[-1]!=len(Ids)):
  partes.append(len(Ids))
  
Tab_model=[]
colunas_explode=['contrato','VlSOP','Data de referência','Produto','Portifolio','Data_Mora']
for i in range(0,len(partes)-1):
  print("\nExecutando a parte "+str(i+1)+" de "+str(len(partes)-1)+"...")  
  IDs_partes=Ids[partes[i]:(partes[i+1])]
  Query_model=[
    {
        "$match" : {
            "_id" : {"$in" : IDs_partes}
        }
    },
    {
        "$project" : {
            "document" : "$document",
            "debts" : {
                        "$filter" : {
                            "input" : "$debts",
                            "cond" : {"$and" : [
                                            {"$eq" : ["$$this.creditor","recovery"]},
                                            {"$eq" : ["$$this.status","active"]}
                                        ]
                                }
                        }
                    }
        }
    },
    {
        "$project" : {
            "_id" : 0,
            "documento" : "$document",
            "contrato" : "$debts.contract",
            "VlSOP" : "$debts.originalAmount",
            "Data_Mora" : "$debts.dueDate",
            "Data de referência" : "$debts.updatedAt",
            "Produto" : "$debts.product",
            "Portifolio" : "$debts.portfolio"     
        }
    }
  ]
  Tab_model.append(pd.DataFrame(list(col_person.aggregate(pipeline=Query_model,allowDiskUse=True))).explode(column=colunas_explode))
Tab_model=pd.concat(Tab_model)  

# COMMAND ----------

# DBTITLE 1,Ajustando as variáveis Portifolio, Produto e Carteira
#colunas_explode=['contrato','VlSOP','Data_Mora','Data de referência','Produto','Portifolio']
#Tab_model=Tab_model.explode(colunas_explode,ignore_index=True)
Tab_model['Produto']=Tab_model['Produto'].str.encode('ascii',errors='ignore').str.decode('utf-8',errors='ignore').str.lower()
###Criando a variável carteira##
Tab_model['Carteira']=Tab_model['Portifolio'].str.split(":",n=1,expand=True).iloc[:,1]
##Arrumando a variável Portifolio##
Tab_model['Portifolio']=Tab_model['Portifolio'].str.split(":",n=1,expand=True).iloc[:,0]

# COMMAND ----------

# DBTITLE 1,Ajustando as tabelas de classificação
###Carregando as tabelas de calssificação##
dir_orig='/dbfs/mnt/qq-data-studies/Recovery_model'
dir_class=dir_orig+'/Tabela das classificações'
Tabs=sorted(os.listdir(dir_class))
columns=['Carteira','Produto','Portifolio']
for i in range(0,len(Tabs)):
  print("Montando a classificação da variável "+columns[i]+"...\n")
  aux_tabs=pd.read_csv(dir_class+'/'+Tabs[i],sep=';')
  aux_format=Tab_model[columns[i]].unique()
  Tab_result=[difflib.get_close_matches(str(word), aux_tabs.iloc[:,0],n=1,cutoff=0.75) for word in aux_format]
  Tab_result=[["Outros"] if x ==[] else x for x in Tab_result]
  Tab_result=list(chain.from_iterable(Tab_result))
  Tab_result=pd.DataFrame(data=list(zip(Tab_result,aux_format)),columns=[list(aux_tabs.columns)[0],'real_name'])
  Tab_result=pd.merge(Tab_result,aux_tabs,how='left',on=list(aux_tabs.columns)[0])
  Tab_result=Tab_result.drop(list(aux_tabs.columns)[0],axis=1)
  Tab_result=Tab_result.rename(columns={'real_name' : columns[i]})
  Tab_model=pd.merge(Tab_model,Tab_result,how='left',on=[list(Tab_result.columns)[0]])
  
Tab_model=Tab_model.drop(columns,axis=1)

# COMMAND ----------

# DBTITLE 1,Variável de acionamento
data_hj=datetime.strptime(str((datetime.today()-timedelta(hours=3)).strftime("%d/%m/%Y"))+' 03:00:00',"%d/%m/%Y %H:%M:%S")
data_fim=data_hj+timedelta(days=1)
data_inic=data_hj-timedelta(days=5)
Query_acions_val=[
    {
        "$match" : {
            "sentAt" : {"$gte" : data_inic,"$lt" :data_fim },
            "status" : {"$nin" : ["not_answered","not_received"]}
        }
    },
    {
        "$addFields" : {
            "creditor" : {"$ifNull" : ["$creditor","unknown"]}
        }
    },
    {
        "$match" : {
            "creditor" : {"$in" : ["recovery","unknown"]}
        }
    },
    {
        "$group" : {
            "_id" : "$document"
        }
    },
    {
        "$project" : {
            "_id" : 0,
            "documento": "$_id"
        }
    }
]
col_interaction = getPyMongoCollection('col_interaction')
Tab_acion=pd.DataFrame(list(col_interaction.aggregate(pipeline=Query_acions_val,allowDiskUse=True,batchSize=100000)))
Tab_model['Acion_5']=Tab_model['documento'].isin(Tab_acion['documento'])
del col_interaction,Tab_acion

# COMMAND ----------

# DBTITLE 1,Arrumando as variáveis numéricas
##Datas##
Tab_model['Data_Mora']=Tab_model['Data_Mora'].dt.strftime('%Y-%m-%d 03:00:00')
Tab_model['Data de referência']=Tab_model['Data de referência']-timedelta(hours=3)
Tab_model['Data de referência']=Tab_model['Data de referência'].dt.strftime('%Y-%m-%d 03:00:00')
##Criando a variável Aging##

Tab_model['Aging']=list(map(lambda a: (data_hj-pd.to_datetime(a)).days,Tab_model['Data_Mora']))
##Valores##
Tab_model['VlSOP']=round(Tab_model['VlSOP'].astype(float),2)

# COMMAND ----------

# DBTITLE 1,Formatando as variáveis de tempo do modelo
def transform_time(X):
    """Convert Datetime objects to seconds for numerical/quantitative parsing"""
    df = pd.DataFrame(X)
    return df.apply(lambda x: pd.to_datetime(x).apply(lambda x: x.timestamp()))

###Retirando os NaN, no caso é quem fez acordo nesse meio tempo de extração das base###  
Tab_model.dropna(subset=['Data de referência'],inplace=True)

###Separando as variáveis chaves##
Base_recov=Tab_model.drop(columns=['documento','contrato'],axis=1)
Tab_final_class=Tab_model.loc[:,['documento','contrato']]

del Tab_model

##Formatando as variáveis tempo em milesegundos##
Base_recov['Data de referência']=transform_time(Base_recov['Data de referência'])
Base_recov['Data_Mora']=transform_time(Base_recov['Data_Mora'])

# COMMAND ----------

# DBTITLE 1,Pré-processamento
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
Base_recov_pretransform=preprocessor.fit_transform(Base_recov)

# COMMAND ----------

# DBTITLE 1,Carregando o modelo
modelo_rec=pickle.load(open(dir_orig+'/model_fit_recovery_complete.sav', 'rb'))

# COMMAND ----------

# DBTITLE 1,Classificando a base
y_pred_prob=modelo_rec.predict_proba(Base_recov_pretransform)[:,1]
y_pred= modelo_rec.predict(Base_recov_pretransform)

# COMMAND ----------

# DBTITLE 1,Montando a tabela de classificação
Tab_final_class['Pred_prob']=y_pred_prob
Tab_final_class['Pred']=y_pred
Tab_final_class=Tab_final_class.groupby('documento',as_index=False).agg(Pred_prob = ('Pred_prob','max'),
                                                                       Pred=('Pred','max'),
                                                                       ScoreAvg=('Pred_prob',np.mean))

# COMMAND ----------

pd.DataFrame({'Predições' : Tab_final_class['Pred'].value_counts()})

# COMMAND ----------

# DBTITLE 1,Criando os escores de classificação
Threshold_tables=pd.read_csv(dir_orig+'/Threshold tables.csv',sep=";")
Threshold_tables=list(Threshold_tables.iloc[0,[0,3]])

cortes=np.linspace(start=0, stop=Threshold_tables[0], num=6)
cortes=list(np.concatenate([cortes,np.linspace(start=Threshold_tables[0], stop=Threshold_tables[1], num=5),np.array([1])],axis=0))
cortes=list(np.unique(cortes))
#labels=list(range(0,10,1))
Tab_final_class['Score']=pd.cut(Tab_final_class['Pred_prob'],bins=cortes,labels=list(range(0,10,1)))

# COMMAND ----------

pd.DataFrame({'Escores' : Tab_final_class['Score'].value_counts()}).sort_index()

# COMMAND ----------

# DBTITLE 1,Colocando no formato de produção
##Layout das colunas##
layout_columns=['Document','Score','ScoreValue','ScoreAvg','Provider','Date','CreatedAt']

##Formatando as tabelas##
Tab_final_class['Pred_prob']=Tab_final_class['Pred_prob']*100
Tab_final_class['ScoreAvg']=Tab_final_class['ScoreAvg']*100
Tab_final_class['Provider']='qq_recovery_propensity'
Tab_final_class['Date']=str(datetime.today().strftime("%Y-%m-%d"))
Tab_final_class['CreatedAt']=Tab_final_class['Date']
Tab_final_class=Tab_final_class.rename(columns={'documento' : 'Document', 'Pred_prob' : 'ScoreValue'})
Tab_final_class=Tab_final_class.drop('Pred',axis=1)
Tab_final_class=Tab_final_class[layout_columns]

# COMMAND ----------

# DBTITLE 1,Ajustando bloob de produção dos modelos
mount_blob_storage_key(dbutils,'qqdatastoragemain','ml-prd','/mnt/ml-prd')

# COMMAND ----------

# DBTITLE 1,Salvando e colocando em produção
dir_save='/dbfs/mnt/ml-prd/ml-data/propensaodeal/santander/output'
Tab_final_class.to_csv(dir_save+'/Recovery_model_to_production_'+(datetime.today()-timedelta(hours=3)).strftime("%d%m%Y")+'.csv',index=False,sep=";")

Tab_final_class.to_csv(dir_orig+'/Recovery_model_to_production_'+(datetime.today()-timedelta(hours=3)).strftime("%d%m%Y")+'.csv',index=False,sep=";")