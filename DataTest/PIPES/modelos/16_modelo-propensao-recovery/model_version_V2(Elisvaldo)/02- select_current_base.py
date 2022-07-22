# Databricks notebook source
# DBTITLE 1,Carregando as funções padrão
# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

# DBTITLE 1,montando qq-data-studies no cluster
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

# DBTITLE 1,Selecionando os IDs ativos
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
  
base_dados=[]
colunas_explode=['contrato','VlSOP','VlDividaAtualizado','Produto','Portifolio','Data_Mora']
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
            "IdContatoSIR" : {"$ifNull" : ["$info.additionalInfo.recovery:IdContatoSIR","0"]},
            "debts" : {
            		"$map" : {
            			"input" : {
			                        "$filter" : {
			                            "input" : "$debts",
			                            "cond" : {"$and" : [
			                                            {"$eq" : ["$$this.creditor","recovery"]},
			                                            {"$eq" : ["$$this.status","active"]}
			                                        ]
			                                }
			                        }
			                    },
			             "as" : "i",
			             "in" : {
                           "contract" : "$$i.contract",
			             	"originalAmount" : "$$i.originalAmount",
			             	"dueDate" : "$$i.dueDate",
			             	"product" : "$$i.product",
			             	"portfolio" : "$$i.portfolio",
			             	"VlDividaAtualizado" : {"$ifNull" : ["$$i.additionalInfo.VlDividaAtualizado","0.00"]}
			             }       
            		}
            	}
        }
    },
    {
        "$project" : {
            "_id" : 0,
            "CPF" : "$document",
            "contrato" : "$debts.contract",
            "IdContatoSIR" : "$IdContatoSIR",
            "VlSOP" : "$debts.originalAmount",
            "VlDividaAtualizado" : "$debts.VlDividaAtualizado",
            "Data_Mora" : "$debts.dueDate",
            "Produto" : "$debts.product",
            "Portifolio" : "$debts.portfolio"     
        }
    }
  ]
  base_dados.append(pd.DataFrame(list(col_person.aggregate(pipeline=Query_model,allowDiskUse=True))).explode(column=colunas_explode))
base_dados=pd.concat(base_dados)
del Ids

# COMMAND ----------

# DBTITLE 1,Dropando as variáveis sem informações de 'IdContatoSIR' e 'VlDividaAtualizado'
base_not_class=base_dados[(base_dados['VlDividaAtualizado']=='0.00') | (base_dados['IdContatoSIR']=="0")].copy()
base_dados=base_dados[(base_dados['VlDividaAtualizado']!='0.00') & (base_dados['IdContatoSIR']!="0")]
base_dados['Chave']=base_dados['CPF']+':'+base_dados['contrato']

# COMMAND ----------

# DBTITLE 1,Salvando os que não serão classificados
dir_not_class="/dbfs/mnt/ml-prd/ml-data/propensaodeal/recovery/not_classified/model_V2 (Elisvaldo)/"
base_not_class.to_csv(dir_not_class+'not_classified_debts_'+(datetime.today()-timedelta(hours=3)).strftime("%d%m%Y")+'.csv',index=False,sep=";")
del dir_not_class,base_not_class

# COMMAND ----------

# DBTITLE 1,Ajustando a formatação das variáveis numéricas
base_dados[['VlSOP','VlDividaAtualizado']]=base_dados[['VlSOP','VlDividaAtualizado']].astype(float)
base_dados[['CPF','IdContatoSIR']]=base_dados[['CPF','IdContatoSIR']].astype(int)

# COMMAND ----------

# DBTITLE 1,Ajustando as variáveis Portifolio, Produto e Carteira
####Separando produtos de portifólio###
base_dados['Produto']=base_dados['Produto'].str.encode('ascii',errors='ignore').str.decode('utf-8',errors='ignore').str.lower()
###Criando a variável carteira##
base_dados['Carteira']=base_dados['Portifolio'].str.split(":",n=1,expand=True).iloc[:,1]
##Arrumando a variável Portifolio##
base_dados['Portifolio']=base_dados['Portifolio'].str.split(":",n=1,expand=True).iloc[:,0]

# COMMAND ----------

# DBTITLE 1,Ajustando as tabelas de classificação
###Carregando as tabelas de calssificação##
dir_class="/dbfs/mnt/ml-prd/ml-data/propensaodeal/recovery/classification_tables"
Tabs=sorted(os.listdir(dir_class))
columns=['Carteira','Produto','Portifolio']
for i in range(0,len(Tabs)):
  print("Montando a classificação da variável "+columns[i]+"...\n")
  aux_tabs=pd.read_csv(dir_class+'/'+Tabs[i],sep=';')
  aux_format=base_dados[columns[i]].unique()
  Tab_result=[difflib.get_close_matches(str(word), aux_tabs.iloc[:,0],n=1,cutoff=0.75) for word in aux_format]
  Tab_result=[["Outros"] if x ==[] else x for x in Tab_result]
  Tab_result=list(chain.from_iterable(Tab_result))
  Tab_result=pd.DataFrame(data=list(zip(Tab_result,aux_format)),columns=[list(aux_tabs.columns)[0],'real_name'])
  Tab_result=pd.merge(Tab_result,aux_tabs,how='left',on=list(aux_tabs.columns)[0])
  Tab_result=Tab_result.drop(list(aux_tabs.columns)[0],axis=1)
  Tab_result=Tab_result.rename(columns={'real_name' : columns[i]})
  base_dados=pd.merge(base_dados,Tab_result,how='left',on=[list(Tab_result.columns)[0]])
  
base_dados=base_dados.drop(columns,axis=1)

# COMMAND ----------

# DBTITLE 1,Calculando o aging
###Data de hoje##
data_hj=datetime.strptime(str((datetime.today()-timedelta(hours=3)).strftime("%d/%m/%Y"))+' 03:00:00',"%d/%m/%Y %H:%M:%S")
##Datas##
base_dados['Data_Mora']=pd.to_datetime(base_dados['Data_Mora'].dt.strftime('%Y-%m-%d 03:00:00'))

base_dados['Data_hoje']=data_hj
##Criando a variável Aging##
base_dados['Aging']=(base_dados['Data_hoje']-base_dados['Data_Mora']).dt.days
##Excluindo as colunas de datas##
base_dados=base_dados.drop(['Data_Mora','Data_hoje','VlSOP'],axis=1)

# COMMAND ----------

# DBTITLE 1,Salvando
dir_save="/dbfs/mnt/ml-prd/ml-data/propensaodeal/recovery/processed/model_V2 (Elisvaldo)/base_to_score"
base_dados.to_csv(dir_save+'/recovery_model_v2_to_score_'+(datetime.today()-timedelta(hours=3)).strftime("%d%m%Y")+'.csv',index=False,sep=";")
del base_dados,dir_save,data_hj,Tabs,Tab_result,aux_format,aux_tabs,dir_class,columns