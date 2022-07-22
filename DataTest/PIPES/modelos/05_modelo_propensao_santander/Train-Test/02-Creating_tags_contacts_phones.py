# Databricks notebook source
# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

# MAGIC %run "/ml-prd/propensao-deal/santander/training/1-func-common"

# COMMAND ----------

# DBTITLE 1,Ajustando os diretórios
import pandas as pd
import sys 
import os
import numpy as np
import difflib
from itertools import chain
from datetime import datetime,timedelta

###Carregando os bloobs##
mount_blob_storage_key(dbutils,'qqdatastoragemain','etlsantander','/mnt/etlsantander')
dir_files='/dbfs/mnt/etlsantander/etl_santander_locs_campaign'
##Selecionando os Parquets###
dir_parquets=os.listdir(dir_files)
dir_parquets=list(filter(lambda x: '2021-12-14' in x, dir_parquets))
#dir_parquets
###
arqs=os.listdir(dir_files+'/'+dir_parquets[0])
arqs=list(filter(lambda x: '.parquet' in x, arqs))
arqs

# COMMAND ----------

# DBTITLE 1,Carregando tabelas e variáveis auxiliares
data_ref=datetime.strptime('2021-12-14 03:00:00',"%Y-%m-%d %H:%M:%S")
col_person = getPyMongoCollection('col_person')

dir_IDs_datas='/dbfs/mnt/etlsantander/Base_UFPEL/IDs_base/Dados=2021-12-14/'
dir_phones='/dbfs/mnt/etlsantander/Base_UFPEL/Tags_contatos_phones'
if not os.path.exists(dir_phones+'/Dados='+data_ref.strftime("%Y-%m-%d")):
  os.makedirs(dir_phones+'/Dados='+data_ref.strftime("%Y-%m-%d"))

dir_phones_datas=dir_phones+'/Dados='+data_ref.strftime("%Y-%m-%d")

# COMMAND ----------

# DBTITLE 1,Montando a base formatada para ser salva no blob
for j in range(0,len(arqs)):
  print('\nLendo o arquivo '+str(j+1)+' de '+str(len(arqs))+'...')
  tab_training=pd.read_parquet(dir_files+'/'+dir_parquets[0]+'/'+arqs[j])
  
  ####FORMATANDO AS VARIÁVEIS DA BASE#####
  print('\nFormatando as variáveis da base...')
  columns_contacts=['Numero_Documento','Telefone1','Telefone2','Telefone3','Telefone4','Telefone5','Telefone6','Telefone7','Telefone8','Telefone9','Telefone10']
  tab_training=tab_training[columns_contacts]
  tab_training['Numero_Documento']=tab_training['Numero_Documento'].str.replace("[.,-]","").str.strip().astype(str)
  for i in range(1,11):
    tab_training[columns_contacts[i]]=tab_training[columns_contacts[i]].str.replace("[(,),-]","").str.replace(" ","").astype(str)
    
  ####SELECIONANDO OS TELEFONES VÁLIDOS####
  print('\nSelecionando os telefones válidos...')
  tab_training=tab_training.drop_duplicates(subset='Numero_Documento').reset_index(drop=True)
  tab_training['Telefones']=tab_training[columns_contacts[1:11]].values.tolist()
  tab_phone=tab_training[['Numero_Documento','Telefones']].copy().explode(column=['Telefones']).reset_index(drop=True)
  tab_phone=tab_phone[tab_phone['Telefones']!=""]
  tab_phone['Telefones']='55'+tab_phone['Telefones']
  tab_phone=tab_phone.groupby('Numero_Documento',as_index=False).agg(Telefones=('Telefones',";".join))
  aux=tab_phone['Telefones'].str.split(pat=";",n=10,expand=True)
  if aux.shape[1]<10:
    for i in range(0,10):
      if i not in list(aux.columns):
        aux[i]=None
  tab_phone=pd.concat([tab_phone['Numero_Documento'],aux],axis=1)
  tab_phone.columns=['document']+columns_contacts[1:11]
  tab_training=tab_training[['Numero_Documento']]
  tab_training=tab_training.rename(columns={'Numero_Documento' : 'document'},inplace=False)
  tab_training=pd.merge(tab_training,tab_phone,how='left',on='document')
  del aux,tab_phone
  
  ####SELECIONANDO AS TAGS DOS TELEFONES####
  print('\nSelecionando as tags dos telefones...')
  aux=tab_training.dropna(subset=['Telefone1'])['document'].unique()
  partes=list(range(0,len(aux),10000))
  if(partes[-1]!=len(aux)):
    partes.append(len(aux))

  info_tels=[]  
  for i in range(0,len(partes)-1):
    print("\nExecutando a parte "+str(i+1)+" de "+str(len(partes)-1)+"...")  
    lista_CPFs=list(aux[partes[i]:(partes[i+1])])
    query_tels=[
      {
          "$match" : {
              "document" : {"$in" : lista_CPFs}
          }
      },
      {
          "$project" : {
              "_id" : 0,
              "Document" : "$document",
              "phones" : {
                          "$map" : {
                              "input" : {"$ifNull" : ["$info.phones",[]]},
                              "as" : "i",
                              "in" : {
                                      "phone" : {"$ifNull" : ["$$i.phone",""] } ,
                                      "tags" : {"$ifNull" : ["$$i.tags",["sem_tags"]]}
                                  }
                          }
                      }
          }
      }
    ]		
    info_tels.append(list(col_person.aggregate(pipeline=query_tels,allowDiskUse=True)))
  info_tels=pd.DataFrame(list(chain.from_iterable(info_tels))).explode(column=['phones'])
  aux_info_tels=pd.json_normalize(info_tels['phones'])
  info_tels=info_tels.reset_index(drop=True)
  info_tels=pd.concat([info_tels['Document'],aux_info_tels],axis=1)
  info_tels=info_tels.dropna(subset=['phone']).reset_index(drop=True)
  del aux_info_tels
  tags_values=['skip:hot','skip:alto','skip:medio','skip:baixo','skip:nhot','sem_tags']
  info_tels['tags']=info_tels['tags'].map(lambda x : list(set(map(str,x)).intersection(tags_values))[0] if  len(list(set(map(str,x)).intersection(tags_values)))>0 else 'sem_tags')
  
  ####MONTANDO AS DUMMIES DOS TELEFONES###
  print('\nMontando as dummies dos telefones...')
  for i in range(1,11):
    print("\nSelecionando as dummies da coluna "+list(tab_training.columns)[i]+"...")  
    tab_training['chave']=tab_training['document']+':'+tab_training[list(tab_training.columns)[i]]
    info_tels['chave']=info_tels['Document']+':'+info_tels['phone']
    tab_training=pd.merge(tab_training,info_tels,how='left',on=['chave']).reset_index(drop=True)
    tab_training.loc[tab_training['tags'].isna(),'tags']=pd.Series(['sem_tags'] * tab_training['tags'].isna().sum()).values
    tab_training['tags']=list(tab_training.columns)[i]+'_'+tab_training['tags']
    tab_training=pd.concat([tab_training,pd.get_dummies(tab_training['tags'],columns=list(map(lambda x:  list(tab_training.columns)[i]+'_'+x, tags_values))).T.reindex(list(map(lambda x:  list(tab_training.columns)[i]+'_'+x, tags_values))).T.fillna(0).astype(bool)],axis=1)
    tab_training=tab_training.drop(labels=['chave','Document','phone','tags'],axis=1)

  tab_training.drop(labels=tab_training.columns[1:11],axis=1,inplace=True)
  del lista_CPFs,info_tels,partes,query_tels
  
  ####ENCAIXANDO OS IDS DOS DOCUMENTOS###
  #print('\nEncaixando os IDs dos documentos...')
  #IDs=pd.read_csv(dir_IDs_datas+'IDs_documents_std_P'+str(j).zfill(2)+'.csv', sep=';')
  #IDs['document']=IDs['document'].astype(str).str.pad(width=11,fillchar='0')
  #IDs=IDs.drop(labels='chave',axis=1).drop_duplicates(subset=['document']).reset_index(drop=True)
  #tab_save=pd.merge(IDs,tab_training,how='right',on=['document'])
  #tab_save=tab_save.drop(labels='document',axis=1)
  #del IDs,tab_training
  
  ####SALVANDO####
  print('\nSalvando...')
  tab_training.to_csv(dir_phones_datas+'/Base_UFPEL_QQ_Model_std_phones_P'+str(j).zfill(2)+'.csv',index=False,sep=";")
  #del tab_save

# COMMAND ----------

df2 = spark.read.options(delimiter =';', header = True).csv('/mnt/etlsantander/Base_UFPEL/Tags_contatos_phones/Dados=2021-12-14')
display(df2)