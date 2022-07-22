# Databricks notebook source
import datetime
date = datetime.datetime.today()
date = str(date.year)+'-'+str(date.month).zfill(2)+'-'+str(date.day).zfill(2)
#date = '2022-06-21'
date

# COMMAND ----------

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
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime,timedelta

###Carregando os bloobs##
mount_blob_storage_key(dbutils,'qqdatastoragemain','etlsantander','/mnt/etlsantander')
dir_files='/dbfs/mnt/etlsantander/etl_santander_locs_campaign'
##Selecionando os Parquets###
dir_parquets=os.listdir(dir_files)
dir_parquets=list(filter(lambda x: date in x, dir_parquets))  #'2021-11-13'
#dir_parquets
###
arqs=os.listdir(dir_files+'/'+dir_parquets[0])
arqs=list(filter(lambda x: '.parquet' in x, arqs))
arqs

# COMMAND ----------

# DBTITLE 1,Carregando as tabelas e variáveis auxiliares 
dir_prod='/dbfs/mnt/etlsantander/Base_UFPEL'
Tab_prod=pd.read_csv(dir_prod+'/Contagem dos produtos.csv',sep=',')
Tab_prod=Tab_prod['Produto'][0:12]
Tab_prod=Tab_prod.append(pd.Series("Outros")).reset_index(drop=True)
param_date_time = "{0} {1}".format(date, '03:00:00')
data_ref=datetime.strptime(param_date_time,"%Y-%m-%d %H:%M:%S") #'2021-11-13 03:00:00'
data_hj=datetime.strptime(str((datetime.today()-timedelta(hours=3)).strftime("%d/%m/%Y"))+' 03:00:00',"%d/%m/%Y %H:%M:%S")
col_person = getPyMongoCollection('col_person')

dir_debts='/dbfs/mnt/etlsantander/Base_UFPEL/Skips_divids'
if not os.path.exists(dir_debts+'/Dados='+data_ref.strftime("%Y-%m-%d")):
  os.makedirs(dir_debts+'/Dados='+data_ref.strftime("%Y-%m-%d"))
  
dir_debts_datas=dir_debts+'/Dados='+data_ref.strftime("%Y-%m-%d")

# COMMAND ----------

#Tab_vals1[Tab_vals1['Numero_Documento']== '00402495306']

# COMMAND ----------

#Tab_vals1['Indicador_Correntista'].str.replace("N","0").str.replace("S","1").astype(int).astype(bool)

# COMMAND ----------

# DBTITLE 1,Montando a base formatada para ser salva no blob
for j in range(0,len(arqs)):
  print('\nLendo o arquivo '+str(j+1)+' de '+str(len(arqs))+'...')
  tab_training=pd.read_parquet(dir_files+'/'+dir_parquets[0]+'/'+arqs[j])
  tab_training = tab_training.loc[tab_training['Endereco1']!='R R ANTONIO R ANTONIO PORTELA, S']
  
  ####FORMATANDO AS VARIÁVEIS DA BASE####
  print('\nFormatando as variáveis da base...')
  tab_training['Saldo_Devedor_Contrato']=tab_training['Saldo_Devedor_Contrato'].str.strip('+').str.replace("EMPRESTIMO EM FOLHA                                         ","0").str.replace(",",".").astype(float)
  tab_training['Taxa_Padrao']=tab_training['Taxa_Padrao'].str.strip('+').str.replace(",",".").astype(float)
  tab_training['Desconto_Padrao']=tab_training['Desconto_Padrao'].str.strip('+').str.replace(",",".").astype(float)
  tab_training['Percentual_Min_Entrada']=tab_training['Percentual_Min_Entrada'].str.strip('+').str.replace(",",".").astype(float)
  tab_training['Valor_Min_Parcela']=tab_training['Valor_Min_Parcela'].str.strip('+').str.replace(",",".").astype(float)
  tab_training['Numero_Documento']=tab_training['Numero_Documento'].str.replace("[.,-]","").str.strip().astype(str)
  tab_training['Dias_Atraso'].fillna(0, inplace=True)
  tab_training['Dias_Atraso']=tab_training['Dias_Atraso'].astype(int)
  tab_training['Qtde_Min_Parcelas'].fillna(0, inplace=True)
  tab_training['Qtde_Min_Parcelas']=tab_training['Qtde_Min_Parcelas'].astype(int)
  tab_training['Qtde_Max_Parcelas'].fillna(0, inplace=True)
  tab_training['Qtde_Max_Parcelas']=tab_training['Qtde_Max_Parcelas'].astype(int)
  tab_training['Qtde_Parcelas_Padrao'].fillna(0, inplace=True)
  tab_training['Qtde_Parcelas_Padrao']=tab_training['Qtde_Parcelas_Padrao'].astype(int)
  tab_training['Descricao_Produto']=tab_training['Descricao_Produto'].str.rstrip()
  tab_training['Contrato_Altair']=tab_training['Contrato_Altair'].str.slice(14,26)
  colunas_tab1=['Numero_Documento','Contrato_Altair','Indicador_Correntista','Cep1','Cep2','Cep3','Forma_Pgto_Padrao','Taxa_Padrao','Desconto_Padrao','Qtde_Min_Parcelas','Qtde_Max_Parcelas','Qtde_Parcelas_Padrao','Valor_Min_Parcela','Codigo_Politica','Saldo_Devedor_Contrato','Dias_Atraso','Descricao_Produto']
  Tab_vals1=tab_training[colunas_tab1]

  Tab_vals1.insert(0,'ID_Credor','std')
  Tab_vals1['Indicador_Correntista'].fillna('N', inplace=True)
  Tab_vals1['Indicador_Correntista']=Tab_vals1['Indicador_Correntista'].str.replace("N","0").str.replace("S","1").astype(int).astype(bool)
  del tab_training
  
  ####INCLUINDO AS DUMMIES DE PRODUTOS####
  print('\nIncluindo as dummies de produtos...')
  aux_format=Tab_vals1['Descricao_Produto'].unique()
  Tab_result=[difflib.get_close_matches(str(word), Tab_prod,n=1,cutoff=0.75) for word in aux_format]
  Tab_result=[["Outros"] if x ==[] else x for x in Tab_result]
  Tab_result=pd.DataFrame(data=list(zip(list(chain.from_iterable(Tab_result)),aux_format)),columns=['Dummie_produto','Descricao_Produto'])
  Tab_vals1=pd.merge(Tab_vals1,Tab_result,how='left',on='Descricao_Produto')
  Tab_vals1=pd.concat([Tab_vals1,pd.get_dummies(Tab_vals1['Dummie_produto'],columns=Tab_prod).T.reindex(Tab_prod).T.fillna(0).astype(bool)],axis=1)
  Tab_vals1=Tab_vals1.drop('Dummie_produto',axis=1)
  Tab_vals1=Tab_vals1.rename(columns={'Numero_Documento' : 'document'})
  
  ####SELECIONANDO AS DEMAIS INFORMAÇÕES####
  print("\nSelecionando as informações da plataforma QQ...")
  partes=list(range(0,Tab_vals1.shape[0],10000))
  if(partes[-1]!=Tab_vals1.shape[0]):
    partes.append(Tab_vals1.shape[0])

  Tab_rest=[]
  for i in range(0,len(partes)-1):
    print("\nExecutando a parte "+str(i+1)+" de "+str(len(partes)-1)+" da querys das informações...")  
    lista_CPFs=list(Tab_vals1['document'][partes[i]:(partes[i+1])])
    query=[
      {
          "$match" : {
              "document" : {"$in" : lista_CPFs}
          }
      },
      {
          "$project" : {
              "_id" : 1,
              "document" : 1,
              "genero" : {"$ifNull" : ["$info.gender",""]},
              "faixa_etaria" : {
                                  "$floor": {
                                          "$divide": [{
                                              "$subtract": [data_hj,{"$ifNull" : ["$info.birthDate",data_hj]}]
                                          }, 31540000000]
                                      }
                           },
             "debts" : {
                          "$filter" : {
                              "input" : {"$ifNull" : ["$debts",[]]},
                              "cond" : {"$eq" : ["$$this.creditor","santander"]}
                          }
                      }
          }
        }
      ]
    Tab_rest.append(list(col_person.aggregate(pipeline=query,allowDiskUse=True)))

  Tab_rest=pd.DataFrame(list(chain.from_iterable(Tab_rest)))
  del lista_CPFs,partes,query
  
  ####IDs, GÊNERO E FAIXA ETÁRIA####
  print('\nMontando IDs, gênero e faixa etária...')
  aux_IDs=Tab_rest[['_id','document','genero','faixa_etaria']].copy()
  aux_IDs['genero']= np.where((aux_IDs['genero']!='F') & (aux_IDs['genero']!='M'),"",aux_IDs['genero'])
  aux_IDs['faixa_etaria']=np.where((aux_IDs['faixa_etaria']>=120) | (aux_IDs['faixa_etaria']<0),0,aux_IDs['faixa_etaria'])
  Tab_vals1=pd.merge(Tab_vals1,aux_IDs,how='left',on='document')
  del aux_IDs
  Tab_vals1=Tab_vals1.rename(columns={'_id' : 'ID_DEVEDOR'},inplace=False)
  
  ####DEBTS###
  print('\nInformações da debts...')
  Tab_debts=Tab_rest[['document','debts']].copy().explode(column=['debts'])
  aux=pd.json_normalize(Tab_debts['debts'])
  Tab_debts=Tab_debts.reset_index(drop=True)
  Tab_debts=pd.concat([Tab_debts['document'],aux],axis=1)
  del aux
  Tab_debts=Tab_debts[['document','contract','createdAt','tags']]
  ###Formatando###
  Tab_debts['createdAt']=Tab_debts['createdAt'].dt.strftime('%d/%m/%Y')
  isna = Tab_debts['tags'].isna()
  Tab_debts.loc[isna,'tags']=pd.Series([['sem_skip']] * isna.sum()).values
  tags_values=['rank:a','rank:b','rank:c','rank:d','rank:e','sem_skip']
  Tab_debts['tags']=Tab_debts['tags'].map(lambda x : list(set(x).intersection(tags_values))[0] if  len(list(set(x).intersection(tags_values)))>0 else 'sem_skip')
  Tab_debts=pd.concat([Tab_debts,pd.get_dummies(Tab_debts['tags'],columns=tags_values).T.reindex(tags_values).T.fillna(0).astype(bool)],axis=1)
  Tab_debts=Tab_debts.drop(columns=['tags'])
  Tab_debts['chave']=Tab_debts['document']+":"+Tab_debts['contract']
  Tab_debts=Tab_debts.drop(columns=['document','contract'])
  ###Concatenando###
  Tab_vals1['chave']=Tab_vals1['document']+":"+Tab_vals1['Contrato_Altair']
  Tab_vals1=pd.merge(Tab_vals1,Tab_debts,how='left',on='chave')
  del Tab_debts
  Tab_vals1=Tab_vals1.rename(columns={'createdAt' : 'DATA_ENTRADA_DIVIDA'},inplace=False)
    
  #####FORMATANDO A TABELA PRINCIPAL E DROPANDO AS COLUNAS DESNECESSÁRIAS####
  print('\nFormatando as tabelas principais e dropando as colunas desnecessárias...')
  
  Tab_vals1.insert(2,'data_referencia',data_ref.strftime("%d/%m/%Y"))
  columns_vals=['ID_Credor','document','Contrato_Altair','genero','faixa_etaria','data_referencia','DATA_ENTRADA_DIVIDA','Indicador_Correntista','Cep1','Cep2','Cep3','Forma_Pgto_Padrao','Taxa_Padrao','Desconto_Padrao','Qtde_Min_Parcelas','Qtde_Max_Parcelas','Qtde_Parcelas_Padrao','Valor_Min_Parcela','Codigo_Politica','Saldo_Devedor_Contrato','Dias_Atraso','Descricao_Produto','ADIANTAMENTOS A DEPOSITANTES','CAPITAL DE GIRO MESOP','CARTAO FLEX INTERNACIONAL MC','CARTAO FREE GOLD MC','CARTAO FREE GOLD VISA','CARTAO SANTANDER FIT MC','CHEQUE ESPECIAL BANESPA','Outros','REFIN','SANTANDER STYLE PLATINUM MC','SANTANDER SX MASTER', 'rank:a','rank:b','rank:c','rank:d','rank:e','sem_skip']
  Tab_vals1=Tab_vals1[columns_vals]
  Tab_vals1=Tab_vals1.rename(columns={'document' : 'ID_DEVEDOR'},inplace=False)
  
  ####SALVANDO####
  print('\nSalvando as informações de débitos...')
  Tab_vals1.to_csv(dir_debts_datas+'/Base_UFPEL_QQ_Model_std_P'+str(j).zfill(2)+'.csv',index=False,sep=";")

  del Tab_vals1

# COMMAND ----------

dir_debts_datas+'/Base_UFPEL_QQ_Model_std_P'+str(j).zfill(2)+'.csv'