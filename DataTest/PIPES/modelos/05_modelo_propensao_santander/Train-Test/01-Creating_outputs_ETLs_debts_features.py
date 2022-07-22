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
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime,timedelta

##Selecionando os Parquets###
dir_parquets=os.listdir(dir_files)
dir_parquets=list(filter(lambda x: '2021-12-14' in x, dir_parquets))
#dir_parquets
###
arqs=os.listdir(dir_files+'/'+dir_parquets[0])
arqs=list(filter(lambda x: '.parquet' in x, arqs))
arqs

# COMMAND ----------

data_ref=datetime.strptime('2021-12-14 03:00:00',"%Y-%m-%d %H:%M:%S")
data_hj=datetime.strptime(str((datetime.today()-timedelta(hours=3)).strftime("%d/%m/%Y"))+' 03:00:00',"%d/%m/%Y %H:%M:%S")
data_limit=datetime.strptime(str(data_ref+timedelta(days=31)),"%Y-%m-%d %H:%M:%S")
#data_limit.strftime("%Y-%m-%d %H:%M:%S")
str(data_limit)

# COMMAND ----------

# DBTITLE 1,Carregando as tabelas e variáveis auxiliares 
dir_prod='/dbfs/mnt/etlsantander/Base_UFPEL'
Tab_prod=pd.read_csv(dir_prod+'/Contagem dos produtos.csv',sep=',')
Tab_prod=Tab_prod['Produto'][0:12]
Tab_prod=Tab_prod.append(pd.Series("Outros")).reset_index(drop=True)

data_ref=datetime.strptime('2021-12-14 03:00:00',"%Y-%m-%d %H:%M:%S")
data_hj=datetime.strptime(str((datetime.today()-timedelta(hours=3)).strftime("%d/%m/%Y"))+' 03:00:00',"%d/%m/%Y %H:%M:%S")
data_limit=datetime.strptime(str(data_ref+timedelta(days=31)),"%Y-%m-%d %H:%M:%S")
col_person = getPyMongoCollection('col_person')

dir_IDs='/dbfs/mnt/etlsantander/Base_UFPEL/IDs_base'
if not os.path.exists(dir_IDs+'/Dados='+data_ref.strftime("%Y-%m-%d")):
  os.makedirs(dir_IDs+'/Dados='+data_ref.strftime("%Y-%m-%d"))

dir_IDs_datas=dir_IDs+'/Dados='+data_ref.strftime("%Y-%m-%d")

dir_debts='/dbfs/mnt/etlsantander/Base_UFPEL/Skips_pags'
if not os.path.exists(dir_debts+'/Dados='+data_ref.strftime("%Y-%m-%d")):
  os.makedirs(dir_debts+'/Dados='+data_ref.strftime("%Y-%m-%d"))
  
dir_debts_datas=dir_debts+'/Dados='+data_ref.strftime("%Y-%m-%d")

# COMMAND ----------

# DBTITLE 1,Montando a base formatada para ser salva no blob
for j in range(0,len(arqs)):
  print('\nLendo o arquivo '+str(j+1)+' de '+str(len(arqs))+'...')
  tab_training=pd.read_parquet(dir_files+'/'+dir_parquets[0]+'/'+arqs[j])
  
  ####FORMATANDO AS VARIÁVEIS DA BASE####
  print('\nFormatando as variáveis da base...')
  tab_training['Saldo_Devedor_Contrato']=tab_training['Saldo_Devedor_Contrato'].str.strip('+').str.replace(",",".").astype(float)
  tab_training['Taxa_Padrao']=tab_training['Taxa_Padrao'].str.strip('+').str.replace(",",".").astype(float)
  tab_training['Desconto_Padrao']=tab_training['Desconto_Padrao'].str.strip('+').str.replace(",",".").astype(float)
  tab_training['Percentual_Min_Entrada']=tab_training['Percentual_Min_Entrada'].str.strip('+').str.replace(",",".").astype(float)
  tab_training['Valor_Min_Parcela']=tab_training['Valor_Min_Parcela'].str.strip('+').str.replace(",",".").astype(float)
  tab_training['Numero_Documento']=tab_training['Numero_Documento'].str.replace("[.,-]","").str.strip().astype(str)
  tab_training['Dias_Atraso']=tab_training['Dias_Atraso'].astype(int)
  tab_training['Qtde_Min_Parcelas']=tab_training['Qtde_Min_Parcelas'].astype(int)
  tab_training['Qtde_Max_Parcelas']=tab_training['Qtde_Max_Parcelas'].astype(int)
  tab_training['Qtde_Parcelas_Padrao']=tab_training['Qtde_Parcelas_Padrao'].astype(int)
  tab_training['Descricao_Produto']=tab_training['Descricao_Produto'].str.rstrip()
  tab_training['Contrato_Altair']=tab_training['Contrato_Altair'].str.slice(14,26)
  colunas_tab1=['Numero_Documento','Contrato_Altair','Indicador_Correntista','Cep1','Cep2','Cep3','Forma_Pgto_Padrao','Taxa_Padrao','Desconto_Padrao','Qtde_Min_Parcelas','Qtde_Max_Parcelas','Qtde_Parcelas_Padrao','Valor_Min_Parcela','Codigo_Politica','Saldo_Devedor_Contrato','Dias_Atraso','Descricao_Produto']
  Tab_vals1=tab_training[colunas_tab1]

  Tab_vals1.insert(0,'ID_Credor','std')
  Tab_vals1['Indicador_Correntista']=Tab_vals1['Indicador_Correntista'].str.replace("N","0").str.replace("S","1").astype(int).astype(bool)
  del tab_training
  
  ####INCLUINDO AS DUMMIES DE PRODUTOS####
 # print('\nIncluindo as dummies de produtos...')
 # aux_format=Tab_vals1['Descricao_Produto'].unique()
  #Tab_result=[difflib.get_close_matches(str(word), Tab_prod,n=1,cutoff=0.75) for word in aux_format]
  #Tab_result=[["Outros"] if x ==[] else x for x in Tab_result]
  #Tab_result=pd.DataFrame(data=list(zip(list(chain.from_iterable(Tab_result)),aux_format)),columns=['Dummie_produto','Descricao_Produto'])
  #Tab_vals1=pd.merge(Tab_vals1,Tab_result,how='left',on='Descricao_Produto')
  #Tab_vals1=pd.concat([Tab_vals1,pd.get_dummies(Tab_vals1['Dummie_produto'],columns=Tab_prod).T.reindex(Tab_prod).T.fillna(0).astype(bool)],axis=1)
  #Tab_vals1=Tab_vals1.drop('Dummie_produto',axis=1)
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
                      },	
              "deals" : {
                          "$filter" : {
                              "input" : {"$ifNull" : ["$deals",[]]},
                              "cond" : {"$and" : [
                                              {"$eq" : ["$$this.creditor","santander"]},
                                              {"$ne" : ["$$this.status","error"]},
                                              {"$gte" : ["$$this.createdAt",data_ref]},
                                              {"$lt" : ["$$this.createdAt",data_limit]}
                                          ]
                                  }
                          }
                      },
			"installments" : {
								"$filter" : {
									"input" : {"$ifNull" : [{"$map": {
															            "input" : "$installments",
															            "as": "i",
															            "in": {
															                "totalIstallments" : "$$i.totalInstallments",
															                "installment" : "$$i.installment",
															                "paidAmount" : "$$i.payment.paidAmount",
															                "installmentAmount" : "$$i.installmentAmount",
															                "dealID" : "$$i.dealID",
															                "creditor" : "$$i.creditor",
															                "status" : "$$i.status",
															                "createdAt" : "$$i.createdAt",
															                "paidAt" : { "$toString" : "$$i.payment.paidAt" }                
															            } 
															         }
															},
															[]]
											},
									"cond" : {"$and" : [
													{"$eq" : ["$$this.creditor","santander"]},
													{"$eq" : ["$$this.status","paid"]},
													{"$gte" : ["$$this.createdAt",data_ref]},
                                                    {"$lt" : ["$$this.createdAt",data_limit]}
												]
										}
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
  
  ####SELECIONANDO OS OUTPUTS DE ACORDO###
  print('\nSelecionando os outputs de acordo...')
  Deals_tab=Tab_rest[['document','deals']].copy().explode(column=['deals']).dropna(subset=['deals'])
  aux=pd.json_normalize(Deals_tab['deals'])
  aux=aux[['_id','totalAmount','installmentValue','upfront','totalInstallments','offer.debts','createdAt']]
  Deals_tab=Deals_tab.reset_index(drop=True)
  Deals_tab=pd.concat([Deals_tab['document'],aux],axis=1)
  aux=Deals_tab[['document','_id','offer.debts']].copy().explode(column=['offer.debts'])
  aux2=pd.json_normalize(aux['offer.debts'])
  aux2=aux2[['contract','createdAt']]
  aux=aux.reset_index(drop=True)
  aux=pd.concat([aux,aux2],axis=1)
  aux=aux.drop(columns=['offer.debts','document','createdAt'])
  Deals_tab=pd.merge(Deals_tab,aux,how='left',on='_id')
  Deals_tab['chave']=Deals_tab['document']+":"+Deals_tab['contract']
  Deals_tab=Deals_tab.sort_values(by=['createdAt'])
  Deals_tab=Deals_tab.groupby('chave',as_index=False).agg(Val_acordo=('totalAmount','last'),
                                                            Qtd_parcelas=('totalInstallments','last'),
                                                           ID_acordo=('_id','last'))
  Deals_tab['ACORDO']=True
  Deals_tab['ACORDO_A_VISTA']= np.where(Deals_tab['Qtd_parcelas']>1, False, True)
  del aux,aux2
  ###Concatenando###
  Tab_vals1=pd.merge(Tab_vals1,Deals_tab,how='left',on=['chave'])
  Tab_vals1['ACORDO']=Tab_vals1['ACORDO'].fillna(False)
  Tab_vals1['ACORDO_A_VISTA']=Tab_vals1['ACORDO_A_VISTA'].fillna(False)
  del  Deals_tab
  
  ####SELECIONANDO OS OUTPUTS DE PAGAMENTO####
  print('\nSelecionando os outputs de pagamento...')
  inst_tab=Tab_rest[['document','installments']].copy().explode(column=['installments']).dropna(subset=['installments'])
  aux=pd.json_normalize(inst_tab['installments'])
  aux=aux[['dealID','paidAmount','installmentAmount','installment']]
  aux['val_pago']= np.where(aux['paidAmount']==0,aux['installmentAmount'],aux['paidAmount'])
  aux['Pagto_Parcela1']=np.where(aux['installment']==1,aux['val_pago'],0)
  aux['Pagto_Demais_Parcelas']=np.where(aux['installment']==1,0,aux['val_pago'])
  inst_tab=inst_tab.reset_index(drop=True)
  inst_tab=pd.concat([inst_tab['document'],aux],axis=1)
  inst_tab=inst_tab.groupby('dealID',as_index=False).agg(Pagto_Parcela1=('Pagto_Parcela1','first'),
                                                        Pagto_Demais_Parcelas=('Pagto_Demais_Parcelas','sum'))
  del aux,Tab_rest
  inst_tab['PAGTO']=True
  inst_tab=inst_tab.rename(columns={'dealID' : 'ID_acordo'},inplace=False)
  ###Concatenando###
  Tab_vals1=pd.merge(Tab_vals1,inst_tab,how='left',on=['ID_acordo'])
  Tab_vals1['PAGTO']=Tab_vals1['PAGTO'].fillna(False)
  Tab_vals1['Pagto_Parcela1']=Tab_vals1['Pagto_Parcela1'].fillna(0)
  Tab_vals1['Pagto_Demais_Parcelas']=Tab_vals1['Pagto_Demais_Parcelas'].fillna(0)
  Tab_vals1['PAGTO_A_VISTA']= np.where((Tab_vals1['ACORDO_A_VISTA'] & Tab_vals1['PAGTO']), True, False)
  Tab_vals1['R$_PAGTO_A_VISTA']=np.where(Tab_vals1['PAGTO_A_VISTA'],Tab_vals1['Pagto_Parcela1'], 0)
  del inst_tab
  Tab_vals1=Tab_vals1.rename(columns={'Pagto_Parcela1' : 'R$_Pagto_Parcela1', 'Pagto_Demais_Parcelas' : 'R$_Pagto_Demais_Parcelas'},inplace=False)
  
  #####FORMATANDO A TABELA PRINCIPAL E DROPANDO AS COLUNAS DESNECESSÁRIAS####
  print('\nFormatando as tabelas principais e dropando as colunas desnecessárias...')
  #Tab_IDs=Tab_vals1[['ID_DEVEDOR','document','chave']].copy()
  Tab_vals1.insert(2,'data_referencia',data_ref.strftime("%d/%m/%Y"))
  columns_vals=['ID_Credor','document','Contrato_Altair','genero','faixa_etaria','data_referencia','DATA_ENTRADA_DIVIDA','Indicador_Correntista','Cep1','Cep2','Cep3','Forma_Pgto_Padrao','Taxa_Padrao','Desconto_Padrao','Qtde_Min_Parcelas','Qtde_Max_Parcelas','Qtde_Parcelas_Padrao','Valor_Min_Parcela','Codigo_Politica','Saldo_Devedor_Contrato','Dias_Atraso','Descricao_Produto','rank:a','rank:b','rank:c','rank:d','rank:e','sem_skip','Val_acordo','Qtd_parcelas','ID_acordo','ACORDO','ACORDO_A_VISTA','R$_Pagto_Parcela1','R$_Pagto_Demais_Parcelas','PAGTO','PAGTO_A_VISTA','R$_PAGTO_A_VISTA']
  Tab_vals1=Tab_vals1[columns_vals]
  
  ####SALVANDO####
  print('\nSalvando as informações de débitos...')
  Tab_vals1.to_csv(dir_debts_datas+'/Base_UFPEL_QQ_Model_std_P'+str(j).zfill(2)+'.csv',index=False,sep=";")

  print('\nSalvando as informações dos IDs dos documentos...')
  #Tab_IDs.to_csv(dir_IDs_datas+'/IDs_documents_std_P'+str(j).zfill(2)+'.csv',index=False,sep=";")
  del Tab_vals1

# COMMAND ----------

df = spark.read.options(delimiter =';', header = True).csv('/mnt/etlsantander/Base_UFPEL/Skips_pags/Dados=2021-12-14')


# COMMAND ----------

df.filter(F.col('PAGTO_A_VISTA')==True).count()

# COMMAND ----------

df.filter(F.col('PAGTO')==True).count()

# COMMAND ----------

df.filter(F.col('ACORDO')==True).count()