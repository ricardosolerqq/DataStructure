# Databricks notebook source
# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

#mount_blob_storage_oauth(dbutils,'qqprd','ml-prd',"/mnt/ml-prd")
dir_models='/mnt/ml-prd/ml-data/propensaodeal/santander/processed_score'
dir_outputs='/mnt/bi-reports/VALIDACAO-MODELOS-ML'

# COMMAND ----------

# DBTITLE 1,Carregando os pacotes do Python
from datetime import datetime, timedelta
import pandas as pd
import dateutil.relativedelta
from itertools import chain

# COMMAND ----------

#Tab_phones=[]
#query_tels=[
#  {
#      "$match" : {
#          "document" : {"$in" : lista_docs}
#      }
#  },
#  {
#      "$project" : {
#          "_id" : 0,
#          "Document" : "$document",
#          "phones" : {
#                      "$map" : {
#                          "input" : {"$ifNull" : ["$info.phones",[]]},
#                          "as" : "i",
#                          "in" : {
#                                  "tags" : {"$ifNull" : ["$$i.tags",["sem_tags"]]}
#                              }
#                      }
#                  }
#      }
#  }
#]
#
#for t in range(0,(len(partes)-1)):
#  print("\nExecutando a parte "+str(t+1)+" de "+str(len(partes)-1)+" da querys dos telefones...")  
#  lista_docs=list(set(DOCs_deals[partes[t]:(partes[t+1])]))
#  query_payments=[
#      {
#          "$match" : {
#              "document" : {"$in" : lista_docs},
#          }
#      },
#      {
#          "$match" : {
#              "creditor" : i.Credores,
#          }
#      },
#      {
#          "$unwind" : "$info"
#      },
#      {
#          "$match" : {
#              "installments.dealID" : {"$in" : lista_IDs},
#              "installments.status" : "paid"
#          }
#      },
#      {
#          "$group" : {
#              "_id" : "$installments.dealID",
#              "tot_pay" : {"$sum" : {"$cond" : [{"$eq" : ["$installments.payment.paidAmount",0]},
#                                                  "$installments.installmentAmount",
#                                                  "$installments.payment.paidAmount"]}},
#              "tot_parc_pags" : {"$sum" : 1.0}
#          }
#      },
#      {
#          "$project" : {
#              "_id" : 0,
#              "ID_acordo" : "$_id",
#              "document" : "$document",
#          }
#      }
#    ]
#
#

# COMMAND ----------

# DBTITLE 1,Selecionando os credores com modelo
Credores=spark.createDataFrame(dbutils.fs.ls(dir_outputs+'/Tabelas_Acionamentos_News')).withColumn('Credores',F.regexp_replace('name','/','')).withColumn('path',F.regexp_replace("path","dbfs:","")).collect()
col_interaction = getPyMongoCollection('col_interaction')
col_person = getPyMongoCollection('col_person')
for i in Credores:
  print("Montando o arquivo de "+i.Credores+'...')
  df_actual=spark.createDataFrame(dbutils.fs.ls(dir_outputs+'/Tabelas_Acionamentos_News/'+i.Credores)).withColumn('path',F.regexp_replace("path","dbfs:","")).filter(F.col('name').contains('last_files')).collect()
  df_actual=spark.read.format("csv").option("header", True).option("delimiter", ";").load(df_actual[0].path).withColumn('First_Data',F.to_date(F.col('First_Data'))).withColumn('Last_Data',F.to_date(F.col('Last_Data'))).cache()
  providers=df_actual.select('provider').distinct().rdd.flatMap(lambda x: x).collect()
  for w in providers:
    print('Selecionando as bases do modelo '+w+'...')
    df_actual_aux=df_actual.filter(F.col('provider')==w)
    data_min=df_actual_aux.agg({"First_Data": "min"}).rdd.flatMap(lambda x: x).collect()[0]
    data_min=datetime.strptime(data_min.strftime("%d/%m/%Y %H:%M:%S"),"%d/%m/%Y %H:%M:%S")+timedelta(hours=3)
    data_max=df_actual_aux.agg({"Last_Data": "max"}).rdd.flatMap(lambda x: x).collect()[0]
    data_max=datetime.strptime(data_max.strftime("%d/%m/%Y %H:%M:%S"),"%d/%m/%Y %H:%M:%S")+timedelta(hours=3)

    datas_limits=pd.date_range(data_min,data_max,freq='MS').strftime("%d/%m/%Y %H:%M:%S").tolist()
    datas_limits=[data_min.strftime("%d/%m/%Y %H:%M:%S")]+datas_limits
    if int(datas_limits[-1][3:5])==12:
      add_data=datas_limits[-1][0:3]+'01/'+str(int(datas_limits[-1][6:10])+1)+datas_limits[-1][10:]
    else:
      add_data=datas_limits[-1][0:3]+str(int(datas_limits[-1][3:5])+1).zfill(2)+datas_limits[-1][5:]

    datas_limits.append(add_data)
    del add_data
    
    print('Selecionando os acionamentos')
    
    acions=[]
    for j in range(0,(len(datas_limits)-1)):
      print("\nVerificando os acionamentos de "+datetime.strptime(datas_limits[j],"%d/%m/%Y %H:%M:%S").strftime("%B")+" de "+datetime.strptime(datas_limits[j],"%d/%m/%Y %H:%M:%S").strftime("%Y")+"...")

      query_acions=[
        {
            "$match" : {
                "creditor" : i.Credores,
                "sentAt" : {"$gte" : datetime.strptime(datas_limits[j],"%d/%m/%Y %H:%M:%S"), "$lt" : datetime.strptime(datas_limits[j],"%d/%m/%Y %H:%M:%S") + dateutil.relativedelta.relativedelta(months=1)}
            }
        },
        {
            "$match" : {
                "status" : {"$nin" : ["not_answered","not_received"]}
            }
        },
        {
            "$addFields" : {
                "sentAt" : {
                            "$dateToString": {
                                "date": "$sentAt",
                                "format": "%Y-%m-%d",
                                "timezone": "-0300"
                            }
                        }
            }
        },
        {
            "$group" : {
                "_id" : {"channel"  : {"$ifNull" : ["$channel",{ "$literal": "open_sms" }]}, "document" : "$document", "data" : "$sentAt"},
                "Tot_acions" : {"$sum" : 1.0}
            }
        },
        {
            "$project" : {
                "_id" : 0,
                "Document" : "$_id.document",
                "channel" : "$_id.channel",
                "data" : "$_id.data",
                "total_acions" : "$Tot_acions"
            }
        }
      ]
      acions.append(list(col_interaction.aggregate(pipeline=query_acions,allowDiskUse=True)))


    acions=spark.sparkContext.parallelize(list(chain.from_iterable(acions))).toDF()  
    acions=acions.groupBy('Document','data').agg(F.sum(F.when(F.col('channel').contains('sms'),F.col('total_acions')).otherwise(0)).alias('Acions_SMS'),
                                               F.sum(F.when(F.col('channel').contains('email'),F.col('total_acions')).otherwise(0)).alias('Acions_EMAIL'),
                                                      F.sum(F.when(F.col('channel').contains('voice'),F.col('total_acions')).otherwise(0)).alias('Acions_VOICE'))
    print('Concatenando os acionamentos com a base escorada...')   
    df_actual_concat=df_actual_aux.select(['Document','Score','provider','First_Data','Last_Data']).join(acions,'Document','left').withColumn('data',F.when(F.col('data').isNull(),F.col('First_Data')).otherwise(F.col('data'))).na.fill(0)
    
    print('Selecionando os acordos...')
    Tab_deals=[]
    query_acordos=[
        {
            "$match" : {
               "deals" : {
                    "$elemMatch" : {
                        "creditor" : i.Credores,
                        "status" : {"$ne" : "error"},
                        "createdAt" : {"$gte" : datetime.strptime(datas_limits[0],"%d/%m/%Y %H:%M:%S")}
                    }
                }
            }
        },
        {
            "$unwind" : "$deals"
        },
        {
            "$match" : {
                "deals.creditor" : i.Credores,
                "deals.status" : {"$ne" : "error"},
                "deals.createdAt" : {"$gte" :  datetime.strptime(datas_limits[0],"%d/%m/%Y %H:%M:%S")} 
            }
        },
        {
            "$project" : {
                "_id" : 0,
                "_id" : "$deals._id",
                "Document" : "$document",
                "total_acordo" : "$deals.totalAmount",
                "data_acordo" : "$deals.createdAt",
                "total_parcelas" : {"$toDouble" : "$deals.totalInstallments"},
                "valor_parcelas" : "$deals.installmentValue",
                "valor_entrada" : "$deals.upfront",
                "canal" : {
                            "$cond": [
                                { "$ifNull": ["$deals.tracking.channel", False] },
                                "$deals.tracking.channel",
                                {
                                    "$cond": [
                                        { "$ifNull": ["$deals.tracking.utms.source", False] },
                                        "$deals.tracking.utms.source",
                                        {
                                            "$cond": [
                                                { "$ifNull": ["$deals.offer.tokenData.channel", False] },
                                                "$deals.offer.tokenData.channel",
                                                { "$cond": [{ "$ifNull": ["$deals.simulationID", False] }, "web", "unknown"] }
                                            ]
                                        }
                                    ]
                                }
                            ]
                        } 
            }
        }
    ]

    Tab_deals.append(list(col_person.aggregate(pipeline=query_acordos,allowDiskUse=True)))
    try:
      print('Agregando e concatenando com a tabela principal...')
      Tab_deals=spark.sparkContext.parallelize(list(chain.from_iterable(Tab_deals))).toDF()
      Tab_deals=Tab_deals.sort(F.col('data_acordo').asc()).groupBy('Document').agg(F.max(F.col('data_acordo')).alias('data_acordo'),
                                                                                   F.last(F.col('total_acordo')).alias('total_acordo'),
                                                                                   F.last(F.col('_id')).alias('ID_acordo'),
                                                                                   F.last(F.col('total_parcelas')).alias('total_parcelas'),
                                                                                   F.last(F.col('valor_parcelas')).alias('valor_parcelas'),
                                                                                   F.last(F.col('valor_entrada')).alias('valor_entrada'),
                                                                                   F.last(F.col('canal')).alias('canal'))
      Tab_deals=Tab_deals.withColumn("data_acordo",F.col('data_acordo') - F.expr("INTERVAL 3 HOURS"))
      
      df_actual_concat=df_actual_concat.join(Tab_deals,'Document','left')
      df_actual_concat=df_actual_concat.withColumn('DEAL',F.when(F.col('data_acordo').isNull(),False).otherwise(F.when(F.col('data_acordo')>F.col('data'),True).otherwise(False))).withColumn('data_acordo',F.when(F.col('DEAL')==False,None).otherwise(F.col('data_acordo'))).withColumn('total_acordo',F.when(F.col('DEAL')==False,None).otherwise(F.col('total_acordo'))).withColumn('ID_acordo',F.when(F.col('DEAL')==False,None).otherwise(F.col('ID_acordo'))).withColumn('total_parcelas',F.when(F.col('DEAL')==False,None).otherwise(F.col('total_parcelas'))).withColumn('valor_parcelas',F.when(F.col('DEAL')==False,None).otherwise(F.col('valor_parcelas'))).withColumn('valor_entrada',F.when(F.col('DEAL')==False,None).otherwise(F.col('valor_entrada'))).withColumn('canal',F.when(F.col('DEAL')==False,None).otherwise(F.col('canal')))
    except:
      print('Nenhum acordo encontrado. Formatando a tabela para dar continuidade ao código...')
      df_actual_concat=df_actual_concat.withColumn('data_acordo',F.lit(None)).withColumn('total_acordo',F.lit(None)).withColumn('ID_acordo',F.lit(None)).withColumn('total_parcelas',F.lit(None)).withColumn('valor_parcelas',F.lit(None)).withColumn('valor_entrada',F.lit(None)).withColumn('canal',F.lit(None)).withColumn('DEAL',F.lit('0'))
    
      
    try:
      print('Verificando os pagamentos...')
      ID_deals=Tab_deals.select('ID_acordo').rdd.flatMap(lambda x: x).collect()
      DOCs_deals=Tab_deals.select('Document').rdd.flatMap(lambda x: x).collect()

      payments=[]

      partes=list(range(0,len(ID_deals),10000))

      if(partes[-1]!=len(ID_deals)):
        partes.append(len(ID_deals))

      for j in range(0,(len(partes)-1)):
        print("\nExecutando a parte "+str(j+1)+" de "+str(len(partes)-1)+" da querys dos pagamentos...")  
        lista_IDs=ID_deals[partes[j]:(partes[j+1])]
        lista_docs=list(set(DOCs_deals[partes[j]:(partes[j+1])]))
        query_payments=[
            {
                "$match" : {
                    "document" : {"$in" : lista_docs},
                }
            },
            {
                "$unwind" : "$installments"
            },
            {
                "$match" : {
                    "installments.dealID" : {"$in" : lista_IDs},
                    "installments.status" : "paid"
                }
            },
            {
                "$group" : {
                    "_id" : "$installments.dealID",
                    "tot_pay" : {"$sum" : {"$cond" : [{"$eq" : ["$installments.payment.paidAmount",0]},
                                                        "$installments.installmentAmount",
                                                        "$installments.payment.paidAmount"]}},
                    "tot_parc_pags" : {"$sum" : 1.0}
                }
            },
            {
                "$project" : {
                    "_id" : 0,
                    "ID_acordo" : "$_id",
                    "total_pago" : "$tot_pay",
                    "total_parcelas_pagas" : "$tot_parc_pags"
                }
            }
          ]
        payments.append(list(col_person.aggregate(pipeline=query_payments,allowDiskUse=True)))

      payments=spark.sparkContext.parallelize(list(chain.from_iterable(payments))).toDF()

      print('Concatenando os pagamentos os pagamentos...')
      df_actual_concat=df_actual_concat.join(payments,'ID_acordo','left')
    
    except:
      print('Não foram encontrados pagamentos. Formatando a tabela para dar continuidade ao código...')
      df_actual_concat=df_actual_concat.withColumn('total_pago',F.lit(None)).withColumn('total_parcelas_pagas',F.lit(None))
    df_actual_concat=df_actual_concat.withColumn('PAYMENTS',F.when(F.col('total_pago').isNull(),False).otherwise(True)).select(['Document','Score','provider','First_Data','Last_Data','data','Acions_SMS','Acions_EMAIL','Acions_VOICE','DEAL','data_acordo','total_acordo','total_parcelas','valor_entrada','canal','PAYMENTS','total_pago','total_parcelas_pagas','ID_acordo']).withColumn('Credor',F.lit(i.Credores))
    
    print('Excluindo o arquivo anterior, caso exista...')
    for  file in dbutils.fs.ls(dir_outputs+'/Tabelas_Acionamentos_News/'+i.Credores):
      if ('Analise_modelo_'+w) in file.name:
        dbutils.fs.rm(file.path, True)
      
    print('Salvando o arquivo final do modelo...')
    df_actual_concat.write.mode('overwrite').option("header",True).option("delimiter",";").option("emptyValue",'').option("nullValue",None).csv(dir_outputs+'/Tabelas_Acionamentos_News/'+i.Credores+'/Analise_modelo_'+w)

# COMMAND ----------

# DBTITLE 1,Carregando e salvando os arquivos na pasta dos BIs
#for i in Credores:
#  print('Carregando os arquivos dos modelos de '+i.Credores+'...')
#  basePaths=dir_outputs+'/Tabelas_Acionamentos_News/'+i.Credores+'/'
#  paths=[basePaths+'Analise_modelo_qq_*']
#  df_to_save=spark.read.option('header',True).option('delimiter',";").option('basePaths',basePaths).csv(*paths).withColumn('Separator',F.col('data')[0:7])
#  
#  print('Salvando particionando no arquivo do Power Bi...')
#  
#  df_to_save.coalesce(1).write.partitionBy('Separator').mode('overwrite').option("header",True).option("delimiter",";").option("emptyValue",'').csv(dir_outputs+'/arquivos_originais/Base acionada '+i.Credores[0].upper()+i.Credores[1:].lower())
#  
#  for directory in dbutils.fs.ls(dir_outputs+'/arquivos_originais/Base acionada '+i.Credores[0].upper()+i.Credores[1:].lower()):
#    if 'Separator' in  directory.name:
#      print (directory.name)
#      for file in dbutils.fs.ls(directory.path):
#        if '.csv' in file.name:
#          dbutils.fs.cp(file.path, dir_outputs+'/arquivos_originais/Base acionada '+i.Credores[0].upper()+i.Credores[1:].lower()+'/Base acionada '+i.Credores[0].upper()+i.Credores[1:].lower()+'_'+directory.name.split('=')[-1].replace('/','')+'.csv')
#        
#  print('Removendo os demais arquivos...')
#  
#  for removes in dbutils.fs.ls(dir_outputs+'/arquivos_originais/Base acionada '+i.Credores[0].upper()+i.Credores[1:].lower()):
#    if '.csv' not in removes.name:
#      dbutils.fs.rm(removes.path, True)
#
#  del df_to_save

# COMMAND ----------

Credores=spark.createDataFrame(dbutils.fs.ls(dir_outputs+'/Tabelas_Acionamentos_News')).withColumn('Credores',F.regexp_replace('name','/','')).withColumn('path',F.regexp_replace("path","dbfs:","")).collect()


# COMMAND ----------

for i in Credores:
  print('Carregando os arquivos dos modelos de '+i.Credores+'...')
  basePaths=dir_outputs+'/Tabelas_Acionamentos_News/'+i.Credores+'/'
  paths=[basePaths+'Analise_modelo_qq_*']
  df_to_save=spark.read.option('header',True).option('delimiter',";").option('basePaths',basePaths).csv(*paths).withColumn('Separator',F.col('data')[0:7])
  
  print('Salvando particionando no arquivo do Power Bi...')
  
  df_to_save.coalesce(1).write.partitionBy('Separator').mode('overwrite').option("header",True).option("delimiter",";").option("emptyValue",'').csv(dir_outputs+'/arquivos_originais/TMP/Base acionada '+i.Credores[0].upper()+i.Credores[1:].lower())
  
 
  for directory in dbutils.fs.ls(dir_outputs+'/arquivos_originais/TMP/Base acionada '+i.Credores[0].upper()+i.Credores[1:].lower()):
    if 'Separator' in  directory.name:
      for file in dbutils.fs.ls(directory.path):
        if '.csv' in file.name:
          dbutils.fs.cp(file.path, dir_outputs+'/arquivos_originais/Base acionada '+i.Credores[0].upper()+i.Credores[1:].lower()+'_'+directory.name.split('=')[-1].replace('/','')+'.csv')
 
  df_to_save.coalesce(1).write.mode('overwrite').option("header",True).option("delimiter",";").option("emptyValue",'').csv(dir_outputs+'/arquivos_originais/TMP/Base acionada '+i.Credores[0].upper()+i.Credores[1:].lower())
 
  for directory in dbutils.fs.ls(dir_outputs+'/arquivos_originais/TMP/Base acionada '+i.Credores[0].upper()+i.Credores[1:].lower()):
    for file in dbutils.fs.ls(directory.path):
      if '.csv' in file.name:
        dbutils.fs.cp(file.path, dir_outputs+'/arquivos_originais/Base acionada '+i.Credores[0].upper()+i.Credores[1:].lower()+'_Consolidado.csv')  

        
  print('Removendo os demais arquivos...')
    
  for removes in dbutils.fs.ls(dir_outputs+'/arquivos_originais/TMP/Base acionada '+i.Credores[0].upper()+i.Credores[1:].lower()):
    dbutils.fs.rm(removes.path, True)   
  dbutils.fs.rm(dir_outputs+'/arquivos_originais/TMP', True)
  
  del df_to_save

# COMMAND ----------

#dfVal = spark.read.options(header = 'True', delimiter = ';').csv(os.path.join(dir_outputs, 'arquivos_originais/Base acionada Bahamas_2022-05.csv'))
#
#display(dfVal.dropDuplicates(['ID_acordo']).filter(F.col('data_acordo').isNotNull()).groupBy(F.col('data_acordo')[0:7].alias('ANO_MES')).count())