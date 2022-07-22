# Databricks notebook source
# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

#%run /Users/bruno.caravieri@queroquitar.com.br/SparkSet

# COMMAND ----------

#st = SparkTools()

# COMMAND ----------

#mount_blob_storage_oauth(dbutils,'qqprd','ml-prd',"/mnt/ml-prd")
dir_models='/mnt/ml-prd/ml-data/propensaodeal/santander/processed_score'
dir_outputs='/mnt/bi-reports/VALIDACAO-MODELOS-ML'

# COMMAND ----------

# DBTITLE 1,Selecionando os credores ativos
col_creditor = getPyMongoCollection('col_creditor')

query_creditor=[
  {
    "$match" : {
        "active" : True,
        "_id" : {"$nin" : ["fake","fake2","pernambucanas"]}
      }
  },
  {
    "$project" : {
        "_id" : 1
      }
  }
]

Credores=pd.DataFrame(list(col_creditor.aggregate(pipeline=query_creditor,allowDiskUse=True)))['_id'].to_list()
Credores=pd.DataFrame({'Credores' : Credores,'Cod_leitura' : Credores})
Credores.iloc[Credores['Cod_leitura']=='santander',1]='part-'
Credores=spark.createDataFrame(Credores).collect()
display(Credores)

# COMMAND ----------

# DBTITLE 1,Selecionando os arquivos que serão lidos
df_data=spark.createDataFrame(dbutils.fs.ls(dir_models)).withColumn('path',F.regexp_replace("path","dbfs:",""))
#df_data_read=df_data.filter(F.col('name').contains(Credores[21].Cod_leitura)).withColumn('index',F.monotonically_increasing_id()).collect()
df_data_read=df_data.filter((F.col('name').contains(Credores[22].Cod_leitura)) & (F.col('size')>0))
df_readed_files=spark.read.format("csv").option("header", True).option("delimiter", ";").load(dir_outputs+'/Tabelas_Acionamentos_News/'+Credores[22].Credores+'/readed_files_'+Credores[22].Credores+'.csv').rdd.flatMap(lambda x: x).collect()
df_data_read=df_data_read.filter(~(F.col('name').isin(df_readed_files)))
#df_data_read=df_data_read.filter(F.col('name').isin(df_readed_files))
display(df_data_read)
#files_read_all=df_data_read.select('path').rdd.flatMap(lambda x: x).collect()
#files_read_all=['"'+x+'"' for x in files_read_all]
#files_read_all=','.join(files_read_all)

# COMMAND ----------

df_data_read_2=df_data_read.collect()
df_data_read_2 = [ x.path for x in df_data_read_2]
#df_data_read2


BASES_ORIGINAL = [spark.read.option("header", "true")\
         .option("delimiter", ";")\
         .csv(x)\
         for x in df_data_read_2]
head = True
for x in BASES_ORIGINAL:
  try:
    if head == True:
      df = x
      head = False
    else:
      df = df.union(x)
  except:
    print(x)
    
df=df.withColumn('CreatedAt',F.to_date(F.col('CreatedAt'))).sort(F.col('CreatedAt').asc())    

del BASES_ORIGINAL, df_data_read_2

df=df.groupby('Document','provider').agg(F.last('Score').alias('Score'),
                                        F.last('ScoreValue').alias('ScoreValue'),
                                        F.last('ScoreAvg').alias('ScoreAvg'),
                                        F.first('CreatedAt').alias('First_Data'),
                                        F.last('CreatedAt').alias('Last_Data')).cache()

# COMMAND ----------

# DBTITLE 1,Carregando as bases anteriores (caso existam)
try:
  print("Concatenando com a base anterior...")
  last_table=spark.createDataFrame(dbutils.fs.ls(dir_outputs+'/Tabelas_Acionamentos_News/'+Credores[22].Credores)).withColumn('path',F.regexp_replace("path","dbfs:","")).filter(F.col('name').contains('last_files')).collect()
  last_table=spark.read.format("csv").option("header", True).option("delimiter", ";").load(last_table[0].path).withColumn('First_Data',F.to_date(F.col('First_Data'))).withColumn('Last_Data',F.to_date(F.col('Last_Data'))).cache()
  df=df.union(last_table).sort(F.col('Last_Data').asc())
  #del last_table
except:
  print("Nenhum arquivo anterior encontrado, acumulando os já lidos...")
  pass


df_scores_model=df.groupby('Document','provider').agg(F.last('Score').alias('Score'),
                                        F.last('ScoreValue').alias('ScoreValue'),
                                        F.last('ScoreAvg').alias('ScoreAvg'),
                                        F.min('First_Data').alias('First_Data'),
                                        F.max('Last_Data').alias('Last_Data')).cache()
#data_min=df_scores_model.agg({"First_Data": "min"}).collect()[0]
#data_max=df_scores_model.agg({"Last_Data": "max"}).collect()[0]
del df

# COMMAND ----------

# DBTITLE 1,Atualizando e salvando a lista de arquivos já lidos
df_readed_files=spark.read.format("csv").option("header", True).option("delimiter", ";").load(dir_outputs+'/Tabelas_Acionamentos_News/'+Credores[22].Credores+'/readed_files_'+Credores[22].Credores+'.csv')
df_readed_files=df_readed_files.union(df_data_read.select('name'))

df_readed_files.coalesce(1).write.mode('overwrite').option("header",True).option("delimiter",";").option("emptyValue",'').csv(dir_outputs+'/Tabelas_Acionamentos_News/'+Credores[22].Credores+'/readed_files_'+Credores[22].Credores)

for file in dbutils.fs.ls(dir_outputs+'/Tabelas_Acionamentos_News/'+Credores[22].Credores+'/readed_files_'+Credores[22].Credores):
  if file.name.split('.')[-1] == 'csv':
    print (file)
    dbutils.fs.cp(file.path, dir_outputs+'/Tabelas_Acionamentos_News/'+Credores[22].Credores+'/readed_files_'+Credores[22].Credores+'.csv')
dbutils.fs.rm(dir_outputs+'/Tabelas_Acionamentos_News/'+Credores[22].Credores+'/readed_files_'+Credores[22].Credores, True)

del df_readed_files, df_data_read

# COMMAND ----------

# DBTITLE 1,Salvando todos os escorados
from datetime import datetime, timedelta
data_hj=(datetime.today()-timedelta(hours=3)).strftime("%Y_%m_%d")
df_scores_model.coalesce(1).write.mode('overwrite').option("header",True).option("delimiter",";").option("emptyValue",'').csv(dir_outputs+'/Tabelas_Acionamentos_News/'+Credores[22].Credores+'/last_files_'+Credores[22].Credores)

for file in dbutils.fs.ls(dir_outputs+'/Tabelas_Acionamentos_News/'+Credores[22].Credores+'/last_files_'+Credores[22].Credores):
  if file.name.split('.')[-1] == 'csv':
    print (file)
    dbutils.fs.cp(file.path, dir_outputs+'/Tabelas_Acionamentos_News/'+Credores[22].Credores+'/last_files_'+Credores[22].Credores+'_'+data_hj+'.csv')
dbutils.fs.rm(dir_outputs+'/Tabelas_Acionamentos_News/'+Credores[22].Credores+'/last_files_'+Credores[22].Credores, True)

del df_scores_model

# COMMAND ----------

# DBTITLE 1,Excluindo os arquivos anteriores caso existam
try:
  last_file_remove=spark.createDataFrame(dbutils.fs.ls(dir_outputs+'/Tabelas_Acionamentos_News/'+Credores[22].Credores)).filter(~(F.col('name')=='last_files_'+Credores[22].Credores+'_'+data_hj+'.csv')).collect()
  for i in last_file_remove:
    if 'last_files' in i.name:
      dbutils.fs.rm(i.path, True)
  del last_file_remove    
except:
  pass

# COMMAND ----------

# DBTITLE 1,Montando a query dos acionamentos
from datetime import datetime, timedelta
import pandas as pd
from itertools import chain
###Carregando a base atualizada###

df_actual=spark.createDataFrame(dbutils.fs.ls(dir_outputs+'/Tabelas_Acionamentos_News/'+Credores[22].Credores)).withColumn('path',F.regexp_replace("path","dbfs:","")).filter(F.col('name').contains('last_files')).collect()
df_actual=spark.read.format("csv").option("header", True).option("delimiter", ";").load(df_actual[0].path).withColumn('First_Data',F.to_date(F.col('First_Data'))).withColumn('Last_Data',F.to_date(F.col('Last_Data'))).cache()

providers=df_actual.select('provider').distinct().rdd.flatMap(lambda x: x).collect()
data_min=df_actual.agg({"First_Data": "min"}).rdd.flatMap(lambda x: x).collect()[0]
data_min=datetime.strptime(data_min.strftime("%d/%m/%Y %H:%M:%S"),"%d/%m/%Y %H:%M:%S")+timedelta(hours=3)
data_max=df_actual.agg({"Last_Data": "max"}).rdd.flatMap(lambda x: x).collect()[0]
data_max=datetime.strptime(data_max.strftime("%d/%m/%Y %H:%M:%S"),"%d/%m/%Y %H:%M:%S")+timedelta(hours=3)

datas_limits=pd.date_range(data_min,data_max,freq='MS').strftime("%d/%m/%Y %H:%M:%S").tolist()
if int(datas_limits[-1][3:5])==12:
  add_data=datas_limits[-1][0:3]+'01/'+str(int(datas_limits[-1][6:10])+1)+datas_limits[-1][10:]
else:
  add_data=datas_limits[-1][0:3]+str(int(datas_limits[-1][3:5])+1).zfill(2)+datas_limits[-1][5:]
  
datas_limits.append(add_data)
datas_limits=[data_min.strftime("%d/%m/%Y %H:%M:%S")]+datas_limits
del add_data
datas_limits

# COMMAND ----------

acions=[]

col_interaction = getPyMongoCollection('col_interaction')
#for j in range(1,2):
for j in range(0,(len(datas_limits)-1)):
  print("\nVerificando os acionamentos de "+datetime.strptime(datas_limits[j],"%d/%m/%Y %H:%M:%S").strftime("%B")+" de "+datetime.strptime(datas_limits[j],"%d/%m/%Y %H:%M:%S").strftime("%Y")+"...")
  
  query_acions=[
	{
		"$match" : {
			"creditor" : Credores[22].Credores,
			"sentAt" : {"$gte" : datetime.strptime(datas_limits[j],"%d/%m/%Y %H:%M:%S"), "$lt" : datetime.strptime(datas_limits[j+1],"%d/%m/%Y %H:%M:%S")}
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

# COMMAND ----------

acions=acions.groupBy('Document','data').agg(F.sum(F.when(F.col('channel').contains('sms'),F.col('total_acions')).otherwise(0)).alias('Acions_SMS'),
                                               F.sum(F.when(F.col('channel').contains('email'),F.col('total_acions')).otherwise(0)).alias('Acions_EMAIL'),
                                                      F.sum(F.when(F.col('channel').contains('voice'),F.col('total_acions')).otherwise(0)).alias('Acions_VOICE'))

# COMMAND ----------

# DBTITLE 1,Concatenando com o arquivo escorado
df_actual_concat=df_actual.select(['Document','Score','provider','First_Data','Last_Data']).join(acions,'Document','left').withColumn('data',F.when(F.col('data').isNull(),F.col('First_Data')).otherwise(F.col('data'))).na.fill(0)
df_actual_concat.show(15)
#del col_interaction

# COMMAND ----------

# DBTITLE 1,Selecionando a base de acordos
col_person = getPyMongoCollection('col_person')

Tab_deals=[]
query_acordos=[
	{
		"$match" : {
			"documentType" : "cpf",
			"deals" : {
				"$elemMatch" : {
					"creditor" : Credores[22].Credores,
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
			"deals.creditor" : Credores[22].Credores,
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

#schema=T.StructType([T.StructField("ID_acordo", T.StringType(), True),
 #                   T.StructField("CPF", T.StringType(), True),
  #                  T.StructField("total_acordo", T.DoubleType(), True),
   #                 T.StructField("data_acordo", T.DateType(), True),
    #                T.StructField("total_parcelas", T.StringType(), True),
     #               T.StructField("valor_parcelas", T.DoubleType(), True),
      #              T.StructField("valor_entrada", T.DoubleType(), True),
       #             T.StructField("canal", T.StringType(), True)])


Tab_deals=spark.sparkContext.parallelize(list(chain.from_iterable(Tab_deals))).toDF()

# COMMAND ----------

Tab_deals_formatted=Tab_deals.sort(F.col('data_acordo').asc()).groupBy('Document').agg(F.max(F.col('data_acordo')).alias('data_acordo'),
                                                                                 F.last(F.col('total_acordo')).alias('total_acordo'),
                                                                                 F.last(F.col('_id')).alias('ID_acordo'),
                                                                                 F.last(F.col('total_parcelas')).alias('total_parcelas'),
                                                                                 F.last(F.col('valor_parcelas')).alias('valor_parcelas'),
                                                                                 F.last(F.col('valor_entrada')).alias('valor_entrada'),
                                                                                 F.last(F.col('canal')).alias('canal'))
Tab_deals_formatted=Tab_deals_formatted.withColumn("data_acordo",F.col('data_acordo') - F.expr("INTERVAL 3 HOURS"))
display(Tab_deals_formatted)

# COMMAND ----------

df_actual_concat=df_actual_concat.join(Tab_deals_formatted,'Document','left')
df_actual_concat=df_actual_concat.withColumn('DEAL',F.when(F.col('data_acordo').isNull(),False).otherwise(F.when(F.col('data_acordo')>F.col('data'),True).otherwise(False))).withColumn('data_acordo',F.when(F.col('DEAL')==False,None).otherwise(F.col('data_acordo'))).withColumn('total_acordo',F.when(F.col('DEAL')==False,None).otherwise(F.col('total_acordo'))).withColumn('ID_acordo',F.when(F.col('DEAL')==False,None).otherwise(F.col('ID_acordo'))).withColumn('total_parcelas',F.when(F.col('DEAL')==False,None).otherwise(F.col('total_parcelas'))).withColumn('valor_parcelas',F.when(F.col('DEAL')==False,None).otherwise(F.col('valor_parcelas'))).withColumn('valor_entrada',F.when(F.col('DEAL')==False,None).otherwise(F.col('valor_entrada'))).withColumn('canal',F.when(F.col('DEAL')==False,None).otherwise(F.col('canal')))

# COMMAND ----------

ID_deals=Tab_deals_formatted.select('ID_acordo').rdd.flatMap(lambda x: x).collect()
DOCs_deals=Tab_deals_formatted.select('Document').rdd.flatMap(lambda x: x).collect()

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

# COMMAND ----------

df_actual_concat=df_actual_concat.join(payments,'ID_acordo','left')
#display(df_actual_concat)

# COMMAND ----------

df_actual_concat=df_actual_concat.withColumn('PAYMENTS',F.when(F.col('total_pago').isNull(),False).otherwise(True)).select(['Document','Score','provider','First_Data','Last_Data','data','Acions_SMS','Acions_EMAIL','Acions_VOICE','DEAL','data_acordo','total_acordo','total_parcelas','valor_entrada','canal','PAYMENTS','total_pago','total_parcelas_pagas','ID_acordo']).withColumn('Credor',F.lit(Credores[22].Credores))
#display(df_actual_concat)

# COMMAND ----------

# DBTITLE 1,Salvando no drive auxiliar de atualização
df_actual_concat.write.mode('overwrite').option("header",True).option("delimiter",";").option("emptyValue",'').csv(dir_outputs+'/Tabelas_Acionamentos_News/'+Credores[22].Credores+'/Analise_modelo_'+providers[0])

# COMMAND ----------

paths=dir_outputs+'/Tabelas_Acionamentos_News/'+Credores[22].Credores+'/Analise_modelo_'+providers[0]+'/*.csv'
df_to_save=spark.read.format("csv").option("header", True).option("delimiter", ";").csv(paths)

# COMMAND ----------

display(df_to_save)

# COMMAND ----------

# DBTITLE 1,Carregando e salvando os arquivos na pasta dos BIs
paths=dir_outputs+'/Tabelas_Acionamentos_News/'+Credores[22].Credores+'/Analise_modelo_'+providers[0]+'/*.csv'
df_to_save=spark.read.format("csv").option("header", True).option("delimiter", ";").csv(paths).withColumn('Separator',F.col('data')[0:7])

df_to_save.coalesce(1).write.partitionBy('Separator').mode('overwrite').option("header",True).option("delimiter",";").option("emptyValue",'').csv(dir_outputs+'/arquivos_originais/Base acionada '+Credores[22].Credores[0].upper()+Credores[22].Credores[1:].lower())

for directory in dbutils.fs.ls(dir_outputs+'/arquivos_originais/'+'Base acionada '+Credores[22].Credores[0].upper()+Credores[22].Credores[1:].lower()):
  if 'Separator' in  directory.name:
    print (directory.name)
    for file in dbutils.fs.ls(directory.path):
      if '.csv' in file.name:
        dbutils.fs.cp(file.path, dir_outputs+'/arquivos_originais/Base acionada '+Credores[22].Credores[0].upper()+Credores[22].Credores[1:].lower()+'/Base acionada '+Credores[22].Credores[0].upper()+Credores[22].Credores[1:].lower()+'_'+directory.name.split('=')[-1].replace('/','')+'.csv')
      
    
#REMOVENDO OS DEMAIS ARQUIVOS

for removes in dbutils.fs.ls(dir_outputs+'/arquivos_originais/'+'Base acionada '+Credores[22].Credores[0].upper()+Credores[22].Credores[1:].lower()):
  if '.csv' not in removes.name:
    dbutils.fs.rm(removes.path, True)

del df_to_save

# COMMAND ----------



# COMMAND ----------

dir_separator[3].name

# COMMAND ----------

dir_acions=dbutils.fs.ls(dir_outputs+'/arquivos_originais/'+'Base acionada '+Credores[22].Credores[0].upper()+Credores[22].Credores[1:].lower())

dir_separator=dbutils.fs.ls(dir_acions[0].path)

dir_separator