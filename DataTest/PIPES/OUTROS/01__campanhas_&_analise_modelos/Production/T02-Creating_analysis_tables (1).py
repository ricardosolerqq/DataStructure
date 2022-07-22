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
from itertools import chain

# COMMAND ----------

col_person = getPyMongoCollection('col_person')


# COMMAND ----------

Credores=spark.createDataFrame(dbutils.fs.ls(dir_outputs+'/Tabelas_Acionamentos_News')).withColumn('Credores',F.regexp_replace('name','/','')).withColumn('path',F.regexp_replace("path","dbfs:","")).collect()


# COMMAND ----------

basePaths=dir_outputs+'/Tabelas_Acionamentos_News/'+Credores[13].Credores+'/'
paths=[basePaths+'Analise_modelo_qq_*']
df_to_save=spark.read.option('header',True).option('delimiter',";").option('basePaths',basePaths).csv(*paths).withColumn('Separator',F.col('data')[0:7])


# COMMAND ----------

# DBTITLE 1,Schema person
spark.conf.set('spark.sql.caseSensitive', True)

DATA_PATH = "/dbfs/SCHEMAS/DATA_SCHEMA.json"
with open(DATA_PATH) as f:
  SCHEMA = T.StructType.fromJson(json.load(f))

# COMMAND ----------

# DBTITLE 1,Info
dfInfo = spark.read.schema(SCHEMA).json('dbfs:/mnt/bi-reports/export_full/person_without_project/20220522/')\
.select('_id','document',F.col('info.phones.tags').alias('tag_telefone'), F.col('info.emails.tags').alias('tag_email'))\

listTel= ['skip:hot','skip:alto','skip:medio','skip:baixo','skip:nhot','sem_tags']
listMail= ['skip:hot','skip:alto','skip:medio','skip:baixo','skip:nhot','sem_tags']

for item in range(10):
  for i in listTel:
    if 'sem_tags' in i:
      dfInfo = dfInfo.withColumn('Telefone{0}_{1}'.format(item+1,i),F.when(F.array_contains(F.col('tag_telefone')[item], i).isNull(), True).otherwise(False))
    else:
      dfInfo = dfInfo.withColumn('Telefone{0}_{1}'.format(item+1,i),F.when(F.array_contains(F.col('tag_telefone')[item], i) == True, True).otherwise(False))
      
dfInfo = dfInfo.drop('tag_telefone')

for item in range(3):
  for i in listMail:
    if 'sem_tags' in i:
      dfInfo = dfInfo.withColumn('Email{0}_{1}'.format(item+1,i),F.when(F.array_contains(F.col('tag_email')[item], i).isNull(), True).otherwise(False))
    else:
      dfInfo = dfInfo.withColumn('Email{0}_{1}'.format(item+1,i),F.when(F.array_contains(F.col('tag_email')[item], i) == True, True).otherwise(False))
  
dfInfo = dfInfo.drop('tag_email')

#dfInfo.printSchema()

# COMMAND ----------

dfInfo.write.mode('overwrite').format('delta').save(os.path.join(dir_outputs, 'Info3'))


# COMMAND ----------

spark.sql("OPTIMIZE delta.`/mnt/bi-reports/VALIDACAO-MODELOS-ML/Info3` ZORDER BY (_id)")

# COMMAND ----------

dfInfo = spark.read.format('delta').load(os.path.join(dir_outputs, 'Info3'))

# COMMAND ----------

#16
dfInfo.count()

# COMMAND ----------

#17
dfInfo.count()

# COMMAND ----------

# DBTITLE 1,Divida
dfDebts = spark.read.schema(SCHEMA).json('dbfs:/mnt/bi-reports/export_full/person_without_project/20220522/')\
.select('_id', F.col('debts.tags').alias('tag_divida'), F.col('debts.creditor').alias('creditor'), F.col('debts.dueDate').alias('dueDate'))\
.withColumn('debts',F.arrays_zip('creditor', 'tag_divida', 'dueDate'))\
.select('_id', F.explode('debts').alias('debts'))\
.select('_id', 'debts.tag_divida', 'debts.creditor', F.to_date(F.col('debts.dueDate')[0:10], 'yyyy-M-d').alias('INICIO_DIVIDA'))\

listDiv = ['rank:a','rank:b','rank:c','rank:d','rank:e', 'sem_tags']


for i in listDiv:
  if 'sem_tags' in i:
    dfDebts = dfDebts.withColumn('SkipDivida_{0}'.format(i),F.when(F.array_contains(F.col('tag_divida'), i).isNull(), True).otherwise(False))
  else:
    dfDebts = dfDebts.withColumn('SkipDivida_{0}'.format(i),F.when(F.array_contains(F.col('tag_divida'), i) == True, True).otherwise(False))
    
dfDebts = dfDebts.drop('tag_divida')

dfDebts = dfDebts.distinct()

dfDebts = dfDebts.join(dfDebts.groupBy('_id','creditor').agg(F.min('INICIO_DIVIDA').alias('INICIO_DIVIDA')),on=(['_id','creditor','INICIO_DIVIDA']),how='leftsemi')
#dfDebts = dfDebts.join(dfDebts.groupBy('_id','creditor').agg(F.min('INICIO_DIVIDA').alias('INICIO_DIVIDA')),on='INICIO_DIVIDA',how='leftsemi')\
#          .orderBy(F.col('INICIO_DIVIDA').asc()).dropDuplicates(['_id'])

#dfDebts.printSchema()

# COMMAND ----------

display(dfDebts.filter(F.col('_id')=='96db51f69b2db9efbb80e2b6dd9ff9c0'))

# COMMAND ----------

display(dfDebts.filter(F.col('_id')=='ea8ae4374cde8b9526e671035fe2fdc8'))

# COMMAND ----------

display(dfDebts.filter(F.col('_id')=='ea8ae4374cde8b9526e671035fe2fdc8'))

# COMMAND ----------

display(dfDebts.groupBy('_id','creditor').agg(F.min('INICIO_DIVIDA').alias('INICIO_DIVIDA')).filter(F.col('_id')=='ea8ae4374cde8b9526e671035fe2fdc8'))

# COMMAND ----------

dfDebts.write.mode('overwrite').format('delta').save(os.path.join(dir_outputs, 'Debts'))


# COMMAND ----------

spark.sql("OPTIMIZE delta.`/mnt/bi-reports/VALIDACAO-MODELOS-ML/Debts` ZORDER BY (creditor)")

# COMMAND ----------

dfDebts = spark.read.format('delta').load(os.path.join(dir_outputs, 'Debts'))


# COMMAND ----------

#data 17
dfDebts.count()

# COMMAND ----------

#data 16
dfDebts.count()

# COMMAND ----------

##Escrevendo dividas dos credores

for i in range(len(Credores)):
  print('Processando credor {0}'.format(Credores[i].Credores))
  dfCredor = dfDebts.filter(F.col('creditor') == Credores[i].Credores).orderBy(F.col('_id').asc())
  dfCredor.write.mode('overwrite').format('delta').save(os.path.join(dir_outputs, 'Dividas3', 'Divida_'+Credores[i].Credores))
    
  dir_files = os.path.join(dir_outputs, 'Dividas3', 'Divida_'+Credores[i].Credores)

  dfCredor = spark.read.format('delta').load(os.path.join(dir_files))
    
    #Join dos DFs de Info e Debts do credor
  dfBase = dfInfo.join(dfCredor, on='_id')
    
  dfBase.write.mode('overwrite').parquet(os.path.join(dir_outputs, dir_files+'_parquet'))
  print('--------------------------------')

# COMMAND ----------

##Escrevendo dividas dos credores

for i in range(len(Credores)):
  print('Processando credor {0}'.format(Credores[i].Credores))
  dfCredor = dfDebts.filter(F.col('creditor') == Credores[i].Credores).orderBy(F.col('_id').asc())
  dfCredor.write.mode('overwrite').parquet(os.path.join(dir_outputs, 'Dividas4', 'Divida_'+Credores[i].Credores))

  dir_files = os.path.join(dir_outputs, 'Dividas4', 'Divida_'+Credores[i].Credores)

  dir_parquets=os.listdir('/dbfs'+dir_files)
  arqs=list(filter(lambda x: '.parquet' in x, dir_parquets))
  
  ##Loop que pega cada parte do parquet e faz o join
  for j in range(0,len(arqs)):
    print('    Parte {0} de {1}'.format(j+1,len(arqs)))
    dfCredor = spark.read.parquet(os.path.join(dir_files,arqs[j]))
    
    #Join dos DFs de Info e Debts do credor
    dfBase = dfInfo.join(dfCredor, on='_id')
    
    dfBase.write.parquet(os.path.join(dir_outputs, dir_files, 'parquet', 'part{0}'.format(j)))
  print('--------------------------------')

# COMMAND ----------

dfDebts = spark.read.parquet(os.path.join(dir_outputs, 'Dividas3/Divida_recovery_parquet/'))\
          .orderBy(F.col('INICIO_DIVIDA').asc()).dropDuplicates(['_id']).drop(*['creditor','_id'])


# COMMAND ----------

dfDebts.select('document').count()

# COMMAND ----------

df_to_save.count()

# COMMAND ----------

a = df_to_save.join(dfDebts, df_to_save.Document == dfDebts.document, 'inner').drop('document')
a.count()

# COMMAND ----------

display(df_to_save.join(a, on = 'Document', how = 'leftanti'))

# COMMAND ----------

display(dfDebts.filter(F.col('_id')=='de157d5ab5506b3baa807c18cdc59b01'))

# COMMAND ----------

display(dfInfo.filter(F.col('document')=='00012946362'))

# COMMAND ----------

display(a.groupBy(F.col('Document')).count().filter(F.col('count')>1))

# COMMAND ----------

df_to_save.count()

# COMMAND ----------

display(dfInfo.filter(F.col('document').like('%385279108%')))

# COMMAND ----------

display(a.filter(F.col('INICIO_DIVIDA').isNull()))

# COMMAND ----------

basePaths=dir_outputs+'/Tabelas_Acionamentos_News2/avenida/'
paths=[basePaths+'Analise_modelo_qq_*']
df_to_save=a.withColumn('Separator',F.col('data')[0:7])

print('Salvando particionando no arquivo do Power Bi...')

df_to_save.coalesce(1).write.partitionBy('Separator').mode('overwrite').option("header",True).option("delimiter",";").option("emptyValue",'').csv(dir_outputs+'/arquivos_originais2/TMP/Base acionada Avenida')


for directory in dbutils.fs.ls(dir_outputs+'/arquivos_originais2/TMP/Base acionada Avenida'):
  if 'Separator' in  directory.name:
    for file in dbutils.fs.ls(directory.path):
      if '.csv' in file.name:
        dbutils.fs.cp(file.path, dir_outputs+'/arquivos_originais2/Base acionada Avenida_'+directory.name.split('=')[-1].replace('/','')+'.csv')

df_to_save.coalesce(1).write.mode('overwrite').option("header",True).option("delimiter",";").option("emptyValue",'').csv(dir_outputs+'/arquivos_originais2/TMP/Base acionada Avenida')

for directory in dbutils.fs.ls(dir_outputs+'/arquivos_originais2/TMP/Base acionada Avenida'):
  for file in dbutils.fs.ls(directory.path):
    if '.csv' in file.name:
      dbutils.fs.cp(file.path, dir_outputs+'/arquivos_originais2/Base acionada Avenida_Consolidado.csv')  

      
print('Removendo os demais arquivos...')
  
for removes in dbutils.fs.ls(dir_outputs+'/arquivos_originais2/TMP/Base acionada Avenida'):
  dbutils.fs.rm(removes.path, True)   
dbutils.fs.rm(dir_outputs+'/arquivos_originais2/TMP', True)


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