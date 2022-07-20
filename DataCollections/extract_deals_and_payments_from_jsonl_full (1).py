# Databricks notebook source
import time
time.sleep(300)

# COMMAND ----------

#IMPORTS 
import json
import string as s
import pandas as pd
from datetime import datetime
from datetime import timedelta
import os.path,inspect, re 
from pyspark.sql import types as T
from pyspark.sql import functions as F
from pyspark.sql import functions
from pyspark.sql.window import Window
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark import SparkConf
from pyspark import SparkContext
import random as r
import pyspark
from types import SimpleNamespace
import sys
import pathlib
import time
import os
import zipfile
import concurrent

#SPARK SETTINGS

storage_account_name="qqdatastoragemain",
storage_account_access_key="mRI1Hf9yj0LARoe/BCXg49HjYkaYxZ5ERnpgiA7rOJPQ9YS633rEKh9b8kwepJInz7dDJSH99pQ3RM4uKMbmvw==",
spark.conf.set("fs.azure.account.key.qqdatastoragemain.blob.core.windows.net",  str(storage_account_access_key))
spark.conf.set('spark.sql.sources.partitionOverwriteMode', 'dynamic')
spark.conf.set('spark.sql.legacy.avro.datetimeRebaseModeInWrite', 'CORRECTED')
spark.conf.set("spark.sql.session.timeZone", "America/Sao_Paulo")
spark.conf.set('spark.sql.execution.arrow.pyspark.enabled' , False)
spark.conf.set('spark.sql.caseSensitive', True)

# COMMAND ----------

# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

def getSchema(sampleSize = 50000, mode="local"):
  DATA_PATH = "/dbfs/SCHEMAS/DATA_SCHEMA.json" 
  SQL_DF_FORMAT = "com.mongodb.spark.sql.DefaultSource"

  if mode =="local":
    with open(DATA_PATH) as f:
      DATA_SCHEMA = T.StructType.fromJson(json.load(f))

  else:
    DATA = spark.read\
    .format(SQL_DF_FORMAT)\
    .option('spark.mongodb.input.sampleSize', sampleSize)\
    .option("database", "qq")\
    .option("spark.mongodb.input.collection", "col_person")\
    .option("badRecordsPath", "/tmp/badRecordsPath")\
    .load().schema

    with open("DATA_SCHEMA.json", "w") as f:
      json.dump(DATA.jsonValue(), f)

    with open("DATA_SCHEMA.json") as f:
      DATA_SCHEMA = T.StructType.fromJson(json.load(f))

  return DATA_SCHEMA

DATA_SCHEMA = getSchema()

# COMMAND ----------

# DBTITLE 1,filepath
selected_path = "DEALS_REPORT"

caminho_origin = '/mnt/bi-reports/export_full/person'
caminho_origin_dbfs = '/dbfs/mnt/bi-reports/export_full/person'

caminho_temp = '/mnt/bi-reports/' + str(selected_path) + '/temp'
caminho_temp_dbfs = '/dbfs/mnt/bi-reports/' + str(selected_path) + '/temp'

caminho_unzip = '/mnt/bi-reports/' + str(selected_path) + '/unzip'
caminho_unzip_dbfs = '/dbfs/mnt/bi-reports/' + str(selected_path) + '/unzip'

caminho_stage = '/mnt/bi-reports/' + str(selected_path) + '/stage'
caminho_stage_dbfs = '/dbfs/mnt/bi-reports/' + str(selected_path) + '/stage'

caminho_trusted = '/mnt/bi-reports/' + str(selected_path) + '/trusted'
caminho_trusted_dbfs = '/dbfs/mnt/bi-reports/' + str(selected_path) + '/trusted'

path_list = [
        ["caminho_origin", caminho_origin], 
        ["caminho_origin_dbfs", caminho_origin_dbfs], 
        ["caminho_temp", caminho_temp], 
        ["caminho_temp_dbfs", caminho_temp_dbfs], 
        ["caminho_unzip", caminho_unzip], 
        ["caminho_unzip_dbfs", caminho_unzip_dbfs], 
        ["caminho_stage", caminho_stage], 
        ["caminho_stage_dbfs", caminho_stage_dbfs], 
        ["caminho_trusted", caminho_trusted], 
        ["caminho_trusted_dbfs", caminho_trusted_dbfs]
            ]
caminho_deltaControle = '/mnt/bi-reports/controleDelta_deals_payments'

# COMMAND ----------

# DBTITLE 1,select newest folder
date = datetime.today() - timedelta(days=1) 
dateToday = datetime.today()
dateToday = str(dateToday.year)+str(dateToday.month).zfill(2)+str(dateToday.day).zfill(2)
ArqDate = str(date.year)+str(date.month).zfill(2)+str(date.day).zfill(2)
date2 = str(date.year)+'-'+str(date.month).zfill(2)+'-'+str(date.day).zfill(2)


pastas = []
for folder in os.listdir(caminho_origin_dbfs):
  try:
    pastas.append(int(folder))
  except:
    pass
  
max_folder = str(max(pastas))

#if max_folder == dateToday:
#  max_folder = ArqDate
  
max_folder

# COMMAND ----------

# DBTITLE 1,exclude old files
paths_to_exclude = [caminho_temp,caminho_unzip, caminho_stage]

for path in paths_to_exclude:
  try:
    for file in dbutils.fs.ls(path):
      dbutils.fs.rm(file.path, True)
      print (file.name, 'excluido')
  except:
    pass
  try:
    dbutils.fs.rm(path)
    print (path, 'excluido')
  except:
    pass

# COMMAND ----------

# DBTITLE 1,deszipando sem threads e então garantindo que todos os arquivos tenham sido deszipados
arquivos_origin_ref = [file.name for file in dbutils.fs.ls(os.path.join(caminho_origin, max_folder))]

qtd_arquivos = len(arquivos_origin_ref)
  
for file in arquivos_origin_ref:
  with zipfile.ZipFile(os.path.join(caminho_origin_dbfs, max_folder, file), "r") as arquivozipado:
      try:
        arquivozipado.extractall(path = caminho_unzip_dbfs)
      except Exception as e:
        print (e)
        
arquivos_unzip_ref = [file.name for file in dbutils.fs.ls(caminho_unzip)]
print ('deszipados', len(arquivos_unzip_ref), 'arquivos com sucesso')

# COMMAND ----------

# DBTITLE 1,verificando quantidade da pasta unzip, deve ser 1:1 para com arquivos jsonl
if len(arquivos_unzip_ref) == qtd_arquivos:
  print (len(arquivos_unzip_ref), '==', qtd_arquivos)
else:
   raise Exception(len(arquivos_unzip_ref), '!=', qtd_arquivos)

# COMMAND ----------

def size_in_mb(size):
  size_in_mb = round(size / 1048576, 1)
  return size_in_mb

files_size = {}

for file in dbutils.fs.ls(caminho_unzip):
  files_size.update({file.path:size_in_mb(file.size)})

# COMMAND ----------

def select_files(files_size, megabytes_threshold = 28000): #max_memory em gigas
  files = list(files_size)
  roadmap_matrix = []
  inner_roadmap_matrix = []
  memory_per_file_list = 0
  i = 0 #indice de file
  while i <= len(files)-1:
    if memory_per_file_list + files_size[files[i]] < megabytes_threshold:
      memory_per_file_list = memory_per_file_list + files_size[files[i]]
      inner_roadmap_matrix.append(files[i])
      i = i+1
    else:
      roadmap_matrix.append(inner_roadmap_matrix)
      inner_roadmap_matrix = []
      memory_per_file_list = 0
  roadmap_matrix.append(inner_roadmap_matrix)
  ### realizando verificação ###
  c = 0
  for subarray in roadmap_matrix:
    for item in subarray:
      c = c+1
  if c != len(files):
    raise Exception ('geração de listas para processamento não utilizou todos os arquivos!',  str(len(files)), '!=', c)
  
  return roadmap_matrix

roadmap_matrix = select_files(files_size)

# COMMAND ----------

i = 1

def model_arrays(df):
  arrays = ['deals', 'payments']
  for array in arrays:
    if array == 'deals':
      dfDeals = df.withColumn("deals",F.explode(df.deals))\
                .withColumn("creditor", F.col("deals").getItem("creditor")).alias("creditor")\
                .withColumn("DEALS_ID", F.col("deals").getItem("_id"))\
                .filter(F.col('DEALS_ID').isNotNull())\
                .withColumn("installmentValue",F.col("deals").getItem("installmentValue"))\
                .withColumn("createdAt",F.col("deals").getItem("createdAt"))\
                .withColumn("totalAmount",F.col("deals").getItem("totalAmount"))\
                .withColumn("totalInstallments",F.col("deals").getItem("totalInstallments"))\
                .withColumn("upfront",F.col("deals").getItem("upfront"))\
                .withColumn("status",F.col("deals").getItem("status"))\
                .withColumn("offer",F.col("deals").getItem("offer"))\
                .withColumn("tracking", F.col("offer").getItem("tracking"))\
                .withColumn("channel", F.col("tracking").getItem("channel"))\
                .withColumn("origin", F.col("tracking").getItem("origin"))\
                .withColumn("debts", F.col("offer").getItem("debts"))\
                .withColumn("dueDate", F.col("debts").getItem("dueDate"))\
                .withColumn("portfolio", F.col("debts").getItem("portfolio"))\
                .withColumn("product", F.col("debts").getItem("product"))\
                .withColumn("originalAmount", F.col("debts").getItem("originalAmount"))\
                .withColumn('originalAmount', F.aggregate("originalAmount", F.lit(0.0), lambda acc, x: acc + x))\
                .withColumn("dueDate", F.array_min(F.col("dueDate")))\
                .withColumn("aging", F.datediff(F.col("createdAt"), F.col("dueDate")))\
                .withColumn("createdAt", F.split(F.col("createdAt").cast(T.StringType()), " ").getItem(0))\
                .withColumn("dueDate", F.split(F.col("dueDate").cast(T.StringType()), " ").getItem(0))\
                .withColumn('addresses' , F.array_min('info.addresses').alias('addresses'))\
                .select("_id", "DEALS_ID",'documentType',"creditor","dueDate", "createdAt", "channel", "totalAmount", "totalInstallments",\
                        "upfront", "status", "installmentValue", "product", "portfolio","origin","aging","originalAmount", 'addresses.address',  'addresses.city', 'addresses.complement', 'addresses.country', 'addresses.neighborhood', 'addresses.number', 'addresses.state', 'addresses.type', 'addresses.updatedAt', 'addresses.zipcode' ,'info.birthDate', 'info.gender', 'document', df.documentType.alias('tipoDocumento'))
                
    else:
      dfPayments = df.withColumn("installments",F.explode(df.installments))\
              .withColumn("installmentID", F.col("installments").getItem("_id"))\
              .filter(F.col('installmentID').isNotNull())\
              .withColumn("creditor", F.col("installments").getItem("creditor")).alias("creditor")\
              .withColumn('dealID', F.col('installments').getItem('dealID'))\
              .withColumn('installment', F.col('installments').getItem('installment'))\
              .withColumn("installmentAmount",F.col("installments").getItem("installmentAmount"))\
              .withColumn("status",F.col("installments").getItem("status"))\
              .filter(F.col('status')=='paid')\
              .withColumn('payment', F.col('installments').getItem('payment'))\
              .withColumn('paidAmount', F.col('payment').getItem('paidAmount'))\
              .withColumn('paidAt', F.col('payment').getItem('paidAt'))\
              .withColumn('dueAt', F.col('installments').getItem('dueAt'))\
              .withColumn('addresses' , F.array_min('info.addresses').alias('addresses'))\
              .select("_id", "installmentID", 'installment', 'dealID', "creditor", "dueAt",'status', 'installmentAmount', 'paidAt', 'paidAmount', '_t', 'addresses.address',  'addresses.city', 'addresses.complement', 'addresses.country', 'addresses.neighborhood', 'addresses.number', 'addresses.state', 'addresses.type', 'addresses.updatedAt', 'addresses.zipcode' ,'info.birthDate', 'info.gender', 'document', df.documentType.alias('tipoDocumento'))\

        
  return dfDeals, dfPayments

for subarray in roadmap_matrix:
  print (len(subarray),"/",qtd_arquivos)
  
  df = spark.read.schema(DATA_SCHEMA).json(subarray)

  dfDeals, dfPayments = model_arrays(df)

  dfDeals.distinct().coalesce(1).write.mode("overwrite").parquet(os.path.join(caminho_stage, 'deals|'+file.name.split(".")[0]))
  dfPayments.distinct().coalesce(1).write.mode("overwrite").parquet(os.path.join(caminho_stage, 'payments|'+file.name.split(".")[0]))

  print ("\tstage escrita!")
  i = i+1

# COMMAND ----------

arrays = ['deals', 'payments']
dict_arquivos_por_array = {}
dict_united_arrays = {}

for folder in os.listdir(caminho_stage_dbfs):
  for array in arrays:
    if folder.split('|')[0] == array:
      try:
        dict_arquivos_por_array[array].append(folder)
      except:
        dict_arquivos_por_array.update({array:[folder]})
        
for array in dict_arquivos_por_array:
  primeiraLeitura = True
  for file in dict_arquivos_por_array[array]:
    if primeiraLeitura:
      df = spark.read.parquet(os.path.join(caminho_stage,file))
      primeiraLeitura = False
    else:
      df_to_union = spark.read.parquet(os.path.join(caminho_stage, file))
      df = df.union(df_to_union)
  dict_united_arrays.update({array:df})

# COMMAND ----------

# DBTITLE 1,realiza controle e escreve os arquivos
def createDeltaControle(horario_execucao, array, df_count, writeFlag, caminho_deltaControle):
  print ("criando controle delta")
  controleDelta = spark.createDataFrame([[horario_execucao, array, df_count, writeFlag]]).withColumnRenamed('_1', 'date').withColumnRenamed('_2', 'array').withColumnRenamed('_3', 'count').withColumnRenamed('_4', 'WRITTEN')
  controleDelta.write.format('delta').save(os.path.join(caminho_deltaControle, 'deltaControle.PARQUET'))
  
def updateDeltaControle(previousDelta_deltaTable, controleDelta, array):
  previousDelta_deltaTable.alias('previous').merge(controleDelta.alias('atual'), 'previous.date = atual.date').whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()
  
def getDeltaTable(caminho_deltaControle):
  previousDelta_deltaTable = DeltaTable.forPath(spark, os.path.join(caminho_deltaControle, 'deltaControle.PARQUET'))
  return previousDelta_deltaTable

from delta.tables import *

arrays_raise_exception = {}

for array in arrays:
  horario_execucao = datetime.now()
  print ('array:', array)
  df = dict_united_arrays[array]
  df_count = df.count()
  print ("current_count:", df_count)
  try:
    previousDelta = spark.read.format('delta').load(os.path.join(caminho_deltaControle, 'deltaControle.PARQUET'))
    previousDelta_deltaTable = getDeltaTable(caminho_deltaControle)
    try:
      last_count = previousDelta.filter(F.col('array')==array).filter(F.col('WRITTEN')==True).orderBy(F.desc(F.col('date'))).limit(1).select('count').rdd.map(lambda Row:Row[0]).collect()[0]
    except Exception as e:
      print (e)
      last_count = 0
    
  except Exception as e:
    print (e)
    writeFlag = True
    createDeltaControle(horario_execucao, array, df_count, writeFlag, caminho_deltaControle)
    previousDelta_deltaTable = getDeltaTable(caminho_deltaControle)
    last_count = df_count
  
  print (last_count, df_count)
  if last_count > df_count:
    arrays_raise_exception.update({array:[last_count, df_count]})
    writeFlag = False
  else:
    if array == 'deals':
      df = df.filter(F.col('createdAt')<=date2)
    else:
      df = df.withColumn('pay_at', F.when(F.col('paidAt').isNull(),F.col('dueAt')).otherwise(F.col('paidAt')))
      df = df.filter(F.col('pay_at')[0:10]<=date2)
    df.coalesce(1).write.mode('overwrite').parquet('/mnt/bi-reports/FINAL/'+array+'/'+array+'_updated_report.parquet/tmp')

    for files in dbutils.fs.ls('/mnt/bi-reports/FINAL/'+array+'/'+array+'_updated_report.parquet/tmp'):
      if files.name.split('.')[-1] == 'parquet':
        dbutils.fs.rm('/mnt/bi-reports/FINAL/'+array+'/'+array+'_updated_report.parquet/F_'+array.upper()+'.parquet', recurse=True)
        dbutils.fs.cp(files.path, '/mnt/bi-reports/FINAL/'+array+'/'+array+'_updated_report.parquet/F_'+array.upper()+'.parquet')
        dbutils.fs.rm('/mnt/s3_qq-data-bi-us/reports/FINAL/'+array+'/'+array+'_updated_report.parquet/F_'+array.upper()+'.parquet', recurse=True)
        dbutils.fs.cp(files.path, '/mnt/s3_qq-data-bi-us/reports/FINAL/'+array+'/'+array+'_updated_report.parquet/F_'+array.upper()+'.parquet')
        dbutils.fs.rm('/mnt/bi-reports/FINAL/'+array+'/'+array+'_updated_report.parquet/tmp', recurse=True)  
    print (array, 'escrita.')
    writeFlag = True
        
  controleDelta = spark.createDataFrame([[horario_execucao, array, df_count, writeFlag]]).withColumnRenamed('_1', 'date').withColumnRenamed('_2', 'array').withColumnRenamed('_3', 'count').withColumnRenamed('_4', 'WRITTEN')
  updateDeltaControle (previousDelta_deltaTable, controleDelta, array)
if len(arrays_raise_exception)>0:
  raise Exception ('houveram problemas na contagem das arrays', str(arrays_raise_exception))

# COMMAND ----------

deltaAtual = spark.read.format('delta').load(os.path.join(caminho_deltaControle, 'deltaControle.PARQUET')).orderBy(F.desc('date'))
display(deltaAtual)

#deltaAtual = previousDelta.withColumn('WRITTEN', F.when(F.col('date')[0:10]=='2022-07-18', False).otherwise(F.col('WRITTEN')))
#deltaAtual.write.mode('overwrite').format('delta').save(os.path.join(caminho_deltaControle, 'deltaControle.PARQUET'))

# COMMAND ----------

display(dfDeals.groupBy(F.col('createdAt').alias('Dia')).count().orderBy(F.col('Dia').desc()))

# COMMAND ----------

display(dfPayments.groupBy(F.col('paidAt')[0:10].alias('Dia')).count().orderBy(F.col('Dia').desc()))

# COMMAND ----------

dfPayments.distinct().filter(F.col('paidAt').isNull()).count()
