# Databricks notebook source
# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

import os
import datetime

# COMMAND ----------

blob_account_source_ml = "qqdatastoragemain"
blob_container_source_ml = "ml-prd"

blob_account_source_lake = "saqueroquitar"
blob_container_source_lake = "trusted"

mount_blob_storage_key(dbutils,blob_account_source_ml,blob_container_source_ml,'/mnt/ml-prd')
mount_blob_storage_key(dbutils,blob_account_source_lake,blob_container_source_lake,'/mnt/ml-prd')

pre_outputpath = '/mnt/ml-prd/ml-data/propensaodeal/credsystem/output'
pre_outputpath_dbfs = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/credsystem/output'

santander_output = '/mnt/ml-prd/ml-data/propensaodeal/santander/output'

# COMMAND ----------

os.listdir(pre_outputpath_dbfs)

# COMMAND ----------

for file in os.listdir(pre_outputpath_dbfs):
  print ('file:',file)
  
date = file.split(':')[1]
date = date.split('.')[0]
date = date.split('_')[2]
date = date[0:4]+'-'+date[4:6]+'-'+date[6:8]
print ('date:',date)

createdAt = datetime.datetime.today().date()
print ('createdAt:', createdAt)

# COMMAND ----------

output = spark.read.format('csv').option('header','True').load(os.path.join(pre_outputpath, file))
output = output.select('CPF', 'P_1', 'GH')

# COMMAND ----------

output = spark.read.format('csv').option('header','True').load(os.path.join(pre_outputpath, file))
output = output.drop('_c0')
output = output.withColumn('DOCUMENTO:ID_DIVIDA', F.split("CPF", ':'))
output = output.withColumn('Document', F.lpad(F.col("CPF"),11,'0'))
output = output.groupBy(F.col('Document')).agg(F.max(F.col('GH')), F.max(F.col('P_1')), F.avg(F.col('P_1')))
output = output.withColumn('Provider', F.lit('qq_credsystem_propensity_v1'))
output = output.withColumn('Date', F.lit(date))
output = output.withColumn('CreatedAt', F.lit(createdAt))
output = changeColumnNames(output, ['Document','Score','ScoreValue','ScoreAvg','Provider','Date','CreatedAt'])

# COMMAND ----------

display(output)

# COMMAND ----------

# DBTITLE 1,escrevendo arquivo no output
output.coalesce(1)\
   .write.format("com.databricks.spark.csv")\
   .option("sep",";")\
   .option("header", "true")\
   .save(os.path.join(pre_outputpath, 'temp_output.csv'))

# COMMAND ----------

for arq in dbutils.fs.ls(os.path.join(pre_outputpath, 'temp_output.csv')):
  if arq.name.split('.')[-1] == 'csv':
    print (arq.path)
    dbutils.fs.cp(arq.path, os.path.join(pre_outputpath+'/credsystem_model_to_production_'+str(createdAt)+'.csv'))

# COMMAND ----------

# DBTITLE 1,removendo arquivos de apoio
dbutils.fs.rm(os.path.join(pre_outputpath,'temp_output.csv'), True)
dbutils.fs.rm(os.path.join(pre_outputpath, file), True)

# COMMAND ----------

# DBTITLE 1,copiando para pasta SANTANDER OUTPUT
dbutils.fs.cp(os.path.join(pre_outputpath+'/credsystem_model_to_production_'+str(createdAt)+'.csv'), os.path.join(santander_output+'/credsystem_model_to_production_'+str(createdAt)+'.csv'))
santander_output+'/credsystem_model_to_production_'+str(createdAt)+'.csv'

# COMMAND ----------

santander_output+'/credsystem_model_to_production_'+str(createdAt)+'.csv'