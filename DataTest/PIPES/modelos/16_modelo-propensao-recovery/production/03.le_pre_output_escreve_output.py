# Databricks notebook source
# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

import os
import datetime

# COMMAND ----------

pre_outputpath = '/mnt/ml-prd/ml-data/propensaodeal/recovery/output'
pre_outputpath_dbfs = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/recovery/output'

santander_output = '/mnt/ml-prd/ml-data/propensaodeal/santander/output'

# COMMAND ----------

for file in os.listdir(pre_outputpath_dbfs):
  print ('file:',file)
  
date = file.split('.')[0]
date = (date[::-1][0:8])[::-1]
date = date[0:4]+'-'+date[4:6]+'-'+date[6:8]
print ('date:',date)

createdAt = datetime.datetime.today().date()
print ('createdAt:', createdAt)

# COMMAND ----------

for file in os.listdir(pre_outputpath_dbfs):
  print ('file:',file)
  
createdAt = datetime.datetime.today().date()
print ('createdAt:', createdAt)

# COMMAND ----------

output = spark.read.format('csv').option('header','True').load(os.path.join(pre_outputpath, file))
output = output.drop('_c0')
output = output.withColumn('Document', F.lpad(F.col("document"),11,'0'))
output = output.groupBy(F.col('Document')).agg(F.max(F.col('GH')), F.max(F.col('P_1')), F.avg(F.col('P_1')))
output = output.withColumn('Provider', F.lit('qq_recovery_propensity_v3'))
output = output.withColumn('Date', F.lit(date))
output = output.withColumn('CreatedAt', F.lit(createdAt))
output = changeColumnNames(output, ['Document','Score','ScoreValue','ScoreAvg','Provider','Date','CreatedAt'])

# COMMAND ----------

display(output)

# COMMAND ----------

output.coalesce(1).write.mode('overwrite').options(header='True', delimiter=';').csv(pre_outputpath+'/tmp')

for files in dbutils.fs.ls(pre_outputpath+'/tmp'):
  if files.name.split('.')[-1] == 'csv':
    dbutils.fs.cp(files.path, pre_outputpath+'/recovery_model_to_production_'+str(createdAt)+'.csv')
    dbutils.fs.rm(pre_outputpath+'/tmp', recurse=True)
    dbutils.fs.rm(pre_outputpath+'/'+file, recurse=True)

# COMMAND ----------

dbutils.fs.cp(pre_outputpath+'/recovery_model_to_production_'+str(createdAt)+'.csv', santander_output+'/recovery_model_to_production_'+str(createdAt)+'.csv')