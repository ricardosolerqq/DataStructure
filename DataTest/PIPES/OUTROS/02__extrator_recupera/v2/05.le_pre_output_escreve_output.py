# Databricks notebook source
# MAGIC %run "/pipe_modelos/extrator_recupera/v2/00.le_regras_disponibiliza_variaveis_v2"

# COMMAND ----------

import os
import datetime

# COMMAND ----------

dbutils.widgets.text('credor', 'fort_brasil')
credor = dbutils.widgets.get('credor')

credor = Credor(credor)

# COMMAND ----------

santander_output = '/mnt/ml-prd/ml-data/propensaodeal/santander/output'

# COMMAND ----------

for file in os.listdir(credor.caminho_pre_output_dbfs):
  print ('file:',file)
  
date = file.split('_')[-1].split('.')[0]
date = datetime.date(int(date[:4]), int(date[4:6]), int(date[6:]))
print ('date:',date)

createdAt = datetime.datetime.today().date()
createdAt_datetime = datetime.datetime.today()
print ('createdAt:', createdAt)

# COMMAND ----------

output = spark.read.parquet(os.path.join(credor.caminho_pre_output, file))
output = output.select('DOCUMENTO', 'P_1', 'GH')
output = output.withColumn('Document', F.lpad(F.col("DOCUMENTO"),11,'0'))
output = output.groupBy(F.col('Document')).agg(F.max(F.col('GH')), F.max(F.col('P_1')), F.avg(F.col('P_1')))
if dbutils.widgets.get('credor') == 'fort_brasil':
  output = output.withColumn('Provider', F.lit('qq_'+credor.nome+'_propensity_v2'))
else:
  output = output.withColumn('Provider', F.lit('qq_'+credor.nome+'_propensity_v1'))
output = output.withColumn('Date', F.lit(date))
output = output.withColumn('CreatedAt', F.lit(createdAt))
output = changeColumnNames(output, ['Document','Score','ScoreValue','ScoreAvg','Provider','Date','CreatedAt'])

# COMMAND ----------

output.show(5,False)

# COMMAND ----------

# DBTITLE 1,escrevendo arquivo no output
output.coalesce(1)\
   .write.format("com.databricks.spark.csv")\
   .option('overwrite', 'True')\
   .option("sep",";")\
   .option("header", "true")\
   .save(os.path.join(credor.caminho_pre_output, 'pre_output.csv'))

# COMMAND ----------

for arq in dbutils.fs.ls(os.path.join(credor.caminho_pre_output, 'pre_output.csv')):
  if arq.name.split('.')[-1] == 'csv':
    print (arq.path)
    dbutils.fs.cp(arq.path, os.path.join(credor.caminho_output,str(createdAt),'qq_'+credor.nome+'_propensity_v1'+"_"+str(createdAt_datetime).replace(' ', '_').split('.')[0].replace(":",'_')+'_output.csv'))

# COMMAND ----------

# DBTITLE 1,removendo arquivos de apoio
dbutils.fs.rm(os.path.join(credor.caminho_pre_output,'pre_output.csv'), True)
dbutils.fs.rm(os.path.join(credor.caminho_pre_output, file), True)
dbutils.fs.rm(credor.caminho_pre_output, True)

# COMMAND ----------

# DBTITLE 1,copiando para pasta SANTANDER OUTPUT
dbutils.fs.cp(os.path.join(credor.caminho_output,str(createdAt),'qq_'+credor.nome+'_propensity_v1'+"_"+str(createdAt_datetime).replace(' ', '_').split('.')[0].replace(":",'_')+'_output.csv'), os.path.join(santander_output+'/'+credor.nome+'model_to_production_'+str(createdAt)+'.csv'))
santander_output+'/'+credor.nome+'_model_to_production_'+str(createdAt_datetime).replace(' ', '_').split('.')[0].replace(":",'_')+'_output'+'.csv'


# COMMAND ----------

dbutils.notebook.exit('DONE')