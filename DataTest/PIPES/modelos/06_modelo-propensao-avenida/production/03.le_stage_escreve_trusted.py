# Databricks notebook source
# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

blob_account_source_ml = "qqdatastoragemain"
blob_container_source_ml = "ml-prd"

mount_blob_storage_key(dbutils,blob_account_source_ml,blob_account_source_ml,'/mnt/ml-prd')

readpath_stage = '/mnt/ml-prd/ml-data/propensaodeal/avenida/stage'
readpath_stage_dbfs = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/avenida/stage'
writepath_trusted = '/mnt/ml-prd/ml-data/propensaodeal/avenida/trusted'
list_readpath = os.listdir(readpath_stage_dbfs)

writepath_sample = '/mnt/ml-prd/ml-data/propensaodeal/avenida/sample'

spark.conf.set('spark.sql.sources.partitionOverwriteMode', 'dynamic')

# COMMAND ----------

# DBTITLE 1,escolha a data, data_escolhida = max(list_readpath) para padrão
data_escolhida = max(list_readpath)

# COMMAND ----------

parquetfilepath = '/mnt/ml-prd/ml-data/propensaodeal/avenida/pre-trusted'+'/'+data_escolhida+'/'
print(parquetfilepath)

# COMMAND ----------

dbutils.fs.ls(parquetfilepath)

# COMMAND ----------

# DBTITLE 1,lendo dataframes para join
df = spark.read.format('parquet').load(readpath_stage+'/'+data_escolhida+'/'+'geral_sem_detalhes.PARQUET').alias('df')
detalhes_clientes_df = spark.read.format('parquet').load(readpath_stage+'/'+data_escolhida+'/'+'detalhes_clientes.PARQUET').alias('detalhes_clientes')

# COMMAND ----------

# DBTITLE 1,join dataframes
df = df.join(detalhes_clientes_df, on = 'ID_PESSOA:ID_DIVIDA', how='left').alias('df')

# COMMAND ----------

display(df)

# COMMAND ----------

# DBTITLE 1,transform datatypes
df = df.withColumn('DOCUMENTO_PESSOA', F.col('DOCUMENTO_PESSOA').cast(T.LongType()))
df = df.withColumn('ID_DIVIDA', F.col('ID_DIVIDA').cast(T.LongType()))
df = df.alias('df')

# COMMAND ----------

# DBTITLE 1,DROP CPF's null
df = df.filter(F.col('DOCUMENTO_PESSOA').isNotNull())

# COMMAND ----------

#from pyspark.sql.functions import lpad
df = df.withColumn("DOCUMENTO_PESSOA", F.lpad(F.col("DOCUMENTO_PESSOA"),11,"0"))

# COMMAND ----------

display(df)

# COMMAND ----------

# DBTITLE 1,dtype para string se for array
dtypes = dict(df.dtypes)
for item in dtypes:
  if 'array' in dtypes[item]:
    df = df.withColumn(item, F.col(item).cast(T.StringType()))

# COMMAND ----------

# DBTITLE 1,escreve csv
df.coalesce(1).write.option('header', 'True').option('sep', ';').csv(writepath_trusted+'/'+data_escolhida+'/'+'tempfile_avenida'+'.csv')

writepath_trusted+'/'+data_escolhida+'/'+'tempfile_avenida'+'.csv'

# COMMAND ----------

# DBTITLE 1,exclui csv tempfile
for file in dbutils.fs.ls(writepath_trusted+'/'+data_escolhida+'/'+'tempfile_avenida'+'.csv'):
  if file.name.split('.')[-1] == 'csv':
    dbutils.fs.cp(file.path, writepath_trusted+'/'+data_escolhida+'/'+'trustedFile_avenida'+'.csv')
  else:
    dbutils.fs.rm(file.path, True)
dbutils.fs.rm(writepath_trusted+'/'+data_escolhida+'/'+'tempfile_avenida'+'.csv', True)