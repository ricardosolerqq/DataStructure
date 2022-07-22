# Databricks notebook source
# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

blob_account_source_ml = "qqdatastoragemain"
blob_container_source_ml = "ml-prd"

mount_blob_storage_key(dbutils,blob_account_source_ml,blob_account_source_ml,'/mnt/ml-prd')

readpath_stage = '/mnt/ml-prd/ml-data/propensaodeal/portocred/stage'
readpath_stage_dbfs = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/portocred/stage'
writepath_trusted = '/mnt/ml-prd/ml-data/propensaodeal/portocred/trusted'
list_readpath = os.listdir(readpath_stage_dbfs)

writepath_sample = '/mnt/ml-prd/ml-data/propensaodeal/portocred/sample'

spark.conf.set('spark.sql.sources.partitionOverwriteMode', 'dynamic')

# COMMAND ----------

data_escolhida = max(list_readpath)
parquetfilepath = '/mnt/ml-prd/ml-data/propensaodeal/portocred/pre-trusted'+'/'+data_escolhida+'/'
print(parquetfilepath)

# COMMAND ----------

# DBTITLE 1,lendo dataframes para join
df = spark.read.format('parquet').load(readpath_stage+'/'+data_escolhida+'/'+'geral_sem_detalhes.PARQUET').alias('df')
detalhes_clientes_df = spark.read.format('parquet').load(readpath_stage+'/'+data_escolhida+'/'+'detalhes_clientes.PARQUET').alias('detalhes_clientes')
detalhes_contratos_df = spark.read.format('parquet').load(readpath_stage+'/'+data_escolhida+'/'+'detalhes_contratos.PARQUET').alias('detalhes_contratos')
detalhes_dividas_df = spark.read.format('parquet').load(readpath_stage+'/'+data_escolhida+'/'+'detalhes_dividas.PARQUET').alias('detalhes_dividas')

# COMMAND ----------

# DBTITLE 1,join dataframes
df = df.join(detalhes_clientes_df, on = 'ID_PESSOA:ID_DIVIDA', how='left').alias('df')
df = df.join(detalhes_contratos_df, on = 'ID_PESSOA:ID_DIVIDA', how='left').alias('df')
df = df.join(detalhes_dividas_df, on = 'ID_PESSOA:ID_DIVIDA', how='left').alias('df')

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

# DBTITLE 1,escreve variavel resposta
#df = escreve_variavel_resposta_acordo(df, 'portocred', datetime(2021,10,28), 'DOCUMENTO_PESSOA', drop_null=True)

# COMMAND ----------

# DBTITLE 1,dtype para string se for array
dtypes = dict(df.dtypes)
for item in dtypes:
  if 'array' in dtypes[item]:
    df = df.withColumn(item, F.col(item).cast(T.StringType()))

# COMMAND ----------

# DBTITLE 1,escreve parquet
#df.write.mode('overwrite').parquet(writepath_trusted+'/'+data_escolhida+'/'+'trustedFile_portocred'+'.PARQUET')
#df.write.option("header","true").option("delimiter",";").csv(writepath_trusted+'/'+data_escolhida+'/'+'trustedFile_portocred'+'.csv')
df.coalesce(1).write.option('header', 'True').option('sep', ';').csv(writepath_trusted+'/'+data_escolhida+'/'+'tempfile_portocred'+'.csv')

writepath_trusted+'/'+data_escolhida+'/'+'tempfile_portocred'+'.csv'

# COMMAND ----------

for file in dbutils.fs.ls(writepath_trusted+'/'+data_escolhida+'/'+'tempfile_portocred'+'.csv'):
  if file.name.split('.')[-1] == 'csv':
    dbutils.fs.cp(file.path, writepath_trusted+'/'+data_escolhida+'/'+'trustedFile_portocred'+'.csv')
  else:
    dbutils.fs.rm(file.path, True)
dbutils.fs.rm(writepath_trusted+'/'+data_escolhida+'/'+'tempfile_portocred'+'.csv', True)

# COMMAND ----------

# DBTITLE 1,sample
#df_representativo, df_aleatorio = gera_sample(df, max_sample = 100000)

#samples = {'df_representativo':df_representativo, 'df_aleatorio':df_aleatorio}

#print ('escrevendo...')
#for sample in samples:
#  samples[sample].write.csv(writepath_sample+'/'+data_escolhida+'/temp_sample.csv', header=True)
#  for file in dbutils.fs.ls(writepath_sample+'/'+data_escolhida+'/temp_sample.csv'):
#    if file.name.split('.')[-1] == 'csv':
#      dbutils.fs.cp(file.path, writepath_sample+'/'+data_escolhida+'/'+sample+'.csv')
#    else:
#      dbutils.fs.rm(file.path, True)
#  dbutils.fs.rm(writepath_sample+'/'+data_escolhida+'/temp_sample.csv', True)

# COMMAND ----------

display(spark.read.option('sep', ';')\
        .option('header', 'True')\
        .csv("/mnt/ml-prd/ml-data/propensaodeal/portocred/trusted/"+data_escolhida+"/trustedFile_portocred.csv"))

# COMMAND ----------

spark.read.option('sep', ';')\
        .option('header', 'True')\
        .csv("/mnt/ml-prd/ml-data/propensaodeal/portocred/trusted/"+data_escolhida+"/trustedFile_portocred.csv")\
        .select(F.col('DOCUMENTO_PESSOA')).dropDuplicates().count()