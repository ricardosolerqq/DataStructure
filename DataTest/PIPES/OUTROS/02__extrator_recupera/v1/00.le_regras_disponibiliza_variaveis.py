# Databricks notebook source
# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

blob_account_source_prd = "qqprd"
blob_account_source_ml = "qqdatastoragemain"
blob_container_source_prd = "qq-integrator"
blob_container_source_ml = "ml-prd"

mount_blob_storage_key(dbutils,blob_account_source_prd,blob_account_source_prd,'/mnt/qq-integrator')
mount_blob_storage_key(dbutils,blob_account_source_ml,blob_account_source_ml,'/mnt/ml-prd')

caminho_regras = '/mnt/ml-prd/extrator_recupera/regras/'
caminho_regras_dbfs = '/dbfs/mnt/ml-prd/extrator_recupera/regras/'

# COMMAND ----------

def obtem_variaveis_caminho(credor):
  prefix = "etl/"+credor+"/processed"

  caminho_base = '/mnt/qq-integrator/etl/'+credor+'/processed'
  caminho_base_dbfs = '/dbfs/mnt//qq-integrator/etl/'+credor+'/processed/'
  caminho_raw = '/mnt/ml-prd/ml-data/propensaodeal/'+credor+'/raw'
  caminho_raw_dbfs = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/'+credor+'/raw'
  caminho_trusted = '/mnt/ml-prd/ml-data/propensaodeal/'+credor+'/trusted'
  caminho_trusted_dbfs = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/'+credor+'/trusted'
  caminho_joined_trusted = '/mnt/ml-prd/ml-data/propensaodeal/'+credor+'/joined_trusted'
  caminho_joined_trusted_dbfs = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/'+credor+'/joined_trusted'
  caminho_sample = '/mnt/ml-prd/ml-data/propensaodeal/'+credor+'/sample'
  caminho_sample_dbfs = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/'+credor+'/sample'
  
  return caminho_base,caminho_base_dbfs,caminho_raw,caminho_raw_dbfs,caminho_trusted,caminho_trusted_dbfs,caminho_joined_trusted,caminho_joined_trusted_dbfs,caminho_sample,caminho_sample_dbfs

# COMMAND ----------

df_regras = spark.read.option("encoding", "UTF-8").option('header', 'True').csv(caminho_regras)
df_regras = df_regras.withColumn('arquivo', F.input_file_name())
df_regras = df_regras.withColumn('arquivo',F.element_at(F.split(F.col('arquivo'), '/'),-1))
df_regras = df_regras.withColumn('arquivo',F.element_at(F.split(F.col('arquivo'), '.csv'),1))
df_regras = df_regras.withColumn('arquivo',F.element_at(F.split(F.col('arquivo'), 'Registro_'),2))
df_regras = df_regras.filter(~F.col('arquivo').isin(['Trailer', 'Header']))
df_regras = changeColumnNames(df_regras, ['nome', 'inicio', 'fim', 'desc', 'tipo'])
df_regras = df_regras.withColumn('tipo', F.lower(F.regexp_replace(F.col('tipo'), '\W+', '')))

# COMMAND ----------

df_regras_tipo_reg_interface = df_regras.filter(F.col('Nome').isin(['tip_reg', 'tip_inter']))
df_regras_tipo_reg_interface = df_regras_tipo_reg_interface.withColumn('desc', F.substring(F.col('desc'),-4, 1))
df_regras_tipo_reg_interface = df_regras_tipo_reg_interface.withColumn('tip_reg', F.when(F.col('Nome')=='tip_reg', F.col('desc')).otherwise(None))
df_regras_tipo_reg_interface = df_regras_tipo_reg_interface.withColumn('tip_inter', F.when(F.col('Nome')=='tip_inter', F.col('desc')).otherwise(None))
df_regras_tipo_reg_interface = df_regras_tipo_reg_interface.groupBy('tipo').agg(F.first('tip_reg'), F.last('tip_inter'))
df_regras_tipo_reg_interface = changeColumnNames(df_regras_tipo_reg_interface, ['tipo', 'reg', 'inter'])

# COMMAND ----------

# DBTITLE 1,variáveis de separação
dict_regras_sep = {}
regras_sep = df_regras_tipo_reg_interface.rdd.map(lambda Row:{Row[0].lower():Row[1:]}).collect()
for regra in regras_sep:
  dict_regras_sep.update(regra)

del regras_sep

# COMMAND ----------

def regras_por_tipo(df_regras, tipo):
  df = df_regras.filter(F.col('tipo')==tipo)
  df = df.drop('desc').drop('tipo')
  regras = df.rdd.map(lambda Row:{Row[0]:Row[1:]}).collect()
  dict_regras = {}
  for regra in regras:
    dict_regras.update(regra)
  return (dict_regras)

# COMMAND ----------

def display_regra_por_tipo(df_regras, tipo):
  display(df_regras.filter(F.col('tipo')==tipo))