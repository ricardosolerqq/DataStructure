# Databricks notebook source
import time
time.sleep(300)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC <h1> este notebook lê o cnab e escreve a pre_trusted

# COMMAND ----------

import os
import csv
from azure.storage.blob import BlockBlobService
import zipfile

# COMMAND ----------

# MAGIC %run "/pipe_modelos/modelo-propensao-avenida/00.funcoes_transformacao_cnab"

# COMMAND ----------

blob_account_source_prd = "qqprd"
blob_account_source_ml = "qqdatastoragemain"
blob_container_source_prd = "qq-integrator"
blob_container_source_ml = "ml-prd"

prefix = "etl/avenida/processed"

mount_blob_storage_key(dbutils,blob_account_source_prd,blob_account_source_prd,'/mnt/qq-integrator')
mount_blob_storage_key(dbutils,blob_account_source_ml,blob_account_source_ml,'/mnt/ml-prd')

caminho_base = 'mnt/qq-integrator/etl/avenida/processed'
caminho_base_dbfs = '/dbfs/mnt/qq-integrator/etl/avenida/processed'
caminho_raw = '/mnt/ml-prd/ml-data/propensaodeal/avenida/raw'
caminho_raw_dbfs = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/avenida/raw'
caminho_pre_trusted = '/mnt/ml-prd/ml-data/propensaodeal/avenida/pre-trusted'

caminho_log = '/mnt/ml-prd/ml-data/propensaodeal/avenida/log'

# COMMAND ----------

arquivos_etl = get_creditor_etl_file_dates('avenida')
arquivos_processaveis = {}
for arq in arquivos_etl:
  if arq.split('.')[1]=='BATIMENTO':
    arquivos_processaveis.update({arquivos_etl[arq]:arq})
    
max_data = max(list(arquivos_processaveis))
print(max_data)
newest_file = arquivos_processaveis[max_data]
newest_file

# COMMAND ----------

# DBTITLE 1,altere arquivo_escolhido para processar outro arquivo. padrão newest_file
arquivo_escolhido = newest_file

# COMMAND ----------

# DBTITLE 1,copiando arquivo para pasta
data_arq = getBlobFileDate('qqprd','qq-integrator', arquivo_escolhido, prefix = "etl/avenida/processed", str_scope = "scope_qqdatastoragemain", str_key = "qqprd-key")
data_arq = data_arq.strftime('%Y-%m-%d_%H%M%S')
arquivo_escolhido_path = os.path.join(caminho_base,arquivo_escolhido)
print ("arquivo a ser copiado:", arquivo_escolhido_path)
try:
  dbutils.fs.rm('/tmp_zip',True)
  dbutils.fs.mkdir('tmp_zip')
  print ('limpando pasta tmp_zip')
except:
  pass

dbutils.fs.cp(arquivo_escolhido_path, 'tmp/file.txt')
print ('arquivo copiado!')

# COMMAND ----------

data_arq

# COMMAND ----------

df_raw = spark.read.csv('/tmp/file.txt').withColumnRenamed('_c0', 'TXT_ORIGINAL')

# COMMAND ----------

spark.conf.set('spark.sql.sources.partitionOverwriteMode', 'dynamic')

dict_dfs = processo_leitura_transformacao(df_raw, regras)
for tipo_registro in dict_dfs:
  df = dict_dfs[tipo_registro]
  print ('SALVANDO EM',str(caminho_pre_trusted+'/'+data_arq+'/'+tipo_registro+'.PARQUET'))
  df.write.mode('overwrite').parquet(caminho_pre_trusted+'/'+data_arq+'/'+tipo_registro+'.PARQUET') 
dbutils.fs.rm('/tmp',True) # excluindo arquivos de unzip após escrever dataframe