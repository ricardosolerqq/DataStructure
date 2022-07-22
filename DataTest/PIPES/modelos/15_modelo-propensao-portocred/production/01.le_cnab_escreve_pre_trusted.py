# Databricks notebook source
import time
time.sleep(300)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC <h1> este notebook lê o cnab e escreve a pre_trusted

# COMMAND ----------

# MAGIC %run "/pipe_modelos/modelo-propensao-portocred/00.funcoes_transformacao_cnab"

# COMMAND ----------

import os
import csv
from azure.storage.blob import BlockBlobService
import zipfile

# COMMAND ----------

blob_account_source_prd = "qqprd"
blob_account_source_ml = "qqdatastoragemain"
blob_container_source_prd = "qq-integrator"
blob_container_source_ml = "ml-prd"

prefix = "etl/portocred/processed"

mount_blob_storage_key(dbutils,blob_account_source_prd,blob_account_source_prd,'/mnt/qq-integrator')
mount_blob_storage_key(dbutils,blob_account_source_ml,blob_account_source_ml,'/mnt/ml-prd')

caminho_base = 'mnt/qq-integrator/etl/portocred/processed'
caminho_base_dbfs = '/dbfs/mnt/qq-integrator/etl/portocred/processed'
caminho_raw = '/mnt/ml-prd/ml-data/propensaodeal/portocred/raw'
caminho_raw_dbfs = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/portocred/raw'
caminho_pre_trusted = '/mnt/ml-prd/ml-data/propensaodeal/portocred/pre-trusted'

# COMMAND ----------

# DBTITLE 1,cria lista de arquivos a consumir
arquivos_batimento = {}
listas_datas = []
for file in os.listdir(caminho_base_dbfs):
  data_arq = getBlobFileDate(blob_account_source_prd, blob_container_source_prd, file, prefix = prefix, str_scope = "scope_qqdatastoragemain", str_key = "qqprd-key")
  listas_datas.append(data_arq)
  if '335' in file:
    arquivos_batimento.update({file:data_arq})
max_data = max(listas_datas)
for newest_file in arquivos_batimento:
  if str(arquivos_batimento[newest_file])[0:10] == str(max_data)[0:10]:
    break

# COMMAND ----------

#arquivos_etl = get_creditor_etl_file_dates('portocred')
#arquivos_batimento = {}
#for arq in arquivos_etl:
#  if arq.split('.')[1]=='BATIMENTO':
#    arquivos_batimento.update({arquivos_etl[arq]:arq})
#    
#arquivos_batimento

# COMMAND ----------

arquivo_escolhido = newest_file
arquivo_escolhido

# COMMAND ----------

# DBTITLE 1,copiando arquivo para pasta
data_arq = getBlobFileDate('qqprd','qq-integrator', arquivo_escolhido, prefix = "etl/portocred/processed", str_scope = "scope_qqdatastoragemain", str_key = "qqprd-key")
data_arq = data_arq.strftime('%Y-%m-%d')
arquivo_escolhido_path = os.path.join(caminho_base,arquivo_escolhido)
print ("arquivo a ser copiado:", arquivo_escolhido_path)
try:
  dbutils.fs.rm('/tmp_zip',True)
  dbutils.fs.mkdir('tmp_zip')
  print ('limpando pasta tmp_zip')
except:
  pass

dbutils.fs.cp(arquivo_escolhido_path, 'tmp_zip/zipped-file.zip')
print ('arquivo copiado!')

# COMMAND ----------

import zipfile
with zipfile.ZipFile('/dbfs/tmp_zip/zipped-file.zip',"r") as zip_ref:
    print ("DESZIPANDO")
    zip_ref.extractall(path = '/dbfs/tmp_zip')
    print ('deszipado!')
for file in os.listdir('/dbfs/tmp_zip'):
  if file != "zipped-file.zip":
    break

# COMMAND ----------

spark.conf.set('spark.sql.sources.partitionOverwriteMode', 'dynamic')

dict_dfs = processo_leitura_transformacao(os.path.join('/tmp_zip/', file), regras)
for tipo_registro in dict_dfs:
  df = dict_dfs[tipo_registro]
  print ('SALVANDO EM',str(caminho_pre_trusted+'/'+data_arq+'/'+tipo_registro+'.PARQUET'))
  df.write.mode('overwrite').parquet(caminho_pre_trusted+'/'+data_arq+'/'+tipo_registro+'.PARQUET') 
dbutils.fs.rm('/tmp_zip',True) # excluindo arquivos de unzip após escrever dataframe