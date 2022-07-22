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

# MAGIC %run "/pipe_modelos/modelo-propensao-credz/00.funcoes_transformacao_cnab"

# COMMAND ----------

blob_account_source_prd = "qqprd"
blob_account_source_ml = "qqdatastoragemain"
blob_container_source_prd = "qq-integrator"
blob_container_source_ml = "ml-prd"

prefix = "etl/credz/processed"

mount_blob_storage_key(dbutils,blob_account_source_prd,blob_account_source_prd,'/mnt/qq-integrator')
mount_blob_storage_key(dbutils,blob_account_source_ml,blob_account_source_ml,'/mnt/ml-prd')

caminho_base = 'mnt/qq-integrator/etl/credz/processed'
caminho_base_dbfs = '/dbfs/mnt/qq-integrator/etl/credz/processed'
caminho_raw = '/mnt/ml-prd/ml-data/propensaodeal/credz/raw'
caminho_raw_dbfs = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/credz/raw'
caminho_pre_trusted = '/mnt/ml-prd/ml-data/propensaodeal/credz/pre-trusted'

# COMMAND ----------

# DBTITLE 1,excluindo widget
try:
  dbutils.widgets.remove('ARQUIVO_ESCOLHIDO')
except:
  pass

# COMMAND ----------

# DBTITLE 1,cria lista de arquivos a consumir
arquivos_batimento = {}
listas_datas = []
for file in os.listdir(caminho_base_dbfs):
  if 'Batimento' in file:
    data_arq = getBlobFileDate(blob_account_source_prd, blob_container_source_prd, file, prefix = prefix, str_scope = "scope_qqdatastoragemain", str_key = "qqprd-key")
    listas_datas.append(data_arq)
    arquivos_batimento.update({file:data_arq})
max_data = max(listas_datas)
for newest_file in arquivos_batimento:
  if arquivos_batimento[newest_file] == max_data:
    break

# COMMAND ----------

# DBTITLE 1,criando widget
dbutils.widgets.combobox('ARQUIVO_ESCOLHIDO', newest_file, list(arquivos_batimento))

arquivo_escolhido = dbutils.widgets.get('ARQUIVO_ESCOLHIDO')

# COMMAND ----------

# DBTITLE 1,copiando arquivo para pasta
data_arq = getBlobFileDate('qqprd','qq-integrator', arquivo_escolhido, prefix = "etl/credz/processed", str_scope = "scope_qqdatastoragemain", str_key = "qqprd-key")
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

with zipfile.ZipFile('/dbfs/tmp_zip/zipped-file.zip',"r") as zip_ref:
    print ("DESZIPANDO")
    zip_ref.extractall(path = '/dbfs/tmp_zip')
    print ('deszipado!')
for file in os.listdir('/dbfs/tmp_zip'):
  if file != "zipped-file.zip":
    break

# COMMAND ----------

spark.conf.set('spark.sql.sources.partitionOverwriteMode', 'dynamic')

dict_dfs = processo_leitura_transformacao('/tmp_zip/'+file, regras)
for tipo_registro in dict_dfs:
  df = dict_dfs[tipo_registro]
  print ('SALVANDO EM',str(caminho_pre_trusted+'/'+data_arq+'/'+tipo_registro+'.PARQUET'))
  df.write.mode('overwrite').parquet(caminho_pre_trusted+'/'+data_arq+'/'+tipo_registro+'.PARQUET') 
dbutils.fs.rm('/tmp_zip',True) # excluindo arquivos de unzip após escrever dataframe