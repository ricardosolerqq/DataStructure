# Databricks notebook source
import os
import csv
from azure.storage.blob import BlockBlobService
import zipfile

# COMMAND ----------

# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

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

# COMMAND ----------

# DBTITLE 1,escolhe arquivo mais recente
"""
arquivos_batimento = {}
listas_datas = []
for file in list_caminho_base:
  if 'Batimento' in file:
    data_arq = getBlobFileDate(blob_account_source_prd, blob_container_source_prd, file, prefix = prefix, str_scope = "scope_qqdatastoragemain", str_key = "qqprd-key")
    listas_datas.append(data_arq)
    arquivos_batimento.update({file:data_arq})

max_data = max(listas_datas)
for f in arquivos_batimento:
  if arquivos_batimento[f]==max_data:
    arquivo_escolhido = f
    
"""

arquivo_escolhido = '20210806-Batimento.zip'

# COMMAND ----------

# DBTITLE 1,copiando arquivo para pasta e então deszipando
data_arq = getBlobFileDate('qqprd','qq-integrator', arquivo_escolhido, prefix = "etl/credz/processed", str_scope = "scope_qqdatastoragemain", str_key = "qqprd-key")
data_arq = data_arq.strftime('%Y-%m-%d')
arquivo_escolhido_path = os.path.join(caminho_base,arquivo_escolhido)
print ("arquivo a ser copiado:", arquivo_escolhido_path)
try:
  dbutils.fs.rm(os.path.join(caminho_raw, 'zipped-file'),True)
  dbutils.fs.mkdir(os.path.join(caminho_raw, 'zipped-file'))
  print ('limpando pasta zipped-file')
except:
  pass

dbutils.fs.cp(arquivo_escolhido_path, caminho_raw + '/zipped-file/' + 'zipped-file.zip')
print ('arquivo copiado!')

# COMMAND ----------

caminho_raw + '/zipped-file/' + 'zipped-file.zip'

# COMMAND ----------

arquivo_escolhido_path = '/dbfs' + caminho_raw + '/zipped-file/' + 'zipped-file.zip'
with zipfile.ZipFile(arquivo_escolhido_path,"r") as zip_ref:
    print ("DESZIPANDO",arquivo_escolhido_path)
    zip_ref.extractall(path = '/tmp_zip')
    print ('deszipado!')

# COMMAND ----------

os.path.join(caminho_raw, data_arq, data_arq)+'.csv'

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC ls /

# COMMAND ----------

#os.mkdir('/tmp_zip')
#os.remove('/tmp_dir')
os.listdir('/tmp_zip')