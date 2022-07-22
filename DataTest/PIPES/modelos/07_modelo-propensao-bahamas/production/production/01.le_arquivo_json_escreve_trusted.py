# Databricks notebook source
import time
time.sleep(300)

# COMMAND ----------

# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

try:
  dbutils.widgets.remove('ARQUIVO_ESCOLHIDO')
except:
  pass

# COMMAND ----------

# DBTITLE 1,Imports e mount
import os
import datetime
import zipfile
import pyspark.sql.functions as F
from dateutil.relativedelta import relativedelta

blob_account_source_prd = "qqprd"
blob_account_source_ml = "qqdatastoragemain"
blob_container_source_prd = "qq-integrator"
blob_container_source_ml = "ml-prd"

prefix = "etl/bahamas/processed"

mount_blob_storage_key(dbutils,blob_account_source_prd,blob_account_source_prd,'/mnt/qq-integrator')
mount_blob_storage_key(dbutils,blob_account_source_ml,blob_account_source_ml,'/mnt/ml-prd')

caminho_base = '/mnt/qq-integrator/etl/bahamas/processed'
caminho_base_dbfs = '/dbfs/mnt/qq-integrator/etl/bahamas/processed'
caminho_trusted = '/mnt/ml-prd/ml-data/propensaodeal/bahamas/trusted'
caminho_sample = '/mnt/ml-prd/ml-data/propensaodeal/bahamas/sample'

# COMMAND ----------

# DBTITLE 1,Configurando widget processamento
dbutils.widgets.dropdown('processamento', 'auto', ['auto', 'manual'])
processo_auto = dbutils.widgets.get('processamento')
if processo_auto == 'auto':
  processo_auto = True
else:
  processo_auto = False
  
processo_auto

# COMMAND ----------

# DBTITLE 1,Configurando widget variavel resposta
dbutils.widgets.dropdown('ESCREVER_VARIAVEL_RESPOSTA', 'False', ['True', 'False'])
escreverVariavelResposta = dbutils.widgets.get('ESCREVER_VARIAVEL_RESPOSTA')
if escreverVariavelResposta == 'True':
  variavelResposta = True
else:
  variavelResposta = False
  
variavelResposta

# COMMAND ----------

# DBTITLE 1,Configurando widget de arquivo escolhido
files = {}
fileList = []
for file in os.listdir(caminho_base_dbfs):
  files.update({int(file.split('_')[1].split('.')[0]):file})
  fileList.append(file)
max_date = max(files)
max_file = files[max_date]

if processo_auto:
  try:
    dbutils.widgets.remove('ARQUIVO_ESCOLHIDO')
  except:
    pass
  arquivo_escolhido = max_file
else:
  dbutils.widgets.dropdown('ARQUIVO_ESCOLHIDO', max_file, fileList)
  arquivo_escolhido = dbutils.widgets.get('ARQUIVO_ESCOLHIDO')
  
print ('processamento', processo_auto)
arquivo_escolhido

# COMMAND ----------

# DBTITLE 1,Pegando a data máxima do arquivo
arquivo_escolhido_date = arquivo_escolhido.split('_')[1].split('.')[0]
ano = int(arquivo_escolhido_date[0:4])
mes = int(arquivo_escolhido_date[4:6])
dia = int(arquivo_escolhido_date[6:8])
arquivo_escolhido_date = datetime.date(ano, mes, dia)
#arquivo_escolhido_date = arquivo_escolhido_date - relativedelta(months=1)
arquivo_escolhido_date = arquivo_escolhido_date - relativedelta(years=1)
arquivo_escolhido_date

# COMMAND ----------

# DBTITLE 1,Criando Dataframe Spark
df = spark.read.json(os.path.join(caminho_base,arquivo_escolhido))
df = df.drop("emails","enderecos","telefones")

# COMMAND ----------

display(df)

# COMMAND ----------

# DBTITLE 1,Condicional Para Variável Resposta e DataFrame
if variavelResposta:
  df = escreve_variavel_resposta_acordo(df, 'bahamas', arquivo_escolhido_date, 'cic', drop_null = True)
  df_representativo, df_aleatorio = gera_sample(df)
  df_aleatorio.coalesce(1).write.option('sep', ';').option('header', 'True').csv(os.path.join(caminho_sample, 'aleatorio_temp'))
  for file in dbutils.fs.ls(os.path.join(caminho_sample, 'aleatorio_temp')):
    if file.name.split('.')[-1] == 'csv':
      dbutils.fs.cp(file.path, os.path.join(caminho_sample, 'base_aleatoria.csv'))
  dbutils.fs.rm(os.path.join(caminho_sample, 'aleatorio_temp'), True)


  df_representativo.coalesce(1).write.option('sep', ';').option('header', 'True').csv(os.path.join(caminho_sample, 'representativo_temp'))
  for file in dbutils.fs.ls(os.path.join(caminho_sample, 'representativo_temp')):
    if file.name.split('.')[-1] == 'csv':
      dbutils.fs.cp(file.path, os.path.join(caminho_sample, 'base_representativa.csv'))
  dbutils.fs.rm(os.path.join(caminho_sample, 'representativo_temp'), True)

else:
  df.coalesce(1).write.option('sep', ';').option('header', 'True').csv(os.path.join(caminho_trusted, 'trusted_tmp'))
  for file in dbutils.fs.ls(os.path.join(caminho_trusted, 'trusted_tmp')):
    if file.name.split('.')[-1] == 'csv':
      dbutils.fs.cp(file.path, os.path.join(caminho_trusted, arquivo_escolhido.split('.')[0]+'.csv'))
  dbutils.fs.rm(os.path.join(caminho_trusted, 'trusted_tmp'), True)