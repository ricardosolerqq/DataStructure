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

import os
import datetime
import zipfile
import pyspark.sql.functions as F

blob_account_source_prd = "qqprd"
blob_account_source_ml = "qqdatastoragemain"
blob_container_source_prd = "qq-integrator"
blob_container_source_ml = "ml-prd"

prefix = "etl/credsystem/processed"

mount_blob_storage_key(dbutils,blob_account_source_prd,blob_account_source_prd,'/mnt/qq-integrator')
mount_blob_storage_key(dbutils,blob_account_source_ml,blob_account_source_ml,'/mnt/ml-prd')

caminho_base = 'mnt/qq-integrator/etl/credsystem/processed'
caminho_base_dbfs = '/dbfs/mnt/qq-integrator/etl/credsystem/processed'
caminho_trusted = '/mnt/ml-prd/ml-data/propensaodeal/credsystem/trusted'
caminho_sample = '/mnt/ml-prd/ml-data/propensaodeal/credsystem/sample'

# COMMAND ----------

# DBTITLE 1,Configurando Processo e Arquivo a ser Tratado
dbutils.widgets.dropdown('processamento', 'auto', ['auto', 'manual'])
processo_auto = dbutils.widgets.get('processamento')
if processo_auto == 'auto':
  processo_auto = True
else:
  processo_auto = False
  
if processo_auto:
  try:
    dbutils.widgets.remove('ARQUIVO_ESCOLHIDO')
  except:
    pass
  arquivo_escolhido = list(get_creditor_etl_file_dates('credsystem', latest=True))[0]
  arquivo_escolhido_path = os.path.join(caminho_base, arquivo_escolhido)
  arquivo_escolhido_path_dbfs = os.path.join('/dbfs',caminho_base, arquivo_escolhido)
  
else:
  ##list_arquivo = os.listdir('/dbfs/mnt/qq-integrator/etl/credsystem/processed')
  ##lista_arquivo = []
  ##for i in lista_arquivos:
  ##  if "QUERO_QUITAR_" in i:
  ##    lista_arquivo.append(i)
  ##dict_arq = {datetime.date(int(item.split('QUERO_QUITAR_')[1].split('.')[0].split('-')[2]), int(item.split('QUERO_QUITAR_')[1].split('.')[0].split('-')[1]), int(item.split('QUERO_QUITAR_')[1].split('.')[0].split('-')[0])):item for item in lista_arquivos}
  ##dbutils.widgets.dropdown('ARQUIVO_ESCOLHIDO', max(str(item) for item in dict_arq), [str(item) for item in dict_arq])
  ##arquivo_escolhido = dbutils.widgets.get('ARQUIVO_ESCOLHIDO')
  arquivo_escolhido_path = 'mnt/qq-integrator/etl/credsystem/processed/QUERO_QUITAR_202206292100.CSV' ##os.path.join(caminho_base,dict_arq[datetime.date(int(arquivo_escolhido.split('-')[0]), int(arquivo_escolhido.split('-')[1]), int(arquivo_escolhido.split('-')[2]))])
  ##arquivo_escolhido_path_dbfs = os.path.join('/dbfs',caminho_base,dict_arq[datetime.date(int(arquivo_escolhido.split('-')[0]), int(arquivo_escolhido.split('-')[1]), int(arquivo_escolhido.split('-')[2]))])

arquivo_escolhido_fileformat = arquivo_escolhido_path.split('.')[-1]
arquivo_escolhido_fileformat


#arquivo_escolhido_path
file = arquivo_escolhido_path
file

# COMMAND ----------

# DBTITLE 1,Criando Dataframe Spark
df = spark.read.option('delimiter',';').option('header', 'True').csv("/"+file)

# COMMAND ----------

df.show(10, False)

# COMMAND ----------

# DBTITLE 1,Tratando Campo de Data
df = df.withColumn('nova_data', F.to_date(F.col('PRIMEIRO_VENCIMENTO'), 'ddMMMyyyy'))
df = df.drop("PRIMEIRO_VENCIMENTO")
df = df.withColumnRenamed("nova_data", "PRIMEIRO_VENCIMENTO")
df.show(10, False)

# COMMAND ----------

dbutils.widgets.dropdown('ESCREVER_VARIAVEL_RESPOSTA', 'False', ['False', 'True'])
escreverVariavelResposta = dbutils.widgets.get('ESCREVER_VARIAVEL_RESPOSTA')
if escreverVariavelResposta == 'True':
  df = escreve_variavel_resposta_acordo(df, 'credsystem', datetime.datetime(2021,9,11), 'CPF', 'VARIAVEL_RESPOSTA', drop_null = True)
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
  #Escrevendo DataFrame
  df.coalesce(1).write.option('sep', ';').option('header', 'True').csv(os.path.join(caminho_trusted, 'tmp'))
  for file in dbutils.fs.ls(os.path.join(caminho_trusted, 'tmp')):
    if file.name.split('.')[-1] == 'csv':
      dbutils.fs.cp(file.path, os.path.join(caminho_trusted, arquivo_escolhido.split('.')[0]+'.csv'))
  dbutils.fs.rm(os.path.join(caminho_trusted, 'tmp'), True)