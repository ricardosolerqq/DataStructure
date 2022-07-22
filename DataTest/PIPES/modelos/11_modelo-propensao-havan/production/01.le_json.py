# Databricks notebook source
import time
time.sleep(300)

# COMMAND ----------

# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

import os
import datetime
import json

blob_account_source_prd = "qqprd"
blob_account_source_ml = "qqdatastoragemain"
blob_container_source_prd = "qq-integrator"
blob_container_source_ml = "ml-prd"

prefix = "etl/dmcard/processed"

mount_blob_storage_key(dbutils,blob_account_source_prd,blob_account_source_prd,'/mnt/qq-integrator')
mount_blob_storage_key(dbutils,blob_account_source_ml,blob_account_source_ml,'/mnt/ml-prd')

caminho_base = '/mnt/qq-integrator/etl/havan/processed'
caminho_base_dbfs = '/dbfs/mnt/qq-integrator/etl/havan/processed'
caminho_trusted = '/mnt/ml-prd/ml-data/propensaodeal/havan/trusted'
caminho_sample = '/mnt/ml-prd/ml-data/propensaodeal/havan/sample'

spark.conf.set("spark.sql.session.timeZone", "Etc/UCT")

# COMMAND ----------

# DBTITLE 1,config autoMode
dbutils.widgets.dropdown('MODE', 'AUTO', ['AUTO', 'MANUAL'])
if dbutils.widgets.get('MODE') == 'AUTO':
  autoMode = True
else:
  autoMode = False
  
autoMode

# COMMAND ----------

# DBTITLE 1,config variavelResposta
dbutils.widgets.dropdown('VARIAVEL_RESPOSTA', 'False', ['True', 'False'])
if dbutils.widgets.get('VARIAVEL_RESPOSTA') == 'True':
  variavelResposta = True
else:
  variavelResposta = False
  
variavelResposta

# COMMAND ----------

files = {}
fileList = []
for file in os.listdir(caminho_base_dbfs):
  if file.split('_')[1] == 'clients':
    files.update({int(file.split('_')[-1].split('.')[0]):file})
    fileList.append(file)
max_date = max(files)
max_file = files[max_date]

if autoMode:
  try:
    dbutils.widgets.remove('ARQUIVO_ESCOLHIDO')
  except:
    pass
  arquivo_escolhido = max_file
else:
  dbutils.widgets.dropdown('ARQUIVO_ESCOLHIDO', max_file, fileList)
  arquivo_escolhido = dbutils.widgets.get('ARQUIVO_ESCOLHIDO')
  
print ('autoMode', autoMode)
arquivo_escolhido

# COMMAND ----------

arquivo_escolhido_date = arquivo_escolhido.split('_')[-1].split('.')[0]
arquivo_escolhido_date = arquivo_escolhido_date[0:-4]
ano = int(arquivo_escolhido_date[0:4])
mes = int(arquivo_escolhido_date[4:6])
dia = int(arquivo_escolhido_date[6:8])
arquivo_escolhido_date = datetime.date(ano, mes, dia)
arquivo_escolhido_date

# COMMAND ----------

df = spark.read.json(os.path.join(caminho_base,arquivo_escolhido))

# COMMAND ----------

# DBTITLE 1,realizando transformações no df

df = df.withColumn('dataMenorVencimento', F.to_date(F.col('dataMenorVencimento')))

df = df.withColumn('documento', F.lpad(F.translate(F.translate(F.col('documento'), '-', ''), ".", ''), 11, "0"))

### limpando caracteres que podem impedir a leitura como numerico na proxima etapa 
cols_to_numeric= ['celularPrincipal',
                  'codigo',
                  'diaVencimentoCartao',
                  'valorVencido']

for col in cols_to_numeric:
  df = df.withColumn(col, F.translate(F.col(col), '-', ''))
  df = df.withColumn(col, F.translate(F.col(col), ' ', ''))
  df = df.withColumn(col, F.regexp_replace(col, "[a-zA-Z]", ''))

#corrigindo datas  
df = df.withColumn('dataMenorVencimento', F.when(F.col('dataMenorVencimento')<'1970-01-01', None).otherwise(F.col('dataMenorVencimento')))
display(df)

# COMMAND ----------

if variavelResposta:
  df = escreve_variavel_resposta_acordo(df, 'havan', arquivo_escolhido_date, 'documento', drop_null = True)
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
  arquivo_escolhido_date_str = arquivo_escolhido_date.strftime('%Y%m%d')
  df.coalesce(1).write.mode('overwrite').option('sep', ';').option('header', 'True').csv(os.path.join(caminho_trusted, arquivo_escolhido_date_str,'full_temp'))
  for file in dbutils.fs.ls(os.path.join(caminho_trusted, arquivo_escolhido_date_str, 'full_temp')):
    if file.name.split('.')[-1] == 'csv':
      dbutils.fs.cp(file.path, os.path.join(caminho_trusted, arquivo_escolhido_date_str, 'havan_full.csv'))
    else:
      dbutils.fs.rm(file.path, True)
  dbutils.fs.rm(os.path.join(caminho_trusted, arquivo_escolhido_date_str, 'full_temp'), True)