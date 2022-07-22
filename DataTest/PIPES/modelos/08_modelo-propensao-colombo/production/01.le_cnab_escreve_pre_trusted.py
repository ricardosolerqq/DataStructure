# Databricks notebook source
import time
time.sleep(300)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC <h1> este notebook lê o cnab e escreve a trusted

# COMMAND ----------

credor = 'colombo'

# COMMAND ----------

# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

import datetime

# COMMAND ----------

blob_account_source_prd = "qqprd"
blob_account_source_ml = "qqdatastoragemain"
blob_container_source_prd = "qq-integrator"
blob_container_source_ml = "ml-prd"

prefix = "etl/colombo/processed"

mount_blob_storage_key(dbutils,blob_account_source_prd,blob_account_source_prd,'/mnt/qq-integrator')
mount_blob_storage_key(dbutils,blob_account_source_ml,blob_account_source_ml,'/mnt/ml-prd')

caminho_base = 'mnt/qq-integrator/etl/colombo/processed'
caminho_base_dbfs = '/dbfs/mnt/qq-integrator/etl/colombo/processed'
caminho_trusted = '/mnt/ml-prd/ml-data/propensaodeal/colombo/trusted'
caminho_trusted_dbfs = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/colombo/trusted'
caminho_sample= '/mnt/ml-prd/ml-data/propensaodeal/colombo/sample'
caminho_sample_dbfs = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/colombo/sample'

# COMMAND ----------

arquivos_etl = get_creditor_etl_file_dates('colombo')
arquivos_processaveis_por_data = {}
for arq in arquivos_etl:
  try:
    arquivos_processaveis_por_data[arquivos_etl[arq].date()].append(arq)
  except:
    arquivos_processaveis_por_data.update({arquivos_etl[arq].date():[arq]})

max_date = max(arquivos_processaveis_por_data)
arquivos_a_processar = arquivos_processaveis_por_data[max_date]
arquivos_a_processar

# COMMAND ----------

# DBTITLE 1,copiando arquivo para pasta tmp
data_arq_datetime = getBlobFileDate('qqprd','qq-integrator', arquivos_a_processar[0], prefix = "etl/colombo/processed", str_scope = "scope_qqdatastoragemain", str_key = "qqprd-key")
data_arq = data_arq_datetime.strftime('%Y-%m-%d_%H%M%S')

try:
  dbutils.fs.rm('/tmp/'+credor,True)
  dbutils.fs.mkdirs('tmp/'+credor)
  print ('limpando pasta tmp')
except Exception as e:
  print (e)
  pass

for arquivo_escolhido in arquivos_a_processar:
  arquivo_escolhido_path = os.path.join(caminho_base,arquivo_escolhido)
  print ("arquivo a ser copiado:", arquivo_escolhido_path)


  dbutils.fs.cp(arquivo_escolhido_path, os.path.join('/tmp/'+credor+'/', arquivo_escolhido))
  print ('arquivo copiado!')
  
data_arq_datetime

# COMMAND ----------

df = spark.read.option('header', 'True').option('sep', ',').csv('/tmp/'+credor)
df = changeColumnNames(df, ['clientID', 
                               'clientFirstName', 
                               'dueDate', 
                               'DOCUMENT', 
                               'param_5', 
                               'amount', 
                               'param_7', 
                               'param_8', 
                               'param_9',
                               'activation',
                               'name', 
                               'birthDate', 
                               'identifier',
                              'tel1',
                              'tel2',
                              'tel3',
                              'tel4',
                              'tel5',
                              'tel6',
                              'tel7',
                              'tel8',
                              'tel9',
                              'tel10',
                              'tel11',
                              'tel12',
                              'tel13',
                              'param_25'])

for col in df.columns:
  df = df.withColumn(col, F.when(F.col(col)=='-', F.lit('')).otherwise(F.col(col)))

# COMMAND ----------

display(df)

# COMMAND ----------

# DBTITLE 1,escrever variável resposta?
dbutils.widgets.dropdown('VARIAVEL_RESPOSTA', 'False', ['True', 'False'])
variavelResposta = dbutils.widgets.get('VARIAVEL_RESPOSTA')
if variavelResposta == 'True':
  variavelResposta = True
else:
  variavelResposta = False
  
variavelResposta

# COMMAND ----------

if variavelResposta:
  df = escreve_variavel_resposta_acordo(df, 'colombo', data_arq_datetime, 'DOCUMENT',drop_null = True)
  df_representativo, df_aleatorio = gera_sample(df)
  df_representativo.coalesce(1).write.option('header', 'True').option('sep', ';').mode('overwrite').csv(os.path.join(caminho_sample, data_arq, 'amostra_representativa_temp'))
  for file in dbutils.fs.ls(os.path.join(caminho_sample, data_arq, 'amostra_representativa_temp')):
    if file.name.split('.')[-1] == 'csv':
      dbutils.fs.cp(file.path, os.path.join(caminho_sample, data_arq, 'amostra_representativa.csv'))
    else:
      dbutils.fs.rm(file.path, True)
  dbutils.fs.rm(os.path.join(caminho_sample, data_arq, 'amostra_representativa_temp'), True)

  df_aleatorio.coalesce(1).write.option('header', 'True').option('sep', ';').mode('overwrite').csv(os.path.join(caminho_sample, data_arq, 'amostra_aleatoria_temp'))
  for file in dbutils.fs.ls(os.path.join(caminho_sample, data_arq, 'amostra_aleatoria_temp')):
    if file.name.split('.')[-1] == 'csv':
      dbutils.fs.cp(file.path, os.path.join(caminho_sample, data_arq, 'amostra_aleatoria.csv'))
    else:
      dbutils.fs.rm(file.path, True)
  dbutils.fs.rm(os.path.join(caminho_sample, data_arq, 'amostra_aleatoria_temp'), True)
else:
  df.coalesce(1).write.option('header', 'True').option('sep', ';').mode('overwrite').csv(os.path.join(caminho_trusted, data_arq.split('_')[0].replace('-',''), 'colombo_full_temp'))
  for file in dbutils.fs.ls(os.path.join(caminho_trusted, data_arq.split('_')[0].replace('-',''), 'colombo_full_temp')):
    if file.name.split('.')[-1] == 'csv':
      dbutils.fs.cp(file.path, os.path.join(caminho_trusted, data_arq.split('_')[0].replace('-',''), 'colombo_full.csv'))
    else:
      dbutils.fs.rm(file.path, True)
  dbutils.fs.rm(os.path.join(caminho_trusted, data_arq.split('_')[0].replace('-',''), 'colombo_full_temp'), True)