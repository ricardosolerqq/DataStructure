# Databricks notebook source
# MAGIC %run "/pipe_modelos/extrator_recupera/v2/00.le_regras_disponibiliza_variaveis_v2"

# COMMAND ----------

credor = 'bmg'

# COMMAND ----------

# DBTITLE 1,obtendo classe Credor e variáveis de ambiente
credor = Credor(credor)

# COMMAND ----------

files = {}
for file in os.listdir(credor.caminho_base_dbfs):
  if credor.nome_arquivos[0] in file or credor.nome_arquivos[1] in file:
    files.update({getBlobFileDate(blob_account_source_prd, blob_container_source_prd, file, prefix = credor.prefix, str_scope = "scope_qqdatastoragemain", str_key = "qqprd-key"):file})

# COMMAND ----------

for date in sorted(files, reverse=True):
  if credor.nome_arquivos[0] in files[date]:
    latest_bt = (files[date])
    latest_bt_date = date
    break
    
print(latest_bt)
print (latest_bt_date)

# COMMAND ----------

arquivos_escolhidos = []
for date in sorted(files, reverse=True):
  if date >= latest_bt_date:
    arquivos_escolhidos.append(files[date])
max_date = max(sorted(files))
max_date

# COMMAND ----------

try:
  dbutils.fs.rm('tmp',True)
  dbutils.fs.mkdirs('tmp')
  print ('limpando pasta tmp')
  dbutils.fs.rm('tmp/txt_puro',True)
  dbutils.fs.mkdirs('tmp/txt_puro')
  print ('limpando pasta tmp/txt_puro')
except:
  pass

for file in arquivos_escolhidos:
  print (file, end='')
  if file.split('.')[-1].lower() == 'txt':
    dbutils.fs.cp(os.path.join(credor.caminho_base, file), 'tmp/txt_puro
  else:
    dbutils.fs.cp(os.path.join(credor.caminho_base, file), 'tmp')
  print (' copiado!')

# COMMAND ----------

if credor.formato_arquivo == 'zip':
  raise Exception ('formato ZIP não implementado em cria_base_full!!!')
elif credor.formato_arquivo == '7z':
  raise Exception ('formato 7z não implementado em cria_base_full!!!')
elif credor.formato_arquivo == 'txt':
  pass
else:
  print ('formato', credor.formato_arquivo, 'não implementado em cria_base_full!!!')

# COMMAND ----------

for file in os.listdir('/dbfs/tmp'):
  if file.split('.')[-1].lower() == 'txt':
    print (file, end='')
    dbutils.fs.cp(os.path.join('tmp', file),'tmp/txt_puro')
    print ('copiado!')

# COMMAND ----------

df = spark.read.csv('/tmp/txt_puro')

# COMMAND ----------

data_inicio = str(latest_bt_date.year)+str(latest_bt_date.month)+str(latest_bt_date.day)
data_fim = str(max_date.year)+str(max_date.month)+str(max_date.day)

# COMMAND ----------

df.write.parquet(os.path.join(credor.caminho_base_consolidada, data_inicio + "_a_" + data_fim + '_g_' + str(datetime.today().date())))
str(latest_bt_date.date()) + "_a_" + str(max_date.date()) + '_g_' + str(datetime.today().date())