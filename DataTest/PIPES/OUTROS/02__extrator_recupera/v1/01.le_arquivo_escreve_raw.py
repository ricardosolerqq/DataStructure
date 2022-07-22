# Databricks notebook source
# MAGIC %run "/pipe_modelos/extrator_recupera/v1/00.le_regras_disponibiliza_variaveis"

# COMMAND ----------

try:
  dbutils.widgets.remove('CREDOR_ESCOLHIDO')
except:
  pass
try:
  dbutils.widgets.remove('ARQUIVO_ESCOLHIDO')
except:
  pass

# COMMAND ----------

# DBTITLE 1,configurando widget de credores
credores_recupera = ['agibank','bmg','fort','tribanco','trigg','valia','zema']
dbutils.widgets.dropdown('CREDOR_ESCOLHIDO', 'bmg', credores_recupera)
credor = dbutils.widgets.get('CREDOR_ESCOLHIDO')

caminho_base,caminho_base_dbfs,caminho_raw,caminho_raw_dbfs,caminho_trusted,caminho_trusted_dbfs,caminho_joined_trusted,caminho_joined_trusted_dbfs,caminho_sample,caminho_sample_dbfs = obtem_variaveis_caminho(credor)

# COMMAND ----------

dict_files = {}
for file in os.listdir(caminho_base_dbfs):
  if 'BT' in file:
    file_num = file.split('BT')[1]
    file_num = file_num.split('QQUITAR')[0]
    dict_files.update({file_num:file})
  else:
    dict_files.update({file:file})

# COMMAND ----------

try:
  dbutils.widgets.remove('ARQUIVO_ESCOLHIDO')
except:
  pass

# COMMAND ----------

dbutils.widgets.dropdown('ARQUIVO_ESCOLHIDO', str(max(list(dict_files))), [str(item) for item in sorted(dict_files,reverse=True)[0:1023]])

# COMMAND ----------

arquivo_escolhido = dbutils.widgets.get("ARQUIVO_ESCOLHIDO")
arquivo_escolhido = dict_files[arquivo_escolhido]
arquivo_escolhido

# COMMAND ----------

# DBTITLE 1,lendo arquivo
filepath = os.path.join(caminho_base, arquivo_escolhido)
df = spark.read.option("encoding", "UTF-8").csv(filepath)
df = changeColumnNames(df, ['raw_info'])
df = df.withColumn('reg', F.substring(F.col('raw_info'), 1,1)).withColumn('inter', F.substring(F.col('raw_info'), 12,1))

# COMMAND ----------

# DBTITLE 1,obtendo data do arquivo
data_arquivo =  df.withColumn('dat_movto', F.substring(F.col('raw_info'),4,8)).select('dat_movto').limit(1).rdd.map(lambda Row:Row[0]).collect()[0]
data_arquivo

# COMMAND ----------

# DBTITLE 1,separando dataframes por tipo (registro e interface)
dict_dfs = {}

for regra in dict_regras_sep:
  dict_dfs.update({regra:df.filter((F.col('reg')==dict_regras_sep[regra][0])&(F.col('inter')==dict_regras_sep[regra][1]))})

# COMMAND ----------

for regra in dict_dfs:
  print (regra)
  dict_dfs[regra].write.mode('overwrite').parquet(os.path.join(caminho_raw, data_arquivo, str(regra+'.PARQUET')))