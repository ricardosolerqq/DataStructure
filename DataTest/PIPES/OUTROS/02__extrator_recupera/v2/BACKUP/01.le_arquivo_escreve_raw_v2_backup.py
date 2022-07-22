# Databricks notebook source
# MAGIC %run "/pipe_modelos/extrator_recupera/v2/00.le_regras_disponibiliza_variaveis_v2"

# COMMAND ----------

# DBTITLE 1,drop widgets
try:
  dbutils.widgets.remove('ARQUIVO_ESCOLHIDO')
except:
  pass
try:
  dbutils.widgets.remove('CREDOR_ESCOLHIDO')
except:
  pass
try:
  dbutils.widgets.remove('LE_BASE_CONSOLIDADA')
except:
  pass


# COMMAND ----------

# DBTITLE 1,configurando widgets e arquivos
dbutils.widgets.dropdown('CREDOR_ESCOLHIDO', 'bmg', credores_recupera)
credor = dbutils.widgets.get('CREDOR_ESCOLHIDO')

credor = Credor(credor)

dbutils.widgets.dropdown('LE_BASE_CONSOLIDADA', 'False', ['True', 'False'])
le_base_consolidada = dbutils.widgets.get('LE_BASE_CONSOLIDADA')
print ('base consolidada: ',le_base_consolidada)

dict_files = {}
if le_base_consolidada == "False":
  for file in os.listdir(credor.caminho_base_dbfs):
    for filename_possibility in credor.nome_arquivos:
      if filename_possibility in file:
        file_num = file.split(filename_possibility)[1]
        dict_files.update({file_num:file})
      else:
        dict_files.update({file:file})
  dbutils.widgets.combobox('ARQUIVO_ESCOLHIDO', str(max(list(dict_files))), [str(item) for item in sorted(dict_files,reverse=True)[0:1023]])
else:
  for file in os.listdir(credor.caminho_base_consolidada_dbfs):
    dict_files.update({file.split('_g_')[0]:file})
  dbutils.widgets.combobox('ARQUIVO_ESCOLHIDO', str(max(list(dict_files))), list(dict_files))

# COMMAND ----------

arquivo_escolhido = dbutils.widgets.get("ARQUIVO_ESCOLHIDO")
arquivo_escolhido = dict_files[arquivo_escolhido]
arquivo_escolhido

# COMMAND ----------

# DBTITLE 1,copiando para local temporário no storage
try:
  dbutils.fs.rm(credor.caminho_temp, True)
  print ('tmp removido!')
  dbutils.fs.mkdirs(credor.caminho_temp)
except:
  pass

dbutils.fs.cp(os.path.join(credor.caminho_base, arquivo_escolhido), os.path.join(credor.caminho_temp,arquivo_escolhido))

# COMMAND ----------

# DBTITLE 1,lendo arquivo - tratando possibilidade de zip e 7z
filepath = descomprime_arquivo_obtem_caminho(credor, arquivo_escolhido)
df = spark.read.option("encoding", "UTF-8").csv(filepath)
df = changeColumnNames(df, ['raw_info'])  
df = df.withColumn('reg', F.substring(F.col('raw_info'), 1,1)).withColumn('inter', F.substring(F.col('raw_info'), 12,1))

# COMMAND ----------

# DBTITLE 1,obtendo data do arquivo
if le_base_consolidada == "True":
  data_arquivo = arquivo_escolhido.split('_g_')[1]
else:
  data_arquivo =  df.withColumn('dat_movto', F.substring(F.col('raw_info'),4,8)).select('dat_movto').limit(1).rdd.map(lambda Row:Row[0]).collect()[0]
data_arquivo

# COMMAND ----------

# DBTITLE 1,separando dataframes por tipo (registro e interface)
dict_dfs = {}

for regra in dict_regras_sep:
  dict_dfs.update({regra:df.filter((F.col('reg')==dict_regras_sep[regra][0])&(F.col('inter')==dict_regras_sep[regra][1]))})

# COMMAND ----------

if le_base_consolidada == 'True':
  pasta = data_arquivo+'_consolidada'
else:
  pasta = data_arquivo
for regra in dict_dfs:
  print (regra)
  dict_dfs[regra].write.mode('overwrite').parquet(os.path.join(credor.caminho_raw, pasta, str(regra+'.PARQUET')))
  
os.path.join(credor.caminho_raw, pasta, str(regra+'.PARQUET'))