# Databricks notebook source
# MAGIC %run "/pipe_modelos/extrator_recupera/v2/00.le_regras_disponibiliza_variaveis_v2"

# COMMAND ----------

# DBTITLE 1,configurando widgets e arquivos
dbutils.widgets.text('credor', '', '')
credor = dbutils.widgets.get('credor')

credor = Credor(credor)
credor.nome

# COMMAND ----------

# DBTITLE 1,lendo arquivo
filepath = credor.caminho_temp

df = spark.read.option("encoding", "UTF-8").csv(filepath)
df = changeColumnNames(df, ['raw_info'])  
df = df.withColumn('reg', F.substring(F.col('raw_info'), 1,1)).withColumn('inter', F.substring(F.col('raw_info'), 12,1))

# COMMAND ----------

# DBTITLE 1,separando dataframes por tipo (registro e interface)
dict_dfs = {}

for regra in credor.dict_regras_sep:
  dict_dfs.update({regra:df.filter((F.col('reg')==credor.dict_regras_sep[regra][0])&(F.col('inter')==credor.dict_regras_sep[regra][1]))})

# COMMAND ----------

dict_dfs

# COMMAND ----------

for regra in dict_dfs:
  print (regra)
  dict_dfs[regra].write.mode('overwrite').parquet(os.path.join(credor.caminho_raw, str(regra+'.PARQUET')))
  
os.path.join(credor.caminho_raw, str(regra+'.PARQUET'))

# COMMAND ----------

dbutils.notebook.exit('OK')