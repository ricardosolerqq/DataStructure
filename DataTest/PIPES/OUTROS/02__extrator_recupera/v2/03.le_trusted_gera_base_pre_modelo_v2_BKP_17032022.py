# Databricks notebook source
# MAGIC %run "/pipe_modelos/extrator_recupera/v2/00.le_regras_disponibiliza_variaveis_v2"

# COMMAND ----------

import datetime

# COMMAND ----------

# DBTITLE 1,configurando widget de credores
dbutils.widgets.text('credor', '', '')
credor = dbutils.widgets.get('credor')

credor = Credor(credor)

# COMMAND ----------

# DBTITLE 1,configurando widget de variável resposta
dbutils.widgets.text('criar_variavel_resposta', 'False', '')
criar_variavel_resposta = dbutils.widgets.get('criar_variavel_resposta')
criar_variavel_resposta

# COMMAND ----------

dict_dfs = {}
for file in os.listdir(credor.caminho_trusted_dbfs):
  dict_dfs.update({file.split(".")[0]: spark.read.parquet(os.path.join(credor.caminho_trusted, file)).alias(file.split(".")[0])})
dict_dfs

# COMMAND ----------

# DBTITLE 1,juntando contratos com clientes
colunas_clientes = []
for col in dict_dfs['registroclientes'].columns:
  if col == 'des_regis':
    colunas_clientes.append(col)
  if col not in dict_dfs['registrocontratos'].columns:
    colunas_clientes.append(col)
df = dict_dfs['registrocontratos'].join(dict_dfs['registroclientes'].select(colunas_clientes), on='DES_REGIS', how='left')

# COMMAND ----------

# DBTITLE 1,corrigindo des_regis para documento formato MONGO - lendo des_cpf quando existente
df = df.withColumn('DOCUMENTO', F.when(F.col('des_cpf').isNotNull(), F.lpad(F.col('des_cpf'),11,"0")).otherwise(F.lpad(F.col('des_regis'), 11, "0")))

# COMMAND ----------

df = df.drop('des_regis').drop('des_cpf')

# COMMAND ----------

# DBTITLE 1,date para datetype
#Transforma toda coluna que tem numeros e tamanho 8 em data
for col in df.columns:
  if 'dat_' in col:    
    df = df.withColumn(col, F.when(F.length(F.col(col).cast(T.IntegerType()))==8, F.to_date(F.col(col), format='yyyyMMdd')).otherwise(None))

# COMMAND ----------

# DBTITLE 1,corrigindo data de nascimento
if 'dat_nasci' in df.columns:
  df = df.withColumn('dat_nasci', F.when(F.col('dat_nasci')<'1900-01-01', None).otherwise(F.col('dat_nasci')))

# COMMAND ----------

# DBTITLE 1,valores para integer round 2
#14 inteiros e 4 decimais sem ',' ou '.' preencher com zeros à esquerda

for col in df.columns:
  if 'val_' in col:
    df = df.withColumn(col, F.round((F.col(col).cast(T.IntegerType())/10000),2))

# COMMAND ----------

# DBTITLE 1,null para linhas com len(0)
for col in df.columns:
  df = df.withColumn(col, F.when(F.length(F.col('cod_credor'))==0, None).otherwise(F.col(col)))

# COMMAND ----------

df = df.withColumn('DOCUMENTO', F.lpad('DOCUMENTO',11,'0'))

# COMMAND ----------

# DBTITLE 1,gerando variavel resposta e data_arquivo a partir do widget
if criar_variavel_resposta != 'False':
  data_arquivo = criar_variavel_resposta.split(",")[1].replace(']','')
  data_arquivo = datetime.datetime(int(data_arquivo[0:4]), int(data_arquivo[4:6]), int(data_arquivo[6:8]), 3)
  print (data_arquivo)
  df = escreve_variavel_resposta_acordo(df, credor.nome, data_arquivo, 'DOCUMENTO', varName='VARIAVEL_RESPOSTA', drop_null = True)

# COMMAND ----------

dfbak = df

# COMMAND ----------

if criar_variavel_resposta != 'False':
  from pyspark.sql.window import Window
  df = df.withColumn('SK', F.concat(F.col('DOCUMENTO'),F.lit(':'), F.col('des_contr')))
  windowSpec  = Window.partitionBy("SK").orderBy(F.desc("VARIAVEL_RESPOSTA"))

  df = df.withColumn("row_number",F.row_number().over(windowSpec))
  df = df.filter(F.col('row_number')==1).drop('row_number')

# COMMAND ----------

# DBTITLE 1,escrevendo arquivo final - csv
if criar_variavel_resposta != 'False':
  
  df_representativo, df_aleatorio = gera_sample(df)
  
  data_arquivo_str = str(data_arquivo.year) + str(data_arquivo.month).zfill(2) + str(data_arquivo.day).zfill(2)
  df_representativo.coalesce(1).write.option('header', 'True').option('delimiter',';').csv(os.path.join(credor.caminho_sample, data_arquivo_str, 'amostra_representativa_temp.csv'), mode='overwrite')
  df_aleatorio.coalesce(1).write.option('header', 'True').option('delimiter',';').csv(os.path.join(credor.caminho_sample, data_arquivo_str, 'amostra_aleatoria_temp.csv'), mode='overwrite')
  
  for file in dbutils.fs.ls(os.path.join(credor.caminho_sample, data_arquivo_str, 'amostra_representativa_temp.csv')):
      if file.name.split('.')[-1] == 'csv':
        dbutils.fs.cp(file.path, os.path.join(credor.caminho_sample, data_arquivo_str, "amostra_representativa_"+str(datetime.datetime.today()).replace(' ', '_').split('.')[0].replace(":",'_')+".csv"))
  dbutils.fs.rm(os.path.join(credor.caminho_sample, data_arquivo_str, 'amostra_representativa_temp.csv'), True)

  for file in dbutils.fs.ls(os.path.join(credor.caminho_sample, data_arquivo_str, 'amostra_aleatoria_temp.csv')):
      if file.name.split('.')[-1] == 'csv':
        dbutils.fs.cp(file.path, os.path.join(credor.caminho_sample, data_arquivo_str, "amostra_aleatoria_"+str(datetime.datetime.today()).replace(' ', '_').split('.')[0].replace(":",'_')+".csv"))
  dbutils.fs.rm(os.path.join(credor.caminho_sample, data_arquivo_str, 'amostra_aleatoria_temp.csv'), True)  

  
else:
  df.coalesce(1).write.mode('overwrite').option('sep', ';').option('header', 'True').csv(os.path.join(credor.caminho_joined_trusted, 'temp_output.csv'))
  for file in dbutils.fs.ls(os.path.join(credor.caminho_joined_trusted, 'temp_output.csv')):
    if file.name.split('.')[-1] == 'csv':
      print (file)
      dbutils.fs.cp(file.path, os.path.join(credor.caminho_joined_trusted,'pre_output.csv'))
  dbutils.fs.rm(os.path.join(credor.caminho_joined_trusted, 'temp_output.csv'), True)

# COMMAND ----------

dbutils.notebook.exit('OK')