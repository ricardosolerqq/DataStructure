# Databricks notebook source
# MAGIC %run "/pipe_modelos/extrator_recupera/v1/00.le_regras_disponibiliza_variaveis"

# COMMAND ----------

import datetime

# COMMAND ----------

# DBTITLE 1,configurando widget de credores
credores_recupera = ['agibank','bmg','fort','tribanco','trigg','valia','zema']
dbutils.widgets.dropdown('CREDOR_ESCOLHIDO', 'bmg', credores_recupera)
credor = dbutils.widgets.get('CREDOR_ESCOLHIDO')

caminho_base,caminho_base_dbfs,caminho_raw,caminho_raw_dbfs,caminho_trusted,caminho_trusted_dbfs,caminho_joined_trusted,caminho_joined_trusted_dbfs,caminho_sample,caminho_sample_dbfs = obtem_variaveis_caminho(credor)

# COMMAND ----------

files = []
for file in os.listdir(caminho_trusted_dbfs):
  if '.PARQUET' not in file:
    files.append(file)

# COMMAND ----------

try:
  dbutils.widgets.remove('PASTA_ESCOLHIDA')
except:
  pass

# COMMAND ----------

dbutils.widgets.dropdown('PASTA_ESCOLHIDA', max(files), files)

# COMMAND ----------

pasta_escolhida = dbutils.widgets.get("PASTA_ESCOLHIDA")
dict_dfs = {}
for file in os.listdir(os.path.join(caminho_trusted_dbfs, pasta_escolhida)):
  dict_dfs.update({file.split(".")[0]: spark.read.parquet(os.path.join(caminho_trusted, pasta_escolhida, file)).alias(file.split(".")[0])})

# COMMAND ----------

# DBTITLE 1,juntando contratos com clientes
colunas_clientes = []
for col in dict_dfs['cliente'].columns:
  if col == 'des_regis':
    colunas_clientes.append(col)
  if col not in dict_dfs['contratos'].columns:
    colunas_clientes.append(col)
df = dict_dfs['contratos'].join(dict_dfs['cliente'].select(colunas_clientes), on='DES_REGIS', how='left')

# COMMAND ----------

# DBTITLE 1,corrigindo des_regis para documento formato MONGO - lendo des_cpf quando existente
df = df.withColumn('DOCUMENTO', F.when(F.col('des_cpf').isNotNull(), F.lpad(F.col('des_cpf'),11,"0")).otherwise(F.lpad(F.col('des_regis'), 11, "0")))

# COMMAND ----------

df = df.drop('des_regis').drop('des_cpf')

# COMMAND ----------

# DBTITLE 1,date para datetype
for col in df.columns:
  if 'dat_' in col:
    df = df.withColumn(col, F.to_date(F.col(col), format='yyyyMMdd'))

# COMMAND ----------

# DBTITLE 1,valores para integer round 2
#14 inteiros e 4 decimais sem ',' ou '.' preencher com zeros à esquerda

for col in df.columns:
  if 'val_' in col:
    df = df.withColumn(col, F.round((F.col(col).cast(T.IntegerType())/10000),2))

# COMMAND ----------

data_arquivo = datetime.datetime(int(pasta_escolhida[:4]), int(pasta_escolhida[4:6]), int(pasta_escolhida[6:]))

df = escreve_variavel_resposta_acordo(df, 'bmg', data_arquivo, 'DOCUMENTO', varName='VARIAVEL_RESPOSTA', drop_null = True)

# COMMAND ----------

df = df.filter(F.col('VARIAVEL_RESPOSTA').isNotNull())

# COMMAND ----------

# DBTITLE 1,escrevendo arquivo
df.write.mode('overwrite').parquet(os.path.join(caminho_joined_trusted, pasta_escolhida+'.PARQUET'))
os.path.join(caminho_joined_trusted, pasta_escolhida+'.PARQUET')

# COMMAND ----------

# DBTITLE 1,gerando amostra
df2 = df.filter(F.col('VARIAVEL_RESPOSTA').isNotNull())

df_representativo, df_aleatorio = gera_sample(df2)

# COMMAND ----------

#df_representativo.coalesce(1).write.option('header', 'True').option('delimiter',';').csv(os.path.join(caminho_sample, pasta_escolhida, pasta_escolhida+'_amostra_representativa.csv'))
#df_aleatorio.coalesce(1).write.option('header', 'True').option('delimiter',';').csv(os.path.join(caminho_sample, pasta_escolhida, pasta_escolhida+'_amostra_aleatoria.csv'))
os.path.join(caminho_sample, pasta_escolhida, pasta_escolhida+'_amostra_aleatoria.csv')