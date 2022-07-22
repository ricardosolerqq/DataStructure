# Databricks notebook source
# MAGIC %run "/pipe_modelos/extrator_recupera/v2/00.le_regras_disponibiliza_variaveis_v2"

# COMMAND ----------

import datetime

# COMMAND ----------

# DBTITLE 1,configurando widget de credores
credores_recupera = ['agibank','bmg','fort','tribanco','trigg','valia','zema']
dbutils.widgets.dropdown('CREDOR_ESCOLHIDO', 'bmg', credores_recupera)
credor = dbutils.widgets.get('CREDOR_ESCOLHIDO')

credor = Credor(credor)

# COMMAND ----------

files = []
for file in os.listdir(credor.caminho_trusted_dbfs):
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
for file in os.listdir(os.path.join(credor.caminho_trusted_dbfs, pasta_escolhida)):
  dict_dfs.update({file.split(".")[0]: spark.read.parquet(os.path.join(credor.caminho_trusted, pasta_escolhida, file)).alias(file.split(".")[0])})
pasta_escolhida

# COMMAND ----------

display(dict_dfs['contratos'].groupBy('cod_produt').agg(F.count(F.col('cod_produt'))))

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
    df = df.withColumn(col, F.when(F.length(F.col(col).cast(T.IntegerType()))==8, F.to_date(F.col(col), format='yyyyMMdd')).otherwise(None))

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

display(df.select('des_fones_resid').filter(F.col('des_fones_resid').contains('536857828')).orderBy(F.desc(F.col('des_fones_resid'))))

# COMMAND ----------

display(df.select('des_fones_resid').orderBy(F.desc(F.col('des_fones_resid'))))

# COMMAND ----------

# DBTITLE 1,gerando variavel resposta
"""
data_arquivo = datetime.datetime(int(pasta_escolhida[:4]), int(pasta_escolhida[4:6]), int(pasta_escolhida[6:]))

df = escreve_variavel_resposta_acordo(df, 'bmg', data_arquivo, 'DOCUMENTO', varName='VARIAVEL_RESPOSTA', drop_null = True)
"""

# COMMAND ----------

# DBTITLE 1,escrevendo arquivo
df.write.mode('overwrite').parquet(os.path.join(credor.caminho_joined_trusted, pasta_escolhida+'.PARQUET'))
os.path.join(credor.caminho_joined_trusted, pasta_escolhida+'.PARQUET')

# COMMAND ----------

df.coalesce(1).write.mode('overwrite').option('sep', ';').option('header', 'True').csv(os.path.join(credor.caminho_joined_trusted, 'temp_output.csv'))

# COMMAND ----------

for file in dbutils.fs.ls(os.path.join(credor.caminho_joined_trusted, 'temp_output.csv')):
  if file.name.split('.')[-1] == 'csv':
    print (file)
    dbutils.fs.cp(file.path, os.path.join(credor.caminho_joined_trusted, pasta_escolhida+'.csv'))
dbutils.fs.rm(os.path.join(credor.caminho_joined_trusted, 'temp_output.csv'), True)

# COMMAND ----------

# DBTITLE 1,gerando amostra
"""
df2 = df.filter(F.col('VARIAVEL_RESPOSTA').isNotNull())

df_representativo, df_aleatorio = gera_sample(df2)
"""