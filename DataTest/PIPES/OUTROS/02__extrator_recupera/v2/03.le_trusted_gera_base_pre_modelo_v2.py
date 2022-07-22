# Databricks notebook source
# MAGIC %run "/pipe_modelos/extrator_recupera/v2/00.le_regras_disponibiliza_variaveis_v2"

# COMMAND ----------

import datetime

# COMMAND ----------

# DBTITLE 1,configurando widget de credores
dbutils.widgets.text('credor', '', '')
credores = dbutils.widgets.get('credor')

credor = Credor(credores)

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

for i in dict_dfs:
  print(i)

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

# MAGIC %run /env_mvp/fork_improvements/FUNCTIONS

# COMMAND ----------

if criar_variavel_resposta != 'False':
  ##Leitura do ENV
  env = Env()
  info = spark.read.format('delta').load(os.path.join(env.caminho_trusted, 'person_'+'info'+'.PARQUET'))
  debts = spark.read.format('delta').load(os.path.join(env.caminho_trusted, 'person_'+'debts'+'.PARQUET'))
  deals = spark.read.format('delta').load(os.path.join(env.caminho_trusted, 'person_'+'deals'+'.PARQUET'))
  installments = spark.read.format('delta').load(os.path.join(env.caminho_trusted, 'person_'+'installments'+'.PARQUET'))
  
  ##filtrando informações da env
  info = info.select(F.col('_id'),F.col('document'))
  deals = deals.filter(F.col('creditor')==credores)
  installments = installments.filter((F.col('creditor')==credores)&(F.col('status')=='paid'))
  installments = installments.orderBy(F.col('_id'),F.col('installment').desc())
  installments = installments.dropDuplicates(['_id','dealID'])


# COMMAND ----------

if criar_variavel_resposta != 'False':
  from pyspark.sql.types import DateType
  
  ##Criando dataframe com dados da env
  df2 = deals.join(info, on='_id', how='inner').select(F.col('_id'),F.col('document'),F.col('dealID'),F.col('createdAt')[0:10].alias('createdAt'),F.explode(F.col('debtsIDs')).alias('debtsIDs'))
  df2 = df2.withColumn('debtsIDs', F.split((F.col('debtsIDs')),':')[0])
  df2 = df2.filter(F.col('status') != 'error')
  df2 = df2.withColumn('createdAt', (F.col('createdAt').cast(DateType())))
  df2 = df2.orderBy(F.col('createdAt').desc(),F.col('dealID').desc()).dropDuplicates(['document','debtsIDs'])
  df2 = df2.join(installments.select(F.col('_id'),F.col('dealID'),F.col('installmentID'),F.col('installment'),F.col('paidAmount')), on=(['_id','dealID']), how='left')
  print('Dataframe ENV')
  df2.show(5,False)
  
  ##Criando tabela de analise completa da env e etl
  df.join(df2, (df.des_contr == df2.debtsIDs) | (df.DOCUMENTO == df2.document)).createOrReplaceTempView('analise')
  
  ##Criando variavel resposta de acordos
  spark.sql('select *, CASE WHEN CAST(months_between(createdAt, dat_movto) AS INT) >= 0 THEN True else False END as ACORDO from analise').createOrReplaceTempView('analise')
  
  ##Criando dataset com variavel resposta de pagamentos e pagamento a vista
  df3 = spark.sql('select *, CASE WHEN installment is not null and ACORDO = True THEN True else False END as PAGAMENTO, CASE WHEN installment = 1 and ACORDO = True THEN True else False END as PAGAMENTO_A_VISTA from analise')
  df3 = df3.orderBy(F.col('dat_movto').desc()).dropDuplicates(['DOCUMENTO','des_contr'])
  df3 = df3.drop('installmentID').drop('debtsIDs').drop('document').drop('dealID').drop('_id')
  

# COMMAND ----------

# DBTITLE 1,escrevendo arquivo final - csv
if criar_variavel_resposta != 'False':
  import time
  createdAt = time.strftime("%Y-%m-%d")
  
  df3.coalesce(1).write.option('header', 'True').option('delimiter',';').csv(os.path.join(credor.caminho_sample, 'temp'), mode='overwrite')
  for file in dbutils.fs.ls(os.path.join(credor.caminho_sample, 'temp')):
      if file.name.split('.')[-1] == 'csv':
        dbutils.fs.cp(file.path, os.path.join(credor.caminho_sample, "amostra_representativa_"+createdAt+".csv"))
  dbutils.fs.rm(os.path.join(credor.caminho_sample, 'temp'), True)

  
else:
  df.coalesce(1).write.mode('overwrite').option('sep', ';').option('header', 'True').csv(os.path.join(credor.caminho_joined_trusted, 'temp_output.csv'))
  for file in dbutils.fs.ls(os.path.join(credor.caminho_joined_trusted, 'temp_output.csv')):
    if file.name.split('.')[-1] == 'csv':
      print (file)
      dbutils.fs.cp(file.path, os.path.join(credor.caminho_joined_trusted,'pre_output.csv'))
  dbutils.fs.rm(os.path.join(credor.caminho_joined_trusted, 'temp_output.csv'), True)

# COMMAND ----------

dbutils.notebook.exit('OK')