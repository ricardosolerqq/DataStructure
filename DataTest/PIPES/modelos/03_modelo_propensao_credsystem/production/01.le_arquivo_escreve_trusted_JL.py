# Databricks notebook source
# MAGIC %md
# MAGIC ###Imports

# COMMAND ----------

# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

try:
  dbutils.widgets.remove('ARQUIVO_ESCOLHIDO')
except:
  pass

# COMMAND ----------

import os
import datetime
import zipfile
import pyspark.sql.functions as F

blob_account_source_prd = "qqprd"
blob_account_source_ml = "qqdatastoragemain"
blob_container_source_prd = "qq-integrator"
blob_container_source_ml = "ml-prd"

prefix = "etl/credsystem/processed"

mount_blob_storage_key(dbutils,blob_account_source_prd,blob_account_source_prd,'/mnt/qq-integrator')
mount_blob_storage_key(dbutils,blob_account_source_ml,blob_account_source_ml,'/mnt/ml-prd')

caminho_base = 'mnt/qq-integrator/etl/credsystem/processed'
caminho_base_dbfs = '/dbfs/mnt/qq-integrator/etl/credsystem/processed'
caminho_trusted = '/mnt/ml-prd/ml-data/propensaodeal/credsystem/trusted'
caminho_sample = '/mnt/ml-prd/ml-data/propensaodeal/credsystem/sample'

# COMMAND ----------

# DBTITLE 1,Configurando Processo e Arquivo a ser Tratado
dbutils.widgets.dropdown('processamento', 'auto', ['auto', 'manual'])
processo_auto = dbutils.widgets.get('processamento')
if processo_auto == 'auto':
  processo_auto = True
else:
  processo_auto = False
  
if processo_auto:
  try:
    dbutils.widgets.remove('ARQUIVO_ESCOLHIDO')
  except:
    pass
  arquivo_escolhido = list(get_creditor_etl_file_dates('credsystem', latest=True))[0]
  arquivo_escolhido_path = os.path.join(caminho_base, arquivo_escolhido)
  arquivo_escolhido_path_dbfs = os.path.join('/dbfs',caminho_base, arquivo_escolhido)
  
else:
  ##list_arquivo = os.listdir('/dbfs/mnt/qq-integrator/etl/credsystem/processed')
  ##lista_arquivo = []
  ##for i in lista_arquivos:
  ##  if "QUERO_QUITAR_" in i:
  ##    lista_arquivo.append(i)
  ##dict_arq = {datetime.date(int(item.split('QUERO_QUITAR_')[1].split('.')[0].split('-')[2]), int(item.split('QUERO_QUITAR_')[1].split('.')[0].split('-')[1]), int(item.split('QUERO_QUITAR_')[1].split('.')[0].split('-')[0])):item for item in lista_arquivos}
  ##dbutils.widgets.dropdown('ARQUIVO_ESCOLHIDO', max(str(item) for item in dict_arq), [str(item) for item in dict_arq])
  ##arquivo_escolhido = dbutils.widgets.get('ARQUIVO_ESCOLHIDO')
  arquivo_escolhido_path = 'mnt/qq-integrator/etl/credsystem/processed/QUERO_QUITAR_202112032100.CSV' ##os.path.join(caminho_base,dict_arq[datetime.date(int(arquivo_escolhido.split('-')[0]), int(arquivo_escolhido.split('-')[1]), int(arquivo_escolhido.split('-')[2]))])
  ##arquivo_escolhido_path_dbfs = os.path.join('/dbfs',caminho_base,dict_arq[datetime.date(int(arquivo_escolhido.split('-')[0]), int(arquivo_escolhido.split('-')[1]), int(arquivo_escolhido.split('-')[2]))])

arquivo_escolhido_fileformat = arquivo_escolhido_path.split('.')[-1]
arquivo_escolhido_fileformat


#arquivo_escolhido_path
file = arquivo_escolhido_path.split('/')[-1]

fileDate = (file.split('.')[0].split('_')[2])[0:8]

file, fileDate 

# COMMAND ----------

# DBTITLE 1,Criando Dataframe Spark
df = spark.read.option('delimiter',';').option('header', 'True').csv("/"+arquivo_escolhido_path)

# COMMAND ----------

# DBTITLE 1,Tratando Campo de Data
df = df.withColumn('nova_data', F.to_date(F.col('PRIMEIRO_VENCIMENTO'), 'ddMMMyyyy'))
df = df.drop("PRIMEIRO_VENCIMENTO")
df = df.withColumnRenamed("nova_data", "PRIMEIRO_VENCIMENTO")
df = df.withColumn('DATA_ARQ', F.to_date(F.lit(fileDate), 'yyyyMMdd'))
df.show(10, False)

# COMMAND ----------

spark.conf.set('spark.sql.caseSensitive', True)

DATA_PATH = "/dbfs/SCHEMAS/DATA_SCHEMA.json"
with open(DATA_PATH) as f:
  SCHEMA = T.StructType.fromJson(json.load(f))

#SCHEMA = spark.read\
#            .format( "com.mongodb.spark.sql.DefaultSource")\
#            .option('spark.mongodb.input.sampleSize', 50000)\
#            .option("database", "qq")\
#            .option("spark.mongodb.input.collection", "col_person")\
#            .option("badRecordsPath", "/tmp/badRecordsPath")\
#            .load().limit(1).schema

# COMMAND ----------

# DBTITLE 1,Info
dfInfo = spark.read.schema(SCHEMA).json('dbfs:/mnt/bi-reports/export_full/person_without_project/20220403/datalake_202204030025_0000000000.jsonl')\
.select('_id','document',F.col('info.phones.tags').alias('tag_telefone'),F.col('info.gender').alias('gender'),F.col('info.birthDate').alias('birthDate'),F.col('info.addresses.state').alias('state'),\
        F.col('info.emails.tags').alias('tag_email'), F.col('info.emails.domain').alias('domain'))\
.withColumn('birthDate', F.to_date(F.col('birthDate')[0:10], 'yyyy-M-d'))\
.withColumn('age', F.floor(F.months_between(F.current_date(), F.col('birthDate'))/12)).drop('birthDate')

listTel= ['skip:hot','skip:alto','skip:medio','skip:baixo','skip:nhot','sem_tags']
listMail= ['skip:hot','skip:alto','skip:medio','skip:baixo','skip:nhot','sem_tags']

for item in range(10):
  for i in listTel:
    if 'sem_tags' in i:
      dfInfo = dfInfo.withColumn('Telefone{0}_{1}'.format(item+1,i),F.when(F.array_contains(F.col('tag_telefone')[item], i).isNull(), True).otherwise(False))
    else:
      dfInfo = dfInfo.withColumn('Telefone{0}_{1}'.format(item+1,i),F.when(F.array_contains(F.col('tag_telefone')[item], i) == True, True).otherwise(False))
      
dfInfo = dfInfo.drop('tag_telefone')

for item in range(3):
  for i in listMail:
    if 'sem_tags' in i:
      dfInfo = dfInfo.withColumn('Email{0}_{1}'.format(item+1,i),F.when(F.array_contains(F.col('tag_email')[item], i).isNull(), True).otherwise(False))
    else:
      dfInfo = dfInfo.withColumn('Email{0}_{1}'.format(item+1,i),F.when(F.array_contains(F.col('tag_email')[item], i) == True, True).otherwise(False))
  dfInfo = dfInfo.withColumn('Email{0}_dominio'.format(item+1),F.col('domain')[item])
  
dfInfo = dfInfo.drop('tag_email')
dfInfo = dfInfo.drop('domain')

for item in range(3):
  dfInfo = dfInfo.withColumn('UF{0}'.format(item+1,i),F.upper(F.when(F.col('state')[item] == '!!!<<Cadastrar>>!!!', None).otherwise(F.col('state')[item])))
  
dfInfo = dfInfo.drop('state')


#dfInfo.printSchema()

# COMMAND ----------

# DBTITLE 1,Dividas
dfDebts = spark.read.schema(SCHEMA).json('dbfs:/mnt/bi-reports/export_full/person_without_project/20220403/datalake_202204030025_0000000000.jsonl')\
.select('_id', F.col('debts.contract').alias('contract'),F.col('debts.portfolio').alias('portfolio'), F.col('debts.originalAmount').alias('originalAmount'), F.col('debts.product').alias('product'),
        F.col('debts.tags').alias('tag_divida'), F.col('debts.creditor').alias('creditor'), F.col('debts.dueDate').alias('dueDate'))\
.withColumn('debts',F.arrays_zip('contract','portfolio','originalAmount', 'product', 'creditor', 'tag_divida', 'dueDate'))\
.select('_id', F.explode('debts').alias('debts'))\
.select('_id', 'debts.contract', 'debts.portfolio','debts.originalAmount', 'debts.product', 'debts.tag_divida', 'debts.creditor', F.to_date(F.col('debts.dueDate')[0:10], 'yyyy-M-d').alias('dueDate'))\
#.withColumn('tag_divida', F.explode(F.col('tag_divida')))\
#.filter((F.col('creditor') == 'credsystem'))

listDiv = ['rank:a','rank:b','rank:c','rank:d','rank:e', 'sem_tags']


for i in listDiv:
  if 'sem_tags' in i:
    dfDebts = dfDebts.withColumn('SkipDivida_{0}'.format(i),F.when(F.array_contains(F.col('tag_divida'), i).isNull(), True).otherwise(False))
  else:
    dfDebts = dfDebts.withColumn('SkipDivida_{0}'.format(i),F.when(F.array_contains(F.col('tag_divida'), i) == True, True).otherwise(False))
    
dfDebts = dfDebts.drop('tag_divida')

dfDebts = dfDebts.distinct()

#dfDebts.printSchema()

# COMMAND ----------

# DBTITLE 1,Tratamentos e joins
##Filtrando apenas dividas do credor
dfDebtsCre = dfDebts.filter(F.col('creditor') == 'credsystem')
dfValCredor =  dfDebtsCre.select('_id','contract')

##Criando DF de divida nos demais credores
dfDebtsOut = dfDebts.filter(F.col('creditor') != 'credsystem').select('_id','creditor')
dfDebtsOut = dfDebtsOut.dropDuplicates(['_id'])

#Join dos DFs de Info e Debts do credor
dfBase = dfInfo.join(dfDebtsCre, on='_id')

##Join do arquivo ETL com os dados do mongo
dfArq = df.join(dfBase, (df.CONTRATO == dfBase.contract) & (df.CPF == dfBase.document), 'left')
dfArq = dfArq.drop('creditor').drop('document').drop('originalAmoun').drop('contract')

#Join do DF completo com o DF de divida em outros credores e criando FLAG
dfArq = dfArq.join(dfDebtsOut, on= '_id', how='left')
dfArq = dfArq.withColumn('Divida_Outro_Credor', F.when(F.col('creditor').isNull(),False).otherwise(True)).drop('creditor')

##Criando Flag de outras dividas no mesmo credor
dfValCredor = dfValCredor.join(dfArq, (dfValCredor.contract == dfArq.CONTRATO) & (dfValCredor._id == dfArq._id ), 'left').select(dfValCredor._id,'contract','CPF').filter(F.col('CPF').isNull()).drop('CPF')
dfArq = dfArq.join(dfValCredor, on='_id', how='left')
dfArq = dfArq.withColumn('Divida_Mesmo_Credor', F.when(F.col('contract').isNull(),False).otherwise(True)).drop('contract')


# COMMAND ----------

display(dfArq)

# COMMAND ----------

# MAGIC %md
# MAGIC ###MONGO

# COMMAND ----------

# DBTITLE 1,Autenticação
#PYSPARK
SCHEMA_AUTH = spark.read\
        .format("com.mongodb.spark.sql.DefaultSource")\
        .option('spark.mongodb.input.database', "qq")\
        .option('spark.mongodb.input.collection', "col_authentication")\
        .load().limit(1).schema


DATA_Auth = spark.read.schema(SCHEMA_AUTH)\
        .format("com.mongodb.spark.sql.DefaultSource")\
        .option('spark.mongodb.input.database', "qq")\
        .option('spark.mongodb.input.collection', "col_authentication")\
        .load()
#display(dfAuth)

# COMMAND ----------

##Filtrando colunas de autenticações
dfAuth = DATA_Auth.select('tokenData.document','tokenData.contact','tokenData.contract','tokenData.creditor','createdAt')\
        .filter(F.col('creditor')=='credsystem')\
        .filter(F.col('createdAt') <= (F.to_date(F.lit(fileDate), 'yyyyMMdd'))).orderBy(F.col('createdAt').desc())


##Contagem de autenticação por CPF
dfAuthQtd = dfAuth.groupBy('document').count()\
            .select(F.col('document').alias('CPF'), F.col('count').alias('QTD_AUTH'))

##Contagem de autenticação por CPF e Contrato --Não disponivel para Credsystem
#dfAuthQtdContract = dfAuth.groupBy('document','contract').count()\
#            .select(F.col('document').alias('CPF'), F.col('contract').alias('CONTRATO'), F.col('count').alias('QTD_AUTH_CONTRATO'))


display(dfAuth)

# COMMAND ----------

##Criando coluna de quantidade de autenticações por CPF no DF principal
dfArq = dfArq.join(dfAuthQtd, on= 'CPF', how = 'left')\
        .withColumn('QTD_AUTH', (F.when(F.col('QTD_AUTH').isNull(), 0).otherwise(F.col('QTD_AUTH'))))


##Criando coluna de quantidade de autenticações por CPF e Contrato no DF principal --Não disponivel para Credsystem
#dfArq = dfArq.join(dfAuthQtdContract, on= 'CPF', how = 'left')\
#            .withColumn('QTD_AUTH_CONTRATO', (F.when(F.col('QTD_AUTH_CONTRATO').isNull(), 0).otherwise(F.col('QTD_AUTH_CONTRATO'))))

# COMMAND ----------

display(dfArq)

# COMMAND ----------

# DBTITLE 1,Simulação
#PYSPARK
SCHEMA_SIMU = spark.read\
        .format("com.mongodb.spark.sql.DefaultSource")\
        .option('spark.mongodb.input.database', "qq")\
        .option('spark.mongodb.input.collection', "col_simulation")\
        .load().limit(1).schema


DATA_Simu = spark.read.schema(SCHEMA_SIMU)\
        .format("com.mongodb.spark.sql.DefaultSource")\
        .option('spark.mongodb.input.database', "qq")\
        .option('spark.mongodb.input.collection', "col_simulation")\
        .load()
#display(DATA)

# COMMAND ----------

dfSimu = DATA_Simu.select('personID','creditor','createdAt')\
        .filter(F.col('creditor')=='credsystem')\
        .filter(F.col('createdAt') <= (F.to_date(F.lit(fileDate), 'yyyyMMdd'))).orderBy(F.col('createdAt').desc())

##Contagem de simulações por CPF
dfSimuQtd = dfSimu.groupBy('personID').count()\
            .select(F.col('personID').alias('_id'), F.col('count').alias('QTD_SIMU'))

#display(dfSimu)

# COMMAND ----------

##Criando coluna de quantidade de simulações por CPF no DF principal
dfArq = dfArq.join(dfSimuQtd, on= '_id', how = 'left')\
        .withColumn('QTD_SIMU', (F.when(F.col('QTD_SIMU').isNull(), 0).otherwise(F.col('QTD_SIMU'))))

#display(dfArq.orderBy(F.col('QTD_SIMU').desc()))

# COMMAND ----------

# DBTITLE 1,Interações
# MAGIC %fs ls /mnt/bi-reports/engenharia_de_dados/ad_hoc/valid_interaction_20220422

# COMMAND ----------

schemaInt = spark.read.parquet("/mnt/bi-reports/engenharia_de_dados/ad_hoc/valid_interaction_20220422/folder_date=2022-04-09/").limit(1).schema
df_int = spark.read.schema(schemaInt).parquet('/mnt/bi-reports/engenharia_de_dados/ad_hoc/valid_interaction_20220422/folder_date=202*/')


# COMMAND ----------

display(df_int)

# COMMAND ----------

df_interaction = df_int.select(F.col('personID').alias('_id'), 'channel', 'creditor', 'document','contact', 'status', 'type', 'sentAt_c', 'createdAt_c')\
                .filter((F.col('creditor') == 'santander')&(~F.col('status').isin(["not_answered","not_received"]))&(F.col('document')=='79648169934'))


# COMMAND ----------

display(df_interaction)

# COMMAND ----------

display(df_interaction.groupBy('type').count())

# COMMAND ----------

# MAGIC %md
# MAGIC ###VARIAVEL RESPOSTA ACORDO

# COMMAND ----------

#Buscando todos acordos do credor
dfRespostaAcordo = variavel_resposta_acordo_atualizado('credsystem', datetime.datetime(2021,12,3), datetime.datetime(2022,1,3))
dfRespostaAcordo = spark.createDataFrame(dfRespostaAcordo)\
                  .select(F.col('document').alias('CPF'),F.col('contract').alias('CONTRATO'),F.col('varName').alias('VARIAVEL_RESPOSTA'), F.col('createdAt').alias('DT_ACORDO'))
display(dfRespostaAcordo)

# COMMAND ----------

dfRespostaAcordo = dfRespostaAcordo.withColumn('QTD_DIAS', F.datediff('DT_ACORDO',F.to_date(F.lit(fileDate), 'yyyyMMdd')))\
.withColumn('QTD_MES', F.floor(F.months_between('DT_ACORDO',F.to_date(F.lit(fileDate), 'yyyyMMdd'))))
#fileDate

# COMMAND ----------

dfRespostaAcordo.withColumn('ACORDO_7dias', F.when(F.col('QTD_DIAS') <= 7, True).otherwise(False))\
.withColumn('ACORDO_15dias', F.when((F.col('QTD_DIAS') > 7) & (F.col('QTD_DIAS') <= 15), True).otherwise(False))\
.withColumn('ACORDO_30dias', F.when((F.col('QTD_DIAS') > 15) & (F.col('QTD_DIAS') <= 30), True).otherwise(False)).show(100,False)



# COMMAND ----------

#Criando variavel resposta de acordo
dfArq = dfArq.join(dfRespostaAcordo, on=(['CPF','CONTRATO']), how= 'left')
dfArq = dfArq.withColumn('VARIAVEL_RESPOSTA', (F.when(F.col('VARIAVEL_RESPOSTA').isNull(), False).otherwise(F.col('VARIAVEL_RESPOSTA'))))
display(dfArq)
###Criar coluna com quantidade de dias entre a entrada do arquivo e fechamendo de acordo

# COMMAND ----------

# MAGIC %md
# MAGIC ###VARIAVEL RESPOSTA PAGAMENTOS

# COMMAND ----------

#Recuperando contratos da deals
dfDeals = spark.read.schema(SCHEMA).json('dbfs:/mnt/bi-reports/export_full/person_without_project/20220403/datalake_202204030025_001*')\
.select('_id',  F.col('deals.creditor').alias('creditor'), F.col('deals._id').alias('dealID'),(F.col('deals.offer.debts')).alias('debts'))\
.withColumn('deals',F.arrays_zip('dealID','debts','creditor'))\
.select('_id', F.explode('deals').alias('deals'))\
.select('_id', F.col('deals.dealID').alias('dealID'), F.explode(F.col('deals.debts')).alias('debts'), F.col('deals.creditor').alias('creditor'))\
.filter((F.col('creditor') == 'credsystem'))\
.select('_id', 'dealID', 'debts.contract')\


#display(dfDeals)

#dfDebts.printSchema()

# COMMAND ----------

#Buscando todos pagamentos do credor
dfRespostaPagamento = variavel_resposta_pagamento('credsystem', datetime.datetime(2021,12,3), datetime.datetime(2022,1,3))
dfRespostaPagamento = spark.createDataFrame(dfRespostaPagamento)

#Recuperando codigo do contrato
dfRespostaPagamento = dfRespostaPagamento.join(dfDeals, on=('dealID'), how= 'inner').drop('Pagto_Parcela1').drop('Pagto_Demais_Parcelas').drop('dealID').drop('_id')
display(dfRespostaPagamento)

# COMMAND ----------

#Criando variavel resposta de pagamento
dfArq = dfArq.join(dfRespostaPagamento, (dfArq.CPF == dfRespostaPagamento.document) & (dfArq.CONTRATO == dfRespostaPagamento.contract), how= 'left').drop('document').drop('contract')
dfArq = dfArq.withColumn('PAGTO_A_VISTA', (F.when(F.col('PAGTO_A_VISTA').isNull(), False).otherwise(F.col('PAGTO_A_VISTA'))))\
.withColumn('PAGTO', (F.when(F.col('PAGTO').isNull(), False).otherwise(F.col('PAGTO'))))\
.withColumn('Qtd_parcelas', (F.when(F.col('Qtd_parcelas').isNull(), 0).otherwise(F.col('Qtd_parcelas'))))
display(dfArq)

# COMMAND ----------

dbutils.widgets.dropdown('ESCREVER_VARIAVEL_RESPOSTA', 'False', ['False', 'True'])
escreverVariavelResposta = dbutils.widgets.get('ESCREVER_VARIAVEL_RESPOSTA')
if escreverVariavelResposta == 'True':

  for file in dbutils.fs.ls(os.path.join(caminho_sample, 'aleatorio_temp')):
    if file.name.split('.')[-1] == 'csv':
      dbutils.fs.cp(file.path, os.path.join(caminho_sample, 'base_aleatoria.csv'))
  dbutils.fs.rm(os.path.join(caminho_sample, 'aleatorio_temp'), True)


else:
  #Escrevendo DataFrame
  df.coalesce(1).write.option('sep', ';').option('header', 'True').csv(os.path.join(caminho_trusted, 'tmp'))
  for file in dbutils.fs.ls(os.path.join(caminho_trusted, 'tmp')):
    if file.name.split('.')[-1] == 'csv':
      dbutils.fs.cp(file.path, os.path.join(caminho_trusted, arquivo_escolhido.split('.')[0]+'.csv'))
  dbutils.fs.rm(os.path.join(caminho_trusted, 'tmp'), True)