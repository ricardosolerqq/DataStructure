# Databricks notebook source
import datetime
ArqDate = datetime.datetime.today() - datetime.timedelta(days=1)
date = datetime.datetime.today()
ArqDate = str(ArqDate.year)+'-'+str(ArqDate.month).zfill(2)+'-'+str(ArqDate.day).zfill(2)
date = str(date.year)+'-'+str(date.month).zfill(2)+'-'+str(date.day).zfill(2)

date = '2022-07-20'
ArqDate = '2022-07-18'
date, ArqDate

# COMMAND ----------

try:
  dbutils.widgets.remove('ARQUIVO_ESCOLHIDO')
except:
  pass

# COMMAND ----------

# DBTITLE 1,Carregamento de funções pré-definidas
# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

from pyspark.sql.functions import substring
from pyspark.sql.functions import desc
import os
import datetime
import zipfile

# COMMAND ----------

# DBTITLE 1,Definição dos diretórios do blob
# Todos realizados de forma manual
prefix = "etl/hoepers/processed"

caminho_base = '/mnt/qq-integrator/etl/hoepers/processed'
caminho_base_dbfs = '/dbfs/mnt/qq-integrator/etl/hoepers/processed'
caminho_trusted = '/mnt/ml-prd/ml-data/propensaodeal/hoepers/trusted'
caminho_base_models = '/mnt/ml-prd/ml-data/base_models'

file_name = 'Base_hoepers_QQ_Model_'+date+'.csv'

# COMMAND ----------

file = ''
df = spark.read.options(header= True, delimiter = ';').csv(os.path.join(caminho_base, file))

display(df)

# COMMAND ----------

dates = []
files_dates = get_creditor_etl_file_dates('hoepers', as_date=True)
for file in files_dates:
  if 'base_diaria' in file.lower():
    if files_dates[file] not in dates:
      dates.append(files_dates[file])
dates = sorted(dates, reverse=True)

# COMMAND ----------

arquivos_escolhidos = []
def escolhe_arquivos(files_dates, datas_escolhidas):
  if type(datas_escolhidas) != list:
    datas_escolhidas = [datas_escolhidas]
  lista_dates = []
  for date in datas_escolhidas:
    if type(date)==str:
      date = date.split('-')
      date = [int(d) for d in date]
      date = datetime.date(date[0], date[1], date[2])
    lista_dates.append(date)
  arquivos_escolhidos = []
  for date in lista_dates:
    for file in files_dates:
      if files_dates[file] == date:
        if 'base_diaria' in file.lower():
          arquivos_escolhidos.append([date.strftime('%Y%m%d'), file])
  return arquivos_escolhidos

dbutils.widgets.dropdown('MODE', 'AUTO', ['AUTO', 'MANUAL'])
if dbutils.widgets.get('MODE') == 'AUTO':
  autoMode = True
else:
  autoMode = False
  
if autoMode:
  try:
    dbutils.widgets.remove('DATAS_ARQUIVOS')
  except:
    pass
    datas_escolhidas = max(dates)
else:
  dbutils.widgets.multiselect('DATAS_ARQUIVOS', str(max(dates)), [str(date) for date in dates])
  datas_escolhidas = dbutils.widgets.get('DATAS_ARQUIVOS')
  datas_escolhidas = datas_escolhidas.split(',')
  
arquivos_escolhidos = escolhe_arquivos(files_dates, datas_escolhidas)  

print ('autoMode',autoMode)
print ('arquivos a processar:')
for arq in arquivos_escolhidos:
  print ('\t',arq)

# COMMAND ----------

i = 0
for file in arquivos_escolhidos:
  file = file[1]
  print ('lendo aquivo ',file)
  if i == 0:
    i += 1
    df = spark.read.options(header= True, delimiter = ';').csv(os.path.join(caminho_base, file))
  else:
    dfu = spark.read.options(header= True, delimiter = ';').csv(os.path.join(caminho_base, file))
    try:
      df = df.union(dfu)
    except:
      pass
df = df.distinct()

# COMMAND ----------

df.count()

# COMMAND ----------

fileDate = arquivos_escolhidos[0][0]

fileDate

# COMMAND ----------

display(df)

# COMMAND ----------

# DBTITLE 1,Info
dfInfo = spark.read.format('delta').load(os.path.join(caminho_base_models, 'Info',ArqDate)).filter(F.col('documentType')=='cpf')

dfInfo = dfInfo.select('_id','document','documentType','tag_telefone','state','tag_email','domain')


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
  
dfInfo = dfInfo.drop('state').orderBy(F.col('_id').asc())

# COMMAND ----------

# DBTITLE 1,Debts
dfDebts = spark.read.format('delta').load(os.path.join(caminho_base_models, 'Debts/CPF',ArqDate))

listDiv = ['rank:a','rank:b','rank:c','rank:d','rank:e', 'sem_tags']

for i in listDiv:
  if 'sem_tags' in i:
    dfDebts = dfDebts.withColumn('SkipDivida_{0}'.format(i),F.when(F.array_contains(F.col('tag_divida'), i).isNull(), True).otherwise(False))
  else:
    dfDebts = dfDebts.withColumn('SkipDivida_{0}'.format(i),F.when(F.array_contains(F.col('tag_divida'), i) == True, True).otherwise(False))
    
dfDebts = dfDebts.drop('tag_divida')

dfDebts = dfDebts.distinct()

dfDebts = dfDebts.join(dfDebts.groupBy('_id','creditor').agg(F.min('INICIO_DIVIDA').alias('INICIO_DIVIDA')),on=(['_id','creditor','INICIO_DIVIDA']),how='leftsemi')

# COMMAND ----------

##Filtrando apenas dividas do credor
dfDebtsCre = dfDebts.filter(F.col('creditor') == 'hoepers').orderBy(F.col('_id').asc())
dfValCredor =  dfDebtsCre.select('_id','contract')

##Criando DF de divida nos demais credores
dfDebtsOut = dfDebts.filter(F.col('creditor') != 'hoepers').select('_id','creditor').orderBy(F.col('_id').asc())
dfDebtsOut = dfDebtsOut.dropDuplicates(['_id'])

#Join dos DFs de Info e Debts do credor
dfBase = dfInfo.join(dfDebtsCre.drop('document'), on='_id').orderBy(F.col('document').asc(),F.col('contract').asc())

##Join do arquivo ETL com os dados do mongo
dfArq = df.join(dfBase, (df.nr_cpf == dfBase.document), 'left')
dfArq = dfArq.drop('creditor').drop('document').drop('originalAmoun').drop('contract')

#Join do DF completo com o DF de divida em outros credores e criando FLAG
dfArq = dfArq.join(dfDebtsOut, on= '_id', how='left')
dfArq = dfArq.withColumn('Divida_Outro_Credor', F.when(F.col('creditor').isNull(),False).otherwise(True)).drop('creditor')

##Criando Flag de outras dividas no mesmo credor
dfValCredor = dfValCredor.join(dfArq, (dfValCredor.contract == dfArq.nr_conta) & (dfValCredor._id == dfArq._id ), 'left').select(dfValCredor._id,'contract','nr_conta').filter(F.col('nr_conta').isNull()).drop('nr_conta').orderBy(F.col('_id').asc())
dfArq = dfArq.join(dfValCredor, on='_id', how='left')
dfArq = dfArq.withColumn('Divida_Mesmo_Credor', F.when(F.col('contract').isNull(),False).otherwise(True)).drop(*['contract','nr_conta'])

dfArq = dfArq.distinct()

# COMMAND ----------

# DBTITLE 1,Simulação
DATA_Simu = spark.read.format('delta').load(os.path.join(caminho_base_models, 'Simulation',ArqDate))

# COMMAND ----------

dfSimu = DATA_Simu.select('personID','creditor','createdAt')\
        .filter(F.col('creditor')=='hoepers')\
        .filter(F.col('createdAt') <= (F.to_date(F.lit(fileDate), 'yyyyMMdd'))).orderBy(F.col('createdAt').desc())

##Contagem de simulações por CPF
dfSimuQtd = dfSimu.groupBy('personID').count()\
            .select(F.col('personID').alias('_id'), F.col('count').alias('QTD_SIMU')).orderBy(F.col('_id').asc())

#display(dfSimu)

# COMMAND ----------

##Criando coluna de quantidade de simulações por CPF no DF principal
dfArq = dfArq.join(dfSimuQtd, on= '_id', how = 'left')\
        .withColumn('QTD_SIMU', (F.when(F.col('QTD_SIMU').isNull(), 0).otherwise(F.col('QTD_SIMU'))))

# COMMAND ----------

# DBTITLE 1,Autenticação
DATA_Auth = spark.read.format('delta').load(os.path.join(caminho_base_models, 'Authentication',ArqDate))

# COMMAND ----------

##Filtrando colunas de autenticações
dfAuth = DATA_Auth.select('tokenData.document','tokenData.contact','tokenData.creditor','createdAt')\
        .filter(F.col('creditor')=='hoepers')\
        .filter(F.col('createdAt') <= (F.to_date(F.lit(fileDate), 'yyyyMMdd'))).orderBy(F.col('createdAt').desc())


##Contagem de autenticação por CPF
dfAuthQtd = dfAuth.groupBy('document').count()\
            .select(F.col('document').alias('nr_cpf'), F.col('count').alias('QTD_AUTH')).orderBy(F.col('nr_cpf').asc())

# COMMAND ----------

##Criando coluna de quantidade de autenticações por CPF no DF principal
dfArq = dfArq.join(dfAuthQtd, on= 'nr_cpf', how = 'left')\
        .withColumn('QTD_AUTH', (F.when(F.col('QTD_AUTH').isNull(), 0).otherwise(F.col('QTD_AUTH'))))

# COMMAND ----------

# DBTITLE 1,Acordos
#Buscando todos acordos do credor
dfdeals = spark.read.format('delta').load(os.path.join(caminho_base_models, 'Deals', ArqDate)).filter(F.col('creditor')=='hoepers')

dfdeals = dfdeals.filter((F.col('status')!='error') & (F.col('createdAt')>='2020-01-01'))\
          .select(F.col('document').alias('nr_cpf'),F.col('createdAt').alias('DT_ACORDO'),'dealID','status')\
          .withColumn('VARIAVEL_RESPOSTA_ACORDO', F.lit(True))

# COMMAND ----------

#Buscando todos acordos do credor
dfArq = dfArq.join(dfdeals.drop('status'), on=('nr_cpf'), how= 'left')
dfArq = dfArq.withColumn('VARIAVEL_RESPOSTA_ACORDO', (F.when(F.col('VARIAVEL_RESPOSTA_ACORDO').isNull(), False).otherwise(F.col('VARIAVEL_RESPOSTA_ACORDO'))))

# COMMAND ----------

# DBTITLE 1,Pagamentos
dfinstallments = spark.read.format('delta').load(os.path.join(caminho_base_models, 'Installments', ArqDate))\
                 .withColumn('DT_PAGAMENTO', F.when(F.col('paidAt').isNull(),F.col('dueAt')[0:10]).otherwise(F.col('paidAt'))[0:10])\
                 .filter((F.col('DT_PAGAMENTO')>='2020-01-01') & (F.col('creditor')=='hoepers'))\
                 .select('DT_PAGAMENTO', 'dealID', 'QTD_PARCELAS') 

# COMMAND ----------

#Criando variavel resposta de pagamento
dfArq = dfArq.join(dfinstallments, on = 'dealID', how= 'left')
dfArq = dfArq.withColumn('PAGTO', (F.when(F.col('DT_PAGAMENTO').isNull(), False).otherwise(True)))\
.withColumn('QTD_PARCELAS', (F.when(F.col('QTD_PARCELAS').isNull(), 0).otherwise(F.col('QTD_PARCELAS'))))

# COMMAND ----------

display(dfArq.groupBy('PAGTO','VARIAVEL_RESPOSTA_ACORDO').count())

# COMMAND ----------

# DBTITLE 1,Interações
INTER_PATH = "/mnt/bi-reports/MONGO_ANALYTICS/COL_INTERACTION/RAW_DISP/folderDate=*"
INTER_DATA = spark.read.option("delimiter", ";").option("header", True).parquet(INTER_PATH)
dfinteractions = INTER_DATA.filter(F.col('creditor') == 'hoepers')\
                           .filter(F.col('status')!='not_received')\
                           .select('document', F.to_date(F.col('sentAt')[0:10], 'yyyy-M-d').alias('DT_INTERACAO'), 'type')\

dfinteractions = dfinteractions.withColumn('aux', F.lit(1)).groupby('document','DT_INTERACAO').pivot('type').sum('aux')

dfinteractions = dfinteractions.join(dfdeals, on = dfinteractions.document == dfdeals.nr_cpf, how = 'left').drop(*['VARIAVEL_RESPOSTA_ACORDO','nr_cpf'])

dfinteractions = dfinteractions.withColumn('ATIVACAO', F.when((F.col('DT_INTERACAO') <= F.col('DT_ACORDO'))\
                                                              | ((F.col('DT_INTERACAO') > F.col('DT_ACORDO')) & (F.col('status').isin(['expired','broken'])))\
                                                              | (F.col('DT_ACORDO').isNull())
                                                              , 'ATIVADO').otherwise('NAO_ATIVADO'))\
                 .orderBy(F.col('DT_ACORDO').desc()).dropDuplicates(['document','DT_INTERACAO',])\
                 .withColumn('email', (F.when(F.col('email').isNull(), 0).otherwise(F.col('email'))))\
                 .withColumn('sms', (F.when(F.col('sms').isNull(), 0).otherwise(F.col('sms'))))



dfValInt = dfinteractions.select(F.col('document').alias('nr_cpf'),F.col('DT_INTERACAO').alias('PRIMEIRO_ACIONAMENTO')).orderBy('nr_cpf',F.col('PRIMEIRO_ACIONAMENTO').asc()).dropDuplicates(['nr_cpf'])

dfinteractions = dfinteractions.groupby('document')\
                .pivot('ATIVACAO').sum('email','sms')\
                .select(F.col('document').alias('nr_cpf'), F.col('ATIVADO_sum(email)').alias('ACION_EMAIL_QTD_ATIVADO'),\
                        F.col('NAO_ATIVADO_sum(email)').alias('ACION_EMAIL_QTD_NAO_ATIVADO'), F.col('ATIVADO_sum(sms)').alias('ACION_SMS_QTD_ATIVADO'),\
                        F.col('NAO_ATIVADO_sum(sms)').alias('ACION_SMS_QTD_NAO_ATIVADO'))\
                .withColumn('ACION_EMAIL_QTD_ATIVADO', (F.when(F.col('ACION_EMAIL_QTD_ATIVADO').isNull(), 0).otherwise(F.col('ACION_EMAIL_QTD_ATIVADO'))))\
                .withColumn('ACION_EMAIL_QTD_NAO_ATIVADO', (F.when(F.col('ACION_EMAIL_QTD_NAO_ATIVADO').isNull(), 0).otherwise(F.col('ACION_EMAIL_QTD_NAO_ATIVADO'))))\
                .withColumn('ACION_SMS_QTD_ATIVADO', (F.when(F.col('ACION_SMS_QTD_ATIVADO').isNull(), 0).otherwise(F.col('ACION_SMS_QTD_ATIVADO'))))\
                .withColumn('ACION_SMS_QTD_NAO_ATIVADO', (F.when(F.col('ACION_SMS_QTD_NAO_ATIVADO').isNull(), 0).otherwise(F.col('ACION_SMS_QTD_NAO_ATIVADO'))))


display(dfinteractions)

# COMMAND ----------

#Adicionando coluna de primeiro acionamento
dfArq = dfArq.join(dfValInt, on = 'nr_cpf' , how='inner').drop('document')

#Adicinando as quantidades no DF principal
dfArq = dfArq.join(dfinteractions, on = 'nr_cpf', how='left').drop(*['dealID','_id','status'])

# COMMAND ----------

display(dfArq)

# COMMAND ----------

dfArq.coalesce(1).write.option('sep', ';').option('header', 'True').csv(os.path.join(caminho_trusted, 'full_temp'))
    
for file in dbutils.fs.ls(os.path.join(caminho_trusted, 'full_temp')):
  if file.name.split('.')[-1] == 'csv':
    dbutils.fs.cp(file.path, os.path.join(caminho_trusted,file_name))
  else:
    dbutils.fs.rm(file.path, True)
dbutils.fs.rm(os.path.join(caminho_trusted, 'full_temp'), True)

# COMMAND ----------

dfArq = spark.read.option('sep', ';').option('header', 'True').csv(os.path.join(caminho_trusted,file_name))

display(dfArq)

# COMMAND ----------

display(dfArq.groupBy('VARIAVEL_RESPOSTA_ACORDO').count())