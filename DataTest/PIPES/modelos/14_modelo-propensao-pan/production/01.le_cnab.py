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

# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

import os
import datetime
import json

blob_account_source_prd = "qqprd"
blob_account_source_ml = "qqdatastoragemain"
blob_container_source_prd = "qq-integrator"
blob_container_source_ml = "ml-prd"


mount_blob_storage_key(dbutils,blob_account_source_prd,blob_account_source_prd,'/mnt/qq-integrator')
mount_blob_storage_key(dbutils,blob_account_source_ml,blob_account_source_ml,'/mnt/ml-prd')

caminho_layout = '/mnt/ml-prd/ml-data/propensaodeal/pan/layout_cnab'
caminho_base = '/mnt/qq-integrator/etl/pan/processed'
caminho_base_dbfs = '/dbfs/mnt/qq-integrator/etl/pan/processed'
caminho_stage = '/mnt/ml-prd/ml-data/propensaodeal/pan/stage'
caminho_trusted = '/mnt/ml-prd/ml-data/propensaodeal/pan/trusted'
caminho_sample = '/mnt/ml-prd/ml-data/propensaodeal/pan/sample'
caminho_trusted_dbfs = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/pan/trusted'
caminho_base_models = '/mnt/ml-prd/ml-data/base_models'

file_name = 'Base_Pan_QQ_Model_'+date+'.csv'

spark.conf.set("spark.sql.session.timeZone", "Etc/UCT")

# COMMAND ----------

def get_creditor_etl_file_dates(credor, latest = False, as_date = False):
  """
  esta função verifica o blob qq-integrator e retorna todos os blobs na pasta de ETL's processados
  retorna ou TODOS os arquivos com suas respectivas datas ou, quando latest = True, apenas o mais recente
  """
  
  
  ackey = dbutils.secrets.get(scope = "scope_qqdatastoragemain", key = "qqprd-key")
  block_blob_service = BlockBlobService(account_name='qqprd', account_key=ackey)
  
  prefix = "etl/"+credor+"/backup_processed"
  
  generator = block_blob_service.list_blobs('qq-integrator', prefix=prefix)
  
  file_date = {}
  
  for blob in generator:   
    nome = blob.name.split('/')[-1]
    data = BlockBlobService.get_blob_properties(block_blob_service,'qq-integrator',blob.name).properties.last_modified
    if as_date:
      data = datetime.date(data.year, data.month, data.day)
    if not latest:
      file_date.update({nome:data})
    else:
      file_date.update ({data:nome})
      
  if not latest:
    return (file_date)
  else:
    newest_date = max(list(file_date))
    nome = file_date[newest_date]
    return ({nome:newest_date})

# COMMAND ----------

dates = []
files_dates = get_creditor_etl_file_dates('pan', as_date=True)
for file in files_dates:
  if 'remessa' in file.lower():
    if files_dates[file] not in dates:
      dates.append(files_dates[file])
dates = sorted(dates, reverse=True)

# COMMAND ----------

# DBTITLE 1,config autoMode
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
        if 'remessa' in file:
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
  dbutils.widgets.dropdown('VARIAVEL_RESPOSTA', 'False', ['True', 'False'])
  if dbutils.widgets.get('VARIAVEL_RESPOSTA') == 'True':
    variavelResposta = True
  else:
    variavelResposta = False
  if variavelResposta:
    hoje = datetime.datetime.today()
    mesAtual = datetime.datetime(hoje.year, hoje.month, 1)
    mesAnterior = mesAtual - datetime.timedelta(days=28)
    mesAnterior = datetime.date(mesAnterior.year, mesAnterior.month, 1)
    tresMesesAtras = mesAnterior - datetime.timedelta(days=84)
    tresMesesAtras = datetime.date(tresMesesAtras.year, tresMesesAtras.month, 1)
    print ('populando arquivos de,', tresMesesAtras, ' até', mesAnterior,'para a variável resposta\n')
    datas_escolhidas = []
    for date in dates:
      if date < mesAnterior and date >= tresMesesAtras:
        datas_escolhidas.append(date)
  else:
    datas_escolhidas = max(dates)
else:
  print ('MODE = Auto para liberar possibilidade de variável resposta')
  variavelResposta = False
  try:
    dbutils.widgets.remove('VARIAVEL_RESPOSTA')
  except:
    pass
  
  dbutils.widgets.multiselect('DATAS_ARQUIVOS', str(max(dates)), [str(date) for date in dates])
  datas_escolhidas = dbutils.widgets.get('DATAS_ARQUIVOS')
  datas_escolhidas = datas_escolhidas.split(',')
  
arquivos_escolhidos = escolhe_arquivos(files_dates, datas_escolhidas)  

print ('autoMode',autoMode)
print('variavelResposta',variavelResposta)
print ('arquivos a processar:')
for arq in arquivos_escolhidos:
  print ('\t',arq)

# COMMAND ----------

# DBTITLE 1,construindo regras do cnab
cnabLayout = spark.read.option('sep',';').option('header','True').csv(dbutils.fs.ls(caminho_layout)[0].path)
cnabLayout_dict = cnabLayout.rdd.map(lambda Row:{Row[0]:[int(Row[1]), int(Row[3])]}).collect()
cnabLayout_dict_temp = {}
for l in cnabLayout_dict:
  for k in l:
    v = l[k]
    cnabLayout_dict_temp.update({k: v})
  
cnabLayout_dict = cnabLayout_dict_temp
del cnabLayout_dict_temp

# COMMAND ----------

arquivos_escolhidos_dict = {}
for date in arquivos_escolhidos:
  try:
    arquivos_escolhidos_dict[date[0]].append(date[1])
  except Exception as e:
    arquivos_escolhidos_dict.update({date[0]:[date[1]]})
    


# COMMAND ----------

arquivos_escolhidos = arquivos_escolhidos[0:1770]

# COMMAND ----------

dfs = {}
for date in arquivos_escolhidos_dict:
  print (date)
  i = 0
  for file in arquivos_escolhidos_dict[date]:
    if i == 0:
      i = i+1
      df = spark.read.option('header', 'False').csv(os.path.join(caminho_base, file)).withColumnRenamed('_c0', 'raw')
    else:
      dfu = spark.read.option('header', 'False').csv(os.path.join(caminho_base, file)).withColumnRenamed('_c0', 'raw')
      try:
        df = df.union(dfu)
      except:
        pass
  for col in cnabLayout_dict:
    df = df.withColumn(col, F.substring(F.col('raw'), cnabLayout_dict[col][0], cnabLayout_dict[col][1]))
  df = df.withColumn('CPF/CNPJ Cliente', F.when(F.col('PJ ou PF')=='F', F.lpad(F.col('CPF/CNPJ Cliente').cast(T.StringType()),11,'0')).otherwise(F.col('CPF/CNPJ Cliente')))
  df = df.withColumn('CPF/CNPJ Cliente', F.when(F.col('PJ ou PF')=='J', F.lpad(F.col('CPF/CNPJ Cliente').cast(T.StringType()),14,'0')).otherwise(F.col('CPF/CNPJ Cliente')))
  df = df.drop('raw')
  df = df.dropDuplicates(subset = ['CPF/CNPJ Cliente','Número do Contrato (Chave)'])
  dfs.update({date:df})

# COMMAND ----------

i = 0
for date in dfs:
  if i == 0:
    df = dfs[date]
  else:
    ### garantindo que df será apenas com dado novo
    df = df.union(dfs[date]).distinct()
    df = df.dropDuplicates(subset = ['CPF/CNPJ Cliente','Número do Contrato (Chave)'])
  i = i+1

# COMMAND ----------

df.count()

# COMMAND ----------

df.count()

# COMMAND ----------

df = df.withColumnRenamed('Número do Contrato (Chave)',"nr_contrato")\
       .withColumnRenamed('CPF/CNPJ Cliente',"nr_cpf")

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
dfDebtsCre = dfDebts.filter(F.col('creditor') == 'pan').orderBy(F.col('_id').asc())
dfValCredor =  dfDebtsCre.select('_id','contract')

##Criando DF de divida nos demais credores
dfDebtsOut = dfDebts.filter(F.col('creditor') != 'pan').select('_id','creditor').orderBy(F.col('_id').asc())
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
dfValCredor = dfValCredor.join(dfArq, (dfValCredor.contract == dfArq.nr_contrato) & (dfValCredor._id == dfArq._id ), 'left').select(dfValCredor._id,'contract','nr_contrato').filter(F.col('nr_contrato').isNull()).drop('nr_contrato').orderBy(F.col('_id').asc())
dfArq = dfArq.join(dfValCredor, on='_id', how='left')
dfArq = dfArq.withColumn('Divida_Mesmo_Credor', F.when(F.col('contract').isNull(),False).otherwise(True)).drop(*['contract','nr_conta'])

dfArq = dfArq.distinct()

# COMMAND ----------

# DBTITLE 1,Simulação
DATA_Simu = spark.read.format('delta').load(os.path.join(caminho_base_models, 'Simulation',ArqDate))

# COMMAND ----------

dfSimu = DATA_Simu.select('personID','creditor','createdAt')\
        .filter(F.col('creditor')=='pan')\
        .filter(F.col('createdAt') <= (F.to_date(F.lit(date), 'yyyyMMdd'))).orderBy(F.col('createdAt').desc())

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
        .filter(F.col('creditor')=='pan')\
        .filter(F.col('createdAt') <= (F.to_date(F.lit(date), 'yyyyMMdd'))).orderBy(F.col('createdAt').desc())


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
dfdeals = spark.read.format('delta').load(os.path.join(caminho_base_models, 'Deals', ArqDate)).filter(F.col('creditor')=='pan')

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
                 .filter((F.col('DT_PAGAMENTO')>='2020-01-01') & (F.col('creditor')=='pan'))\
                 .select('DT_PAGAMENTO', 'dealID', 'QTD_PARCELAS') 
display(dfinstallments)

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
dfinteractions = INTER_DATA.filter(F.col('creditor') == 'pan')\
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

dfArq.coalesce(1).write.option('sep', ';').option('header', 'True').csv(os.path.join(caminho_trusted, 'full_temp'))
    
for file in dbutils.fs.ls(os.path.join(caminho_trusted, 'full_temp')):
  if file.name.split('.')[-1] == 'csv':
    dbutils.fs.cp(file.path, os.path.join(caminho_trusted,file_name))
  else:
    dbutils.fs.rm(file.path, True)
dbutils.fs.rm(os.path.join(caminho_trusted, 'full_temp'), True)

# COMMAND ----------

display(dfArq)

# COMMAND ----------

