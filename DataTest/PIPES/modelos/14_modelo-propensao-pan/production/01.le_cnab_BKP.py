# Databricks notebook source
# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

import os
import datetime
import json

blob_account_source_prd = "qqprd"
blob_account_source_ml = "qqdatastoragemain"
blob_container_source_prd = "qq-integrator"
blob_container_source_ml = "ml-prd"

prefix = "etl/dmcard/processed"

mount_blob_storage_key(dbutils,blob_account_source_prd,blob_account_source_prd,'/mnt/qq-integrator')
mount_blob_storage_key(dbutils,blob_account_source_ml,blob_account_source_ml,'/mnt/ml-prd')

caminho_layout = '/mnt/ml-prd/ml-data/propensaodeal/pan/layout_cnab'
caminho_base = '/mnt/qq-integrator/etl/pan/processed'
caminho_base_dbfs = '/dbfs/mnt/qq-integrator/etl/pan/processed'
caminho_stage = '/mnt/ml-prd/ml-data/propensaodeal/pan/stage'
caminho_trusted = '/mnt/ml-prd/ml-data/propensaodeal/pan/trusted'
caminho_sample = '/mnt/ml-prd/ml-data/propensaodeal/pan/sample'

spark.conf.set("spark.sql.session.timeZone", "Etc/UCT")

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

def escreve_variavel_resposta_acordo(df, credor, dataMin, documentCol, varName='VARIAVEL_RESPOSTA', drop_null = False, use_bi_report = False):
  """
  Escreve em um dataframe fornecido uma coluna com a variável resposta de ACORDO verdadeiro/falso (se a pessoa cravou ou não acordo).
  Funciona por DOCUMENT ou ID UNICO DE PESSOA.
  Passe o dataframe, nome do credor (str), data mínima para acordo cravado (objeto date) - apenas datas superiores à passada - , nome da coluna que faz referência à informação do Mongo (str) - ex: 'document', 'id_pessoa', se essa coluna corresponde a DOCUMENT ou _ID (passar via string ---- 'document' ou '_id').
  drop_null diz se devem ser dropadas linhas com valor NULL na variavel resposta (não existem no Mongo)
  """
  def sep_lst(lst, sep):
    for i in range (0, len(lst), sep):
      yield (lst[i:i+sep])
    
    
  import datetime
  dataMin = datetime.datetime(dataMin.year, dataMin.month, dataMin.day, 3)
  #print (dataMin)
  df = df.withColumn(documentCol, F.lpad(F.col(documentCol),11,'0'))
  if not use_bi_report:
    lista_pk = df.select(documentCol).dropDuplicates().rdd.map(lambda Row:Row[0]).collect()
    lista_pk_iter = sep_lst(lista_pk, 1000)

    ### construindo project ###
    project = {'_id':1, 'document':1,'deals._id':1, 'deals.createdAt':1, 'deals.creditor':1}
    col_person = getPyMongoCollection('col_person')

    min_array = 0

    matrix = []

    for l in lista_pk_iter:
      query = col_person.find({'document':{'$in':l}},project)

      for q in query:
        try:
          document = q['document']
          has_deal = False
          try:
            for deal in q['deals']:
              #print (credor, dataMin)
              #print (deal['creditor'], deal['createdAt'])
              if deal['creditor'] == credor and deal['createdAt'] >= dataMin:
                has_deal = True
          except:
            pass
          matrix.append([document, has_deal])
        except Exception as e:
          print (q['_id'], e)

    matrix = spark.sparkContext.parallelize(matrix)
    schema = T.StructType([T.StructField(documentCol, T.StringType(), False), T.StructField(varName, T.BooleanType(), False)])
    df_resposta = spark.createDataFrame(matrix, schema=schema)
    df = df.join(df_resposta, on=documentCol, how = 'left')

    docs_inexistentes = df.filter(F.col(varName).isNull()).select(documentCol)

    if drop_null:
      df = df.filter(F.col(varName).isNotNull())

    print ('drop_null:', drop_null)
    print (df.count(), 'linhas no dataframe final')
    dfTrue = df.select(varName).filter(F.col(varName)==True)
    dfFalse = df.select(varName).filter(F.col(varName)==False)
    print (dfTrue.count(), 'TRUE')
    print (dfFalse.count(), 'FALSE')
    print (docs_inexistentes.count(), 'linhas sem variável resposta - não deram match no mongoDB')
    display(docs_inexistentes)
  
  else:
    bi_report_path = '/mnt/bi-reports/env/trusted/deals/deals_full.PARQUET'
    dataMin = dataMin.strftime("%Y-%m-%d")

    df_bi = spark.read.parquet(bi_report_path)
    df_bi = df_bi.filter(F.col('creditor')==credor).filter(F.col('createdAt')>=dataMin)
    df_bi = df_bi.withColumnRenamed('document', documentCol)
    if drop_null:
      how = 'left'
    else:
      how = 'full'
    df = df.join(df_bi.select(documentCol, 'DEALS_ID'), on=documentCol, how=how)
    df = df.withColumnRenamed('DEALS_ID', varName)
    df = df.withColumn(varName, F.when(F.col(varName).isNull(), False).otherwise(True))
  return df

# COMMAND ----------

data_escolhida = max(datas_escolhidas)
if variavelResposta:
  df = escreve_variavel_resposta_acordo(df, 'pan', data_escolhida, 'CPF/CNPJ Cliente', drop_null = True, use_bi_report = True)
  df_representativo, df_aleatorio = gera_sample(df)
  df_aleatorio.coalesce(1).write.option('sep', ';').option('header', 'True').csv(os.path.join(caminho_sample, 'aleatorio_temp'))
  for file in dbutils.fs.ls(os.path.join(caminho_sample, 'aleatorio_temp')):
    if file.name.split('.')[-1] == 'csv':
      dbutils.fs.cp(file.path, os.path.join(caminho_sample, 'base_aleatoria.csv'))
  dbutils.fs.rm(os.path.join(caminho_sample, 'aleatorio_temp'), True)


  df_representativo.coalesce(1).write.option('sep', ';').option('header', 'True').csv(os.path.join(caminho_sample, 'representativo_temp'))
  for file in dbutils.fs.ls(os.path.join(caminho_sample, 'representativo_temp')):
    if file.name.split('.')[-1] == 'csv':
      dbutils.fs.cp(file.path, os.path.join(caminho_sample, 'base_representativa.csv'))
  dbutils.fs.rm(os.path.join(caminho_sample, 'representativo_temp'), True)

else:
  print ('implemente a escrita do df!')

# COMMAND ----------

df_representativo, df_aleatorio = gera_sample(df, max_sample=100000)
df_aleatorio.coalesce(1).write.option('sep', ';').option('header', 'True').csv(os.path.join(caminho_sample, 'aleatorio_temp'))
for file in dbutils.fs.ls(os.path.join(caminho_sample, 'aleatorio_temp')):
  if file.name.split('.')[-1] == 'csv':
    dbutils.fs.cp(file.path, os.path.join(caminho_sample, 'base_aleatoria.csv'))
  else:
    dbutils.fs.cp(file.path, True)
dbutils.fs.rm(os.path.join(caminho_sample, 'aleatorio_temp'), True)


df_representativo.coalesce(1).write.option('sep', ';').option('header', 'True').csv(os.path.join(caminho_sample, 'representativo_temp'))
for file in dbutils.fs.ls(os.path.join(caminho_sample, 'representativo_temp')):
  if file.name.split('.')[-1] == 'csv':
    dbutils.fs.cp(file.path, os.path.join(caminho_sample, 'base_representativa.csv'))
  else:
    dbutils.fs.cp(file.path, True)
dbutils.fs.rm(os.path.join(caminho_sample, 'representativo_temp'), True)

# COMMAND ----------

df_aleatorio.coalesce(1).write.option('sep', ';').option('header', 'True').csv(os.path.join(caminho_sample, 'aleatorio_temp'))
df_representativo.coalesce(1).write.option('sep', ';').option('header', 'True').csv(os.path.join(caminho_sample, 'representativo_temp'))