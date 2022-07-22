# Databricks notebook source
import zipfile
import datetime
import os
import time
time.sleep(300)

# COMMAND ----------

# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

try:
    dbutils.widgets.remove('ARQUIVO_ESCOLHIDO')
except:
    pass

# COMMAND ----------


blob_account_source_prd = "qqprd"
blob_account_source_ml = "qqdatastoragemain"
blob_container_source_prd = "qq-integrator"
blob_container_source_ml = "ml-prd"

prefix = "etl/agibank/processed"

mount_blob_storage_key(dbutils, blob_account_source_prd,
                       blob_account_source_prd, '/mnt/qq-integrator')
mount_blob_storage_key(dbutils, blob_account_source_ml,
                       blob_account_source_ml, '/mnt/ml-prd')

caminho_base = 'mnt/qq-integrator/etl/agibank/processed'
caminho_base_dbfs = '/dbfs/mnt/qq-integrator/etl/agibank/processed'
caminho_trusted = '/mnt/ml-prd/ml-data/propensaodeal/agibank/trusted'
caminho_sample = '/mnt/ml-prd/ml-data/propensaodeal/agibank/sample'

# COMMAND ----------

# datas_arqs = {}
#  arqs_datas = get_creditor_etl_file_dates('agibank')
#  for arq in arqs_datas:
#    if arq.split('_')[0].lower() == 'mailing':
#      datas_arqs.update({arqs_datas[arq]:arq})
#      ultimo_arquivo = datas_arqs[max(datas_arqs)]

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
    datas_arqs = {}
    arqs_datas = get_creditor_etl_file_dates('agibank')
    for arq in arqs_datas:
        if arq.split('_')[0].lower() == 'mailing':
            datas_arqs.update({arqs_datas[arq]: arq})
            file = caminho_base + "/" + datas_arqs[max(datas_arqs)]
            arquivo_escolhido = datas_arqs[max(datas_arqs)]

# else:
#  lista_arquivos = os.listdir('/dbfs/mnt/qq-integrator/etl/agibank/processed')
#  dict_arq = {datetime.date(int(item.split('mailing_')[1].split('.')[0].split('-')[2]), int(item.split('mailing_')[1].split('.')[0].split('-')[1]), int(item.split('mailing_')[1].split('.')[0].split('-')[0])):item for item in lista_arquivos}
#  dbutils.widgets.dropdown('ARQUIVO_ESCOLHIDO', max(str(item) for item in dict_arq), [str(item) for item in dict_arq])
#  arquivo_escolhido = dbutils.widgets.get('ARQUIVO_ESCOLHIDO')
#  arquivo_escolhido_path = os.path.join(caminho_base,dict_arq[datetime.date(int(arquivo_escolhido.split('-')[0]), int(arquivo_escolhido.split('-')[1]), int(arquivo_escolhido.split('-')[2]))])
#  arquivo_escolhido_path_dbfs = os.path.join('/dbfs',caminho_base,dict_arq[datetime.date(int(arquivo_escolhido.split('-')[0]), int(arquivo_escolhido.split('-')[1]), int(arquivo_escolhido.split('-')[2]))])

#arquivo_escolhido_fileformat = arquivo_escolhido_path.split('.')[-1]
# arquivo_escolhido_fileformat


# arquivo_escolhido_path
#file = arquivo_escolhido_path
file

# COMMAND ----------

#datas_arqs = {}
#arqs_datas = get_creditor_etl_file_dates('agibank')
# for arq in arqs_datas:
#  if arq.split('_')[0].lower() == 'mailing':
#    datas_arqs.update({arqs_datas[arq]:arq})
#ultimo_arquivo = datas_arqs[max(datas_arqs)]
# ultimo_arquivo

# COMMAND ----------

# DBTITLE 1,UnZip de Arquivo
# try:
#  for file in os.listdir('/dbfs/tmp_zip'):
#    os.remove(os.path.join('/dbfs/tmp_zip',file))
# except:
#  pass

# if arquivo_escolhido_fileformat == 'zip':
#  with zipfile.ZipFile(arquivo_escolhido_path_dbfs,"r") as zip_ref:
#    print ("DESZIPANDO")
#    zip_ref.extractall(path = '/dbfs/tmp_zip')
#    print ('deszipado!')
#  for file in os.listdir('/dbfs/tmp_zip'):
#    print (file)
#  file = os.path.join('/tmp_zip',file)
# else:
#  file = arquivo_escolhido_path

# COMMAND ----------

# DBTITLE 1,Criando Dataframe Spark
df = spark.read.option('delimiter', ';').option('header', 'True').csv("/"+file)

# COMMAND ----------

df.show(10)

# COMMAND ----------

# DBTITLE 1,Condicional Para Variável Resposta e DataFrame
dbutils.widgets.dropdown('ESCREVER_VARIAVEL_RESPOSTA',
                         'False', ['False', 'True'])
escreverVariavelResposta = dbutils.widgets.get('ESCREVER_VARIAVEL_RESPOSTA')
if escreverVariavelResposta == 'True':
    df = escreve_variavel_resposta_acordo(df, 'agibank', datetime.datetime(
        2021, 10, 10), 'NR_CPF_CNPJ2', 'VARIAVEL_RESPOSTA', drop_null=True)

    # amostra representativa - todos os verdadeiros mais 3x a quantidade de verdadeiros como falsos no mesmo arquivo
    # amostra aleatória - todos os verdadeiros e o que faltar para completar 50000 com zeros
    dfTrue = df.filter(F.col('VARIAVEL_RESPOSTA') == True)
    dfFalse = df.filter(F.col('VARIAVEL_RESPOSTA') == False)

    dfFalse_representativo = dfFalse.sample((dfTrue.count()*3)/dfFalse.count())
    dfFalse_aleatorio = dfFalse.sample(
        (df.count()-dfTrue.count())/dfFalse.count())

    df_representativo = dfTrue.union(dfFalse_representativo)
    df_aleatorio = dfTrue.union(dfFalse_aleatorio)

    df_representativo.coalesce(1).write.option('header', 'True').option('delimiter', ';').csv(os.path.join(
        caminho_sample, arquivo_escolhido, 'sample', arquivo_escolhido+'_amostra_representativa.csv'))
    df_aleatorio.coalesce(1).write.option('header', 'True').option('delimiter', ';').csv(os.path.join(
        caminho_sample, arquivo_escolhido, 'sample', arquivo_escolhido+'_amostra_aleatoria.csv'))
else:
    # Escrevendo DataFrame
    dbutils.fs.rm(caminho_trusted, True)
    df.coalesce(1).write.option('sep', ';').option(
        'header', 'True').csv(os.path.join(caminho_trusted, 'tmp'))
    for file in dbutils.fs.ls(os.path.join(caminho_trusted, 'tmp')):
        if file.name.split('.')[-1] == 'csv':
            dbutils.fs.cp(file.path, os.path.join(
                caminho_trusted, arquivo_escolhido.split('.')[0]+'.csv'))
    dbutils.fs.rm(os.path.join(caminho_trusted, 'tmp'), True)
