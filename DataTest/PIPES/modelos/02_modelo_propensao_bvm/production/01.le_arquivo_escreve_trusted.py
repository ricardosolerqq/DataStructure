# Databricks notebook source
import zipfile
import datetime
import os
import time
time.sleep(300)

# COMMAND ----------

try:
    dbutils.widgets.remove('ARQUIVO_ESCOLHIDO')
except:
    pass

# COMMAND ----------

# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------


blob_account_source_prd = "qqprd"
blob_account_source_ml = "qqdatastoragemain"
blob_container_source_prd = "qq-integrator"
blob_container_source_ml = "ml-prd"

prefix = "etl/bvm/processed"

mount_blob_storage_key(dbutils, blob_account_source_prd,
                       blob_account_source_prd, '/mnt/qq-integrator')
mount_blob_storage_key(dbutils, blob_account_source_ml,
                       blob_account_source_ml, '/mnt/ml-prd')

caminho_base = '/mnt/qq-integrator/etl/bvm/processed'
caminho_base_dbfs = '/dbfs/mnt/qq-integrator/etl/bvm/processed'
caminho_trusted = '/mnt/ml-prd/ml-data/propensaodeal/bvm/trusted'
caminho_sample = '/mnt/ml-prd/ml-data/propensaodeal/bvm/sample'

# COMMAND ----------

# DBTITLE 1,configurando processo e arquivo a ser tratado
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
    arquivo_escolhido = list(
        get_creditor_etl_file_dates('bvm', latest=True))[0]
    arquivo_escolhido_path = os.path.join(caminho_base, arquivo_escolhido)
    arquivo_escolhido_path_dbfs = os.path.join(
        '/dbfs', caminho_base, arquivo_escolhido)

else:
    lista_arquivos = os.listdir('/dbfs/mnt/qq-integrator/etl/bvm/processed')
    dict_arq = {item.split('_')[-1].split('.')[0]: item for item in lista_arquivos}
    dbutils.widgets.dropdown('ARQUIVO_ESCOLHIDO', max(
        str(item) for item in dict_arq), [str(item) for item in dict_arq])
    arquivo_escolhido = dbutils.widgets.get('ARQUIVO_ESCOLHIDO')
    arquivo_escolhido_path = os.path.join(
        caminho_base, dict_arq[arquivo_escolhido])
    arquivo_escolhido_path_dbfs = '/dbfs/'+arquivo_escolhido_path

arquivo_escolhido_fileformat = arquivo_escolhido_path.split('.')[-1]
arquivo_escolhido_fileformat


# arquivo_escolhido_path
file = arquivo_escolhido_path
file

# COMMAND ----------

# DBTITLE 1,criando dataframe spark
df = spark.read.option('delimiter', ';').option('header', 'True').csv(file)
display(df)

# COMMAND ----------

# DBTITLE 1,criando variavel resposta?
arquivo_escolhido_date = arquivo_escolhido.split('_')[2].split('.')[0]
data_arquivo_escolhido = datetime.date(int(arquivo_escolhido_date[0:4]), int(
    arquivo_escolhido_date[4:6]), int(arquivo_escolhido_date[6:8]))
dbutils.widgets.dropdown('ESCREVER_VARIAVEL_RESPOSTA',
                         'False', ['False', 'True'])
escreverVariavelResposta = dbutils.widgets.get('ESCREVER_VARIAVEL_RESPOSTA')
if escreverVariavelResposta == 'True':
    df = escreve_variavel_resposta_acordo(
        df, 'bvm', data_arquivo_escolhido, 'NUCPFCNPJ', drop_null=True)

# COMMAND ----------

# DBTITLE 1,escrevendo amostra representativa e amostra aleatoria
# amostra representativa - todos os verdadeiros mais 3x a quantidade de verdadeiros como falsos no mesmo arquivo
# amostra aleatória - todos os verdadeiros e o que faltar para completar 50000 com zeros
if escreverVariavelResposta == 'True':

    dfTrue = df.filter(F.col('VARIAVEL_RESPOSTA') == True)
    dfFalse = df.filter(F.col('VARIAVEL_RESPOSTA') == False)

    dfFalse_representativo = dfFalse.sample((dfTrue.count()*3)/dfFalse.count())
    dfFalse_aleatorio = dfFalse.sample((50000-dfTrue.count())/dfFalse.count())

    df_representativo = dfTrue.union(dfFalse_representativo)
    df_aleatorio = dfTrue.union(dfFalse_aleatorio)

    df_representativo.coalesce(1).write.option('header', 'True').option('delimiter', ';').csv(os.path.join(
        caminho_trusted, arquivo_escolhido, 'sample', arquivo_escolhido+'_amostra_representativa.csv'))
    df_aleatorio.coalesce(1).write.option('header', 'True').option('delimiter', ';').csv(os.path.join(
        caminho_trusted, arquivo_escolhido, 'sample', arquivo_escolhido+'_amostra_aleatoria.csv'))

else:
    # Escrevendo DataFrame
    df.coalesce(1).write.option('sep', ';').option(
        'header', 'True').csv(os.path.join(caminho_trusted, 'tmp'))
    for file in dbutils.fs.ls(os.path.join(caminho_trusted, 'tmp')):
        if file.name.split('.')[-1] == 'csv':
            dbutils.fs.cp(file.path, os.path.join(
                caminho_trusted, arquivo_escolhido.split('.')[0]+'.csv'))
    dbutils.fs.rm(os.path.join(caminho_trusted, 'tmp'), True)
