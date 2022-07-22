# Databricks notebook source
import os
import csv

# COMMAND ----------

# DBTITLE 1,VARIÁVEIS COM REGRAS
"""
GOOGLE SHEETS DAS REGRAS
https://docs.google.com/spreadsheets/d/1ZSXOSvVBcBHEcTQg88vh6PFIi5nTg9EQhGyDmzZaHOA/edit#gid=1545203801

"""

regras = {'detalhes' : {'Tipo de Registro':2,
            'ID Cliente EC':10,
            'ID Cedente':10,
            'ID Cliente':50,
            'Nome Cedente':50,
            'ID Detalhe Cliente':10,
            'ID Detalhe':10,
            'Nome Detalhe':50,
            'Tipo':1,
            'Valor':500},


'detalhes_contratos' : {'Tipo de Registro':2,
                      'ID Cliente EC':10,
                      'ID Cedente':10,
                      'ID Cliente':50,
                      'Nome Cedente':50,
                      'ID Contrato EC':10,
                      'ID Contrato':50,
                      'ID Produto':10,
                      'Nome Produto':50,
                      'ID Detalhe Contrato':10,
                      'ID Detalhe':10,
                      'Nome Detalhe':50,
                      'Tipo':1,
                      'Valor':500},

'detalhes_dividas' : {'Tipo de Registro':2,
                    'ID Dívida EC':20,
                    'ID Cliente EC':10,
                    'ID Cedente':10,
                    'ID Cliente':50,
                    'Nome Cedente':50,
                    'ID Contrato EC':10,
                    'ID Contrato':50,
                    'ID Produto':10,
                    'Nome Produto':50,
                    'ID Dívida':50,
                    'ID Detalhe Dívida':20,
                    'ID Detalhe':10,
                    'Nome Detalhe':50,
                    'Tipo':1,
                    'Valor':500}}


# COMMAND ----------

# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

blob_account_source_prd = "qqprd"
blob_account_source_ml = "qqdatastoragemain"
blob_container_source_prd = "qq-integrator"
blob_container_source_ml = "ml-feature-store"

prefix = "etl/credz/processed"

mount_blob_storage_key(dbutils,blob_account_source_prd,blob_account_source_prd,'/mnt/qq-integrator')
mount_blob_storage_key(dbutils,blob_account_source_ml,blob_account_source_ml,'/mnt/ml-feature-store')

caminho_raw_dbfs = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/credz/raw'
list_caminho_raw = os.listdir(caminho_raw_dbfs)
caminho_raw = '/mnt/ml-prd/ml-data/propensaodeal/credz/raw'
detalhespath = 'file://mnt/ml-feature-store/credz/detalhes'

# COMMAND ----------

def leCSV(txtfile):
  schema = [T.StructField('TXT_ORIGINAL', T.StringType(), False)]
  schema_final = T.StructType(schema)
  df = spark.read.format("csv").option("header", False).schema(schema_final).load(txtfile)
  return df

# COMMAND ----------

def criaDFs(df):
  df = df.withColumn("Tipo de Registro", F.col('TXT_ORIGINAL').substr(0,2))

  detalhes_clientes_df = df.filter(F.col('Tipo de Registro')=='04')
  detalhes_contratos_df = df.filter(F.col('Tipo de Registro')=='06')
  detalhes_dividas_df = df.filter(F.col('Tipo de Registro')=='08')
  
  detalhes_clientes = {'detalhes':detalhes_clientes_df}
  detalhes_contratos = {'detalhes_contratos':detalhes_contratos_df}
  detalhes_dividas = {'detalhes_dividas':detalhes_dividas_df}
  
  return [detalhes_clientes, detalhes_contratos, detalhes_dividas]

# COMMAND ----------

def txt_para_col(df, tipo_registro):
  if tipo_registro == 'detalhes':
    df = df.withColumn('DETALHES_CLIENTES', F.col('TXT_ORIGINAL').substr(143,192))
  elif tipo_registro == 'detalhes_dividas':
    df = df.withColumn('DETALHES_DIVIDAS', F.col('TXT_ORIGINAL').substr(343,392))
  elif tipo_registro == 'detalhes_contratos':
      df = df.withColumn('DETALHES_CONTRATOS', F.col('TXT_ORIGINAL').substr(263,312))

  df = df.drop("TXT_ORIGINAL")
  df = df.dropDuplicates()

  return df

# COMMAND ----------

# DBTITLE 1,corrigindo nomes das colunas
def padronizaNomesColunas(df):
  for col in df.columns:
    str_col = ''
    for c in col:
      if c == " " or c == '-' or c == ',' or c == ';' or c == '{' or c == '}' or c == '(' or c == ')' or c == '\n' or c == '\t':
        str_col = str_col + '_'
      elif c.upper() == "Á":
        str_col = str_col + 'A'
      elif c.upper() == "É":
        str_col = str_col + 'E'
      elif c.upper() == "Í":
        str_col = str_col + 'I'
      elif c.upper() == "Ó":
        str_col = str_col + 'O'
      elif c.upper() == "Ú":
        str_col = str_col + 'U'

      elif c.upper() == "Ã":
        str_col = str_col + 'A'

      elif c.upper() == "Â":
        str_col = str_col + 'A'
      elif c.upper() == "Ê":
        str_col = str_col + 'E'
        
      else:
        str_col = str_col + c
    str_col = str_col.upper()
    if str_col[-1] == "_":
      str_col = str_col[:-1]
    df = df.withColumnRenamed(col, str_col)
  return df

# COMMAND ----------

def processo_leitura_transformacao(caminho_raw, txtfile, data, regras):
  txtfile = os.path.join(caminho_raw, data, txtfile)
  df = leCSV(txtfile)
  list_dfs = criaDFs(df)
  dict_return = {}
  for dfs in list_dfs:
    for tipo_registro in dfs:
      if 'detalhes' in tipo_registro:
        df = dfs[tipo_registro]
        df = txt_para_col(df, tipo_registro)
        dict_return.update({tipo_registro:df})
  return dict_return

# COMMAND ----------

datefile = {}
for date in list_caminho_raw:
  arquivos = os.listdir(os.path.join(caminho_raw_dbfs, date))
  if len(arquivos) > 0:
    datefile.update({date:arquivos})

# COMMAND ----------

index = 0
for date in datefile:
  index = index+1
  print (index, '.',date)

# COMMAND ----------

def escolhe_index_le_detalhes(index_escolhido, datefile):
  index = 0
  for date in datefile:
    index = index + 1
    arquivo = datefile[date]
    if index == index_escolhido:
      return arquivo[0], date

# COMMAND ----------

features_encontradas_detalhes_clientes = {}
features_encontradas_detalhes_dividas = {}
features_encontradas_detalhes_contratos = {}

# COMMAND ----------

arquivo, date = escolhe_index_le_detalhes(30, datefile)
print (arquivo, date)
dfs = processo_leitura_transformacao(caminho_raw, arquivo,date, regras)
for tipo_registro in dfs:
  if 'detalhes' in tipo_registro:
    df = dfs[tipo_registro]
    display(df)

# COMMAND ----------

tipos_registros_detalhes = ['04', '06', '08']

# COMMAND ----------

for date in datefile:
  for file in datefile[date]:
    print (date, file)
    with open (os.path.join(caminho_raw_dbfs, date, file)) as openfile:
      lines = openfile.readlines(1024) #corrigir 1024 para trazer tudo
      for l in lines:
        with open (os.path.join(detalhespath, 'features_detalhes.csv')) as outfile:
          escritor = csv.writer(outfile)
          escritor.writerow(l)

# COMMAND ----------

tipos_registros_detalhes = ['04', '06', '08']
for date in datefile:
  for file in (datefile[date]):
    print (file)
    filepath = os.path.join(caminho_raw_dbfs, date,file)
    with open (filepath, 'r', encoding = 'latin-1') as openfile:
      lines = openfile.readlines() #corrigir 1024 para trazer tudo
      for l in lines:
        if l[0:2] in tipos_registros_detalhes:
          print (l)

# COMMAND ----------

# DBTITLE 1,EXCLUINDO ARQUIVOS TEMPORARIOS
"""try:
  list_caminho_raw = os.listdir(caminho_raw)
  for file in list_caminho_raw:
    os.remove(os.path.join(caminho_raw,file))
except:
  pass"""