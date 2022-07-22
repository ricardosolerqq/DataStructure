# Databricks notebook source
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

def get_creditor_etl_file_date(credor, latest = False, as_date = False):
  ackey = dbutils.secrets.get(scope = "scope_qqdatastoragemain", key = "qqprd-key")
  block_blob_service = BlockBlobService(account_name='qqprd', account_key=ackey)
  
  prefix = "etl/"+credor+"/processed"
  
  generator = block_blob_service.list_blobs('qq-integrator', prefix=prefix)
  
  file_date = {}
  
  for blob in generator:   
    nome = blob.name.split('/')[-1]
    if nome.find('AdicionarQueroQuitar',0,20) == 0:
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

import os
import datetime
import zipfile

blob_account_source_prd = "qqprd"
blob_container_source_prd = "qq-integrator"
blob_account_source_ml = "qq-data-studies"
blob_container_source_ml = "ml-prd"

dir_arqs = 'dbfs:/mnt/qq-integrator/etl/recovery/processed/'

mount_blob_storage_key(dbutils,blob_account_source_prd,blob_account_source_prd,'/mnt/qq-integrator', key='qqprd-key')
mount_blob_storage_key(dbutils,blob_account_source_ml,blob_container_source_ml,'/mnt/ml-prd')


prefix = "etl/recovery/processed"

caminho_base = '/mnt/qq-integrator/etl/recovery/processed'
caminho_base_dbfs = '/dbfs/mnt/qq-integrator/etl/recovery/processed'
caminho_trusted = '/mnt/ml-prd/ml-data/propensaodeal/recovery/trusted'

# COMMAND ----------

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
  arquivo_escolhido = list(get_creditor_etl_file_date('recovery', latest=True))[0]
  arquivo_escolhido_path = os.path.join(caminho_base, arquivo_escolhido)
  arquivo_escolhido_path_dbfs = os.path.join('/dbfs', caminho_base, arquivo_escolhido)

  data_arq = arquivo_escolhido.split('.')[0][24:]
  dia = int(data_arq[6:])
  mes = int(data_arq[4:6])
  ano = int(data_arq[0:4])
  data_arq = datetime.date(ano, mes, dia)
  data_arq
  
else:
  fileList = list(get_creditor_etl_file_date('recovery', latest=False))  
  max_file = max(fileList)
  dbutils.widgets.multiselect('ARQUIVO_ESCOLHIDO', max_file, fileList+[''])
  arqs = dbutils.widgets.get('ARQUIVO_ESCOLHIDO').split(',')

# COMMAND ----------

#arqs=spark.createDataFrame(dbutils.fs.ls(dir_arqs)).withColumn('path',F.regexp_replace("path","dbfs:",""))
#arqs=arqs.filter(F.col('name').contains('AdicionarQueroQuitar'))
#arqs=arqs.withColumn('dataArq',F.to_date(F.col('name')[25:8],'yyyyMMdd'))
#arqs=arqs.orderBy(F.col('dataArq').desc())
#arqs=arqs.filter(F.col('dataArq') >= '2021-11-05')
#arqs= arqs.collect()
#arqs

# COMMAND ----------

if processo_auto:
  
    tab_i = spark.read.options(header = True, delimiter = ';', encoding = 'latin1').csv(arquivo_escolhido_path)
    tab_i = tab_i.withColumn('dt_arquivo', F.lit(data_arq))

    ###CPF###
    
    Tab_correct = tab_i.dropna(subset=['CPF']).withColumnRenamed("CPF","document")
    Tab_correct=Tab_correct.withColumn('document', F.lpad('document',11,'0'))
    
    ###CNPJ###
    
    #aux_CNPJs=tab_i.dropna(subset=['CNPJ']).withColumnRenamed("CNPJ","document")
    #aux_CNPJs=aux_CNPJs.withColumn('document', F.lpad('document',14,'0'))

    
    # Agregando as bases:
    
    
    Tab_correct=Tab_correct.withColumn('VlDividaAtualizado',F.regexp_replace(F.col('VlDividaAtualizado'),',','.').cast('float'))\
                                                  .withColumn('VlSOP',F.regexp_replace(F.col('VlSOP'),',','.').cast('float'))\
                                                  .withColumn('IdContatoSIR',F.col("IdContatoSIR").cast('int'))\
                                                  .withColumn('Chave', F.concat_ws(':',F.col('document'), F.col('Numero_Contrato')))\
                                                  .withColumnRenamed('Numero_Contrato','contrato')
    
    # Dropando as variáveis sem informações de 'IdContatoSIR' e 'VlDividaAtualizado'

    Tab_correct = Tab_correct.filter((F.col('IdContatoSIR') !=0) |  (F.col('VlDividaAtualizado') !=0))
    
    
    # Ajustando as variáveis de Carteira, Produto e Portfolio

    Tab_correct = Tab_correct.withColumn('Carteira', F.lower('Carteira'))\
                        .withColumn('Class_Carteira', F.when(F.col('Carteira') == "bradesco npl2","a")\
                                                      .when(F.col('Carteira') == "itau iresolve","b")\
                                                      .when(F.col('Carteira') == "bradesco npl1","c")\
                                                      .when(F.col('Carteira') == "natura","d")\
                                                      .when(F.col('Carteira') == "std-renova-01","e")\
                                                      .when(F.col('Carteira') == "santander npl1","f")\
                                                      .when(F.col('Carteira') == "caixa npl1","g")\
                                                      .when(F.col('Carteira') == "caixa","h")\
                                                      .when(F.col('Carteira') == "pontocred","i")\
                                                      .when(F.col('Carteira') == "santander npl1- 2","j")\
                                                      .otherwise('k'))\
                       .withColumn('Produto', F.lower('Produto'))\
                       .withColumn('Produto',F.regexp_replace(F.col("Produto"), "ã", "a"))\
                       .withColumn('Produto',F.regexp_replace(F.col("Produto"), "é", "e"))\
                       .withColumn('Produto',F.regexp_replace(F.col("Produto"), "á", "a"))\
                       .withColumn('Produto',F.regexp_replace(F.col("Produto"), "ç", "c"))\
                       .withColumn('Class_Produto', F.when(F.col('Produto') == "cartao de credito","a")\
                                                      .when(F.col('Produto') == "emprestimos","b")\
                                                      .when(F.col('Produto') == "cartao mastercard","c")\
                                                      .when(F.col('Produto') == "renegociacao","d")\
                                                      .when(F.col('Produto') == "cartao visa","e")\
                                                      .when(F.col('Produto') == "contas correntes","f")\
                                                      .when(F.col('Produto') == "financiamento","g")\
                                                      .when(F.col('Produto') == "compra natura","h")\
                                                      .when(F.col('Produto') == "crediario","i")\
                                                      .when(F.col('Produto') == "cartao - renegociacao","j")\
                                                      .otherwise('k'))\
                         .withColumn('Portfolio', F.split('Carteira',' ').getItem(0))\
                         .withColumn('Class_Portfolio', F.when(F.col('Portfolio') == "bradesco","a")\
                                                      .when(F.col('Portfolio') == "itau","b")\
                                                      .when(F.col('Portfolio') == "santander","c")\
                                                      .when(F.col('Portfolio') == "caixa","d")\
                                                      .when(F.col('Portfolio') == "natura","e")\
                                                      .when(F.col('Portfolio') == "std-renova-01","f")\
                                                      .when(F.col('Portfolio') == "pontocred","g")\
                                                      .when(F.col('Portfolio') == "banco","h")\
                                                      .when(F.col('Portfolio') == "sorocred","i")\
                                                      .when(F.col('Portfolio') == "citi","j")\
                                                      .otherwise('k'))\
                         .withColumn('Data_Mora', F.from_unixtime(F.unix_timestamp('Data_Mora', 'dd/MM/yyyy')))
    
    Tab_correct =  Tab_correct.select("document","contrato","IdContatoSIR","VlDividaAtualizado","Chave","Class_Carteira","Class_Produto","Class_Portfolio",'Data_Mora','dt_arquivo') 

  
else:
  count = 0

  for file in arqs:
    if count == 0:
      data_arq = file.split('.')[0][24:]
      dia = int(data_arq[6:])
      mes = int(data_arq[4:6])
      ano = int(data_arq[0:4])
      data_arq = datetime.date(ano, mes, dia)
      print('Lendo e formatando o primeiro arquivo: '+file)
      # Inclusão da data no banco de dados
      tab_i = spark.read.options(header = True, delimiter = ';', encoding = 'latin1').csv(os.path.join(caminho_base, file))
      tab_i = tab_i.withColumn('dt_arquivo', F.lit(data_arq))
  
      ###CPF###
      
      Tab_correct = tab_i.dropna(subset=['CPF']).withColumnRenamed("CPF","document")
      Tab_correct=Tab_correct.withColumn('document', F.lpad('document',11,'0'))
      
      ###CNPJ###
      
      #aux_CNPJs=tab_i.dropna(subset=['CNPJ']).withColumnRenamed("CNPJ","document")
      #aux_CNPJs=aux_CNPJs.withColumn('document', F.lpad('document',14,'0'))
  
      
      # Agregando as bases:
      
      
      Tab_correct=Tab_correct.withColumn('VlDividaAtualizado',F.regexp_replace(F.col('VlDividaAtualizado'),',','.').cast('float'))\
                                                    .withColumn('VlSOP',F.regexp_replace(F.col('VlSOP'),',','.').cast('float'))\
                                                    .withColumn('IdContatoSIR',F.col("IdContatoSIR").cast('int'))\
                                                    .withColumn('Chave', F.concat_ws(':',F.col('document'), F.col('Numero_Contrato')))\
                                                    .withColumnRenamed('Numero_Contrato','contrato')
      
      # Dropando as variáveis sem informações de 'IdContatoSIR' e 'VlDividaAtualizado'
  
      Tab_correct = Tab_correct.filter((F.col('IdContatoSIR') !=0) |  (F.col('VlDividaAtualizado') !=0))
      
      
      # Ajustando as variáveis de Carteira, Produto e Portfolio
  
      Tab_correct = Tab_correct.withColumn('Carteira', F.lower('Carteira'))\
                          .withColumn('Class_Carteira', F.when(F.col('Carteira') == "bradesco npl2","a")\
                                                        .when(F.col('Carteira') == "itau iresolve","b")\
                                                        .when(F.col('Carteira') == "bradesco npl1","c")\
                                                        .when(F.col('Carteira') == "natura","d")\
                                                        .when(F.col('Carteira') == "std-renova-01","e")\
                                                        .when(F.col('Carteira') == "santander npl1","f")\
                                                        .when(F.col('Carteira') == "caixa npl1","g")\
                                                        .when(F.col('Carteira') == "caixa","h")\
                                                        .when(F.col('Carteira') == "pontocred","i")\
                                                        .when(F.col('Carteira') == "santander npl1- 2","j")\
                                                        .otherwise('k'))\
                         .withColumn('Produto', F.lower('Produto'))\
                         .withColumn('Produto',F.regexp_replace(F.col("Produto"), "ã", "a"))\
                         .withColumn('Produto',F.regexp_replace(F.col("Produto"), "é", "e"))\
                         .withColumn('Produto',F.regexp_replace(F.col("Produto"), "á", "a"))\
                         .withColumn('Produto',F.regexp_replace(F.col("Produto"), "ç", "c"))\
                         .withColumn('Class_Produto', F.when(F.col('Produto') == "cartao de credito","a")\
                                                        .when(F.col('Produto') == "emprestimos","b")\
                                                        .when(F.col('Produto') == "cartao mastercard","c")\
                                                        .when(F.col('Produto') == "renegociacao","d")\
                                                        .when(F.col('Produto') == "cartao visa","e")\
                                                        .when(F.col('Produto') == "contas correntes","f")\
                                                        .when(F.col('Produto') == "financiamento","g")\
                                                        .when(F.col('Produto') == "compra natura","h")\
                                                        .when(F.col('Produto') == "crediario","i")\
                                                        .when(F.col('Produto') == "cartao - renegociacao","j")\
                                                        .otherwise('k'))\
                           .withColumn('Portfolio', F.split('Carteira',' ').getItem(0))\
                           .withColumn('Class_Portfolio', F.when(F.col('Portfolio') == "bradesco","a")\
                                                        .when(F.col('Portfolio') == "itau","b")\
                                                        .when(F.col('Portfolio') == "santander","c")\
                                                        .when(F.col('Portfolio') == "caixa","d")\
                                                        .when(F.col('Portfolio') == "natura","e")\
                                                        .when(F.col('Portfolio') == "std-renova-01","f")\
                                                        .when(F.col('Portfolio') == "pontocred","g")\
                                                        .when(F.col('Portfolio') == "banco","h")\
                                                        .when(F.col('Portfolio') == "sorocred","i")\
                                                        .when(F.col('Portfolio') == "citi","j")\
                                                        .otherwise('k'))\
                           .withColumn('Data_Mora', F.from_unixtime(F.unix_timestamp('Data_Mora', 'dd/MM/yyyy')))
      
      Tab_correct =  Tab_correct.select("document","contrato","IdContatoSIR","VlDividaAtualizado","Chave","Class_Carteira","Class_Produto","Class_Portfolio",'Data_Mora','dt_arquivo')  
  
    else:
      print('Lendo e formatando o arquivo: '+file)
      data_arq = file.split('.')[0][24:]
      dia = int(data_arq[6:])
      mes = int(data_arq[4:6])
      ano = int(data_arq[0:4])
      data_arq = datetime.date(ano, mes, dia)
      tab_i = spark.read.options(header = True, delimiter = ';', encoding = 'latin1').csv(os.path.join(caminho_base, file))
      tab_i = tab_i.withColumn('dt_arquivo', F.lit(data_arq))
  
      ###CPF###
      aux_insert = tab_i.dropna(subset=['CPF']).withColumnRenamed("CPF","document")
      aux_insert=aux_insert.withColumn('document', F.lpad('document',11,'0'))
      
      ###CNPJ###  
      #aux_CNPJs=tab_i.dropna(subset=['CNPJ']).withColumnRenamed("CNPJ","document")
      #aux_CNPJs=aux_CNPJs.withColumn('document', F.lpad('document',14,'0'))
      
      
      
      # Agregando as bases:
      
      
      aux_insert=aux_insert.withColumn('VlDividaAtualizado',F.regexp_replace(F.col('VlDividaAtualizado'),',','.').cast('float'))\
                                                    .withColumn('VlSOP',F.regexp_replace(F.col('VlSOP'),',','.').cast('float'))\
                                                    .withColumn('IdContatoSIR',F.col("IdContatoSIR").cast('int'))\
                                                    .withColumn('Chave', F.concat_ws(':',F.col('document'), F.col('Numero_Contrato')))\
                                                    .withColumnRenamed('Numero_Contrato','contrato')
  
      # Dropando as variáveis sem informações de 'IdContatoSIR' e 'VlDividaAtualizado'
  
      aux_insert = aux_insert.filter((F.col('IdContatoSIR') !=0) |  (F.col('VlDividaAtualizado') !=0))
      
      
      # Ajustando as variáveis de Carteira, Produto e Portfolio
  
      aux_insert = aux_insert.withColumn('Carteira', F.lower('Carteira'))\
                          .withColumn('Class_Carteira', F.when(F.col('Carteira') == "bradesco npl2","a")\
                                                        .when(F.col('Carteira') == "itau iresolve","b")\
                                                        .when(F.col('Carteira') == "bradesco npl1","c")\
                                                        .when(F.col('Carteira') == "natura","d")\
                                                        .when(F.col('Carteira') == "std-renova-01","e")\
                                                        .when(F.col('Carteira') == "santander npl1","f")\
                                                        .when(F.col('Carteira') == "caixa npl1","g")\
                                                        .when(F.col('Carteira') == "caixa","h")\
                                                        .when(F.col('Carteira') == "pontocred","i")\
                                                        .when(F.col('Carteira') == "santander npl1- 2","j")\
                                                        .otherwise('k'))\
                         .withColumn('Produto', F.lower('Produto'))\
                         .withColumn('Produto',F.regexp_replace(F.col("Produto"), "ã", "a"))\
                         .withColumn('Produto',F.regexp_replace(F.col("Produto"), "é", "e"))\
                         .withColumn('Produto',F.regexp_replace(F.col("Produto"), "á", "a"))\
                         .withColumn('Produto',F.regexp_replace(F.col("Produto"), "ç", "c"))\
                         .withColumn('Class_Produto', F.when(F.col('Produto') == "cartao de credito","a")\
                                                        .when(F.col('Produto') == "emprestimos","b")\
                                                        .when(F.col('Produto') == "cartao mastercard","c")\
                                                        .when(F.col('Produto') == "renegociacao","d")\
                                                        .when(F.col('Produto') == "cartao visa","e")\
                                                        .when(F.col('Produto') == "contas correntes","f")\
                                                        .when(F.col('Produto') == "financiamento","g")\
                                                        .when(F.col('Produto') == "compra natura","h")\
                                                        .when(F.col('Produto') == "crediario","i")\
                                                        .when(F.col('Produto') == "cartao - renegociacao","j")\
                                                        .otherwise('k'))\
                           .withColumn('Portfolio', F.split('Carteira',' ').getItem(0))\
                           .withColumn('Class_Portfolio', F.when(F.col('Portfolio') == "bradesco","a")\
                                                        .when(F.col('Portfolio') == "itau","b")\
                                                        .when(F.col('Portfolio') == "santander","c")\
                                                        .when(F.col('Portfolio') == "caixa","d")\
                                                        .when(F.col('Portfolio') == "natura","e")\
                                                        .when(F.col('Portfolio') == "std-renova-01","f")\
                                                        .when(F.col('Portfolio') == "pontocred","g")\
                                                        .when(F.col('Portfolio') == "banco","h")\
                                                        .when(F.col('Portfolio') == "sorocred","i")\
                                                        .when(F.col('Portfolio') == "citi","j")\
                                                        .otherwise('k'))\
                           .withColumn('Data_Mora', F.from_unixtime(F.unix_timestamp('Data_Mora', 'dd/MM/yyyy')))
    
    
      #columns_to_drop = ['CNPJ','Nome_Cliente / Empresa','Carteira','Produto','Portfolio','SubTipo Produto','NrCpfCnpjAvalista1', 'NrCpfCnpjAvalista1','NomeAvalista1','NrCpfCnpjAvalista2', 'NomeAvalista2','VlSOP']
      #aux_insert = aux_insert.drop(*columns_to_drop)
      aux_insert =  aux_insert.select("document","contrato","IdContatoSIR","VlDividaAtualizado","Chave","Class_Carteira","Class_Produto","Class_Portfolio",'Data_Mora','dt_arquivo')    
      Tab_correct=Tab_correct.union(aux_insert)
      del aux_insert
    count += 1

# COMMAND ----------

display(Tab_correct)

# COMMAND ----------

# DBTITLE 1,Acumulando todos os únicos (contrato+documento) pela entrada mais recente
# Aqui os arquivos estão ordenados da data mais recente para a data mais antiga
df=Tab_correct.groupBy('document','contrato','IdContatoSIR','Chave','Class_Carteira','Class_Produto','Class_Portfolio','Data_Mora').agg(F.first('VlDividaAtualizado').alias('VlDividaAtualizado'))

# COMMAND ----------

# DBTITLE 1,Realizando o particionamento e salvando no blob
df.coalesce(1).write.option('sep', ';').option('header', 'True').csv(os.path.join(caminho_trusted, 'full_temp'))

if processo_auto:
  for file in dbutils.fs.ls(os.path.join(caminho_trusted, 'full_temp')):
    if file.name.split('.')[-1] == 'csv':
      dbutils.fs.cp(file.path, os.path.join(caminho_trusted,arquivo_escolhido.split('.')[0]+'.csv'))
    else:
      dbutils.fs.rm(file.path, True)
  dbutils.fs.rm(os.path.join(caminho_trusted, 'full_temp'), True)
else:
  from datetime import datetime, timedelta
  timestr = (datetime.today()).strftime('%Y%m%d')
  
  for file in dbutils.fs.ls(os.path.join(caminho_trusted, 'full_temp')):
    if file.name.split('.')[-1] == 'csv':
      dbutils.fs.cp(file.path, os.path.join(caminho_trusted,'MultiFilesRecovery'+timestr+'.csv'))
    else:
      dbutils.fs.rm(file.path, True)
  dbutils.fs.rm(os.path.join(caminho_trusted, 'full_temp'), True)

# COMMAND ----------

