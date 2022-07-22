# Databricks notebook source
spark.conf.set("spark.databricks.io.cache.enabled", "True")

# COMMAND ----------

# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

# DBTITLE 1,Ajustando os diretórios da Azure
blob_account_source_prd = "qqprd"
blob_container_source_prd = "qq-integrator"
blob_account_source_ml = "qq-data-studies"
blob_container_source_ml = "ml-prd"

dir_arqs = 'dbfs:/mnt/qq-integrator/etl/recovery/processed/'

#mount_blob_storage_oauth(dbutils,blob_account_source_prd, blob_container_source_prd, '/mnt/qq-integrator, key="kvdesQQ")
mount_blob_storage_key(dbutils,blob_account_source_prd,blob_account_source_prd,'/mnt/qq-integrator', key='qqprd-key')
mount_blob_storage_key(dbutils,blob_account_source_ml,blob_container_source_ml,'/mnt/ml-prd')

# COMMAND ----------

# DBTITLE 1,Selecionando somente os arquivos de entrada
from pyspark.sql.functions import *

arqs=os.listdir('/dbfs/mnt/qq-integrator/etl/recovery/processed/')
arqs=spark.createDataFrame(arqs,StringType()).toDF('entradas')
arqs=arqs.filter(arqs['entradas'].contains('AdicionarQueroQuitar'))
arqs=arqs.withColumn("diretorios",F.concat(F.lit(dir_arqs),F.col('entradas')))
arqs=arqs.orderBy(F.col('diretorios'),ascending=True).collect()
display(arqs)

# COMMAND ----------

# DBTITLE 1,Lendo e filtrando as variáveis
for filenames in arqs:

  if filenames.entradas==arqs[0].entradas:
    print('Lendo e formatando o arquivo "'+filenames.entradas,'"')
    ###Lendo o arquivo##
    aux_tabs=spark.read.format("csv").option("header", True).option("delimiter", ";").option("encoding", 'latin1').load(filenames.diretorios)
    ###CPF###
    Tab_correct=aux_tabs.select(F.col('CPF'),F.col('Numero_Contrato'),F.col('VlDividaAtualizado')).dropna(subset=['CPF']).withColumnRenamed("CPF","document")
    ###Corrigindo os zeros##
    if Tab_correct.count()>0:
      Tab_correct=Tab_correct.withColumn('document', F.lpad('document',11,'0'))
    ###CNPJ###
    aux_CNPJs=aux_tabs.select(F.col('CNPJ'),F.col('Numero_Contrato'),F.col('VlDividaAtualizado')).dropna(subset=['CNPJ']).withColumnRenamed("CNPJ","document")
    if aux_CNPJs.count()>0:
      aux_CNPJs=aux_CNPJs.withColumn('document', F.lpad('document',14,'0'))
    del aux_tabs
    ###Concatenando###
    Tab_correct=Tab_correct.union(aux_CNPJs).withColumn('VlDividaAtualizado',regexp_replace(F.col('VlDividaAtualizado'),',','.').cast('float'))
    del aux_CNPJs
    
  else:
    print('Lendo e formatando o arquivo "'+filenames.entradas,'"')
    ###Lendo o arquivo##
    aux_tabs=spark.read.format("csv").option("header", True).option("delimiter", ";").option("encoding", 'latin1').load(filenames.diretorios)
    ###CPF###
    aux_insert=aux_tabs.select(F.col('CPF'),F.col('Numero_Contrato'),F.col('VlDividaAtualizado')).dropna(subset=['CPF']).withColumnRenamed("CPF","document")
    ###Corrigindo os zeros##
    if aux_insert.count()>0:
      aux_insert=aux_insert.withColumn('document', F.lpad('document',11,'0'))
    ###CNPJ###
    aux_CNPJs=aux_tabs.select(F.col('CNPJ'),F.col('Numero_Contrato'),F.col('VlDividaAtualizado')).dropna(subset=['CNPJ']).withColumnRenamed("CNPJ","document")
    if aux_CNPJs.count()>0:
      aux_CNPJs=aux_CNPJs.withColumn('document', F.lpad('document',14,'0'))
    del aux_tabs
    ###Concatenando###
    aux_insert=aux_insert.union(aux_CNPJs).withColumn('VlDividaAtualizado',regexp_replace(F.col('VlDividaAtualizado'),',','.').cast('float'))
    del aux_CNPJs
    Tab_correct=Tab_correct.union(aux_insert)
    del aux_insert
#aux_tabs.printSchema()

#aux_insert=aux_insert.withColumn('Data',regexp_extract(F.lit('AdicionarQueroQuitarPFPJ20210201.txt'),"\\d+", 0))
#aux_insert=aux_insert.withColumn("Data",date_format(to_timestamp(col("Data"),"yyyyMMdd"),"dd/MM/yyyy"))
#display(Tab_correct)

# COMMAND ----------

# DBTITLE 1,Agrupando pelos valores mais recentes
Tab_correct=Tab_correct.groupby('document','Numero_Contrato').agg(F.last('VlDividaAtualizado').alias('VlDividaAtualizado'))
#display(Tab_correct)

# COMMAND ----------

# DBTITLE 1,Salvando
dir_save='dbfs:/mnt/ml-prd/ml-data/propensaodeal/recovery/features_to_insert/model_V2 (Elisvaldo)/'

Tab_correct.coalesce(1).write.mode('overwrite').option("header",True).option("delimiter",";").option("emptyValue",'').csv(dir_save+'feature_model_to_insert_recovery_20211011')

for file in dbutils.fs.ls(dir_save+'feature_model_to_insert_recovery_20211011'):
  if file.name.split('.')[-1] == 'csv':
    print (file)
    dbutils.fs.cp(file.path, dir_save+'feature_model_to_insert_recovery_20211011.csv')
dbutils.fs.rm(dir_save+'feature_model_to_insert_recovery_20211011', True)

# COMMAND ----------

# DBTITLE 1,Colocando em .zip
import shutil
import os

os.chdir('/dbfs/mnt/ml-prd/ml-data/propensaodeal/recovery/features_to_insert/model_V2 (Elisvaldo)')
dir_save_zip='feature_model_to_insert_recovery_20211011.csv'


#modelPath = "/dbfs/mnt/databricks/Models/predictBaseTerm/noNormalizationCode/2020-01-10-13-43/9_0.8147903598547376"
zipPath= "/recovery_features_to_mongoDB"
shutil.make_archive(base_dir= dir_save_zip, format='zip',base_name=zipPath)

blobStoragePath = "dbfs:/mnt/ml-prd/ml-data/propensaodeal/recovery/features_to_insert/model_V2 (Elisvaldo)"
dbutils.fs.cp("file:" +zipPath + ".zip", blobStoragePath)