# Databricks notebook source
# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

# DBTITLE 1,Carregando os blobs e ajustando diretórios
mount_blob_storage_key(dbutils,'qqprd','qq-integrator',"/mnt/qq-integrator")
mount_blob_storage_key(dbutils,'qqprd','ml-prd',"/mnt/ml-prd")
dir_zips='/mnt/ml-prd/ml-data/propensaodeal/credigy/zip_extractor'

# COMMAND ----------

# DBTITLE 1,Ajustando o diretório e as bases credigy
from pyspark.sql.functions import *

dir_ETLs="/mnt/qq-integrator/etl/credigy/processed"
files=dbutils.fs.ls(dir_ETLs)
files=spark.createDataFrame(files)
files=files.withColumn("path", regexp_replace("path","dbfs:",""))
files=files.withColumn('data_referencia',F.col('name').substr(16,8))
files=files.withColumn('data_referencia',to_timestamp(col("data_referencia"),"yyyyMMdd"))
files=files.withColumn('conta_arquivo',F.col('name').substr(24,2))
files=files.withColumn('tipo_arquivo',F.col('name').substr(27,3))
files=files.sort(F.col('data_referencia').desc())
zip_files=files.filter(F.col('tipo_arquivo')=='zip')
zip_files=zip_files.withColumn('name',regexp_replace("name","zip","txt")).collect()
files=files.filter(F.col('tipo_arquivo')=='txt')
display(files)

# COMMAND ----------

# DBTITLE 1,Descomprimindo e salvando nas pastas auxiliares os .zip
import zipfile
zip_files_saved=os.listdir('/dbfs/'+dir_zips)

for i in zip_files:
  if i.name in zip_files_saved:
    print('Arquivo '+i.path+' já foi deszipado, indo para o seguinte...')
  else:
    print('Descomprimindo o arquivo '+i.path+'...')
    with zipfile.ZipFile('/dbfs/'+i.path, 'r') as zip_ref:
      zip_ref.extractall('/dbfs/'+dir_zips)

# COMMAND ----------

# DBTITLE 1,Formatando a tabela dos arquivos deszipados e concatenando com a principal
zip_files_saved=dbutils.fs.ls(dir_zips)
zip_files_saved=spark.createDataFrame(zip_files_saved)
zip_files_saved=zip_files_saved.withColumn("path", regexp_replace("path","dbfs:",""))
zip_files_saved=zip_files_saved.withColumn('data_referencia',F.col('name').substr(16,8))
zip_files_saved=zip_files_saved.withColumn('data_referencia',to_timestamp(col("data_referencia"),"yyyyMMdd"))
zip_files_saved=zip_files_saved.withColumn('conta_arquivo',F.col('name').substr(24,2))
zip_files_saved=zip_files_saved.withColumn('tipo_arquivo',F.col('name').substr(27,3))
files=files.union(zip_files_saved)
files=files.sort(F.col('data_referencia').desc()).collect()
del zip_files_saved,zip_files
display(files)

# COMMAND ----------

# DBTITLE 1,Lendo todos os arquivos
for i in files:
  print('Lendo o arquivo "'+i.name+'"')
  if i.path==files[0].path:
    bases_credigy=spark.read.option("header", False).option("delimiter", ";").option("encoding", 'UTF-8').csv(i.path)
  else:
    aux=spark.read.option("header", False).option("delimiter", ";").option("encoding", 'UTF-8').csv(i.path)
    bases_credigy=bases_credigy.union(aux)
    
del aux

# COMMAND ----------

# DBTITLE 1,Selecionando as colunas de definição de tabela e pegando apenas os valores únicos
bases_credigy=bases_credigy.withColumn('TP_ARQ',F.col('_c0').substr(0,2)).withColumn('TP_REG',F.col('_c0').substr(3,2))
count_credigy=bases_credigy.select(['TP_ARQ','TP_REG'])
count_credigy=count_credigy.distinct()

# COMMAND ----------

# DBTITLE 1,Salvando a tabela com as contagens
dir_save='dbfs:/mnt/ml-prd/ml-data/propensaodeal/credigy/others_tables/'

count_credigy.coalesce(1).write.mode('overwrite').option("header",True).option("delimiter",";").option("emptyValue",'').csv(dir_save+'credigy_uniques_tables')

for file in dbutils.fs.ls(dir_save+'credigy_uniques_tables'):
  if file.name.split('.')[-1] == 'csv':
    print (file)
    dbutils.fs.cp(file.path, dir_save+'credigy_uniques_tables.csv')
dbutils.fs.rm(dir_save+'credigy_uniques_tables', True)