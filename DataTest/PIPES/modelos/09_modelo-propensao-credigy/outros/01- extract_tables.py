# Databricks notebook source
import time
time.sleep(300)

# COMMAND ----------

# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

# DBTITLE 1,Carregando os diretórios
mount_blob_storage_oauth(dbutils,'qqprd','qq-integrator',"/mnt/qq-integrator")
mount_blob_storage_oauth(dbutils,'qqdatastoragemain','ml-prd',"/mnt/ml-prd")
dir_zips='/mnt/ml-prd/ml-data/propensaodeal/credigy/zip_extractor'
dir_knabs='/mnt/ml-prd/ml-data/propensaodeal/credigy/pre-trusted/knabs_tables'
dir_pre_trusted='/mnt/ml-prd/ml-data/propensaodeal/credigy/pre-trusted'
dir_s3_tables='/mnt/ml-prd/ml-data/propensaodeal/credigy/tables_s3'

# COMMAND ----------

# DBTITLE 1,Ajustando as tabelas knabs auxiliares
from pyspark.sql.functions import *

df_knabs=dbutils.fs.ls(dir_knabs)
df_knabs=spark.createDataFrame(df_knabs)
df_knabs=df_knabs.withColumn("path", regexp_replace("path","dbfs:",""))
df_knabs=df_knabs.withColumn('Tipo',F.col('name').substr(10,2))
df_knabs=df_knabs.withColumn('Registro',F.col('name').substr(17,2)).filter(F.col('Tipo').isin(['01','21','22'])) .collect()

display(df_knabs)

# COMMAND ----------

###Bases processed do S3###
dir_ETLs="/mnt/qq-integrator/etl/credigy/processed"
df_files=dbutils.fs.ls(dir_ETLs)
df_files=spark.createDataFrame(df_files)
df_files=df_files.withColumn("path", regexp_replace("path","dbfs:",""))
df_files=df_files.withColumn('data_referencia',F.col('name').substr(16,8))
df_files=df_files.withColumn('data_referencia',to_timestamp(col("data_referencia"),"yyyyMMdd"))
df_files=df_files.withColumn('conta_arquivo',F.col('name').substr(24,2))
df_files=df_files.withColumn('tipo_arquivo',F.col('name').substr(27,3))
display(df_files)

# COMMAND ----------

#df_files.agg({"size": "max"}).collect()[0]
display(df_files.sort('size',ascending=False))

# COMMAND ----------

# DBTITLE 1,Carregando as bases
###Bases processed do S3###
dir_ETLs="/mnt/qq-integrator/etl/credigy/processed"
df_files=dbutils.fs.ls(dir_ETLs)
df_files=spark.createDataFrame(df_files)
df_files=df_files.withColumn("path", regexp_replace("path","dbfs:",""))
df_files=df_files.withColumn('data_referencia',F.col('name').substr(16,8))
df_files=df_files.withColumn('data_referencia',to_timestamp(col("data_referencia"),"yyyyMMdd"))
df_files=df_files.withColumn('conta_arquivo',F.col('name').substr(24,2))
df_files=df_files.withColumn('tipo_arquivo',F.col('name').substr(27,3))
###Selecionando somente os zipados###
zip_files=df_files.filter(F.col('tipo_arquivo')=='zip')
zip_files=zip_files.withColumn('name',regexp_replace("name","zip","txt"))
###Bases do S3###
zip_s3_files=dbutils.fs.ls(dir_s3_tables)
zip_s3_files=spark.createDataFrame(zip_s3_files)
zip_s3_files=zip_s3_files.withColumn("path", regexp_replace("path","dbfs:",""))
zip_s3_files=zip_s3_files.withColumn('data_referencia',F.col('name').substr(16,8))
zip_s3_files=zip_s3_files.withColumn('data_referencia',to_timestamp(col("data_referencia"),"yyyyMMdd"))
zip_s3_files=zip_s3_files.withColumn('conta_arquivo',F.col('name').substr(24,2))
zip_s3_files=zip_s3_files.withColumn('tipo_arquivo',F.col('name').substr(27,3))
zip_s3_files=zip_s3_files.filter(F.col('tipo_arquivo')=='zip')
zip_s3_files=zip_s3_files.withColumn('name',regexp_replace("name","zip","txt"))
###Concatenando as duas tabelas a serem dezipadas##
zip_files=zip_files.union(zip_s3_files).collect()
###Separando os txt###
df_files=df_files.filter(F.col('tipo_arquivo')=='txt')

del zip_s3_files
display(zip_files)

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Descomprimindo os arquivos zip e concatenando na tabela principal
import zipfile
zip_files_saved=os.listdir('/dbfs/'+dir_zips)

for i in zip_files:
  if i.name in zip_files_saved:
    print('Arquivo '+i.path+' já foi deszipado, indo para o seguinte...')
  else:
    print('Descomprimindo o arquivo '+i.path+'...')
    with zipfile.ZipFile('/dbfs/'+i.path, 'r') as zip_ref:
      zip_ref.extractall('/dbfs/'+dir_zips)
      
zip_files_saved=dbutils.fs.ls(dir_zips)
zip_files_saved=spark.createDataFrame(zip_files_saved)
zip_files_saved=zip_files_saved.withColumn("path", regexp_replace("path","dbfs:",""))
zip_files_saved=zip_files_saved.withColumn('data_referencia',F.col('name').substr(16,8))
zip_files_saved=zip_files_saved.withColumn('data_referencia',to_timestamp(col("data_referencia"),"yyyyMMdd"))
zip_files_saved=zip_files_saved.withColumn('conta_arquivo',F.col('name').substr(24,2))
zip_files_saved=zip_files_saved.withColumn('tipo_arquivo',F.col('name').substr(27,3))
df_files=df_files.union(zip_files_saved)
datas=df_files.select(F.col('data_referencia')).distinct().sort(F.col('data_referencia').desc())
df_files=df_files.sort(F.col('data_referencia').desc())
del zip_files_saved,zip_files
display(df_files)

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Removendo as datas já existentes
datas=datas.withColumn('data_formatada',date_format(F.col('data_referencia'),"yyyy-MM-dd"))
files_executed=dbutils.fs.ls(dir_pre_trusted)
files_executed=spark.createDataFrame(files_executed)
if files_executed.count()>0:
  files_executed=files_executed.withColumn("path", regexp_replace("path","dbfs:","")).collect()
  files_executed=dbutils.fs.ls(files_executed[13].path)
  files_executed=spark.createDataFrame(files_executed)
  files_executed=files_executed.withColumn('data_to_compare',split(files_executed['name'], '=').getItem(1)).withColumn('data_to_compare',regexp_replace("data_to_compare","/","")).select(F.col('data_to_compare')).rdd.flatMap(lambda x: x).collect()
  datas=datas.filter(~(F.col('data_formatada').isin(files_executed)))

datas=datas.collect()
display(datas)

# COMMAND ----------

# DBTITLE 1,Lendo os arquivos por data
for w in datas:
  print("Lendo os arquivos do dia "+str(w.data_referencia.strftime("%d/%m/%Y")))
  aux=df_files.filter(F.col('data_referencia')==w.data_referencia).collect()
  for j in aux:
    print('Lendo o arquivo '+ j.name +'...')
    df_read=spark.read.option("header", False).option("delimiter", ";").option("encoding", 'UTF-8').csv(j.path)
    print('Salvando por knab...')
    for i in df_knabs:
      print("Selecionando os arquivos do Tipo "+i.Tipo+", Registro "+i.Registro+"...")
      aux_unformated=df_read.filter((F.col('_c0').substr(0,2)==i.Tipo) & (F.col('_c0').substr(3,2)==i.Registro))
      print('Salvando...')
      aux_unformated.write.mode('overwrite').option("sep",";").option("header",False).option('emptyValue','').parquet("{0}{1}/".format(dir_pre_trusted+'/tabelas_tipo_'+i.Tipo+'_registro_'+i.Registro+"/data_arquivo=",str(w.data_referencia.strftime("%Y-%m-%d"))))
      del aux_unformated  

print("Sucesso!!!")