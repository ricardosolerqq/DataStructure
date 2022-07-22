# Databricks notebook source
# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

# DBTITLE 1,Carregando os diretórios
#mount_blob_storage_oauth(dbutils,'qqprd','qq-integrator',"/mnt/qq-integrator")
#mount_blob_storage_oauth(dbutils,'qqdatastoragemain','ml-prd',"/mnt/ml-prd")
dir_zips='/mnt/ml-prd/ml-data/propensaodeal/credigy/zip_extractor'
dir_knabs='/mnt/ml-prd/ml-data/propensaodeal/credigy/pre-trusted/knabs_tables'
dir_pre_trusted='/mnt/ml-prd/ml-data/propensaodeal/credigy/pre-trusted'
dir_ETLs="/mnt/qq-integrator/etl/credigy/processed"
#dir_s3_tables='/mnt/ml-prd/ml-data/propensaodeal/credigy/tables_s3'

# COMMAND ----------

# DBTITLE 1,Ajustando as tabelas knabs auxiliares (somente aquelas das infos descritivas)
from pyspark.sql.functions import *

df_knabs=dbutils.fs.ls(dir_knabs)
df_knabs=spark.createDataFrame(df_knabs)
df_knabs=df_knabs.withColumn("path", regexp_replace("path","dbfs:",""))
df_knabs=df_knabs.withColumn('Tipo',F.col('name').substr(10,2))
df_knabs=df_knabs.withColumn('Registro',F.col('name').substr(17,2)).filter((F.col('Tipo').isin(['01','21'])) & (F.col('Registro')=='01')) .collect()

#display(df_knabs)

# COMMAND ----------

df_files=dbutils.fs.ls(dir_ETLs)
df_files=spark.createDataFrame(df_files)
df_files=df_files.withColumn("path", regexp_replace("path","dbfs:",""))
df_files=df_files.withColumn('data_referencia',F.col('name').substr(16,8))
df_files=df_files.withColumn('data_referencia',to_timestamp(col("data_referencia"),"yyyyMMdd"))
df_files=df_files.withColumn('formato_arquivo',F.col('name').substr(27,3))
df_files=df_files.withColumn('tipo_arquivo',F.col('name').substr(8,2)).filter(F.col('tipo_arquivo').isin(['01','21']))
display(df_files)

# COMMAND ----------

# DBTITLE 1,Selecionando a data mais recente na pré-trusted
df_pre_trusted=spark.createDataFrame(dbutils.fs.ls(dir_pre_trusted+'/tabelas_tipo_21_registro_01'))
df_pre_trusted=df_pre_trusted.union(spark.createDataFrame(dbutils.fs.ls(dir_pre_trusted+'/tabelas_tipo_01_registro_01')))
df_pre_trusted=df_pre_trusted.withColumn('data_arquivo',to_date(regexp_replace(split(F.col('name'),'=').getItem(1),'/','')))
max_pre_trusted=df_pre_trusted.select(F.max(F.col("data_arquivo")).alias("MAX")).limit(1).collect()[0].MAX
del df_pre_trusted

# COMMAND ----------

# DBTITLE 1,Selecionando somente os novos arquivos
df_files=df_files.filter(F.col('data_referencia')>max_pre_trusted)

# COMMAND ----------

# DBTITLE 1,Carregando as bases
###Selecionando somente os zipados###
zip_files=df_files.filter(F.col('formato_arquivo')=='zip')
zip_files=zip_files.withColumn('name',regexp_replace("name","zip","txt"))
###Separando os txt###
df_files=df_files.filter(F.col('formato_arquivo')=='txt')
#display(zip_files)

# COMMAND ----------

# DBTITLE 1,Descomprimindo os arquivos zip e concatenando na tabela principal
if zip_files.count()>0:
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
  zip_files_saved=zip_files_saved.withColumn('data_referencia',F.col('name').substr(16,8))
  zip_files_saved=zip_files_saved.withColumn('data_referencia',to_timestamp(col("data_referencia"),"yyyyMMdd")).filter(F.col('data_referencia')>max_pre_trusted)
  zip_files_saved=zip_files_saved.withColumn("path", regexp_replace("path","dbfs:",""))
  zip_files_saved=zip_files_saved.withColumn('formato_arquivo',F.col('name').substr(27,3)).withColumn('tipo_arquivo',F.col('name').substr(8,2)).filter(F.col('tipo_arquivo').isin(['01','21']))
  df_files=df_files.union(zip_files_saved)
  del zip_files_saved

del zip_files
datas=df_files.select(F.col('data_referencia')).distinct().sort(F.col('data_referencia').asc()).collect()
df_files=df_files.sort(F.col('data_referencia').asc())

#display(df_files)

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