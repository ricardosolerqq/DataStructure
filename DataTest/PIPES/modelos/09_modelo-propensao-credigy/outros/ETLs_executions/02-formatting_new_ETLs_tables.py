# Databricks notebook source
# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

#mount_blob_storage_oauth(dbutils,'qqdatastoragemain','ml-prd',"/mnt/ml-prd")
#dir_zips='/mnt/ml-prd/ml-data/propensaodeal/credigy/zip_extractor'
dir_knabs='/mnt/ml-prd/ml-data/propensaodeal/credigy/pre-trusted/knabs_tables'
dir_pre_trusted='/mnt/ml-prd/ml-data/propensaodeal/credigy/pre-trusted'
dir_trusted='/mnt/ml-prd/ml-data/propensaodeal/credigy/trusted'
dir_output='/mnt/ml-prd/ml-data/propensaodeal/credigy/output'

# COMMAND ----------

# DBTITLE 1,Lendo a tabela dos knabs
from pyspark.sql.functions import *

df_knabs=dbutils.fs.ls(dir_knabs)
df_knabs=spark.createDataFrame(df_knabs)
df_knabs=df_knabs.withColumn("path", regexp_replace("path","dbfs:",""))
df_knabs=df_knabs.withColumn('Tipo',F.col('name').substr(10,2))
df_knabs=df_knabs.withColumn('Registro',F.col('name').substr(17,2)).filter((F.col('Tipo').isin(['01','21'])) & (F.col('Registro')=='01')).collect()
#display(df_knabs)

# COMMAND ----------

tab_files_pre_trusted=dbutils.fs.ls(dir_pre_trusted)
tab_files_pre_trusted=spark.createDataFrame(tab_files_pre_trusted)
tab_files_pre_trusted=tab_files_pre_trusted.withColumn('Tipo',split(tab_files_pre_trusted['name'], '_').getItem(2)).withColumn('Registro',split(tab_files_pre_trusted['name'], '_').getItem(4)).withColumn("Registro", regexp_replace("Registro","/","")).withColumn('path', regexp_replace("path","dbfs:","")).filter((F.col('Tipo').isin(['01','21'])) & (F.col('Registro')=='01'))
#display(tab_files_pre_trusted)

# COMMAND ----------

# DBTITLE 1,Selecionando a data mais recente na trusted
df_pre_trusted=spark.createDataFrame(dbutils.fs.ls(dir_pre_trusted+'/tabelas_tipo_21_registro_01'))
df_pre_trusted=df_pre_trusted.union(spark.createDataFrame(dbutils.fs.ls(dir_pre_trusted+'/tabelas_tipo_01_registro_01')))
df_pre_trusted=df_pre_trusted.withColumn('data_arquivo',to_date(regexp_replace(split(F.col('name'),'=').getItem(1),'/','')))
max_pre_trusted=df_pre_trusted.select(F.max(F.col("data_arquivo")).alias("MAX")).limit(1).collect()[0].MAX
del df_pre_trusted
#max_pre_trusted

# COMMAND ----------

# DBTITLE 1,Selecionando a data mais recente na trusted
df_trusted=spark.createDataFrame(dbutils.fs.ls(dir_trusted+'/tabelas_tipo_21_registro_01'))
df_trusted=df_trusted.union(spark.createDataFrame(dbutils.fs.ls(dir_trusted+'/tabelas_tipo_01_registro_01')))
df_trusted=df_trusted.withColumn('data_arquivo',to_date(regexp_replace(split(F.col('name'),'=').getItem(1),'/','')))
max_trusted=df_trusted.select(F.max(F.col("data_arquivo")).alias("MAX")).limit(1).collect()[0].MAX
del df_trusted

# COMMAND ----------

# DBTITLE 1,Lendo por knab e salvando no diretório
if max_trusted<max_pre_trusted:
  for i in df_knabs:
    dir_aux_files=tab_files_pre_trusted.filter((F.col('Tipo')==i.Tipo)&(F.col('Registro')==i.Registro)).collect()
    dir_aux_files=dbutils.fs.ls(dir_aux_files[0].path)
    dir_aux_files=spark.createDataFrame(dir_aux_files).withColumn("path", regexp_replace("path","dbfs:","")).withColumn('data',F.col('name').substr(14,10)).filter(F.col('data')>max_trusted).collect()
    aux_knabs=spark.read.option("header", True).option("delimiter", ";").option("encoding", 'UTF-8').csv(i.path).collect()
    for j in dir_aux_files:
      aux_format=spark.read.parquet(j.path+'*parquet')

      if aux_format.count()>0:

        for w in aux_knabs:
          aux_format=aux_format.withColumn(w.COL_NAME,F.col('_c0').substr(int(w.Inicio),int(w.Tam)))

        aux_format=aux_format.drop(F.col('_c0'))  
        aux_format.write.mode('overwrite').option("sep",";").option("header",True).option('emptyValue','').parquet(dir_trusted+'/tabelas_tipo_'+i.Tipo+'_registro_'+i.Registro+"/data_arquivo="+j.data)
        aux_format.write.mode('overwrite').option("sep",";").option("header",True).option('emptyValue','').parquet(dir_trusted+'/outputs_tipo_'+i.Tipo+'_registro_'+i.Registro+"/data_arquivo="+j.data)

    del aux_knabs,dir_aux_files
else:
  print("Não há novos arquivos à serem atualizados na trusted.")