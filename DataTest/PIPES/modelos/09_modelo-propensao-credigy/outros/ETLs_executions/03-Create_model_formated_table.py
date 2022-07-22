# Databricks notebook source
# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

# DBTITLE 1,Ajustando diretórios
#mount_blob_storage_oauth(dbutils,'qqprd','ml-prd',"/mnt/ml-prd")
#dir_trusted_read='/dbfs/mnt/ml-prd/ml-data/propensaodeal/credigy/trusted'
dir_outputs_trusted='/mnt/ml-prd/ml-data/propensaodeal/credigy/trusted'
#dir_outputs_01='/mnt/ml-prd/ml-data/propensaodeal/credigy/trusted/outputs_tipo_01_registro_01'
#dir_register='/mnt/ml-prd/ml-data/propensaodeal/credigy/registration_table'

# COMMAND ----------

dir_reads=spark.createDataFrame(dbutils.fs.ls(dir_outputs_trusted)).filter(F.col('path').contains('output')).withColumn("path", F.regexp_replace("path","dbfs:","")).collect()
#display(dir_reads)

# COMMAND ----------

for i in dir_reads:
  if i.path==dir_reads[0].path:
    paths=[i.path+'data_arquivo=*']
    df_final=spark.read.option('i.path',i.path).parquet(*paths).filter(F.col('TP_PESSOA')=="F")
  else:
    paths=[i.path+'data_arquivo=*']
    df_final=df_final.union(spark.read.option('i.path',i.path).parquet(*paths).filter(F.col('TP_PESSOA')=="F"))

#display(df_final)    

# COMMAND ----------

# DBTITLE 1,Selecionando as variáveis
df_final=df_final.select(['ID_PESSOA','DOCUMENT','NAME_PESSOA','DT_NASC','EMAIL','GENERO','EST_CIVIL','TP_PESSOASYS','DT_CADASTRO'])

# COMMAND ----------

# DBTITLE 1,Selecionando os CPFs que possuem data em alguns arquivos e são nulos em outros
df_final.createOrReplaceTempView("cred_model")
df1 = sql("select DOCUMENT, DT_NASC, count(1) as qtd from cred_model where trim(DT_NASC)='' group by DOCUMENT, DT_NASC")
df2 = sql("select DOCUMENT from cred_model where trim(DT_NASC)!=''")
df1=df1.join(df2, 'DOCUMENT', "inner").select('DOCUMENT').distinct().rdd.flatMap(lambda x: x).collect()
del df2

# COMMAND ----------

# DBTITLE 1,Formatando as colunas
df_final=df_final.withColumn('DT_NASC',F.regexp_replace(F.col("DT_NASC"), " ", "")).withColumn('DT_NASC',F.when(F.col('DT_NASC')=="",None).otherwise(F.col('DT_NASC'))).filter(~((F.col('DT_NASC').isNull()) & (F.col('DOCUMENT').isin(df1)))).withColumn('EST_CIVIL',F.regexp_replace(F.col("EST_CIVIL"), " ", "")).withColumn('EST_CIVIL',F.when(F.col('EST_CIVIL')=="",None).otherwise(F.col('EST_CIVIL'))).withColumn('DT_CADASTRO',F.to_date(F.col('DT_CADASTRO'),'ddMMyyyy')).sort('DT_CADASTRO',ascending=True)
df_final=df_final.withColumn('DT_NASC',F.when(F.col('DT_NASC').isNull(),None).otherwise(F.to_date(F.col('DT_NASC'),'ddMMyyyy'))).withColumn('ID_PESSOA',F.col('ID_PESSOA').cast('bigint')).withColumn('DOCUMENT',F.col('DOCUMENT').substr(5,11)).withColumn('GENERO',F.regexp_replace(F.col("GENERO"), " ", "")).withColumn('GENERO',F.when(F.col('GENERO')=="",'U').otherwise(F.col('GENERO')))

# COMMAND ----------

# DBTITLE 1,Agrupando por CPF e pelos mais recentes
df_final=df_final.groupby('DOCUMENT').agg(F.last("ID_PESSOA").alias('ID_PESSOA'),
                                         F.last("NAME_PESSOA").alias('NAME_PESSOA'),
                                         F.last("DT_NASC").alias('DT_NASC'),
                                         F.last("EMAIL").alias('EMAIL'),
                                         F.last("GENERO").alias('GENERO'),
                                         F.last("EST_CIVIL").alias('EST_CIVIL'),
                                         F.last("TP_PESSOASYS").alias('TP_PESSOASYS'),
                                         F.min("DT_CADASTRO").alias('DT_CADASTRO'))
#display(df_final)
###Formatando as datas###
#df_final=df_final.withColumn('DT_CADASTRO',to_date(F.col('DT_NASC'),'ddMMyyyy')).withColumn('DT_NASC',when(F.col('DT_NASC').isNUll(),None).otherwise(to_date(F.col('DT_NASC'),'ddMMyyyy')))
#display(df_final)

# COMMAND ----------

# DBTITLE 1,Salvando
 #df_final.write.mode('overwrite').option("sep",";").option("header",True).option('emptyValue','').parquet(dir_register+'/registration_model_variables')
df_final.write.option('header', 'True').option('delimiter',';').csv(dir_outputs_trusted+'/output_to_score_model')  

# COMMAND ----------

# DBTITLE 1,Excluindo as tabelas anteriores
for i in dir_reads:
  aux_remove=dbutils.fs.ls(i.path)
  for j in aux_remove:
    dbutils.fs.rm(j.path, True)
  dbutils.fs.rm(i.path,True)