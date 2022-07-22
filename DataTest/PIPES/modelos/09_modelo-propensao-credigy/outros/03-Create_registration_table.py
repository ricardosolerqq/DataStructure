# Databricks notebook source
# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

# DBTITLE 1,Ajustando diretórios
mount_blob_storage_oauth(dbutils,'qqprd','ml-prd',"/mnt/ml-prd")
#dir_trusted_read='/dbfs/mnt/ml-prd/ml-data/propensaodeal/credigy/trusted'
dir_trusted='/mnt/ml-prd/ml-data/propensaodeal/credigy/trusted'
dir_register='/mnt/ml-prd/ml-data/propensaodeal/credigy/registration_table'

# COMMAND ----------

#from pyspark.sql.types import *
#df_reg_01=[]
#schema = StructType([ \
 #   StructField("TP_ARQ",StringType(),True), \
  #  StructField("TP_REG",StringType(),True), \
   # StructField("ID_PESSOA",IntegerType(),True), \
   # StructField("DOCUMENT",IntegerType(),True), \
   # StructField("NUM_RG",StringType(),True), \
   # StructField("DS_ORGEXPRG",StringType(),True), \
   # StructField("NAME_PESSOA",StringType(),True),  \
   # StructField("DT_NASC",StringType(),True), \
   # StructField("EMAIL",StringType(),True), \
   # StructField("GENERO",StringType(),True), \
   # StructField("EST_CIVIL",StringType(),True), \
   # StructField("NACIONALIDADE",StringType(),True), \
   # StructField("DT_CADASTRO",StringType(),True), \
   # StructField("TP_PESSOA",StringType(),True), \
   # StructField("TP_PESSOASYS",StringType(),True), \
   # StructField("FUT_IMPL",StringType(),True), \
   # StructField("CD_SEQ",StringType(),True), \
   # StructField("data_arquivo",DateType(),True), \
  #])

# COMMAND ----------

# DBTITLE 1,Registro 01
from pyspark.sql.functions import *
from datetime import datetime
basePath_21=dir_trusted+'/tabelas_tipo_21_registro_01/'
basePath_01=dir_trusted+'/tabelas_tipo_01_registro_01/'
paths=[basePath_21+'data_arquivo=*']
paths_01=[basePath_01+'data_arquivo=*']
df_final=spark.read.option('basePath_01',basePath_01).parquet(*paths_01).filter(F.col('TP_PESSOA')=="F")
df_21=spark.read.option('basePath_21',basePath_21).parquet(*paths).filter(F.col('TP_PESSOA')=="F")
df_final=df_final.union(df_21)

del df_21
#df_21=df_21.select(['ID_PESSOA','DOCUMENT','NAME_PESSOA','DT_NASC','GENERO','EST_CIVIL','TP_PESSOASYS','data_arquivo'])
###Arumando a formatação###
#df_21=df_21.withColumn('ID_PESSOA',F.col('ID_PESSOA').cast('double')).withColumn('DOCUMENT',F.col('DOCUMENT').substr(5,11)).withColumn('DT_NASC',when(F.col('DT_NASC')=="",None).otherwise(to_date(F.col('DT_NASC'),'ddMMyyyy')))
display(df_final)

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

# DBTITLE 1,Retirando esses CPFs
df_final= df_final.filter(~((F.col('DT_NASC').isNull()) & (F.col('DOCUMENT').isin(df1))))

# COMMAND ----------

# DBTITLE 1,Formatando as colunas
df_final=df_final.withColumn('DT_NASC',regexp_replace(F.col("DT_NASC"), " ", "")).withColumn('DT_NASC',when(F.col('DT_NASC')=="",None).otherwise(F.col('DT_NASC'))).filter(~((F.col('DT_NASC').isNull()) & (F.col('DOCUMENT').isin(df1)))).withColumn('EST_CIVIL',regexp_replace(F.col("EST_CIVIL"), " ", "")).withColumn('EST_CIVIL',when(F.col('EST_CIVIL')=="",None).otherwise(F.col('EST_CIVIL'))).withColumn('DT_CADASTRO',to_date(F.col('DT_CADASTRO'),'ddMMyyyy')).sort('DT_CADASTRO',ascending=True)
df_final=df_final.withColumn('DT_NASC',when(F.col('DT_NASC').isNull(),None).otherwise(to_date(F.col('DT_NASC'),'ddMMyyyy'))).withColumn('ID_PESSOA',F.col('ID_PESSOA').cast('bigint')).withColumn('DOCUMENT',F.col('DOCUMENT').substr(5,11))

# COMMAND ----------

# DBTITLE 1,Localizando os outputs
from pyspark.sql.types import *

col_person =getPyMongoCollection('col_person')
data_min=df_final.agg(F.min('DT_CADASTRO').alias('DT_CADASTRO')).collect()
data_min=datetime.strptime(datetime.strftime(data_min[0].DT_CADASTRO,"%Y-%m-%d")+' 03:00:00','%Y-%m-%d %H:%M:%S')

query_output=[
	{
		"$match" : {
			"deals" : {
				"$elemMatch" : {
					"creditor" : "credigy",
					"createdAt" : {"$gte" : data_min},
					"status" : {"$ne" : "error"}
				}
			}
		}
	},
	{
		"$addFields" : {
	           "deals" : {
	            "$filter" : {
	                "input" : {"$map" : {
	                                    "input" : "$deals",
	                                    "as" : "i",
	                                    "in" : {
	                                        "creditor" : "$$i.creditor",
	                                        "status" : "$$i.status",
	                                        "createdAt" : "$$i.createdAt"
	                                    }

	                                }
	                        },
	                "cond": {
	                        "$and" : [
	                            {"$gte" : ["$$this.createdAt",data_min]},
	                            {"$not" : {"$eq" : ["$$this.status","error"]}},
	                            {"$eq" : ["$$this.creditor","credigy"]}
	                        ]
	                    }
	            }

			}
		}
	},
	{
		"$project" : {
			"_id" : 0,
			"DOCUMENT" : "$document",
			"acordo" : "True",
			"data_max" : {"$max" : "$deals.createdAt"}
		}
	}
  ]
Tab_outputs = spark.sparkContext.parallelize(list(col_person.aggregate(pipeline=query_output,allowDiskUse=True))).toDF().withColumn('acordo',F.col('acordo').cast(BooleanType())).withColumn('data_max',to_date(F.col('data_max')))
Tab_outputs.show()

# COMMAND ----------

# DBTITLE 1,Concatenando
df_final=df_final.join(Tab_outputs,'DOCUMENT','left').withColumn('acordo',when(F.col('acordo').isNull(),False).otherwise(F.col('acordo'))).withColumn('data_max',when(F.col('data_max').isNull(),datetime.strptime('1900-01-01',"%Y-%m-%d")).otherwise(F.col('data_max'))).withColumn('data_max',to_date(F.col('data_max')))

# COMMAND ----------

# DBTITLE 1,Corrigindo os acordos feitos antes da data de cadastro
df_final=df_final.withColumn('acordo',when(F.col('data_max')<F.col('DT_CADASTRO'),False).otherwise(F.col('acordo')))

# COMMAND ----------

# DBTITLE 1,Agrupando por CPF e pelos mais recentes
df_final=df_final.groupby('DOCUMENT').agg(F.last("ID_PESSOA").alias('ID_PESSOA'),
                                         F.last("NAME_PESSOA").alias('NAME_PESSOA'),
                                         F.last("DT_NASC").alias('DT_NASC'),
                                         F.last("EMAIL").alias('EMAIL'),
                                         F.last("GENERO").alias('GENERO'),
                                         F.last("EST_CIVIL").alias('EST_CIVIL'),
                                         F.last("TP_PESSOASYS").alias('TP_PESSOASYS'),
                                         F.min("DT_CADASTRO").alias('DT_CADASTRO'),
                                         F.max('acordo').alias('acordo'))
display(df_final)
###Formatando as datas###
#df_final=df_final.withColumn('DT_CADASTRO',to_date(F.col('DT_NASC'),'ddMMyyyy')).withColumn('DT_NASC',when(F.col('DT_NASC').isNUll(),None).otherwise(to_date(F.col('DT_NASC'),'ddMMyyyy')))
#display(df_final)

# COMMAND ----------

df_final.groupBy('acordo').count().show()

# COMMAND ----------

# DBTITLE 1,Salvando
 #df_final.write.mode('overwrite').option("sep",";").option("header",True).option('emptyValue','').parquet(dir_register+'/registration_model_variables')
df_final.coalesce(1).write.option('header', 'True').option('delimiter',';').csv(dir_register+'/sample_test')  