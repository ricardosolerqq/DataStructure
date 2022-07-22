# Databricks notebook source
from pyspark.sql.functions import *

# COMMAND ----------

import datetime
date = datetime.datetime.today()
date = str(date.year)+'-'+str(date.month).zfill(2)+'-'+str(date.day).zfill(2)
#date = '2022-06-21'
date

# COMMAND ----------

def get_csv_files(directory_path):
  """recursively list path of all csv files in path directory """
  csv_files = []
  files_to_treat = dbutils.fs.ls(directory_path.replace('/dbfs',''))
  while files_to_treat:
    path = files_to_treat.pop(0).path
    if path.endswith('/'):
      files_to_treat += dbutils.fs.ls(path)
    elif path.endswith('.csv'):
      csv_files.append(path)
  return csv_files

# COMMAND ----------

param_chave = "ID_DEVEDOR"
param_caminho_base = "{0}{1}/".format("/dbfs/mnt/etlsantander/Base_UFPEL/Join_Skips_Divids_Contact/repart/Data_Arquivo=",date) 
param_caminho_pickle_model = "/dbfs/mnt/ml-prd/ml-data/propensaodeal/santander/pickle_model/model_fit_santander_v2.sav"
param_separador = ";"
param_decimal = "."

source = "{0}{1}/".format("/dbfs/mnt/etlsantander/Base_UFPEL/Join_Skips_Divids_Contact/repart/Data_Arquivo=",date)
print(source)
sourceOK = "{0}{1}/processed/".format("/dbfs/mnt/etlsantander/Base_UFPEL/Join_Skips_Divids_Contact/repart/Data_Arquivo=",date) 
print(sourceOK)
sink = "{0}{1}/score_model/".format("/mnt/etlsantander/Base_UFPEL/Dados=",date)
print(sink)
sink_homogeneous_group_model = "{0}{1}/homogeneous_group_model/".format("/mnt/etlsantander/Base_UFPEL/Dados=",date)
print(sink_homogeneous_group_model)
cont = 1
notebook_model = "./05_model_QQ_santander"


# COMMAND ----------

for part in get_csv_files(source):
  
  param_N_Base = part.replace("{0}{1}/".format("dbfs:/mnt/etlsantander/Base_UFPEL/Join_Skips_Divids_Contact/repart/Data_Arquivo=",date), "")
  pathSink = sink+str(cont)+'.csv'
  pathSinkHomomgeneous = sink_homogeneous_group_model+str(cont)+'.csv'
  
  print('Start Scoring model')
  print('File:' + param_N_Base)
  dbutils.notebook.run(notebook_model, 
                       600, 
                       {
                         "chave": param_chave, 
                         "caminho_base": param_caminho_base,
                         "decimal_":param_decimal,
                         "separador_":param_separador,
                         "caminho_pickle_model":param_caminho_pickle_model,
                         "N_Base":param_N_Base,
                         "sink":pathSink,
                         "sinkHomomgeneous":pathSinkHomomgeneous
                       })  
  print('End Scoring model')
  
  dbutils.fs.mv(source.replace('/dbfs','')+param_N_Base,sourceOK.replace('/dbfs','')+param_N_Base)
  
  cont = cont+1

# COMMAND ----------

param_caminho_pickle_model_p = "/dbfs/mnt/ml-prd/ml-data/propensaodeal/santander/pickle_model/model_fit_santander_payments.sav"

sink_p = "{0}{1}/payments/score_model/".format("/mnt/etlsantander/Base_UFPEL/Dados=",date)
print(sink_p)
sink_homogeneous_group_model_p = "{0}{1}/payments/homogeneous_group_model/".format("/mnt/etlsantander/Base_UFPEL/Dados=",date)
print(sink_homogeneous_group_model_p)
cont = 1
notebook_model = "./05_model_QQ_santander_payments"

# COMMAND ----------

for part in get_csv_files(source):
  
  param_N_Base = part.replace("{0}{1}/".format("dbfs:/mnt/etlsantander/Base_UFPEL/Join_Skips_Divids_Contact/repart/Data_Arquivo=",date), "")
  pathSink = sink_p+str(cont)+'.csv'
  pathSinkHomomgeneous = sink_homogeneous_group_model_p+str(cont)+'.csv'
  
  print('Start Scoring model')
  print('File:' + param_N_Base)
  dbutils.notebook.run(notebook_model, 
                       600, 
                       {
                         "chave": param_chave, 
                         "caminho_base": param_caminho_base,
                         "decimal_":param_decimal,
                         "separador_":param_separador,
                         "caminho_pickle_model":param_caminho_pickle_model_p,
                         "N_Base":param_N_Base,
                         "sink":pathSink,
                         "sinkHomomgeneous":pathSinkHomomgeneous
                       })  
  print('End Scoring model')
  
  dbutils.fs.mv(source.replace('/dbfs','')+param_N_Base,sourceOK.replace('/dbfs','')+param_N_Base)
  
  cont = cont+1

# COMMAND ----------

df_score_model = spark.read.option("delimiter", ";").option("header", True).csv("{0}{1}{2}".format("/mnt/etlsantander/Base_UFPEL/Dados=",date,"/score_model/*/*.csv"))

# COMMAND ----------

df_score_model = df_score_model.withColumn("date_processed",lit(date))
display(df_score_model)

# COMMAND ----------

spark.sql("drop table if exists default.model_qq_santander_logistic_regression")

# COMMAND ----------

df_score_model.write.saveAsTable("default.model_qq_santander_logistic_regression")

# COMMAND ----------

df_homogeneous_group_model = spark.read.option("delimiter", ";").option("header", True).csv("{0}{1}{2}".format("/mnt/etlsantander/Base_UFPEL/Dados=",date,"/homogeneous_group_model/*/*.csv"))

# COMMAND ----------

df_homogeneous_group_model = df_homogeneous_group_model.withColumn("date_processed",lit(date))
display(df_homogeneous_group_model)

# COMMAND ----------

spark.sql("drop table if exists default.model_qq_santander_homogeneous_group")

# COMMAND ----------

df_homogeneous_group_model.write.saveAsTable("default.model_qq_santander_homogeneous_group")

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select *
# MAGIC from default.model_qq_santander_logistic_regression

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select *
# MAGIC from default.model_qq_santander_homogeneous_group

# COMMAND ----------

# DBTITLE 1,Pagamentos
df_score_model_p = spark.read.option("delimiter", ";").option("header", True).csv("{0}{1}{2}".format("/mnt/etlsantander/Base_UFPEL/Dados=",date,"/payments/score_model/*/*.csv"))
df_score_model_p = df_score_model_p.withColumn("date_processed",lit(date))
display(df_score_model_p)

# COMMAND ----------

spark.sql("drop table if exists default.model_qq_santander_payments_logistic_regression")

# COMMAND ----------

df_score_model_p.write.saveAsTable("default.model_qq_santander_payments_logistic_regression")

# COMMAND ----------

df_homogeneous_group_model = spark.read.option("delimiter", ";").option("header", True).csv("{0}{1}{2}".format("/mnt/etlsantander/Base_UFPEL/Dados=",date,"/homogeneous_group_model/*/*.csv"))
df_homogeneous_group_model.count()

# COMMAND ----------

df_homogeneous_group_model_p = spark.read.option("delimiter", ";").option("header", True).csv("{0}{1}{2}".format("/mnt/etlsantander/Base_UFPEL/Dados=",date,"/payments/homogeneous_group_model/*/*.csv"))
df_homogeneous_group_model_p = df_homogeneous_group_model_p.withColumn("date_processed",lit(date))
display(df_homogeneous_group_model_p)

# COMMAND ----------

spark.sql("drop table if exists default.model_qq_santander_payments_homogeneous_group")

# COMMAND ----------

df_homogeneous_group_model_p.write.saveAsTable("default.model_qq_santander_payments_homogeneous_group")

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select *
# MAGIC from default.model_qq_santander_payments_logistic_regression

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select *
# MAGIC from default.model_qq_santander_payments_homogeneous_group

# COMMAND ----------

