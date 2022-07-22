# Databricks notebook source
import datetime
date = datetime.datetime.today()
date = str(date.year)+'-'+str(date.month).zfill(2)+'-'+str(date.day).zfill(2)
#date = '2022-06-21'
date

# COMMAND ----------

#Uma parte dessa base foi vendida, mas eles ainda continuam enviando pra gente. 
#Então tem alguns CPFs que vc vai ter de excluir na tabela final de escoragem

# COMMAND ----------

pathRemoveCpf = "/mnt/ml-prd/ml-data/propensaodeal/santander/cpfs_to_remove/Detalhes_bases_santander_2021_11_01.csv"

# COMMAND ----------

df_remove_cpf = spark.read.options(header='True', inferSchema='True', delimiter=';').csv(pathRemoveCpf)
display(df_remove_cpf)

# COMMAND ----------

from pyspark.sql.functions import lpad

df_remove_cpf = df_remove_cpf.withColumn("cpf", lpad(df_remove_cpf.document, 11, '0'))
display(df_remove_cpf)

# COMMAND ----------

df_remove_cpf.createOrReplaceTempView("remove_cpfs")

# COMMAND ----------

# MAGIC %sql
# MAGIC create or replace temporary view base_final
# MAGIC as
# MAGIC select lpad(a.ID_DEVEDOR, 11, '0') AS Document, 
# MAGIC        max(a.GH) as Score, 
# MAGIC        max(a.P_1) as ScoreValue, 
# MAGIC        avg(a.P_1) as ScoreAvg, 
# MAGIC        'qq_santander_propensity_v3' as provider,
# MAGIC        max(a.date_processed) as Date,
# MAGIC        max(a.date_processed) as CreatedAt
# MAGIC from default.model_qq_santander_homogeneous_group as a
# MAGIC GROUP BY lpad(a.ID_DEVEDOR, 11, '0')

# COMMAND ----------

# MAGIC %sql
# MAGIC create or replace temporary view base_final_dividas
# MAGIC as
# MAGIC select a.*
# MAGIC from base_final as a
# MAGIC left join remove_cpfs as b on b.cpf = a.Document
# MAGIC where cpf is null

# COMMAND ----------

df_final = spark.sql("""
select a.*
from base_final_dividas as a
left join remove_cpfs as b on b.cpf = a.Document
where cpf is null
""")

# COMMAND ----------

pathOutput = "/mnt/ml-prd/ml-data/propensaodeal/santander/output/"
print(pathOutput)
pathbackup = "/mnt/ml-prd/ml-data/propensaodeal/santander/backup/date="+date+"/"
print(pathbackup)

pathTrusted = "/mnt/ml-prd/ml-data/propensaodeal/santander/trusted/"
print(pathTrusted)

# COMMAND ----------

df_final.coalesce(1).write.option("sep",";").option("header","true").csv(pathOutput+'file/')

# COMMAND ----------

for p in dbutils.fs.ls(pathOutput+'file/'):
  if p[0].find(".csv") > 0:
    print(p[0])
    dbutils.fs.mv(p[0], "dbfs:/mnt/ml-prd/ml-data/propensaodeal/santander/output/")

# COMMAND ----------

dbutils.fs.rm("/mnt/ml-prd/ml-data/propensaodeal/santander/output/file/",True)

# COMMAND ----------

df_final.write.option("sep",";").option("header","true").csv(pathbackup)

# COMMAND ----------

df_final.coalesce(1).write.option("sep",";").option("header","true").csv(pathbackup+'file/')

# COMMAND ----------

for p in dbutils.fs.ls(pathbackup+'file/'):
  if p[0].find(".csv") > 0:
    print(p[0])
    dbutils.fs.mv(p[0], "dbfs:/mnt/ml-prd/ml-data/propensaodeal/santander/backup/date="+date+"/")

# COMMAND ----------

dbutils.fs.rm("/mnt/ml-prd/ml-data/propensaodeal/santander/backup/date="+date+"/file/",True)

# COMMAND ----------

# DBTITLE 1,Pagamentos
# MAGIC %sql
# MAGIC create or replace temporary view base_final
# MAGIC as
# MAGIC select lpad(a.ID_DEVEDOR, 11, '0') AS Document, 
# MAGIC        max(a.GH) as Score, 
# MAGIC        max(a.P_1) as ScoreValue, 
# MAGIC        avg(a.P_1) as ScoreAvg, 
# MAGIC        'qq_santander_propensity_payments_v1' as provider,
# MAGIC        max(a.date_processed) as Date,
# MAGIC        max(a.date_processed) as CreatedAt
# MAGIC from default.model_qq_santander_payments_homogeneous_group as a
# MAGIC GROUP BY lpad(a.ID_DEVEDOR, 11, '0')

# COMMAND ----------

# MAGIC %sql
# MAGIC create or replace temporary view base_final_pagamentos
# MAGIC as
# MAGIC select a.*
# MAGIC from base_final as a
# MAGIC left join remove_cpfs as b on b.cpf = a.Document
# MAGIC where cpf is null

# COMMAND ----------

#df_final_p = spark.sql("""
#select a.*
#from base_final_pagamentos as a
#left join remove_cpfs as b on b.cpf = a.Document
#where cpf is null
#""")

# COMMAND ----------

#df_final_p.show(5,False)

# COMMAND ----------

#df_final_p.coalesce(1).write.option("sep",";").option("header","true").csv(pathOutput+'file/')
#
#for p in dbutils.fs.ls(pathOutput+'file/'):
#  if p[0].find(".csv") > 0:
#    print(p[0])
#    dbutils.fs.mv(p[0], "dbfs:/mnt/ml-prd/ml-data/propensaodeal/santander/output/")
#    
#dbutils.fs.rm("/mnt/ml-prd/ml-data/propensaodeal/santander/output/file/",True)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from base_final_pagamentos

# COMMAND ----------

# DBTITLE 1,Blend
df_blend = spark.sql("""
select a.Document, a.ScoreValue as P_1_Acordo, b.ScoreValue as P_1_Pgto 
from base_final_dividas a left join base_final_pagamentos b
on a.Document = b.Document
""")

display(df_blend)

# COMMAND ----------

df_blend.coalesce(1).write.option("sep",";").option("header","true").csv(pathTrusted+'file/')

for p in dbutils.fs.ls(pathTrusted+'file/'):
  if p[0].find(".csv") > 0:
    print(p[0])
    dbutils.fs.cp(p[0], 'dbfs:'+pathTrusted+'/santander_blend_model.csv')
    
dbutils.fs.rm(pathTrusted+'file/',True)

# COMMAND ----------

