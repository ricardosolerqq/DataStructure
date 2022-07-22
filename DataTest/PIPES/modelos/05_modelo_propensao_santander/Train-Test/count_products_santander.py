# Databricks notebook source
# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

mount_blob_storage_oauth(dbutils, "saqueroquitar", "trusted", "/mnt/trusted", key="kvdesQQ")

# COMMAND ----------

#filePath = "/mnt/trusted/collection/full/"
df_debts = spark.read.format("delta").load("/mnt/trusted/collection/full/col_person_debts")
df_debts.createOrReplaceTempView("col_person_debts")
df_debts.printSchema()

# COMMAND ----------

from pyspark.sql.functions import *
#transforma a coluna timestamp em date
df_debts = df_debts.withColumn("DataVencimento", F.to_date(F.col("dueDate")))
##Coluna com a data de hoje##
df_debts=df_debts.withColumn("Data_atual",current_date())
df_debts = df_debts.filter((F.col("status") == "active")&(F.col("creditor") == "santander")&(F.col("documentType")=="cpf"))
df_debts_prod=df_debts.groupby('document','contract').agg(F.first('description').alias('Produto'))
df_debts_prod=df_debts_prod.groupby('Produto').count().orderBy('count',ascending=False)
df_debts_prod=df_debts_prod.filter(~(F.col('Produto')==''))
display(df_debts_prod)