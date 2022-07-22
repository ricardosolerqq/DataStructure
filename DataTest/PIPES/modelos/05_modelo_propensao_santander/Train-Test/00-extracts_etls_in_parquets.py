# Databricks notebook source
# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

# MAGIC %run "/ml-prd/propensao-deal/santander/training/1-func-common"

# COMMAND ----------

mount_path_source_etlsantander = mount_path_source + "/etl/santander/processed"

# COMMAND ----------

'''
mount_path_source = "/mnt/qq-integrator"
mount_path_load = "/mnt/etlsantander"

Verificar se existe os arquivos e pasta input 
'''
from pyspark.sql.functions import *
fileStartNameSource = "OY_QUEROQUITAR_D20211029CRIP"
fileParquetLoad = mount_path_load + "/etl_santander_locs_campaign"

#listar e filtrar os arquivos na pasta input datalake
fileInfo = dbutils.fs.ls(mount_path_source_etlsantander)
dfFile = spark.createDataFrame(fileInfo)
dfFile=dfFile.filter(dfFile["name"].contains("OY_QUEROQUITAR_D20211029CRIP"))
#dfFile = dfFile.filter(dfFile["name"].isin(fileStartNameSource))
#dfFile = dfFile.filter(dfFile["name"].contains("OY_QUEROQUITAR_ACOMP_D20211029CRIP"))
#dfFileAbril = dfFile.filter(dfFile["name"].contains("OY_QUEROQUITAR_D202104"))
dfFile = dfFile.withColumn("path", regexp_replace("path","dbfs:",""))
dfName = dfFile.select(col("path"),col("name")).collect()
#dfNameAbril = dfFileAbril.select(col("name")).collect()
display(dfFile)

# COMMAND ----------

fileParquetLoad

# COMMAND ----------

from pyspark.sql.types import *
date_schema = getStructEtlField("date_schema")
result_schema = getStructEtlField("result_schema")
temp_schema = getStructEtlField("temp_schema")

# COMMAND ----------

dfResult = spark.createDataFrame(spark.sparkContext.emptyRDD(),result_schema)
dfResult = dfResult.drop("Tipo_Registro")
#dfHeaderResult = spark.createDataFrame(spark.sparkContext.emptyRDD(),date_schema)

# COMMAND ----------

for fileName in dfName:
  print(fileName.name)
  dfOne = getDfEtlSantander(fileName.path, fileName.name,temp_schema)
  dfResult = dfResult.union(dfOne)  
dfResult.dropDuplicates()
dfResult.write.mode("overwrite").partitionBy("Data_Arquivo").parquet(fileParquetLoad)