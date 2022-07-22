# Databricks notebook source
import time
time.sleep(300)

# COMMAND ----------

# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

file_names = get_creditor_etl_file_dates('santander')
a_excluir = []
for file in file_names:
  if "OY_QUEROQUITAR_D" not in file:
    a_excluir.append(file)
for file in a_excluir:
  del file_names[file]
  
file_dates = []
for file in file_names:
  file_dates.append(file_names[file])
max_file_date = max(file_dates)

for file in file_names:
  valor = file_names[file]
  if valor == max_file_date:
    file_name = file
    break
file_name


# COMMAND ----------

# MAGIC %run "/ml-prd/propensao-deal/santander/training/1-func-common"

# COMMAND ----------

mount_path_source_etlsantander = mount_path_source + "/etl/santander/processed"

# COMMAND ----------

file_name

# COMMAND ----------

'''
mount_path_source = "/mnt/qq-integrator"
mount_path_load = "/mnt/etlsantander"

Verificar se existe os arquivos e pasta input 
'''
from pyspark.sql.functions import *
fileStartNameSource = file_name
fileParquetLoad = mount_path_load + "/etl_santander_locs_campaign"

#listar e filtrar os arquivos na pasta input datalake
fileInfo = dbutils.fs.ls(mount_path_source_etlsantander)
dfFile = spark.createDataFrame(fileInfo)
dfFile=dfFile.filter(dfFile["name"].contains(file_name))

dfFile = dfFile.withColumn("path", regexp_replace("path","dbfs:",""))
dfName = dfFile.select(col("path"),col("name")).collect()

# COMMAND ----------

from pyspark.sql.types import *
date_schema = getStructEtlField("date_schema")
result_schema = getStructEtlField("result_schema")
temp_schema = getStructEtlField("temp_schema")

# COMMAND ----------

dfResult = spark.createDataFrame(spark.sparkContext.emptyRDD(),result_schema)
dfResult = dfResult.drop("Tipo_Registro")

# COMMAND ----------

for fileName in dfName:
  print(fileName.name)
  dfOne = getDfEtlSantander(fileName.path, fileName.name,temp_schema)
  dfResult = dfResult.union(dfOne)  
dfResult.dropDuplicates()
print(dfResult.count())
dfResult.write.mode("overwrite").partitionBy("Data_Arquivo").parquet(fileParquetLoad)