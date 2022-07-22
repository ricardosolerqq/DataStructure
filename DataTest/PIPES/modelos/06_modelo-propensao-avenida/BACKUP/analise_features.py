# Databricks notebook source
# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

import os


# COMMAND ----------

blob_account_source_ml = "qqdatastoragemain"
blob_container_source_ml = "ml-prd"

mount_blob_storage_key(dbutils,blob_account_source_ml,blob_account_source_ml,'/mnt/ml-prd')

parquetfilepath = '/mnt/ml-prd/ml-data/propensaodeal/credz/pre-trusted'
parquetfilepath_dbfs = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/credz/pre-trusted'
list_parquetfilepath = os.listdir(parquetfilepath_dbfs)
dbutils.widgets.combobox('ARQUIVO ESCOLHIDO', max(list_parquetfilepath), list_parquetfilepath)
folder_escolhido = dbutils.widgets.get('ARQUIVO ESCOLHIDO')
parquetfilepath = '/mnt/ml-prd/ml-data/propensaodeal/credz/pre-trusted'+'/'+folder_escolhido+'/'
parquetfilepath

# COMMAND ----------

# DBTITLE 1,construindo df's
clientes_df = spark.read.parquet(parquetfilepath+'clientes.PARQUET').alias('clientes')
detalhes_clientes_df = spark.read.parquet(parquetfilepath+'detalhes_clientes.PARQUET').alias('detalhes_clientes')
telefones_df = spark.read.parquet(parquetfilepath+'telefones.PARQUET').alias('telefones')
enderecos_df = spark.read.parquet(parquetfilepath+'enderecos.PARQUET').alias('enderecos')
detalhes_dividas_df = spark.read.parquet(parquetfilepath+'detalhes_dividas.PARQUET').alias('detalhes_dividas')
contratos_df = spark.read.parquet(parquetfilepath+'contratos.PARQUET').alias('contratos')
detalhes_contratos_df = spark.read.parquet(parquetfilepath+'detalhes_contratos.PARQUET').alias('detalhes_contratos')
dividas_df = spark.read.parquet(parquetfilepath+'dividas.PARQUET').alias('dividas')
e_mails_dos_clientes_df = spark.read.parquet(parquetfilepath+'e_mails_dos_clientes.PARQUET').alias('clientes')

# COMMAND ----------

'2021-06-18'

detalhes_clientes_features_18062021 = detalhes_clientes_df.select('NOME_DETALHE').dropDuplicates().rdd.map(lambda row : row[0]).collect()
detalhes_dividas_features_18062021 = detalhes_dividas_df.select('NOME_DETALHE').dropDuplicates().rdd.map(lambda row : row[0]).collect()
detalhes_contratos_features_18062021 = detalhes_contratos_df.select('NOME_DETALHE').dropDuplicates().rdd.map(lambda row : row[0]).collect()

# COMMAND ----------

'2021-06-16'

detalhes_clientes_features_16062021 = detalhes_clientes_df.select('NOME_DETALHE').dropDuplicates().rdd.map(lambda row : row[0]).collect()
detalhes_dividas_features_16062021 = detalhes_dividas_df.select('NOME_DETALHE').dropDuplicates().rdd.map(lambda row : row[0]).collect()
detalhes_contratos_features_16062021 = detalhes_contratos_df.select('NOME_DETALHE').dropDuplicates().rdd.map(lambda row : row[0]).collect()

# COMMAND ----------

'2021-04-17'

detalhes_clientes_features_17042021 = detalhes_clientes_df.select('NOME_DETALHE').dropDuplicates().rdd.map(lambda row : row[0]).collect()
detalhes_dividas_features_17042021 = detalhes_dividas_df.select('NOME_DETALHE').dropDuplicates().rdd.map(lambda row : row[0]).collect()
detalhes_contratos_features_17042021 = detalhes_contratos_df.select('NOME_DETALHE').dropDuplicates().rdd.map(lambda row : row[0]).collect()

# COMMAND ----------

'2021-03-12'

detalhes_clientes_features_12032021 = detalhes_clientes_df.select('NOME_DETALHE').dropDuplicates().rdd.map(lambda row : row[0]).collect()
detalhes_dividas_features_12032021 = detalhes_dividas_df.select('NOME_DETALHE').dropDuplicates().rdd.map(lambda row : row[0]).collect()
detalhes_contratos_features_12032021 = detalhes_contratos_df.select('NOME_DETALHE').dropDuplicates().rdd.map(lambda row : row[0]).collect()

# COMMAND ----------

print (len(detalhes_clientes_features_18062021), len(detalhes_clientes_features_16062021), len(detalhes_clientes_features_17042021), len(detalhes_clientes_features_12032021))
print (len(detalhes_dividas_features_18062021), len(detalhes_dividas_features_16062021),len(detalhes_dividas_features_17042021),len(detalhes_dividas_features_12032021))
print (len(detalhes_contratos_features_18062021),len(detalhes_contratos_features_16062021), len(detalhes_contratos_features_17042021), len(detalhes_contratos_features_12032021))


# COMMAND ----------

# DBTITLE 1,detalhes em comum
detalhes_clientes = [detalhes_clientes_features_18062021,detalhes_clientes_features_16062021, detalhes_clientes_features_17042021,detalhes_clientes_features_12032021]
detalhes_dividas = [detalhes_dividas_features_18062021, detalhes_dividas_features_16062021, detalhes_dividas_features_17042021, detalhes_dividas_features_12032021]
detalhes_contratos = [detalhes_contratos_features_18062021, detalhes_contratos_features_16062021, detalhes_contratos_features_17042021, detalhes_contratos_features_12032021]

detalhes_clientes_em_comum = []
detalhes_dividas_em_comum = []
detalhes_contratos_em_comum = []

for i in range (0, 4):
  for d in detalhes_clientes[i]:
    if d in detalhes_clientes[0] and d in detalhes_clientes[1] and d in detalhes_clientes[2] and d in detalhes_clientes[3]:
      if d not in detalhes_clientes_em_comum:
        detalhes_clientes_em_comum.append(d)

for i in range (0, 4):
  for d in detalhes_dividas[i]:
    if d in detalhes_dividas[0] and d in detalhes_dividas[1] and d in detalhes_dividas[2] and d in detalhes_dividas[3]:
      if d not in detalhes_dividas_em_comum:
        detalhes_dividas_em_comum.append(d)
          
for i in range (0, 4):
  for d in detalhes_contratos[i]:
    if d in detalhes_contratos[0] and d in detalhes_contratos[1] and d in detalhes_contratos[2] and d in detalhes_contratos[3]:
      if d not in detalhes_contratos_em_comum:
        detalhes_contratos_em_comum.append(d)


# COMMAND ----------

display(clientes_df)

# COMMAND ----------

detalhes_clientes_em_comum

# COMMAND ----------

detalhes_dividas_em_comum

# COMMAND ----------

detalhes_contratos_em_comum

# COMMAND ----------

detalhes_dividas_em_comum

# COMMAND ----------

for detalhe in detalhes_dividas_em_comum:
  print ('weee')
  display(detalhes_dividas_df.filter(F.col('NOME_DETALHE')==detalhe).groupby('NOME_DETALHE').agg({'VALOR':'first'}))

# COMMAND ----------

display(detalhes_contratos_df.groupby('NOME_DETALHE').agg(
{'VALOR':'first'}))