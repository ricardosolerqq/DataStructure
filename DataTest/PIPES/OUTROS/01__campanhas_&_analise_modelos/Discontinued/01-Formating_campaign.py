# Databricks notebook source
# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

# DBTITLE 1,Selecionando os diretórios
mount_blob_storage_key(dbutils,'qqdatastoragemain','qq-data-studies','/mnt/qq-data-studies')
dir_unprocessed='/dbfs/mnt/qq-data-studiesModels_Campaign/unprocessed/'
dir_processed='/dbfs/mnt/qq-data-studies/Models_Campaign/processed/'
dir_aux_tables='/dbfs/mnt/qq-data-studies/Models_Campaign/auxiliary_folder_of_table/'

# COMMAND ----------

# DBTITLE 1,Carregando os credores ativos
col_creditor=getPyMongoCollection('col_creditor')
query_credores=[
	{
		"$match" : {
			"active" : True
		}
	},
	{
		"$project" : {
			"_id" : 1
		}
	}
]

Credores=pd.DataFrame(list(col_creditor.aggregate(pipeline=query_credores,allowDiskUse=True)))['_id'].to_list()
del query_credores,col_creditor
Credores