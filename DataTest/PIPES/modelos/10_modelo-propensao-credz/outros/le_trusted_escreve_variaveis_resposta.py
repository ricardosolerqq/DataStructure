# Databricks notebook source
# Notebook não encontrado
'''%run "/Users/diego.cohen@queroquitar.com.br/init_called_lake_variables"''' 

# COMMAND ----------

# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

import os
import datetime

# COMMAND ----------

blob_account_source_ml = "qqdatastoragemain"
blob_container_source_ml = "ml-prd"

blob_account_source_lake = "saqueroquitar"
blob_container_source_lake = "trusted"

mount_blob_storage_key(dbutils,blob_account_source_ml,blob_container_source_ml,'/mnt/ml-prd')
mount_blob_storage_key(dbutils,blob_account_source_lake,blob_container_source_lake,'/mnt/ml-prd')


readpath_trusted = '/mnt/ml-prd/ml-data/propensaodeal/credz/trusted'
readpath_trusted_dbfs = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/credz/trusted'

writepath_trusted_variavel_resposta = '/mnt/ml-prd/ml-data/propensaodeal/credz/trusted/variavel_resposta'

list_readpath = os.listdir(readpath_trusted_dbfs)

spark.conf.set('spark.sql.sources.partitionOverwriteMode', 'dynamic')

# COMMAND ----------

import os
try:
  dbutils.widgets.remove('ARQUIVO ESCOLHIDO')
except:
  pass

# COMMAND ----------

# DBTITLE 1,cria widget
dbutils.widgets.combobox('ARQUIVO ESCOLHIDO', max(list_readpath), list_readpath)

# COMMAND ----------

# DBTITLE 1,atribuindo escolha do widget nos arquivos lidos
data_escolhida = dbutils.widgets.get('ARQUIVO ESCOLHIDO')
parquetfilepath = '/mnt/ml-prd/ml-data/propensaodeal/credz/pre-trusted'+'/'+data_escolhida+'/'
print(parquetfilepath)

# COMMAND ----------

df_credz = spark.read.format('parquet').load(readpath_trusted+'/'+data_escolhida+'/'+'trustedFile_credz.PARQUET').alias('df_credz')

# COMMAND ----------

df.display()

# COMMAND ----------

df = df_credz.dropna(subset=['DOCUMENTO_PESSOA'])
df = df.withColumn('DOCUMENTO_PESSOA', F.lpad('DOCUMENTO_PESSOA',11,'0'))\
       .withColumn('ID_DIVIDA', F.lpad('ID_DIVIDA',19,'0'))

# COMMAND ----------

df_teste = variavel_resposta_acordo_atualizado('credz', datetime.datetime(2022,2,1), datetime.datetime(2022,3,2), varName='VARIAVEL_RESPOSTA')
#df = escreve_variavel_resposta_acordo(df, 'credz', datetime.datetime(2022,2,1), 'document', drop_null = True)

# COMMAND ----------

df_teste

# COMMAND ----------

df_teste_v2 = df_teste.drop(columns = ['chave','ID_acordo'])

# COMMAND ----------

from pyspark.sql import SparkSession
#Create PySpark SparkSession
spark = SparkSession.builder \
    .master("local[1]") \
    .appName("SparkByExamples.com") \
    .getOrCreate()
#Create PySpark DataFrame from Pandas
sparkDF=spark.createDataFrame(df_teste_v2) 
sparkDF.printSchema()
sparkDF.show()

# COMMAND ----------

df.display()

# COMMAND ----------

df.count()

# COMMAND ----------

sparkDF.filter(F.col('document') == '00146799810').display()

# COMMAND ----------

df.filter(F.col('DOCUMENTO_PESSOA') == '00146799810').display()

# COMMAND ----------

df_v2 = df.join(sparkDF, (F.col('ID_DIVIDA')==F.col('contract')) & (F.col('DOCUMENTO_PESSOA')==F.col('document')), how='left')

# COMMAND ----------

df_v2.display()

# COMMAND ----------

df_v2.groupBy('varName').count().show()

# COMMAND ----------

df_v2.coalesce(1).write.option('sep', ';').option('header', 'True').csv(os.path.join(writepath_trusted_variavel_resposta+'/'+data_escolhida,'trustedFile_credz_FULL_variavel_resposta'))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Criação do DataFrame por parte do Diego

# COMMAND ----------

# Na época em que o diego criou o DataFrame, ele não utilizava a função escreve_variavel_resposta_acordo.
# Dessa Forma era coletado os acordos do credor conforme as datas mínimas e máximas e posteriormente realizava um cruzamento entre as variáveis (id_deals, document, identificador) de deals_debts
# Pelo que foi avaliado, a única diferença no processo é a variável id_deals.
# Como id_deals é uma variável utilizada apenas para cruzamento entre offer_debts e deals, optamos por não utilizar no momento.

# COMMAND ----------

import datetime
data_minima = datetime.datetime.strptime(data_escolhida, '%Y-%m-%d')
data_maxima = data_minima+datetime.timedelta(days=30)

# COMMAND ----------

# DBTITLE 1,criando e filtrando df_lake_deals
df_lake_deals = get_collection('col_person_deals')
df_lake_deals = df_lake_deals.filter(F.col('creditor')=='credz')
df_lake_deals = df_lake_deals.select('id_deals', 'document', 'createdAt')
df_lake_deals = convertTimeStampToDate(df_lake_deals)
df_lake_deals = df_lake_deals.filter(F.col('createdAt') >= data_minima)
df_lake_deals = df_lake_deals.filter(F.col('createdAt') <= data_maxima).alias('deals')

# COMMAND ----------

# DBTITLE 1,criando e filtrando df_lake_deals_debts
df_lake_deals_debts = spark.read.format('delta').load("/mnt/trusted/collection/full/col_person_deals_offer_debts").filter(F.col('creditor')=='credz')
df_lake_deals_debts = df_lake_deals_debts.select('id_deals', 'document', 'additionalInfo')
df_lake_deals_debts = convertTimeStampToDate(df_lake_deals_debts).alias('debts')
df_lake_deals_debts = unpack_dict_to_columns(df_lake_deals_debts).select('id_deals', 'document', 'Identificador')

# COMMAND ----------

df_lake = df_lake_deals.join(df_lake_deals_debts, (F.col('debts.document')==F.col('deals.document')) & (F.col('debts.id_deals')==F.col('deals.id_deals')), how='left')
df_lake = df_lake.drop(F.col('debts.document'))
df_lake = df_lake.drop(F.col('debts.id_deals'))
df_lake = df_lake.withColumn('document', F.col('document').cast(T.LongType()))
df_lake = df_lake.withColumn('Identificador', F.col('Identificador').cast(T.LongType()))
df_lake = df_lake.alias('df_lake')

# COMMAND ----------

df = df_credz.join(df_lake, (F.col('df_credz.DOCUMENTO_PESSOA') == F.col('df_lake.document')) \
                      & (F.col('df_credz.ID_DIVIDA') == F.col('df_lake.Identificador')), how='left')\
                      .drop(F.col('df_lake.document'))\
                      .drop(F.col('df_lake.createdAt'))\
                      .withColumn('Identificador', F.when(F.col('Identificador').isNull(), False).otherwise(True))\
                      .withColumnRenamed('Identificador', 'VARIAVEL_RESPOSTA')

# COMMAND ----------

df.write.mode('overwrite').parquet(writepath_trusted_variavel_resposta+'/'+data_escolhida+'/'+'trustedFile_credz_FULL_variavel_resposta'+'.PARQUET')

# COMMAND ----------

"""
writepath_trusted = '/mnt/ml-prd/ml-data/propensaodeal/credz/trusted'

df2 = df.limit(int(df.count()/10))
for sch in df2.dtypes:
  nome = sch[0]
  tipo = sch[1]
  if 'array' in tipo:
    df2 = df2.withColumn(nome, F.col(nome).cast(T.StringType()))
    

df2.write.csv(writepath_trusted+'/'+data_escolhida+'/'+'trustedFile_credz_SAMPLE_2'+'.csv', header=True)
"""

# COMMAND ----------

"""
df2 = df.select('DETALHES_CLIENTES_VENCIMENTO_FATURA')
df2 = df2.withColumn('DIA', F.substring(F.col('DETALHES_CLIENTES_VENCIMENTO_FATURA'),1,2))
df2 = df2.withColumn('MES', F.substring(F.col('DETALHES_CLIENTES_VENCIMENTO_FATURA'),3,2))
df2 = df2.withColumn('ANO', F.substring(F.col('DETALHES_CLIENTES_VENCIMENTO_FATURA'),5,4))
print (df2.count())
df3 = df2.filter(F.col('ANO')==2021)
df3 = df3.filter((F.col("MES").cast(T.IntegerType())>=4) & (F.col('DIA').cast(T.IntegerType())>17))
print (df3.count())
"""