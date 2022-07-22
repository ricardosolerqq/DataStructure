# Databricks notebook source
# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

import os
try:
  dbutils.widgets.remove('ARQUIVO_ESCOLHIDO')
except:
  pass

# COMMAND ----------

blob_account_source_ml = "qqdatastoragemain"
blob_container_source_ml = "ml-prd"

mount_blob_storage_key(dbutils,blob_account_source_ml,blob_account_source_ml,'/mnt/ml-prd')

readpath_stage = '/mnt/ml-prd/ml-data/propensaodeal/credz/stage'
readpath_stage_dbfs = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/credz/stage'
writepath_trusted = '/mnt/ml-prd/ml-data/propensaodeal/credz/trusted'
list_readpath = os.listdir(readpath_stage_dbfs)

spark.conf.set('spark.sql.sources.partitionOverwriteMode', 'dynamic')

# COMMAND ----------

# DBTITLE 1,cria widget
dbutils.widgets.combobox('ARQUIVO_ESCOLHIDO', max(list_readpath), list_readpath)
data_escolhida = dbutils.widgets.get('ARQUIVO_ESCOLHIDO')
parquetfilepath = '/mnt/ml-prd/ml-data/propensaodeal/credz/pre-trusted'+'/'+data_escolhida+'/'
print(parquetfilepath)

# COMMAND ----------

readpath_stage+'/'+data_escolhida+'/'+'geral_sem_detalhes.PARQUET'

# COMMAND ----------

# DBTITLE 1,lendo dataframes para join
df = spark.read.format('parquet').load(readpath_stage+'/'+data_escolhida+'/'+'geral_sem_detalhes.PARQUET').alias('df')
detalhes_clientes_df = spark.read.format('parquet').load(readpath_stage+'/'+data_escolhida+'/'+'detalhes_clientes.PARQUET').alias('detalhes_clientes')
detalhes_contratos_df = spark.read.format('parquet').load(readpath_stage+'/'+data_escolhida+'/'+'detalhes_contratos.PARQUET').alias('detalhes_contratos')
detalhes_dividas_df = spark.read.format('parquet').load(readpath_stage+'/'+data_escolhida+'/'+'detalhes_dividas.PARQUET').alias('detalhes_dividas')

# COMMAND ----------

# DBTITLE 1,join dataframes
df = df.join(detalhes_clientes_df, F.col('df.ID_PESSOA:ID_DIVIDA')== F.col('detalhes_clientes.ID_PESSOA:ID_DIVIDA')).drop(F.col('detalhes_clientes.ID_PESSOA:ID_DIVIDA')).alias('df')
df = df.join(detalhes_contratos_df, F.col('df.ID_PESSOA:ID_DIVIDA')== F.col('detalhes_contratos.ID_PESSOA:ID_DIVIDA')).drop(F.col('detalhes_contratos.ID_PESSOA:ID_DIVIDA')).alias('df')
df = df.join(detalhes_dividas_df, F.col('df.ID_PESSOA:ID_DIVIDA')== F.col('detalhes_dividas.ID_PESSOA:ID_DIVIDA')).drop(F.col('detalhes_dividas.ID_PESSOA:ID_DIVIDA')).alias('df')

# COMMAND ----------

# DBTITLE 1,transform datatypes
df = df.withColumn('DOCUMENTO_PESSOA', F.col('DOCUMENTO_PESSOA').cast(T.LongType()))
df = df.withColumn('ID_DIVIDA', F.col('ID_DIVIDA').cast(T.LongType()))
df = df.alias('df')

# COMMAND ----------

# DBTITLE 1,dummerizando tipo de email
df = df.withColumn('TIPO_EMAIL_0_INDEX', F.col('TIPO_EMAIL').getItem(0))
df = df.withColumn('TIPO_EMAIL_1_INDEX', F.col('TIPO_EMAIL').getItem(1))
df = df.withColumn('TIPO_EMAIL_2_INDEX', F.col('TIPO_EMAIL').getItem(2))
df = df.withColumn('TIPO_EMAIL_3_INDEX', F.col('TIPO_EMAIL').getItem(3))

df = df.withColumn('TIPO_EMAIL_0', (F.when(F.col('TIPO_EMAIL_0_INDEX')==0,True)\
                                .otherwise((F.when(F.col('TIPO_EMAIL_1_INDEX')==0, True)\
                                .otherwise((F.when(F.col('TIPO_EMAIL_2_INDEX')==0, True)\
                                .otherwise(F.when(F.col('TIPO_EMAIL_3_INDEX')==0, True)\
                                .otherwise(False))))))))

df = df.withColumn('TIPO_EMAIL_1', (F.when(F.col('TIPO_EMAIL_0_INDEX')==1,True)\
                                .otherwise((F.when(F.col('TIPO_EMAIL_1_INDEX')==1, True)\
                                .otherwise((F.when(F.col('TIPO_EMAIL_2_INDEX')==1, True)\
                                .otherwise(F.when(F.col('TIPO_EMAIL_3_INDEX')==1, True)\
                                .otherwise(False))))))))

df = df.withColumn('TIPO_EMAIL_2', (F.when(F.col('TIPO_EMAIL_0_INDEX')==2,True)\
                                .otherwise((F.when(F.col('TIPO_EMAIL_1_INDEX')==2, True)\
                                .otherwise((F.when(F.col('TIPO_EMAIL_2_INDEX')==2, True)\
                                .otherwise(F.when(F.col('TIPO_EMAIL_3_INDEX')==2, True)\
                                .otherwise(False))))))))

df = df.withColumn('TIPO_EMAIL_3', (F.when(F.col('TIPO_EMAIL_0_INDEX')==3,True)\
                                .otherwise((F.when(F.col('TIPO_EMAIL_1_INDEX')==3, True)\
                                .otherwise((F.when(F.col('TIPO_EMAIL_2_INDEX')==3, True)\
                                .otherwise(F.when(F.col('TIPO_EMAIL_3_INDEX')==3, True)\
                                .otherwise(False))))))))

df = df.drop('TIPO_EMAIL_0_INDEX')
df = df.drop('TIPO_EMAIL_1_INDEX')
df = df.drop('TIPO_EMAIL_2_INDEX')
df = df.drop('TIPO_EMAIL_3_INDEX')
df = df.drop('TIPO_EMAIL')

# COMMAND ----------

# DBTITLE 1,dummerizando telefones
df = df.withColumn('TIPO_TELEFONE_0_INDEX', F.col('TIPOS_TELEFONES').getItem(0))
df = df.withColumn('TIPO_TELEFONE_1_INDEX', F.col('TIPOS_TELEFONES').getItem(1))
df = df.withColumn('TIPO_TELEFONE_2_INDEX', F.col('TIPOS_TELEFONES').getItem(2))
df = df.withColumn('TIPO_TELEFONE_3_INDEX', F.col('TIPOS_TELEFONES').getItem(3))
df = df.withColumn('TIPO_TELEFONE_4_INDEX', F.col('TIPOS_TELEFONES').getItem(4))


df = df.withColumn('TIPO_TELEFONE_0', (F.when(F.col('TIPO_TELEFONE_0_INDEX')==0,True)\
                                .otherwise((F.when(F.col('TIPO_TELEFONE_1_INDEX')==0, True)\
                                .otherwise((F.when(F.col('TIPO_TELEFONE_2_INDEX')==0, True)\
                                .otherwise(F.when(F.col('TIPO_TELEFONE_3_INDEX')==0, True)\
                                .otherwise(False))))))))

df = df.withColumn('TIPO_TELEFONE_1', (F.when(F.col('TIPO_TELEFONE_0_INDEX')==1,True)\
                                .otherwise((F.when(F.col('TIPO_TELEFONE_1_INDEX')==1, True)\
                                .otherwise((F.when(F.col('TIPO_TELEFONE_2_INDEX')==1, True)\
                                .otherwise(F.when(F.col('TIPO_TELEFONE_3_INDEX')==1, True)\
                                .otherwise(False))))))))

df = df.withColumn('TIPO_TELEFONE_2', (F.when(F.col('TIPO_TELEFONE_0_INDEX')==2,True)\
                                .otherwise((F.when(F.col('TIPO_TELEFONE_1_INDEX')==2, True)\
                                .otherwise((F.when(F.col('TIPO_TELEFONE_2_INDEX')==2, True)\
                                .otherwise(F.when(F.col('TIPO_TELEFONE_3_INDEX')==2, True)\
                                .otherwise(False))))))))

df = df.withColumn('TIPO_TELEFONE_3', (F.when(F.col('TIPO_TELEFONE_0_INDEX')==3,True)\
                                .otherwise((F.when(F.col('TIPO_TELEFONE_1_INDEX')==3, True)\
                                .otherwise((F.when(F.col('TIPO_TELEFONE_2_INDEX')==3, True)\
                                .otherwise(F.when(F.col('TIPO_TELEFONE_3_INDEX')==3, True)\
                                .otherwise(False))))))))

df = df.withColumn('TIPO_TELEFONE_4', (F.when(F.col('TIPO_TELEFONE_0_INDEX')==4,True)\
                                .otherwise((F.when(F.col('TIPO_TELEFONE_1_INDEX')==4, True)\
                                .otherwise((F.when(F.col('TIPO_TELEFONE_2_INDEX')==4, True)\
                                .otherwise(F.when(F.col('TIPO_TELEFONE_3_INDEX')==4, True)\
                                .otherwise(False))))))))

df = df.drop('TIPO_TELEFONE_0_INDEX')
df = df.drop('TIPO_TELEFONE_1_INDEX')
df = df.drop('TIPO_TELEFONE_2_INDEX')
df = df.drop('TIPO_TELEFONE_3_INDEX')
df = df.drop('TIPO_TELEFONE_4_INDEX')
df = df.drop('TIPOS_TELEFONES')

# COMMAND ----------

# DBTITLE 1,dummerizando tipos de enderecos
df = df.withColumn('TIPO_ENDERECO_0_INDEX', F.col('TIPO_ENDERECO').getItem(0))
df = df.withColumn('TIPO_ENDERECO_1_INDEX', F.col('TIPO_ENDERECO').getItem(1))
df = df.withColumn('TIPO_ENDERECO_2_INDEX', F.col('TIPO_ENDERECO').getItem(2))
df = df.withColumn('TIPO_ENDERECO_3_INDEX', F.col('TIPO_ENDERECO').getItem(3))

df = df.withColumn('TIPO_ENDERECO_0', (F.when(F.col('TIPO_ENDERECO_0_INDEX')==0,True)\
                                .otherwise((F.when(F.col('TIPO_ENDERECO_1_INDEX')==0, True)\
                                .otherwise((F.when(F.col('TIPO_ENDERECO_2_INDEX')==0, True)\
                                .otherwise(F.when(F.col('TIPO_ENDERECO_3_INDEX')==0, True)\
                                .otherwise(False))))))))

df = df.withColumn('TIPO_ENDERECO_1', (F.when(F.col('TIPO_ENDERECO_0_INDEX')==1,True)\
                                .otherwise((F.when(F.col('TIPO_ENDERECO_1_INDEX')==1, True)\
                                .otherwise((F.when(F.col('TIPO_ENDERECO_2_INDEX')==1, True)\
                                .otherwise(F.when(F.col('TIPO_ENDERECO_3_INDEX')==1, True)\
                                .otherwise(False))))))))

df = df.withColumn('TIPO_ENDERECO_2', (F.when(F.col('TIPO_ENDERECO_0_INDEX')==2,True)\
                                .otherwise((F.when(F.col('TIPO_ENDERECO_1_INDEX')==2, True)\
                                .otherwise((F.when(F.col('TIPO_ENDERECO_2_INDEX')==2, True)\
                                .otherwise(F.when(F.col('TIPO_ENDERECO_3_INDEX')==2, True)\
                                .otherwise(False))))))))

df = df.withColumn('TIPO_ENDERECO_3', (F.when(F.col('TIPO_ENDERECO_0_INDEX')==3,True)\
                                .otherwise((F.when(F.col('TIPO_ENDERECO_1_INDEX')==3, True)\
                                .otherwise((F.when(F.col('TIPO_ENDERECO_2_INDEX')==3, True)\
                                .otherwise(F.when(F.col('TIPO_ENDERECO_3_INDEX')==3, True)\
                                .otherwise(False))))))))

df = df.drop('TIPO_ENDERECO_0_INDEX')
df = df.drop('TIPO_ENDERECO_1_INDEX')
df = df.drop('TIPO_ENDERECO_2_INDEX')
df = df.drop('TIPO_ENDERECO_3_INDEX')
df = df.drop('TIPO_ENDERECO')

# COMMAND ----------

# DBTITLE 1,atribuindo features do modelo no dataframe gerado
features_do_modelo = ['DOCUMENTO:ID_DIVIDA', 'DOCUMENTO_PESSOA','ID_DIVIDA','IDADE_PESSOA', 'ID_CEDENTE', 'VALOR_DIVIDA',
       'AGING', 'NOME_PRODUTO', 'DETALHES_CLIENTES_VENCIMENTO_FATURA',
       'DETALHES_CLIENTES_SCORE_CARGA', 'DETALHES_CLIENTES_VALOR_FATURA',
       'DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO',
       'DETALHES_CLIENTES_RENDA_CONSIDERADA',
       'DETALHES_CLIENTES_COLLECT_SCORE', 'DETALHES_CLIENTES_LIMITE_APROVADO',
       'DETALHES_CONTRATOS_SALDO_ATUAL_ALDIV', 'DETALHES_CONTRATOS_BLOQUEIO2',
       'DETALHES_CONTRATOS_CLASSE', 'DETALHES_CONTRATOS_CODIGO_LOGO',
       'DETALHES_CONTRATOS_HISTORICO_FPD',
       'DETALHES_CONTRATOS_BLOQUEIO2___DESC',
       'DETALHES_CONTRATOS_BLOQUEIO1___DESC',
       'DETALHES_CONTRATOS_CLASSE___DESC', 'DETALHES_CONTRATOS_STATUS_ACORDO',
       'DETALHES_CONTRATOS_VALOR_PRINCIPAL', 'DETALHES_DIVIDAS_VALOR_JUROS',
       'DETALHES_DIVIDAS_TAXA_SERVICO', 'DETALHES_DIVIDAS_TAXA_ATRASO',
       'DETALHES_DIVIDAS_VALOR_JUROS_DIARIO', 'DETALHES_DIVIDAS_TAXA_SEGURO',
       'TIPO_EMAIL_0', 'TIPO_EMAIL_1', 'TIPO_EMAIL_2', 'TIPO_EMAIL_3',
       'TIPO_TELEFONE_0', 'TIPO_TELEFONE_1', 'TIPO_TELEFONE_2',
       'TIPO_TELEFONE_3', 'TIPO_TELEFONE_4', 'TIPO_ENDERECO_0',
       'TIPO_ENDERECO_1', 'TIPO_ENDERECO_2', 'TIPO_ENDERECO_3']

df = df.select(features_do_modelo)

# COMMAND ----------

# DBTITLE 1,cast numeric features para float
numeric_features_to_transform = ['IDADE_PESSOA',
 'VALOR_DIVIDA',
 'AGING',
 'IDADE_PESSOA',
 'VALOR_DIVIDA',
 'AGING',
 'DETALHES_CLIENTES_SCORE_CARGA',
 'DETALHES_CLIENTES_VALOR_FATURA',
 'DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO',
 'DETALHES_CLIENTES_RENDA_CONSIDERADA',
 'DETALHES_CLIENTES_COLLECT_SCORE',
 'DETALHES_CLIENTES_LIMITE_APROVADO',
 'DETALHES_CONTRATOS_SALDO_ATUAL_ALDIV',
 'DETALHES_CONTRATOS_VALOR_PRINCIPAL',
 'DETALHES_DIVIDAS_VALOR_JUROS',
 'DETALHES_DIVIDAS_TAXA_SERVICO',
 'DETALHES_DIVIDAS_TAXA_ATRASO',
 'DETALHES_DIVIDAS_VALOR_JUROS_DIARIO',
 'DETALHES_DIVIDAS_TAXA_SEGURO']

for numeric_feature in numeric_features_to_transform:
  df = df.withColumn(numeric_feature, F.col(numeric_feature).cast(T.FloatType()))

# COMMAND ----------

# DBTITLE 1,trim de categorical features e None nas strings vazias
categorical_features = ['ID_CEDENTE',
 'NOME_PRODUTO',
 'DETALHES_CONTRATOS_BLOQUEIO2',
 'DETALHES_CONTRATOS_CLASSE',
 'DETALHES_CONTRATOS_CODIGO_LOGO',
 'DETALHES_CONTRATOS_HISTORICO_FPD',
 'DETALHES_CONTRATOS_BLOQUEIO2___DESC',
 'DETALHES_CONTRATOS_BLOQUEIO1___DESC',
 'DETALHES_CONTRATOS_CLASSE___DESC',
 'DETALHES_CONTRATOS_STATUS_ACORDO',
 'TIPO_EMAIL_0',
 'TIPO_EMAIL_1',
 'TIPO_EMAIL_2',
 'TIPO_EMAIL_3',
 'TIPO_TELEFONE_0',
 'TIPO_TELEFONE_1',
 'TIPO_TELEFONE_2',
 'TIPO_TELEFONE_3',
 'TIPO_TELEFONE_4',
 'TIPO_ENDERECO_0',
 'TIPO_ENDERECO_1',
 'TIPO_ENDERECO_2',
 'TIPO_ENDERECO_3']

for feature in categorical_features:
  df = df.withColumn(feature, F.trim(F.col(feature)))
  df = df.withColumn(feature, F.when(F.col(feature)=='',None).otherwise(F.col(feature)))

# COMMAND ----------

# DBTITLE 1,selecionando 10 mais relevantes de cada feature categorica, o resto vira 'outros'
for feature in categorical_features:
  top_features = df.groupBy(F.col(feature))\
      .agg(F.count(feature))\
      .orderBy(F.desc(F.col('count('+feature+')'))).limit(10).select(feature)\
      .rdd.map(lambda Row:Row[0]).collect()
  print (feature, top_features)
  df = df.withColumn(feature, F.when(F.col(feature).isin(top_features), F.col(feature)).otherwise('outros'))

# COMMAND ----------

# DBTITLE 1,transformando datatype de detalhes e criando coluna de atraso em vencimento e data arquivo
df = df.withColumn('DETALHES_CLIENTES_VENCIMENTO_FATURA', F.trim(F.col('DETALHES_CLIENTES_VENCIMENTO_FATURA')))\
       .withColumn('dia_vcto', F.substring(F.col('DETALHES_CLIENTES_VENCIMENTO_FATURA'),1,2))\
       .withColumn('mes_vcto', F.substring(F.col('DETALHES_CLIENTES_VENCIMENTO_FATURA'),3,2))\
       .withColumn('ano_vcto', F.substring(F.col('DETALHES_CLIENTES_VENCIMENTO_FATURA'),5,4))\
       .withColumn('DETALHES_CLIENTES_VENCIMENTO_FATURA', F.concat(F.col('ano_vcto'), F.lit('-'), F.col('mes_vcto'), F.lit('-'), F.col('dia_vcto'))\
       .cast('date'))
df = df.drop('dia_vcto')
df = df.drop('mes_vcto')
df = df.drop('ano_vcto')
df = df.withColumn('ATRASO_DETALHES_CLIENTES_VENCIMENTO_FATURA', F.datediff(F.lit(data_escolhida), F.col('DETALHES_CLIENTES_VENCIMENTO_FATURA')))
df = df.drop('DETALHES_CLIENTES_VENCIMENTO_FATURA')

# COMMAND ----------

# DBTITLE 1,dividindo valor de dividas e de colunas numericas de detalhes
df = df.withColumn('VALOR_DIVIDA', F.col('VALOR_DIVIDA')/10)
div_100 = ['DETALHES_CLIENTES_VALOR_FATURA',
            'DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO',
            'DETALHES_DIVIDAS_VALOR_JUROS',
            'DETALHES_DIVIDAS_TAXA_SERVICO',
            'DETALHES_DIVIDAS_TAXA_ATRASO',
            'DETALHES_DIVIDAS_TAXA_SEGURO']
for div in div_100:
  df = df.withColumn(div, F.col(div).cast(T.IntegerType())/100)

# COMMAND ----------

# DBTITLE 1,DROP CPF's null
df = df.filter(F.col('DOCUMENTO_PESSOA').isNotNull())

# COMMAND ----------

# DBTITLE 1,escreve parquet
df.write.mode('overwrite').parquet(writepath_trusted+'/'+data_escolhida+'/'+'trustedFile_credz'+'.PARQUET')

writepath_trusted+'/'+data_escolhida+'/'+'trustedFile_credz'+'.PARQUET'

# COMMAND ----------

# DBTITLE 1,escreve 10% da trusted em csv
"""
df2 = df.limit(int(df.count()/10))
for sch in df2.dtypes:
  nome = sch[0]
  tipo = sch[1]
  if 'array' in tipo:
    df2 = df2.withColumn(nome, F.col(nome).cast(T.StringType()))
    

df2.write.csv(writepath_trusted+'/'+data_escolhida+'/'+'trustedFile_credz_SAMPLE'+'.csv', header=True)
"""