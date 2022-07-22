# Databricks notebook source
import os
import pandas as pd
import numpy as np
import time

from pyspark.sql.types import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# COMMAND ----------

# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

caminho_sample = '/mnt/ml-prd/ml-data/propensaodeal/agibank'
timestr = time.strftime("%d-%m-%Y")
createdAt = time.strftime("%Y-%m-%d")
caminho_trusted = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/agibank/trusted'
santander_output = '/mnt/ml-prd/ml-data/propensaodeal/santander/output'

# COMMAND ----------

#file = caminho_sample+'/sample/MAILING_DIGITAL_QQ_15122021.TXT/sample/MAILING_DIGITAL_QQ_15122021.TXT_amostra_aleatoria.csv/part-00000-tid-7944637052338987946-a09c7290-c5fa-4cc0-93d3-aacf893a27ac-677-1-c000.csv' # Arquivo mais recente
file = caminho_sample+'/trusted/'+(max(os.listdir('/dbfs'+caminho_sample+'/trusted')))
files = max(os.listdir('/dbfs'+caminho_sample+'/trusted'))
df_sp = spark.read.option('delimiter',';').option('header', 'True').csv(file)
df = df_sp.toPandas()

# COMMAND ----------

date = files.split('_')[-1]
date = date.split('.')[0]
date = date[4:8]+'-'+date[2:4]+'-'+date[0:2]

# COMMAND ----------

df.head()

# COMMAND ----------

df_modelo = df[['CHAVE','DIAS_ATRASO_TOTAL', 'VALOR_ATRASO_TOTAL']].copy()

df_modelo['DIAS_ATRASO_TOTAL'] = df_modelo['DIAS_ATRASO_TOTAL'].astype(str).astype(int)
df_modelo['VALOR_ATRASO_TOTAL'] = df_modelo['VALOR_ATRASO_TOTAL'].str.replace(',', '.').astype(str).astype(float)
df_modelo['CHAVE'] = df_modelo['CHAVE'].astype(str).astype(int)


df_modelo['ULT_OCORRENCIA_ENCERCTT'] = np.where(df['ULT_OCORRENCIA'] == 'ENCERCTT', 1, 0)
df_modelo['ULT_OCORRENCIA_DESCOCLI'] = np.where(df['ULT_OCORRENCIA'] == 'DESCOCLI', 1, 0)
df_modelo['ULT_OCORRENCIA_EMISBOL'] = np.where(df['ULT_OCORRENCIA'] == 'EMISBOL', 1, 0)

df_modelo['FONTEPAGADORA_INSS'] = np.where(df['FONTEPAGADORA'] == 'INSS', 1, 0)

df_modelo['CD_UF_SP'] = np.where(df['CD_UF'] == 'SP', 1, 0)
df_modelo['CD_UF_RS'] = np.where(df['CD_UF'] == 'RS', 1, 0)
df_modelo['CD_UF_MG'] = np.where(df['CD_UF'] == 'MG', 1, 0)
df_modelo['CD_UF_RJ'] = np.where(df['CD_UF'] == 'RJ', 1, 0)
df_modelo['CD_UF_PR'] = np.where(df['CD_UF'] == 'PR', 1, 0)
df_modelo['CD_UF_BA'] = np.where(df['CD_UF'] == 'BA', 1, 0)


df_modelo['SEGMENTOFONTEPAGADORA_CP_Estadual'] = np.where(df['SEGMENTOFONTEPAGADORA_CP'] == 'Estadual', 1, 0)
df_modelo['SEGMENTOFONTEPAGADORA_CP_Federal'] = np.where(df['SEGMENTOFONTEPAGADORA_CP'] == 'Federal', 1, 0)
df_modelo['SEGMENTOFONTEPAGADORA_CP_Municipal'] = np.where(df['SEGMENTOFONTEPAGADORA_CP'] == 'Municipal', 1, 0)


df_modelo['MAILING_CP_PURO_MENOR_QUEBRAS'] = np.where(df['MAILING'].str.contains('CP_PURO_MENOR_QUEBRAS') == True, 1, 0)


df_modelo['SCORE_1'] = np.where(df['SCORE'] == '1', 1, 0)
df_modelo['SCORE_2'] = np.where(df['SCORE'] == '2', 1, 0)
df_modelo['SCORE_3'] = np.where(df['SCORE'] == '3', 1, 0)
df_modelo['SCORE_4'] = np.where(df['SCORE'] == '4', 1, 0)
df_modelo['SCORE_5'] = np.where(df['SCORE'] == '5', 1, 0)


df_modelo['ORDEM_MAILING_1'] = np.where(df['ORDEM_MAILING'] == '1', 1, 0)
df_modelo['ORDEM_MAILING_3'] = np.where(df['ORDEM_MAILING'] == '3', 1, 0)
df_modelo['ORDEM_MAILING_99'] = np.where(df['ORDEM_MAILING'] == '99', 1, 0)


df_modelo['PRODUTO_GRUPO_CP_PURO'] = np.where(df['PRODUTO_GRUPO'] == 'CP PURO', 1, 0)
df_modelo['PRODUTO_GRUPO_CC_TOPAZ_PURO'] = np.where(df['PRODUTO_GRUPO'] == 'CC TOPAZ PURO', 1, 0)
df_modelo['PRODUTO_GRUPO_MULTIPRODUTO'] = np.where(df['PRODUTO_GRUPO'] == 'MULTIPRODUTO', 1, 0)


df_modelo['BANCOCONTRATO_BRADESCO'] = np.where(df['BANCOCONTRATO'] == 'BANCO_BRADESCO_S.A.', 1, 0)
df_modelo['BANCOCONTRATO_BANCODOBRASIL'] = np.where(df['BANCOCONTRATO'] == 'BANCO_DO_BRASIL_S.A.', 1, 0)
df_modelo['BANCOCONTRATO_AGIBANK'] = np.where(df['BANCOCONTRATO'] == 'Banco_Agibank_S.A.', 1, 0)


df_modelo['DT_DIFF_ULTIMO_PAGAMENTO'] = (pd.to_datetime("now") - pd.to_datetime(df['ULTIMO_PAGAMENTO'], format='%Y/%m/%d')).dt.days # Há casos de missings e portanto talvez deverá haver mudança na variável em caso de regressão logística
df_modelo['EMAIL'] = np.where(df['DS_EMAIL'].isnull() == True, 0, 1)
df_modelo['EXTRA_1'] = df['EXTRA1'].str.replace(',', '.').astype(float)
df_modelo['EXTRA_2'] = df['EXTRA2'].str.replace(',', '.').astype(float)



# y = np.where(df_modelo['VARIAVEL_RESPOSTA'] == 'true', 1, 0) Não vai existir
x = df_modelo
x = x.drop(columns = ['CHAVE','ORDEM_MAILING_99', 'MAILING_CP_PURO_MENOR_QUEBRAS','FONTEPAGADORA_INSS','PRODUTO_GRUPO_MULTIPRODUTO'])
x = x.fillna(0)

# COMMAND ----------

x.head()

# COMMAND ----------

import pickle
filename = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/agibank/finalized_model.sav'
modelo=pickle.load(open(filename, 'rb'))

# COMMAND ----------

variaveis_numericas = ['DIAS_ATRASO_TOTAL','VALOR_ATRASO_TOTAL','DT_DIFF_ULTIMO_PAGAMENTO', 'EXTRA_1','EXTRA_2']

scaler = StandardScaler()
scaled_numfeats_train = pd.DataFrame(scaler.fit_transform(x[variaveis_numericas]), 
                                     columns=variaveis_numericas, index= x.index)
for col in variaveis_numericas:
    x[col] = scaled_numfeats_train[col]

# COMMAND ----------

chave = df_modelo['CHAVE'].astype(str).str.zfill(11)
p_1 = modelo.predict_proba(x)[:,1]

dt_gh = pd.DataFrame({'Chave': chave, 'P_1':p_1})

# COMMAND ----------

dt_gh['P_1_pk_4_g_1'] = np.where(dt_gh['P_1'] <= 0.01, 0.0,
    np.where(np.bitwise_and(dt_gh['P_1'] > 0.01, dt_gh['P_1'] <= 0.14), 1.0,
    np.where(dt_gh['P_1'] > 0.14,2,0)))

dt_gh['P_1_S_pk_34_g_1'] = np.where(dt_gh['P_1'] <= 0.01, 0.0,
    np.where(np.bitwise_and(dt_gh['P_1'] > 0.01, dt_gh['P_1'] <= 0.03), 1.0,
    np.where(np.bitwise_and(dt_gh['P_1'] > 0.03, dt_gh['P_1'] <= 0.68), 2.0,
    np.where(dt_gh['P_1'] > 0.68,3,0))))

dt_gh['GH'] = np.where(np.bitwise_and(dt_gh['P_1_pk_4_g_1'] == 0, dt_gh['P_1_S_pk_34_g_1'] == 0), 0,
    np.where(np.bitwise_and(dt_gh['P_1_pk_4_g_1'] == 0, dt_gh['P_1_S_pk_34_g_1'] == 1), 0,
    np.where(np.bitwise_and(dt_gh['P_1_pk_4_g_1'] == 0, dt_gh['P_1_S_pk_34_g_1'] == 2), 1,
    np.where(np.bitwise_and(dt_gh['P_1_pk_4_g_1'] == 0, dt_gh['P_1_S_pk_34_g_1'] == 3), 2,
    np.where(np.bitwise_and(dt_gh['P_1_pk_4_g_1'] == 1, dt_gh['P_1_S_pk_34_g_1'] == 0), 1,
    np.where(np.bitwise_and(dt_gh['P_1_pk_4_g_1'] == 1, dt_gh['P_1_S_pk_34_g_1'] == 1), 1,
    np.where(np.bitwise_and(dt_gh['P_1_pk_4_g_1'] == 1, dt_gh['P_1_S_pk_34_g_1'] == 2), 2,
    np.where(np.bitwise_and(dt_gh['P_1_pk_4_g_1'] == 1, dt_gh['P_1_S_pk_34_g_1'] == 3), 2,
    np.where(np.bitwise_and(dt_gh['P_1_pk_4_g_1'] == 2, dt_gh['P_1_S_pk_34_g_1'] == 0), 1,
    np.where(np.bitwise_and(dt_gh['P_1_pk_4_g_1'] == 2, dt_gh['P_1_S_pk_34_g_1'] == 1), 2,
    np.where(np.bitwise_and(dt_gh['P_1_pk_4_g_1'] == 2, dt_gh['P_1_S_pk_34_g_1'] == 2), 3,
    np.where(np.bitwise_and(dt_gh['P_1_pk_4_g_1'] == 2, dt_gh['P_1_S_pk_34_g_1'] == 3), 4,0))))))))))))

del dt_gh['P_1_pk_4_g_1']
del dt_gh['P_1_S_pk_34_g_1']

sparkDF=spark.createDataFrame(dt_gh) 


    

# COMMAND ----------

sparkDF=[]
schema = StructType([ \
    StructField("Chave",StringType(),True), \
    StructField("P_1",FloatType(),True), \
    StructField("GH",IntegerType(),True), \
  ])

# COMMAND ----------

sparkDF=spark.createDataFrame(dt_gh,schema) 


# COMMAND ----------

sparkDF = sparkDF.withColumn('Document', F.lpad(F.col("Chave"), 11, '0'))
sparkDF = sparkDF.withColumn('P_1', F.col("P_1").cast(DecimalType(11, 10)))
sparkDF = sparkDF.groupBy(F.col('Document')).agg(F.max(F.col('GH')), F.max(F.col('P_1')), F.avg(F.col('P_1')))
sparkDF = sparkDF.withColumn('Provider', F.lit('qq_agibank_propensity_v1'))
sparkDF = sparkDF.withColumn('Date', F.lit(date))
sparkDF = sparkDF.withColumn('CreatedAt', F.lit(createdAt))
sparkDF = changeColumnNames(sparkDF, ['Document','Score','ScoreValue','ScoreAvg','Provider','Date','CreatedAt'])

# COMMAND ----------

sparkDF.show(10, truncate = False)
#sparkDF.schema.fields
#sparkDF.filter(sparkDF.Chave == '51206110449').show(truncate = False)

# COMMAND ----------

sparkDF.coalesce(1).write.mode('overwrite').options(header='True', delimiter=';').csv(caminho_sample+'/output/tmp')

for file in dbutils.fs.ls(caminho_sample+'/output/tmp'):
  if file.name.split('.')[-1] == 'csv':
    dbutils.fs.cp(file.path, caminho_sample+'/output/agibank_model_to_production_'+timestr+'.csv')
    dbutils.fs.rm(caminho_sample+'/output/tmp', recurse=True)

# COMMAND ----------

dbutils.fs.cp(caminho_sample+'/output/agibank_model_to_production_'+timestr+'.csv', santander_output+'/agibank_model_to_production_'+timestr+'.csv')