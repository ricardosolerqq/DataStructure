# Databricks notebook source
import datetime
date = datetime.datetime.today()
date = str(date.year)+'-'+str(date.month).zfill(2)+'-'+str(date.day).zfill(2)
date = '2021-12-14'
date

# COMMAND ----------

import os
from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from functools import reduce

# COMMAND ----------

caminho_sample = '/mnt/etlsantander/Base_UFPEL/sample'
caminho_trusted = '/mnt/etlsantander/Base_UFPEL/trusted'

# COMMAND ----------

dbutils.widgets.dropdown('ESCREVER_VARIAVEL_RESPOSTA', 'False', ['True', 'False'])
escreverVariavelResposta = dbutils.widgets.get('ESCREVER_VARIAVEL_RESPOSTA')
if escreverVariavelResposta == 'True':
  variavelResposta = True
else:
  variavelResposta = False

# COMMAND ----------

#a) Débitos: Tab_vals1.to_csv(dir_debts_datas+'/Base_UFPEL_QQ_Model_std_P'+str(j).zfill(2)+'.csv',index=False,sep=";")

path = "{0}{1}{2}".format("/mnt/etlsantander/Base_UFPEL/Skips_pags/Dados=",date,"/Base_UFPEL_QQ_Model_std_*.csv")
df_debitos = (spark.read
               .format("com.databricks.spark.csv")
               .option("header", "true")
               .option("inferSchema", "true")
               .option("delimiter", ";")
               .load(path))

# COMMAND ----------

#b) Contatos: tab_training.to_csv(dir_phones_datas+'/Base_UFPEL_QQ_Model_std_phones_P'+str(j).zfill(2)+'.csv',index=False,sep=";")

path = "{0}{1}{2}".format("/mnt/etlsantander/Base_UFPEL/Tags_contatos_phones/Dados=",date,"/Base_UFPEL_QQ_Model_std_phones_*.csv")

df_contatos = (spark.read
                 .format("com.databricks.spark.csv")
                 .option("header", "true")
                 .option("inferSchema", "true")
                 .option("delimiter", ";")
                 .load(path))

# COMMAND ----------

df_join = df_debitos.join(df_contatos, ['document'], 'left').drop(df_contatos.document)

# COMMAND ----------

df_join = df_join.distinct()

# COMMAND ----------

if variavelResposta:
  df_avista = df_join.filter(F.col('PAGTO_A_VISTA')==True).limit(1500)
  df_pag = df_join.filter(F.col('PAGTO')==True).limit(1500)
  df_acordo = df_join.filter(F.col('ACORDO')==True).limit(1500)
  
  df_limit = df_join.filter((F.col('PAGTO_A_VISTA')==False) & (F.col('PAGTO')==False) & (F.col('ACORDO')==False)).limit(47000)
  
  df_vresposta = [df_avista,df_pag,df_acordo,df_limit]
  df_vresposta = reduce(DataFrame.unionAll, df_vresposta)
  
  df_vresposta.coalesce(1).write.option('sep', ';').option('header', 'True').csv(os.path.join(caminho_sample, 'aleatorio_temp'))
  for file in dbutils.fs.ls(os.path.join(caminho_sample, 'aleatorio_temp')):
    if file.name.split('.')[-1] == 'csv':
      dbutils.fs.cp(file.path, os.path.join(caminho_sample, 'base_aleatoria.csv'))
  dbutils.fs.rm(os.path.join(caminho_sample, 'aleatorio_temp'), True)
  
else:
  df.coalesce(1).write.option('sep', ';').option('header', 'True').csv(os.path.join(caminho_trusted, 'trusted_tmp'))
  for file in dbutils.fs.ls(os.path.join(caminho_trusted, 'trusted_tmp')):
    if file.name.split('.')[-1] == 'csv':
      dbutils.fs.cp(file.path, os.path.join(caminho_trusted, 'santander_'+date+'.csv'))
  dbutils.fs.rm(os.path.join(caminho_trusted, 'trusted_tmp'), True)