# Databricks notebook source
# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

#mount_blob_storage_oauth(dbutils,'qqprd','ml-prd',"/mnt/ml-prd")
dir_models='/mnt/ml-prd/ml-data/propensaodeal/santander/processed_score'
dir_outputs='/mnt/bi-reports/VALIDACAO-MODELOS-ML'

# COMMAND ----------

# DBTITLE 1,Pacotes Python a serem carregados
from datetime import datetime, timedelta
from pyspark.sql import functions as F

# COMMAND ----------

df = spark.read.options(header = True, delimiter = ';').csv(dir_outputs+'/arquivos_originais/Base acionada Recovery_2022-03.csv')
##
df = df.orderBy(F.col('data'))
##
#display(df)

df_aux = df
df_tmp = df_aux.limit(6915447)
df_aux= df_aux.subtract(df_tmp)


#dfa = df.filter((F.col('data')>= '2022-03-01') & (F.col('data') <= '2022-03-29'))
#dfb = df.filter(F.col('data') > '2022-03-29')


#Base acionada Santander_2022-03_1.csv
#Base acionada Santander_2022-03_2.csv
#3.480.995

#df_tmp.coalesce(1).write.options(header = True, delimiter = ';').csv(dir_outputs+'/arquivos_originais/tmpWP/')
#for files in dbutils.fs.ls(dir_outputs+'/arquivos_originais/tmpWP/'):
#  if files.name.split('.')[-1] == 'csv':
#    dbutils.fs.cp(files.path, dir_outputs+'/arquivos_originais/'+'Base acionada Recovery_2022-03_1.csv')
#  else:
#      dbutils.fs.rm(files.path, True)
#dbutils.fs.rm(os.path.join(dir_outputs, 'arquivos_originais/tmpWP'), True) 
                           
df_aux.coalesce(1).write.options(header = True, delimiter = ';').csv(dir_outputs+'/arquivos_originais/tmpWP/')
for files in dbutils.fs.ls(dir_outputs+'/arquivos_originais/tmpWP/'):
  if files.name.split('.')[-1] == 'csv':
    dbutils.fs.cp(files.path, dir_outputs+'/arquivos_originais/'+'Base acionada Recovery_2022-03_2.csv')
  else:
      dbutils.fs.rm(files.path, True)
dbutils.fs.rm(os.path.join(dir_outputs, 'arquivos_originais/tmpWP'), True)         

# COMMAND ----------

display(df)

# COMMAND ----------

display(df.groupBy(F.col('data')).count().orderBy(F.col('data')))

# COMMAND ----------

display(df_tmp.groupBy(F.col('data')).count().orderBy(F.col('data')))

# COMMAND ----------

display(df_aux.groupBy(F.col('data')).count().orderBy(F.col('data')))

# COMMAND ----------

display(dfb.groupBy(F.col('data')).count().orderBy(F.col('data')))

# COMMAND ----------

display(df_tmp.groupBy(F.col('data')).count())

# COMMAND ----------

display(df.groupBy(F.col('data')).count().orderBy(F.col('data')))

# COMMAND ----------

df_aux.count(), df_tmp.count()

# COMMAND ----------

df_data=spark.createDataFrame(dbutils.fs.ls(dir_models)).withColumn('path',F.regexp_replace("path","dbfs:",""))

# COMMAND ----------

df_data = df_data.withColumn('Credor',F.split(F.col('name'),'_')[0])

# COMMAND ----------

df_data.show(5,False)

# COMMAND ----------

df_data.filter(F.col('Credor') == 'credigy').show(5,False)

# COMMAND ----------

dfrec = spark.read.options(delimiter=';',header = True).csv(dir_outputs+'/arquivos_originais/Base acionada Recovery formatada para análise.csv')
dfrec.show(5,False)

# COMMAND ----------

display(dfrec)

# COMMAND ----------

display(dfrec.filter(F.col('CPF') == '00005293383'))

# COMMAND ----------

dfrec.groupBy(F.col('tagContato')).count().show(100,False)

# COMMAND ----------

dfrec.groupBy(F.col('tag')).count().show(100,False)

# COMMAND ----------

dfrec.groupBy(F.col('data_campanha')).count().orderBy((F.to_date(F.col('data_campanha'), 'dd/MM/yyyy')).desc()).show(100,False)

# COMMAND ----------

df_data.groupBy(F.col('Credor')).count().orderBy(F.col('Credor')).show(100,False)

# COMMAND ----------

df_data.show(5,False)

# COMMAND ----------

# DBTITLE 1,Selecionando os credores ativos
col_creditor = getPyMongoCollection('col_creditor')

query_creditor=[
  {
    "$match" : {
        "active" : True      }
  },
  {
    "$project" : {
        "_id" : 1
      }
  }
]

Credores=pd.DataFrame(list(col_creditor.aggregate(pipeline=query_creditor,allowDiskUse=True)))['_id'].to_list()
Credores=pd.DataFrame({'Credores' : Credores,'Cod_leitura' : Credores})
Credores.iloc[Credores['Cod_leitura']=='santander',1]='part-'
#Credores=Credores.iloc[[0,1,2,10],:]
Credores=spark.createDataFrame(Credores)
display(Credores)

# COMMAND ----------

columns = ['Credores','Cod_leitura']
vals = [('credigy','credigy')]
Credor = spark.createDataFrame(vals, columns)
Credor.show()

# COMMAND ----------

Credores = Credores.union(Credor).collect()
display(Credores)

# COMMAND ----------

# DBTITLE 1,Selecionando as bases escoradas por credor
df_data=spark.createDataFrame(dbutils.fs.ls(dir_models)).withColumn('path',F.regexp_replace("path","dbfs:",""))
for i in Credores:
  print('Montando a tabela de análise dos modelos de '+i.Credores+'...\n')
  print('Selecionando somente os arquivos '+i.Credores+'...\n')
  if i.Credores=='santander':
    df_data_read=df_data.filter(((F.col('name').contains(i.Cod_leitura)) |(F.col('name').contains('santander')) ) & (F.col('size')>0))
  else:
    df_data_read=df_data.filter((F.col('name').contains(i.Cod_leitura)) & (F.col('size')>0))
    
  if df_data_read.count()==0 or i.Credores=='pan':
    print('Ainda não há modelos '+i.Credores+' em produção, indo para o credor seguinte...\n')
    continue
    
  try:
    print('Retirando aqueles que já foram lidos em processos anteriores...\n')
    df_readed_files=spark.read.format("csv").option("header", True).option("delimiter", ";").load(dir_outputs+'/Tabelas_Acionamentos_News/'+i.Credores+'/readed_files_'+i.Credores+'.csv').rdd.flatMap(lambda x: x).collect()
    df_data_read=df_data_read.filter(~(F.col('name').isin(df_readed_files)))
    if df_data_read.count()==0:
      print('A base está completamente atualizada, partindo para o credor seguinte...\n')
      continue
  except:
    print('Não há arquivos anteriores desse credor, portanto todos serão lidos...\n')
    pass
  
  print('Executando a leitura e concatenação dos arquivos..\n')
  df_data_read_2=df_data_read.collect()
  df_data_read_2 = [ x.path for x in df_data_read_2]
  #df_data_read2
  BASES_ORIGINAL = [spark.read.option("header", "true")\
           .option("delimiter", ";")\
           .csv(x)\
           for x in df_data_read_2]
  head = True
  for x in BASES_ORIGINAL:
    try:
      if head == True:
        df = x
        head = False
      else:
        df = df.union(x)
    except:
      print(x)

  df=df.withColumn('CreatedAt',F.to_date(F.col('CreatedAt'))).sort(F.col('CreatedAt').asc())    

  del BASES_ORIGINAL, df_data_read_2

  df=df.groupby('Document','provider').agg(F.last('Score').alias('Score'),
                                          F.last('ScoreValue').alias('ScoreValue'),
                                          F.last('ScoreAvg').alias('ScoreAvg'),
                                          F.first('CreatedAt').alias('First_Data'),
                                          F.last('CreatedAt').alias('Last_Data')).cache()
  print('Carregando as bases anteriores (caso existam)...\n')
  try:
    print("Concatenando com a base anterior...\n")
    last_table=spark.createDataFrame(dbutils.fs.ls(dir_outputs+'/Tabelas_Acionamentos_News/'+i.Credores)).withColumn('path',F.regexp_replace("path","dbfs:","")).filter(F.col('name').contains('last_files')).collect()
    last_table=spark.read.format("csv").option("header", True).option("delimiter", ";").load(last_table[0].path).withColumn('First_Data',F.to_date(F.col('First_Data'))).withColumn('Last_Data',F.to_date(F.col('Last_Data'))).cache()
    df=df.union(last_table).sort(F.col('Last_Data').asc())
    #del last_table
  except:
    print("Nenhum arquivo anterior encontrado, acumulando os já lidos...\n")
    pass


  df_scores_model=df.groupby('Document','provider').agg(F.last('Score').alias('Score'),
                                          F.last('ScoreValue').alias('ScoreValue'),
                                          F.last('ScoreAvg').alias('ScoreAvg'),
                                          F.min('First_Data').alias('First_Data'),
                                          F.max('Last_Data').alias('Last_Data')).cache()
  #data_min=df_scores_model.agg({"First_Data": "min"}).collect()[0]
  #data_max=df_scores_model.agg({"Last_Data": "max"}).collect()[0]
  del df
   
  try:
    print("Atualizando e salvando a lista de arquivos já lidos...\n")
    df_readed_files=spark.read.format("csv").option("header", True).option("delimiter", ";").load(dir_outputs+'/Tabelas_Acionamentos_News/'+i.Credores+'/readed_files_'+i.Credores+'.csv')
    df_readed_files=df_readed_files.union(df_data_read.select('name'))

    df_readed_files.coalesce(1).write.mode('overwrite').option("header",True).option("delimiter",";").option("emptyValue",'').csv(dir_outputs+'/Tabelas_Acionamentos_News/'+i.Credores+'/readed_files_'+i.Credores)

    for file in dbutils.fs.ls(dir_outputs+'/Tabelas_Acionamentos_News/'+i.Credores+'/readed_files_'+i.Credores):
      if file.name.split('.')[-1] == 'csv':
        print (file)
        dbutils.fs.cp(file.path, dir_outputs+'/Tabelas_Acionamentos_News/'+i.Credores+'/readed_files_'+i.Credores+'.csv')
    dbutils.fs.rm(dir_outputs+'/Tabelas_Acionamentos_News/'+i.Credores+'/readed_files_'+i.Credores, True)

    del df_readed_files, df_data_read
  except:
    print('Não há arquivos lidos anteriormente, portanto os arquivos lidos nessa consulta serão salvos como os principais...\n')
    df_data_read.select('name').coalesce(1).write.mode('overwrite').option("header",True).option("delimiter",";").option("emptyValue",'').csv(dir_outputs+'/Tabelas_Acionamentos_News/'+i.Credores+'/readed_files_'+i.Credores)

    for file in dbutils.fs.ls(dir_outputs+'/Tabelas_Acionamentos_News/'+i.Credores+'/readed_files_'+i.Credores):
      if file.name.split('.')[-1] == 'csv':
        print (file)
        dbutils.fs.cp(file.path, dir_outputs+'/Tabelas_Acionamentos_News/'+i.Credores+'/readed_files_'+i.Credores+'.csv')
    dbutils.fs.rm(dir_outputs+'/Tabelas_Acionamentos_News/'+i.Credores+'/readed_files_'+i.Credores, True)
    del df_data_read
    
  
  print('Salvando todos os escorados...')
  data_hj=(datetime.today()-timedelta(hours=3)).strftime("%Y_%m_%d")
  df_scores_model.coalesce(1).write.mode('overwrite').option("header",True).option("delimiter",";").option("emptyValue",'').csv(dir_outputs+'/Tabelas_Acionamentos_News/'+i.Credores+'/last_files_'+i.Credores)

  for file in dbutils.fs.ls(dir_outputs+'/Tabelas_Acionamentos_News/'+i.Credores+'/last_files_'+i.Credores):
    if file.name.split('.')[-1] == 'csv':
      print (file)
      dbutils.fs.cp(file.path, dir_outputs+'/Tabelas_Acionamentos_News/'+i.Credores+'/last_files_'+i.Credores+'_'+data_hj+'.csv')
  dbutils.fs.rm(dir_outputs+'/Tabelas_Acionamentos_News/'+i.Credores+'/last_files_'+i.Credores, True)

  del df_scores_model
  
  print('Excluindo o arquivo anterior...')
  try:
    last_file_remove=spark.createDataFrame(dbutils.fs.ls(dir_outputs+'/Tabelas_Acionamentos_News/'+i.Credores)).filter(~(F.col('name')=='last_files_'+i.Credores+'_'+data_hj+'.csv')).collect()
    for i in last_file_remove:
      if 'last_files' in i.name:
        dbutils.fs.rm(i.path, True)
    del last_file_remove    
  except:
    pass