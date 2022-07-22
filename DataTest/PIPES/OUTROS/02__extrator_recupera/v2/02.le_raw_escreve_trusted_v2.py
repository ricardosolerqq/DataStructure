# Databricks notebook source
# MAGIC %run "/pipe_modelos/extrator_recupera/v2/00.le_regras_disponibiliza_variaveis_v2"

# COMMAND ----------

# DBTITLE 1,configurando widget de credores
dbutils.widgets.text('credor','','')
credor = dbutils.widgets.get('credor')

credor = Credor(credor, output = True)

# COMMAND ----------

# DBTITLE 1,construindo leitura dos dataframes geradas no step anterior e escrevendo na trusted
dict_dfs = {}
for file in os.listdir(os.path.join(credor.caminho_raw_dbfs)):
  print (file)
  dict_dfs.update({file.split('.')[0]:spark.read.parquet(os.path.join(credor.caminho_raw,file))})

# COMMAND ----------

# DBTITLE 1,construindo colunas usando as regras e escrevendo df's que possuam informações - dropando colunas de controle
dict_dfs_transform = {}
inventario = []
for tipo in dict_dfs:
  print (tipo)
  df = dict_dfs[tipo]
  regras_do_tipo = regras_por_tipo(credor.df_regras, tipo)
  for regra in regras_do_tipo:
    df = df.withColumn(regra, F.substring(F.col('raw_info'), int(regras_do_tipo[regra][0]), int(regras_do_tipo[regra][1])))
  df = df.drop('raw_info').drop('inter').drop('reg')
  df = df.drop('tip_reg').drop('cod_siste').drop('tip_inter')
  #df = df.drop('tip_reg').drop('cod_siste').drop('dat_movto').drop('tip_inter')
  count = df.count()
  #print ('count', count)
  if count >0:
    dict_dfs_transform.update({tipo:df})

for df in dict_dfs_transform:
  try:
    dict_dfs_transform[df].write.mode('overwrite').parquet(os.path.join(credor.caminho_trusted, df+'.PARQUET'))
    inventario.append([tipo, True])
    print ('df: '+df)

  except:
    print ('EXCEPT no arquivo',df)
    inventario.append([tipo, False])


# COMMAND ----------

regras_do_tipo

# COMMAND ----------

# DBTITLE 1,escrevendo dataframe de inventário do arquivo
schema = T.StructType([T.StructField('tipo', T.StringType(), False),
                       T.StructField('existe', T.BooleanType(), False)])
inventario_df = spark.createDataFrame(inventario, schema = schema)



inventario_df.write.mode('overwrite').parquet(os.path.join(credor.caminho_logs, str(datetime.datetime.today().date())+'_inventario.PARQUET'))
display(inventario_df.filter(F.col('existe')==True))

# COMMAND ----------

features = []
for df in dict_dfs_transform:
  for feature in dict_dfs_transform[df].columns:
    if feature not in features:
      features.append(feature)
features

matriz_existencia_colunas = []
for df in dict_dfs_transform:
  matriz_existencia_colunas_interna = [df]
  for feature in features:
    if feature in dict_dfs_transform[df].columns:
      matriz_existencia_colunas_interna.append('X')
    else:
      matriz_existencia_colunas_interna.append(' ')
  matriz_existencia_colunas.append(matriz_existencia_colunas_interna)
#matriz_existencia_colunas = spark.sparkContext.parallelize(matriz_existencia_colunas)

df_features_schema = T.StructType(
  [ T.StructField('tipo', T.StringType(), True) ]+ [ T.StructField(feature, T.StringType(), True) for feature in features ]
)

df_features = spark.createDataFrame(matriz_existencia_colunas, schema = df_features_schema)
df_features.write.mode('overwrite').parquet(os.path.join(credor.caminho_logs, str(datetime.datetime.today().date())+'_features.PARQUET'))


display(df_features)

# COMMAND ----------

dbutils.notebook.exit('OK')