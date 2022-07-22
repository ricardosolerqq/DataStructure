# Databricks notebook source
# MAGIC %run "/pipe_modelos/extrator_recupera/v1/00.le_regras_disponibiliza_variaveis"

# COMMAND ----------

# DBTITLE 1,configurando widget de credores
credores_recupera = ['agibank','bmg','fort','tribanco','trigg','valia','zema']
dbutils.widgets.dropdown('CREDOR_ESCOLHIDO', 'bmg', credores_recupera)
credor = dbutils.widgets.get('CREDOR_ESCOLHIDO')

caminho_base,caminho_base_dbfs,caminho_raw,caminho_raw_dbfs,caminho_trusted,caminho_trusted_dbfs,caminho_joined_trusted,caminho_joined_trusted_dbfs,caminho_sample,caminho_sample_dbfs = obtem_variaveis_caminho(credor)

# COMMAND ----------

files = []
for file in os.listdir(caminho_raw_dbfs):
  files.append(file)

# COMMAND ----------

try:
  dbutils.widgets.remove('ARQUIVO_ESCOLHIDO')
except:
  pass

# COMMAND ----------

dbutils.widgets.dropdown('ARQUIVO_ESCOLHIDO', max(files), files)

# COMMAND ----------

arquivo_escolhido = dbutils.widgets.get("ARQUIVO_ESCOLHIDO")
arquivo_escolhido

# COMMAND ----------

# DBTITLE 1,construindo leitura dos dataframes geradas no step anterior e escrevendo na trusted
dict_dfs = {}
for file in os.listdir(os.path.join(caminho_raw_dbfs, arquivo_escolhido)):
  print (file)
  dict_dfs.update({file.split('.')[0]:spark.read.parquet(os.path.join(caminho_raw, arquivo_escolhido,file))})

# COMMAND ----------

# DBTITLE 1,construindo colunas usando as regras e escrevendo df's que possuam informações - dropando colunas de controle
dict_dfs_transform = {}
inventario = []
for tipo in dict_dfs:
  #print (tipo)
  df = dict_dfs[tipo]
  regras_do_tipo = regras_por_tipo(df_regras, tipo)
  for regra in regras_do_tipo:
    #print ('\t',regra)
    df = df.withColumn(regra, F.substring(F.col('raw_info'), int(regras_do_tipo[regra][0]), int(regras_do_tipo[regra][1])))
  df = df.drop('raw_info').drop('inter').drop('reg')
  df = df.drop('tip_reg').drop('cod_siste').drop('dat_movto').drop('tip_inter')
  count = df.count()
  #print ('count', count)
  if count >0:
    dict_dfs_transform.update({tipo:df})
    inventario.append([tipo, True])
  else:
    inventario.append([tipo, False])

for df in dict_dfs_transform:
  dict_dfs_transform[df].write.mode('overwrite').parquet(os.path.join(caminho_trusted, arquivo_escolhido, df+'.PARQUET'))

# COMMAND ----------

# DBTITLE 1,escrevendo dataframe de inventário do arquivo
schema = T.StructType([T.StructField('tipo', T.StringType(), False),
                       T.StructField('existe', T.BooleanType(), False)])
inventario_df = spark.createDataFrame(inventario, schema = schema)

inventario_df.write.mode('overwrite').parquet(os.path.join(caminho_trusted, arquivo_escolhido+'_inventario.PARQUET'))
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
display(df_features)