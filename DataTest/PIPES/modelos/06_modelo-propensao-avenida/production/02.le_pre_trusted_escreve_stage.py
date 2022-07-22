# Databricks notebook source
# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

import os

# COMMAND ----------

blob_account_source_ml = "qqdatastoragemain"
blob_container_source_ml = "ml-prd"

mount_blob_storage_key(dbutils,blob_account_source_ml,blob_account_source_ml,'/mnt/ml-prd')

parquetfilepath = '/mnt/ml-prd/ml-data/propensaodeal/avenida/pre-trusted'
parquetfilepath_dbfs = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/avenida/pre-trusted'

writepath_stage = '/mnt/ml-prd/ml-data/propensaodeal/avenida/stage'
writepath_trusted = '/mnt/ml-prd/ml-data/propensaodeal/avenida/trusted'
list_parquetfilepath = os.listdir(parquetfilepath_dbfs)

spark.conf.set('spark.sql.sources.partitionOverwriteMode', 'dynamic')

# COMMAND ----------

# DBTITLE 1,obtendo data do arquivo
data_escolhida = max(list_parquetfilepath)
parquetfilepath = '/mnt/ml-prd/ml-data/propensaodeal/avenida/pre-trusted'+'/'+data_escolhida+'/'
print(parquetfilepath)

data_escolhida

# COMMAND ----------

# DBTITLE 1,funcoes para este notebook
def busca_coluna_nos_dfs(coluna_buscada, dfs):
  dfs_encontrados = []
  for df in dfs:
    if coluna_buscada in dfs[df].columns:
      dfs_encontrados.append(df)
  for item in dfs_encontrados:
    print (item)
    
def getDFs():
  dfs = {'clientes_df':clientes_df,'detalhes_clientes_df':detalhes_clientes_df,'telefones_df':telefones_df,'enderecos_df':enderecos_df,'detalhes_dividas_df':detalhes_dividas_df,'contratos_df':contratos_df,'detalhes_contratos_df':detalhes_contratos_df,'dividas_df':dividas_df,'e_mails_dos_clientes_df':e_mails_dos_clientes_df}
  return dfs

def verificaColunasRepetidas(df):
  colunas_unicas = []
  for column in df.columns:
    if column not in colunas_unicas:
      colunas_unicas.append(column)
    else:
      print (column, 'repetida')
      
def todasColunas():
  dfs = getDFs()
  todas_colunas = {}
  for df in dfs:
    for col in dfs[df].columns:
      try:
        todas_colunas[col][0] = todas_colunas[col][0]+1
        todas_colunas[col][1].append(df)
      except:
        todas_colunas.update({col:[1, [df]]})
  print (len(todas_colunas), 'colunas:')
  for col in sorted(todas_colunas):
    print ('\n',col, todas_colunas[col])
  
def padronizaNomesColunas(df):
  for col in df.columns:
    str_col = ''
    for c in col:
      if c == " " or c == '-' or c == ',' or c == ';' or c == '{' or c == '}' or c == '(' or c == ')' or c == '\n' or c == '\t':
        str_col = str_col + '_'
      elif c.upper() == "Á":
        str_col = str_col + 'A'
      elif c.upper() == "É":
        str_col = str_col + 'E'
      elif c.upper() == "Í":
        str_col = str_col + 'I'
      elif c.upper() == "Ó":
        str_col = str_col + 'O'
      elif c.upper() == "Ú":
        str_col = str_col + 'U'
      elif c.upper() == "Ã":
        str_col = str_col + 'A'
      elif c.upper() == "À":
        str_col = str_col + 'A'
      elif c.upper() == "Â":
        str_col = str_col + 'A'
      elif c.upper() == "Ê":
        str_col = str_col + 'E'
      elif c == " ":
        str_col = str_col+'_'
        
      else:
        str_col = str_col + c
    str_col = str_col.upper()
    if str_col[-1] == "_":
      str_col = str_col[:-1]
    df = df.withColumnRenamed(col, str_col)
  return df

def agregaDetalhes(df, features_a_agregar):
  print (features_a_agregar)
  df_final = df.select('ID_PESSOA:ID_DIVIDA').dropDuplicates().alias('dfFinal')
  for feature in features_a_agregar:
    print (feature)
    nova_df = df.select('ID_PESSOA:ID_DIVIDA', feature)
    print (nova_df.columns)

    nova_df = nova_df.groupBy('ID_PESSOA:ID_DIVIDA').agg(F.first(F.col(feature), ignorenulls = True).alias(feature)).alias('dfNova')
    df_final = df_final.join(nova_df, F.col('dfFinal.ID_PESSOA:ID_DIVIDA')==F.col('dfNova.ID_PESSOA:ID_DIVIDA'), how='left')
    df_final = df_final.drop(F.col('dfNova.ID_PESSOA:ID_DIVIDA'))
    del nova_df
  return df_final

# COMMAND ----------

# DBTITLE 1,construindo df's
clientes_df = spark.read.parquet(parquetfilepath+'clientes.PARQUET')
detalhes_clientes_df = spark.read.parquet(parquetfilepath+'detalhes_clientes.PARQUET')
telefones_df = spark.read.parquet(parquetfilepath+'telefones.PARQUET')
enderecos_df = spark.read.parquet(parquetfilepath+'enderecos.PARQUET')
detalhes_dividas_df = spark.read.parquet(parquetfilepath+'detalhes_dividas.PARQUET')
contratos_df = spark.read.parquet(parquetfilepath+'contratos.PARQUET')
detalhes_contratos_df = spark.read.parquet(parquetfilepath+'detalhes_contratos.PARQUET')
dividas_df = spark.read.parquet(parquetfilepath+'dividas.PARQUET')
e_mails_dos_clientes_df = spark.read.parquet(parquetfilepath+'e_mails_dos_clientes.PARQUET')

# COMMAND ----------

# DBTITLE 1,contagens
dfs = {'clientes_df':clientes_df,'detalhes_clientes_df':detalhes_clientes_df,'telefones_df':telefones_df,'enderecos_df':enderecos_df,'detalhes_dividas_df':detalhes_dividas_df,'contratos_df':contratos_df,'detalhes_contratos_df':detalhes_contratos_df,'dividas_df':dividas_df ,'e_mails_dos_clientes_df':e_mails_dos_clientes_df}

for df in dfs:
  print (df, "{:,}".format(dfs[df].count()))

# COMMAND ----------

# DBTITLE 1,DROP colunas irrelevantes ou duplicadas
"""
dropando informações duplicadas ou sem valor
"""

"""clientes_df"""
clientes_df = clientes_df.drop("TIPO_DE_REGISTRO") # irrelevante
clientes_df = clientes_df.drop('TIPO_DE_PESSOA') # possui informação marcada como desconhecida
clientes_df = clientes_df.drop('SEXO') # possui informação marcada como desconhecida
clientes_df = clientes_df.drop("NOME_CEDENTE") #colunas STRING que tem informação duplicada com ID_CEDENTE
clientes_df = clientes_df.drop("NOME_/_RAZAO_SOCIAL") #info irrelevante
clientes_df = clientes_df.drop("RG") #info irrelevante
clientes_df = clientes_df.drop("ESTADO_CIVIL") #info não populada
clientes_df = clientes_df.drop('NOME_CÔNJUGE') #info irrelevante
clientes_df = clientes_df.drop('NOME_DA_MAE') #info irrelevante
clientes_df = clientes_df.drop('NOME_DO_PAI') #info irrelevante
clientes_df = clientes_df.drop('EMPRESA') #info irrelevante
clientes_df = clientes_df.drop('CARGO') #info não populada
clientes_df = clientes_df.drop('VALOR_RENDA') #info não populada
clientes_df = clientes_df.drop('ID_CEDENTE__K2') #usada info na table DIVIDAS
clientes_df = clientes_df.drop('E_MAIL') #usada info na table EMAIL
clientes_df = clientes_df.drop('NOME_CEDENTE__K2') #possui ID em outra coluna


"""detalhes_clientes_df"""
detalhes_clientes_df = detalhes_clientes_df.drop("TIPO_DE_REGISTRO")# irrelevante
detalhes_clientes_df = detalhes_clientes_df.drop("NOME_CEDENTE")#colunas STRING que tem informação duplicada com ID_CEDENTE
detalhes_clientes_df = detalhes_clientes_df.drop("ID_DETALHE_CLIENTE") # irrelevante
detalhes_clientes_df = detalhes_clientes_df.drop('ID_CEDENTE') #usada info na table DIVIDAS
detalhes_clientes_df = detalhes_clientes_df.drop('ID_DETALHE') #inutil


"""telefones_df"""
telefones_df = telefones_df.drop('TIPO_DE_REGISTRO')
telefones_df = telefones_df.drop('ID_TELEFONE_EC__PK')
telefones_df = telefones_df.drop('NOME_CEDENTE')
telefones_df = telefones_df.drop('ID_CEDENTE') #usada info na table DIVIDAS
telefones_df = telefones_df.drop('TELEFONE') #inutil
telefones_df = telefones_df.drop('RAMAL') #inutil
telefones_df = telefones_df.drop('CONTATO') #inutil

"""enderecos_df"""
enderecos_df = enderecos_df.drop('TIPO_DE_REGISTRO')
enderecos_df = enderecos_df.drop('ID_ENDEREÇO_EC')
enderecos_df = enderecos_df.drop('ID_CEDENTE')
enderecos_df = enderecos_df.drop('NOME_CEDENTE')
enderecos_df = enderecos_df.drop('ENDEREÇO')
enderecos_df = enderecos_df.drop('NUMERO')
enderecos_df = enderecos_df.drop('COMPLEMENTO')
enderecos_df = enderecos_df.drop('BAIRRO')
enderecos_df = enderecos_df.drop('CEP')

"""detalhes_dividas_df"""
detalhes_dividas_df = detalhes_dividas_df.drop('TIPO_DE_REGISTRO')
detalhes_dividas_df = detalhes_dividas_df.drop('ID_CONTRATO_EC')
detalhes_dividas_df = detalhes_dividas_df.drop('ID_DIVIDA_EC')
detalhes_dividas_df = detalhes_dividas_df.drop('ID_CEDENTE')
detalhes_dividas_df = detalhes_dividas_df.drop('NOME_CEDENTE')
detalhes_dividas_df = detalhes_dividas_df.drop('ID_DETALHE_DIVIDA')
detalhes_dividas_df = detalhes_dividas_df.drop('ID_PRODUTO') #presente em contratos_df
detalhes_dividas_df = detalhes_dividas_df.drop('ID_DETALHE') #inutil
detalhes_dividas_df = detalhes_dividas_df.drop('ID_CONTRATO') #igual a id_cliente
detalhes_dividas_df = detalhes_dividas_df.drop('ID_DIVIDA') #coluna duplicada
detalhes_dividas_df = detalhes_dividas_df.drop('ID_DIVIDA_EC') #irrelevante

"""contratos_df"""
contratos_df = contratos_df.drop('TIPO_DE_REGISTRO')
contratos_df = contratos_df.drop('ID_CEDENTE')
contratos_df = contratos_df.drop('NOME_CEDENTE')
contratos_df = contratos_df.drop('ID_CONTRATO__K2')
contratos_df = contratos_df.drop('DATA_EXPIRAÇAO') #data expiração é sempre 1 dia, info irrelevante

"""detalhes_contratos_df"""
detalhes_contratos_df = detalhes_contratos_df.drop('ID_CONTRATO_EC')
detalhes_contratos_df = detalhes_contratos_df.drop('TIPO_DE_REGISTRO')
detalhes_contratos_df = detalhes_contratos_df.drop('ID_CEDENTE')
detalhes_contratos_df = detalhes_contratos_df.drop('NOME_CEDENTE')
detalhes_contratos_df = detalhes_contratos_df.drop('ID_PRODUTO') #presente em contratos_df
detalhes_contratos_df = detalhes_contratos_df.drop('ID_DETALHE') #inutil

"""dividas_df"""
dividas_df = dividas_df.drop('TIPO_DE_REGISTRO')
dividas_df = dividas_df.drop('ID_DIVIDA') # col igual a ID_CLIENTE
dividas_df = dividas_df.drop('ID_PRODUTO') #presente em contratos_df
dividas_df = dividas_df.drop('NOME_PRODUTO') #presente em contratos_df
dividas_df = dividas_df.drop('ID_CONTRATO') #presente em contratos_df
dividas_df = dividas_df.drop('NOME_CEDENTE') 
dividas_df = dividas_df.drop('PRESTAÇAO')
dividas_df = dividas_df.drop('ID_CONTRATO_EC')
dividas_df = dividas_df.drop('ID_DIVIDA_EC__K1')

"""e_mails_dos_clientes_df"""
e_mails_dos_clientes_df = e_mails_dos_clientes_df.drop('TIPO_DE_REGISTRO')
e_mails_dos_clientes_df = e_mails_dos_clientes_df.drop('ID_E_MAIL_EC__PK') #irrevelante
e_mails_dos_clientes_df = e_mails_dos_clientes_df.drop('ID_CEDENTE')
e_mails_dos_clientes_df = e_mails_dos_clientes_df.drop('NOME_CEDENTE')
e_mails_dos_clientes_df = e_mails_dos_clientes_df.drop('CONTATO') #irrelevante


# COMMAND ----------

# DBTITLE 1,CORRIGINDO DF's para CRUZAMENTO - renomeando
"""
renomeando colunas dos dfs
"""
clientes_df = clientes_df.withColumnRenamed('CPF_/_CNPJ', 'DOCUMENTO_PESSOA')


""" ID PESSOA """
clientes_df = clientes_df.withColumnRenamed('ID_CLIENTE_EC__K1', 'ID_PESSOA')
detalhes_clientes_df = detalhes_clientes_df.withColumnRenamed('ID_CLIENTE_EC', 'ID_PESSOA')
telefones_df = telefones_df.withColumnRenamed('ID_CLIENTE_EC', 'ID_PESSOA')
contratos_df = contratos_df.withColumnRenamed('ID_CLIENTE_EC', 'ID_PESSOA')
detalhes_contratos_df = detalhes_contratos_df.withColumnRenamed('ID_CLIENTE_EC', 'ID_PESSOA')
dividas_df = dividas_df.withColumnRenamed('ID_CLIENTE_EC', 'ID_PESSOA')
detalhes_dividas_df = detalhes_dividas_df.withColumnRenamed('ID_CLIENTE_EC', 'ID_PESSOA')
e_mails_dos_clientes_df = e_mails_dos_clientes_df.withColumnRenamed('ID_CLIENTE_EC', 'ID_PESSOA')
enderecos_df = enderecos_df.withColumnRenamed('ID_CLIENTE_EC', 'ID_PESSOA')

""" CONTRATO DIVIDA """
clientes_df = clientes_df.withColumnRenamed('ID_CLIENTE__K2', 'CONTRATO_DIVIDA')
detalhes_clientes_df = detalhes_clientes_df.withColumnRenamed('ID_CLIENTE', 'CONTRATO_DIVIDA')
telefones_df = telefones_df.withColumnRenamed('ID_CLIENTE', 'CONTRATO_DIVIDA')
contratos_df = contratos_df.withColumnRenamed('ID_CLIENTE', 'CONTRATO_DIVIDA')
detalhes_contratos_df = detalhes_contratos_df.withColumnRenamed('ID_CLIENTE', 'CONTRATO_DIVIDA')
dividas_df = dividas_df.withColumnRenamed('ID_CLIENTE', 'CONTRATO_DIVIDA')
detalhes_dividas_df = detalhes_dividas_df.withColumnRenamed('ID_CLIENTE', 'CONTRATO_DIVIDA')
e_mails_dos_clientes_df = e_mails_dos_clientes_df.withColumnRenamed('ID_CLIENTE', 'CONTRATO_DIVIDA')
enderecos_df = enderecos_df.withColumnRenamed('ID_CLIENTE', 'CONTRATO_DIVIDA')
enderecos_df = enderecos_df.withColumnRenamed('DATA_CORREÇAO', 'DATA_CORRECAO')


""" ID CONTRATO DE CREDITO """
contratos_df = contratos_df.withColumnRenamed('ID_CONTRATO', 'ID_CONTRATO_CREDITO')
detalhes_contratos_df = detalhes_contratos_df.withColumnRenamed('ID_CONTRATO', 'ID_CONTRATO_CREDITO')
dividas_df = dividas_df.withColumnRenamed('ID_CONTRATO', 'ID_CONTRATO_CREDITO')
detalhes_dividas_df = detalhes_dividas_df.withColumnRenamed('ID_CONTRATO', 'ID_CONTRATO_CREDITO')

detalhes_dividas_df = detalhes_dividas_df.withColumnRenamed('ID_CONTRATO_EC', 'ID_CONTRATO')
detalhes_contratos_df = detalhes_contratos_df.withColumnRenamed('ID_CONTRATO_EC', 'ID_CONTRATO')
contratos_df = contratos_df.withColumnRenamed('ID_CONTRATO_EC__K1', 'ID_CONTRATO')

""" ID PRODUTO """
contratos_df = contratos_df.withColumnRenamed("ID_PRODUTO__K2", "ID_PRODUTO")

# COMMAND ----------

# DBTITLE 1,criando coluna documento:id_divida (PK1) e id_pessoa:id_divida (PK2)
clientes_df = clientes_df.withColumn('DOCUMENTO:ID_DIVIDA', F.concat(F.col('DOCUMENTO_PESSOA').cast(T.LongType()),F.lit(":"),F.col('CONTRATO_DIVIDA').cast(T.LongType())))
clientes_df = clientes_df.withColumn('ID_PESSOA:ID_DIVIDA', F.concat(F.col('ID_PESSOA').cast(T.LongType()),F.lit(":"),F.col('CONTRATO_DIVIDA').cast(T.LongType())))
clientes_df = clientes_df.withColumnRenamed('CONTRATO_DIVIDA','ID_DIVIDA')
def criaPK2(df):
  df = df.withColumn('ID_PESSOA:ID_DIVIDA', F.concat(F.col('ID_PESSOA').cast(T.LongType()),F.lit(":"),F.col('CONTRATO_DIVIDA').cast(T.LongType())))
  df = df.drop('ID_PESSOA')
  df = df.drop('CONTRATO_DIVIDA')
  return df

detalhes_clientes_df = criaPK2(detalhes_clientes_df)
telefones_df = criaPK2(telefones_df)
enderecos_df = criaPK2(enderecos_df)
detalhes_dividas_df = criaPK2(detalhes_dividas_df)
contratos_df = criaPK2(contratos_df)
detalhes_contratos_df = criaPK2(detalhes_contratos_df)
dividas_df = criaPK2(dividas_df)
e_mails_dos_clientes_df = criaPK2(e_mails_dos_clientes_df)

# COMMAND ----------

# DBTITLE 1,agregando e transformando df's que não de detalhes por pk - CLIENTES
clientes_df = clientes_df.groupby('ID_PESSOA:ID_DIVIDA').agg(
                                                      F.first('DOCUMENTO:ID_DIVIDA').alias('DOCUMENTO:ID_DIVIDA'),
                                                      F.first('DOCUMENTO_PESSOA').alias('DOCUMENTO_PESSOA'),
                                                      F.first('ID_DIVIDA').alias('ID_DIVIDA'),
                                                      F.first('DATA_DE_NASCIMENTO').alias('DATA_DE_NASCIMENTO'))

clientes_df = clientes_df.withColumn('DATA_DE_NASCIMENTO', F.translate(F.col('DATA_DE_NASCIMENTO'),'/','-').cast(T.DateType()))
clientes_df = clientes_df.withColumn('IDADE_PESSOA', (F.datediff(F.lit(data_escolhida).cast(T.DateType()),F.col('DATA_DE_NASCIMENTO'))/365).cast(T.IntegerType()))
clientes_df = clientes_df.drop('DATA_DE_NASCIMENTO')

# COMMAND ----------

# DBTITLE 1,agregando e transformando df's que não de detalhes por pk - TELEFONES - LISTA
telefones_df = telefones_df.withColumn('TIPO', F.col('TIPO').cast(T.IntegerType())).withColumn('DDD', F.col('DDD').cast(T.IntegerType())).withColumn('TIPO_PREFERENCIAL', F.col('TIPO_PREFERENCIAL').cast(T.IntegerType()))

telefones_df = telefones_df.orderBy('ID_PESSOA:ID_DIVIDA', F.desc('TIPO_PREFERENCIAL')).groupBy('ID_PESSOA:ID_DIVIDA').agg(F.collect_set('TIPO').alias('TIPOS_TELEFONES'),F.collect_set('DDD').alias('DDDs'))

# COMMAND ----------

# DBTITLE 1,agregando e transformando df's que não de detalhes por pk - ENDERECOS - LISTA
enderecos_df = enderecos_df.orderBy('ID_PESSOA:ID_DIVIDA', F.desc('TIPO_PREFERENCIA')).groupBy('ID_PESSOA:ID_DIVIDA').agg(F.collect_set('TIPO').alias('TIPO_ENDERECO'), F.collect_set('CIDADE').alias("CIDADE"), F.collect_set("UF").alias('UF'))

# COMMAND ----------

# DBTITLE 1,transformando  CONTRATOS para INTEGERTYPE()
contratos_df = contratos_df.withColumn('ID_CONTRATO', F.col('ID_CONTRATO').cast(T.IntegerType()))
contratos_df = contratos_df.withColumn('ID_PRODUTO', F.col('ID_PRODUTO').cast(T.IntegerType()))
contratos_df = contratos_df.withColumn('NOME_PRODUTO', F.col('NOME_PRODUTO').cast(T.IntegerType()))
contratos_df = contratos_df.withColumn('PLANO', F.col('PLANO').cast(T.IntegerType()))

# COMMAND ----------

# DBTITLE 1,transformando DIVIDAS para INTEGERTYPE e obtendo AGING
dividas_df = dividas_df.withColumn('ID_CEDENTE', F.col('ID_CEDENTE').cast(T.IntegerType()))
dividas_df = dividas_df.withColumn('VALOR_DIVIDA', F.col('VALOR_DIVIDA').cast(T.FloatType()))

dividas_df = dividas_df.withColumn('DATA_DE_VENCIMENTO', F.translate(F.col('DATA_DE_VENCIMENTO'),'/','-').cast(T.DateType()))
dividas_df = dividas_df.withColumn('AGING', F.datediff(F.lit(data_escolhida).cast(T.DateType()),F.col('DATA_DE_VENCIMENTO')))
dividas_df = dividas_df.drop('DATA_DE_VENCIMENTO')

dividas_df = dividas_df.withColumn('VALOR_CORREÇAO', F.col('VALOR_CORREÇAO').cast(T.FloatType())).withColumnRenamed('VALOR_CORREÇAO', 'VALOR_CORRECAO')
dividas_df = dividas_df.withColumn('VALOR_MINIMO', F.col('VALOR_MINIMO').cast(T.FloatType()))

# COMMAND ----------

# DBTITLE 1,agregando e transformando df's que não de detalhes por pk - E-MAILS - TIPO LISTA
e_mails_dos_clientes_df = e_mails_dos_clientes_df.withColumn('DOMINIO_EMAIL', F.split(F.col('E_MAIL'),'@').getItem(1))
e_mails_dos_clientes_df = e_mails_dos_clientes_df.drop('E_MAIL')
e_mails_dos_clientes_df = e_mails_dos_clientes_df.withColumn('DOMINIO_EMAIL', F.lower(F.col('DOMINIO_EMAIL')))
e_mails_dos_clientes_df = e_mails_dos_clientes_df.orderBy('ID_PESSOA:ID_DIVIDA', F.desc('TIPO_PREFERENCIAL')).groupBy('ID_PESSOA:ID_DIVIDA').agg(F.collect_set('TIPO').alias('TIPO_EMAIL'), F.collect_set('DOMINIO_EMAIL').alias('DOMINIO_EMAIL'))

# COMMAND ----------

# DBTITLE 1,definindo ALIAS para df's
clientes_df = clientes_df.alias('clientes')
detalhes_clientes_df = detalhes_clientes_df.alias('detalhes_clientes')
telefones_df = telefones_df.alias('telefones')
enderecos_df = enderecos_df.alias('enderecos')
detalhes_dividas_df = detalhes_dividas_df.alias('detalhes_dividas')
contratos_df = contratos_df.alias('contratos')
detalhes_contratos_df = detalhes_contratos_df.alias('detalhes_contratos')
dividas_df = dividas_df.alias('dividas')
e_mails_dos_clientes_df = e_mails_dos_clientes_df.alias('e-mails')

# COMMAND ----------

# DBTITLE 1,join atraves do PK2 de todas as df's
df = clientes_df.join(dividas_df, F.col('clientes.ID_PESSOA:ID_DIVIDA')==F.col('dividas.ID_PESSOA:ID_DIVIDA'), how='left').drop(F.col('dividas.ID_PESSOA:ID_DIVIDA')).alias('df')

df = df.join(contratos_df, F.col('df.ID_PESSOA:ID_DIVIDA')==F.col('contratos.ID_PESSOA:ID_DIVIDA'), how='left').drop(F.col('contratos.ID_PESSOA:ID_DIVIDA')).alias('df')
df = df.join(telefones_df, F.col('df.ID_PESSOA:ID_DIVIDA')==F.col('telefones.ID_PESSOA:ID_DIVIDA'), how='left').drop(F.col('telefones.ID_PESSOA:ID_DIVIDA')).alias('df')
df = df.join(enderecos_df, F.col('df.ID_PESSOA:ID_DIVIDA')==F.col('enderecos.ID_PESSOA:ID_DIVIDA'), how='left').drop(F.col('enderecos.ID_PESSOA:ID_DIVIDA')).alias('df')

df = df.join(e_mails_dos_clientes_df, F.col('df.ID_PESSOA:ID_DIVIDA')==F.col('e-mails.ID_PESSOA:ID_DIVIDA'), how='left').drop(F.col('e-mails.ID_PESSOA:ID_DIVIDA')).alias('df')

# COMMAND ----------

# DBTITLE 1,escrevendo e excluindo DF geral sem detalhes
spark.conf.set('spark.sql.sources.partitionOverwriteMode', 'dynamic')

print ('SALVANDO EM',str(writepath_stage+'/'+data_escolhida+'/'+'geral_sem_detalhes'+'.PARQUET'))
df.write.mode('overwrite').parquet(writepath_stage+'/'+data_escolhida+'/'+'geral_sem_detalhes'+'.PARQUET')

#del df, clientes_df, telefones_df, enderecos_df, contratos_df, dividas_df, e_mails_dos_clientes_df

# COMMAND ----------

# DBTITLE 1,TRATANDO DETALHES - NOMES DE COLUNAS
def corrige_nomes_colunas_detalhes(df):
  for col in df.columns:
    novoNome_col = col.replace(' ', '_')
    df = df.withColumnRenamed(col, novoNome_col)
  return df

# COMMAND ----------

# DBTITLE 1,TRATANDO DETALHES - ALIAS
detalhes_clientes_df = detalhes_clientes_df.alias('detalhes_clientes')
detalhes_dividas_df = detalhes_dividas_df.alias('detalhes_dividas')
detalhes_contratos_df = detalhes_contratos_df.alias('detalhes_contratos')

# COMMAND ----------

# DBTITLE 1,ESCREVENDO DETALHES CLIENTES
features_a_agregar = []
features_detalhes_clientes = detalhes_clientes_df.select(F.trim(F.col('NOME_DETALHE'))).dropDuplicates().rdd.map(lambda row:row[0]).collect()
for feature in features_detalhes_clientes:
  nomeColuna = 'DETALHES_CLIENTES_'+feature
  detalhes_clientes_df = detalhes_clientes_df.withColumn(nomeColuna, F.when(F.col('NOME_DETALHE').contains(feature), F.col('VALOR')).otherwise(None))
  features_a_agregar.append(nomeColuna)
  
detalhes_clientes_df = agregaDetalhes(detalhes_clientes_df,features_a_agregar)
detalhes_clientes_df = padronizaNomesColunas(detalhes_clientes_df)

print ('SALVANDO EM',str(writepath_stage+'/'+data_escolhida+'/'+'detalhes_clientes'+'.PARQUET'))
detalhes_clientes_df.write.mode('overwrite').parquet(writepath_stage+'/'+data_escolhida+'/'+'detalhes_clientes'+'.PARQUET')

#del detalhes_clientes_df

# COMMAND ----------

# DBTITLE 1,ESCREVENDO DETALHES_DIVIDAS
"""
features_a_agregar = []
features_detalhes_dividas = detalhes_dividas_df.select(F.trim(F.col('NOME_DETALHE'))).dropDuplicates().rdd.map(lambda row:row[0]).collect()
for feature in features_detalhes_dividas:
  nomeColuna = 'DETALHES_DIVIDAS_'+feature
  detalhes_dividas_df = detalhes_dividas_df.withColumn(nomeColuna, F.when(F.col('NOME_DETALHE').contains(feature), F.col('VALOR')).otherwise(None))
  features_a_agregar.append(nomeColuna)

detalhes_dividas_df = agregaDetalhes(detalhes_dividas_df,features_a_agregar)
detalhes_dividas_df = padronizaNomesColunas(detalhes_dividas_df)

print ('SALVANDO EM',str(writepath_stage+'/'+data_escolhida+'/'+'detalhes_dividas'+'.PARQUET'))
detalhes_dividas_df.write.mode('overwrite').parquet(writepath_stage+'/'+data_escolhida+'/'+'detalhes_dividas'+'.PARQUET')

#del detalhes_dividas_df
"""

# COMMAND ----------

# DBTITLE 1,ESCREVENDO DETALHES_CONTRATOS
"""
features_a_agregar = []
features_detalhes_contratos = detalhes_contratos_df.select(F.trim(F.col('NOME_DETALHE'))).dropDuplicates().rdd.map(lambda row:row[0]).collect()
for feature in features_detalhes_contratos:
  nomeColuna = 'DETALHES_CONTRATOS_'+feature
  detalhes_contratos_df = detalhes_contratos_df.withColumn(nomeColuna, F.when(F.col('NOME_DETALHE').contains(feature), F.col('VALOR')).otherwise(None))
  features_a_agregar.append(nomeColuna)
  
detalhes_contratos_df = agregaDetalhes(detalhes_contratos_df,features_a_agregar)
detalhes_contratos_df = padronizaNomesColunas(detalhes_contratos_df)

print ('SALVANDO EM',str(writepath_stage+'/'+data_escolhida+'/'+'detalhes_contratos'+'.PARQUET'))
detalhes_contratos_df.write.mode('overwrite').parquet(writepath_stage+'/'+data_escolhida+'/'+'detalhes_contratos'+'.PARQUET')

#del detalhes_contratos_df
"""