# Databricks notebook source
# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

import py7zr

# COMMAND ----------

def regras_por_tipo(df_regras, tipo):
  df = df_regras.filter(F.col('tipo')==tipo)
  df = df.drop('desc').drop('tipo')
  regras = df.rdd.map(lambda Row:{Row[0]:Row[1:]}).collect()
  dict_regras = {}
  for regra in regras:
    dict_regras.update(regra)
  return (dict_regras)

# COMMAND ----------

blob_account_source_prd = "qqprd"
blob_account_source_ml = "qqdatastoragemain"
blob_container_source_prd = "qq-integrator"
blob_container_source_ml = "ml-prd"

mount_blob_storage_key(dbutils,blob_account_source_prd,blob_account_source_prd,'/mnt/qq-integrator')
mount_blob_storage_key(dbutils,blob_account_source_ml,blob_account_source_ml,'/mnt/ml-prd')

# COMMAND ----------

# DBTITLE 1,lista de credores recupera
credores_recupera = ['agibank','bmg','fort','tribanco','trigg','valia','zema']

# COMMAND ----------

class Credor:
  def cria_regras(self, caminho_regras):
    df_regras = spark.read.option("encoding", "UTF-8").option('header', 'True').csv(caminho_regras)
    df_regras = df_regras.filter(F.col('_c0').isNotNull())
    registros = df_regras.select('_c0').filter(F.col('_c0').contains('Registro')).rdd.map(lambda Row: Row[0]).collect()
    df_regras = df_regras.withColumn('reg', F.when(F.col('_c0').isin(registros), F.col('_c0')).otherwise(None))
    df_regras = df_regras.withColumn('temp', F.lit('temp')).withColumn('row', F.row_number().over(Window.partitionBy(F.col('temp')).orderBy(F.lit('temp')))).drop('temp')
    registros = df_regras.select('_c0', 'row').filter(F.col('_c0').contains('Registro')).rdd.map(lambda Row: [Row[0],Row[1]]).collect()
    chave_registros = {}

    for reg in registros:
      chave_registros.update({re.sub('\W+', '',reg[0]): reg[1]})

    for reg in chave_registros:
      df_regras = df_regras.withColumn('reg', F.when(F.col('row')>=chave_registros[reg], F.lit(reg)).otherwise(F.col('reg')))

    df_regras = df_regras.drop(F.col('row'))
    df_regras = df_regras.filter(~F.col('_c0').contains('Registro')).filter(~F.col('_c0').contains('Nome'))
    df_regras = df_regras.select('_c0', '_c1', '_c2', '_c3', 'reg')
    df_regras = changeColumnNames(df_regras, ['nome', 'inicio', 'fim', 'desc', 'tipo'])
    df_regras = df_regras.withColumn('tipo', F.regexp_replace(F.col('tipo'), 'Registro', 'reg_'))
    df_regras = df_regras.withColumn('tipo', F.lower(F.regexp_replace(F.col('tipo'), '\W+', '')))
    df_regras = df_regras.filter(~F.col('tipo').contains('trailer')).filter(~F.col('tipo').contains('trailler')).filter(~F.col('tipo').contains('header'))
    df_regras = df_regras.filter(F.col('inicio').isNotNull())
    return df_regras
  
  def cria_regras_tipo_reg_interface(self, df_regras):
    df_regras_tipo_reg_interface = df_regras.filter(F.col('nome').isin(['tip_reg', 'tip_inter']))
    df_regras_tipo_reg_interface = df_regras_tipo_reg_interface.withColumn('desc', F.substring(F.col('desc'),-4, 1))
    df_regras_tipo_reg_interface = df_regras_tipo_reg_interface.withColumn('tip_reg', F.when(F.col('Nome')=='tip_reg', F.col('desc')).otherwise(None))
    df_regras_tipo_reg_interface = df_regras_tipo_reg_interface.withColumn('tip_inter', F.when(F.col('Nome')=='tip_inter', F.col('desc')).otherwise(None))
    df_regras_tipo_reg_interface = df_regras_tipo_reg_interface.groupBy('tipo').agg(F.first('tip_reg'), F.last('tip_inter'))
    df_regras_tipo_reg_interface = changeColumnNames(df_regras_tipo_reg_interface, ['tipo', 'reg', 'inter'])
    return df_regras_tipo_reg_interface
  
  def cria_dict_regras_sep(self, df_regras_tipo_reg_interface):
    dict_regras_sep = {}
    regras_sep = df_regras_tipo_reg_interface.rdd.map(lambda Row:{Row[0].lower():Row[1:]}).collect()
    for regra in regras_sep:
      dict_regras_sep.update(regra)
    return dict_regras_sep
  
  def __init__(self, credor):
    if credor in ['valia']:
      raise Excception ("credor não envia arquivo!")
    
    
    print ('credor configurado: ',credor)
    config_credores = {
                      'agibank':
                         {'formato_arquivo':'zip', 'nomes_arquivos':['mailing_']},
                      'bmg':
                         {'ruleset':'20130314','formato_arquivo':'txt', 'nomes_arquivos':['BT', 'T']},
                      'fort_brasil':
                         {'ruleset':'20180629','formato_arquivo':'zip', 'nomes_arquivos':['BT_', 'T_']},
                      'tribanco':
                         {'ruleset':'tribanco_corrigido','formato_arquivo':'7z', 'nomes_arquivos':['BT_', 'C_']},
                      'trigg':
                         {'ruleset':'20180629','formato_arquivo':'zip', 'nomes_arquivos':['T_']},
                      'valia':
                         {'formato_arquivo':'zip', 'nomes_arquivos':['BT_', 'C_']},
                      'zema':
                         {'ruleset':'20180629','formato_arquivo':'zip', 'nomes_arquivos':['BT_', 'T_']},
                      }

    self.nome = credor
    
    self.prefix = "etl/"+credor+"/processed"

    self.caminho_base = '/mnt/qq-integrator/etl/'+credor+'/processed'
    self.caminho_base_dbfs = '/dbfs/mnt//qq-integrator/etl/'+credor+'/processed/'
    self.caminho_base_consolidada = '/mnt/ml-prd/ml-data/propensaodeal/'+credor+'/base_full'
    self.caminho_base_consolidada_dbfs = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/'+credor+'/base_full'
    self.caminho_temp = '/mnt/ml-prd/ml-data/propensaodeal/'+credor+'/temp'
    self.caminho_temp_dbfs = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/'+credor+'/temp'
    self.caminho_raw = '/mnt/ml-prd/ml-data/propensaodeal/'+credor+'/raw'
    self.caminho_raw_dbfs = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/'+credor+'/raw'
    self.caminho_trusted = '/mnt/ml-prd/ml-data/propensaodeal/'+credor+'/trusted'
    self.caminho_trusted_dbfs = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/'+credor+'/trusted'
    self.caminho_joined_trusted = '/mnt/ml-prd/ml-data/propensaodeal/'+credor+'/joined_trusted'
    self.caminho_joined_trusted_dbfs = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/'+credor+'/joined_trusted'
    self.caminho_sample = '/mnt/ml-prd/ml-data/propensaodeal/'+credor+'/sample'
    self.caminho_sample_dbfs = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/'+credor+'/sample'
    self.caminho_pre_output = '/mnt/ml-prd/ml-data/propensaodeal/'+credor+'/pre_output'
    self.caminho_pre_output_dbfs = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/'+credor+'/pre_output'
    self.caminho_output = '/mnt/ml-prd/ml-data/propensaodeal/'+credor+'/output'
    self.caminho_output_dbfs = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/'+credor+'/output'
    
    self.caminho_pickle = '/mnt/ml-prd/ml-data/propensaodeal/'+credor+'/pickle_model'
    self.caminho_pickle_dbfs = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/'+credor+'/pickle_model'
    
    self.formato_arquivo = config_credores[credor]['formato_arquivo']
    self.nome_arquivos = config_credores[credor]['nomes_arquivos']
    self.ruleset = config_credores[credor]['ruleset']
    
    self.caminho_regras = os.path.join('/mnt/ml-prd/extrator_recupera/regras/',str(self.ruleset))
    self.caminho_regras_dbfs = os.path.join('/dbfs/mnt/ml-prd/extrator_recupera/regras/',str(self.ruleset))
    
    self.df_regras = self.cria_regras(self.caminho_regras)
    self.df_regras_tipo_reg_interface = self.cria_regras_tipo_reg_interface(self.df_regras)
    self.dict_regras_sep = self.cria_dict_regras_sep(self.df_regras_tipo_reg_interface)
    
regras_por_tipo(Credor('tribanco').df_regras, 'reg_prestaes')

# COMMAND ----------

def descomprime_arquivo_obtem_caminho(credor, arquivo):
  if arquivo.split('.')[-1].lower() in ['txt', 'csv']:
    return os.path.join(credor.caminho_temp,arquivo)
  elif arquivo.split('.')[-1].lower() in ['7z', '7zip']:
    dbutils.fs.mkdirs(os.path.join(credor.caminho_temp_dbfs, 'un'))
    with py7zr.SevenZipFile(os.path.join(credor.caminho_temp_dbfs,arquivo), mode = 'r') as compressedfile:
      compressedfile.extractall(os.path.join(credor.caminho_temp_dbfs,'un', arquivo))
      arquivo_deszipado = os.listdir(os.path.join(credor.caminho_temp_dbfs,'un', arquivo))[0]
      return(os.path.join(credor.caminho_temp, 'un', arquivo, arquivo_deszipado))

# COMMAND ----------

display(Credor('tribanco').df_regras.filter(F.col('tipo')=='reg_prestaes'))