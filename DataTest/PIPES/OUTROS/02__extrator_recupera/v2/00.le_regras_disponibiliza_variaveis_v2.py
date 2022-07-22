# Databricks notebook source
# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

import py7zr
import datetime
from delta.tables import *
import json

# COMMAND ----------

blob_account_source_prd = "qqprd"
blob_account_source_ml = "qqdatastoragemain"
blob_container_source_prd = "qq-integrator"
blob_container_source_ml = "ml-prd"

mount_blob_storage_key(dbutils,blob_account_source_prd,blob_account_source_prd,'/mnt/qq-integrator')
mount_blob_storage_key(dbutils,blob_account_source_ml,blob_account_source_ml,'/mnt/ml-prd')

caminho_temp_extrator_recupera = '/mnt/ml-prd/extrator_recupera/temp'

# COMMAND ----------

# DBTITLE 1,lista de credores recupera
credores_recupera = ['agibank','bmg','fort_brasil','tribanco','trigg','valia','zema']

# COMMAND ----------

# DBTITLE 1,Classe Credor
class Credor:
  def cria_regras(self, caminho_regras):
    mapa_caracteres_especiais = {'a':
                                     ['à', 'á', 'ã', 'â', 'ä'],
                                 'e':
                                     ['è', 'é', 'ẽ', 'ê', 'ë'],
                                 'i':
                                     ['ì', 'í', 'ĩ', 'î', 'ĩ'],
                                 'o':
                                     ['ò', 'ó', 'õ', 'ô', 'õ'],
                                 'u':
                                     ['ù', 'ú', 'ũ', 'û', 'ü'],
                                 'c':
                                     ['ç']
                                 }
    
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
    df_regras = df_regras.withColumn('tipo', F.lower(F.col('tipo')))
    for character in mapa_caracteres_especiais:
      for special_character in mapa_caracteres_especiais[character]:
        df_regras = df_regras.withColumn('tipo', F.regexp_replace(F.col('tipo'), special_character, character))
    df_regras = df_regras.filter(~F.col('tipo').contains('trailer')).filter(~F.col('tipo').contains('trailler')).filter(~F.col('tipo').contains('header'))
    df_regras = df_regras.filter(F.col('inicio').isNotNull()).filter(F.col('fim').isNotNull())
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
  
  def cria_pipe(self, credor, caminho_pickle_dbfs, caminho_extrator_recupera, criar_variavel_resposta = False, modelo = 'latest'):
    pipe_full = ['01.le_arquivo_escreve_raw_v2',
                 '02.le_raw_escreve_trusted_v2',
                 '03.le_trusted_gera_base_pre_modelo_v2',
                 '04.le_pickle_escora_base',
                 '05.le_pre_output_escreve_output'
                ]
    
    try:
      modelos = {}
      for modelo_iter in os.listdir(caminho_pickle_dbfs):
        if os.path.isdir(os.path.join(caminho_pickle_dbfs,modelo_iter)):
          modelos.update({modelo_iter.split('_')[-1]:modelo_iter})
      pickle_existe = True
      if modelo == 'latest':
        modelo_escolhido = modelos[max(modelos)]
      else:
        try:
          modelo_escolhido = modelos[modelo]
        except:
          raise Exception ('modelo não existe!')        
    except Exception as e:
      pickle_existe = False
      
    if pickle_existe and not criar_variavel_resposta:
      num_pipe_steps = 5
    else:
      num_pipe_steps = 3
    
    pipe = []
    for step in range (1, num_pipe_steps+1):
      if step == 1:
       pipe.append(["./"+pipe_full[step-1]])
      elif step == 2:
       pipe.append(["./"+pipe_full[step-1]])
      elif step == 3:
       pipe.append(["./"+pipe_full[step-1], criar_variavel_resposta])
      elif step == 4:
       pipe.append(["./modelos_credores/" + credor + '/propensaodeal/' + modelo_escolhido + '/' + pipe_full[step-1], modelo_escolhido])
      elif step == 5:
       pipe.append(["./" + pipe_full[step-1]])
    return pipe
  
  def get_arquivos_no_ambiente(self, credor):
    ackey = dbutils.secrets.get(scope = "scope_qqdatastoragemain", key = "qqprd-key")
    block_blob_service = BlockBlobService(account_name='qqprd', account_key=ackey)

    prefix = "etl/"+credor+"/processed"

    generator = block_blob_service.list_blobs('qq-integrator', prefix=prefix)

    arquivos_no_ambiente = {}
    for blob in generator:   
      nome = blob.name.split('/')[-1]
      date = BlockBlobService.get_blob_properties(block_blob_service,'qq-integrator',blob.name).properties.last_modified
      size = BlockBlobService.get_blob_properties(block_blob_service,'qq-integrator',blob.name).properties.content_length/1048576
      arquivos_no_ambiente.update({nome:[date,size]})
    return arquivos_no_ambiente    
  
  def __init__(self, credor, output = False, criar_variavel_resposta = False, modelo = 'latest'):
    if credor in ['valia']:
      raise Exception (credor, 'não funciona com este extrator.')
    if modelo != 'latest':
      try:
        int(modelo)
      except:
        print ('variável modelo precisa ser data, aaaammdd')
    
    if output:
      print ('credor configurado: ',credor)
    config_credores = {
                      'agibank':
                         {'ruleset':'20180629','formato_arquivo':'zip', 'nomes_arquivos':['mailing_'], 'min_date':'2021-10-29'},
                      'bmg':
                         {'ruleset':'20130314','formato_arquivo':'txt', 'nomes_arquivos':['BT', 'T'], 'min_date':'2021-10-29'},
                      'fort_brasil':
                         {'ruleset':'20180629','formato_arquivo':'zip', 'nomes_arquivos':['BT', 'T'], 'min_date':'2021-11-03'},
                      'tribanco':
                         {'ruleset':'tribanco_corrigido','formato_arquivo':'7z', 'nomes_arquivos':['BT', 'C'], 'min_date':'2021-10-29'},
                      'trigg':
                         {'ruleset':'trigg_corrigido','formato_arquivo':'txt', 'nomes_arquivos':['T'], 'min_date':'2021-10-29'},
                      'valia':
                         {'ruleset':'20180629','formato_arquivo':'zip', 'nomes_arquivos':['D'], 'min_date':'2021-10-29'},
                      'zema':
                         {'ruleset':'20180629','formato_arquivo':'zip', 'nomes_arquivos':['BT', 'T'], 'min_date':'2021-10-29'},
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
    self.caminho_logs = '/mnt/ml-prd/ml-data/propensaodeal/'+credor+'/logs'
    self.caminho_logs_dbfs = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/'+credor+'/logs'
    
    self.caminho_pickle = '/mnt/ml-prd/ml-data/propensaodeal/'+credor+'/pickle_model'
    self.caminho_pickle_dbfs = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/'+credor+'/pickle_model'
    
    self.formato_arquivo = config_credores[credor]['formato_arquivo']
    self.nome_arquivos = config_credores[credor]['nomes_arquivos']
    self.ruleset = config_credores[credor]['ruleset']
    self.min_date = config_credores[credor]['min_date']
    
    self.caminho_regras = os.path.join('/mnt/ml-prd/extrator_recupera/regras/',str(self.ruleset))
    self.caminho_regras_dbfs = os.path.join('/dbfs/mnt/ml-prd/extrator_recupera/regras/',str(self.ruleset))
    
    self.caminho_processed = os.path.join('/mnt/ml-prd/extrator_recupera/processed/',str(self.ruleset))
    self.caminho_processed_dbfs = os.path.join('/dbfs/mnt/ml-prd/extrator_recupera/processed/',str(self.ruleset))
    
    self.df_regras = self.cria_regras(self.caminho_regras)
    self.df_regras_tipo_reg_interface = self.cria_regras_tipo_reg_interface(self.df_regras)
    self.dict_regras_sep = self.cria_dict_regras_sep(self.df_regras_tipo_reg_interface)
    
    self.caminho_extrator_recupera = '/pipe_modelos/extrator_recupera/v2/'
    
    self.pipe = self.cria_pipe(self.nome, self.caminho_pickle_dbfs, self.caminho_extrator_recupera, criar_variavel_resposta, modelo)
    
    self.criar_variavel_resposta = criar_variavel_resposta
    
    self.arquivos_no_ambiente = self.get_arquivos_no_ambiente(self.nome)

# COMMAND ----------

# DBTITLE 1,Classe ControleDelta
class ControleDelta:
  def obtem_ultima_table_escrita(self, caminho_controle):
    df_controle = spark.read.format('delta').load('/mnt/ml-prd/extrator_recupera/controle/controle.PARQUET').orderBy(F.desc(F.col('DATA_ARQUIVO')))
    delta_controle = DeltaTable.forPath(spark, caminho_controle)
    version = delta_controle.history(1).select('version').rdd.map(lambda Row:Row[0]).collect()[0]
    print ('obtendo version', version, 'da table controle')
    return (df_controle, delta_controle, version)
  
  def atualiza_necessidade_processamento(self, credor, df_controle, arquivos_a_reprocessar = False):
    df = df_controle.filter(F.col('CREDOR')==credor.nome)
    if arquivos_a_reprocessar:
      print ('reprocessando', len(arquivos_a_reprocessar), 'arquivos')
      df = df.withColumn('DEVE_SER_PROCESSADO', F.when(\
                                                      (F.col('CREDOR')==credor.nome)\
                                                      &\
                                                      (F.col('NOME_ARQUIVO').isin(arquivos_a_reprocessar))\
                                                      &\
                                                      (F.col('DATA_PROCESSAMENTO').isNull()),\
                                                      True).otherwise(F.col('DEVE_SER_PROCESSADO')))
    else:
      for nome_arquivo in credor.nome_arquivos:
        df = df.withColumn('DEVE_SER_PROCESSADO', F.when(\
                                                      (F.substring(F.col('NOME_ARQUIVO'),0,len(nome_arquivo))==nome_arquivo) \
                                                      & \
                                                      (F.col('DATA_ARQUIVO') >= credor.min_date) \
                                                      & \
                                                      (F.col('DATA_PROCESSAMENTO').isNull())
                                                      & \
                                                      (F.col('DEVE_SER_PROCESSADO')==False),
                                                      True).otherwise(False))
    return df
  
  def obtem_novos_arquivos(self, credor, df_controle):
    cred = credor.nome
    arquivos_existentes = df_controle.filter(F.col('CREDOR')==cred).select('NOME_ARQUIVO').rdd.map(lambda Row:Row[0]).collect()
    matrix = []
    dict_arquivos = get_creditor_etl_file_dates(cred)
    for arquivo in dict_arquivos:
      if arquivo not in arquivos_existentes:
        lista_interna = [
                           str(cred + ':' + arquivo),
                           cred, 
                           arquivo, 
                           dict_arquivos[arquivo],
                           datetime.datetime.today()-datetime.timedelta(hours=3),
                           False, 
                           None
                        ]
        matrix.append(lista_interna)

        schema = T.StructType([
                          T.StructField('KEY', T.StringType(), False),
                          T.StructField('CREDOR', T.StringType(), False), 
                          T.StructField('NOME_ARQUIVO', T.StringType(), False), 
                          T.StructField("DATA_ARQUIVO", T.TimestampType(), False), 
                          T.StructField('DATA_ATUALIZACAO_CONTROLE', T.TimestampType(), True),
                          T.StructField('DEVE_SER_PROCESSADO', T.BooleanType(), False), 
                          T.StructField('DATA_PROCESSAMENTO', T.TimestampType(), True)])
    try:
      df_novos_arquivos = spark.createDataFrame(spark.sparkContext.parallelize(matrix), schema=schema)
      print ("foram encontrados", df_novos_arquivos.count(), "novos arquivos para", cred)
    except:
      print ('não foram encontrados novos arquivos para', cred)
      df_novos_arquivos = None
    return df_novos_arquivos
  
  def organiza_controle(self, df):
    df = df.orderBy(F.desc(F.col('DATA_ARQUIVO')))
    return df
  
  def salva_controle(self, delta_controle, df_final, version):
    print ('escrevendo a version', version+1, 'da table controle')
    delta_controle.alias('delta').merge(df_final.alias('atual'), 'delta.KEY = atual.KEY')\
                                  .whenMatchedUpdateAll()\
                                  .whenNotMatchedInsertAll()\
                                  .execute()
    
  def insere_data_processamento(self, credor, arquivos_processados, df_controle):
    df_controle_a_atualizar = df_controle.\
                                        withColumn('DATA_PROCESSAMENTO', F.when((F.col('NOME_ARQUIVO').isin(arquivos_processados))\
                                                                                &\
                                                                                (F.col('CREDOR')==credor.nome),\
                                        F.current_timestamp()).otherwise(F.col('DATA_PROCESSAMENTO'))) # data do processamento
    
    df_controle_a_atualizar = df_controle_a_atualizar.\
                                        withColumn('DEVE_SER_PROCESSADO', F.when((F.col('NOME_ARQUIVO').isin(arquivos_processados))\
                                                                                &\
                                                                                (F.col('CREDOR')==credor.nome),\
                                        F.lit(False)).otherwise(F.col('DEVE_SER_PROCESSADO'))) # não precisa mais atualizar
    return df_controle_a_atualizar
      
    
  def popula_arquivos_para_variavel_resposta(self, credor, df_controle):
    df = df_controle.filter(F.col('CREDOR') == credor.nome)
    df = df.filter(F.col('NOME_ARQUIVO').contains(credor.nome_arquivos[0]))
    df = df.orderBy(F.desc(F.col('DATA_ARQUIVO')))
    data_hoje = datetime.date.today()
    data_hoje = str(data_hoje.year) + '-' + str(data_hoje.month).zfill(2) + '-' + str(data_hoje.day)
    df = df.withColumn('DATA_HOJE', F.lit(data_hoje))
    df = df.withColumn('datediff', F.datediff(F.col('DATA_HOJE'),F.col('DATA_ARQUIVO')))
    datediff = sorted(df.groupBy(F.col('datediff')).agg(F.first(F.col('datediff'))).rdd.map(lambda Row: Row[0]).collect())
    datediff_delta = {}
    target = 30 # dias para obter arquivo
    for day in datediff:
      diferenca = target - day
      if diferenca < 0:
        diferenca = - diferenca
      datediff_delta.update({diferenca:day})
    menor_diferenca = min(list(datediff_delta))
    ref_arquivos_a_escolher = datediff_delta[menor_diferenca]
    df_arquivos_a_processar = df.filter(F.col('datediff')==ref_arquivos_a_escolher)
    
    print('obtendo', df_arquivos_a_processar.count(), 'arquivo(s) com', ref_arquivos_a_escolher, 'dias de idade para gerar variável resposta')
    arquivos_a_processar_variavel_resposta = df_arquivos_a_processar.select("NOME_ARQUIVO").rdd.map(lambda Row: Row[0]).collect()
    data_arquivos_a_processar_variavel_resposta = str(df_arquivos_a_processar.select("DATA_ARQUIVO").rdd.map(lambda Row: Row[0]).collect()[0].date()).replace('-','')

    return (arquivos_a_processar_variavel_resposta, data_arquivos_a_processar_variavel_resposta)
    
    
  def __init__(self, credor = None, arquivos_processados = None, arquivos_a_reprocessar = None):
    """"
    credor - enviar classe Credor para realizar atualizações na table
    arquivos_processados - fim do pipe, para registrar arquivos processados
    lista_reprocessar = lista de arquivos que devem ser reprocessados
    """
    ### ORQUESTRADOR ###
    
    self.caminho_controle = '/mnt/ml-prd/extrator_recupera/controle/controle.PARQUET'
    
    #chama método para obter a table controle mais atualizada
    self.df_controle, self.delta_controle, version = self.obtem_ultima_table_escrita(self.caminho_controle)
    
    if credor == None and arquivos_processados != None:
      raise Exception ('novos processamentos devem ser acompanhados pela classe Credor, com a variavel credor = Class(credor)')
    
    elif credor != None:
      criar_variavel_resposta = credor.criar_variavel_resposta
      
      if criar_variavel_resposta == True:
        self.arquivos_a_processar_variavel_resposta, self.data_arquivos_a_processar_variavel_resposta = self.popula_arquivos_para_variavel_resposta(credor, self.df_controle)
        
      else:
        #configura possibilidade de processamento
        df_controle_a_processar = self.atualiza_necessidade_processamento(credor, self.df_controle)        
        ############### verificando novos arquivos ############################
        #obtem arquivos novos no credor
        if arquivos_processados != None:
          if type(arquivos_processados) == list:
            df_novos_processados = self.insere_data_processamento(credor, arquivos_processados, self.df_controle)
            df_novos_processados = self.organiza_controle(df_novos_processados)
            self.salva_controle(self.delta_controle, df_novos_processados, version)
          else:
            raise Exception ('arquivos_processados deve ser formato lista')

        elif arquivos_a_reprocessar != None:
          if type(arquivos_a_reprocessar) == list:
            df_a_reprocessar = self.atualiza_necessidade_processamento(credor, self.df_controle, arquivos_a_reprocessar = arquivos_a_reprocessar)
            df_a_reprocessar = self.organiza_controle(df_a_reprocessar)
            self.salva_controle(self.delta_controle, df_a_reprocessar, version)
          else:
            raise Exception ('arquivos_a_reprocessar deve ser formato lista')          
            
        else:
          df_novos_arquivos = self.obtem_novos_arquivos(credor, self.df_controle)

          if df_novos_arquivos != None:
            #configura possibilidade de processamento - NOVOS ARQUIVOS
            df_novos_arquivos_a_processar = self.atualiza_necessidade_processamento(credor, df_novos_arquivos)    

            #une os arquivos
            df_final = df_controle_a_processar.union(df_novos_arquivos_a_processar)
            #organiza a união
            df_final = self.organiza_controle(df_final)

            self.salva_controle(self.delta_controle, df_final, version)
          else:
            if df_controle_a_processar.collect() != self.df_controle.collect(): # verifica se dataframe é diferente do original (ex: nao é mais necessario processar algum item)
              df_final = self.organiza_controle(df_controle_a_processar)
              self.salva_controle(self.delta_controle, df_final, version)
              #salva a table controle



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

#def disponibiliza_arquivos(credor, arquivos_por_credor):
#  for arquivo in arquivos_por_credor:
#    caminho_arquivo_base = os.path.join(credor.caminho_base, arquivo)
#    if arquivo.split('.')[-1].lower() in ['txt', 'csv']:
#      dbutils.fs.cp(caminho_arquivo_base, os.path.join(credor.caminho_temp, arquivo))
#    elif arquivo.split('.')[-1].lower() in ['7z', '7zip']:
#      print('Procesando arquivo: '+arquivo)
#      dbutils.fs.mkdirs(os.path.join(credor.caminho_temp_dbfs, 'un'))
#      with py7zr.SevenZipFile(os.path.join(credor.caminho_base_dbfs,arquivo), mode = 'r') as compressedfile:
#        nome_deszipado = '.'.join(arquivo.split('.')[:-1])
#        compressedfile.extractall(os.path.join(credor.caminho_temp_dbfs, arquivo))
#        for file in dbutils.fs.ls(os.path.join(credor.caminho_temp, arquivo)):
#          dbutils.fs.cp(file.path, os.path.join(credor.caminho_temp,nome_deszipado+'_un'))
#          dbutils.fs.rm(os.path.join(credor.caminho_temp,arquivo), True)
#          dbutils.fs.mv(os.path.join(credor.caminho_temp,nome_deszipado+'_un'), os.path.join(credor.caminho_temp,nome_deszipado))
#    else:
#      raise Exception ('tipo de arquivo não configurado em disponibiliza_arquivos')
#

# COMMAND ----------

def disponibiliza_arquivos(credor, arquivos_por_credor):
  for arquivo in arquivos_por_credor:
    caminho_arquivo_base = os.path.join(credor.caminho_base, arquivo)
    if arquivo.split('.')[-1].lower() in ['txt', 'csv']:
      dbutils.fs.cp(caminho_arquivo_base, os.path.join(credor.caminho_temp, arquivo))
    elif arquivo.split('.')[-1].lower() in ['7z', '7zip']:
      print('Procesando arquivo: '+arquivo)
      dbutils.fs.mkdirs(os.path.join(credor.caminho_temp_dbfs, 'un'))
      try:
        with py7zr.SevenZipFile(os.path.join(credor.caminho_base_dbfs,arquivo), mode = 'r') as compressedfile:
          nome_deszipado = '.'.join(arquivo.split('.')[:-1])
          compressedfile.extractall(os.path.join(credor.caminho_temp_dbfs, arquivo))
          for file in dbutils.fs.ls(os.path.join(credor.caminho_temp, arquivo)):
            dbutils.fs.cp(file.path, os.path.join(credor.caminho_temp,nome_deszipado+'_un'))
            dbutils.fs.rm(os.path.join(credor.caminho_temp,arquivo), True)
            dbutils.fs.mv(os.path.join(credor.caminho_temp,nome_deszipado+'_un'), os.path.join(credor.caminho_temp,nome_deszipado))
      except:
        import zipfile
        with zipfile.ZipFile(os.path.join(credor.caminho_base_dbfs,arquivo), mode = 'r') as compressedfile:
          nome_deszipado = '.'.join(arquivo.split('.')[:-1])
          compressedfile.extractall(os.path.join(credor.caminho_temp_dbfs, arquivo))
          for file in dbutils.fs.ls(os.path.join(credor.caminho_temp, arquivo)):
            dbutils.fs.cp(file.path, os.path.join(credor.caminho_temp,nome_deszipado+'_un'))
            dbutils.fs.rm(os.path.join(credor.caminho_temp,arquivo), True)
            dbutils.fs.mv(os.path.join(credor.caminho_temp,nome_deszipado+'_un'), os.path.join(credor.caminho_temp,nome_deszipado))
    else:
      raise Exception ('tipo de arquivo não configurado em disponibiliza_arquivos')

# COMMAND ----------

def preemptive_transform(credor, key):
  def limpa_temp(): ### para limpar os arquivos temporários na pasta
    for item in dbutils.fs.ls('/mnt/ml-prd/extrator_recupera/temp'):
      for file in dbutils.fs.ls(os.path.join('/mnt/ml-prd/extrator_recupera/temp', item.name)):
        try:
          dbutils.fs.rm(os.path.join('/mnt/ml-prd/extrator_recupera/temp', item.name, file.name), True)
        except:
          pass
      try:
        dbutils.fs.rm(os.path.join('/mnt/ml-prd/extrator_recupera/temp', item.name), True)
      except:
        pass
      
  limpa_temp()
  import datetime
  ################# lendo codepath ###############################
  codepath = '/dbfs/mnt/ml-prd/extrator_recupera/current_model_transformation.txt'
  
  ################# lendo dataframe e criando hash ###############
  df = spark.read.option('sep',';').option('header','True').csv(os.path.join(credor.caminho_joined_trusted,'pre_output.csv'))
  df = df.withColumn('HASH', F.lit('H-'))
  columns = df.columns
  for c in range (0, len(columns)):
    df = df.withColumn(columns[c], F.when(F.col(columns[c]).cast(T.StringType()).isNull(),F.lit('0000')).otherwise(F.col(columns[c]).cast(T.StringType())))
    df = df.withColumn('HASH', F.concat(F.col('HASH'), F.col(columns[c])))
  df = df.withColumn('HASH', F.sha2(F.col('HASH'), 512))
  
  with open (codepath, 'r') as code:
  ############### construindo lista com linhas do código ################
    dict_campos_tipos = {}
    linhas_codigo_transformacao = []
    for row in code:
      row = row.replace('\n','') # tirando o pulo de linha
      linhas_codigo_transformacao.append(row)
    
  ############### construindo entendimento de features do modelo ########
  for l in range (0, len(linhas_codigo_transformacao)):
    if "pd.read_csv" in linhas_codigo_transformacao[l]:
      break
  
  ############## obtendo datatypes do código #############################
  for row in linhas_codigo_transformacao:
    # tratando astype
    if 'astype' in row:
      chave =  row.split("=")[0].split("'")[1]
      valor =  row.split('.astype(')[-1].replace('np.','').replace(')','')
      if valor == 'int':
        valor = T.IntegerType()
      elif valor == 'int64':
        valor = T.LongType()
      elif valor == 'float':
        valor = T.FloatType()
      elif valor == 'bool':
        valor = T.BooleanType()
      else:
        raise Exception ('tipo de variável não configurado em preemptive_transform()')
      dict_campos_tipos.update({chave:valor})
    #tratando pd.to_
    elif 'to_' in row:
      valor = row.split('.')[1].split('(')[0]
      if valor == 'to_datetime' or valor == 'to_timestamp':
        chave = row.split('.')[1].split('(')[1].replace("'", '').split('[')[1].replace(']', '').replace(')','')
        valor = T.DateType()
        dict_campos_tipos.update({chave:valor})
        
  ################### convertendo dataframe para o datatype esperado, senão 0 ou None ######
  ################### se não houver datatype esperado, faz select com hash e salva ######

  df_map_features_transform = df
  
  i = 0
  for feature in df.columns:
    print ("processando a coluna",feature)
    if feature == 'HASH':
      continue
    try:
      ##### transform ######
      print ('\ttipo de dado a transformar:', dict_campos_tipos[feature])
      if dict_campos_tipos[feature] != T.DateType():
        df_feature = df.select('HASH', feature).filter(F.col(feature)!='').withColumn(feature, F.col(feature).cast(dict_campos_tipos[feature]))
        df_feature = df_feature.withColumn(feature, F.when(F.col(feature).isNull(), 0).otherwise(F.col(feature)))
      else:
        df_feature = df.select('HASH', feature).filter(F.col(feature)!='').withColumn(feature, F.col(feature).cast(T.TimestampType())).withColumn(feature, F.col(feature).cast(T.DateType()))
        df_feature = df_feature.withColumn(feature, F.when(F.col(feature).isNull(), F.lit(None)).otherwise(F.col(feature)))
        df_feature = df_feature.withColumn(feature, F.when(F.col(feature)<'1900-01-01', F.lit(None)).otherwise(F.col(feature))) #### tratando datas inferiores a 01-01-1900

      df_feature = df_feature.alias('df_feature')
      df_feature.write.mode('overwrite').parquet(os.path.join("/mnt/ml-prd/extrator_recupera/temp", feature+'.PARQUET'))


      ##### log ############
      feature_map_iter = df_map_features_transform.select(feature).filter(F.col(feature).isNotNull()).withColumn('bool', F.col(feature).cast(dict_campos_tipos[feature]))
      feature_map_iter = feature_map_iter.withColumn('bool', F.when(F.col('bool').isNotNull(), True).otherwise(False))
      feature_map_iter = feature_map_iter.drop(feature)
      feature_map_iter = feature_map_iter.withColumn('true', F.when(F.col('bool')==True, 1).otherwise(0))
      feature_map_iter = feature_map_iter.withColumn('false', F.when(F.col('bool')==False, 1).otherwise(0))
      
      feature_map_iter = feature_map_iter.agg(F.sum('true'), F.sum('false'))\
                                          .withColumnRenamed('sum(true)', 'transform_true')\
                                          .withColumnRenamed('sum(false)', 'transform_false')\
                                          .withColumn('feature', F.lit(feature))\
                                          .withColumn('expected_dtype', F.lit(str(dict_campos_tipos[feature])))\
                                          .select('feature', 'expected_dtype', 'transform_true', 'transform_false')
      
    except Exception as e:
      ##### log ############
      print ('EXCEPTION', e)
      if feature not in dict_campos_tipos:
        print ('\t feature não tratada no código do modelo quanto a tipo.')
      feature_map_iter = spark.createDataFrame(data = [(feature, '-', '-', '-')], schema = ['feature', 'expected_dtype', 'transform_true', 'transform_false'])
      df_feature = df.select('HASH', feature)
      df_feature.write.mode('overwrite').parquet(os.path.join("/mnt/ml-prd/extrator_recupera/temp", feature+'.PARQUET'))
    if i == 0:
      feature_map = feature_map_iter
    else:
      feature_map = feature_map.union(feature_map_iter)
    i = i+1
  if feature_map.filter(F.col('transform_false')!='-').filter(F.col('transform_false')>0).count() >0:
    print ("linhas por feature que não puderam ser transformadas:")
    display(feature_map.filter(F.col('transform_false')!='-').filter(F.col('transform_false')>0))

  ##################### zerando variaveis e lendo arquivos gerados ##########################
  del df
  del dict_campos_tipos
  del linhas_codigo_transformacao
  del df_map_features_transform

  arquivos_features = {}
  for file in os.listdir('/dbfs/mnt/ml-prd/extrator_recupera/temp'):
    arquivos_features.update({file:spark.read.parquet(os.path.join('/mnt/ml-prd/extrator_recupera/temp', file))})
  
  print ('juntando output...')
  primeiroArquivo = True
  for file in arquivos_features:
    print ('\tcoluna', file)
    if primeiroArquivo == True:
      df = arquivos_features[file]
      primeiroArquivo = False
    else:
      df = df.join(arquivos_features[file], on='HASH', how = 'full')
  print (primeiroArquivo)
  ################## escrevendo arquivos ###################
  print ('escrevendo...')
  data_atual = datetime.date.today()
  data_atual = str(data_atual.year)+'-'+str(data_atual.month).zfill(2)+'-'+str(data_atual.day).zfill(2)
  print ('\tfeature map...')
  feature_map.write.mode('overwrite').parquet(os.path.join(credor.caminho_logs, data_atual+'_feature_dtype_transform_map.PARQUET'))
  print ('\tdataframe do modelo...')  
  df.drop('HASH').coalesce(1).write.mode('overwrite').option('header', 'True').option('sep',';').csv(os.path.join(credor.caminho_joined_trusted,'temp.csv'))
  for file in dbutils.fs.ls(os.path.join(credor.caminho_joined_trusted,'temp.csv')):
    if file.name.split('.')[-1] == 'csv':
      print (file)
      dbutils.fs.cp(file.path, os.path.join(credor.caminho_joined_trusted,'pre_processed_pre_output.csv'))
    else:
        dbutils.fs.rm(os.path.join(credor.caminho_joined_trusted,file.name), True)
  dbutils.fs.rm(os.path.join(credor.caminho_joined_trusted,'temp.csv'), True)
  limpa_temp()

# COMMAND ----------

def preemptive_transform(credor, modelo_escolhido):
  import datetime
  
  chaves_valores_tipo_dado = {'int':T.IntegerType(),
                            'int64':T.LongType(),
                            'float':T.FloatType(),
                            'bool':T.BooleanType(),
                            'date':T.DateType()}
  
  for file in os.listdir(os.path.join(credor.caminho_pickle_dbfs, modelo_escolhido)):
    if file.split('.')[-1]=='json':
      jsonfile = file
  try:
    jsonfile
  except:
    raise Exception ('não existem arquivos JSON de configuração para o modelo escolhido!')

  jsonrules = json.load(open(os.path.join(credor.caminho_pickle_dbfs, modelo_escolhido, jsonfile),'r'))
  input_variables = jsonrules['input_variables']
  transform_variables = jsonrules['transform_variables']
  ################# lendo codepath ###############################
  codepath = '/dbfs/mnt/ml-prd/extrator_recupera/current_model_transformation.txt'

  ################# lendo dataframe e selecionando colunas do modelo ###############
  df = spark.read.option('sep',';').option('header','True').csv(os.path.join(credor.caminho_joined_trusted,'pre_output.csv'))
  df = df.select(jsonrules['input_variables'])

  df_map_features_transform = df
  
  primeiraEscritaFeatureMap = True
  for feature in df.columns:
    if feature in transform_variables:
      try:
        tipo_dado = chaves_valores_tipo_dado[transform_variables[feature]]
      except:
        raise Exception ('tipo de variável não configurado em preemptive_transform()')
      
      if tipo_dado == T.DateType():
        df = df.withColumn(feature, F.col(feature).cast(T.TimestampType())).withColumn(feature, F.col(feature).cast(T.DateType()))
        df = df.withColumn(feature, F.when(F.col(feature).isNull(), F.lit(None)).otherwise(F.col(feature)))
        df = df.withColumn(feature, F.when(F.col(feature)<'1900-01-01', F.lit(None)).otherwise(F.col(feature))) #### tratando datas inferiores a 01-01-1900
      else:
        df = df.withColumn(feature, F.col(feature).cast(tipo_dado))
        df = df.withColumn(feature, F.when(F.col(feature).isNull(), 0).otherwise(F.col(feature)))

    ##### log ############
    if feature in transform_variables:
      feature_map_iter = df_map_features_transform.select(feature).filter(F.col(feature).isNotNull()).withColumn('bool', F.col(feature).cast(tipo_dado))
      feature_map_iter = feature_map_iter.withColumn('bool', F.when(F.col('bool').isNotNull(), True).otherwise(False))
      feature_map_iter = feature_map_iter.drop(feature)
      feature_map_iter = feature_map_iter.withColumn('true', F.when(F.col('bool')==True, 1).otherwise(0))
      feature_map_iter = feature_map_iter.withColumn('false', F.when(F.col('bool')==False, 1).otherwise(0))
      feature_map_iter = feature_map_iter.agg(F.sum('true'), F.sum('false'))\
                                        .withColumnRenamed('sum(true)', 'transform_true')\
                                        .withColumnRenamed('sum(false)', 'transform_false')\
                                        .withColumn('feature', F.lit(feature))\
                                        .withColumn('expected_dtype', F.lit(str(tipo_dado)))\
                                        .select('feature', 'expected_dtype', 'transform_true', 'transform_false')
      
    else:
      feature_map_iter = spark.createDataFrame(data = [(feature, '-', '-', '-')], schema = ['feature', 'expected_dtype', 'transform_true', 'transform_false'])
    if primeiraEscritaFeatureMap:
      feature_map = feature_map_iter
      primeiraEscritaFeatureMap = False
    else:
      feature_map = feature_map.union(feature_map_iter)
  display(feature_map)
  
  ################## escrevendo arquivos ###################
  print ('escrevendo...')
  data_atual = datetime.date.today()
  data_atual = str(data_atual.year)+'-'+str(data_atual.month).zfill(2)+'-'+str(data_atual.day).zfill(2)
  print ('\tfeature map...')
  feature_map.write.mode('overwrite').parquet(os.path.join(credor.caminho_logs, data_atual+'_feature_dtype_transform_map.PARQUET'))
  print ('\tdataframe do modelo...')  
  df.drop('HASH').coalesce(1).write.mode('overwrite').option('header', 'True').option('sep',';').csv(os.path.join(credor.caminho_joined_trusted,'temp.csv'))
  for file in dbutils.fs.ls(os.path.join(credor.caminho_joined_trusted,'temp.csv')):
    if file.name.split('.')[-1] == 'csv':
      print (file)
      dbutils.fs.cp(file.path, os.path.join(credor.caminho_joined_trusted,'pre_processed_pre_output.csv'))
    else:
        dbutils.fs.rm(os.path.join(credor.caminho_joined_trusted,file.name), True)
  dbutils.fs.rm(os.path.join(credor.caminho_joined_trusted,'temp.csv'), True)