# Databricks notebook source
import time
time.sleep(300)

# COMMAND ----------

# MAGIC %run "/pipe_modelos/extrator_recupera/v2/00.le_regras_disponibiliza_variaveis_v2"

# COMMAND ----------

# MAGIC %md # ORQUESTRADOR

# COMMAND ----------

# MAGIC %md ### obtendo lista de arquivos a processar

# COMMAND ----------

# DBTITLE 1,selecionando credores a processar
dbutils.widgets.text('credores_a_processar', "")
credores_a_processar_widget = dbutils.widgets.get('credores_a_processar').split(',')

credores_a_processar_widget_temp = []
for credor in credores_a_processar_widget: 
  credores_a_processar_widget_temp.append(credor.replace(' ',''))
credores_a_processar_widget = credores_a_processar_widget_temp
del credores_a_processar_widget_temp

for credor in credores_a_processar_widget: 
  ControleDelta(Credor(credor.replace(' ',''), output=True))
  
dbutils.widgets.multiselect('popula_variavel_resposta_gera_sample', credores_recupera[0], credores_recupera)
credores_variavel_resposta_True = dbutils.widgets.get('popula_variavel_resposta_gera_sample').split(',')

dbutils.widgets.multiselect('processa_base_full', "", credores_recupera+[''])
credores_carga_full = dbutils.widgets.get('processa_base_full').split(',')
carga_full = {}
for credor in credores_carga_full:
  if credor != "":
    carga_full.update({credor:True})

a_processar = ControleDelta().df_controle.filter(F.col('DEVE_SER_PROCESSADO')==True)
credores_a_processar_controle = sorted(a_processar.groupBy('CREDOR').agg(F.first(F.col('CREDOR'))).rdd.map(lambda Row:Row[0]).collect())
credores_a_processar = []

for credor in credores_a_processar_widget:
  if credor in credores_a_processar_controle or credor in carga_full:
    credores_a_processar.append(credor)
    
credores_a_processar = sorted(credores_a_processar)

for credor in credores_a_processar:
  if credor not in carga_full:
    carga_full.update({credor:False})

try:
  credores_a_processar.remove('')
except:
  pass
credores_a_processar

# COMMAND ----------

# DBTITLE 1,selecionando arquivos por credor para serem processados até o fim da run do orchestrator
arquivos_por_credor = {}
datas_variavel_resposta = {}
for credor in credores_a_processar:
  print (credor)
  if credor in credores_variavel_resposta_True:
    print ('obtendo variável resposta para', credor)
    credor = Credor(credor, criar_variavel_resposta = True)
    controle = ControleDelta(credor)
    arquivos_a_processar_variavel_resposta = controle.arquivos_a_processar_variavel_resposta
    data_arquivos_a_processar_variavel_resposta = controle.data_arquivos_a_processar_variavel_resposta
    arquivos_por_credor.update({credor.nome:arquivos_a_processar_variavel_resposta})
    datas_variavel_resposta.update({credor.nome:data_arquivos_a_processar_variavel_resposta})
  else:  
    for credor in carga_full:
      if carga_full[credor]:
        arquivos_por_credor.update({credor:ControleDelta().df_controle.filter(F.col('CREDOR')==credor).select(F.col('NOME_ARQUIVO')).rdd.map(lambda Row: Row[0]).collect()})
      else:
        arquivos_por_credor.update({credor:a_processar.filter(F.col('CREDOR')==credor).select(F.col('NOME_ARQUIVO')).rdd.map(lambda Row: Row[0]).collect()})

# COMMAND ----------

#arquivos_por_credor = {}
#for credor in credores_a_processar:
#  for credor in carga_full:
#    arquivos_por_credor.update({credor:ControleDelta().df_controle.filter((F.col('CREDOR')==credor) & ((F.col('DATA_ARQUIVO')>="2021-12-14 00:00:00") & (F.col('DATA_ARQUIVO')<="2022-01-15 00:00:00")) ).select((F.col('NOME_ARQUIVO'))).rdd.map(lambda Row: Row[0]).collect()})

# COMMAND ----------

arquivos_por_credor

# COMMAND ----------

# DBTITLE 1,SUMÁRIO DE EXECUÇÃO
matriz_sumario = []
for credor in credores_a_processar:
  submatrix = [credor, 'X']
  if credor in credores_variavel_resposta_True:
    submatrix.append('X')
  else:
    submatrix.append('-')
  if credor in carga_full:
    submatrix.append('X')
  else:
    submatrix.append('-')
  submatrix.append(len(arquivos_por_credor[credor]))
  matriz_sumario.append(submatrix)
try:
  df_matriz_sumario = changeColumnNames(spark.createDataFrame(sc.parallelize(matriz_sumario)), ['credor', 'processar', 'variavel_resposta_true', 'processa_base_full', 'qtd_arquivos'])
  display(df_matriz_sumario)
except:
  pass

# COMMAND ----------

# DBTITLE 1,SUMÁRIO DE ARQUIVOS A PROCESSAR
try: 
  max_file_amount = df_matriz_sumario.agg(F.max(F.col('qtd_arquivos'))).rdd.map(lambda Row:Row[0]).collect()[0]

  matriz_arquivos = []
  for i in range (0, max_file_amount):
    row = []
    for credor in arquivos_por_credor:
      try:
        row.append(arquivos_por_credor[credor][i])
      except Exception:
        row.append('')
    matriz_arquivos.append(row)
     
  matriz_arquivos = changeColumnNames(spark.createDataFrame(sc.parallelize(matriz_arquivos)), [credor for credor in arquivos_por_credor])
  display(matriz_arquivos)
except:
  pass

# COMMAND ----------

if len(list(arquivos_por_credor)) == 0:
  dbutils.notebook.exit('N/E')

# COMMAND ----------

# MAGIC %md ### disponibiliza arquivos

# COMMAND ----------

# DBTITLE 1,excluir arquivos temporários do ambiente
def exclui_temporarios_ambiente(caminho_a_excluir):
  try:
    for file in dbutils.fs.ls(caminho_a_excluir):
      dbutils.fs.rm(os.path.join(caminho_a_excluir, file.name), True)
      dbutils.fs.rm(caminho_a_excluir, recurse = True)
  except Exception as e:
    pass
  print ('excluída pasta', caminho_a_excluir)

# COMMAND ----------

# DBTITLE 1,orquestrador de arquivos - seleciona arquivos, disponibiliza e retorna processados e faltantes
def obtem_batch_arquivos_a_processar(credor, arquivos_a_processar, megabytes_threshold = 2000, max_files = 50, carga_full = False):
  caminhos_a_excluir = [credor.caminho_temp,
                        credor.caminho_raw,
                        credor.caminho_trusted,
                        credor.caminho_joined_trusted,
                        credor.caminho_pre_output,
                        caminho_temp_extrator_recupera]
  
  for caminho_a_excluir in caminhos_a_excluir:
    exclui_temporarios_ambiente(caminho_a_excluir)
  
  #verificando quais arquivos disponibilizar
  file_date_size = {}
  list_dates = []
  tamanho_arquivos_total = 0
  for file in arquivos_a_processar:
    file_date_size.update({file:credor.arquivos_no_ambiente[file]})  
    list_dates.append(credor.arquivos_no_ambiente[file][0])
    tamanho_arquivos_total = tamanho_arquivos_total+credor.arquivos_no_ambiente[file][1]
  list_dates = sorted(list_dates)
  temp_list_dates = []
  for item in list_dates:
    if item not in temp_list_dates:
      temp_list_dates.append(item)
  list_dates = temp_list_dates
  del temp_list_dates
  lista_a_processar = []
  lista_espera = []
  tamanho_arquivos_a_processar = 0 # em mb
  
  for date in list_dates:
    for file in file_date_size:
      if file_date_size[file][0]==date:
        if carga_full:
          lista_a_processar = arquivos_a_processar
        else:
          if tamanho_arquivos_a_processar <= megabytes_threshold and len(lista_a_processar) < max_files: # um gigabyte de processamento
            lista_a_processar.append(file)
            tamanho_arquivos_a_processar = tamanho_arquivos_a_processar + file_date_size[file][1]
  for file in arquivos_a_processar:
    if file not in lista_a_processar:
      lista_espera.append(file)
  
  if len(lista_a_processar) != 0:
    print ('serão processados', len(lista_a_processar), 'arquivos, com ', len(lista_espera), 'em espera.\n\t',round(tamanho_arquivos_a_processar, 2), 'MB a processar neste batch.\n\t',round(tamanho_arquivos_total-tamanho_arquivos_a_processar,2),'MB faltantes.')
    print ("\tdisponibilizando",len(lista_a_processar), "arquivos...")
  disponibiliza_arquivos(credor, lista_a_processar)
  if len(lista_a_processar) == 1:
    print ('arquivo a processar:', lista_a_processar[0])
  elif len(lista_a_processar) >1:
    print ('arquivos a processar:', lista_a_processar)
  return (lista_a_processar, lista_espera)

# COMMAND ----------

# MAGIC %md ### chamando notebooks e processando

# COMMAND ----------

# DBTITLE 1,orquestrador principal
for credor in credores_a_processar:
  print ('processando credor', credor)
  
  if credor in credores_variavel_resposta_True:
    print ('\tconfigurando busca de variável resposta')
    credor = Credor(credor, criar_variavel_resposta = True)
  else:
    credor = Credor(credor)
  pipe = credor.pipe
  print (pipe)
  
  # obtendo arquivos
  lista_a_processar, lista_espera = obtem_batch_arquivos_a_processar(credor, arquivos_por_credor[credor.nome], carga_full = carga_full[credor.nome])
  while(len(lista_a_processar) != 0):    
    ### inserindo arquivos no pipe1
    for step in range (1, len(pipe)+1):
      append_command = {'credor': credor.nome}
      if step in [3, 4]: #estes dois steps recebem informações adicionais por dbutils
          append_command_list = pipe[step-1][1]
          if step == 3:
              if append_command_list:
                  append_command = {**append_command, **{'criar_variavel_resposta':'[True,'+datas_variavel_resposta[credor.nome]+']'}}
              else:
                  append_command = {**append_command, **{'criar_variavel_resposta':'False'}}
          elif step == 4:
              append_command = {**append_command, **{'modelo_escolhido':append_command_list}}
        
        ### rodando steps ###
      print ('\tSTEP',step)
      print ('\t\t@', credor.pipe[step-1][0], end = ' ')
      end_note = dbutils.notebook.run(credor.pipe[step-1][0], 0, append_command)
      print ("- ",end_note)
    lista_processada = lista_a_processar # mudando nome de variavel pelo arquivo ter sido processado
    ControleDelta(credor, arquivos_processados = lista_processada)
    lista_a_processar, lista_espera = obtem_batch_arquivos_a_processar(credor, lista_espera, carga_full = carga_full[credor.nome])

# COMMAND ----------

# DBTITLE 1,Para reconfigurar arquivos processados por credor

#ControleDelta(Credor('fort_brasil'),arquivos_a_reprocessar = ["BT00000919QQUITAR.TXT"])


# COMMAND ----------

"""
ControleDelta(Credor('fort_brasil'), arquivos_processados=["T00000733QQUITAR.TXT"])
"""

# COMMAND ----------

dbutils.notebook.exit('OK')