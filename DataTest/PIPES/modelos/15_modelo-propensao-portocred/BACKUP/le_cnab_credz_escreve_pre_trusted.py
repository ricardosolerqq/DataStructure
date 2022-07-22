# Databricks notebook source
import os
import csv
from azure.storage.blob import BlockBlobService
import zipfile

# COMMAND ----------

# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/mount-repository"

# COMMAND ----------

blob_account_source_prd = "qqprd"
blob_account_source_ml = "qqdatastoragemain"
blob_container_source_prd = "qq-integrator"
blob_container_source_ml = "ml-prd"

prefix = "etl/credz/processed"

mount_blob_storage_key(dbutils,blob_account_source_prd,blob_account_source_prd,'/mnt/qq-integrator')
mount_blob_storage_key(dbutils,blob_account_source_ml,blob_account_source_ml,'/mnt/ml-prd')

caminho_base = '/dbfs/mnt/qq-integrator/etl/credz/processed'
list_caminho_base = os.listdir(caminho_base)
caminho_temp = "/mnt/temp"
parquetfilepath = '/mnt/ml-prd/ml-data/propensaodeal/credz/pre-trusted'

# COMMAND ----------

# DBTITLE 1,EXCLUINDO ARQUIVOS INTERNOS
"""
try:
  list_caminho_temp = os.listdir(caminho_temp)
  for file in list_caminho_temp:
    os.remove(os.path.join(caminho_temp,file))
except:
  pass
"""

# COMMAND ----------

# DBTITLE 1,escolhe arquivo mais recente
"""
arquivos_batimento = {}
listas_datas = []
for file in list_caminho_base:
  if 'Batimento' in file:
    data_arq = getBlobFileDate(blob_account_source, blob_container_source, file, prefix = prefix, str_scope = "scope_qqdatastoragemain", str_key = "qqprd-key")
    listas_datas.append(data_arq)
    arquivos_batimento.update({file:data_arq})

max_data = max(listas_datas)
for f in arquivos_batimento:
  if arquivos_batimento[f]==max_data:
    arquivo_escolhido = f
"""    

# COMMAND ----------

arquivo_escolhido = "20210416-Batimento.zip"
data_arq_escolhido = getBlobFileDate(blob_account_source_prd, blob_container_source_prd, arquivo_escolhido, prefix = prefix, str_scope = "scope_qqdatastoragemain", str_key = "qqprd-key").date()
data_arq_escolhido = data_arq_escolhido.strftime('%Y-%m-%d')
data_arq_escolhido

# COMMAND ----------

arquivo_escolhido_path = os.path.join(caminho_base, arquivo_escolhido)
print (arquivo_escolhido_path)
with zipfile.ZipFile(arquivo_escolhido_path,"r") as zip_ref:
    zip_ref.extractall(caminho_temp)

dir_extract_list = os.listdir(caminho_temp)
file_extract = dir_extract_list[0]
dir_extract_list = os.listdir(caminho_temp)
file_extract = dir_extract_list[0]
file_extract_path = os.path.join(caminho_temp, file_extract)
file_extract_path

# COMMAND ----------

# DBTITLE 1,URL COM REGRAS EM SHEETS
"""

https://docs.google.com/spreadsheets/d/1ZSXOSvVBcBHEcTQg88vh6PFIi5nTg9EQhGyDmzZaHOA/edit#gid=1545203801

"""

# COMMAND ----------

# DBTITLE 1,VARIÁVEIS COM REGRAS
  clientes = {'Tipo de Registro':2,
              'ID Cliente EC (K1)':10,
              'ID Cedente (K2)':10,
              'ID Cliente (K2)':50,
              'Nome Cedente (K2)':50,
              'Nome / Razão Social':50,
              'CPF / CNPJ':14,
              'RG':15,
              'Data de Nascimento':10,
              'Sexo':1,
              'E-mail':100,
              'Estado Civil':1,
              'Nome Cônjuge':50,
              'Nome da Mãe':50,
              'Nome do Pai':50,
              'Empresa':50,
              'Cargo':50,
              'Valor Renda':15,
              'Tipo de Pessoa':1}

  clientes_file = os.path.join(caminho_temp, 'clientes.csv')
  
  telefones = {'Tipo de Registro':2,
              'ID Telefone EC (PK)':10,
              'ID Cliente EC':10,
              'ID Cedente':10,
              'ID Cliente':50,
              'Nome Cedente':50,
              'Tipo':1,
              'DDD':2,
              'Telefone':12,
              'Ramal':5,
              'Contato':50,
              'Tipo Preferencial':1}
  
  telefones_file = os.path.join(caminho_temp, 'telefones.csv')
  
  enderecos = {'Tipo de Registro':2,
              'ID Endereço EC':10,
              'ID Cliente EC':10,
              'ID Cedente':10,
              'ID Cliente':50,
              'Nome Cedente':50,
              'Tipo':1,
              'Endereço':100,
              'Número':10,
              'Complemento':100,
              'Bairro':100,
              'Cidade':100,
              'UF':2,
              'CEP':8,
              'Tipo Preferencia':1}
  
  enderecos_file = os.path.join(caminho_temp, 'enderecos.csv')

  detalhes = {'Tipo de Registro':2,
              'ID Cliente EC':10,
              'ID Cedente':10,
              'ID Cliente':50,
              'Nome Cedente':50,
              'ID Detalhe Cliente':10,
              'ID Detalhe':10,
              'Nome Detalhe':50,
              'Tipo':1,
              'Valor':500}

  detalhes_file = os.path.join(caminho_temp, 'detalhes.csv')

  
  contratos = {'Tipo de Registro':2,
              'ID Cliente EC':10,
              'ID Cedente':10,
              'ID Cliente':50,
              'Nome Cedente':50,
              'ID Contrato EC (K1)':10,
              'ID Contrato (K2)':50,
              'ID Produto (K2)':10,
              'Nome Produto':50,
              'Plano':5,
              'Data Expiração':10}
  
  contratos_file = os.path.join(caminho_temp, 'contratos.csv')

  
  detalhes_contratos = {'Tipo de Registro':2,
                        'ID Cliente EC':10,
                        'ID Cedente':10,
                        'ID Cliente':50,
                        'Nome Cedente':50,
                        'ID Contrato EC':10,
                        'ID Contrato':50,
                        'ID Produto':10,
                        'Nome Produto':50,
                        'ID Detalhe Contrato':10,
                        'ID Detalhe':10,
                        'Nome Detalhe':50,
                        'Tipo':1,
                        'Valor':500}
  
  detalhes_contratos_file = os.path.join(caminho_temp, 'detalhes_contratos.csv')
  
  dividas = {'Tipo de Registro':2,
                        'ID Dívida EC (K1)':20,
                        'ID Cliente EC':10,
                        'ID Cedente':10,
                        'ID Cliente':50,
                        'Nome Cedente':50,
                        'ID Contrato EC':10,
                        'ID Contrato':50,
                        'ID Produto':10,
                        'Nome Produto':50,
                        'ID Dívida':50,
                        'Valor Dívida':15,
                        'Data de Vencimento':10,
                        'Prestação':5,
                        'Data Correção':10,
                        'Valor Correção':15,
                        'Valor Mínimo':15}
  
  dividas_file = os.path.join(caminho_temp, 'dividas.csv')
  
  detalhes_dividas = {'Tipo de Registro':2,
                      'ID Dívida EC':20,
                      'ID Cliente EC':10,
                      'ID Cedente':10,
                      'ID Cliente':50,
                      'Nome Cedente':50,
                      'ID Contrato EC':10,
                      'ID Contrato':50,
                      'ID Produto':10,
                      'Nome Produto':50,
                      'ID Dívida':50,
                      'ID Detalhe Dívida':20,
                      'ID Detalhe':10,
                      'Nome Detalhe':50,
                      'Tipo':1,
                      'Valor':500}
  
  detalhes_dividas_file = os.path.join(caminho_temp, 'detalhes_dividas.csv')
  
  
"""  retirado por CREDZ não ter esse tipo e ser o mesmo código de tipo de registro do detalhes_contratos  
garantias_mercadorias = {'Tipo de Registro':2,
                            'ID Cliente EC':10,
                            'ID Cedente':10,
                            'ID Cliente':50,
                            'Nome Cedente':50,
                            'ID Contrato EC':10,
                            'ID Contrato':50,
                            'ID Produto':10,
                            'Nome Produto':50,
                            'ID Garantia':10,
                            'Garantia Cedente':50,
                            'Nome':50,
                            'Descrição':500,
                            'Valor':15,
                            'Data Compra':10}
  """
  e_mails_dos_clientes = {'Tipo de Registro':2,
                            'ID E-mail EC (PK)':10,
                            'ID Cliente EC':10,
                            'ID Cedente':10,
                            'ID Cliente':50,
                            'Nome Cedente':50,
                            'Tipo':1,
                            'E-mail':100,
                            'Contato':50,
                            'Tipo Preferencial':1}
  
  e_mails_dos_clientes_file = os.path.join(caminho_temp, 'emails_dos_clientes.csv')

# COMMAND ----------

def separaPorTipoRegistro(l, tipo_registro, primeiraLinha=False):
  if tipo_registro == '01':
    tipo_registro = clientes
    arquivo_escrita = clientes_file
  elif tipo_registro == '02':
    tipo_registro = telefones
    arquivo_escrita = telefones_file
  elif tipo_registro == '03':
    tipo_registro = enderecos
    arquivo_escrita = enderecos_file
  elif tipo_registro == '04':
    tipo_registro = detalhes
    arquivo_escrita = detalhes_file
  elif tipo_registro == '05':
    tipo_registro = contratos
    arquivo_escrita = contratos_file
  elif tipo_registro == '06':
    tipo_registro = detalhes_contratos
    arquivo_escrita = detalhes_contratos_file
  elif tipo_registro == '07':
    tipo_registro = dividas
    arquivo_escrita = dividas_file
  elif tipo_registro == '08':
    tipo_registro = detalhes_dividas
    arquivo_escrita = detalhes_dividas_file
  elif tipo_registro == '10':
    tipo_registro = e_mails_dos_clientes
    arquivo_escrita = e_mails_dos_clientes_file
  else:
    tipo_registro = None
    arquivo_escrita = clientes_file
  
  if tipo_registro != None:
    lista_a_escrever = []
    if primeiraLinha == True:
      for i in tipo_registro:
        lista_a_escrever.append(i)
    else:
      indice_inicial = 0
      for i in tipo_registro:
        indice_final = indice_inicial + int(tipo_registro[i])
        lista_a_escrever.append(l[indice_inicial:indice_final])
        indice_inicial = indice_final
    with open(arquivo_escrita, 'a') as arqescrita:
      escritor = csv.writer(arqescrita)
      escritor.writerow(lista_a_escrever)
    
    

# COMMAND ----------

tipos_registros_escritos = [] # lista de tipos de registros ja escritos

file_write_path = os.path.join(caminho_temp, 'output.csv')

with open (file_extract_path, 'r', encoding = 'latin-1') as openfile:
  lines = openfile.readlines() #corrigir 1024 para trazer tudo
  for l in lines:
    print(l)
    input()
    tipo_registro = l[0:2]
    if tipo_registro in tipos_registros_escritos:
      primeiraEscrita = False
    else:
      tipos_registros_escritos.append(tipo_registro)
      primeiraEscrita = True
      print (tipo_registro)
    separaPorTipoRegistro(l, tipo_registro, primeiraLinha = primeiraEscrita)
      

# COMMAND ----------

caminho_temp = "/mnt/temp"
list_caminho_temp = os.listdir(caminho_temp)
list_caminho_temp

# COMMAND ----------

# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

clientes_df = spark.read.option('header', True).csv("file:///mnt/temp/clientes.csv")
telefones_df = spark.read.option('header', True).csv("file:///mnt/temp/telefones.csv")
enderecos_df = spark.read.option('header', True).csv("file:///mnt/temp/enderecos.csv")
detalhes_clientes_df = spark.read.option('header', True).csv("file:///mnt/temp/detalhes.csv")
detalhes_dividas_df = spark.read.option('header', True).csv("file:///mnt/temp/detalhes_dividas.csv")
contratos_df = spark.read.option('header', True).csv("file:///mnt/temp/contratos.csv")
detalhes_contratos_df = spark.read.option('header', True).csv("file:///mnt/temp/detalhes_contratos.csv")
dividas_df = spark.read.option('header', True).csv("file:///mnt/temp/dividas.csv")
e_mails_dos_clientes_df = spark.read.option('header', True).csv("file:///mnt/temp/emails_dos_clientes.csv")

# COMMAND ----------

# DBTITLE 1,corrigindo nomes das colunas
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

      elif c.upper() == "Â":
        str_col = str_col + 'A'
      elif c.upper() == "Ê":
        str_col = str_col + 'E'
        
      else:
        str_col = str_col + c
    str_col = str_col.upper()
    if str_col[-1] == "_":
      str_col = str_col[:-1]
    df = df.withColumnRenamed(col, str_col)
  return df

clientes_df = padronizaNomesColunas(clientes_df)
detalhes_clientes_df = padronizaNomesColunas(detalhes_clientes_df)
telefones_df = padronizaNomesColunas(telefones_df)
enderecos_df = padronizaNomesColunas(enderecos_df)
detalhes_dividas_df = padronizaNomesColunas(detalhes_dividas_df)
contratos_df = padronizaNomesColunas(contratos_df)
detalhes_contratos_df = padronizaNomesColunas(detalhes_contratos_df)
dividas_df = padronizaNomesColunas(dividas_df)
detalhes_dividas_df = padronizaNomesColunas(detalhes_dividas_df)
e_mails_dos_clientes_df = padronizaNomesColunas(e_mails_dos_clientes_df)

# COMMAND ----------

# DBTITLE 1,escrevendo no parquet
spark.conf.set('spark.sql.sources.partitionOverwriteMode', 'dynamic')

clientes_df.write.mode('overwrite').parquet(parquetfilepath+'/'+data_arq_escolhido+'/'+'clientes.PARQUET') 
detalhes_clientes_df.write.mode('overwrite').parquet(parquetfilepath+'/'+data_arq_escolhido+'/'+'detalhes_clientes.PARQUET') 
telefones_df.write.mode('overwrite').parquet(parquetfilepath+'/'+data_arq_escolhido+'/'+'telefones.PARQUET') 
enderecos_df.write.mode('overwrite').parquet(parquetfilepath+'/'+data_arq_escolhido+'/'+'enderecos.PARQUET') 
detalhes_dividas_df.write.mode('overwrite').parquet(parquetfilepath+'/'+data_arq_escolhido+'/'+'detalhes_dividas.PARQUET') 
contratos_df.write.mode('overwrite').parquet(parquetfilepath+'/'+data_arq_escolhido+'/'+'contratos.PARQUET') 
detalhes_contratos_df.write.mode('overwrite').parquet(parquetfilepath+'/'+data_arq_escolhido+'/'+'detalhes_contratos.PARQUET') 
dividas_df.write.mode('overwrite').parquet(parquetfilepath+'/'+data_arq_escolhido+'/'+'dividas.PARQUET') 
e_mails_dos_clientes_df.write.mode('overwrite').parquet(parquetfilepath+'/'+data_arq_escolhido+'/'+'e_mails_dos_clientes.PARQUET') 