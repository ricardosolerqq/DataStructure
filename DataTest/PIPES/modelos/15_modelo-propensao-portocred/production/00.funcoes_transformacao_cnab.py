# Databricks notebook source
# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

# DBTITLE 1,VARIÁVEIS COM REGRAS
"""
GOOGLE SHEETS DAS REGRAS
https://docs.google.com/spreadsheets/d/1ZSXOSvVBcBHEcTQg88vh6PFIi5nTg9EQhGyDmzZaHOA/edit#gid=1545203801

"""

regras = {'clientes' : {'Tipo de Registro':2,
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
            'Tipo de Pessoa':1},

'telefones' : {'Tipo de Registro':2,
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
            'Tipo Preferencial':1},

'enderecos' : {'Tipo de Registro':2,
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
            'Tipo Preferencia':1},

'detalhes_clientes' : {'Tipo de Registro':2,
            'ID Cliente EC':10,
            'ID Cedente':10,
            'ID Cliente':50,
            'Nome Cedente':50,
            'ID Detalhe Cliente':10,
            'ID Detalhe':10,
            'Nome Detalhe':50,
            'Tipo':1,
            'Valor':500},

'contratos' : {'Tipo de Registro':2,
            'ID Cliente EC':10,
            'ID Cedente':10,
            'ID Cliente':50,
            'Nome Cedente':50,
            'ID Contrato EC (K1)':10,
            'ID Contrato (K2)':50,
            'ID Produto (K2)':10,
            'Nome Produto':50,
            'Plano':5,
            'Data Expiração':10},

'detalhes_contratos' : {'Tipo de Registro':2,
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
                      'Valor':500},

'dividas' : {'Tipo de Registro':2,
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
                      'Valor Mínimo':15},

'detalhes_dividas' : {'Tipo de Registro':2,
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
                    'Valor':500},

'e_mails_dos_clientes' : {'Tipo de Registro':2,
                          'ID E-mail EC (PK)':10,
                          'ID Cliente EC':10,
                          'ID Cedente':10,
                          'ID Cliente':50,
                          'Nome Cedente':50,
                          'Tipo':1,
                          'E-mail':100,
                          'Contato':50,
                          'Tipo Preferencial':1}}


# COMMAND ----------

def leCSV(txtfile):
  schema = [T.StructField('TXT_ORIGINAL', T.StringType(), False)]
  schema_final = T.StructType(schema)
  df = spark.read.format("csv").option("header", False).schema(schema_final).load(txtfile)
  return df

# COMMAND ----------

def criaDFs(df):
  df = df.withColumn("Tipo de Registro", F.col('TXT_ORIGINAL').substr(0,2))

  clientes_df = df.filter(F.col('Tipo de Registro')=='01')
  telefones_df = df.filter(F.col('Tipo de Registro')=='02')
  enderecos_df = df.filter(F.col('Tipo de Registro')=='03')
  detalhes_clientes_df = df.filter(F.col('Tipo de Registro')=='04')
  contratos_df = df.filter(F.col('Tipo de Registro')=='05')
  detalhes_contratos_df = df.filter(F.col('Tipo de Registro')=='06')
  dividas_df = df.filter(F.col('Tipo de Registro')=='07')
  detalhes_dividas_df = df.filter(F.col('Tipo de Registro')=='08')
  e_mails_dos_clientes_df = df.filter(F.col('Tipo de Registro')=='10')


  clientes = {'clientes':clientes_df}
  telefones = {'telefones': telefones_df}
  enderecos = {'enderecos': enderecos_df}
  detalhes_clientes = {'detalhes_clientes':detalhes_clientes_df}
  contratos = {'contratos':contratos_df} 
  detalhes_contratos = {'detalhes_contratos':detalhes_contratos_df}
  dividas = {'dividas':dividas_df}
  detalhes_dividas = {'detalhes_dividas':detalhes_dividas_df}
  e_mails_dos_clientes = {'e_mails_dos_clientes':e_mails_dos_clientes_df}

  
  return [clientes, telefones, enderecos, detalhes_clientes, contratos, detalhes_contratos, dividas, detalhes_dividas, e_mails_dos_clientes]

# COMMAND ----------

def txt_para_col(df, tipo_registro, regras):
  regras = regras[tipo_registro]
  indice = 1
  for coluna in regras:
    tamanho = regras[coluna]
    #print (coluna, regras[coluna])
    df = df.withColumn(coluna, F.col('TXT_ORIGINAL').substr(indice, tamanho))
    indice = indice+tamanho
  df = df.drop("TXT_ORIGINAL")

  return df

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

# COMMAND ----------

def processo_leitura_transformacao(txtfile,regras):
  df = leCSV(txtfile)
  list_dfs = criaDFs(df)
  dict_return = {}
  for dfs in list_dfs:
    for tipo_registro in dfs:
      df = dfs[tipo_registro]
      df = txt_para_col(df, tipo_registro,regras)
      df = padronizaNomesColunas(df)
      dict_return.update({tipo_registro:df})
  return dict_return