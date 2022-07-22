# Databricks notebook source
# MAGIC %md
# MAGIC # <font color='blue'>IA - Feature Selection</font>
# MAGIC 
# MAGIC # <font color='blue'>Ferramenta de Criação de Variáveis</font>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importando os pacotes Python

# COMMAND ----------

# Imports
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inserindo hiperparâmetros do Algoritmo

# COMMAND ----------

## Parâmetros do Algoritmo

#Variável chave-primaria
chave = 'DOCUMENT'

#Nome da Base de Dados
N_Base = "amostra_aleatoria_f.csv"

#Caminho da base de dados
caminho_base = "Base_Dados_Ferramenta/Colombo/"

#Separador
separador_ = ";"

#Decimal
decimal_ = "."

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importação da Base de Dados

# COMMAND ----------

base_dados = pd.read_csv(caminho_base+N_Base, sep=separador_, decimal=decimal_)
base_dados = base_dados[[chave,'amount', 'clientID', 'dueDate', 'param_9', 'param_5', 'param_7', 'tel1', 'tel10', 'tel2', 'tel3', 'tel4', 'tel5', 'tel6', 'tel8', 'tel9', 'activation']]

base_dados.fillna(-3)

#string
base_dados['activation'] = base_dados['activation'].replace(np.nan, '-3')

#numericas
base_dados['DOCUMENT'] = base_dados['DOCUMENT'].replace(np.nan, '-3')
base_dados['clientID'] = base_dados['clientID'].replace(np.nan, '-3')
base_dados['param_5'] = base_dados['param_5'].replace(np.nan, '-3')
base_dados['amount'] = base_dados['amount'].replace(np.nan, '-3')
base_dados['param_7'] = base_dados['param_7'].replace(np.nan, '-3')
base_dados['tel1'] = base_dados['tel1'].replace(np.nan, '-3')
base_dados['tel2'] = base_dados['tel2'].replace(np.nan, '-3')
base_dados['tel3'] = base_dados['tel3'].replace(np.nan, '-3')
base_dados['tel4'] = base_dados['tel4'].replace(np.nan, '-3')
base_dados['tel5'] = base_dados['tel5'].replace(np.nan, '-3')
base_dados['tel6'] = base_dados['tel6'].replace(np.nan, '-3')
base_dados['tel8'] = base_dados['tel8'].replace(np.nan, '-3')
base_dados['tel9'] = base_dados['tel9'].replace(np.nan, '-3')
base_dados['tel10'] = base_dados['tel10'].replace(np.nan, '-3')

base_dados = base_dados.apply(pd.to_numeric, errors='ignore')

base_dados['DOCUMENT'] = base_dados['DOCUMENT'].astype(np.int64)
base_dados['clientID'] = base_dados['clientID'].astype(np.int64)
base_dados['param_5'] = base_dados['param_5'].astype(float)
base_dados['amount'] = base_dados['amount'].astype(float)
base_dados['param_7'] = base_dados['param_7'].astype(np.int64)
base_dados['tel1'] = base_dados['tel1'].astype(np.int64)
base_dados['tel2'] = base_dados['tel2'].astype(np.int64)
base_dados['tel3'] = base_dados['tel3'].astype(np.int64)
base_dados['tel4'] = base_dados['tel4'].astype(np.int64)
base_dados['tel5'] = base_dados['tel5'].astype(np.int64)
base_dados['tel6'] = base_dados['tel6'].astype(np.int64)
base_dados['tel8'] = base_dados['tel8'].astype(np.int64)
base_dados['tel9'] = base_dados['tel9'].astype(np.int64)
base_dados['tel10'] = base_dados['tel10'].astype(np.int64)


base_dados['dueDate'] = pd.to_datetime(base_dados['dueDate'])
base_dados['param_9'] = pd.to_datetime(base_dados['param_9'])

base_dados['mob_dueDate'] = ((datetime.today()) - base_dados.dueDate)/np.timedelta64(1, 'M')
base_dados['year_dueDate'] = ((datetime.today()) - base_dados.dueDate)/np.timedelta64(1, 'Y')
base_dados['mob_param_9'] = ((datetime.today()) - base_dados.param_9)/np.timedelta64(1, 'M')
base_dados['year_param_9'] = ((datetime.today()) - base_dados.param_9)/np.timedelta64(1, 'Y')

del base_dados['dueDate']
del base_dados['param_9']


base_dados.drop_duplicates(keep=False, inplace=True)

print("shape da Base de Dados:",base_dados.shape)

base_dados.head()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Gerando variáveis Categóricas

# COMMAND ----------

base_dados['activation_gh30'] = np.where(base_dados['activation'] == 'ACIONADO MANUAL', 0,
np.where(base_dados['activation'] == 'ACORDO FECHADO', 1,
np.where(base_dados['activation'] == 'ACORDO PAY NOW', 2,
np.where(base_dados['activation'] == 'ACORDO WHATS APP', 3,
np.where(base_dados['activation'] == 'ASSE - DESCONHECIDO OUTROS', 4,
np.where(base_dados['activation'] == 'ASSE - DESLIGOU CAIU', 5,
np.where(base_dados['activation'] == 'ASSE - E-MAIL', 6,
np.where(base_dados['activation'] == 'ASSE - SMS', 7,
np.where(base_dados['activation'] == 'ASSE - URA', 8,
np.where(base_dados['activation'] == 'CAMP - ACORDO FECHADO', 9,
np.where(base_dados['activation'] == 'CAMP - CLIENTE VAI VERIFICAR', 10,
np.where(base_dados['activation'] == 'CAMP - MUDO CAIXA POSTAL', 11,
np.where(base_dados['activation'] == 'CELULAR DESCONHECIDO', 12,
np.where(base_dados['activation'] == 'CLIENTE DESLIGA CIENTE', 13,
np.where(base_dados['activation'] == 'CLIENTE ENVIADO PARA HIGIENIZACAO - CREDILINK', 14,
np.where(base_dados['activation'] == 'CLIENTE ENVIADO PARA HIGIENIZACAO - LEMIT', 15,
np.where(base_dados['activation'] == 'CLIENTE VAI VERIFICAR', 16,
np.where(base_dados['activation'] == 'COBR - CARTA 5 DIAS', 17,
np.where(base_dados['activation'] == 'COBR - PROMESSA DE PAGAMENTO', 18,
np.where(base_dados['activation'] == 'COBR - RECADO', 19,
np.where(base_dados['activation'] == 'CX POSTAL', 20,
np.where(base_dados['activation'] == 'DESABILITAR FONE DESCONHECIDO', 21,
np.where(base_dados['activation'] == 'DESLIGOU CAIU', 22,
np.where(base_dados['activation'] == 'EMAIL AUTOMATICO', 23,
np.where(base_dados['activation'] == 'ENC MAIORES DEVEDORES CP', 24,
np.where(base_dados['activation'] == 'ENVIADO LOJA', 25,
np.where(base_dados['activation'] == 'FALECIDO   ATESTADO OBITO', 26,
np.where(base_dados['activation'] == 'FONADA AUTOMATICO', 27,
np.where(base_dados['activation'] == 'LEMBRETE ACORDO', 28,
np.where(base_dados['activation'] == 'LEMBRETE ACORDO S  CONTATO', 29,
np.where(base_dados['activation'] == 'LOJA - ACORDO FECHADO', 30,
np.where(base_dados['activation'] == 'LOJA - CLIENTE VAI VERIFICAR', 31,
np.where(base_dados['activation'] == 'LOJA - CX POSTAL MUDO', 32,
np.where(base_dados['activation'] == 'LOJA - DESLIGOU CAIU', 33,
np.where(base_dados['activation'] == 'LOJA - FALECIDO   ATESTADO OBITO', 34,
np.where(base_dados['activation'] == 'LOJA - LOCALIZACAO VIA LOJA', 35,
np.where(base_dados['activation'] == 'LOJA - MUTIRAO DE COBRANCA', 36,
np.where(base_dados['activation'] == 'LOJA - NAO ATENDE SO CHAMA', 37,
np.where(base_dados['activation'] == 'LOJA - PROMESSA DE PAGAMENTO', 38,
np.where(base_dados['activation'] == 'LOJA - RECADO', 39,
np.where(base_dados['activation'] == 'LOJA - RETORNO LOCALIZACAO', 40,
np.where(base_dados['activation'] == 'MENSAGEM TELECOM', 41,
np.where(base_dados['activation'] == 'MUDO', 42,
np.where(base_dados['activation'] == 'NAO PASSA RECADO', 43,
np.where(base_dados['activation'] == 'PESQUISADO SEM CONTATO', 44,
np.where(base_dados['activation'] == 'PROMESSA DE PAGAMENTO', 45,
np.where(base_dados['activation'] == 'RECADO COM CONJUGE', 46,
np.where(base_dados['activation'] == 'RECADO NA REF', 47,
np.where(base_dados['activation'] == 'RECADO NO RESIDENCIAL', 48,
np.where(base_dados['activation'] == 'RECEPTIVO â€“ INFORMACAO', 49,
np.where(base_dados['activation'] == 'RETORNAR LIGACAO', 50,
np.where(base_dados['activation'] == 'SMS AUTOMATICO', 51,
np.where(base_dados['activation'] == 'WHATSAPP AUTOMATICO', 52,
0)))))))))))))))))))))))))))))))))))))))))))))))))))))
base_dados['activation_gh31'] = np.where(base_dados['activation_gh30'] == 0, 0,
np.where(base_dados['activation_gh30'] == 1, 1,
np.where(base_dados['activation_gh30'] == 2, 1,
np.where(base_dados['activation_gh30'] == 3, 1,
np.where(base_dados['activation_gh30'] == 4, 4,
np.where(base_dados['activation_gh30'] == 5, 4,
np.where(base_dados['activation_gh30'] == 6, 6,
np.where(base_dados['activation_gh30'] == 7, 7,
np.where(base_dados['activation_gh30'] == 8, 8,
np.where(base_dados['activation_gh30'] == 9, 9,
np.where(base_dados['activation_gh30'] == 10, 10,
np.where(base_dados['activation_gh30'] == 11, 11,
np.where(base_dados['activation_gh30'] == 12, 12,
np.where(base_dados['activation_gh30'] == 13, 13,
np.where(base_dados['activation_gh30'] == 14, 14,
np.where(base_dados['activation_gh30'] == 15, 14,
np.where(base_dados['activation_gh30'] == 16, 16,
np.where(base_dados['activation_gh30'] == 17, 17,
np.where(base_dados['activation_gh30'] == 18, 18,
np.where(base_dados['activation_gh30'] == 19, 19,
np.where(base_dados['activation_gh30'] == 20, 20,
np.where(base_dados['activation_gh30'] == 21, 21,
np.where(base_dados['activation_gh30'] == 22, 22,
np.where(base_dados['activation_gh30'] == 23, 23,
np.where(base_dados['activation_gh30'] == 24, 23,
np.where(base_dados['activation_gh30'] == 25, 23,
np.where(base_dados['activation_gh30'] == 26, 23,
np.where(base_dados['activation_gh30'] == 27, 27,
np.where(base_dados['activation_gh30'] == 28, 28,
np.where(base_dados['activation_gh30'] == 29, 28,
np.where(base_dados['activation_gh30'] == 30, 30,
np.where(base_dados['activation_gh30'] == 31, 31,
np.where(base_dados['activation_gh30'] == 32, 31,
np.where(base_dados['activation_gh30'] == 33, 33,
np.where(base_dados['activation_gh30'] == 34, 33,
np.where(base_dados['activation_gh30'] == 35, 35,
np.where(base_dados['activation_gh30'] == 36, 36,
np.where(base_dados['activation_gh30'] == 37, 37,
np.where(base_dados['activation_gh30'] == 38, 38,
np.where(base_dados['activation_gh30'] == 39, 38,
np.where(base_dados['activation_gh30'] == 40, 40,
np.where(base_dados['activation_gh30'] == 41, 41,
np.where(base_dados['activation_gh30'] == 42, 42,
np.where(base_dados['activation_gh30'] == 43, 43,
np.where(base_dados['activation_gh30'] == 44, 44,
np.where(base_dados['activation_gh30'] == 45, 45,
np.where(base_dados['activation_gh30'] == 46, 46,
np.where(base_dados['activation_gh30'] == 47, 47,
np.where(base_dados['activation_gh30'] == 48, 48,
np.where(base_dados['activation_gh30'] == 49, 48,
np.where(base_dados['activation_gh30'] == 50, 50,
np.where(base_dados['activation_gh30'] == 51, 51,
np.where(base_dados['activation_gh30'] == 52, 52,
0)))))))))))))))))))))))))))))))))))))))))))))))))))))
base_dados['activation_gh32'] = np.where(base_dados['activation_gh31'] == 0, 0,
np.where(base_dados['activation_gh31'] == 1, 1,
np.where(base_dados['activation_gh31'] == 4, 2,
np.where(base_dados['activation_gh31'] == 6, 3,
np.where(base_dados['activation_gh31'] == 7, 4,
np.where(base_dados['activation_gh31'] == 8, 5,
np.where(base_dados['activation_gh31'] == 9, 6,
np.where(base_dados['activation_gh31'] == 10, 7,
np.where(base_dados['activation_gh31'] == 11, 8,
np.where(base_dados['activation_gh31'] == 12, 9,
np.where(base_dados['activation_gh31'] == 13, 10,
np.where(base_dados['activation_gh31'] == 14, 11,
np.where(base_dados['activation_gh31'] == 16, 12,
np.where(base_dados['activation_gh31'] == 17, 13,
np.where(base_dados['activation_gh31'] == 18, 14,
np.where(base_dados['activation_gh31'] == 19, 15,
np.where(base_dados['activation_gh31'] == 20, 16,
np.where(base_dados['activation_gh31'] == 21, 17,
np.where(base_dados['activation_gh31'] == 22, 18,
np.where(base_dados['activation_gh31'] == 23, 19,
np.where(base_dados['activation_gh31'] == 27, 20,
np.where(base_dados['activation_gh31'] == 28, 21,
np.where(base_dados['activation_gh31'] == 30, 22,
np.where(base_dados['activation_gh31'] == 31, 23,
np.where(base_dados['activation_gh31'] == 33, 24,
np.where(base_dados['activation_gh31'] == 35, 25,
np.where(base_dados['activation_gh31'] == 36, 26,
np.where(base_dados['activation_gh31'] == 37, 27,
np.where(base_dados['activation_gh31'] == 38, 28,
np.where(base_dados['activation_gh31'] == 40, 29,
np.where(base_dados['activation_gh31'] == 41, 30,
np.where(base_dados['activation_gh31'] == 42, 31,
np.where(base_dados['activation_gh31'] == 43, 32,
np.where(base_dados['activation_gh31'] == 44, 33,
np.where(base_dados['activation_gh31'] == 45, 34,
np.where(base_dados['activation_gh31'] == 46, 35,
np.where(base_dados['activation_gh31'] == 47, 36,
np.where(base_dados['activation_gh31'] == 48, 37,
np.where(base_dados['activation_gh31'] == 50, 38,
np.where(base_dados['activation_gh31'] == 51, 39,
np.where(base_dados['activation_gh31'] == 52, 40,
0)))))))))))))))))))))))))))))))))))))))))
base_dados['activation_gh33'] = np.where(base_dados['activation_gh32'] == 0, 0,
np.where(base_dados['activation_gh32'] == 1, 1,
np.where(base_dados['activation_gh32'] == 2, 2,
np.where(base_dados['activation_gh32'] == 3, 3,
np.where(base_dados['activation_gh32'] == 4, 4,
np.where(base_dados['activation_gh32'] == 5, 5,
np.where(base_dados['activation_gh32'] == 6, 6,
np.where(base_dados['activation_gh32'] == 7, 7,
np.where(base_dados['activation_gh32'] == 8, 8,
np.where(base_dados['activation_gh32'] == 9, 9,
np.where(base_dados['activation_gh32'] == 10, 10,
np.where(base_dados['activation_gh32'] == 11, 11,
np.where(base_dados['activation_gh32'] == 12, 12,
np.where(base_dados['activation_gh32'] == 13, 13,
np.where(base_dados['activation_gh32'] == 14, 14,
np.where(base_dados['activation_gh32'] == 15, 15,
np.where(base_dados['activation_gh32'] == 16, 16,
np.where(base_dados['activation_gh32'] == 17, 17,
np.where(base_dados['activation_gh32'] == 18, 18,
np.where(base_dados['activation_gh32'] == 19, 19,
np.where(base_dados['activation_gh32'] == 20, 20,
np.where(base_dados['activation_gh32'] == 21, 21,
np.where(base_dados['activation_gh32'] == 22, 22,
np.where(base_dados['activation_gh32'] == 23, 23,
np.where(base_dados['activation_gh32'] == 24, 24,
np.where(base_dados['activation_gh32'] == 25, 25,
np.where(base_dados['activation_gh32'] == 26, 26,
np.where(base_dados['activation_gh32'] == 27, 27,
np.where(base_dados['activation_gh32'] == 28, 28,
np.where(base_dados['activation_gh32'] == 29, 29,
np.where(base_dados['activation_gh32'] == 30, 30,
np.where(base_dados['activation_gh32'] == 31, 31,
np.where(base_dados['activation_gh32'] == 32, 32,
np.where(base_dados['activation_gh32'] == 33, 33,
np.where(base_dados['activation_gh32'] == 34, 34,
np.where(base_dados['activation_gh32'] == 35, 35,
np.where(base_dados['activation_gh32'] == 36, 36,
np.where(base_dados['activation_gh32'] == 37, 37,
np.where(base_dados['activation_gh32'] == 38, 38,
np.where(base_dados['activation_gh32'] == 39, 39,
np.where(base_dados['activation_gh32'] == 40, 40,
0)))))))))))))))))))))))))))))))))))))))))
base_dados['activation_gh34'] = np.where(base_dados['activation_gh33'] == 0, 39,
np.where(base_dados['activation_gh33'] == 1, 39,
np.where(base_dados['activation_gh33'] == 2, 8,
np.where(base_dados['activation_gh33'] == 3, 3,
np.where(base_dados['activation_gh33'] == 4, 8,
np.where(base_dados['activation_gh33'] == 5, 39,
np.where(base_dados['activation_gh33'] == 6, 8,
np.where(base_dados['activation_gh33'] == 7, 39,
np.where(base_dados['activation_gh33'] == 8, 8,
np.where(base_dados['activation_gh33'] == 9, 39,
np.where(base_dados['activation_gh33'] == 10, 39,
np.where(base_dados['activation_gh33'] == 11, 8,
np.where(base_dados['activation_gh33'] == 12, 39,
np.where(base_dados['activation_gh33'] == 13, 39,
np.where(base_dados['activation_gh33'] == 14, 39,
np.where(base_dados['activation_gh33'] == 15, 39,
np.where(base_dados['activation_gh33'] == 16, 8,
np.where(base_dados['activation_gh33'] == 17, 8,
np.where(base_dados['activation_gh33'] == 18, 39,
np.where(base_dados['activation_gh33'] == 19, 8,
np.where(base_dados['activation_gh33'] == 20, 8,
np.where(base_dados['activation_gh33'] == 21, 39,
np.where(base_dados['activation_gh33'] == 22, 39,
np.where(base_dados['activation_gh33'] == 23, 39,
np.where(base_dados['activation_gh33'] == 24, 8,
np.where(base_dados['activation_gh33'] == 25, 39,
np.where(base_dados['activation_gh33'] == 26, 39,
np.where(base_dados['activation_gh33'] == 27, 8,
np.where(base_dados['activation_gh33'] == 28, 39,
np.where(base_dados['activation_gh33'] == 29, 39,
np.where(base_dados['activation_gh33'] == 30, 8,
np.where(base_dados['activation_gh33'] == 31, 8,
np.where(base_dados['activation_gh33'] == 32, 39,
np.where(base_dados['activation_gh33'] == 33, 39,
np.where(base_dados['activation_gh33'] == 34, 39,
np.where(base_dados['activation_gh33'] == 35, 39,
np.where(base_dados['activation_gh33'] == 36, 39,
np.where(base_dados['activation_gh33'] == 37, 39,
np.where(base_dados['activation_gh33'] == 38, 39,
np.where(base_dados['activation_gh33'] == 39, 39,
np.where(base_dados['activation_gh33'] == 40, 39,
0)))))))))))))))))))))))))))))))))))))))))
base_dados['activation_gh35'] = np.where(base_dados['activation_gh34'] == 3, 0,
np.where(base_dados['activation_gh34'] == 8, 1,
np.where(base_dados['activation_gh34'] == 39, 2,
0)))
base_dados['activation_gh36'] = np.where(base_dados['activation_gh35'] == 0, 1,
np.where(base_dados['activation_gh35'] == 1, 0,
np.where(base_dados['activation_gh35'] == 2, 2,
1)))
base_dados['activation_gh37'] = np.where(base_dados['activation_gh36'] == 0, 1,
np.where(base_dados['activation_gh36'] == 1, 1,
np.where(base_dados['activation_gh36'] == 2, 2,
0)))
base_dados['activation_gh38'] = np.where(base_dados['activation_gh37'] == 1, 0,
np.where(base_dados['activation_gh37'] == 2, 1,
0))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Gerando variáveis numéricas contínuas

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parte 1 de 2

# COMMAND ----------

base_dados['mob_param_9__pu_20'] = np.where(base_dados['mob_param_9'] <= 2.860768261418403, 0.0,
np.where(np.bitwise_and(base_dados['mob_param_9'] > 2.860768261418403, base_dados['mob_param_9'] <= 4.897771074617853), 1.0,
np.where(np.bitwise_and(base_dados['mob_param_9'] > 4.897771074617853, base_dados['mob_param_9'] <= 6.869064119649578), 2.0,
np.where(np.bitwise_and(base_dados['mob_param_9'] > 6.869064119649578, base_dados['mob_param_9'] <= 8.971776701016752), 3.0,
np.where(np.bitwise_and(base_dados['mob_param_9'] > 8.971776701016752, base_dados['mob_param_9'] <= 10.614520905209856), 4.0,
np.where(np.bitwise_and(base_dados['mob_param_9'] > 10.614520905209856, base_dados['mob_param_9'] <= 12.585813950241583), 5.0,
np.where(np.bitwise_and(base_dados['mob_param_9'] > 12.585813950241583, base_dados['mob_param_9'] <= 14.91851072019579), 6.0,
np.where(np.bitwise_and(base_dados['mob_param_9'] > 14.91851072019579, base_dados['mob_param_9'] <= 15.83844747454393), 7.0,
np.where(np.bitwise_and(base_dados['mob_param_9'] > 15.83844747454393, base_dados['mob_param_9'] <= 35.387103504441875), 16.0,
np.where(base_dados['mob_param_9'] > 35.387103504441875, 19.0,
 0))))))))))
base_dados['mob_param_9__pu_20_g_1_1'] = np.where(base_dados['mob_param_9__pu_20'] == 0.0, 1,
np.where(base_dados['mob_param_9__pu_20'] == 1.0, 0,
np.where(base_dados['mob_param_9__pu_20'] == 2.0, 2,
np.where(base_dados['mob_param_9__pu_20'] == 3.0, 0,
np.where(base_dados['mob_param_9__pu_20'] == 4.0, 2,
np.where(base_dados['mob_param_9__pu_20'] == 5.0, 2,
np.where(base_dados['mob_param_9__pu_20'] == 6.0, 2,
np.where(base_dados['mob_param_9__pu_20'] == 7.0, 2,
np.where(base_dados['mob_param_9__pu_20'] == 16.0, 2,
np.where(base_dados['mob_param_9__pu_20'] == 19.0, 2,
 0))))))))))
base_dados['mob_param_9__pu_20_g_1_2'] = np.where(base_dados['mob_param_9__pu_20_g_1_1'] == 0, 1,
np.where(base_dados['mob_param_9__pu_20_g_1_1'] == 1, 2,
np.where(base_dados['mob_param_9__pu_20_g_1_1'] == 2, 0,
 0)))
base_dados['mob_param_9__pu_20_g_1'] = np.where(base_dados['mob_param_9__pu_20_g_1_2'] == 0, 0,
np.where(base_dados['mob_param_9__pu_20_g_1_2'] == 1, 1,
np.where(base_dados['mob_param_9__pu_20_g_1_2'] == 2, 2,
 0)))
         
         
         
         
         
         
       
         
base_dados['mob_param_9__S'] = np.sin(base_dados['mob_param_9'])
np.where(base_dados['mob_param_9__S'] == 0, -1, base_dados['mob_param_9__S'])
base_dados['mob_param_9__S'] = base_dados['mob_param_9__S'].fillna(-2)
base_dados['mob_param_9__S__p_3'] = np.where(base_dados['mob_param_9__S'] <= -0.8901265191322705, 0.0,
np.where(np.bitwise_and(base_dados['mob_param_9__S'] > -0.8901265191322705, base_dados['mob_param_9__S'] <= -0.6714909474215969), 1.0,
np.where(base_dados['mob_param_9__S'] > -0.6714909474215969, 2.0,
 0)))
base_dados['mob_param_9__S__p_3_g_1_1'] = np.where(base_dados['mob_param_9__S__p_3'] == 0.0, 1,
np.where(base_dados['mob_param_9__S__p_3'] == 1.0, 2,
np.where(base_dados['mob_param_9__S__p_3'] == 2.0, 0,
 0)))
base_dados['mob_param_9__S__p_3_g_1_2'] = np.where(base_dados['mob_param_9__S__p_3_g_1_1'] == 0, 2,
np.where(base_dados['mob_param_9__S__p_3_g_1_1'] == 1, 1,
np.where(base_dados['mob_param_9__S__p_3_g_1_1'] == 2, 0,
 0)))
base_dados['mob_param_9__S__p_3_g_1'] = np.where(base_dados['mob_param_9__S__p_3_g_1_2'] == 0, 0,
np.where(base_dados['mob_param_9__S__p_3_g_1_2'] == 1, 1,
np.where(base_dados['mob_param_9__S__p_3_g_1_2'] == 2, 2,
 0)))
         
         
         
         
         
         
                
base_dados['tel1__pu_20'] = np.where(base_dados['tel1'] <= 4635363940.0, 0.0,
np.where(np.bitwise_and(base_dados['tel1'] > 4635363940.0, base_dados['tel1'] <= 5538233967.0), 1.0,
np.where(np.bitwise_and(base_dados['tel1'] > 5538233967.0, base_dados['tel1'] <= 13997793145.0), 2.0,
np.where(np.bitwise_and(base_dados['tel1'] > 13997793145.0, base_dados['tel1'] <= 17991087219.0), 3.0,
np.where(np.bitwise_and(base_dados['tel1'] > 17991087219.0, base_dados['tel1'] <= 19998816287.0), 4.0,
np.where(np.bitwise_and(base_dados['tel1'] > 19998816287.0, base_dados['tel1'] <= 35999199854.0), 7.0,
np.where(np.bitwise_and(base_dados['tel1'] > 35999199854.0, base_dados['tel1'] <= 41999650741.0), 8.0,
np.where(np.bitwise_and(base_dados['tel1'] > 41999650741.0, base_dados['tel1'] <= 46999317083.0), 9.0,
np.where(np.bitwise_and(base_dados['tel1'] > 46999317083.0, base_dados['tel1'] <= 49999833098.0), 10.0,
np.where(np.bitwise_and(base_dados['tel1'] > 49999833098.0, base_dados['tel1'] <= 55999963759.0), 11.0,
np.where(np.bitwise_and(base_dados['tel1'] > 55999963759.0, base_dados['tel1'] <= 61991537009.0), 13.0,
np.where(np.bitwise_and(base_dados['tel1'] > 61991537009.0, base_dados['tel1'] <= 67998495249.0), 14.0,
np.where(np.bitwise_and(base_dados['tel1'] > 67998495249.0, base_dados['tel1'] <= 77998593896.0), 16.0,
np.where(np.bitwise_and(base_dados['tel1'] > 77998593896.0, base_dados['tel1'] <= 81987880430.0), 17.0,
np.where(base_dados['tel1'] > 81987880430.0, 19.0,
 0)))))))))))))))
base_dados['tel1__pu_20_g_1_1'] = np.where(base_dados['tel1__pu_20'] == 0.0, 0,
np.where(base_dados['tel1__pu_20'] == 1.0, 2,
np.where(base_dados['tel1__pu_20'] == 2.0, 3,
np.where(base_dados['tel1__pu_20'] == 3.0, 2,
np.where(base_dados['tel1__pu_20'] == 4.0, 1,
np.where(base_dados['tel1__pu_20'] == 7.0, 3,
np.where(base_dados['tel1__pu_20'] == 8.0, 1,
np.where(base_dados['tel1__pu_20'] == 9.0, 3,
np.where(base_dados['tel1__pu_20'] == 10.0, 2,
np.where(base_dados['tel1__pu_20'] == 11.0, 1,
np.where(base_dados['tel1__pu_20'] == 13.0, 3,
np.where(base_dados['tel1__pu_20'] == 14.0, 3,
np.where(base_dados['tel1__pu_20'] == 16.0, 3,
np.where(base_dados['tel1__pu_20'] == 17.0, 3,
np.where(base_dados['tel1__pu_20'] == 19.0, 3,
 0)))))))))))))))
base_dados['tel1__pu_20_g_1_2'] = np.where(base_dados['tel1__pu_20_g_1_1'] == 0, 0,
np.where(base_dados['tel1__pu_20_g_1_1'] == 1, 2,
np.where(base_dados['tel1__pu_20_g_1_1'] == 2, 0,
np.where(base_dados['tel1__pu_20_g_1_1'] == 3, 2,
 0))))
base_dados['tel1__pu_20_g_1'] = np.where(base_dados['tel1__pu_20_g_1_2'] == 0, 0,
np.where(base_dados['tel1__pu_20_g_1_2'] == 2, 1,
 0))
                                         
                                         
                                         
                                         
                                         
                                         
                                         
                                         
base_dados['tel1__S'] = np.sin(base_dados['tel1'])
np.where(base_dados['tel1__S'] == 0, -1, base_dados['tel1__S'])
base_dados['tel1__S'] = base_dados['tel1__S'].fillna(-2)
base_dados['tel1__S__p_6'] = np.where(base_dados['tel1__S'] <= -0.8407800674843219, 0.0,
np.where(np.bitwise_and(base_dados['tel1__S'] > -0.8407800674843219, base_dados['tel1__S'] <= -0.4376120950754989), 1.0,
np.where(np.bitwise_and(base_dados['tel1__S'] > -0.4376120950754989, base_dados['tel1__S'] <= 0.05133403408977871), 2.0,
np.where(np.bitwise_and(base_dados['tel1__S'] > 0.05133403408977871, base_dados['tel1__S'] <= 0.5665490035316437), 3.0,
np.where(np.bitwise_and(base_dados['tel1__S'] > 0.5665490035316437, base_dados['tel1__S'] <= 0.8927230198788805), 4.0,
np.where(base_dados['tel1__S'] > 0.8927230198788805, 5.0,
 0))))))
base_dados['tel1__S__p_6_g_1_1'] = np.where(base_dados['tel1__S__p_6'] == 0.0, 1,
np.where(base_dados['tel1__S__p_6'] == 1.0, 0,
np.where(base_dados['tel1__S__p_6'] == 2.0, 0,
np.where(base_dados['tel1__S__p_6'] == 3.0, 2,
np.where(base_dados['tel1__S__p_6'] == 4.0, 1,
np.where(base_dados['tel1__S__p_6'] == 5.0, 2,
 0))))))
base_dados['tel1__S__p_6_g_1_2'] = np.where(base_dados['tel1__S__p_6_g_1_1'] == 0, 1,
np.where(base_dados['tel1__S__p_6_g_1_1'] == 1, 0,
np.where(base_dados['tel1__S__p_6_g_1_1'] == 2, 1,
 0)))
base_dados['tel1__S__p_6_g_1'] = np.where(base_dados['tel1__S__p_6_g_1_2'] == 0, 0,
np.where(base_dados['tel1__S__p_6_g_1_2'] == 1, 1,
 0))
                                          
                                          
                                          
                                          
                                          
                                          
                                          
                                          
base_dados['tel4__L'] = np.log(base_dados['tel4'])
np.where(base_dados['tel4__L'] == 0, -1, base_dados['tel4__L'])
base_dados['tel4__L'] = base_dados['tel4__L'].fillna(-2)
base_dados['tel4__L__pe_6'] = np.where(np.bitwise_and(base_dados['tel4__L'] >= -2.0, base_dados['tel4__L'] <= 17.429876788590487), 3.0,
np.where(np.bitwise_and(base_dados['tel4__L'] > 17.429876788590487, base_dados['tel4__L'] <= 22.278117294698564), 4.0,
np.where(base_dados['tel4__L'] > 22.278117294698564, 5.0,
 -2)))
base_dados['tel4__L__pe_6_g_1_1'] = np.where(base_dados['tel4__L__pe_6'] == -2.0, 1,
np.where(base_dados['tel4__L__pe_6'] == 3.0, 1,
np.where(base_dados['tel4__L__pe_6'] == 4.0, 0,
np.where(base_dados['tel4__L__pe_6'] == 5.0, 0,
 0))))
base_dados['tel4__L__pe_6_g_1_2'] = np.where(base_dados['tel4__L__pe_6_g_1_1'] == 0, 1,
np.where(base_dados['tel4__L__pe_6_g_1_1'] == 1, 0,
 0))
base_dados['tel4__L__pe_6_g_1'] = np.where(base_dados['tel4__L__pe_6_g_1_2'] == 0, 0,
np.where(base_dados['tel4__L__pe_6_g_1_2'] == 1, 1,
 0))

                                           
                                           
                                           
                                           
                                           
                                           
base_dados['tel4__C'] = np.cos(base_dados['tel4'])
np.where(base_dados['tel4__C'] == 0, -1, base_dados['tel4__C'])
base_dados['tel4__C'] = base_dados['tel4__C'].fillna(-2)
base_dados['tel4__C__p_8'] = np.where(base_dados['tel4__C'] <= -0.9902215255359477, 0.0,
np.where(np.bitwise_and(base_dados['tel4__C'] > -0.9902215255359477, base_dados['tel4__C'] <= -0.8605840727612905), 1.0,
np.where(np.bitwise_and(base_dados['tel4__C'] > -0.8605840727612905, base_dados['tel4__C'] <= -0.5232622589504722), 2.0,
np.where(np.bitwise_and(base_dados['tel4__C'] > -0.5232622589504722, base_dados['tel4__C'] <= -0.034478918564036565), 3.0,
np.where(np.bitwise_and(base_dados['tel4__C'] > -0.034478918564036565, base_dados['tel4__C'] <= 0.39466654431632947), 4.0,
np.where(np.bitwise_and(base_dados['tel4__C'] > 0.39466654431632947, base_dados['tel4__C'] <= 0.8123979355644105), 5.0,
np.where(base_dados['tel4__C'] > 0.8123979355644105, 6.0,
 0)))))))
base_dados['tel4__C__p_8_g_1_1'] = np.where(base_dados['tel4__C__p_8'] == 0.0, 0,
np.where(base_dados['tel4__C__p_8'] == 1.0, 1,
np.where(base_dados['tel4__C__p_8'] == 2.0, 0,
np.where(base_dados['tel4__C__p_8'] == 3.0, 1,
np.where(base_dados['tel4__C__p_8'] == 4.0, 0,
np.where(base_dados['tel4__C__p_8'] == 5.0, 0,
np.where(base_dados['tel4__C__p_8'] == 6.0, 1,
 0)))))))
base_dados['tel4__C__p_8_g_1_2'] = np.where(base_dados['tel4__C__p_8_g_1_1'] == 0, 1,
np.where(base_dados['tel4__C__p_8_g_1_1'] == 1, 0,
 0))
base_dados['tel4__C__p_8_g_1'] = np.where(base_dados['tel4__C__p_8_g_1_2'] == 0, 0,
np.where(base_dados['tel4__C__p_8_g_1_2'] == 1, 1,
 0))
                                          
                                          
                                          
                                          
                                          
                                          
                                          
base_dados['tel3__pk_4'] = np.where(base_dados['tel3'] <= 11996228724.0, 0.0,
np.where(np.bitwise_and(base_dados['tel3'] > 11996228724.0, base_dados['tel3'] <= 35998923412.0), 1.0,
np.where(np.bitwise_and(base_dados['tel3'] > 35998923412.0, base_dados['tel3'] <= 55999994731.0), 2.0,
np.where(base_dados['tel3'] > 55999994731.0, 3.0,
 0))))
base_dados['tel3__pk_4_g_1_1'] = np.where(base_dados['tel3__pk_4'] == 0.0, 0,
np.where(base_dados['tel3__pk_4'] == 1.0, 1,
np.where(base_dados['tel3__pk_4'] == 2.0, 1,
np.where(base_dados['tel3__pk_4'] == 3.0, 1,
 0))))
base_dados['tel3__pk_4_g_1_2'] = np.where(base_dados['tel3__pk_4_g_1_1'] == 0, 0,
np.where(base_dados['tel3__pk_4_g_1_1'] == 1, 1,
 0))
base_dados['tel3__pk_4_g_1'] = np.where(base_dados['tel3__pk_4_g_1_2'] == 0, 0,
np.where(base_dados['tel3__pk_4_g_1_2'] == 1, 1,
 0))
                                        
                                        
                                        
                                        
                                        
                                        
                                        
base_dados['tel3__S'] = np.sin(base_dados['tel3'])
np.where(base_dados['tel3__S'] == 0, -1, base_dados['tel3__S'])
base_dados['tel3__S'] = base_dados['tel3__S'].fillna(-2)
base_dados['tel3__S__p_7'] = np.where(base_dados['tel3__S'] <= -0.8798679152745891, 0.0,
np.where(np.bitwise_and(base_dados['tel3__S'] > -0.8798679152745891, base_dados['tel3__S'] <= -0.4832476317316815), 1.0,
np.where(np.bitwise_and(base_dados['tel3__S'] > -0.4832476317316815, base_dados['tel3__S'] <= -0.14404319343079658), 2.0,
np.where(np.bitwise_and(base_dados['tel3__S'] > -0.14404319343079658, base_dados['tel3__S'] <= 0.015913646208187988), 3.0,
np.where(np.bitwise_and(base_dados['tel3__S'] > 0.015913646208187988, base_dados['tel3__S'] <= 0.5247962296550654), 4.0,
np.where(np.bitwise_and(base_dados['tel3__S'] > 0.5247962296550654, base_dados['tel3__S'] <= 0.8908732442241725), 5.0,
np.where(base_dados['tel3__S'] > 0.8908732442241725, 6.0,
 0)))))))
base_dados['tel3__S__p_7_g_1_1'] = np.where(base_dados['tel3__S__p_7'] == 0.0, 0,
np.where(base_dados['tel3__S__p_7'] == 1.0, 0,
np.where(base_dados['tel3__S__p_7'] == 2.0, 0,
np.where(base_dados['tel3__S__p_7'] == 3.0, 1,
np.where(base_dados['tel3__S__p_7'] == 4.0, 2,
np.where(base_dados['tel3__S__p_7'] == 5.0, 2,
np.where(base_dados['tel3__S__p_7'] == 6.0, 0,
 0)))))))
base_dados['tel3__S__p_7_g_1_2'] = np.where(base_dados['tel3__S__p_7_g_1_1'] == 0, 2,
np.where(base_dados['tel3__S__p_7_g_1_1'] == 1, 0,
np.where(base_dados['tel3__S__p_7_g_1_1'] == 2, 1,
 0)))
base_dados['tel3__S__p_7_g_1'] = np.where(base_dados['tel3__S__p_7_g_1_2'] == 0, 0,
np.where(base_dados['tel3__S__p_7_g_1_2'] == 1, 1,
np.where(base_dados['tel3__S__p_7_g_1_2'] == 2, 2,
 0)))
         
         
         
         
         
         
         
base_dados['param_7__pu_6'] = np.where(base_dados['param_7'] <= -33.0, 0.0,
np.where(np.bitwise_and(base_dados['param_7'] > -33.0, base_dados['param_7'] <= -3.0), 1.0,
np.where(np.bitwise_and(base_dados['param_7'] > -3.0, base_dados['param_7'] <= 43.0), 2.0,
np.where(np.bitwise_and(base_dados['param_7'] > 43.0, base_dados['param_7'] <= 68.0), 3.0,
np.where(np.bitwise_and(base_dados['param_7'] > 68.0, base_dados['param_7'] <= 94.0), 4.0,
np.where(base_dados['param_7'] > 94.0, 5.0,
 0))))))
base_dados['param_7__pu_6_g_1_1'] = np.where(base_dados['param_7__pu_6'] == 0.0, 1,
np.where(base_dados['param_7__pu_6'] == 1.0, 1,
np.where(base_dados['param_7__pu_6'] == 2.0, 0,
np.where(base_dados['param_7__pu_6'] == 3.0, 0,
np.where(base_dados['param_7__pu_6'] == 4.0, 1,
np.where(base_dados['param_7__pu_6'] == 5.0, 1,
 0))))))
base_dados['param_7__pu_6_g_1_2'] = np.where(base_dados['param_7__pu_6_g_1_1'] == 0, 1,
np.where(base_dados['param_7__pu_6_g_1_1'] == 1, 0,
 0))
base_dados['param_7__pu_6_g_1'] = np.where(base_dados['param_7__pu_6_g_1_2'] == 0, 0,
np.where(base_dados['param_7__pu_6_g_1_2'] == 1, 1,
 0))
                                           
                                           
                                           
                                        
base_dados['param_7__pk_8'] = np.where(base_dados['param_7'] <= -33.0, 0.0,
np.where(np.bitwise_and(base_dados['param_7'] > -33.0, base_dados['param_7'] <= -3.0), 1.0,
np.where(np.bitwise_and(base_dados['param_7'] > -3.0, base_dados['param_7'] <= 32.0), 2.0,
np.where(np.bitwise_and(base_dados['param_7'] > 32.0, base_dados['param_7'] <= 45.0), 3.0,
np.where(np.bitwise_and(base_dados['param_7'] > 45.0, base_dados['param_7'] <= 58.0), 4.0,
np.where(np.bitwise_and(base_dados['param_7'] > 58.0, base_dados['param_7'] <= 72.0), 5.0,
np.where(np.bitwise_and(base_dados['param_7'] > 72.0, base_dados['param_7'] <= 94.0), 6.0,
np.where(base_dados['param_7'] > 94.0, 7.0,
 0))))))))
base_dados['param_7__pk_8_g_1_1'] = np.where(base_dados['param_7__pk_8'] == 0.0, 1,
np.where(base_dados['param_7__pk_8'] == 1.0, 1,
np.where(base_dados['param_7__pk_8'] == 2.0, 0,
np.where(base_dados['param_7__pk_8'] == 3.0, 0,
np.where(base_dados['param_7__pk_8'] == 4.0, 0,
np.where(base_dados['param_7__pk_8'] == 5.0, 1,
np.where(base_dados['param_7__pk_8'] == 6.0, 1,
np.where(base_dados['param_7__pk_8'] == 7.0, 1,
 0))))))))
base_dados['param_7__pk_8_g_1_2'] = np.where(base_dados['param_7__pk_8_g_1_1'] == 0, 1,
np.where(base_dados['param_7__pk_8_g_1_1'] == 1, 0,
 0))
base_dados['param_7__pk_8_g_1'] = np.where(base_dados['param_7__pk_8_g_1_2'] == 0, 0,
np.where(base_dados['param_7__pk_8_g_1_2'] == 1, 1,
 0))
                                           
                                           
                                           
                                           
                                           
                                           
                                           
base_dados['tel2__pe_10'] = np.where(np.bitwise_and(base_dados['tel2'] >= -3.0, base_dados['tel2'] <= 5539313500.0), 0.0,
np.where(np.bitwise_and(base_dados['tel2'] > 5539313500.0, base_dados['tel2'] <= 9936633586.0), 1.0,
np.where(np.bitwise_and(base_dados['tel2'] > 9936633586.0, base_dados['tel2'] <= 15997973283.0), 2.0,
np.where(np.bitwise_and(base_dados['tel2'] > 15997973283.0, base_dados['tel2'] <= 19997200509.0), 3.0,
np.where(np.bitwise_and(base_dados['tel2'] > 19997200509.0, base_dados['tel2'] <= 22998906124.0), 4.0,
np.where(np.bitwise_and(base_dados['tel2'] > 22998906124.0, base_dados['tel2'] <= 32988579333.0), 5.0,
np.where(np.bitwise_and(base_dados['tel2'] > 32988579333.0, base_dados['tel2'] <= 35999275674.0), 6.0,
np.where(np.bitwise_and(base_dados['tel2'] > 35999275674.0, base_dados['tel2'] <= 43999660299.0), 7.0,
np.where(np.bitwise_and(base_dados['tel2'] > 43999660299.0, base_dados['tel2'] <= 49999705060.0), 8.0,
np.where(base_dados['tel2'] > 49999705060.0, 9.0,
 -2))))))))))
base_dados['tel2__pe_10_g_1_1'] = np.where(base_dados['tel2__pe_10'] == -2.0, 2,
np.where(base_dados['tel2__pe_10'] == 0.0, 0,
np.where(base_dados['tel2__pe_10'] == 1.0, 2,
np.where(base_dados['tel2__pe_10'] == 2.0, 2,
np.where(base_dados['tel2__pe_10'] == 3.0, 2,
np.where(base_dados['tel2__pe_10'] == 4.0, 2,
np.where(base_dados['tel2__pe_10'] == 5.0, 2,
np.where(base_dados['tel2__pe_10'] == 6.0, 2,
np.where(base_dados['tel2__pe_10'] == 7.0, 1,
np.where(base_dados['tel2__pe_10'] == 8.0, 2,
np.where(base_dados['tel2__pe_10'] == 9.0, 1,
 0)))))))))))
base_dados['tel2__pe_10_g_1_2'] = np.where(base_dados['tel2__pe_10_g_1_1'] == 0, 1,
np.where(base_dados['tel2__pe_10_g_1_1'] == 1, 2,
np.where(base_dados['tel2__pe_10_g_1_1'] == 2, 0,
 0)))
base_dados['tel2__pe_10_g_1'] = np.where(base_dados['tel2__pe_10_g_1_2'] == 0, 0,
np.where(base_dados['tel2__pe_10_g_1_2'] == 1, 1,
np.where(base_dados['tel2__pe_10_g_1_2'] == 2, 2,
 0)))
         
         
         
         
         
         
         
        
base_dados['tel2__C'] = np.cos(base_dados['tel2'])
np.where(base_dados['tel2__C'] == 0, -1, base_dados['tel2__C'])
base_dados['tel2__C'] = base_dados['tel2__C'].fillna(-2)
base_dados['tel2__C__pk_8'] = np.where(base_dados['tel2__C'] <= -0.8382456987276166, 0.0,
np.where(np.bitwise_and(base_dados['tel2__C'] > -0.8382456987276166, base_dados['tel2__C'] <= -0.5829225777045096), 1.0,
np.where(np.bitwise_and(base_dados['tel2__C'] > -0.5829225777045096, base_dados['tel2__C'] <= -0.310820207987551), 2.0,
np.where(np.bitwise_and(base_dados['tel2__C'] > -0.310820207987551, base_dados['tel2__C'] <= -0.012910557819010801), 3.0,
np.where(np.bitwise_and(base_dados['tel2__C'] > -0.012910557819010801, base_dados['tel2__C'] <= 0.27794624464248585), 4.0,
np.where(np.bitwise_and(base_dados['tel2__C'] > 0.27794624464248585, base_dados['tel2__C'] <= 0.5367264006136926), 5.0,
np.where(np.bitwise_and(base_dados['tel2__C'] > 0.5367264006136926, base_dados['tel2__C'] <= 0.7978033225987485), 6.0,
np.where(base_dados['tel2__C'] > 0.7978033225987485, 7.0,
 0))))))))
base_dados['tel2__C__pk_8_g_1_1'] = np.where(base_dados['tel2__C__pk_8'] == 0.0, 1,
np.where(base_dados['tel2__C__pk_8'] == 1.0, 0,
np.where(base_dados['tel2__C__pk_8'] == 2.0, 2,
np.where(base_dados['tel2__C__pk_8'] == 3.0, 0,
np.where(base_dados['tel2__C__pk_8'] == 4.0, 1,
np.where(base_dados['tel2__C__pk_8'] == 5.0, 2,
np.where(base_dados['tel2__C__pk_8'] == 6.0, 1,
np.where(base_dados['tel2__C__pk_8'] == 7.0, 2,
 0))))))))
base_dados['tel2__C__pk_8_g_1_2'] = np.where(base_dados['tel2__C__pk_8_g_1_1'] == 0, 1,
np.where(base_dados['tel2__C__pk_8_g_1_1'] == 1, 0,
np.where(base_dados['tel2__C__pk_8_g_1_1'] == 2, 1,
 0)))
base_dados['tel2__C__pk_8_g_1'] = np.where(base_dados['tel2__C__pk_8_g_1_2'] == 0, 0,
np.where(base_dados['tel2__C__pk_8_g_1_2'] == 1, 1,
 0))
                                           
                                           
                                           
                                           
                                           
                                           
                                           
base_dados['tel9__C'] = np.cos(base_dados['tel9'])
np.where(base_dados['tel9__C'] == 0, -1, base_dados['tel9__C'])
base_dados['tel9__C'] = base_dados['tel9__C'].fillna(-2)
base_dados['tel9__C__pe_10'] = np.where(np.bitwise_and(base_dados['tel9__C'] >= -0.9991907421620916, base_dados['tel9__C'] <= 0.19733287688820592), 0.0,
np.where(np.bitwise_and(base_dados['tel9__C'] > 0.19733287688820592, base_dados['tel9__C'] <= 0.3930823715357824), 1.0,
np.where(np.bitwise_and(base_dados['tel9__C'] > 0.3930823715357824, base_dados['tel9__C'] <= 0.5806514964998784), 2.0,
np.where(np.bitwise_and(base_dados['tel9__C'] > 0.5806514964998784, base_dados['tel9__C'] <= 0.7869468876197453), 3.0,
np.where(np.bitwise_and(base_dados['tel9__C'] > 0.7869468876197453, base_dados['tel9__C'] <= 0.9900246781851166), 4.0,
np.where(base_dados['tel9__C'] > 0.9900246781851166, 5.0,
 -2))))))
base_dados['tel9__C__pe_10_g_1_1'] = np.where(base_dados['tel9__C__pe_10'] == -2.0, 0,
np.where(base_dados['tel9__C__pe_10'] == 0.0, 1,
np.where(base_dados['tel9__C__pe_10'] == 1.0, 1,
np.where(base_dados['tel9__C__pe_10'] == 2.0, 1,
np.where(base_dados['tel9__C__pe_10'] == 3.0, 1,
np.where(base_dados['tel9__C__pe_10'] == 4.0, 1,
np.where(base_dados['tel9__C__pe_10'] == 5.0, 1,
 0)))))))
base_dados['tel9__C__pe_10_g_1_2'] = np.where(base_dados['tel9__C__pe_10_g_1_1'] == 0, 0,
np.where(base_dados['tel9__C__pe_10_g_1_1'] == 1, 1,
 0))
base_dados['tel9__C__pe_10_g_1'] = np.where(base_dados['tel9__C__pe_10_g_1_2'] == 0, 0,
np.where(base_dados['tel9__C__pe_10_g_1_2'] == 1, 1,
 0))
                                            
                                            
                                            
                                            
                                            
                                            
                                            
base_dados['tel9__T'] = np.tan(base_dados['tel9'])
np.where(base_dados['tel9__T'] == 0, -1, base_dados['tel9__T'])
base_dados['tel9__T'] = base_dados['tel9__T'].fillna(-2)
base_dados['tel9__T__p_34'] = np.where(base_dados['tel9__T'] <= -1.7090084026364913, 0.0,
np.where(np.bitwise_and(base_dados['tel9__T'] > -1.7090084026364913, base_dados['tel9__T'] <= -0.8287996020958032), 1.0,
np.where(np.bitwise_and(base_dados['tel9__T'] > -0.8287996020958032, base_dados['tel9__T'] <= -0.12254164363877192), 2.0,
np.where(np.bitwise_and(base_dados['tel9__T'] > -0.12254164363877192, base_dados['tel9__T'] <= 0.14231368638598105), 3.0,
np.where(np.bitwise_and(base_dados['tel9__T'] > 0.14231368638598105, base_dados['tel9__T'] <= 0.15666567632077463), 4.0,
np.where(np.bitwise_and(base_dados['tel9__T'] > 0.15666567632077463, base_dados['tel9__T'] <= 0.6381759533765292), 5.0,
np.where(np.bitwise_and(base_dados['tel9__T'] > 0.6381759533765292, base_dados['tel9__T'] <= 1.972430799255809), 6.0,
np.where(base_dados['tel9__T'] > 1.972430799255809, 7.0,
 0))))))))
base_dados['tel9__T__p_34_g_1_1'] = np.where(base_dados['tel9__T__p_34'] == 0.0, 1,
np.where(base_dados['tel9__T__p_34'] == 1.0, 1,
np.where(base_dados['tel9__T__p_34'] == 2.0, 1,
np.where(base_dados['tel9__T__p_34'] == 3.0, 1,
np.where(base_dados['tel9__T__p_34'] == 4.0, 0,
np.where(base_dados['tel9__T__p_34'] == 5.0, 1,
np.where(base_dados['tel9__T__p_34'] == 6.0, 1,
np.where(base_dados['tel9__T__p_34'] == 7.0, 1,
 0))))))))
base_dados['tel9__T__p_34_g_1_2'] = np.where(base_dados['tel9__T__p_34_g_1_1'] == 0, 0,
np.where(base_dados['tel9__T__p_34_g_1_1'] == 1, 1,
 0))
base_dados['tel9__T__p_34_g_1'] = np.where(base_dados['tel9__T__p_34_g_1_2'] == 0, 0,
np.where(base_dados['tel9__T__p_34_g_1_2'] == 1, 1,
 0))
                                           
                                           
                                           
                                           
                                           
                                           
base_dados['mob_dueDate__pu_13'] = np.where(base_dados['mob_dueDate'] <= 23.066521972993588, 0.0,
np.where(np.bitwise_and(base_dados['mob_dueDate'] > 23.066521972993588, base_dados['mob_dueDate'] <= 48.13479852898037), 1.0,
np.where(np.bitwise_and(base_dados['mob_dueDate'] > 48.13479852898037, base_dados['mob_dueDate'] <= 66.9935019931172), 2.0,
np.where(np.bitwise_and(base_dados['mob_dueDate'] > 66.9935019931172, base_dados['mob_dueDate'] <= 97.54854419110895), 3.0,
np.where(np.bitwise_and(base_dados['mob_dueDate'] > 97.54854419110895, base_dados['mob_dueDate'] <= 123.17535377652138), 4.0,
np.where(np.bitwise_and(base_dados['mob_dueDate'] > 123.17535377652138, base_dados['mob_dueDate'] <= 148.17792056434044), 5.0,
np.where(np.bitwise_and(base_dados['mob_dueDate'] > 148.17792056434044, base_dados['mob_dueDate'] <= 167.82514124648998), 6.0,
np.where(np.bitwise_and(base_dados['mob_dueDate'] > 167.82514124648998, base_dados['mob_dueDate'] <= 198.2159090240624), 7.0,
np.where(np.bitwise_and(base_dados['mob_dueDate'] > 198.2159090240624, base_dados['mob_dueDate'] <= 223.54702465272004), 8.0,
np.where(np.bitwise_and(base_dados['mob_dueDate'] > 223.54702465272004, base_dados['mob_dueDate'] <= 246.18403978650102), 9.0,
np.where(np.bitwise_and(base_dados['mob_dueDate'] > 246.18403978650102, base_dados['mob_dueDate'] <= 273.585013112442), 10.0,
np.where(np.bitwise_and(base_dados['mob_dueDate'] > 273.585013112442, base_dados['mob_dueDate'] <= 298.75185432068037), 11.0,
np.where(base_dados['mob_dueDate'] > 298.75185432068037, 12.0,
 0)))))))))))))
base_dados['mob_dueDate__pu_13_g_1_1'] = np.where(base_dados['mob_dueDate__pu_13'] == 0.0, 0,
np.where(base_dados['mob_dueDate__pu_13'] == 1.0, 1,
np.where(base_dados['mob_dueDate__pu_13'] == 2.0, 2,
np.where(base_dados['mob_dueDate__pu_13'] == 3.0, 2,
np.where(base_dados['mob_dueDate__pu_13'] == 4.0, 2,
np.where(base_dados['mob_dueDate__pu_13'] == 5.0, 2,
np.where(base_dados['mob_dueDate__pu_13'] == 6.0, 2,
np.where(base_dados['mob_dueDate__pu_13'] == 7.0, 1,
np.where(base_dados['mob_dueDate__pu_13'] == 8.0, 1,
np.where(base_dados['mob_dueDate__pu_13'] == 9.0, 1,
np.where(base_dados['mob_dueDate__pu_13'] == 10.0, 2,
np.where(base_dados['mob_dueDate__pu_13'] == 11.0, 2,
np.where(base_dados['mob_dueDate__pu_13'] == 12.0, 1,
 0)))))))))))))
base_dados['mob_dueDate__pu_13_g_1_2'] = np.where(base_dados['mob_dueDate__pu_13_g_1_1'] == 0, 2,
np.where(base_dados['mob_dueDate__pu_13_g_1_1'] == 1, 1,
np.where(base_dados['mob_dueDate__pu_13_g_1_1'] == 2, 0,
 0)))
base_dados['mob_dueDate__pu_13_g_1'] = np.where(base_dados['mob_dueDate__pu_13_g_1_2'] == 0, 0,
np.where(base_dados['mob_dueDate__pu_13_g_1_2'] == 1, 1,
np.where(base_dados['mob_dueDate__pu_13_g_1_2'] == 2, 2,
 0)))
         
         
         
         
         
        
base_dados['mob_dueDate__pe_17'] = np.where(base_dados['mob_dueDate'] <= 17.152642837898412, 0.0,
np.where(np.bitwise_and(base_dados['mob_dueDate'] > 17.152642837898412, base_dados['mob_dueDate'] <= 36.96413794046725), 1.0,
np.where(np.bitwise_and(base_dados['mob_dueDate'] > 36.96413794046725, base_dados['mob_dueDate'] <= 54.64006557758506), 2.0,
np.where(np.bitwise_and(base_dados['mob_dueDate'] > 54.64006557758506, base_dados['mob_dueDate'] <= 66.9935019931172), 3.0,
np.where(np.bitwise_and(base_dados['mob_dueDate'] > 66.9935019931172, base_dados['mob_dueDate'] <= 92.2260529695233), 4.0,
np.where(np.bitwise_and(base_dados['mob_dueDate'] > 92.2260529695233, base_dados['mob_dueDate'] <= 111.15046620182785), 5.0,
np.where(np.bitwise_and(base_dados['mob_dueDate'] > 111.15046620182785, base_dados['mob_dueDate'] <= 129.12208779570042), 6.0,
np.where(np.bitwise_and(base_dados['mob_dueDate'] > 129.12208779570042, base_dados['mob_dueDate'] <= 148.17792056434044), 7.0,
np.where(np.bitwise_and(base_dados['mob_dueDate'] > 148.17792056434044, base_dados['mob_dueDate'] <= 166.80663983989024), 8.0,
np.where(np.bitwise_and(base_dados['mob_dueDate'] > 166.80663983989024, base_dados['mob_dueDate'] <= 185.17252004276915), 9.0,
np.where(np.bitwise_and(base_dados['mob_dueDate'] > 185.17252004276915, base_dados['mob_dueDate'] <= 203.86694908648667), 10.0,
np.where(np.bitwise_and(base_dados['mob_dueDate'] > 203.86694908648667, base_dados['mob_dueDate'] <= 222.06855486894625), 11.0,
np.where(np.bitwise_and(base_dados['mob_dueDate'] > 222.06855486894625, base_dados['mob_dueDate'] <= 231.82645544185328), 12.0,
np.where(np.bitwise_and(base_dados['mob_dueDate'] > 231.82645544185328, base_dados['mob_dueDate'] <= 259.1945738837104), 13.0,
np.where(np.bitwise_and(base_dados['mob_dueDate'] > 259.1945738837104, base_dados['mob_dueDate'] <= 277.79043827517637), 14.0,
np.where(np.bitwise_and(base_dados['mob_dueDate'] > 277.79043827517637, base_dados['mob_dueDate'] <= 296.5177222029778), 15.0,
np.where(base_dados['mob_dueDate'] > 296.5177222029778, 16.0,
 -2)))))))))))))))))
base_dados['mob_dueDate__pe_17_g_1_1'] = np.where(base_dados['mob_dueDate__pe_17'] == -2.0, 1,
np.where(base_dados['mob_dueDate__pe_17'] == 0.0, 0,
np.where(base_dados['mob_dueDate__pe_17'] == 1.0, 0,
np.where(base_dados['mob_dueDate__pe_17'] == 2.0, 3,
np.where(base_dados['mob_dueDate__pe_17'] == 3.0, 3,
np.where(base_dados['mob_dueDate__pe_17'] == 4.0, 3,
np.where(base_dados['mob_dueDate__pe_17'] == 5.0, 3,
np.where(base_dados['mob_dueDate__pe_17'] == 6.0, 3,
np.where(base_dados['mob_dueDate__pe_17'] == 7.0, 3,
np.where(base_dados['mob_dueDate__pe_17'] == 8.0, 3,
np.where(base_dados['mob_dueDate__pe_17'] == 9.0, 3,
np.where(base_dados['mob_dueDate__pe_17'] == 10.0, 1,
np.where(base_dados['mob_dueDate__pe_17'] == 11.0, 1,
np.where(base_dados['mob_dueDate__pe_17'] == 12.0, 1,
np.where(base_dados['mob_dueDate__pe_17'] == 13.0, 3,
np.where(base_dados['mob_dueDate__pe_17'] == 14.0, 2,
np.where(base_dados['mob_dueDate__pe_17'] == 15.0, 2,
np.where(base_dados['mob_dueDate__pe_17'] == 16.0, 3,
 0))))))))))))))))))
base_dados['mob_dueDate__pe_17_g_1_2'] = np.where(base_dados['mob_dueDate__pe_17_g_1_1'] == 0, 3,
np.where(base_dados['mob_dueDate__pe_17_g_1_1'] == 1, 2,
np.where(base_dados['mob_dueDate__pe_17_g_1_1'] == 2, 1,
np.where(base_dados['mob_dueDate__pe_17_g_1_1'] == 3, 0,
 0))))
base_dados['mob_dueDate__pe_17_g_1'] = np.where(base_dados['mob_dueDate__pe_17_g_1_2'] == 0, 0,
np.where(base_dados['mob_dueDate__pe_17_g_1_2'] == 1, 1,
np.where(base_dados['mob_dueDate__pe_17_g_1_2'] == 2, 2,
np.where(base_dados['mob_dueDate__pe_17_g_1_2'] == 3, 3,
 0))))
         
         
         
         
         
         
         
base_dados['tel10__p_10'] = np.where(base_dados['tel10'] <= 5134524333.0, 0.0,
np.where(base_dados['tel10'] > 5134524333.0, 1.0,
 0))
base_dados['tel10__p_10_g_1_1'] = np.where(base_dados['tel10__p_10'] == 0.0, 0,
np.where(base_dados['tel10__p_10'] == 1.0, 1,
 0))
base_dados['tel10__p_10_g_1_2'] = np.where(base_dados['tel10__p_10_g_1_1'] == 0, 0,
np.where(base_dados['tel10__p_10_g_1_1'] == 1, 1,
 0))
base_dados['tel10__p_10_g_1'] = np.where(base_dados['tel10__p_10_g_1_2'] == 0, 0,
np.where(base_dados['tel10__p_10_g_1_2'] == 1, 1,
 0))
                                         
                                         
                                         
                                         
                                         
                                         
base_dados['tel10__L'] = np.log(base_dados['tel10'])
np.where(base_dados['tel10__L'] == 0, -1, base_dados['tel10__L'])
base_dados['tel10__L'] = base_dados['tel10__L'].fillna(-2)
base_dados['tel10__L__p_10'] = np.where(base_dados['tel10__L'] <= 22.35902286094301, 0.0,
np.where(base_dados['tel10__L'] > 22.35902286094301, 1.0,
 0))
base_dados['tel10__L__p_10_g_1_1'] = np.where(base_dados['tel10__L__p_10'] == 0.0, 0,
np.where(base_dados['tel10__L__p_10'] == 1.0, 1,
 0))
base_dados['tel10__L__p_10_g_1_2'] = np.where(base_dados['tel10__L__p_10_g_1_1'] == 0, 0,
np.where(base_dados['tel10__L__p_10_g_1_1'] == 1, 1,
 0))
base_dados['tel10__L__p_10_g_1'] = np.where(base_dados['tel10__L__p_10_g_1_2'] == 0, 0,
np.where(base_dados['tel10__L__p_10_g_1_2'] == 1, 1,
 0))
                                            
                                            
                                            
                                            
                                            
                                            
base_dados['tel8__p_25'] = np.where(base_dados['tel8'] <= -3.0, 0.0,
np.where(np.bitwise_and(base_dados['tel8'] > -3.0, base_dados['tel8'] <= 4136731973.0), 1.0,
np.where(np.bitwise_and(base_dados['tel8'] > 4136731973.0, base_dados['tel8'] <= 5134939771.0), 2.0,
np.where(np.bitwise_and(base_dados['tel8'] > 5134939771.0, base_dados['tel8'] <= 35998126064.0), 3.0,
np.where(np.bitwise_and(base_dados['tel8'] > 35998126064.0, base_dados['tel8'] <= 48993547000.0), 4.0,
np.where(np.bitwise_and(base_dados['tel8'] > 48993547000.0, base_dados['tel8'] <= 51989547018.0), 5.0,
np.where(np.bitwise_and(base_dados['tel8'] > 51989547018.0, base_dados['tel8'] <= 51998349025.0), 6.0,
np.where(base_dados['tel8'] > 51998349025.0, 7.0,
 0))))))))
base_dados['tel8__p_25_g_1_1'] = np.where(base_dados['tel8__p_25'] == 0.0, 0,
np.where(base_dados['tel8__p_25'] == 1.0, 1,
np.where(base_dados['tel8__p_25'] == 2.0, 1,
np.where(base_dados['tel8__p_25'] == 3.0, 1,
np.where(base_dados['tel8__p_25'] == 4.0, 1,
np.where(base_dados['tel8__p_25'] == 5.0, 1,
np.where(base_dados['tel8__p_25'] == 6.0, 1,
np.where(base_dados['tel8__p_25'] == 7.0, 1,
 0))))))))
base_dados['tel8__p_25_g_1_2'] = np.where(base_dados['tel8__p_25_g_1_1'] == 0, 0,
np.where(base_dados['tel8__p_25_g_1_1'] == 1, 1,
 0))
base_dados['tel8__p_25_g_1'] = np.where(base_dados['tel8__p_25_g_1_2'] == 0, 0,
np.where(base_dados['tel8__p_25_g_1_2'] == 1, 1,
 0))
                                        
                                        
                                        
                                        
                                        
base_dados['tel8__L'] = np.log(base_dados['tel8'])
np.where(base_dados['tel8__L'] == 0, -1, base_dados['tel8__L'])
base_dados['tel8__L'] = base_dados['tel8__L'].fillna(-2)
base_dados['tel8__L__p_6'] = np.where(base_dados['tel8__L'] <= 23.55624476923622, 0.0,
np.where(base_dados['tel8__L'] > 23.55624476923622, 1.0,
 0))
base_dados['tel8__L__p_6_g_1_1'] = np.where(base_dados['tel8__L__p_6'] == 0.0, 0,
np.where(base_dados['tel8__L__p_6'] == 1.0, 1,
 0))
base_dados['tel8__L__p_6_g_1_2'] = np.where(base_dados['tel8__L__p_6_g_1_1'] == 0, 0,
np.where(base_dados['tel8__L__p_6_g_1_1'] == 1, 1,
 0))
base_dados['tel8__L__p_6_g_1'] = np.where(base_dados['tel8__L__p_6_g_1_2'] == 0, 0,
np.where(base_dados['tel8__L__p_6_g_1_2'] == 1, 1,
 0))


# COMMAND ----------

# MAGIC %md
# MAGIC ### Parte 2 de 2

# COMMAND ----------

base_dados['mob_param_9__S__p_3_g_1_c1_54_1'] = np.where(np.bitwise_and(base_dados['mob_param_9__pu_20_g_1'] == 0, base_dados['mob_param_9__S__p_3_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['mob_param_9__pu_20_g_1'] == 0, base_dados['mob_param_9__S__p_3_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['mob_param_9__pu_20_g_1'] == 0, base_dados['mob_param_9__S__p_3_g_1'] == 2), 2,
np.where(np.bitwise_and(base_dados['mob_param_9__pu_20_g_1'] == 1, base_dados['mob_param_9__S__p_3_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['mob_param_9__pu_20_g_1'] == 1, base_dados['mob_param_9__S__p_3_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['mob_param_9__pu_20_g_1'] == 1, base_dados['mob_param_9__S__p_3_g_1'] == 2), 3,
np.where(np.bitwise_and(base_dados['mob_param_9__pu_20_g_1'] == 2, base_dados['mob_param_9__S__p_3_g_1'] == 0), 4,
np.where(np.bitwise_and(base_dados['mob_param_9__pu_20_g_1'] == 2, base_dados['mob_param_9__S__p_3_g_1'] == 1), 4,
np.where(np.bitwise_and(base_dados['mob_param_9__pu_20_g_1'] == 2, base_dados['mob_param_9__S__p_3_g_1'] == 2), 4,
 0)))))))))
base_dados['mob_param_9__S__p_3_g_1_c1_54_2'] = np.where(base_dados['mob_param_9__S__p_3_g_1_c1_54_1'] == 0, 0,
np.where(base_dados['mob_param_9__S__p_3_g_1_c1_54_1'] == 1, 1,
np.where(base_dados['mob_param_9__S__p_3_g_1_c1_54_1'] == 2, 2,
np.where(base_dados['mob_param_9__S__p_3_g_1_c1_54_1'] == 3, 3,
np.where(base_dados['mob_param_9__S__p_3_g_1_c1_54_1'] == 4, 4,
0)))))
base_dados['mob_param_9__S__p_3_g_1_c1_54'] = np.where(base_dados['mob_param_9__S__p_3_g_1_c1_54_2'] == 0, 0,
np.where(base_dados['mob_param_9__S__p_3_g_1_c1_54_2'] == 1, 1,
np.where(base_dados['mob_param_9__S__p_3_g_1_c1_54_2'] == 2, 2,
np.where(base_dados['mob_param_9__S__p_3_g_1_c1_54_2'] == 3, 3,
np.where(base_dados['mob_param_9__S__p_3_g_1_c1_54_2'] == 4, 4,
 0)))))
         
         
         
         
         
         
                
base_dados['tel1__pu_20_g_1_c1_15_1'] = np.where(np.bitwise_and(base_dados['tel1__pu_20_g_1'] == 0, base_dados['tel1__S__p_6_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['tel1__pu_20_g_1'] == 0, base_dados['tel1__S__p_6_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['tel1__pu_20_g_1'] == 1, base_dados['tel1__S__p_6_g_1'] == 0), 2,
np.where(np.bitwise_and(base_dados['tel1__pu_20_g_1'] == 1, base_dados['tel1__S__p_6_g_1'] == 1), 2,
 0))))
base_dados['tel1__pu_20_g_1_c1_15_2'] = np.where(base_dados['tel1__pu_20_g_1_c1_15_1'] == 0, 0,
np.where(base_dados['tel1__pu_20_g_1_c1_15_1'] == 1, 1,
np.where(base_dados['tel1__pu_20_g_1_c1_15_1'] == 2, 2,
0)))
base_dados['tel1__pu_20_g_1_c1_15'] = np.where(base_dados['tel1__pu_20_g_1_c1_15_2'] == 0, 0,
np.where(base_dados['tel1__pu_20_g_1_c1_15_2'] == 1, 1,
np.where(base_dados['tel1__pu_20_g_1_c1_15_2'] == 2, 2,
 0)))
         
         
         
         
         
         
               
base_dados['tel4__C__p_8_g_1_c1_15_1'] = np.where(np.bitwise_and(base_dados['tel4__L__pe_6_g_1'] == 0, base_dados['tel4__C__p_8_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['tel4__L__pe_6_g_1'] == 0, base_dados['tel4__C__p_8_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['tel4__L__pe_6_g_1'] == 1, base_dados['tel4__C__p_8_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['tel4__L__pe_6_g_1'] == 1, base_dados['tel4__C__p_8_g_1'] == 1), 2,
 0))))
base_dados['tel4__C__p_8_g_1_c1_15_2'] = np.where(base_dados['tel4__C__p_8_g_1_c1_15_1'] == 0, 0,
np.where(base_dados['tel4__C__p_8_g_1_c1_15_1'] == 1, 1,
np.where(base_dados['tel4__C__p_8_g_1_c1_15_1'] == 2, 2,
0)))
base_dados['tel4__C__p_8_g_1_c1_15'] = np.where(base_dados['tel4__C__p_8_g_1_c1_15_2'] == 0, 0,
np.where(base_dados['tel4__C__p_8_g_1_c1_15_2'] == 1, 1,
np.where(base_dados['tel4__C__p_8_g_1_c1_15_2'] == 2, 2,
 0)))
         
         
         
         
         
         
         
base_dados['tel3__pk_4_g_1_c1_26_1'] = np.where(np.bitwise_and(base_dados['tel3__pk_4_g_1'] == 0, base_dados['tel3__S__p_7_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['tel3__pk_4_g_1'] == 0, base_dados['tel3__S__p_7_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['tel3__pk_4_g_1'] == 0, base_dados['tel3__S__p_7_g_1'] == 2), 1,
np.where(np.bitwise_and(base_dados['tel3__pk_4_g_1'] == 1, base_dados['tel3__S__p_7_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['tel3__pk_4_g_1'] == 1, base_dados['tel3__S__p_7_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['tel3__pk_4_g_1'] == 1, base_dados['tel3__S__p_7_g_1'] == 2), 3,
 0))))))
base_dados['tel3__pk_4_g_1_c1_26_2'] = np.where(base_dados['tel3__pk_4_g_1_c1_26_1'] == 0, 0,
np.where(base_dados['tel3__pk_4_g_1_c1_26_1'] == 1, 2,
np.where(base_dados['tel3__pk_4_g_1_c1_26_1'] == 2, 1,
np.where(base_dados['tel3__pk_4_g_1_c1_26_1'] == 3, 3,
0))))
base_dados['tel3__pk_4_g_1_c1_26'] = np.where(base_dados['tel3__pk_4_g_1_c1_26_2'] == 0, 0,
np.where(base_dados['tel3__pk_4_g_1_c1_26_2'] == 1, 1,
np.where(base_dados['tel3__pk_4_g_1_c1_26_2'] == 2, 2,
np.where(base_dados['tel3__pk_4_g_1_c1_26_2'] == 3, 3,
 0))))
         
         
         
         
         
        
base_dados['param_7__pk_8_g_1_c1_36_1'] = np.where(np.bitwise_and(base_dados['param_7__pu_6_g_1'] == 0, base_dados['param_7__pk_8_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['param_7__pu_6_g_1'] == 0, base_dados['param_7__pk_8_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['param_7__pu_6_g_1'] == 1, base_dados['param_7__pk_8_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['param_7__pu_6_g_1'] == 1, base_dados['param_7__pk_8_g_1'] == 1), 2,
 0))))
base_dados['param_7__pk_8_g_1_c1_36_2'] = np.where(base_dados['param_7__pk_8_g_1_c1_36_1'] == 0, 0,
np.where(base_dados['param_7__pk_8_g_1_c1_36_1'] == 1, 1,
np.where(base_dados['param_7__pk_8_g_1_c1_36_1'] == 2, 2,
0)))
base_dados['param_7__pk_8_g_1_c1_36'] = np.where(base_dados['param_7__pk_8_g_1_c1_36_2'] == 0, 0,
np.where(base_dados['param_7__pk_8_g_1_c1_36_2'] == 1, 1,
np.where(base_dados['param_7__pk_8_g_1_c1_36_2'] == 2, 2,
 0)))
         
         
         
         
         
         
         
         
base_dados['tel2__C__pk_8_g_1_c1_13_1'] = np.where(np.bitwise_and(base_dados['tel2__pe_10_g_1'] == 0, base_dados['tel2__C__pk_8_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['tel2__pe_10_g_1'] == 0, base_dados['tel2__C__pk_8_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['tel2__pe_10_g_1'] == 1, base_dados['tel2__C__pk_8_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['tel2__pe_10_g_1'] == 1, base_dados['tel2__C__pk_8_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['tel2__pe_10_g_1'] == 2, base_dados['tel2__C__pk_8_g_1'] == 0), 2,
np.where(np.bitwise_and(base_dados['tel2__pe_10_g_1'] == 2, base_dados['tel2__C__pk_8_g_1'] == 1), 3,
 0))))))
base_dados['tel2__C__pk_8_g_1_c1_13_2'] = np.where(base_dados['tel2__C__pk_8_g_1_c1_13_1'] == 0, 0,
np.where(base_dados['tel2__C__pk_8_g_1_c1_13_1'] == 1, 1,
np.where(base_dados['tel2__C__pk_8_g_1_c1_13_1'] == 2, 2,
np.where(base_dados['tel2__C__pk_8_g_1_c1_13_1'] == 3, 3,
0))))
base_dados['tel2__C__pk_8_g_1_c1_13'] = np.where(base_dados['tel2__C__pk_8_g_1_c1_13_2'] == 0, 0,
np.where(base_dados['tel2__C__pk_8_g_1_c1_13_2'] == 1, 1,
np.where(base_dados['tel2__C__pk_8_g_1_c1_13_2'] == 2, 2,
np.where(base_dados['tel2__C__pk_8_g_1_c1_13_2'] == 3, 3,
 0))))
         
         
         
         
         
         
         
base_dados['tel9__C__pe_10_g_1_c1_29_1'] = np.where(np.bitwise_and(base_dados['tel9__C__pe_10_g_1'] == 0, base_dados['tel9__T__p_34_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['tel9__C__pe_10_g_1'] == 0, base_dados['tel9__T__p_34_g_1'] == 1), 0,
np.where(np.bitwise_and(base_dados['tel9__C__pe_10_g_1'] == 1, base_dados['tel9__T__p_34_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['tel9__C__pe_10_g_1'] == 1, base_dados['tel9__T__p_34_g_1'] == 1), 1,
 0))))
base_dados['tel9__C__pe_10_g_1_c1_29_2'] = np.where(base_dados['tel9__C__pe_10_g_1_c1_29_1'] == 0, 0,
np.where(base_dados['tel9__C__pe_10_g_1_c1_29_1'] == 1, 1,
0))
base_dados['tel9__C__pe_10_g_1_c1_29'] = np.where(base_dados['tel9__C__pe_10_g_1_c1_29_2'] == 0, 0,
np.where(base_dados['tel9__C__pe_10_g_1_c1_29_2'] == 1, 1,
 0))
                                                  
                                                  
                                                  
                                                  
                                                  
                                                  
                                                  
base_dados['mob_dueDate__pe_17_g_1_c1_54_1'] = np.where(np.bitwise_and(base_dados['mob_dueDate__pu_13_g_1'] == 0, base_dados['mob_dueDate__pe_17_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['mob_dueDate__pu_13_g_1'] == 0, base_dados['mob_dueDate__pe_17_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['mob_dueDate__pu_13_g_1'] == 0, base_dados['mob_dueDate__pe_17_g_1'] == 2), 2,
np.where(np.bitwise_and(base_dados['mob_dueDate__pu_13_g_1'] == 0, base_dados['mob_dueDate__pe_17_g_1'] == 3), 2,
np.where(np.bitwise_and(base_dados['mob_dueDate__pu_13_g_1'] == 1, base_dados['mob_dueDate__pe_17_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['mob_dueDate__pu_13_g_1'] == 1, base_dados['mob_dueDate__pe_17_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['mob_dueDate__pu_13_g_1'] == 1, base_dados['mob_dueDate__pe_17_g_1'] == 2), 3,
np.where(np.bitwise_and(base_dados['mob_dueDate__pu_13_g_1'] == 1, base_dados['mob_dueDate__pe_17_g_1'] == 3), 4,
np.where(np.bitwise_and(base_dados['mob_dueDate__pu_13_g_1'] == 2, base_dados['mob_dueDate__pe_17_g_1'] == 0), 4,
np.where(np.bitwise_and(base_dados['mob_dueDate__pu_13_g_1'] == 2, base_dados['mob_dueDate__pe_17_g_1'] == 1), 4,
np.where(np.bitwise_and(base_dados['mob_dueDate__pu_13_g_1'] == 2, base_dados['mob_dueDate__pe_17_g_1'] == 2), 4,
np.where(np.bitwise_and(base_dados['mob_dueDate__pu_13_g_1'] == 2, base_dados['mob_dueDate__pe_17_g_1'] == 3), 4,
 0))))))))))))
base_dados['mob_dueDate__pe_17_g_1_c1_54_2'] = np.where(base_dados['mob_dueDate__pe_17_g_1_c1_54_1'] == 0, 0,
np.where(base_dados['mob_dueDate__pe_17_g_1_c1_54_1'] == 1, 1,
np.where(base_dados['mob_dueDate__pe_17_g_1_c1_54_1'] == 2, 2,
np.where(base_dados['mob_dueDate__pe_17_g_1_c1_54_1'] == 3, 3,
np.where(base_dados['mob_dueDate__pe_17_g_1_c1_54_1'] == 4, 4,
0)))))
base_dados['mob_dueDate__pe_17_g_1_c1_54'] = np.where(base_dados['mob_dueDate__pe_17_g_1_c1_54_2'] == 0, 0,
np.where(base_dados['mob_dueDate__pe_17_g_1_c1_54_2'] == 1, 1,
np.where(base_dados['mob_dueDate__pe_17_g_1_c1_54_2'] == 2, 2,
np.where(base_dados['mob_dueDate__pe_17_g_1_c1_54_2'] == 3, 3,
np.where(base_dados['mob_dueDate__pe_17_g_1_c1_54_2'] == 4, 4,
 0)))))
         
         
         
         
         
         
base_dados['tel10__L__p_10_g_1_c1_8_1'] = np.where(np.bitwise_and(base_dados['tel10__p_10_g_1'] == 0, base_dados['tel10__L__p_10_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['tel10__p_10_g_1'] == 0, base_dados['tel10__L__p_10_g_1'] == 1), 0,
np.where(np.bitwise_and(base_dados['tel10__p_10_g_1'] == 1, base_dados['tel10__L__p_10_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['tel10__p_10_g_1'] == 1, base_dados['tel10__L__p_10_g_1'] == 1), 1,
 0))))
base_dados['tel10__L__p_10_g_1_c1_8_2'] = np.where(base_dados['tel10__L__p_10_g_1_c1_8_1'] == 0, 0,
np.where(base_dados['tel10__L__p_10_g_1_c1_8_1'] == 1, 1,
0))
base_dados['tel10__L__p_10_g_1_c1_8'] = np.where(base_dados['tel10__L__p_10_g_1_c1_8_2'] == 0, 0,
np.where(base_dados['tel10__L__p_10_g_1_c1_8_2'] == 1, 1,
 0))
                                                 
                                                 
                                                 
                                                 
                                                 
                                                 
base_dados['tel8__L__p_6_g_1_c1_28_1'] = np.where(np.bitwise_and(base_dados['tel8__p_25_g_1'] == 0, base_dados['tel8__L__p_6_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['tel8__p_25_g_1'] == 0, base_dados['tel8__L__p_6_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['tel8__p_25_g_1'] == 1, base_dados['tel8__L__p_6_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['tel8__p_25_g_1'] == 1, base_dados['tel8__L__p_6_g_1'] == 1), 2,
 0))))
base_dados['tel8__L__p_6_g_1_c1_28_2'] = np.where(base_dados['tel8__L__p_6_g_1_c1_28_1'] == 0, 0,
np.where(base_dados['tel8__L__p_6_g_1_c1_28_1'] == 1, 1,
np.where(base_dados['tel8__L__p_6_g_1_c1_28_1'] == 2, 2,
0)))
base_dados['tel8__L__p_6_g_1_c1_28'] = np.where(base_dados['tel8__L__p_6_g_1_c1_28_2'] == 0, 0,
np.where(base_dados['tel8__L__p_6_g_1_c1_28_2'] == 1, 1,
np.where(base_dados['tel8__L__p_6_g_1_c1_28_2'] == 2, 2,
 0)))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Mantendo apenas as variáveis do modelo

# COMMAND ----------

import pickle
modelo=pickle.load(open(caminho_base + 'model_fit_colombo_r_forest.sav', 'rb'))

base_teste_c0 = base_dados[[chave,'mob_dueDate__pe_17_g_1_c1_54', 'mob_param_9__S__p_3_g_1_c1_54', 'param_7__pk_8_g_1_c1_36', 'tel1__pu_20_g_1_c1_15', 'tel10__L__p_10_g_1_c1_8', 'tel2__C__pk_8_g_1_c1_13', 'tel3__pk_4_g_1_c1_26', 'tel4__C__p_8_g_1_c1_15', 'tel8__L__p_6_g_1_c1_28', 'tel9__C__pe_10_g_1_c1_29', 'activation_gh38']]

var_fin_c0=['mob_dueDate__pe_17_g_1_c1_54', 'mob_param_9__S__p_3_g_1_c1_54', 'param_7__pk_8_g_1_c1_36', 'tel1__pu_20_g_1_c1_15', 'tel10__L__p_10_g_1_c1_8', 'tel2__C__pk_8_g_1_c1_13', 'tel3__pk_4_g_1_c1_26', 'tel4__C__p_8_g_1_c1_15', 'tel8__L__p_6_g_1_c1_28', 'tel9__C__pe_10_g_1_c1_29', 'activation_gh38']

print(var_fin_c0)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Rodando Regressão Logística

# COMMAND ----------

# Datasets de treino e de teste
x_teste = base_teste_c0[var_fin_c0]
z_teste = base_teste_c0[chave]

# Previsões
valores_previstos = modelo.predict(x_teste)

probabilidades = modelo.predict_proba(x_teste)
data_prob = pd.DataFrame({'P_1': probabilidades[:, 1]})

z_teste1 = z_teste.reset_index(drop=True)
data_prob1 = data_prob.reset_index(drop=True)

x_teste2 = pd.concat([z_teste1,data_prob1], axis=1)

x_teste2


# COMMAND ----------

# MAGIC %md
# MAGIC # Modelo de Grupo Homogêneo

# COMMAND ----------

    
x_teste2['GH'] = np.where(x_teste2['P_1'] <= 0.0243, 0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.0243, x_teste2['P_1'] <= 0.0548), 1,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.0548, x_teste2['P_1'] <= 0.1292), 2,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.1292, x_teste2['P_1'] <= 0.2493), 3,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.2493, x_teste2['P_1'] <= 0.5129), 4,5)))))

x_teste2

# COMMAND ----------

x_teste2.groupby(['GH'])['P_1'].count()

# COMMAND ----------

