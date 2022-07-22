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
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from datetime import datetime

%matplotlib inline

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inserindo hiperparâmetros do Algoritmo

# COMMAND ----------

## Parâmetros do Algoritmo

#Variável chave-primaria
chave = 'DOCUMENTO:ID_DIVIDA'

#Nome da Base de Dados
N_Base = "df_aleatorio__.csv"

#Caminho da base de dados
caminho_base = "Base_Dados_Ferramenta/Avenidas/"

#Separador
separador_ = ";"

#Decimal
decimal_ = "."

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importação da Base de Dados

# COMMAND ----------

#carregar o arquivo em formato tabela
base_dados = pd.read_csv(caminho_base+N_Base, sep=separador_, decimal=decimal_)
base_dados = base_dados[[chave,'DETALHES_CLIENTES_RETIRA_MAGNO', 'DETALHES_CLIENTES_RETI_DIGITAL', 'DETALHES_CLIENTES_EMISSAO_BOLETO_DEMAIS_PARCELAS', 'DETALHES_CLIENTES_CAMP_DEZ2018', 'DETALHES_CLIENTES_FATURA', 'VALOR_DIVIDA', 'DETALHES_CLIENTES_ACAO_FIM_ANO19', 'DETALHES_CLIENTES_RET_MOV_MAGNO', 'DETALHES_CLIENTES_CAMPANHA20161214']]

#string
base_dados['DETALHES_CLIENTES_RETI_DIGITAL'] = base_dados['DETALHES_CLIENTES_RETI_DIGITAL'].replace(np.nan, '-3')
base_dados['DETALHES_CLIENTES_RETIRA_MAGNO'] = base_dados['DETALHES_CLIENTES_RETIRA_MAGNO'].replace(np.nan, '-3')
base_dados['DETALHES_CLIENTES_EMISSAO_BOLETO_DEMAIS_PARCELAS'] = base_dados['DETALHES_CLIENTES_EMISSAO_BOLETO_DEMAIS_PARCELAS'].replace(np.nan, '-3')
base_dados['DETALHES_CLIENTES_CAMP_DEZ2018'] = base_dados['DETALHES_CLIENTES_CAMP_DEZ2018'].replace(np.nan, '-3')
base_dados['DETALHES_CLIENTES_FATURA'] = base_dados['DETALHES_CLIENTES_FATURA'].replace(np.nan, '-3')
base_dados['DETALHES_CLIENTES_ACAO_FIM_ANO19'] = base_dados['DETALHES_CLIENTES_ACAO_FIM_ANO19'].replace(np.nan, '-3')
base_dados['DETALHES_CLIENTES_RET_MOV_MAGNO'] = base_dados['DETALHES_CLIENTES_RET_MOV_MAGNO'].replace(np.nan, '-3')

#numericas
base_dados['VALOR_DIVIDA'] = base_dados['VALOR_DIVIDA'].replace(np.nan, '-3')
base_dados['DETALHES_CLIENTES_CAMPANHA20161214'] = base_dados['DETALHES_CLIENTES_CAMPANHA20161214'].replace(np.nan, '-3')

base_dados = base_dados.apply(pd.to_numeric, errors='ignore')

base_dados.fillna(-3)

base_dados['VALOR_DIVIDA'] = base_dados['VALOR_DIVIDA'].astype(float)
base_dados['DETALHES_CLIENTES_CAMPANHA20161214'] = base_dados['DETALHES_CLIENTES_CAMPANHA20161214'].astype(np.int64)

base_dados.drop_duplicates(keep=False, inplace=True)

print("shape da Base de Dados:",base_dados.shape)

base_dados.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gerando variáveis Categóricas

# COMMAND ----------

         
base_dados['DETALHES_CLIENTES_FATURA_gh30'] = np.where(base_dados['DETALHES_CLIENTES_FATURA'] == '-3', 0,
np.where(base_dados['DETALHES_CLIENTES_FATURA'] == 'OFERTA', 1,0))

base_dados['DETALHES_CLIENTES_FATURA_gh31'] = np.where(base_dados['DETALHES_CLIENTES_FATURA_gh30'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_FATURA_gh30'] == 1, 1,0))

base_dados['DETALHES_CLIENTES_FATURA_gh32'] = np.where(base_dados['DETALHES_CLIENTES_FATURA_gh31'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_FATURA_gh31'] == 1, 1,0))

base_dados['DETALHES_CLIENTES_FATURA_gh33'] = np.where(base_dados['DETALHES_CLIENTES_FATURA_gh32'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_FATURA_gh32'] == 1, 1,0))

base_dados['DETALHES_CLIENTES_FATURA_gh34'] = np.where(base_dados['DETALHES_CLIENTES_FATURA_gh33'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_FATURA_gh33'] == 1, 1,0))

base_dados['DETALHES_CLIENTES_FATURA_gh35'] = np.where(base_dados['DETALHES_CLIENTES_FATURA_gh34'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_FATURA_gh34'] == 1, 1,0))

base_dados['DETALHES_CLIENTES_FATURA_gh36'] = np.where(base_dados['DETALHES_CLIENTES_FATURA_gh35'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_FATURA_gh35'] == 1, 1,0))

base_dados['DETALHES_CLIENTES_FATURA_gh37'] = np.where(base_dados['DETALHES_CLIENTES_FATURA_gh36'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_FATURA_gh36'] == 1, 1,0))

base_dados['DETALHES_CLIENTES_FATURA_gh38'] = np.where(base_dados['DETALHES_CLIENTES_FATURA_gh37'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_FATURA_gh37'] == 1, 1,0))
                                                           
                                                           
                                                           
                                                           
                                                           
                                                           
base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh30'] = np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL'] == '-3', 0,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL'] == '130819', 1,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL'] == '20621', 2,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL'] == 'CORRECAO', 3,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL'] == 'CORRECAO3', 4,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL'] == 'DEVOLVER', 5,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL'] == 'ENVIO BVS', 6,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL'] == 'MALTA2012', 7,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL'] == 'MASTERSERVICE', 8,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL'] == 'MOV080920', 9,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL'] == 'QQ010921', 10,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL'] == 'SERV050320', 11,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL'] == 'VAR DIG', 12,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL'] == 'VERRES 050421', 13,
0))))))))))))))

base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh31'] = np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh30'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh30'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh30'] == 2, 2,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh30'] == 3, 3,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh30'] == 4, 4,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh30'] == 5, 5,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh30'] == 6, 6,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh30'] == 7, 7,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh30'] == 8, 8,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh30'] == 9, 9,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh30'] == 10, 10,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh30'] == 11, 11,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh30'] == 12, 12,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh30'] == 13, 13,
0))))))))))))))

base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh32'] = np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh31'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh31'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh31'] == 2, 2,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh31'] == 3, 3,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh31'] == 4, 4,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh31'] == 5, 5,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh31'] == 6, 6,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh31'] == 7, 7,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh31'] == 8, 8,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh31'] == 9, 9,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh31'] == 10, 10,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh31'] == 11, 11,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh31'] == 12, 12,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh31'] == 13, 13,
0))))))))))))))

base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh33'] = np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh32'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh32'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh32'] == 2, 2,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh32'] == 3, 3,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh32'] == 4, 4,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh32'] == 5, 5,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh32'] == 6, 6,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh32'] == 7, 7,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh32'] == 8, 8,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh32'] == 9, 9,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh32'] == 10, 10,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh32'] == 11, 11,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh32'] == 12, 12,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh32'] == 13, 13,
0))))))))))))))

base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh34'] = np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh33'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh33'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh33'] == 2, 1,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh33'] == 3, 0,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh33'] == 4, 1,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh33'] == 5, 5,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh33'] == 6, 0,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh33'] == 7, 0,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh33'] == 8, 1,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh33'] == 9, 0,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh33'] == 10, 10,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh33'] == 11, 0,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh33'] == 12, 0,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh33'] == 13, 0,
0))))))))))))))

base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh35'] = np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh34'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh34'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh34'] == 5, 2,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh34'] == 10, 3,
0))))

base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh36'] = np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh35'] == 0, 3,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh35'] == 1, 0,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh35'] == 2, 1,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh35'] == 3, 1,
0))))

base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh37'] = np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh36'] == 0, 1,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh36'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh36'] == 3, 2,
0)))

base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh38'] = np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh37'] == 1, 0,
np.where(base_dados['DETALHES_CLIENTES_RETI_DIGITAL_gh37'] == 2, 1,
0))
                                                             
                                                             
                                                             
                                                             
                                                             
base_dados['DETALHES_CLIENTES_RETIRA_MAGNO_gh30'] = np.where(base_dados['DETALHES_CLIENTES_RETIRA_MAGNO'] == '-3', 0,
np.where(base_dados['DETALHES_CLIENTES_RETIRA_MAGNO'] == 'ADIMP060319', 1,
np.where(base_dados['DETALHES_CLIENTES_RETIRA_MAGNO'] == 'FASE33', 2,
np.where(base_dados['DETALHES_CLIENTES_RETIRA_MAGNO'] == 'NELSONW060519', 3,
np.where(base_dados['DETALHES_CLIENTES_RETIRA_MAGNO'] == 'SERVIC060319', 4,
0)))))

base_dados['DETALHES_CLIENTES_RETIRA_MAGNO_gh31'] = np.where(base_dados['DETALHES_CLIENTES_RETIRA_MAGNO_gh30'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_RETIRA_MAGNO_gh30'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_RETIRA_MAGNO_gh30'] == 2, 2,
np.where(base_dados['DETALHES_CLIENTES_RETIRA_MAGNO_gh30'] == 3, 3,
np.where(base_dados['DETALHES_CLIENTES_RETIRA_MAGNO_gh30'] == 4, 4,
0)))))

base_dados['DETALHES_CLIENTES_RETIRA_MAGNO_gh32'] = np.where(base_dados['DETALHES_CLIENTES_RETIRA_MAGNO_gh31'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_RETIRA_MAGNO_gh31'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_RETIRA_MAGNO_gh31'] == 2, 2,
np.where(base_dados['DETALHES_CLIENTES_RETIRA_MAGNO_gh31'] == 3, 3,
np.where(base_dados['DETALHES_CLIENTES_RETIRA_MAGNO_gh31'] == 4, 4,
0)))))

base_dados['DETALHES_CLIENTES_RETIRA_MAGNO_gh33'] = np.where(base_dados['DETALHES_CLIENTES_RETIRA_MAGNO_gh32'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_RETIRA_MAGNO_gh32'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_RETIRA_MAGNO_gh32'] == 2, 2,
np.where(base_dados['DETALHES_CLIENTES_RETIRA_MAGNO_gh32'] == 3, 3,
np.where(base_dados['DETALHES_CLIENTES_RETIRA_MAGNO_gh32'] == 4, 4,
0)))))

base_dados['DETALHES_CLIENTES_RETIRA_MAGNO_gh34'] = np.where(base_dados['DETALHES_CLIENTES_RETIRA_MAGNO_gh33'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_RETIRA_MAGNO_gh33'] == 1, 2,
np.where(base_dados['DETALHES_CLIENTES_RETIRA_MAGNO_gh33'] == 2, 2,
np.where(base_dados['DETALHES_CLIENTES_RETIRA_MAGNO_gh33'] == 3, 4,
np.where(base_dados['DETALHES_CLIENTES_RETIRA_MAGNO_gh33'] == 4, 0,
0)))))

base_dados['DETALHES_CLIENTES_RETIRA_MAGNO_gh35'] = np.where(base_dados['DETALHES_CLIENTES_RETIRA_MAGNO_gh34'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_RETIRA_MAGNO_gh34'] == 2, 1,
np.where(base_dados['DETALHES_CLIENTES_RETIRA_MAGNO_gh34'] == 4, 2,
0)))

base_dados['DETALHES_CLIENTES_RETIRA_MAGNO_gh36'] = np.where(base_dados['DETALHES_CLIENTES_RETIRA_MAGNO_gh35'] == 0, 2,
np.where(base_dados['DETALHES_CLIENTES_RETIRA_MAGNO_gh35'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_RETIRA_MAGNO_gh35'] == 2, 0,
0)))

base_dados['DETALHES_CLIENTES_RETIRA_MAGNO_gh37'] = np.where(base_dados['DETALHES_CLIENTES_RETIRA_MAGNO_gh36'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_RETIRA_MAGNO_gh36'] == 1, 0,
np.where(base_dados['DETALHES_CLIENTES_RETIRA_MAGNO_gh36'] == 2, 2,
0)))

base_dados['DETALHES_CLIENTES_RETIRA_MAGNO_gh38'] = np.where(base_dados['DETALHES_CLIENTES_RETIRA_MAGNO_gh37'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_RETIRA_MAGNO_gh37'] == 2, 1,
0))
                                                              
                                                              
                                                              
                                                              
                                                              
                                                              
base_dados['DETALHES_CLIENTES_ACAO_FIM_ANO19_gh30'] = np.where(base_dados['DETALHES_CLIENTES_ACAO_FIM_ANO19'] == '-3', 0,
np.where(base_dados['DETALHES_CLIENTES_ACAO_FIM_ANO19'] == 'DIGITAL', 1,
np.where(base_dados['DETALHES_CLIENTES_ACAO_FIM_ANO19'] == 'IMPRESSO', 2,
0)))

base_dados['DETALHES_CLIENTES_ACAO_FIM_ANO19_gh31'] = np.where(base_dados['DETALHES_CLIENTES_ACAO_FIM_ANO19_gh30'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_ACAO_FIM_ANO19_gh30'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_ACAO_FIM_ANO19_gh30'] == 2, 2,
0)))

base_dados['DETALHES_CLIENTES_ACAO_FIM_ANO19_gh32'] = np.where(base_dados['DETALHES_CLIENTES_ACAO_FIM_ANO19_gh31'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_ACAO_FIM_ANO19_gh31'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_ACAO_FIM_ANO19_gh31'] == 2, 2,
0)))

base_dados['DETALHES_CLIENTES_ACAO_FIM_ANO19_gh33'] = np.where(base_dados['DETALHES_CLIENTES_ACAO_FIM_ANO19_gh32'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_ACAO_FIM_ANO19_gh32'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_ACAO_FIM_ANO19_gh32'] == 2, 2,
0)))

base_dados['DETALHES_CLIENTES_ACAO_FIM_ANO19_gh34'] = np.where(base_dados['DETALHES_CLIENTES_ACAO_FIM_ANO19_gh33'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_ACAO_FIM_ANO19_gh33'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_ACAO_FIM_ANO19_gh33'] == 2, 0,
0)))

base_dados['DETALHES_CLIENTES_ACAO_FIM_ANO19_gh35'] = np.where(base_dados['DETALHES_CLIENTES_ACAO_FIM_ANO19_gh34'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_ACAO_FIM_ANO19_gh34'] == 1, 1,
0))

base_dados['DETALHES_CLIENTES_ACAO_FIM_ANO19_gh36'] = np.where(base_dados['DETALHES_CLIENTES_ACAO_FIM_ANO19_gh35'] == 0, 1,
np.where(base_dados['DETALHES_CLIENTES_ACAO_FIM_ANO19_gh35'] == 1, 0,
0))

base_dados['DETALHES_CLIENTES_ACAO_FIM_ANO19_gh37'] = np.where(base_dados['DETALHES_CLIENTES_ACAO_FIM_ANO19_gh36'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_ACAO_FIM_ANO19_gh36'] == 1, 1,
0))

base_dados['DETALHES_CLIENTES_ACAO_FIM_ANO19_gh38'] = np.where(base_dados['DETALHES_CLIENTES_ACAO_FIM_ANO19_gh37'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_ACAO_FIM_ANO19_gh37'] == 1, 1,
0))
                                                               
                                                               
                                                               
                                                               
                                                               
                                                               
base_dados['DETALHES_CLIENTES_CAMP_DEZ2018_gh30'] = np.where(base_dados['DETALHES_CLIENTES_CAMP_DEZ2018'] == '-3', 0,
np.where(base_dados['DETALHES_CLIENTES_CAMP_DEZ2018'] == '1 A 2', 1,
np.where(base_dados['DETALHES_CLIENTES_CAMP_DEZ2018'] == '3', 2,
np.where(base_dados['DETALHES_CLIENTES_CAMP_DEZ2018'] == '4', 3,
np.where(base_dados['DETALHES_CLIENTES_CAMP_DEZ2018'] == '5', 4,
np.where(base_dados['DETALHES_CLIENTES_CAMP_DEZ2018'] == '6 A 7', 5,
np.where(base_dados['DETALHES_CLIENTES_CAMP_DEZ2018'] == '9', 6,
0)))))))

base_dados['DETALHES_CLIENTES_CAMP_DEZ2018_gh31'] = np.where(base_dados['DETALHES_CLIENTES_CAMP_DEZ2018_gh30'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_CAMP_DEZ2018_gh30'] == 1, 0,
np.where(base_dados['DETALHES_CLIENTES_CAMP_DEZ2018_gh30'] == 2, 2,
np.where(base_dados['DETALHES_CLIENTES_CAMP_DEZ2018_gh30'] == 3, 3,
np.where(base_dados['DETALHES_CLIENTES_CAMP_DEZ2018_gh30'] == 4, 4,
np.where(base_dados['DETALHES_CLIENTES_CAMP_DEZ2018_gh30'] == 5, 5,
np.where(base_dados['DETALHES_CLIENTES_CAMP_DEZ2018_gh30'] == 6, 6,
0)))))))

base_dados['DETALHES_CLIENTES_CAMP_DEZ2018_gh32'] = np.where(base_dados['DETALHES_CLIENTES_CAMP_DEZ2018_gh31'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_CAMP_DEZ2018_gh31'] == 2, 1,
np.where(base_dados['DETALHES_CLIENTES_CAMP_DEZ2018_gh31'] == 3, 2,
np.where(base_dados['DETALHES_CLIENTES_CAMP_DEZ2018_gh31'] == 4, 3,
np.where(base_dados['DETALHES_CLIENTES_CAMP_DEZ2018_gh31'] == 5, 4,
np.where(base_dados['DETALHES_CLIENTES_CAMP_DEZ2018_gh31'] == 6, 5,
0))))))

base_dados['DETALHES_CLIENTES_CAMP_DEZ2018_gh33'] = np.where(base_dados['DETALHES_CLIENTES_CAMP_DEZ2018_gh32'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_CAMP_DEZ2018_gh32'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_CAMP_DEZ2018_gh32'] == 2, 2,
np.where(base_dados['DETALHES_CLIENTES_CAMP_DEZ2018_gh32'] == 3, 3,
np.where(base_dados['DETALHES_CLIENTES_CAMP_DEZ2018_gh32'] == 4, 4,
np.where(base_dados['DETALHES_CLIENTES_CAMP_DEZ2018_gh32'] == 5, 5,
0))))))

base_dados['DETALHES_CLIENTES_CAMP_DEZ2018_gh34'] = np.where(base_dados['DETALHES_CLIENTES_CAMP_DEZ2018_gh33'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_CAMP_DEZ2018_gh33'] == 1, 0,
np.where(base_dados['DETALHES_CLIENTES_CAMP_DEZ2018_gh33'] == 2, 2,
np.where(base_dados['DETALHES_CLIENTES_CAMP_DEZ2018_gh33'] == 3, 0,
np.where(base_dados['DETALHES_CLIENTES_CAMP_DEZ2018_gh33'] == 4, 0,
np.where(base_dados['DETALHES_CLIENTES_CAMP_DEZ2018_gh33'] == 5, 6,
0))))))

base_dados['DETALHES_CLIENTES_CAMP_DEZ2018_gh35'] = np.where(base_dados['DETALHES_CLIENTES_CAMP_DEZ2018_gh34'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_CAMP_DEZ2018_gh34'] == 2, 1,
np.where(base_dados['DETALHES_CLIENTES_CAMP_DEZ2018_gh34'] == 6, 2,
0)))

base_dados['DETALHES_CLIENTES_CAMP_DEZ2018_gh36'] = np.where(base_dados['DETALHES_CLIENTES_CAMP_DEZ2018_gh35'] == 0, 2,
np.where(base_dados['DETALHES_CLIENTES_CAMP_DEZ2018_gh35'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_CAMP_DEZ2018_gh35'] == 2, 0,
0)))

base_dados['DETALHES_CLIENTES_CAMP_DEZ2018_gh37'] = np.where(base_dados['DETALHES_CLIENTES_CAMP_DEZ2018_gh36'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_CAMP_DEZ2018_gh36'] == 1, 0,
np.where(base_dados['DETALHES_CLIENTES_CAMP_DEZ2018_gh36'] == 2, 2,
0)))

base_dados['DETALHES_CLIENTES_CAMP_DEZ2018_gh38'] = np.where(base_dados['DETALHES_CLIENTES_CAMP_DEZ2018_gh37'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_CAMP_DEZ2018_gh37'] == 2, 1,
0))
                                                             
                                                             
                                                             
                                                             
                                                             
base_dados['DETALHES_CLIENTES_RET_MOV_MAGNO_gh30'] = np.where(base_dados['DETALHES_CLIENTES_RET_MOV_MAGNO'] == '-3', 0,
np.where(base_dados['DETALHES_CLIENTES_RET_MOV_MAGNO'] == '10819', 1,
np.where(base_dados['DETALHES_CLIENTES_RET_MOV_MAGNO'] == 'NELSON2', 2,
np.where(base_dados['DETALHES_CLIENTES_RET_MOV_MAGNO'] == 'RET2', 3,
np.where(base_dados['DETALHES_CLIENTES_RET_MOV_MAGNO'] == 'VERRESCHI', 4,
0)))))

base_dados['DETALHES_CLIENTES_RET_MOV_MAGNO_gh31'] = np.where(base_dados['DETALHES_CLIENTES_RET_MOV_MAGNO_gh30'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_RET_MOV_MAGNO_gh30'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_RET_MOV_MAGNO_gh30'] == 2, 1,
np.where(base_dados['DETALHES_CLIENTES_RET_MOV_MAGNO_gh30'] == 3, 3,
np.where(base_dados['DETALHES_CLIENTES_RET_MOV_MAGNO_gh30'] == 4, 4,
0)))))

base_dados['DETALHES_CLIENTES_RET_MOV_MAGNO_gh32'] = np.where(base_dados['DETALHES_CLIENTES_RET_MOV_MAGNO_gh31'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_RET_MOV_MAGNO_gh31'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_RET_MOV_MAGNO_gh31'] == 3, 2,
np.where(base_dados['DETALHES_CLIENTES_RET_MOV_MAGNO_gh31'] == 4, 3,
0))))

base_dados['DETALHES_CLIENTES_RET_MOV_MAGNO_gh33'] = np.where(base_dados['DETALHES_CLIENTES_RET_MOV_MAGNO_gh32'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_RET_MOV_MAGNO_gh32'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_RET_MOV_MAGNO_gh32'] == 2, 2,
np.where(base_dados['DETALHES_CLIENTES_RET_MOV_MAGNO_gh32'] == 3, 3,
0))))

base_dados['DETALHES_CLIENTES_RET_MOV_MAGNO_gh34'] = np.where(base_dados['DETALHES_CLIENTES_RET_MOV_MAGNO_gh33'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_RET_MOV_MAGNO_gh33'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_RET_MOV_MAGNO_gh33'] == 2, 0,
np.where(base_dados['DETALHES_CLIENTES_RET_MOV_MAGNO_gh33'] == 3, 4,
0))))

base_dados['DETALHES_CLIENTES_RET_MOV_MAGNO_gh35'] = np.where(base_dados['DETALHES_CLIENTES_RET_MOV_MAGNO_gh34'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_RET_MOV_MAGNO_gh34'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_RET_MOV_MAGNO_gh34'] == 4, 2,
0)))

base_dados['DETALHES_CLIENTES_RET_MOV_MAGNO_gh36'] = np.where(base_dados['DETALHES_CLIENTES_RET_MOV_MAGNO_gh35'] == 0, 2,
np.where(base_dados['DETALHES_CLIENTES_RET_MOV_MAGNO_gh35'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_RET_MOV_MAGNO_gh35'] == 2, 0,
0)))

base_dados['DETALHES_CLIENTES_RET_MOV_MAGNO_gh37'] = np.where(base_dados['DETALHES_CLIENTES_RET_MOV_MAGNO_gh36'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_RET_MOV_MAGNO_gh36'] == 1, 0,
np.where(base_dados['DETALHES_CLIENTES_RET_MOV_MAGNO_gh36'] == 2, 2,
0)))

base_dados['DETALHES_CLIENTES_RET_MOV_MAGNO_gh38'] = np.where(base_dados['DETALHES_CLIENTES_RET_MOV_MAGNO_gh37'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_RET_MOV_MAGNO_gh37'] == 2, 1,
0))
                                                                
                                                                
                                                                
                                                                
                                                                
                                                                
                                                                
base_dados['DETALHES_CLIENTES_EMISSAO_BOLETO_DEMAIS_PARCELAS_gh30'] = np.where(base_dados['DETALHES_CLIENTES_EMISSAO_BOLETO_DEMAIS_PARCELAS'] == '-3', 0,
np.where(base_dados['DETALHES_CLIENTES_EMISSAO_BOLETO_DEMAIS_PARCELAS'] == 'CORREIO', 1,
np.where(base_dados['DETALHES_CLIENTES_EMISSAO_BOLETO_DEMAIS_PARCELAS'] == 'LOJA', 2,
0)))

base_dados['DETALHES_CLIENTES_EMISSAO_BOLETO_DEMAIS_PARCELAS_gh31'] = np.where(base_dados['DETALHES_CLIENTES_EMISSAO_BOLETO_DEMAIS_PARCELAS_gh30'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_EMISSAO_BOLETO_DEMAIS_PARCELAS_gh30'] == 1, 1,
np.where(base_dados['DETALHES_CLIENTES_EMISSAO_BOLETO_DEMAIS_PARCELAS_gh30'] == 2, 1,
0)))

base_dados['DETALHES_CLIENTES_EMISSAO_BOLETO_DEMAIS_PARCELAS_gh32'] = np.where(base_dados['DETALHES_CLIENTES_EMISSAO_BOLETO_DEMAIS_PARCELAS_gh31'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_EMISSAO_BOLETO_DEMAIS_PARCELAS_gh31'] == 1, 1,
0))

base_dados['DETALHES_CLIENTES_EMISSAO_BOLETO_DEMAIS_PARCELAS_gh33'] = np.where(base_dados['DETALHES_CLIENTES_EMISSAO_BOLETO_DEMAIS_PARCELAS_gh32'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_EMISSAO_BOLETO_DEMAIS_PARCELAS_gh32'] == 1, 1,
0))

base_dados['DETALHES_CLIENTES_EMISSAO_BOLETO_DEMAIS_PARCELAS_gh34'] = np.where(base_dados['DETALHES_CLIENTES_EMISSAO_BOLETO_DEMAIS_PARCELAS_gh33'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_EMISSAO_BOLETO_DEMAIS_PARCELAS_gh33'] == 1, 1,
0))

base_dados['DETALHES_CLIENTES_EMISSAO_BOLETO_DEMAIS_PARCELAS_gh35'] = np.where(base_dados['DETALHES_CLIENTES_EMISSAO_BOLETO_DEMAIS_PARCELAS_gh34'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_EMISSAO_BOLETO_DEMAIS_PARCELAS_gh34'] == 1, 1,
0))

base_dados['DETALHES_CLIENTES_EMISSAO_BOLETO_DEMAIS_PARCELAS_gh36'] = np.where(base_dados['DETALHES_CLIENTES_EMISSAO_BOLETO_DEMAIS_PARCELAS_gh35'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_EMISSAO_BOLETO_DEMAIS_PARCELAS_gh35'] == 1, 1,
0))

base_dados['DETALHES_CLIENTES_EMISSAO_BOLETO_DEMAIS_PARCELAS_gh37'] = np.where(base_dados['DETALHES_CLIENTES_EMISSAO_BOLETO_DEMAIS_PARCELAS_gh36'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_EMISSAO_BOLETO_DEMAIS_PARCELAS_gh36'] == 1, 1,
0))

base_dados['DETALHES_CLIENTES_EMISSAO_BOLETO_DEMAIS_PARCELAS_gh38'] = np.where(base_dados['DETALHES_CLIENTES_EMISSAO_BOLETO_DEMAIS_PARCELAS_gh37'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_EMISSAO_BOLETO_DEMAIS_PARCELAS_gh37'] == 1, 1,
0))
                                                            
                                                            
                                                            
                                                            
                                                            
base_dados['DETALHES_CLIENTES_CAMPANHA20161214_gh30'] = np.where(base_dados['DETALHES_CLIENTES_CAMPANHA20161214'] == -3, 0,
np.where(base_dados['DETALHES_CLIENTES_CAMPANHA20161214'] == 20161214, 1,
0))

base_dados['DETALHES_CLIENTES_CAMPANHA20161214_gh31'] = np.where(base_dados['DETALHES_CLIENTES_CAMPANHA20161214_gh30'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_CAMPANHA20161214_gh30'] == 1, 1,
0))

base_dados['DETALHES_CLIENTES_CAMPANHA20161214_gh32'] = np.where(base_dados['DETALHES_CLIENTES_CAMPANHA20161214_gh31'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_CAMPANHA20161214_gh31'] == 1, 1,
0))

base_dados['DETALHES_CLIENTES_CAMPANHA20161214_gh33'] = np.where(base_dados['DETALHES_CLIENTES_CAMPANHA20161214_gh32'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_CAMPANHA20161214_gh32'] == 1, 1,
0))

base_dados['DETALHES_CLIENTES_CAMPANHA20161214_gh34'] = np.where(base_dados['DETALHES_CLIENTES_CAMPANHA20161214_gh33'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_CAMPANHA20161214_gh33'] == 1, 1,
0))

base_dados['DETALHES_CLIENTES_CAMPANHA20161214_gh35'] = np.where(base_dados['DETALHES_CLIENTES_CAMPANHA20161214_gh34'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_CAMPANHA20161214_gh34'] == 1, 1,
0))

base_dados['DETALHES_CLIENTES_CAMPANHA20161214_gh36'] = np.where(base_dados['DETALHES_CLIENTES_CAMPANHA20161214_gh35'] == 0, 1,
np.where(base_dados['DETALHES_CLIENTES_CAMPANHA20161214_gh35'] == 1, 0,
0))

base_dados['DETALHES_CLIENTES_CAMPANHA20161214_gh37'] = np.where(base_dados['DETALHES_CLIENTES_CAMPANHA20161214_gh36'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_CAMPANHA20161214_gh36'] == 1, 1,
0))

base_dados['DETALHES_CLIENTES_CAMPANHA20161214_gh38'] = np.where(base_dados['DETALHES_CLIENTES_CAMPANHA20161214_gh37'] == 0, 0,
np.where(base_dados['DETALHES_CLIENTES_CAMPANHA20161214_gh37'] == 1, 1,
0))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gerando variáveis numéricas contínuas

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parte 1 de 2

# COMMAND ----------

base_dados['VALOR_DIVIDA__R'] = np.sqrt(base_dados['VALOR_DIVIDA'])
np.where(base_dados['VALOR_DIVIDA__R'] == 0, -1, base_dados['VALOR_DIVIDA__R'])
base_dados['VALOR_DIVIDA__R'] = base_dados['VALOR_DIVIDA__R'].fillna(-2)

base_dados['VALOR_DIVIDA__R__p_3'] = np.where(np.bitwise_and(base_dados['VALOR_DIVIDA__R'] >= -2.0, base_dados['VALOR_DIVIDA__R'] <= 159.3361227091961), 0.0,
np.where(np.bitwise_and(base_dados['VALOR_DIVIDA__R'] > 159.3361227091961, base_dados['VALOR_DIVIDA__R'] <= 198.38598740838526), 1.0,
np.where(base_dados['VALOR_DIVIDA__R'] > 198.38598740838526, 2.0,
 -2)))

base_dados['VALOR_DIVIDA__R__p_3_g_1_1'] = np.where(base_dados['VALOR_DIVIDA__R__p_3'] == -2.0, 0,
np.where(base_dados['VALOR_DIVIDA__R__p_3'] == 0.0, 0,
np.where(base_dados['VALOR_DIVIDA__R__p_3'] == 1.0, 1,
np.where(base_dados['VALOR_DIVIDA__R__p_3'] == 2.0, 0,
 0))))

base_dados['VALOR_DIVIDA__R__p_3_g_1_2'] = np.where(base_dados['VALOR_DIVIDA__R__p_3_g_1_1'] == 0, 1,
np.where(base_dados['VALOR_DIVIDA__R__p_3_g_1_1'] == 1, 0,
 0))

base_dados['VALOR_DIVIDA__R__p_3_g_1'] = np.where(base_dados['VALOR_DIVIDA__R__p_3_g_1_2'] == 0, 0,
np.where(base_dados['VALOR_DIVIDA__R__p_3_g_1_2'] == 1, 1,
 0))
                                                  
                                                  
                                                  
                                                  
                                                  
base_dados['VALOR_DIVIDA__L'] = np.log(base_dados['VALOR_DIVIDA'])
np.where(base_dados['VALOR_DIVIDA__L'] == 0, -1, base_dados['VALOR_DIVIDA__L'])
base_dados['VALOR_DIVIDA__L'] = base_dados['VALOR_DIVIDA__L'].fillna(-2)

base_dados['VALOR_DIVIDA__L__p_17'] = np.where(np.bitwise_and(base_dados['VALOR_DIVIDA__L'] >= -2.0, base_dados['VALOR_DIVIDA__L'] <= 8.727454116899434), 0.0,
np.where(np.bitwise_and(base_dados['VALOR_DIVIDA__L'] > 8.727454116899434, base_dados['VALOR_DIVIDA__L'] <= 9.435481925953065), 1.0,
np.where(np.bitwise_and(base_dados['VALOR_DIVIDA__L'] > 9.435481925953065, base_dados['VALOR_DIVIDA__L'] <= 9.788301142727011), 2.0,
np.where(np.bitwise_and(base_dados['VALOR_DIVIDA__L'] > 9.788301142727011, base_dados['VALOR_DIVIDA__L'] <= 9.95456082272013), 3.0,
np.where(np.bitwise_and(base_dados['VALOR_DIVIDA__L'] > 9.95456082272013, base_dados['VALOR_DIVIDA__L'] <= 10.091294039560653), 4.0,
np.where(np.bitwise_and(base_dados['VALOR_DIVIDA__L'] > 10.091294039560653, base_dados['VALOR_DIVIDA__L'] <= 11.024105921466852), 14.0,
np.where(np.bitwise_and(base_dados['VALOR_DIVIDA__L'] > 11.024105921466852, base_dados['VALOR_DIVIDA__L'] <= 11.324871509327872), 15.0,
np.where(base_dados['VALOR_DIVIDA__L'] > 11.324871509327872, 16.0,
 -2))))))))

base_dados['VALOR_DIVIDA__L__p_17_g_1_1'] = np.where(base_dados['VALOR_DIVIDA__L__p_17'] == -2.0, 1,
np.where(base_dados['VALOR_DIVIDA__L__p_17'] == 0.0, 2,
np.where(base_dados['VALOR_DIVIDA__L__p_17'] == 1.0, 1,
np.where(base_dados['VALOR_DIVIDA__L__p_17'] == 2.0, 1,
np.where(base_dados['VALOR_DIVIDA__L__p_17'] == 3.0, 0,
np.where(base_dados['VALOR_DIVIDA__L__p_17'] == 4.0, 0,
np.where(base_dados['VALOR_DIVIDA__L__p_17'] == 14.0, 0,
np.where(base_dados['VALOR_DIVIDA__L__p_17'] == 15.0, 2,
np.where(base_dados['VALOR_DIVIDA__L__p_17'] == 16.0, 1,
 0)))))))))
         
base_dados['VALOR_DIVIDA__L__p_17_g_1_2'] = np.where(base_dados['VALOR_DIVIDA__L__p_17_g_1_1'] == 0, 1,
np.where(base_dados['VALOR_DIVIDA__L__p_17_g_1_1'] == 1, 2,
np.where(base_dados['VALOR_DIVIDA__L__p_17_g_1_1'] == 2, 0,
 0)))
         
base_dados['VALOR_DIVIDA__L__p_17_g_1'] = np.where(base_dados['VALOR_DIVIDA__L__p_17_g_1_2'] == 0, 0,
np.where(base_dados['VALOR_DIVIDA__L__p_17_g_1_2'] == 1, 1,
np.where(base_dados['VALOR_DIVIDA__L__p_17_g_1_2'] == 2, 2,
 0)))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parte 2 de 2

# COMMAND ----------


base_dados['VALOR_DIVIDA__L__p_17_g_1_c1_14_1'] = np.where(np.bitwise_and(base_dados['VALOR_DIVIDA__R__p_3_g_1'] == 0, base_dados['VALOR_DIVIDA__L__p_17_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['VALOR_DIVIDA__R__p_3_g_1'] == 0, base_dados['VALOR_DIVIDA__L__p_17_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['VALOR_DIVIDA__R__p_3_g_1'] == 0, base_dados['VALOR_DIVIDA__L__p_17_g_1'] == 2), 2,
np.where(np.bitwise_and(base_dados['VALOR_DIVIDA__R__p_3_g_1'] == 1, base_dados['VALOR_DIVIDA__L__p_17_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['VALOR_DIVIDA__R__p_3_g_1'] == 1, base_dados['VALOR_DIVIDA__L__p_17_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['VALOR_DIVIDA__R__p_3_g_1'] == 1, base_dados['VALOR_DIVIDA__L__p_17_g_1'] == 2), 3,
1))))))
base_dados['VALOR_DIVIDA__L__p_17_g_1_c1_14_2'] = np.where(base_dados['VALOR_DIVIDA__L__p_17_g_1_c1_14_1'] == 0, 0,
np.where(base_dados['VALOR_DIVIDA__L__p_17_g_1_c1_14_1'] == 1, 1,
np.where(base_dados['VALOR_DIVIDA__L__p_17_g_1_c1_14_1'] == 2, 2,
np.where(base_dados['VALOR_DIVIDA__L__p_17_g_1_c1_14_1'] == 3, 3,
0))))

base_dados['VALOR_DIVIDA__L__p_17_g_1_c1_14'] = np.where(base_dados['VALOR_DIVIDA__L__p_17_g_1_c1_14_2'] == 0, 0,
np.where(base_dados['VALOR_DIVIDA__L__p_17_g_1_c1_14_2'] == 1, 1,
np.where(base_dados['VALOR_DIVIDA__L__p_17_g_1_c1_14_2'] == 2, 2,
np.where(base_dados['VALOR_DIVIDA__L__p_17_g_1_c1_14_2'] == 3, 3,
0))))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Mantendo apenas as variáveis do modelo

# COMMAND ----------

import pickle
modelo=pickle.load(open(caminho_base + 'model_fit_avenidas.sav', 'rb'))

base_teste_c0 = base_dados[[chave,'DETALHES_CLIENTES_RETIRA_MAGNO_gh38', 'DETALHES_CLIENTES_RETI_DIGITAL_gh38', 'DETALHES_CLIENTES_EMISSAO_BOLETO_DEMAIS_PARCELAS_gh38','DETALHES_CLIENTES_CAMP_DEZ2018_gh38', 'DETALHES_CLIENTES_FATURA_gh38','DETALHES_CLIENTES_ACAO_FIM_ANO19_gh38','DETALHES_CLIENTES_RET_MOV_MAGNO_gh38','DETALHES_CLIENTES_CAMPANHA20161214_gh38','VALOR_DIVIDA__L__p_17_g_1_c1_14']]

var_fin_c0=['DETALHES_CLIENTES_RETIRA_MAGNO_gh38', 'DETALHES_CLIENTES_RETI_DIGITAL_gh38', 'DETALHES_CLIENTES_EMISSAO_BOLETO_DEMAIS_PARCELAS_gh38','DETALHES_CLIENTES_CAMP_DEZ2018_gh38', 'DETALHES_CLIENTES_FATURA_gh38','DETALHES_CLIENTES_ACAO_FIM_ANO19_gh38','DETALHES_CLIENTES_RET_MOV_MAGNO_gh38','DETALHES_CLIENTES_CAMPANHA20161214_gh38','VALOR_DIVIDA__L__p_17_g_1_c1_14']

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


#transformando a variável original pela raiz quadrada
x_teste2['P_1_R'] = np.sqrt(x_teste2['P_1']) 
       
#marcando os valores igual a 0 como -1
x_teste2['P_1_R'] = np.where(x_teste2['P_1_R'] == 0, -1, x_teste2['P_1_R'])        
        
#marcando os valores igual a missing como -2
x_teste2['P_1_R'] = x_teste2['P_1_R'].fillna(-2)

x_teste2['P_1_R_p_8_g_1'] = np.where(x_teste2['P_1_R'] <= 0.34560633, 0.0,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.34560633, x_teste2['P_1_R'] <= 0.428114168), 1.0,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.428114168, x_teste2['P_1_R'] <= 0.481930164), 2.0,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.481930164, x_teste2['P_1_R'] <= 0.582274876), 3.0,
    np.where(x_teste2['P_1_R'] > 0.582274876,4,1)))))

x_teste2['P_1_pe_17_g_1'] = np.where(x_teste2['P_1'] <= 0.120324881, 0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.120324881, x_teste2['P_1'] <= 0.328140162), 2.0,
    np.where(x_teste2['P_1'] > 0.328140162, 3.0,0)))

x_teste2['GH'] = np.where(np.bitwise_and(x_teste2['P_1_R_p_8_g_1'] == 0, x_teste2['P_1_pe_17_g_1'] == 0), 0,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_8_g_1'] == 0, x_teste2['P_1_pe_17_g_1'] == 2), 1,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_8_g_1'] == 0, x_teste2['P_1_pe_17_g_1'] == 3), 1,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_8_g_1'] == 1, x_teste2['P_1_pe_17_g_1'] == 0), 1,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_8_g_1'] == 1, x_teste2['P_1_pe_17_g_1'] == 2), 2,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_8_g_1'] == 1, x_teste2['P_1_pe_17_g_1'] == 3), 2,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_8_g_1'] == 2, x_teste2['P_1_pe_17_g_1'] == 0), 2,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_8_g_1'] == 2, x_teste2['P_1_pe_17_g_1'] == 2), 3,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_8_g_1'] == 2, x_teste2['P_1_pe_17_g_1'] == 3), 4,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_8_g_1'] == 3, x_teste2['P_1_pe_17_g_1'] == 0), 3,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_8_g_1'] == 3, x_teste2['P_1_pe_17_g_1'] == 2), 4,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_8_g_1'] == 3, x_teste2['P_1_pe_17_g_1'] == 3), 5,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_8_g_1'] == 4, x_teste2['P_1_pe_17_g_1'] == 0), 4,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_8_g_1'] == 4, x_teste2['P_1_pe_17_g_1'] == 2), 5,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_8_g_1'] == 4, x_teste2['P_1_pe_17_g_1'] == 3), 5,
    1)))))))))))))))

del x_teste2['P_1_R']
del x_teste2['P_1_R_p_8_g_1']
del x_teste2['P_1_pe_17_g_1']

x_teste2


# COMMAND ----------

