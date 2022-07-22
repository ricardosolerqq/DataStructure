# Databricks notebook source
# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

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
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inserindo hiperparâmetros do Algoritmo

# COMMAND ----------

## Parâmetros do Algoritmo

#Variável chave-primaria
chave = 'ID_DEVEDOR'

#Nome da Base de Dados

#Caminho da base de dados
caminho_base = "/dbfs/mnt/ml-prd/ml-data/propensaodeal/santander/trusted_PJ/"
list_base = os.listdir(caminho_base)

#Nome da Base de Dados
N_Base = max(list_base)

#Separador
separador_ = ";"

#Decimal
decimal_ = "."


pickle_path = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/santander/pickle_model/'

outputpath = 'mnt/ml-prd/ml-data/propensaodeal/santander/outputPJ/'
outputpath_dbfs = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/santander/outputPJ/'

N_Base

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importação da Base de Dados

# COMMAND ----------

#carregar o arquivo em formato tabela
base_dados = pd.read_csv(caminho_base+N_Base, sep=separador_, decimal=decimal_)

base_dados['TIPO_PRODUTO'] = np.where(base_dados['Descricao_Produto'].str.contains('CARTAO')==True,'CARTAO',
                                    np.where(base_dados['Descricao_Produto'].str.contains('MC')==True,'CARTAO',
                                    np.where(base_dados['Descricao_Produto'].str.contains('VS')==True,'CARTAO',
                                    np.where(base_dados['Descricao_Produto'].str.contains('MASTERCARD')==True,'CARTAO',
                                    np.where(base_dados['Descricao_Produto'].str.contains('MASTER')==True,'CARTAO',
                                    np.where(base_dados['Descricao_Produto'].str.contains('VISA')==True,'CARTAO',
                                    np.where(base_dados['Descricao_Produto'].str.contains('GOLD')==True,'CARTAO',
                                    np.where(base_dados['Descricao_Produto'].str.contains('BLACK')==True,'CARTAO',
                                    np.where(base_dados['Descricao_Produto'].str.contains('PLATINUM')==True,'CARTAO',
                                    np.where(base_dados['Descricao_Produto'].str.contains('INFINITE')==True,'CARTAO',
                                    np.where(base_dados['Descricao_Produto'].str.contains('NACIONAL')==True,'CARTAO',
                                    np.where(base_dados['Descricao_Produto'].str.contains('INTERNACIONAL')==True,'CARTAO',
                                    np.where(base_dados['Descricao_Produto'].str.contains('CONTA')==True,'CCG',            
                                    np.where(base_dados['Descricao_Produto'].str.contains('ADI')==True,'AD',
                                    np.where(base_dados['Descricao_Produto'].str.contains('COMPO')==True,'REFIN',
                                    np.where(base_dados['Descricao_Produto'].str.contains('GIRO')==True,'GIRO',
                                    np.where(base_dados['Descricao_Produto'].str.contains('CHEQUE')==True,'CHEQUE ESPECIAL',
                                    np.where(base_dados['Descricao_Produto'].str.contains('REFIN')==True,'REFIN','OUTROS'))))))))))))))))))




base_dados['Telefone1'] = np.where(base_dados['Telefone1_skip:hot']==True,'skip_hot',
                                    np.where(base_dados['Telefone1_skip:alto']==True,'skip_alto',
                                    np.where(base_dados['Telefone1_skip:medio']==True,'skip_medio',
                                    np.where(base_dados['Telefone1_skip:baixo']==True,'skip_baixo',
                                    np.where(base_dados['Telefone1_skip:nhot']==True,'skip_nhot','sem_tags')))))


base_dados['Telefone10'] = np.where(base_dados['Telefone10_skip:hot']==True,'skip_hot',
                                    np.where(base_dados['Telefone10_skip:alto']==True,'skip_alto',
                                    np.where(base_dados['Telefone10_skip:medio']==True,'skip_medio',
                                    np.where(base_dados['Telefone10_skip:baixo']==True,'skip_baixo',
                                    np.where(base_dados['Telefone10_skip:nhot']==True,'skip_nhot','sem_tags')))))


base_dados['Email1'] = np.where(base_dados['Email1_skip:hot']==True,'skip_hot',
                                    np.where(base_dados['Email1_skip:alto']==True,'skip_alto',
                                    np.where(base_dados['Email1_skip:medio']==True,'skip_medio',
                                    np.where(base_dados['Email1_skip:baixo']==True,'skip_baixo',
                                    np.where(base_dados['Email1_skip:nhot']==True,'skip_nhot','sem_tags')))))

del base_dados['Email1_skip:hot']
del base_dados['Email1_skip:alto']
del base_dados['Email1_skip:medio']
del base_dados['Email1_skip:baixo']
del base_dados['Email1_skip:nhot']
del base_dados['Email1_sem_tags']
del base_dados['Telefone1_skip:hot']
del base_dados['Telefone1_skip:alto']
del base_dados['Telefone1_skip:medio']
del base_dados['Telefone1_skip:baixo']
del base_dados['Telefone1_skip:nhot']
del base_dados['Telefone1_sem_tags']
del base_dados['Telefone10_skip:hot']
del base_dados['Telefone10_skip:alto']
del base_dados['Telefone10_skip:medio']
del base_dados['Telefone10_skip:baixo']
del base_dados['Telefone10_skip:nhot']
del base_dados['Telefone10_sem_tags']

base_dados['Indicador_Correntista'] = base_dados['Indicador_Correntista'].map({True:1,False:0},na_action=None)
base_dados['Divida_Outro_Credor'] = base_dados['Divida_Outro_Credor'].map({True:1,False:0},na_action=None)
base_dados['Divida_Mesmo_Credor'] = base_dados['Divida_Mesmo_Credor'].map({True:1,False:0},na_action=None)
                                             
base_dados['PRODUCT_1'] = np.where(base_dados['product'] > 9999,9999,base_dados['product'])

base_dados.fillna(-3)

base_dados['Cep1'] = base_dados['Cep1'].replace(np.nan, '')

base_dados['Cep1_2'] = base_dados['Cep1'].str[:2]

base_dados = base_dados.apply(pd.to_numeric, errors='ignore')

base_dados['Cep1'] = base_dados['Cep1'].replace(np.nan, -3)

base_dados.fillna(-3)

base_dados.drop_duplicates(keep='first', inplace=True)

base_dados = base_dados[[chave, 'QTD_AUTH', 'Email1', 'Divida_Outro_Credor', 'Telefone10','Desconto_Padrao','Telefone1','TIPO_PRODUTO','PRODUCT_1','UF1','Divida_Mesmo_Credor', 'Descricao_Produto', 'Codigo_Politica','Saldo_Devedor_Contrato', 'Cep1_2']]
print("shape da Base de Dados:",base_dados.shape)

base_dados.head()
#61216

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gerando variáveis Categóricas

# COMMAND ----------


base_dados['Descricao_Produto_gh30'] = np.where(base_dados['Descricao_Produto'] == 'ADIANTAMENTOS A DEPOSITANTES', 0,
np.where(base_dados['Descricao_Produto'] == 'BR GIRO AUTOMATICO BANESPA CONVERSAO', 1,
np.where(base_dados['Descricao_Produto'] == 'BR GIRO PARCELADO BANESPA CONVERSAO', 2,
np.where(base_dados['Descricao_Produto'] == 'BR REFIN CONVERSAO', 3,
np.where(base_dados['Descricao_Produto'] == 'CAPITAL DE GIRO', 4,
np.where(base_dados['Descricao_Produto'] == 'CAPITAL DE GIRO MESOP', 5,
np.where(base_dados['Descricao_Produto'] == 'CARTAO SANTANDER BUSINESS MC', 6,
np.where(base_dados['Descricao_Produto'] == 'CARTAO SANTANDER MEI', 7,
np.where(base_dados['Descricao_Produto'] == 'CDC MAQUINAS/EQUIPAMENTOS P.JURIDICA-PRE', 8,
np.where(base_dados['Descricao_Produto'] == 'CHEQUE EMPRESA BNP', 9,
np.where(base_dados['Descricao_Produto'] == 'CONTA CORRENTE GARANTIDA', 10,
np.where(base_dados['Descricao_Produto'] == 'CREDITO AUTOMATICO BANESPA', 11,
np.where(base_dados['Descricao_Produto'] == 'DESCONTO DE CHEQUE PRE-DATADO', 12,
np.where(base_dados['Descricao_Produto'] == 'DESCONTO DE DUPLICATA', 13,
np.where(base_dados['Descricao_Produto'] == 'DESCONTO DE VISANET', 14,
np.where(base_dados['Descricao_Produto'] == 'EMPRESARIAL MASTERCARD', 15,
np.where(base_dados['Descricao_Produto'] == 'FINANCIAMENTO DE VEICULOS', 16,
np.where(base_dados['Descricao_Produto'] == 'GIRO AUTOMATICO BANESPA', 17,
np.where(base_dados['Descricao_Produto'] == 'GIRO BONIFICADO', 18,
np.where(base_dados['Descricao_Produto'] == 'GIRO CARTOES PARCELADO PRE', 19,
np.where(base_dados['Descricao_Produto'] == 'GIRO PARCELADO BANESPA', 20,
np.where(base_dados['Descricao_Produto'] == 'GIRO PARCELADO PRE', 21,
np.where(base_dados['Descricao_Produto'] == 'GIRO REORGANIZACAO', 22,
np.where(base_dados['Descricao_Produto'] == 'GIRO SOLUCAO PARCELADO', 23,
np.where(base_dados['Descricao_Produto'] == 'GIRO UNIFICADO', 24,
np.where(base_dados['Descricao_Produto'] == 'REAL EMPRESARIAL BUSINESS VISA', 25,
np.where(base_dados['Descricao_Produto'] == 'REAL HOTEL VISA', 26,
np.where(base_dados['Descricao_Produto'] == 'REFIN', 27,
np.where(base_dados['Descricao_Produto'] == 'SANTANDER EMP BUSINESS MASTERCARD', 28,
np.where(base_dados['Descricao_Produto'] == 'SANTANDER NEGOCIOS & EMPRESAS MASTERCARD', 29,
np.where(base_dados['Descricao_Produto'] == 'SANTANDER NEGOCIOS & EMPRESAS PLAT MC', 30,
np.where(base_dados['Descricao_Produto'] == 'SANTANDER NEGOCIOS & EMPRESAS PLAT VISA', 31,
np.where(base_dados['Descricao_Produto'] == 'SANTANDER NEGOCIOS & EMPRESAS VISA', 32,
np.where(base_dados['Descricao_Produto'] == 'VISA BUSINESS CARD', 33,
np.where(base_dados['Descricao_Produto'] == 'VISA EMPRESARIAL CENTRALIZADO', 34,
np.where(base_dados['Descricao_Produto'] == 'VISA EMPRESARIAL CENTRALIZADO MI REAL', 35,
np.where(base_dados['Descricao_Produto'] == 'VISA EMPRESARIAL INDIVIDUALIZADO', 36,
0)))))))))))))))))))))))))))))))))))))

base_dados['Descricao_Produto_gh31'] = np.where(base_dados['Descricao_Produto_gh30'] == 0, 0,
np.where(base_dados['Descricao_Produto_gh30'] == 1, 1,
np.where(base_dados['Descricao_Produto_gh30'] == 2, 1,
np.where(base_dados['Descricao_Produto_gh30'] == 3, 1,
np.where(base_dados['Descricao_Produto_gh30'] == 4, 4,
np.where(base_dados['Descricao_Produto_gh30'] == 5, 5,
np.where(base_dados['Descricao_Produto_gh30'] == 6, 6,
np.where(base_dados['Descricao_Produto_gh30'] == 7, 7,
np.where(base_dados['Descricao_Produto_gh30'] == 8, 7,
np.where(base_dados['Descricao_Produto_gh30'] == 9, 9,
np.where(base_dados['Descricao_Produto_gh30'] == 10, 10,
np.where(base_dados['Descricao_Produto_gh30'] == 11, 10,
np.where(base_dados['Descricao_Produto_gh30'] == 12, 10,
np.where(base_dados['Descricao_Produto_gh30'] == 13, 10,
np.where(base_dados['Descricao_Produto_gh30'] == 14, 10,
np.where(base_dados['Descricao_Produto_gh30'] == 15, 10,
np.where(base_dados['Descricao_Produto_gh30'] == 16, 16,
np.where(base_dados['Descricao_Produto_gh30'] == 17, 17,
np.where(base_dados['Descricao_Produto_gh30'] == 18, 18,
np.where(base_dados['Descricao_Produto_gh30'] == 19, 18,
np.where(base_dados['Descricao_Produto_gh30'] == 20, 20,
np.where(base_dados['Descricao_Produto_gh30'] == 21, 21,
np.where(base_dados['Descricao_Produto_gh30'] == 22, 21,
np.where(base_dados['Descricao_Produto_gh30'] == 23, 21,
np.where(base_dados['Descricao_Produto_gh30'] == 24, 24,
np.where(base_dados['Descricao_Produto_gh30'] == 25, 25,
np.where(base_dados['Descricao_Produto_gh30'] == 26, 25,
np.where(base_dados['Descricao_Produto_gh30'] == 27, 27,
np.where(base_dados['Descricao_Produto_gh30'] == 28, 28,
np.where(base_dados['Descricao_Produto_gh30'] == 29, 29,
np.where(base_dados['Descricao_Produto_gh30'] == 30, 30,
np.where(base_dados['Descricao_Produto_gh30'] == 31, 31,
np.where(base_dados['Descricao_Produto_gh30'] == 32, 32,
np.where(base_dados['Descricao_Produto_gh30'] == 33, 33,
np.where(base_dados['Descricao_Produto_gh30'] == 34, 34,
np.where(base_dados['Descricao_Produto_gh30'] == 35, 34,
np.where(base_dados['Descricao_Produto_gh30'] == 36, 34,
0)))))))))))))))))))))))))))))))))))))


base_dados['Descricao_Produto_gh32'] = np.where(base_dados['Descricao_Produto_gh31'] == 0, 0,
np.where(base_dados['Descricao_Produto_gh31'] == 1, 1,
np.where(base_dados['Descricao_Produto_gh31'] == 4, 2,
np.where(base_dados['Descricao_Produto_gh31'] == 5, 3,
np.where(base_dados['Descricao_Produto_gh31'] == 6, 4,
np.where(base_dados['Descricao_Produto_gh31'] == 7, 5,
np.where(base_dados['Descricao_Produto_gh31'] == 9, 6,
np.where(base_dados['Descricao_Produto_gh31'] == 10, 7,
np.where(base_dados['Descricao_Produto_gh31'] == 16, 8,
np.where(base_dados['Descricao_Produto_gh31'] == 17, 9,
np.where(base_dados['Descricao_Produto_gh31'] == 18, 10,
np.where(base_dados['Descricao_Produto_gh31'] == 20, 11,
np.where(base_dados['Descricao_Produto_gh31'] == 21, 12,
np.where(base_dados['Descricao_Produto_gh31'] == 24, 13,
np.where(base_dados['Descricao_Produto_gh31'] == 25, 14,
np.where(base_dados['Descricao_Produto_gh31'] == 27, 15,
np.where(base_dados['Descricao_Produto_gh31'] == 28, 16,
np.where(base_dados['Descricao_Produto_gh31'] == 29, 17,
np.where(base_dados['Descricao_Produto_gh31'] == 30, 18,
np.where(base_dados['Descricao_Produto_gh31'] == 31, 19,
np.where(base_dados['Descricao_Produto_gh31'] == 32, 20,
np.where(base_dados['Descricao_Produto_gh31'] == 33, 21,
np.where(base_dados['Descricao_Produto_gh31'] == 34, 22,
0)))))))))))))))))))))))

base_dados['Descricao_Produto_gh33'] = np.where(base_dados['Descricao_Produto_gh32'] == 0, 0,
np.where(base_dados['Descricao_Produto_gh32'] == 1, 1,
np.where(base_dados['Descricao_Produto_gh32'] == 2, 2,
np.where(base_dados['Descricao_Produto_gh32'] == 3, 3,
np.where(base_dados['Descricao_Produto_gh32'] == 4, 4,
np.where(base_dados['Descricao_Produto_gh32'] == 5, 5,
np.where(base_dados['Descricao_Produto_gh32'] == 6, 6,
np.where(base_dados['Descricao_Produto_gh32'] == 7, 7,
np.where(base_dados['Descricao_Produto_gh32'] == 8, 8,
np.where(base_dados['Descricao_Produto_gh32'] == 9, 9,
np.where(base_dados['Descricao_Produto_gh32'] == 10, 10,
np.where(base_dados['Descricao_Produto_gh32'] == 11, 11,
np.where(base_dados['Descricao_Produto_gh32'] == 12, 12,
np.where(base_dados['Descricao_Produto_gh32'] == 13, 13,
np.where(base_dados['Descricao_Produto_gh32'] == 14, 14,
np.where(base_dados['Descricao_Produto_gh32'] == 15, 15,
np.where(base_dados['Descricao_Produto_gh32'] == 16, 16,
np.where(base_dados['Descricao_Produto_gh32'] == 17, 17,
np.where(base_dados['Descricao_Produto_gh32'] == 18, 18,
np.where(base_dados['Descricao_Produto_gh32'] == 19, 19,
np.where(base_dados['Descricao_Produto_gh32'] == 20, 20,
np.where(base_dados['Descricao_Produto_gh32'] == 21, 21,
np.where(base_dados['Descricao_Produto_gh32'] == 22, 22,
0)))))))))))))))))))))))

base_dados['Descricao_Produto_gh34'] = np.where(base_dados['Descricao_Produto_gh33'] == 0, 0,
np.where(base_dados['Descricao_Produto_gh33'] == 1, 22,
np.where(base_dados['Descricao_Produto_gh33'] == 2, 22,
np.where(base_dados['Descricao_Produto_gh33'] == 3, 3,
np.where(base_dados['Descricao_Produto_gh33'] == 4, 22,
np.where(base_dados['Descricao_Produto_gh33'] == 5, 22,
np.where(base_dados['Descricao_Produto_gh33'] == 6, 6,
np.where(base_dados['Descricao_Produto_gh33'] == 7, 22,
np.where(base_dados['Descricao_Produto_gh33'] == 8, 0,
np.where(base_dados['Descricao_Produto_gh33'] == 9, 6,
np.where(base_dados['Descricao_Produto_gh33'] == 10, 22,
np.where(base_dados['Descricao_Produto_gh33'] == 11, 6,
np.where(base_dados['Descricao_Produto_gh33'] == 12, 22,
np.where(base_dados['Descricao_Produto_gh33'] == 13, 0,
np.where(base_dados['Descricao_Produto_gh33'] == 14, 22,
np.where(base_dados['Descricao_Produto_gh33'] == 15, 15,
np.where(base_dados['Descricao_Produto_gh33'] == 16, 22,
np.where(base_dados['Descricao_Produto_gh33'] == 17, 17,
np.where(base_dados['Descricao_Produto_gh33'] == 18, 22,
np.where(base_dados['Descricao_Produto_gh33'] == 19, 22,
np.where(base_dados['Descricao_Produto_gh33'] == 20, 0,
np.where(base_dados['Descricao_Produto_gh33'] == 21, 22,
np.where(base_dados['Descricao_Produto_gh33'] == 22, 22,
0)))))))))))))))))))))))

base_dados['Descricao_Produto_gh35'] = np.where(base_dados['Descricao_Produto_gh34'] == 0, 0,
np.where(base_dados['Descricao_Produto_gh34'] == 3, 1,
np.where(base_dados['Descricao_Produto_gh34'] == 6, 2,
np.where(base_dados['Descricao_Produto_gh34'] == 15, 3,
np.where(base_dados['Descricao_Produto_gh34'] == 17, 4,
np.where(base_dados['Descricao_Produto_gh34'] == 22, 5,
0))))))

base_dados['Descricao_Produto_gh36'] = np.where(base_dados['Descricao_Produto_gh35'] == 0, 5,
np.where(base_dados['Descricao_Produto_gh35'] == 1, 2,
np.where(base_dados['Descricao_Produto_gh35'] == 2, 1,
np.where(base_dados['Descricao_Produto_gh35'] == 3, 2,
np.where(base_dados['Descricao_Produto_gh35'] == 4, 2,
np.where(base_dados['Descricao_Produto_gh35'] == 5, 0,
0))))))

base_dados['Descricao_Produto_gh37'] = np.where(base_dados['Descricao_Produto_gh36'] == 0, 1,
np.where(base_dados['Descricao_Produto_gh36'] == 1, 1,
np.where(base_dados['Descricao_Produto_gh36'] == 2, 2,
np.where(base_dados['Descricao_Produto_gh36'] == 5, 3,
0))))

base_dados['Descricao_Produto_gh38'] = np.where(base_dados['Descricao_Produto_gh37'] == 1, 0,
np.where(base_dados['Descricao_Produto_gh37'] == 2, 1,
np.where(base_dados['Descricao_Produto_gh37'] == 3, 2,
0)))       
         
         
         
         
         
base_dados['Telefone1_gh30'] = np.where(base_dados['Telefone1'] == 'sem_tags', 0,
np.where(base_dados['Telefone1'] == 'skip_alto', 1,
np.where(base_dados['Telefone1'] == 'skip_baixo', 2,
np.where(base_dados['Telefone1'] == 'skip_hot', 3,
np.where(base_dados['Telefone1'] == 'skip_medio', 4,
np.where(base_dados['Telefone1'] == 'skip_nhot', 5,
0))))))

base_dados['Telefone1_gh31'] = np.where(base_dados['Telefone1_gh30'] == 0, 0,
np.where(base_dados['Telefone1_gh30'] == 1, 1,
np.where(base_dados['Telefone1_gh30'] == 2, 2,
np.where(base_dados['Telefone1_gh30'] == 3, 3,
np.where(base_dados['Telefone1_gh30'] == 4, 4,
np.where(base_dados['Telefone1_gh30'] == 5, 5,
0))))))

base_dados['Telefone1_gh32'] = np.where(base_dados['Telefone1_gh31'] == 0, 0,
np.where(base_dados['Telefone1_gh31'] == 1, 1,
np.where(base_dados['Telefone1_gh31'] == 2, 2,
np.where(base_dados['Telefone1_gh31'] == 3, 3,
np.where(base_dados['Telefone1_gh31'] == 4, 4,
np.where(base_dados['Telefone1_gh31'] == 5, 5,
0))))))

base_dados['Telefone1_gh33'] = np.where(base_dados['Telefone1_gh32'] == 0, 0,
np.where(base_dados['Telefone1_gh32'] == 1, 1,
np.where(base_dados['Telefone1_gh32'] == 2, 2,
np.where(base_dados['Telefone1_gh32'] == 3, 3,
np.where(base_dados['Telefone1_gh32'] == 4, 4,
np.where(base_dados['Telefone1_gh32'] == 5, 5,
0))))))

base_dados['Telefone1_gh34'] = np.where(base_dados['Telefone1_gh33'] == 0, 0,
np.where(base_dados['Telefone1_gh33'] == 1, 1,
np.where(base_dados['Telefone1_gh33'] == 2, 2,
np.where(base_dados['Telefone1_gh33'] == 3, 1,
np.where(base_dados['Telefone1_gh33'] == 4, 4,
np.where(base_dados['Telefone1_gh33'] == 5, 1,
0))))))

base_dados['Telefone1_gh35'] = np.where(base_dados['Telefone1_gh34'] == 0, 0,
np.where(base_dados['Telefone1_gh34'] == 1, 1,
np.where(base_dados['Telefone1_gh34'] == 2, 2,
np.where(base_dados['Telefone1_gh34'] == 4, 3,
0))))

base_dados['Telefone1_gh36'] = np.where(base_dados['Telefone1_gh35'] == 0, 2,
np.where(base_dados['Telefone1_gh35'] == 1, 3,
np.where(base_dados['Telefone1_gh35'] == 2, 1,
np.where(base_dados['Telefone1_gh35'] == 3, 0,
0))))

base_dados['Telefone1_gh37'] = np.where(base_dados['Telefone1_gh36'] == 0, 0,
np.where(base_dados['Telefone1_gh36'] == 1, 1,
np.where(base_dados['Telefone1_gh36'] == 2, 2,
np.where(base_dados['Telefone1_gh36'] == 3, 3,
0))))

base_dados['Telefone1_gh38'] = np.where(base_dados['Telefone1_gh37'] == 0, 0,
np.where(base_dados['Telefone1_gh37'] == 1, 1,
np.where(base_dados['Telefone1_gh37'] == 2, 2,
np.where(base_dados['Telefone1_gh37'] == 3, 3,
0))))
         
         
         
         
         
base_dados['Telefone10_gh30'] = np.where(base_dados['Telefone10'] == 'sem_tags', 0,
np.where(base_dados['Telefone10'] == 'skip_alto', 1,
np.where(base_dados['Telefone10'] == 'skip_baixo', 2,
np.where(base_dados['Telefone10'] == 'skip_medio', 3,
np.where(base_dados['Telefone10'] == 'skip_nhot', 4,
0)))))

base_dados['Telefone10_gh31'] = np.where(base_dados['Telefone10_gh30'] == 0, 0,
np.where(base_dados['Telefone10_gh30'] == 1, 1,
np.where(base_dados['Telefone10_gh30'] == 2, 1,
np.where(base_dados['Telefone10_gh30'] == 3, 3,
np.where(base_dados['Telefone10_gh30'] == 4, 4,
0)))))

base_dados['Telefone10_gh32'] = np.where(base_dados['Telefone10_gh31'] == 0, 0,
np.where(base_dados['Telefone10_gh31'] == 1, 1,
np.where(base_dados['Telefone10_gh31'] == 3, 2,
np.where(base_dados['Telefone10_gh31'] == 4, 3,
0))))

base_dados['Telefone10_gh33'] = np.where(base_dados['Telefone10_gh32'] == 0, 0,
np.where(base_dados['Telefone10_gh32'] == 1, 1,
np.where(base_dados['Telefone10_gh32'] == 2, 2,
np.where(base_dados['Telefone10_gh32'] == 3, 3,
0))))

base_dados['Telefone10_gh34'] = np.where(base_dados['Telefone10_gh33'] == 0, 0,
np.where(base_dados['Telefone10_gh33'] == 1, 2,
np.where(base_dados['Telefone10_gh33'] == 2, 2,
np.where(base_dados['Telefone10_gh33'] == 3, 0,
0))))

base_dados['Telefone10_gh35'] = np.where(base_dados['Telefone10_gh34'] == 0, 0,
np.where(base_dados['Telefone10_gh34'] == 2, 1,
0))

base_dados['Telefone10_gh36'] = np.where(base_dados['Telefone10_gh35'] == 0, 1,
np.where(base_dados['Telefone10_gh35'] == 1, 0,
0))

base_dados['Telefone10_gh37'] = np.where(base_dados['Telefone10_gh36'] == 0, 0,
np.where(base_dados['Telefone10_gh36'] == 1, 1,
0))

base_dados['Telefone10_gh38'] = np.where(base_dados['Telefone10_gh37'] == 0, 0,
np.where(base_dados['Telefone10_gh37'] == 1, 1,
0))
                                         
                                         
                                         
                                         
base_dados['Email1_gh30'] = np.where(base_dados['Email1'] == 'sem_tags', 0,
np.where(base_dados['Email1'] == 'skip_alto', 1,
np.where(base_dados['Email1'] == 'skip_baixo', 2,
np.where(base_dados['Email1'] == 'skip_medio', 3,
np.where(base_dados['Email1'] == 'skip_nhot', 4,
0)))))

base_dados['Email1_gh31'] = np.where(base_dados['Email1_gh30'] == 0, 0,
np.where(base_dados['Email1_gh30'] == 1, 1,
np.where(base_dados['Email1_gh30'] == 2, 2,
np.where(base_dados['Email1_gh30'] == 3, 3,
np.where(base_dados['Email1_gh30'] == 4, 4,
0)))))

base_dados['Email1_gh32'] = np.where(base_dados['Email1_gh31'] == 0, 0,
np.where(base_dados['Email1_gh31'] == 1, 1,
np.where(base_dados['Email1_gh31'] == 2, 2,
np.where(base_dados['Email1_gh31'] == 3, 3,
np.where(base_dados['Email1_gh31'] == 4, 4,
0)))))

base_dados['Email1_gh33'] = np.where(base_dados['Email1_gh32'] == 0, 0,
np.where(base_dados['Email1_gh32'] == 1, 1,
np.where(base_dados['Email1_gh32'] == 2, 2,
np.where(base_dados['Email1_gh32'] == 3, 3,
np.where(base_dados['Email1_gh32'] == 4, 4,
0)))))

base_dados['Email1_gh34'] = np.where(base_dados['Email1_gh33'] == 0, 0,
np.where(base_dados['Email1_gh33'] == 1, 1,
np.where(base_dados['Email1_gh33'] == 2, 0,
np.where(base_dados['Email1_gh33'] == 3, 3,
np.where(base_dados['Email1_gh33'] == 4, 5,
0)))))

base_dados['Email1_gh35'] = np.where(base_dados['Email1_gh34'] == 0, 0,
np.where(base_dados['Email1_gh34'] == 1, 1,
np.where(base_dados['Email1_gh34'] == 3, 2,
np.where(base_dados['Email1_gh34'] == 5, 3,
0))))

base_dados['Email1_gh36'] = np.where(base_dados['Email1_gh35'] == 0, 1,
np.where(base_dados['Email1_gh35'] == 1, 3,
np.where(base_dados['Email1_gh35'] == 2, 2,
np.where(base_dados['Email1_gh35'] == 3, 0,
0))))

base_dados['Email1_gh37'] = np.where(base_dados['Email1_gh36'] == 0, 1,
np.where(base_dados['Email1_gh36'] == 1, 1,
np.where(base_dados['Email1_gh36'] == 2, 2,
np.where(base_dados['Email1_gh36'] == 3, 2,
0))))

base_dados['Email1_gh38'] = np.where(base_dados['Email1_gh37'] == 1, 0,
np.where(base_dados['Email1_gh37'] == 2, 1,
0))
                                     
                                       
                                       
                                       
                                       
                                       
base_dados['Divida_Outro_Credor_gh30'] = np.where(base_dados['Divida_Outro_Credor'] == 0, 0,
np.where(base_dados['Divida_Outro_Credor'] == 1, 1,
0))
base_dados['Divida_Outro_Credor_gh31'] = np.where(base_dados['Divida_Outro_Credor_gh30'] == 0, 0,
np.where(base_dados['Divida_Outro_Credor_gh30'] == 1, 1,
0))
base_dados['Divida_Outro_Credor_gh32'] = np.where(base_dados['Divida_Outro_Credor_gh31'] == 0, 0,
np.where(base_dados['Divida_Outro_Credor_gh31'] == 1, 1,
0))
base_dados['Divida_Outro_Credor_gh33'] = np.where(base_dados['Divida_Outro_Credor_gh32'] == 0, 0,
np.where(base_dados['Divida_Outro_Credor_gh32'] == 1, 1,
0))
base_dados['Divida_Outro_Credor_gh34'] = np.where(base_dados['Divida_Outro_Credor_gh33'] == 0, 0,
np.where(base_dados['Divida_Outro_Credor_gh33'] == 1, 1,
0))
base_dados['Divida_Outro_Credor_gh35'] = np.where(base_dados['Divida_Outro_Credor_gh34'] == 0, 0,
np.where(base_dados['Divida_Outro_Credor_gh34'] == 1, 1,
0))
base_dados['Divida_Outro_Credor_gh36'] = np.where(base_dados['Divida_Outro_Credor_gh35'] == 0, 1,
np.where(base_dados['Divida_Outro_Credor_gh35'] == 1, 0,
0))
base_dados['Divida_Outro_Credor_gh37'] = np.where(base_dados['Divida_Outro_Credor_gh36'] == 0, 0,
np.where(base_dados['Divida_Outro_Credor_gh36'] == 1, 1,
0))
base_dados['Divida_Outro_Credor_gh38'] = np.where(base_dados['Divida_Outro_Credor_gh37'] == 0, 0,
np.where(base_dados['Divida_Outro_Credor_gh37'] == 1, 1,
0))
                                     
                                     
                                     
base_dados['PRODUCT_1_gh30'] = np.where(base_dados['PRODUCT_1'] == -3.0, 0,
np.where(base_dados['PRODUCT_1'] == 173, 1,
np.where(base_dados['PRODUCT_1'] == 261, 2,
np.where(base_dados['PRODUCT_1'] == 809, 3,
np.where(base_dados['PRODUCT_1'] == 1323, 4,
np.where(base_dados['PRODUCT_1'] == 1357, 5,
np.where(base_dados['PRODUCT_1'] == 1358, 6,
np.where(base_dados['PRODUCT_1'] == 1359, 7,
np.where(base_dados['PRODUCT_1'] == 1360, 8,
np.where(base_dados['PRODUCT_1'] == 2012, 9,
np.where(base_dados['PRODUCT_1'] == 2536, 10,
np.where(base_dados['PRODUCT_1'] == 5120, 11,
np.where(base_dados['PRODUCT_1'] == 8018, 12,
np.where(base_dados['PRODUCT_1'] == 8179, 13,
np.where(base_dados['PRODUCT_1'] == 8180, 14,
np.where(base_dados['PRODUCT_1'] == 9999, 15,
0))))))))))))))))

base_dados['PRODUCT_1_gh31'] = np.where(base_dados['PRODUCT_1_gh30'] == 0, 0,
np.where(base_dados['PRODUCT_1_gh30'] == 1, 1,
np.where(base_dados['PRODUCT_1_gh30'] == 2, 2,
np.where(base_dados['PRODUCT_1_gh30'] == 3, 3,
np.where(base_dados['PRODUCT_1_gh30'] == 4, 3,
np.where(base_dados['PRODUCT_1_gh30'] == 5, 5,
np.where(base_dados['PRODUCT_1_gh30'] == 6, 6,
np.where(base_dados['PRODUCT_1_gh30'] == 7, 7,
np.where(base_dados['PRODUCT_1_gh30'] == 8, 8,
np.where(base_dados['PRODUCT_1_gh30'] == 9, 9,
np.where(base_dados['PRODUCT_1_gh30'] == 10, 9,
np.where(base_dados['PRODUCT_1_gh30'] == 11, 9,
np.where(base_dados['PRODUCT_1_gh30'] == 12, 9,
np.where(base_dados['PRODUCT_1_gh30'] == 13, 9,
np.where(base_dados['PRODUCT_1_gh30'] == 14, 9,
np.where(base_dados['PRODUCT_1_gh30'] == 15, 15,
0))))))))))))))))

base_dados['PRODUCT_1_gh32'] = np.where(base_dados['PRODUCT_1_gh31'] == 0, 0,
np.where(base_dados['PRODUCT_1_gh31'] == 1, 1,
np.where(base_dados['PRODUCT_1_gh31'] == 2, 2,
np.where(base_dados['PRODUCT_1_gh31'] == 3, 3,
np.where(base_dados['PRODUCT_1_gh31'] == 5, 4,
np.where(base_dados['PRODUCT_1_gh31'] == 6, 5,
np.where(base_dados['PRODUCT_1_gh31'] == 7, 6,
np.where(base_dados['PRODUCT_1_gh31'] == 8, 7,
np.where(base_dados['PRODUCT_1_gh31'] == 9, 8,
np.where(base_dados['PRODUCT_1_gh31'] == 15, 9,
0))))))))))

base_dados['PRODUCT_1_gh33'] = np.where(base_dados['PRODUCT_1_gh32'] == 0, 0,
np.where(base_dados['PRODUCT_1_gh32'] == 1, 1,
np.where(base_dados['PRODUCT_1_gh32'] == 2, 2,
np.where(base_dados['PRODUCT_1_gh32'] == 3, 3,
np.where(base_dados['PRODUCT_1_gh32'] == 4, 4,
np.where(base_dados['PRODUCT_1_gh32'] == 5, 5,
np.where(base_dados['PRODUCT_1_gh32'] == 6, 6,
np.where(base_dados['PRODUCT_1_gh32'] == 7, 7,
np.where(base_dados['PRODUCT_1_gh32'] == 8, 8,
np.where(base_dados['PRODUCT_1_gh32'] == 9, 9,
0))))))))))

base_dados['PRODUCT_1_gh34'] = np.where(base_dados['PRODUCT_1_gh33'] == 0, 0,
np.where(base_dados['PRODUCT_1_gh33'] == 1, 6,
np.where(base_dados['PRODUCT_1_gh33'] == 2, 6,
np.where(base_dados['PRODUCT_1_gh33'] == 3, 3,
np.where(base_dados['PRODUCT_1_gh33'] == 4, 0,
np.where(base_dados['PRODUCT_1_gh33'] == 5, 3,
np.where(base_dados['PRODUCT_1_gh33'] == 6, 6,
np.where(base_dados['PRODUCT_1_gh33'] == 7, 3,
np.where(base_dados['PRODUCT_1_gh33'] == 8, 3,
np.where(base_dados['PRODUCT_1_gh33'] == 9, 9,
0))))))))))

base_dados['PRODUCT_1_gh35'] = np.where(base_dados['PRODUCT_1_gh34'] == 0, 0,
np.where(base_dados['PRODUCT_1_gh34'] == 3, 1,
np.where(base_dados['PRODUCT_1_gh34'] == 6, 2,
np.where(base_dados['PRODUCT_1_gh34'] == 9, 3,
0))))

base_dados['PRODUCT_1_gh36'] = np.where(base_dados['PRODUCT_1_gh35'] == 0, 2,
np.where(base_dados['PRODUCT_1_gh35'] == 1, 0,
np.where(base_dados['PRODUCT_1_gh35'] == 2, 2,
np.where(base_dados['PRODUCT_1_gh35'] == 3, 1,
0))))

base_dados['PRODUCT_1_gh37'] = np.where(base_dados['PRODUCT_1_gh36'] == 0, 1,
np.where(base_dados['PRODUCT_1_gh36'] == 1, 1,
np.where(base_dados['PRODUCT_1_gh36'] == 2, 2,
0)))

base_dados['PRODUCT_1_gh38'] = np.where(base_dados['PRODUCT_1_gh37'] == 1, 0,
np.where(base_dados['PRODUCT_1_gh37'] == 2, 1,
0))
                                        
                                                        
                                                        
                                                        
                                                        
                                                        
base_dados['QTD_AUTH_gh30'] = np.where(base_dados['QTD_AUTH'] == 0, 0,
np.where(base_dados['QTD_AUTH'] == 1, 1,
np.where(base_dados['QTD_AUTH'] == 2, 2,
np.where(base_dados['QTD_AUTH'] == 3, 3,
np.where(base_dados['QTD_AUTH'] == 4, 4,
np.where(base_dados['QTD_AUTH'] == 5, 5,
np.where(base_dados['QTD_AUTH'] == 6, 6,
np.where(base_dados['QTD_AUTH'] == 7, 7,
np.where(base_dados['QTD_AUTH'] == 8, 8,
np.where(base_dados['QTD_AUTH'] == 9, 9,
np.where(base_dados['QTD_AUTH'] == 10, 10,
np.where(base_dados['QTD_AUTH'] == 11, 11,
np.where(base_dados['QTD_AUTH'] == 12, 12,
np.where(base_dados['QTD_AUTH'] == 13, 13,
np.where(base_dados['QTD_AUTH'] == 14, 14,
np.where(base_dados['QTD_AUTH'] == 15, 15,
np.where(base_dados['QTD_AUTH'] == 16, 16,
np.where(base_dados['QTD_AUTH'] == 17, 17,
np.where(base_dados['QTD_AUTH'] == 18, 18,
np.where(base_dados['QTD_AUTH'] == 19, 19,
np.where(base_dados['QTD_AUTH'] == 20, 20,
np.where(base_dados['QTD_AUTH'] == 22, 21,
np.where(base_dados['QTD_AUTH'] == 24, 22,
np.where(base_dados['QTD_AUTH'] == 30, 23,
np.where(base_dados['QTD_AUTH'] == 32, 24,
np.where(base_dados['QTD_AUTH'] == 38, 25,
np.where(base_dados['QTD_AUTH'] == 51, 26,
21)))))))))))))))))))))))))))

base_dados['QTD_AUTH_gh31'] = np.where(base_dados['QTD_AUTH_gh30'] == 0, 0,
np.where(base_dados['QTD_AUTH_gh30'] == 1, 1,
np.where(base_dados['QTD_AUTH_gh30'] == 2, 1,
np.where(base_dados['QTD_AUTH_gh30'] == 3, 1,
np.where(base_dados['QTD_AUTH_gh30'] == 4, 1,
np.where(base_dados['QTD_AUTH_gh30'] == 5, 1,
np.where(base_dados['QTD_AUTH_gh30'] == 6, 1,
np.where(base_dados['QTD_AUTH_gh30'] == 7, 1,
np.where(base_dados['QTD_AUTH_gh30'] == 8, 8,
np.where(base_dados['QTD_AUTH_gh30'] == 9, 9,
np.where(base_dados['QTD_AUTH_gh30'] == 10, 9,
np.where(base_dados['QTD_AUTH_gh30'] == 11, 11,
np.where(base_dados['QTD_AUTH_gh30'] == 12, 12,
np.where(base_dados['QTD_AUTH_gh30'] == 13, 13,
np.where(base_dados['QTD_AUTH_gh30'] == 14, 14,
np.where(base_dados['QTD_AUTH_gh30'] == 15, 15,
np.where(base_dados['QTD_AUTH_gh30'] == 16, 15,
np.where(base_dados['QTD_AUTH_gh30'] == 17, 15,
np.where(base_dados['QTD_AUTH_gh30'] == 18, 15,
np.where(base_dados['QTD_AUTH_gh30'] == 19, 15,
np.where(base_dados['QTD_AUTH_gh30'] == 20, 15,
np.where(base_dados['QTD_AUTH_gh30'] == 21, 15,
np.where(base_dados['QTD_AUTH_gh30'] == 22, 15,
np.where(base_dados['QTD_AUTH_gh30'] == 23, 15,
np.where(base_dados['QTD_AUTH_gh30'] == 24, 15,
np.where(base_dados['QTD_AUTH_gh30'] == 25, 15,
np.where(base_dados['QTD_AUTH_gh30'] == 26, 26,
0)))))))))))))))))))))))))))

base_dados['QTD_AUTH_gh32'] = np.where(base_dados['QTD_AUTH_gh31'] == 0, 0,
np.where(base_dados['QTD_AUTH_gh31'] == 1, 1,
np.where(base_dados['QTD_AUTH_gh31'] == 8, 2,
np.where(base_dados['QTD_AUTH_gh31'] == 9, 3,
np.where(base_dados['QTD_AUTH_gh31'] == 11, 4,
np.where(base_dados['QTD_AUTH_gh31'] == 12, 5,
np.where(base_dados['QTD_AUTH_gh31'] == 13, 6,
np.where(base_dados['QTD_AUTH_gh31'] == 14, 7,
np.where(base_dados['QTD_AUTH_gh31'] == 15, 8,
np.where(base_dados['QTD_AUTH_gh31'] == 26, 9,
0))))))))))

base_dados['QTD_AUTH_gh33'] = np.where(base_dados['QTD_AUTH_gh32'] == 0, 0,
np.where(base_dados['QTD_AUTH_gh32'] == 1, 1,
np.where(base_dados['QTD_AUTH_gh32'] == 2, 2,
np.where(base_dados['QTD_AUTH_gh32'] == 3, 3,
np.where(base_dados['QTD_AUTH_gh32'] == 4, 4,
np.where(base_dados['QTD_AUTH_gh32'] == 5, 5,
np.where(base_dados['QTD_AUTH_gh32'] == 6, 6,
np.where(base_dados['QTD_AUTH_gh32'] == 7, 7,
np.where(base_dados['QTD_AUTH_gh32'] == 8, 8,
np.where(base_dados['QTD_AUTH_gh32'] == 9, 9,
0))))))))))

base_dados['QTD_AUTH_gh34'] = np.where(base_dados['QTD_AUTH_gh33'] == 0, 0,
np.where(base_dados['QTD_AUTH_gh33'] == 1, 1,
np.where(base_dados['QTD_AUTH_gh33'] == 2, 1,
np.where(base_dados['QTD_AUTH_gh33'] == 3, 0,
np.where(base_dados['QTD_AUTH_gh33'] == 4, 0,
np.where(base_dados['QTD_AUTH_gh33'] == 5, 0,
np.where(base_dados['QTD_AUTH_gh33'] == 6, 6,
np.where(base_dados['QTD_AUTH_gh33'] == 7, 0,
np.where(base_dados['QTD_AUTH_gh33'] == 8, 1,
np.where(base_dados['QTD_AUTH_gh33'] == 9, 6,
0))))))))))

base_dados['QTD_AUTH_gh35'] = np.where(base_dados['QTD_AUTH_gh34'] == 0, 0,
np.where(base_dados['QTD_AUTH_gh34'] == 1, 1,
np.where(base_dados['QTD_AUTH_gh34'] == 6, 2,
0)))
base_dados['QTD_AUTH_gh36'] = np.where(base_dados['QTD_AUTH_gh35'] == 0, 1,
np.where(base_dados['QTD_AUTH_gh35'] == 1, 2,
np.where(base_dados['QTD_AUTH_gh35'] == 2, 0,
0)))
base_dados['QTD_AUTH_gh37'] = np.where(base_dados['QTD_AUTH_gh36'] == 0, 1,
np.where(base_dados['QTD_AUTH_gh36'] == 1, 1,
np.where(base_dados['QTD_AUTH_gh36'] == 2, 2,
0)))
base_dados['QTD_AUTH_gh38'] = np.where(base_dados['QTD_AUTH_gh37'] == 1, 0,
np.where(base_dados['QTD_AUTH_gh37'] == 2, 1,
0))

                                       
                                     
base_dados['Divida_Mesmo_Credor_gh30'] = np.where(base_dados['Divida_Mesmo_Credor'] == 0, 0,
np.where(base_dados['Divida_Mesmo_Credor'] == 1, 1,
0))
base_dados['Divida_Mesmo_Credor_gh31'] = np.where(base_dados['Divida_Mesmo_Credor_gh30'] == 0, 0,
np.where(base_dados['Divida_Mesmo_Credor_gh30'] == 1, 1,
0))
base_dados['Divida_Mesmo_Credor_gh32'] = np.where(base_dados['Divida_Mesmo_Credor_gh31'] == 0, 0,
np.where(base_dados['Divida_Mesmo_Credor_gh31'] == 1, 1,
0))
base_dados['Divida_Mesmo_Credor_gh33'] = np.where(base_dados['Divida_Mesmo_Credor_gh32'] == 0, 0,
np.where(base_dados['Divida_Mesmo_Credor_gh32'] == 1, 1,
0))
base_dados['Divida_Mesmo_Credor_gh34'] = np.where(base_dados['Divida_Mesmo_Credor_gh33'] == 0, 0,
np.where(base_dados['Divida_Mesmo_Credor_gh33'] == 1, 1,
0))
base_dados['Divida_Mesmo_Credor_gh35'] = np.where(base_dados['Divida_Mesmo_Credor_gh34'] == 0, 0,
np.where(base_dados['Divida_Mesmo_Credor_gh34'] == 1, 1,
0))
base_dados['Divida_Mesmo_Credor_gh36'] = np.where(base_dados['Divida_Mesmo_Credor_gh35'] == 0, 0,
np.where(base_dados['Divida_Mesmo_Credor_gh35'] == 1, 1,
0))
base_dados['Divida_Mesmo_Credor_gh37'] = np.where(base_dados['Divida_Mesmo_Credor_gh36'] == 0, 0,
np.where(base_dados['Divida_Mesmo_Credor_gh36'] == 1, 1,
0))
base_dados['Divida_Mesmo_Credor_gh38'] = np.where(base_dados['Divida_Mesmo_Credor_gh37'] == 0, 0,
np.where(base_dados['Divida_Mesmo_Credor_gh37'] == 1, 1,
0))




base_dados['Desconto_Padrao_gh40'] = np.where(base_dados['Desconto_Padrao'] == 0, 1,
np.where(base_dados['Desconto_Padrao'] >= 90, 2, 0))

base_dados['Desconto_Padrao_gh41'] = np.where(base_dados['Desconto_Padrao_gh40'] == 0, 0,
np.where(base_dados['Desconto_Padrao_gh40'] == 1, 1,
np.where(base_dados['Desconto_Padrao_gh40'] == 2, 2,
0)))

base_dados['Desconto_Padrao_gh42'] = np.where(base_dados['Desconto_Padrao_gh41'] == -5, 0,
np.where(base_dados['Desconto_Padrao_gh41'] == 0, 1,
np.where(base_dados['Desconto_Padrao_gh41'] == 1, 2,
0)))

base_dados['Desconto_Padrao_gh43'] = np.where(base_dados['Desconto_Padrao_gh42'] == 0, 2,
np.where(base_dados['Desconto_Padrao_gh42'] == 1, 0,
np.where(base_dados['Desconto_Padrao_gh42'] == 2, 1,
0)))

base_dados['Desconto_Padrao_gh44'] = np.where(base_dados['Desconto_Padrao_gh43'] == 0, 0,
np.where(base_dados['Desconto_Padrao_gh43'] == 1, 1,
np.where(base_dados['Desconto_Padrao_gh43'] == 2, 2,
0)))
base_dados['Desconto_Padrao_gh45'] = np.where(base_dados['Desconto_Padrao_gh44'] == 0, 0,
np.where(base_dados['Desconto_Padrao_gh44'] == 1, 1,
np.where(base_dados['Desconto_Padrao_gh44'] == 2, 2,
0)))
base_dados['Desconto_Padrao_gh46'] = np.where(base_dados['Desconto_Padrao_gh45'] == 0, 0,
np.where(base_dados['Desconto_Padrao_gh45'] == 1, 1,
np.where(base_dados['Desconto_Padrao_gh45'] == 2, 2,
0)))
base_dados['Desconto_Padrao_gh47'] = np.where(base_dados['Desconto_Padrao_gh46'] == 0, 0,
np.where(base_dados['Desconto_Padrao_gh46'] == 1, 1,
np.where(base_dados['Desconto_Padrao_gh46'] == 2, 2,
0)))
base_dados['Desconto_Padrao_gh48'] = np.where(base_dados['Desconto_Padrao_gh47'] == 0, 0,
np.where(base_dados['Desconto_Padrao_gh47'] == 1, 1,
np.where(base_dados['Desconto_Padrao_gh47'] == 2, 2,
0)))
base_dados['Desconto_Padrao_gh49'] = np.where(base_dados['Desconto_Padrao_gh48'] == 0, 0,
np.where(base_dados['Desconto_Padrao_gh48'] == 1, 1,
np.where(base_dados['Desconto_Padrao_gh48'] == 2, 2,
0)))
base_dados['Desconto_Padrao_gh50'] = np.where(base_dados['Desconto_Padrao_gh49'] == 0, 0,
np.where(base_dados['Desconto_Padrao_gh49'] == 1, 1,
np.where(base_dados['Desconto_Padrao_gh49'] == 2, 2,
0)))
base_dados['Desconto_Padrao_gh51'] = np.where(base_dados['Desconto_Padrao_gh50'] == 0, 0,
np.where(base_dados['Desconto_Padrao_gh50'] == 1, 1,
np.where(base_dados['Desconto_Padrao_gh50'] == 2, 2,
0)))




base_dados['TIPO_PRODUTO_gh40'] = np.where(base_dados['TIPO_PRODUTO'] == 'AD', 0,
np.where(base_dados['TIPO_PRODUTO'] == 'CARTAO', 1,
np.where(base_dados['TIPO_PRODUTO'] == 'CCG', 2,
np.where(base_dados['TIPO_PRODUTO'] == 'CHEQUE ESPECIAL', 3,
np.where(base_dados['TIPO_PRODUTO'] == 'GIRO', 4,
np.where(base_dados['TIPO_PRODUTO'] == 'OUTROS', 5,
np.where(base_dados['TIPO_PRODUTO'] == 'REFIN', 6,
0)))))))

base_dados['TIPO_PRODUTO_gh41'] = np.where(base_dados['TIPO_PRODUTO_gh40'] == 0, -5,
np.where(base_dados['TIPO_PRODUTO_gh40'] == 1, 0,
np.where(base_dados['TIPO_PRODUTO_gh40'] == 2, -5,
np.where(base_dados['TIPO_PRODUTO_gh40'] == 3, 1,
np.where(base_dados['TIPO_PRODUTO_gh40'] == 4, -5,
np.where(base_dados['TIPO_PRODUTO_gh40'] == 5, -5,
np.where(base_dados['TIPO_PRODUTO_gh40'] == 6, -5,
0)))))))

base_dados['TIPO_PRODUTO_gh42'] = np.where(base_dados['TIPO_PRODUTO_gh41'] == -5, 0,
np.where(base_dados['TIPO_PRODUTO_gh41'] == 0, 1,
np.where(base_dados['TIPO_PRODUTO_gh41'] == 1, 2,
0)))

base_dados['TIPO_PRODUTO_gh43'] = np.where(base_dados['TIPO_PRODUTO_gh42'] == 0, 2,
np.where(base_dados['TIPO_PRODUTO_gh42'] == 1, 1,
np.where(base_dados['TIPO_PRODUTO_gh42'] == 2, 0,
0)))

base_dados['TIPO_PRODUTO_gh44'] = np.where(base_dados['TIPO_PRODUTO_gh43'] == 0, 0,
np.where(base_dados['TIPO_PRODUTO_gh43'] == 1, 1,
np.where(base_dados['TIPO_PRODUTO_gh43'] == 2, 2,
0)))

base_dados['TIPO_PRODUTO_gh45'] = np.where(base_dados['TIPO_PRODUTO_gh44'] == 0, 0,
np.where(base_dados['TIPO_PRODUTO_gh44'] == 1, 1,
np.where(base_dados['TIPO_PRODUTO_gh44'] == 2, 2,
0)))

base_dados['TIPO_PRODUTO_gh46'] = np.where(base_dados['TIPO_PRODUTO_gh45'] == 0, 0,
np.where(base_dados['TIPO_PRODUTO_gh45'] == 1, 1,
np.where(base_dados['TIPO_PRODUTO_gh45'] == 2, 2,
0)))

base_dados['TIPO_PRODUTO_gh47'] = np.where(base_dados['TIPO_PRODUTO_gh46'] == 0, 0,
np.where(base_dados['TIPO_PRODUTO_gh46'] == 1, 1,
np.where(base_dados['TIPO_PRODUTO_gh46'] == 2, 2,
0)))
base_dados['TIPO_PRODUTO_gh48'] = np.where(base_dados['TIPO_PRODUTO_gh47'] == 0, 0,
np.where(base_dados['TIPO_PRODUTO_gh47'] == 1, 1,
np.where(base_dados['TIPO_PRODUTO_gh47'] == 2, 2,
0)))
base_dados['TIPO_PRODUTO_gh49'] = np.where(base_dados['TIPO_PRODUTO_gh48'] == 0, 0,
np.where(base_dados['TIPO_PRODUTO_gh48'] == 1, 1,
np.where(base_dados['TIPO_PRODUTO_gh48'] == 2, 2,
0)))
base_dados['TIPO_PRODUTO_gh50'] = np.where(base_dados['TIPO_PRODUTO_gh49'] == 0, 0,
np.where(base_dados['TIPO_PRODUTO_gh49'] == 1, 1,
np.where(base_dados['TIPO_PRODUTO_gh49'] == 2, 2,
0)))
base_dados['TIPO_PRODUTO_gh51'] = np.where(base_dados['TIPO_PRODUTO_gh50'] == 0, 0,
np.where(base_dados['TIPO_PRODUTO_gh50'] == 1, 1,
np.where(base_dados['TIPO_PRODUTO_gh50'] == 2, 2,
0)))




base_dados['UF1_gh40'] = np.where(base_dados['UF1'] == '-3', 0,
np.where(base_dados['UF1'] == 'AC', 1,
np.where(base_dados['UF1'] == 'AL', 2,
np.where(base_dados['UF1'] == 'AM', 3,
np.where(base_dados['UF1'] == 'AP', 4,
np.where(base_dados['UF1'] == 'BA', 5,
np.where(base_dados['UF1'] == 'CE', 6,
np.where(base_dados['UF1'] == 'DF', 7,
np.where(base_dados['UF1'] == 'ES', 8,
np.where(base_dados['UF1'] == 'GO', 9,
np.where(base_dados['UF1'] == 'MA', 10,
np.where(base_dados['UF1'] == 'MG', 11,
np.where(base_dados['UF1'] == 'MS', 12,
np.where(base_dados['UF1'] == 'MT', 13,
np.where(base_dados['UF1'] == 'PA', 14,
np.where(base_dados['UF1'] == 'PB', 15,
np.where(base_dados['UF1'] == 'PE', 16,
np.where(base_dados['UF1'] == 'PI', 17,
np.where(base_dados['UF1'] == 'PR', 18,
np.where(base_dados['UF1'] == 'RJ', 19,
np.where(base_dados['UF1'] == 'RN', 20,
np.where(base_dados['UF1'] == 'RO', 21,
np.where(base_dados['UF1'] == 'RS', 22,
np.where(base_dados['UF1'] == 'SC', 23,
np.where(base_dados['UF1'] == 'SE', 24,
np.where(base_dados['UF1'] == 'SP', 25,
np.where(base_dados['UF1'] == 'TO', 26,
0)))))))))))))))))))))))))))

base_dados['UF1_gh41'] = np.where(base_dados['UF1_gh40'] == 0, 1,
np.where(base_dados['UF1_gh40'] == 1, -5,
np.where(base_dados['UF1_gh40'] == 2, -5,
np.where(base_dados['UF1_gh40'] == 3, -5,
np.where(base_dados['UF1_gh40'] == 4, -5,
np.where(base_dados['UF1_gh40'] == 5, -5,
np.where(base_dados['UF1_gh40'] == 6, -5,
np.where(base_dados['UF1_gh40'] == 7, -5,
np.where(base_dados['UF1_gh40'] == 8, -5,
np.where(base_dados['UF1_gh40'] == 9, -5,
np.where(base_dados['UF1_gh40'] == 10, -5,
np.where(base_dados['UF1_gh40'] == 11, -5,
np.where(base_dados['UF1_gh40'] == 12, -5,
np.where(base_dados['UF1_gh40'] == 13, -5,
np.where(base_dados['UF1_gh40'] == 14, -5,
np.where(base_dados['UF1_gh40'] == 15, -5,
np.where(base_dados['UF1_gh40'] == 16, -5,
np.where(base_dados['UF1_gh40'] == 17, -5,
np.where(base_dados['UF1_gh40'] == 18, -5,
np.where(base_dados['UF1_gh40'] == 19, 2,
np.where(base_dados['UF1_gh40'] == 20, -5,
np.where(base_dados['UF1_gh40'] == 21, -5,
np.where(base_dados['UF1_gh40'] == 22, -5,
np.where(base_dados['UF1_gh40'] == 23, -5,
np.where(base_dados['UF1_gh40'] == 24, -5,
np.where(base_dados['UF1_gh40'] == 25, 0,
np.where(base_dados['UF1_gh40'] == 26, -5,
0)))))))))))))))))))))))))))

base_dados['UF1_gh42'] = np.where(base_dados['UF1_gh41'] == -5, 0,
np.where(base_dados['UF1_gh41'] == 0, 1,
np.where(base_dados['UF1_gh41'] == 1, 2,
np.where(base_dados['UF1_gh41'] == 2, 3,
0))))
base_dados['UF1_gh43'] = np.where(base_dados['UF1_gh42'] == 0, 1,
np.where(base_dados['UF1_gh42'] == 1, 1,
np.where(base_dados['UF1_gh42'] == 2, 3,
np.where(base_dados['UF1_gh42'] == 3, 0,
0))))
base_dados['UF1_gh44'] = np.where(base_dados['UF1_gh43'] == 0, 0,
np.where(base_dados['UF1_gh43'] == 1, 1,
np.where(base_dados['UF1_gh43'] == 3, 2,
0)))
base_dados['UF1_gh45'] = np.where(base_dados['UF1_gh44'] == 0, 0,
np.where(base_dados['UF1_gh44'] == 1, 1,
np.where(base_dados['UF1_gh44'] == 2, 2,
0)))
base_dados['UF1_gh46'] = np.where(base_dados['UF1_gh45'] == 0, 0,
np.where(base_dados['UF1_gh45'] == 1, 1,
np.where(base_dados['UF1_gh45'] == 2, 2,
0)))
base_dados['UF1_gh47'] = np.where(base_dados['UF1_gh46'] == 0, 0,
np.where(base_dados['UF1_gh46'] == 1, 1,
np.where(base_dados['UF1_gh46'] == 2, 2,
0)))
base_dados['UF1_gh48'] = np.where(base_dados['UF1_gh47'] == 0, 0,
np.where(base_dados['UF1_gh47'] == 1, 1,
np.where(base_dados['UF1_gh47'] == 2, 2,
0)))
base_dados['UF1_gh49'] = np.where(base_dados['UF1_gh48'] == 0, 0,
np.where(base_dados['UF1_gh48'] == 1, 1,
np.where(base_dados['UF1_gh48'] == 2, 2,
0)))
base_dados['UF1_gh50'] = np.where(base_dados['UF1_gh49'] == 0, 0,
np.where(base_dados['UF1_gh49'] == 1, 1,
np.where(base_dados['UF1_gh49'] == 2, 2,
0)))
base_dados['UF1_gh51'] = np.where(base_dados['UF1_gh50'] == 0, 0,
np.where(base_dados['UF1_gh50'] == 1, 1,
np.where(base_dados['UF1_gh50'] == 2, 2,
0)))




# COMMAND ----------

# MAGIC %md
# MAGIC ## Gerando variáveis numéricas contínuas

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parte 1 de 2

# COMMAND ----------

base_dados['Codigo_Politica__p_5'] = np.where(base_dados['Codigo_Politica'] <= 413.0, 0.0,
np.where(np.bitwise_and(base_dados['Codigo_Politica'] > 413.0, base_dados['Codigo_Politica'] <= 434.0), 1.0,
np.where(np.bitwise_and(base_dados['Codigo_Politica'] > 434.0, base_dados['Codigo_Politica'] <= 495.0), 2.0,
np.where(np.bitwise_and(base_dados['Codigo_Politica'] > 495.0, base_dados['Codigo_Politica'] <= 522.0), 3.0,
np.where(base_dados['Codigo_Politica'] > 522.0, 4.0,
 0)))))

base_dados['Codigo_Politica__p_5_g_1_1'] = np.where(base_dados['Codigo_Politica__p_5'] == 0.0, 0,
np.where(base_dados['Codigo_Politica__p_5'] == 1.0, 3,
np.where(base_dados['Codigo_Politica__p_5'] == 2.0, 4,
np.where(base_dados['Codigo_Politica__p_5'] == 3.0, 1,
np.where(base_dados['Codigo_Politica__p_5'] == 4.0, 2,
 0)))))

base_dados['Codigo_Politica__p_5_g_1_2'] = np.where(base_dados['Codigo_Politica__p_5_g_1_1'] == 0, 4,
np.where(base_dados['Codigo_Politica__p_5_g_1_1'] == 1, 1,
np.where(base_dados['Codigo_Politica__p_5_g_1_1'] == 2, 0,
np.where(base_dados['Codigo_Politica__p_5_g_1_1'] == 3, 3,
np.where(base_dados['Codigo_Politica__p_5_g_1_1'] == 4, 2,
 0)))))

base_dados['Codigo_Politica__p_5_g_1'] = np.where(base_dados['Codigo_Politica__p_5_g_1_2'] == 0, 0,
np.where(base_dados['Codigo_Politica__p_5_g_1_2'] == 1, 1,
np.where(base_dados['Codigo_Politica__p_5_g_1_2'] == 2, 2,
np.where(base_dados['Codigo_Politica__p_5_g_1_2'] == 3, 3,
np.where(base_dados['Codigo_Politica__p_5_g_1_2'] == 4, 4,
 0)))))






base_dados['Codigo_Politica__C'] = np.cos(base_dados['Codigo_Politica'])
np.where(base_dados['Codigo_Politica__C'] == 0, -1, base_dados['Codigo_Politica__C'])
base_dados['Codigo_Politica__C'] = base_dados['Codigo_Politica__C'].fillna(-2)
base_dados['Codigo_Politica__C__pk_10'] = np.where(base_dados['Codigo_Politica__C'] <= -0.8532358291526332, 0.0,
np.where(np.bitwise_and(base_dados['Codigo_Politica__C'] > -0.8532358291526332, base_dados['Codigo_Politica__C'] <= -0.6735294441417006), 1.0,
np.where(np.bitwise_and(base_dados['Codigo_Politica__C'] > -0.6735294441417006, base_dados['Codigo_Politica__C'] <= -0.45599593257909704), 2.0,
np.where(np.bitwise_and(base_dados['Codigo_Politica__C'] > -0.45599593257909704, base_dados['Codigo_Politica__C'] <= -0.23244705899272397), 3.0,
np.where(np.bitwise_and(base_dados['Codigo_Politica__C'] > -0.23244705899272397, base_dados['Codigo_Politica__C'] <= -0.022156893225121342), 4.0,
np.where(np.bitwise_and(base_dados['Codigo_Politica__C'] > -0.022156893225121342, base_dados['Codigo_Politica__C'] <= 0.19784312260506454), 5.0,
np.where(np.bitwise_and(base_dados['Codigo_Politica__C'] > 0.19784312260506454, base_dados['Codigo_Politica__C'] <= 0.35910055492590737), 6.0,
np.where(np.bitwise_and(base_dados['Codigo_Politica__C'] > 0.35910055492590737, base_dados['Codigo_Politica__C'] <= 0.5984359185578068), 7.0,
np.where(np.bitwise_and(base_dados['Codigo_Politica__C'] > 0.5984359185578068, base_dados['Codigo_Politica__C'] <= 0.7821745479202525), 8.0,
np.where(base_dados['Codigo_Politica__C'] > 0.7821745479202525, 9.0,
 0))))))))))

base_dados['Codigo_Politica__C__pk_10_g_1_1'] = np.where(base_dados['Codigo_Politica__C__pk_10'] == 0.0, 2,
np.where(base_dados['Codigo_Politica__C__pk_10'] == 1.0, 2,
np.where(base_dados['Codigo_Politica__C__pk_10'] == 2.0, 2,
np.where(base_dados['Codigo_Politica__C__pk_10'] == 3.0, 1,
np.where(base_dados['Codigo_Politica__C__pk_10'] == 4.0, 0,
np.where(base_dados['Codigo_Politica__C__pk_10'] == 5.0, 1,
np.where(base_dados['Codigo_Politica__C__pk_10'] == 6.0, 2,
np.where(base_dados['Codigo_Politica__C__pk_10'] == 7.0, 2,
np.where(base_dados['Codigo_Politica__C__pk_10'] == 8.0, 2,
np.where(base_dados['Codigo_Politica__C__pk_10'] == 9.0, 1,
 0))))))))))

base_dados['Codigo_Politica__C__pk_10_g_1_2'] = np.where(base_dados['Codigo_Politica__C__pk_10_g_1_1'] == 0, 0,
np.where(base_dados['Codigo_Politica__C__pk_10_g_1_1'] == 1, 2,
np.where(base_dados['Codigo_Politica__C__pk_10_g_1_1'] == 2, 0,
 0)))

base_dados['Codigo_Politica__C__pk_10_g_1'] = np.where(base_dados['Codigo_Politica__C__pk_10_g_1_2'] == 0, 0,
np.where(base_dados['Codigo_Politica__C__pk_10_g_1_2'] == 2, 1,
 0))


                                                       
                                                       
                                                       
                                                       
                                                       
base_dados['Saldo_Devedor_Contrato__R'] = np.sqrt(base_dados['Saldo_Devedor_Contrato'])
np.where(base_dados['Saldo_Devedor_Contrato__R'] == 0, -1, base_dados['Saldo_Devedor_Contrato__R'])
base_dados['Saldo_Devedor_Contrato__R'] = base_dados['Saldo_Devedor_Contrato__R'].fillna(-2)
base_dados['Saldo_Devedor_Contrato__R__pu_20'] = np.where(base_dados['Saldo_Devedor_Contrato__R'] <= 29.17790259768512, 0.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R'] > 29.17790259768512, base_dados['Saldo_Devedor_Contrato__R'] <= 59.339784293507506), 1.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R'] > 59.339784293507506, base_dados['Saldo_Devedor_Contrato__R'] <= 89.51876898170573), 2.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R'] > 89.51876898170573, base_dados['Saldo_Devedor_Contrato__R'] <= 119.54890212795766), 3.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R'] > 119.54890212795766, base_dados['Saldo_Devedor_Contrato__R'] <= 149.96162842540755), 4.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R'] > 149.96162842540755, base_dados['Saldo_Devedor_Contrato__R'] <= 180.34808011176608), 5.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R'] > 180.34808011176608, base_dados['Saldo_Devedor_Contrato__R'] <= 209.6838334254694), 6.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R'] > 209.6838334254694, base_dados['Saldo_Devedor_Contrato__R'] <= 238.9060484793133), 7.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R'] > 238.9060484793133, base_dados['Saldo_Devedor_Contrato__R'] <= 269.2118125194361), 8.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R'] > 269.2118125194361, base_dados['Saldo_Devedor_Contrato__R'] <= 301.25087551739995), 9.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R'] > 301.25087551739995, base_dados['Saldo_Devedor_Contrato__R'] <= 328.68837825514913), 10.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R'] > 328.68837825514913, base_dados['Saldo_Devedor_Contrato__R'] <= 355.33315353341294), 11.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R'] > 355.33315353341294, base_dados['Saldo_Devedor_Contrato__R'] <= 372.71011523702975), 12.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R'] > 372.71011523702975, base_dados['Saldo_Devedor_Contrato__R'] <= 446.65640037953114), 14.0,
np.where(base_dados['Saldo_Devedor_Contrato__R'] > 446.65640037953114, 19.0,
 0)))))))))))))))

base_dados['Saldo_Devedor_Contrato__R__pu_20_g_1_1'] = np.where(base_dados['Saldo_Devedor_Contrato__R__pu_20'] == 0.0, 0,
np.where(base_dados['Saldo_Devedor_Contrato__R__pu_20'] == 1.0, 0,
np.where(base_dados['Saldo_Devedor_Contrato__R__pu_20'] == 2.0, 1,
np.where(base_dados['Saldo_Devedor_Contrato__R__pu_20'] == 3.0, 2,
np.where(base_dados['Saldo_Devedor_Contrato__R__pu_20'] == 4.0, 2,
np.where(base_dados['Saldo_Devedor_Contrato__R__pu_20'] == 5.0, 3,
np.where(base_dados['Saldo_Devedor_Contrato__R__pu_20'] == 6.0, 3,
np.where(base_dados['Saldo_Devedor_Contrato__R__pu_20'] == 7.0, 3,
np.where(base_dados['Saldo_Devedor_Contrato__R__pu_20'] == 8.0, 3,
np.where(base_dados['Saldo_Devedor_Contrato__R__pu_20'] == 9.0, 3,
np.where(base_dados['Saldo_Devedor_Contrato__R__pu_20'] == 10.0, 3,
np.where(base_dados['Saldo_Devedor_Contrato__R__pu_20'] == 11.0, 3,
np.where(base_dados['Saldo_Devedor_Contrato__R__pu_20'] == 12.0, 3,
np.where(base_dados['Saldo_Devedor_Contrato__R__pu_20'] == 14.0, 3,
np.where(base_dados['Saldo_Devedor_Contrato__R__pu_20'] == 19.0, 3,
 0)))))))))))))))

base_dados['Saldo_Devedor_Contrato__R__pu_20_g_1_2'] = np.where(base_dados['Saldo_Devedor_Contrato__R__pu_20_g_1_1'] == 0, 3,
np.where(base_dados['Saldo_Devedor_Contrato__R__pu_20_g_1_1'] == 1, 2,
np.where(base_dados['Saldo_Devedor_Contrato__R__pu_20_g_1_1'] == 2, 1,
np.where(base_dados['Saldo_Devedor_Contrato__R__pu_20_g_1_1'] == 3, 0,
 0))))

base_dados['Saldo_Devedor_Contrato__R__pu_20_g_1'] = np.where(base_dados['Saldo_Devedor_Contrato__R__pu_20_g_1_2'] == 0, 0,
np.where(base_dados['Saldo_Devedor_Contrato__R__pu_20_g_1_2'] == 1, 1,
np.where(base_dados['Saldo_Devedor_Contrato__R__pu_20_g_1_2'] == 2, 2,
np.where(base_dados['Saldo_Devedor_Contrato__R__pu_20_g_1_2'] == 3, 3,
 0))))

base_dados['Saldo_Devedor_Contrato__L'] = np.log(base_dados['Saldo_Devedor_Contrato'])
np.where(base_dados['Saldo_Devedor_Contrato__L'] == 0, -1, base_dados['Saldo_Devedor_Contrato__L'])
base_dados['Saldo_Devedor_Contrato__L'] = base_dados['Saldo_Devedor_Contrato__L'].fillna(-2)
base_dados['Saldo_Devedor_Contrato__L__p_7'] = np.where(base_dados['Saldo_Devedor_Contrato__L'] <= 4.930725910867953, 0.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__L'] > 4.930725910867953, base_dados['Saldo_Devedor_Contrato__L'] <= 6.496201673533415), 1.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__L'] > 6.496201673533415, base_dados['Saldo_Devedor_Contrato__L'] <= 7.576394579887531), 2.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__L'] > 7.576394579887531, base_dados['Saldo_Devedor_Contrato__L'] <= 8.191197087983973), 3.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__L'] > 8.191197087983973, base_dados['Saldo_Devedor_Contrato__L'] <= 8.89426979887939), 4.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__L'] > 8.89426979887939, base_dados['Saldo_Devedor_Contrato__L'] <= 9.816979321674527), 5.0,
np.where(base_dados['Saldo_Devedor_Contrato__L'] > 9.816979321674527, 6.0,
 0)))))))

base_dados['Saldo_Devedor_Contrato__L__p_7_g_1_1'] = np.where(base_dados['Saldo_Devedor_Contrato__L__p_7'] == 0.0, 0,
np.where(base_dados['Saldo_Devedor_Contrato__L__p_7'] == 1.0, 2,
np.where(base_dados['Saldo_Devedor_Contrato__L__p_7'] == 2.0, 0,
np.where(base_dados['Saldo_Devedor_Contrato__L__p_7'] == 3.0, 0,
np.where(base_dados['Saldo_Devedor_Contrato__L__p_7'] == 4.0, 1,
np.where(base_dados['Saldo_Devedor_Contrato__L__p_7'] == 5.0, 1,
np.where(base_dados['Saldo_Devedor_Contrato__L__p_7'] == 6.0, 3,
 0)))))))

base_dados['Saldo_Devedor_Contrato__L__p_7_g_1_2'] = np.where(base_dados['Saldo_Devedor_Contrato__L__p_7_g_1_1'] == 0, 2,
np.where(base_dados['Saldo_Devedor_Contrato__L__p_7_g_1_1'] == 1, 1,
np.where(base_dados['Saldo_Devedor_Contrato__L__p_7_g_1_1'] == 2, 3,
np.where(base_dados['Saldo_Devedor_Contrato__L__p_7_g_1_1'] == 3, 0,
 0))))

base_dados['Saldo_Devedor_Contrato__L__p_7_g_1'] = np.where(base_dados['Saldo_Devedor_Contrato__L__p_7_g_1_2'] == 0, 0,
np.where(base_dados['Saldo_Devedor_Contrato__L__p_7_g_1_2'] == 1, 1,
np.where(base_dados['Saldo_Devedor_Contrato__L__p_7_g_1_2'] == 2, 2,
np.where(base_dados['Saldo_Devedor_Contrato__L__p_7_g_1_2'] == 3, 3,
 0))))

         
         
         
         
    
         
base_dados['Cep1_2__p_6'] = np.where(base_dados['Cep1_2'] <= 7.0, 0.0,
np.where(np.bitwise_and(base_dados['Cep1_2'] > 7.0, base_dados['Cep1_2'] <= 13.0), 1.0,
np.where(np.bitwise_and(base_dados['Cep1_2'] > 13.0, base_dados['Cep1_2'] <= 24.0), 2.0,
np.where(np.bitwise_and(base_dados['Cep1_2'] > 24.0, base_dados['Cep1_2'] <= 44.0), 3.0,
np.where(np.bitwise_and(base_dados['Cep1_2'] > 44.0, base_dados['Cep1_2'] <= 73.0), 4.0,
np.where(base_dados['Cep1_2'] > 73.0, 5.0,
 0))))))

base_dados['Cep1_2__p_6_g_1_1'] = np.where(base_dados['Cep1_2__p_6'] == 0.0, 0,
np.where(base_dados['Cep1_2__p_6'] == 1.0, 0,
np.where(base_dados['Cep1_2__p_6'] == 2.0, 1,
np.where(base_dados['Cep1_2__p_6'] == 3.0, 2,
np.where(base_dados['Cep1_2__p_6'] == 4.0, 1,
np.where(base_dados['Cep1_2__p_6'] == 5.0, 0,
 0))))))

base_dados['Cep1_2__p_6_g_1_2'] = np.where(base_dados['Cep1_2__p_6_g_1_1'] == 0, 0,
np.where(base_dados['Cep1_2__p_6_g_1_1'] == 1, 2,
np.where(base_dados['Cep1_2__p_6_g_1_1'] == 2, 0,
 0)))

base_dados['Cep1_2__p_6_g_1'] = np.where(base_dados['Cep1_2__p_6_g_1_2'] == 0, 0,
np.where(base_dados['Cep1_2__p_6_g_1_2'] == 2, 1,
 0))






base_dados['Cep1_2__C'] = np.cos(base_dados['Cep1_2'])
np.where(base_dados['Cep1_2__C'] == 0, -1, base_dados['Cep1_2__C'])
base_dados['Cep1_2__C'] = base_dados['Cep1_2__C'].fillna(-2)
base_dados['Cep1_2__C__pu_7'] = np.where(base_dados['Cep1_2__C'] <= -2.0, 0.0,
np.where(np.bitwise_and(base_dados['Cep1_2__C'] > -2.0, base_dados['Cep1_2__C'] <= -0.7361927182273159), 2.0,
np.where(np.bitwise_and(base_dados['Cep1_2__C'] > -0.7361927182273159, base_dados['Cep1_2__C'] <= -0.2921388087338362), 3.0,
np.where(np.bitwise_and(base_dados['Cep1_2__C'] > -0.2921388087338362, base_dados['Cep1_2__C'] <= 0.1367372182078336), 4.0,
np.where(np.bitwise_and(base_dados['Cep1_2__C'] > 0.1367372182078336, base_dados['Cep1_2__C'] <= 0.569750334265312), 5.0,
np.where(base_dados['Cep1_2__C'] > 0.569750334265312, 6.0,
 0))))))
base_dados['Cep1_2__C__pu_7_g_1_1'] = np.where(base_dados['Cep1_2__C__pu_7'] == 0.0, 2,
np.where(base_dados['Cep1_2__C__pu_7'] == 2.0, 0,
np.where(base_dados['Cep1_2__C__pu_7'] == 3.0, 0,
np.where(base_dados['Cep1_2__C__pu_7'] == 4.0, 0,
np.where(base_dados['Cep1_2__C__pu_7'] == 5.0, 1,
np.where(base_dados['Cep1_2__C__pu_7'] == 6.0, 2,
 0))))))
base_dados['Cep1_2__C__pu_7_g_1_2'] = np.where(base_dados['Cep1_2__C__pu_7_g_1_1'] == 0, 1,
np.where(base_dados['Cep1_2__C__pu_7_g_1_1'] == 1, 0,
np.where(base_dados['Cep1_2__C__pu_7_g_1_1'] == 2, 1,
 0)))
base_dados['Cep1_2__C__pu_7_g_1'] = np.where(base_dados['Cep1_2__C__pu_7_g_1_2'] == 0, 0,
np.where(base_dados['Cep1_2__C__pu_7_g_1_2'] == 1, 1,
 0))


# COMMAND ----------

# MAGIC %md
# MAGIC ### Parte 2 de 2

# COMMAND ----------

base_dados['Codigo_Politica__p_5_g_1_c1_5_1'] = np.where(np.bitwise_and(base_dados['Codigo_Politica__p_5_g_1'] == 0, base_dados['Codigo_Politica__C__pk_10_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['Codigo_Politica__p_5_g_1'] == 0, base_dados['Codigo_Politica__C__pk_10_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['Codigo_Politica__p_5_g_1'] == 1, base_dados['Codigo_Politica__C__pk_10_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['Codigo_Politica__p_5_g_1'] == 1, base_dados['Codigo_Politica__C__pk_10_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['Codigo_Politica__p_5_g_1'] == 2, base_dados['Codigo_Politica__C__pk_10_g_1'] == 0), 3,
np.where(np.bitwise_and(base_dados['Codigo_Politica__p_5_g_1'] == 2, base_dados['Codigo_Politica__C__pk_10_g_1'] == 1), 4,
np.where(np.bitwise_and(base_dados['Codigo_Politica__p_5_g_1'] == 3, base_dados['Codigo_Politica__C__pk_10_g_1'] == 0), 4,
np.where(np.bitwise_and(base_dados['Codigo_Politica__p_5_g_1'] == 3, base_dados['Codigo_Politica__C__pk_10_g_1'] == 1), 4,
np.where(np.bitwise_and(base_dados['Codigo_Politica__p_5_g_1'] == 4, base_dados['Codigo_Politica__C__pk_10_g_1'] == 0), 5,
np.where(np.bitwise_and(base_dados['Codigo_Politica__p_5_g_1'] == 4, base_dados['Codigo_Politica__C__pk_10_g_1'] == 1), 5,
 0))))))))))
base_dados['Codigo_Politica__p_5_g_1_c1_5_2'] = np.where(base_dados['Codigo_Politica__p_5_g_1_c1_5_1'] == 0, 0,
np.where(base_dados['Codigo_Politica__p_5_g_1_c1_5_1'] == 1, 1,
np.where(base_dados['Codigo_Politica__p_5_g_1_c1_5_1'] == 2, 3,
np.where(base_dados['Codigo_Politica__p_5_g_1_c1_5_1'] == 3, 2,
np.where(base_dados['Codigo_Politica__p_5_g_1_c1_5_1'] == 4, 4,
np.where(base_dados['Codigo_Politica__p_5_g_1_c1_5_1'] == 5, 5,
0))))))
base_dados['Codigo_Politica__p_5_g_1_c1_5'] = np.where(base_dados['Codigo_Politica__p_5_g_1_c1_5_2'] == 0, 0,
np.where(base_dados['Codigo_Politica__p_5_g_1_c1_5_2'] == 1, 1,
np.where(base_dados['Codigo_Politica__p_5_g_1_c1_5_2'] == 2, 2,
np.where(base_dados['Codigo_Politica__p_5_g_1_c1_5_2'] == 3, 3,
np.where(base_dados['Codigo_Politica__p_5_g_1_c1_5_2'] == 4, 4,
np.where(base_dados['Codigo_Politica__p_5_g_1_c1_5_2'] == 5, 5,
 0))))))

         
         
          
base_dados['Saldo_Devedor_Contrato__L__p_7_g_1_c1_48_1'] = np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R__pu_20_g_1'] == 0, base_dados['Saldo_Devedor_Contrato__L__p_7_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R__pu_20_g_1'] == 0, base_dados['Saldo_Devedor_Contrato__L__p_7_g_1'] == 1), 0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R__pu_20_g_1'] == 0, base_dados['Saldo_Devedor_Contrato__L__p_7_g_1'] == 2), 0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R__pu_20_g_1'] == 0, base_dados['Saldo_Devedor_Contrato__L__p_7_g_1'] == 3), 0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R__pu_20_g_1'] == 1, base_dados['Saldo_Devedor_Contrato__L__p_7_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R__pu_20_g_1'] == 1, base_dados['Saldo_Devedor_Contrato__L__p_7_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R__pu_20_g_1'] == 1, base_dados['Saldo_Devedor_Contrato__L__p_7_g_1'] == 2), 1,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R__pu_20_g_1'] == 1, base_dados['Saldo_Devedor_Contrato__L__p_7_g_1'] == 3), 1,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R__pu_20_g_1'] == 2, base_dados['Saldo_Devedor_Contrato__L__p_7_g_1'] == 0), 2,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R__pu_20_g_1'] == 2, base_dados['Saldo_Devedor_Contrato__L__p_7_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R__pu_20_g_1'] == 2, base_dados['Saldo_Devedor_Contrato__L__p_7_g_1'] == 2), 2,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R__pu_20_g_1'] == 2, base_dados['Saldo_Devedor_Contrato__L__p_7_g_1'] == 3), 2,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R__pu_20_g_1'] == 3, base_dados['Saldo_Devedor_Contrato__L__p_7_g_1'] == 0), 3,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R__pu_20_g_1'] == 3, base_dados['Saldo_Devedor_Contrato__L__p_7_g_1'] == 1), 3,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R__pu_20_g_1'] == 3, base_dados['Saldo_Devedor_Contrato__L__p_7_g_1'] == 2), 3,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R__pu_20_g_1'] == 3, base_dados['Saldo_Devedor_Contrato__L__p_7_g_1'] == 3), 3,
 0))))))))))))))))
base_dados['Saldo_Devedor_Contrato__L__p_7_g_1_c1_48_2'] = np.where(base_dados['Saldo_Devedor_Contrato__L__p_7_g_1_c1_48_1'] == 0, 0,
np.where(base_dados['Saldo_Devedor_Contrato__L__p_7_g_1_c1_48_1'] == 1, 1,
np.where(base_dados['Saldo_Devedor_Contrato__L__p_7_g_1_c1_48_1'] == 2, 2,
np.where(base_dados['Saldo_Devedor_Contrato__L__p_7_g_1_c1_48_1'] == 3, 3,
0))))
base_dados['Saldo_Devedor_Contrato__L__p_7_g_1_c1_48'] = np.where(base_dados['Saldo_Devedor_Contrato__L__p_7_g_1_c1_48_2'] == 0, 0,
np.where(base_dados['Saldo_Devedor_Contrato__L__p_7_g_1_c1_48_2'] == 1, 1,
np.where(base_dados['Saldo_Devedor_Contrato__L__p_7_g_1_c1_48_2'] == 2, 2,
np.where(base_dados['Saldo_Devedor_Contrato__L__p_7_g_1_c1_48_2'] == 3, 3,
 0))))

         
         
         
         
         
base_dados['Cep1_2__C__pu_7_g_1_c1_4_1'] = np.where(np.bitwise_and(base_dados['Cep1_2__p_6_g_1'] == 0, base_dados['Cep1_2__C__pu_7_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['Cep1_2__p_6_g_1'] == 0, base_dados['Cep1_2__C__pu_7_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['Cep1_2__p_6_g_1'] == 1, base_dados['Cep1_2__C__pu_7_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['Cep1_2__p_6_g_1'] == 1, base_dados['Cep1_2__C__pu_7_g_1'] == 1), 2,
 0))))
base_dados['Cep1_2__C__pu_7_g_1_c1_4_2'] = np.where(base_dados['Cep1_2__C__pu_7_g_1_c1_4_1'] == 0, 0,
np.where(base_dados['Cep1_2__C__pu_7_g_1_c1_4_1'] == 1, 1,
np.where(base_dados['Cep1_2__C__pu_7_g_1_c1_4_1'] == 2, 2,
0)))
base_dados['Cep1_2__C__pu_7_g_1_c1_4'] = np.where(base_dados['Cep1_2__C__pu_7_g_1_c1_4_2'] == 0, 0,
np.where(base_dados['Cep1_2__C__pu_7_g_1_c1_4_2'] == 1, 1,
np.where(base_dados['Cep1_2__C__pu_7_g_1_c1_4_2'] == 2, 2,
 0)))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Mantendo apenas as variáveis do modelo

# COMMAND ----------

import pickle
modelo=pickle.load(open(pickle_path + 'model_fit_santander_PJ.sav', 'rb'))

base_teste_c0 = base_dados[[chave,'QTD_AUTH_gh38','Email1_gh38','Divida_Outro_Credor_gh38','Telefone10_gh38','Desconto_Padrao_gh51','Telefone1_gh38','TIPO_PRODUTO_gh51','PRODUCT_1_gh38','UF1_gh51','Divida_Mesmo_Credor_gh38','Descricao_Produto_gh38','Codigo_Politica__p_5_g_1_c1_5','Saldo_Devedor_Contrato__L__p_7_g_1_c1_48','Cep1_2__C__pu_7_g_1_c1_4']]

var_fin_c0=['QTD_AUTH_gh38','Email1_gh38','Divida_Outro_Credor_gh38','Telefone10_gh38','Desconto_Padrao_gh51','Telefone1_gh38','TIPO_PRODUTO_gh51','PRODUCT_1_gh38','UF1_gh51','Divida_Mesmo_Credor_gh38','Descricao_Produto_gh38','Codigo_Politica__p_5_g_1_c1_5','Saldo_Devedor_Contrato__L__p_7_g_1_c1_48','Cep1_2__C__pu_7_g_1_c1_4']

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


x_teste2['P_1_L'] = np.log(x_teste2['P_1'])
x_teste2['P_1_L'] = np.where(x_teste2['P_1'] == 0, -1, x_teste2['P_1_L'])
x_teste2['P_1_L'] = np.where(x_teste2['P_1'] == np.nan, -2, x_teste2['P_1_L'])
x_teste2['P_1_L'] = x_teste2['P_1_L'].fillna(-2)

x_teste2['P_1_p_40_g_1'] = np.where(x_teste2['P_1'] <=0.0058383, 0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.0058383, x_teste2['P_1'] <= 0.017751681), 1,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.017751681, x_teste2['P_1'] <= 0.047498861), 2,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.047498861, x_teste2['P_1'] <= 0.261719456), 3,4))))

x_teste2['P_1_L_p_8_g_1'] = np.where(x_teste2['P_1_L'] <= -5.608567648, 0,
    np.where(np.bitwise_and(x_teste2['P_1_L'] > -5.608567648, x_teste2['P_1_L'] <= -4.796583975), 1,
    np.where(np.bitwise_and(x_teste2['P_1_L'] > -4.796583975, x_teste2['P_1_L'] <= -3.663754042), 2,
    np.where(np.bitwise_and(x_teste2['P_1_L'] > -3.663754042, x_teste2['P_1_L'] <= -2.230398893), 3,
    np.where(np.bitwise_and(x_teste2['P_1_L'] > -2.230398893, x_teste2['P_1_L'] <= -1.472562499), 4,
    np.where(np.bitwise_and(x_teste2['P_1_L'] > -1.472562499, x_teste2['P_1_L'] <= -0.347651467), 5,6))))))

x_teste2['GH'] = np.where(np.bitwise_and(x_teste2['P_1_L_p_8_g_1'] == 0, x_teste2['P_1_p_40_g_1'] == 0), 0,
    np.where(np.bitwise_and(x_teste2['P_1_L_p_8_g_1'] == 0, x_teste2['P_1_p_40_g_1'] == 1), 0,
    np.where(np.bitwise_and(x_teste2['P_1_L_p_8_g_1'] == 0, x_teste2['P_1_p_40_g_1'] == 2), 0,
    np.where(np.bitwise_and(x_teste2['P_1_L_p_8_g_1'] == 0, x_teste2['P_1_p_40_g_1'] == 3), 0,
    np.where(np.bitwise_and(x_teste2['P_1_L_p_8_g_1'] == 0, x_teste2['P_1_p_40_g_1'] == 4), 1,
    np.where(np.bitwise_and(x_teste2['P_1_L_p_8_g_1'] == 1, x_teste2['P_1_p_40_g_1'] == 0), 1,
    np.where(np.bitwise_and(x_teste2['P_1_L_p_8_g_1'] == 1, x_teste2['P_1_p_40_g_1'] == 1), 1,
    np.where(np.bitwise_and(x_teste2['P_1_L_p_8_g_1'] == 1, x_teste2['P_1_p_40_g_1'] == 2), 1,
    np.where(np.bitwise_and(x_teste2['P_1_L_p_8_g_1'] == 1, x_teste2['P_1_p_40_g_1'] == 3), 1,
    np.where(np.bitwise_and(x_teste2['P_1_L_p_8_g_1'] == 1, x_teste2['P_1_p_40_g_1'] == 4), 2,
    np.where(np.bitwise_and(x_teste2['P_1_L_p_8_g_1'] == 2, x_teste2['P_1_p_40_g_1'] == 0), 2,
    np.where(np.bitwise_and(x_teste2['P_1_L_p_8_g_1'] == 2, x_teste2['P_1_p_40_g_1'] == 1), 2,
    np.where(np.bitwise_and(x_teste2['P_1_L_p_8_g_1'] == 2, x_teste2['P_1_p_40_g_1'] == 2), 2,
    np.where(np.bitwise_and(x_teste2['P_1_L_p_8_g_1'] == 2, x_teste2['P_1_p_40_g_1'] == 3), 2,
    np.where(np.bitwise_and(x_teste2['P_1_L_p_8_g_1'] == 2, x_teste2['P_1_p_40_g_1'] == 4), 3,
    np.where(np.bitwise_and(x_teste2['P_1_L_p_8_g_1'] == 3, x_teste2['P_1_p_40_g_1'] == 0), 3,
    np.where(np.bitwise_and(x_teste2['P_1_L_p_8_g_1'] == 3, x_teste2['P_1_p_40_g_1'] == 1), 3,
    np.where(np.bitwise_and(x_teste2['P_1_L_p_8_g_1'] == 3, x_teste2['P_1_p_40_g_1'] == 2), 3,
    np.where(np.bitwise_and(x_teste2['P_1_L_p_8_g_1'] == 3, x_teste2['P_1_p_40_g_1'] == 3), 3,
    np.where(np.bitwise_and(x_teste2['P_1_L_p_8_g_1'] == 3, x_teste2['P_1_p_40_g_1'] == 4), 4,
    np.where(np.bitwise_and(x_teste2['P_1_L_p_8_g_1'] == 4, x_teste2['P_1_p_40_g_1'] == 0), 4,
    np.where(np.bitwise_and(x_teste2['P_1_L_p_8_g_1'] == 4, x_teste2['P_1_p_40_g_1'] == 1), 4,
    np.where(np.bitwise_and(x_teste2['P_1_L_p_8_g_1'] == 4, x_teste2['P_1_p_40_g_1'] == 2), 4,
    np.where(np.bitwise_and(x_teste2['P_1_L_p_8_g_1'] == 4, x_teste2['P_1_p_40_g_1'] == 3), 4,
    np.where(np.bitwise_and(x_teste2['P_1_L_p_8_g_1'] == 4, x_teste2['P_1_p_40_g_1'] == 4), 4,
    np.where(np.bitwise_and(x_teste2['P_1_L_p_8_g_1'] == 5, x_teste2['P_1_p_40_g_1'] == 0), 4,
    np.where(np.bitwise_and(x_teste2['P_1_L_p_8_g_1'] == 5, x_teste2['P_1_p_40_g_1'] == 1), 4,
    np.where(np.bitwise_and(x_teste2['P_1_L_p_8_g_1'] == 5, x_teste2['P_1_p_40_g_1'] == 2), 4,
    np.where(np.bitwise_and(x_teste2['P_1_L_p_8_g_1'] == 5, x_teste2['P_1_p_40_g_1'] == 3), 4,
    np.where(np.bitwise_and(x_teste2['P_1_L_p_8_g_1'] == 5, x_teste2['P_1_p_40_g_1'] == 4), 5,
    np.where(np.bitwise_and(x_teste2['P_1_L_p_8_g_1'] == 6, x_teste2['P_1_p_40_g_1'] == 0), 5,
    np.where(np.bitwise_and(x_teste2['P_1_L_p_8_g_1'] == 6, x_teste2['P_1_p_40_g_1'] == 1), 5,
    np.where(np.bitwise_and(x_teste2['P_1_L_p_8_g_1'] == 6, x_teste2['P_1_p_40_g_1'] == 2), 6,
    np.where(np.bitwise_and(x_teste2['P_1_L_p_8_g_1'] == 6, x_teste2['P_1_p_40_g_1'] == 3), 6,
    np.where(np.bitwise_and(x_teste2['P_1_L_p_8_g_1'] == 6, x_teste2['P_1_p_40_g_1'] == 4), 6,
             2)))))))))))))))))))))))))))))))))))

del x_teste2['P_1_L']
del x_teste2['P_1_p_40_g_1']
del x_teste2['P_1_L_p_8_g_1']

x_teste2


# COMMAND ----------

x_teste2.groupby(['GH'])['P_1'].count()

# COMMAND ----------

try:
  dbutils.fs.rm(outputpath, True)
except:
  pass
dbutils.fs.mkdirs(outputpath)

x_teste2.to_csv(open(os.path.join(outputpath_dbfs, 'pre_output:'+N_Base),'wb'))
os.path.join(outputpath_dbfs, 'pre_output:'+N_Base)