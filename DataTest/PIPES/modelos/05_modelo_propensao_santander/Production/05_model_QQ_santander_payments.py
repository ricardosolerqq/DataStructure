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

dbutils.widgets.text("chave","")
print('This is Chave widget :', dbutils.widgets.get("chave"))
chave = dbutils.widgets.get("chave")

dbutils.widgets.text("caminho_base","")
print('This is caminho_base widget :', dbutils.widgets.get("caminho_base"))
caminho_base = dbutils.widgets.get("caminho_base")

dbutils.widgets.text("caminho_pickle_model","")
print('This is caminho_pickle_model widget :', dbutils.widgets.get("caminho_pickle_model"))
caminho_pickle_model = dbutils.widgets.get("caminho_pickle_model")

dbutils.widgets.text("decimal_","")
print('This is decimal_ widget :', dbutils.widgets.get("decimal_"))
decimal_ = dbutils.widgets.get("decimal_")

dbutils.widgets.text("separador_","")
print('This is separador_ widget :', dbutils.widgets.get("separador_"))
separador_ = dbutils.widgets.get("separador_")

dbutils.widgets.text("N_Base","")
print('This is N_Base widget :', dbutils.widgets.get("N_Base"))
N_Base = dbutils.widgets.get("N_Base")

dbutils.widgets.text("sink","")
print('This is sink widget :', dbutils.widgets.get("sink"))
sink = dbutils.widgets.get("sink")

dbutils.widgets.text("sinkHomomgeneous","")
print('This is sinkHomomgeneous widget :', dbutils.widgets.get("sinkHomomgeneous"))
sinkHomomgeneous = dbutils.widgets.get("sinkHomomgeneous")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Importação da Base de Dados

# COMMAND ----------

base_dados = pd.read_csv(caminho_base+N_Base, sep=separador_, decimal=decimal_)
base_dados = base_dados[[chave,'Cep1', 'Qtde_Parcelas_Padrao', 'genero', 'DATA_ENTRADA_DIVIDA','Desconto_Padrao', 'Contrato_Altair', 'Indicador_Correntista', 'Descricao_Produto','Telefone1_skip:hot', 'Telefone1_skip:alto', 'Telefone1_skip:medio','Telefone1_skip:baixo', 'Telefone1_skip:nhot','Telefone1_sem_tags', 'Telefone2_skip:hot', 'Telefone2_skip:alto', 'Telefone2_skip:medio', 'Telefone2_skip:baixo','Telefone2_skip:nhot', 'Telefone2_sem_tags', 'Telefone3_skip:hot', 'Telefone3_skip:alto', 'Telefone3_skip:medio','Telefone3_skip:baixo', 'Telefone3_skip:nhot', 'Telefone3_sem_tags','rank:a','rank:b','rank:c','rank:d','rank:e','sem_skip']]

base_dados['PRODUTO'] = np.where(base_dados['Descricao_Produto'].str.contains('CARTAO')==True,'CARTAO',
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
                                    np.where(base_dados['Descricao_Produto'].str.contains('EMPRESTIMO')==True,'EMPRESTIMO FOLHA',            
                                    np.where(base_dados['Descricao_Produto'].str.contains('ADIANTAMENTO')==True,'AD',
                                    np.where(base_dados['Descricao_Produto'].str.contains('ADIANTAMENTO')==True,'AD',
                                    np.where(base_dados['Descricao_Produto'].str.contains('CHEQUE')==True,'CHEQUE ESPECIAL',
                                    np.where(base_dados['Descricao_Produto'].str.contains('CREDITO')==True,'EMPRESTIMO',
                                    np.where(base_dados['Descricao_Produto'].str.contains('REFIN')==True,'REFIN','OUTROS'))))))))))))))))))


base_dados['PRODUTO_TOP'] = np.where(base_dados['Descricao_Produto'].str.contains('CARTAO FREE GOLD MC')==True,'CARTAO FREE GOLD MC',
    np.where(base_dados['Descricao_Produto'].str.contains('CHEQUE ESPECIAL BANESPA')==True,'CHEQUE ESPECIAL BANESPA',
    np.where(base_dados['Descricao_Produto'].str.contains('CARTAO FREE GOLD VISA')==True,'CARTAO FREE GOLD VISA',
    np.where(base_dados['Descricao_Produto'].str.contains('REFIN')==True,'REFIN',
    np.where(base_dados['Descricao_Produto'].str.contains('CREDITO PESSOAL ELETRONICO')==True,'CREDITO PESSOAL ELETRONICO',
    np.where(base_dados['Descricao_Produto'].str.contains('SANTANDER SX MASTER')==True,'SANTANDER SX MASTER',
    np.where(base_dados['Descricao_Produto'].str.contains('EMPRESTIMOS EM FOLHA - CARNE')==True,'EMPRESTIMOS EM FOLHA - CARNE',
    np.where(base_dados['Descricao_Produto'].str.contains('SANTANDER SX VISA')==True,'SANTANDER SX VISA',
    np.where(base_dados['Descricao_Produto'].str.contains('CREDITO PESSOAL')==True,'CREDITO PESSOAL',
    np.where(base_dados['Descricao_Produto'].str.contains('ADIANTAMENTOS A DEPOSITANTES')==True,'ADIANTAMENTOS A DEPOSITANTES',
    np.where(base_dados['Descricao_Produto'].str.contains('CREDITO RENOVADO BANESPA')==True,'CREDITO RENOVADO BANESPA',
    np.where(base_dados['Descricao_Produto'].str.contains('CREDITO PREVENTIVO')==True,'CREDITO PREVENTIVO',
    np.where(base_dados['Descricao_Produto'].str.contains('EMPRESTIMO EM FOLHA')==True,'EMPRESTIMO EM FOLHA',
    np.where(base_dados['Descricao_Produto'].str.contains('SANTANDER STYLE PLATINUM MC')==True,'SANTANDER STYLE PLATINUM MC',
    np.where(base_dados['Descricao_Produto'].str.contains('CARTAO SANTANDER FIT MC')==True,'CARTAO SANTANDER FIT MC',
    np.where(base_dados['Descricao_Produto'].str.contains('CARTAO FLEX INTERNACIONAL MC')==True,'CARTAO FLEX INTERNACIONAL MC',
    np.where(base_dados['Descricao_Produto'].str.contains('CREDITO SOLUCOES')==True,'CREDITO SOLUCOES',
    np.where(base_dados['Descricao_Produto'].str.contains('SANTANDER FLEX MASTERCARD')==True,'SANTANDER FLEX MASTERCARD',
    np.where(base_dados['Descricao_Produto'].str.contains('CREDITO REORGANIZACAO')==True,'CREDITO REORGANIZACAO',
    np.where(base_dados['Descricao_Produto'].str.contains('CONTA DA TURMA, INDEPENDENTE E UNIVERSIDADE BANESPA')==True,'CONTA DA TURMA, INDEPENDENTE E UNIVERSIDADE BANESPA',
    np.where(base_dados['Descricao_Produto'].str.contains('SANTANDER STYLE PLATINUM')==True,'SANTANDER STYLE PLATINUM',
    np.where(base_dados['Descricao_Produto'].str.contains('CARTAO MASTERCARD GOLD')==True,'CARTAO MASTERCARD GOLD',
    np.where(base_dados['Descricao_Produto'].str.contains('MICROCREDITO')==True,'MICROCREDITO',
    np.where(base_dados['Descricao_Produto'].str.contains('CARTAO FLEX MASTERCARD')==True,'CARTAO FLEX MASTERCARD',
    np.where(base_dados['Descricao_Produto'].str.contains('SANTANDER ELITE PLATINUM MC')==True,'SANTANDER ELITE PLATINUM MC',
    np.where(base_dados['Descricao_Produto'].str.contains('SANTANDER ELITE PLATINUM VISA')==True,'SANTANDER ELITE PLATINUM VISA',
    np.where(base_dados['Descricao_Produto'].str.contains('SANTANDER BASICO MASTERCARD')==True,'SANTANDER BASICO MASTERCARD',
    np.where(base_dados['Descricao_Produto'].str.contains('SANTANDER FLEX NACIONAL MASTERCARD')==True,'SANTANDER FLEX NACIONAL MASTERCARD',
    np.where(base_dados['Descricao_Produto'].str.contains('CREDITO PESSOAL RENOVADO')==True,'CREDITO PESSOAL RENOVADO',
    np.where(base_dados['Descricao_Produto'].str.contains('1  2  3 DO SANTANDER')==True,'1  2  3 DO SANTANDER',
    np.where(base_dados['Descricao_Produto'].str.contains('MICROCREDITO RESOLUCAO 3422')==True,'MICROCREDITO RESOLUCAO 3422','OUTROS')))))))))))))))))))))))))))))))


base_dados['Telefone1'] = np.where(base_dados['Telefone1_skip:hot']==True,'skip_hot',
                                    np.where(base_dados['Telefone1_skip:alto']==True,'skip_alto',
                                    np.where(base_dados['Telefone1_skip:medio']==True,'skip_medio',
                                    np.where(base_dados['Telefone1_skip:baixo']==True,'skip_baixo',
                                    np.where(base_dados['Telefone1_skip:nhot']==True,'skip_nhot','sem_tags')))))

base_dados['Telefone2'] = np.where(base_dados['Telefone2_skip:hot']==True,'skip_hot',
                                    np.where(base_dados['Telefone2_skip:alto']==True,'skip_alto',
                                    np.where(base_dados['Telefone2_skip:medio']==True,'skip_medio',
                                    np.where(base_dados['Telefone2_skip:baixo']==True,'skip_baixo',
                                    np.where(base_dados['Telefone2_skip:nhot']==True,'skip_nhot','sem_tags')))))

base_dados['Telefone3'] = np.where(base_dados['Telefone3_skip:hot']==True,'skip_hot',
                                    np.where(base_dados['Telefone3_skip:alto']==True,'skip_alto',
                                    np.where(base_dados['Telefone3_skip:medio']==True,'skip_medio',
                                    np.where(base_dados['Telefone3_skip:baixo']==True,'skip_baixo',
                                    np.where(base_dados['Telefone3_skip:nhot']==True,'skip_nhot','sem_tags')))))


del base_dados['Telefone1_skip:hot']
del base_dados['Telefone1_skip:alto']
del base_dados['Telefone1_skip:medio']
del base_dados['Telefone1_skip:baixo']
del base_dados['Telefone1_skip:nhot']
del base_dados['Telefone1_sem_tags']
del base_dados['Telefone2_skip:hot']
del base_dados['Telefone2_skip:alto']
del base_dados['Telefone2_skip:medio']
del base_dados['Telefone2_skip:baixo']
del base_dados['Telefone2_skip:nhot']
del base_dados['Telefone2_sem_tags']
del base_dados['Telefone3_skip:hot']
del base_dados['Telefone3_skip:alto']
del base_dados['Telefone3_skip:medio']
del base_dados['Telefone3_skip:baixo']
del base_dados['Telefone3_skip:nhot']
del base_dados['Telefone3_sem_tags']

base_dados['Indicador_Correntista'] = base_dados['Indicador_Correntista'].map({True:1,False:0},na_action=None)
                                             
base_dados['rank'] = np.where(base_dados['rank:a']==True,'rank:a',
                                    np.where(base_dados['rank:b']==True,'rank:b',
                                    np.where(base_dados['rank:c']==True,'rank:c',
                                    np.where(base_dados['rank:d']==True,'rank:d',
                                    np.where(base_dados['rank:e']==True,'rank:e','sem_skip')))))

base_dados.fillna(-3)

base_dados['genero'] = base_dados['genero'].replace(np.nan, '-3')
base_dados['Cep1'] = base_dados['Cep1'].replace(np.nan, '').replace('-  0', '')


base_dados['Cep1'] = base_dados['Cep1'].str[:5]

base_dados['DATA_ENTRADA_DIVIDA'] = pd.to_datetime(base_dados['DATA_ENTRADA_DIVIDA'])
base_dados['MOB_ENTRADA'] = ((datetime.today()) - base_dados.DATA_ENTRADA_DIVIDA)/np.timedelta64(1, 'M')

del base_dados['DATA_ENTRADA_DIVIDA']

base_dados = base_dados.apply(pd.to_numeric, errors='ignore')

base_dados['Cep1'] = base_dados['Cep1'].replace(np.nan, -3)

base_dados.fillna(-3)

print("shape da Base de Dados:",base_dados.shape)

base_dados.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gerando variáveis Categóricas

# COMMAND ----------

                                                
base_dados['PRODUTO_gh30'] = np.where(base_dados['PRODUTO'] == 'AD', 0,
np.where(base_dados['PRODUTO'] == 'CARTAO', 1,
np.where(base_dados['PRODUTO'] == 'CHEQUE ESPECIAL', 2,
np.where(base_dados['PRODUTO'] == 'EMPRESTIMO', 3,
np.where(base_dados['PRODUTO'] == 'EMPRESTIMO FOLHA', 4,
np.where(base_dados['PRODUTO'] == 'OUTROS', 5,
np.where(base_dados['PRODUTO'] == 'REFIN', 6,
0)))))))
base_dados['PRODUTO_gh31'] = np.where(base_dados['PRODUTO_gh30'] == 0, 0,
np.where(base_dados['PRODUTO_gh30'] == 1, 1,
np.where(base_dados['PRODUTO_gh30'] == 2, 2,
np.where(base_dados['PRODUTO_gh30'] == 3, 3,
np.where(base_dados['PRODUTO_gh30'] == 4, 4,
np.where(base_dados['PRODUTO_gh30'] == 5, 5,
np.where(base_dados['PRODUTO_gh30'] == 6, 5,
0)))))))
base_dados['PRODUTO_gh32'] = np.where(base_dados['PRODUTO_gh31'] == 0, 0,
np.where(base_dados['PRODUTO_gh31'] == 1, 1,
np.where(base_dados['PRODUTO_gh31'] == 2, 2,
np.where(base_dados['PRODUTO_gh31'] == 3, 3,
np.where(base_dados['PRODUTO_gh31'] == 4, 4,
np.where(base_dados['PRODUTO_gh31'] == 5, 5,
0))))))
base_dados['PRODUTO_gh33'] = np.where(base_dados['PRODUTO_gh32'] == 0, 0,
np.where(base_dados['PRODUTO_gh32'] == 1, 1,
np.where(base_dados['PRODUTO_gh32'] == 2, 2,
np.where(base_dados['PRODUTO_gh32'] == 3, 3,
np.where(base_dados['PRODUTO_gh32'] == 4, 4,
np.where(base_dados['PRODUTO_gh32'] == 5, 5,
0))))))
base_dados['PRODUTO_gh34'] = np.where(base_dados['PRODUTO_gh33'] == 0, 4,
np.where(base_dados['PRODUTO_gh33'] == 1, 1,
np.where(base_dados['PRODUTO_gh33'] == 2, 2,
np.where(base_dados['PRODUTO_gh33'] == 3, 3,
np.where(base_dados['PRODUTO_gh33'] == 4, 4,
np.where(base_dados['PRODUTO_gh33'] == 5, 5,
0))))))
base_dados['PRODUTO_gh35'] = np.where(base_dados['PRODUTO_gh34'] == 1, 0,
np.where(base_dados['PRODUTO_gh34'] == 2, 1,
np.where(base_dados['PRODUTO_gh34'] == 3, 2,
np.where(base_dados['PRODUTO_gh34'] == 4, 3,
np.where(base_dados['PRODUTO_gh34'] == 5, 4,
0)))))
base_dados['PRODUTO_gh36'] = np.where(base_dados['PRODUTO_gh35'] == 0, 4,
np.where(base_dados['PRODUTO_gh35'] == 1, 3,
np.where(base_dados['PRODUTO_gh35'] == 2, 1,
np.where(base_dados['PRODUTO_gh35'] == 3, 0,
np.where(base_dados['PRODUTO_gh35'] == 4, 2,
0)))))
base_dados['PRODUTO_gh37'] = np.where(base_dados['PRODUTO_gh36'] == 0, 1,
np.where(base_dados['PRODUTO_gh36'] == 1, 1,
np.where(base_dados['PRODUTO_gh36'] == 2, 2,
np.where(base_dados['PRODUTO_gh36'] == 3, 3,
np.where(base_dados['PRODUTO_gh36'] == 4, 4,
0)))))
base_dados['PRODUTO_gh38'] = np.where(base_dados['PRODUTO_gh37'] == 1, 0,
np.where(base_dados['PRODUTO_gh37'] == 2, 1,
np.where(base_dados['PRODUTO_gh37'] == 3, 2,
np.where(base_dados['PRODUTO_gh37'] == 4, 3,
0))))
         
         
         
         
         
         
         

         
         
         
         
         
         
base_dados['Telefone1_gh30'] = np.where(base_dados['Telefone1'] == 'sem_tags', 0,
np.where(base_dados['Telefone1'] == 'skip_alto', 1,
np.where(base_dados['Telefone1'] == 'skip_baixo', 2,
np.where(base_dados['Telefone1'] == 'skip_hot', 3,
np.where(base_dados['Telefone1'] == 'skip_medio', 4,
np.where(base_dados['Telefone1'] == 'skip_nhot', 5,
0))))))
base_dados['Telefone1_gh31'] = np.where(base_dados['Telefone1_gh30'] == 0, 0,
np.where(base_dados['Telefone1_gh30'] == 1, 0,
np.where(base_dados['Telefone1_gh30'] == 2, 2,
np.where(base_dados['Telefone1_gh30'] == 3, 3,
np.where(base_dados['Telefone1_gh30'] == 4, 4,
np.where(base_dados['Telefone1_gh30'] == 5, 5,
0))))))
base_dados['Telefone1_gh32'] = np.where(base_dados['Telefone1_gh31'] == 0, 0,
np.where(base_dados['Telefone1_gh31'] == 2, 1,
np.where(base_dados['Telefone1_gh31'] == 3, 2,
np.where(base_dados['Telefone1_gh31'] == 4, 3,
np.where(base_dados['Telefone1_gh31'] == 5, 4,
0)))))
base_dados['Telefone1_gh33'] = np.where(base_dados['Telefone1_gh32'] == 0, 0,
np.where(base_dados['Telefone1_gh32'] == 1, 1,
np.where(base_dados['Telefone1_gh32'] == 2, 2,
np.where(base_dados['Telefone1_gh32'] == 3, 3,
np.where(base_dados['Telefone1_gh32'] == 4, 4,
0)))))
base_dados['Telefone1_gh34'] = np.where(base_dados['Telefone1_gh33'] == 0, 0,
np.where(base_dados['Telefone1_gh33'] == 1, 1,
np.where(base_dados['Telefone1_gh33'] == 2, 2,
np.where(base_dados['Telefone1_gh33'] == 3, 3,
np.where(base_dados['Telefone1_gh33'] == 4, 5,
0)))))
base_dados['Telefone1_gh35'] = np.where(base_dados['Telefone1_gh34'] == 0, 0,
np.where(base_dados['Telefone1_gh34'] == 1, 1,
np.where(base_dados['Telefone1_gh34'] == 2, 2,
np.where(base_dados['Telefone1_gh34'] == 3, 3,
np.where(base_dados['Telefone1_gh34'] == 5, 4,
0)))))
base_dados['Telefone1_gh36'] = np.where(base_dados['Telefone1_gh35'] == 0, 3,
np.where(base_dados['Telefone1_gh35'] == 1, 1,
np.where(base_dados['Telefone1_gh35'] == 2, 4,
np.where(base_dados['Telefone1_gh35'] == 3, 1,
np.where(base_dados['Telefone1_gh35'] == 4, 0,
0)))))
base_dados['Telefone1_gh37'] = np.where(base_dados['Telefone1_gh36'] == 0, 1,
np.where(base_dados['Telefone1_gh36'] == 1, 1,
np.where(base_dados['Telefone1_gh36'] == 3, 2,
np.where(base_dados['Telefone1_gh36'] == 4, 3,
0))))
base_dados['Telefone1_gh38'] = np.where(base_dados['Telefone1_gh37'] == 1, 0,
np.where(base_dados['Telefone1_gh37'] == 2, 1,
np.where(base_dados['Telefone1_gh37'] == 3, 2,
0)))
         
         
         
         
         
         
         
base_dados['Telefone2_gh30'] = np.where(base_dados['Telefone2'] == 'sem_tags', 0,
np.where(base_dados['Telefone2'] == 'skip_alto', 1,
np.where(base_dados['Telefone2'] == 'skip_baixo', 2,
np.where(base_dados['Telefone2'] == 'skip_hot', 3,
np.where(base_dados['Telefone2'] == 'skip_medio', 4,
np.where(base_dados['Telefone2'] == 'skip_nhot', 5,
0))))))
base_dados['Telefone2_gh31'] = np.where(base_dados['Telefone2_gh30'] == 0, 0,
np.where(base_dados['Telefone2_gh30'] == 1, 0,
np.where(base_dados['Telefone2_gh30'] == 2, 2,
np.where(base_dados['Telefone2_gh30'] == 3, 3,
np.where(base_dados['Telefone2_gh30'] == 4, 4,
np.where(base_dados['Telefone2_gh30'] == 5, 5,
0))))))
base_dados['Telefone2_gh32'] = np.where(base_dados['Telefone2_gh31'] == 0, 0,
np.where(base_dados['Telefone2_gh31'] == 2, 1,
np.where(base_dados['Telefone2_gh31'] == 3, 2,
np.where(base_dados['Telefone2_gh31'] == 4, 3,
np.where(base_dados['Telefone2_gh31'] == 5, 4,
0)))))
base_dados['Telefone2_gh33'] = np.where(base_dados['Telefone2_gh32'] == 0, 0,
np.where(base_dados['Telefone2_gh32'] == 1, 1,
np.where(base_dados['Telefone2_gh32'] == 2, 2,
np.where(base_dados['Telefone2_gh32'] == 3, 3,
np.where(base_dados['Telefone2_gh32'] == 4, 4,
0)))))
base_dados['Telefone2_gh34'] = np.where(base_dados['Telefone2_gh33'] == 0, 0,
np.where(base_dados['Telefone2_gh33'] == 1, 1,
np.where(base_dados['Telefone2_gh33'] == 2, 3,
np.where(base_dados['Telefone2_gh33'] == 3, 3,
np.where(base_dados['Telefone2_gh33'] == 4, 5,
0)))))
base_dados['Telefone2_gh35'] = np.where(base_dados['Telefone2_gh34'] == 0, 0,
np.where(base_dados['Telefone2_gh34'] == 1, 1,
np.where(base_dados['Telefone2_gh34'] == 3, 2,
np.where(base_dados['Telefone2_gh34'] == 5, 3,
0))))
base_dados['Telefone2_gh36'] = np.where(base_dados['Telefone2_gh35'] == 0, 2,
np.where(base_dados['Telefone2_gh35'] == 1, 1,
np.where(base_dados['Telefone2_gh35'] == 2, 3,
np.where(base_dados['Telefone2_gh35'] == 3, 0,
0))))
base_dados['Telefone2_gh37'] = np.where(base_dados['Telefone2_gh36'] == 0, 0,
np.where(base_dados['Telefone2_gh36'] == 1, 0,
np.where(base_dados['Telefone2_gh36'] == 2, 2,
np.where(base_dados['Telefone2_gh36'] == 3, 3,
0))))
base_dados['Telefone2_gh38'] = np.where(base_dados['Telefone2_gh37'] == 0, 0,
np.where(base_dados['Telefone2_gh37'] == 2, 1,
np.where(base_dados['Telefone2_gh37'] == 3, 2,
0)))
         
         
         
         
         
         
base_dados['Telefone3_gh30'] = np.where(base_dados['Telefone3'] == 'sem_tags', 0,
np.where(base_dados['Telefone3'] == 'skip_alto', 1,
np.where(base_dados['Telefone3'] == 'skip_baixo', 2,
np.where(base_dados['Telefone3'] == 'skip_hot', 3,
np.where(base_dados['Telefone3'] == 'skip_medio', 4,
np.where(base_dados['Telefone3'] == 'skip_nhot', 5,
0))))))
base_dados['Telefone3_gh31'] = np.where(base_dados['Telefone3_gh30'] == 0, 0,
np.where(base_dados['Telefone3_gh30'] == 1, 1,
np.where(base_dados['Telefone3_gh30'] == 2, 2,
np.where(base_dados['Telefone3_gh30'] == 3, 3,
np.where(base_dados['Telefone3_gh30'] == 4, 4,
np.where(base_dados['Telefone3_gh30'] == 5, 5,
0))))))
base_dados['Telefone3_gh32'] = np.where(base_dados['Telefone3_gh31'] == 0, 0,
np.where(base_dados['Telefone3_gh31'] == 1, 1,
np.where(base_dados['Telefone3_gh31'] == 2, 2,
np.where(base_dados['Telefone3_gh31'] == 3, 3,
np.where(base_dados['Telefone3_gh31'] == 4, 4,
np.where(base_dados['Telefone3_gh31'] == 5, 5,
0))))))
base_dados['Telefone3_gh33'] = np.where(base_dados['Telefone3_gh32'] == 0, 0,
np.where(base_dados['Telefone3_gh32'] == 1, 1,
np.where(base_dados['Telefone3_gh32'] == 2, 2,
np.where(base_dados['Telefone3_gh32'] == 3, 3,
np.where(base_dados['Telefone3_gh32'] == 4, 4,
np.where(base_dados['Telefone3_gh32'] == 5, 5,
0))))))
base_dados['Telefone3_gh34'] = np.where(base_dados['Telefone3_gh33'] == 0, 0,
np.where(base_dados['Telefone3_gh33'] == 1, 5,
np.where(base_dados['Telefone3_gh33'] == 2, 2,
np.where(base_dados['Telefone3_gh33'] == 3, 4,
np.where(base_dados['Telefone3_gh33'] == 4, 4,
np.where(base_dados['Telefone3_gh33'] == 5, 5,
0))))))
base_dados['Telefone3_gh35'] = np.where(base_dados['Telefone3_gh34'] == 0, 0,
np.where(base_dados['Telefone3_gh34'] == 2, 1,
np.where(base_dados['Telefone3_gh34'] == 4, 2,
np.where(base_dados['Telefone3_gh34'] == 5, 3,
0))))
base_dados['Telefone3_gh36'] = np.where(base_dados['Telefone3_gh35'] == 0, 1,
np.where(base_dados['Telefone3_gh35'] == 1, 1,
np.where(base_dados['Telefone3_gh35'] == 2, 3,
np.where(base_dados['Telefone3_gh35'] == 3, 0,
0))))
base_dados['Telefone3_gh37'] = np.where(base_dados['Telefone3_gh36'] == 0, 1,
np.where(base_dados['Telefone3_gh36'] == 1, 1,
np.where(base_dados['Telefone3_gh36'] == 3, 2,
0)))
base_dados['Telefone3_gh38'] = np.where(base_dados['Telefone3_gh37'] == 1, 0,
np.where(base_dados['Telefone3_gh37'] == 2, 1,
0))
                                        
                                        
                                        
                                        
                                        
                                        
                                        
base_dados['rank_gh30'] = np.where(base_dados['rank'] == 'rank:a', 0,
np.where(base_dados['rank'] == 'rank:b', 1,
np.where(base_dados['rank'] == 'rank:c', 2,
np.where(base_dados['rank'] == 'rank:d', 3,
np.where(base_dados['rank'] == 'rank:e', 4,
np.where(base_dados['rank'] == 'sem_skip', 5,
0))))))
base_dados['rank_gh31'] = np.where(base_dados['rank_gh30'] == 0, 0,
np.where(base_dados['rank_gh30'] == 1, 1,
np.where(base_dados['rank_gh30'] == 2, 2,
np.where(base_dados['rank_gh30'] == 3, 3,
np.where(base_dados['rank_gh30'] == 4, 4,
np.where(base_dados['rank_gh30'] == 5, 5,
0))))))
base_dados['rank_gh32'] = np.where(base_dados['rank_gh31'] == 0, 0,
np.where(base_dados['rank_gh31'] == 1, 1,
np.where(base_dados['rank_gh31'] == 2, 2,
np.where(base_dados['rank_gh31'] == 3, 3,
np.where(base_dados['rank_gh31'] == 4, 4,
np.where(base_dados['rank_gh31'] == 5, 5,
0))))))
base_dados['rank_gh33'] = np.where(base_dados['rank_gh32'] == 0, 0,
np.where(base_dados['rank_gh32'] == 1, 1,
np.where(base_dados['rank_gh32'] == 2, 2,
np.where(base_dados['rank_gh32'] == 3, 3,
np.where(base_dados['rank_gh32'] == 4, 4,
np.where(base_dados['rank_gh32'] == 5, 5,
0))))))
base_dados['rank_gh34'] = np.where(base_dados['rank_gh33'] == 0, 0,
np.where(base_dados['rank_gh33'] == 1, 1,
np.where(base_dados['rank_gh33'] == 2, 2,
np.where(base_dados['rank_gh33'] == 3, 3,
np.where(base_dados['rank_gh33'] == 4, 4,
np.where(base_dados['rank_gh33'] == 5, 0,
0))))))
base_dados['rank_gh35'] = np.where(base_dados['rank_gh34'] == 0, 0,
np.where(base_dados['rank_gh34'] == 1, 1,
np.where(base_dados['rank_gh34'] == 2, 2,
np.where(base_dados['rank_gh34'] == 3, 3,
np.where(base_dados['rank_gh34'] == 4, 4,
0)))))
base_dados['rank_gh36'] = np.where(base_dados['rank_gh35'] == 0, 4,
np.where(base_dados['rank_gh35'] == 1, 3,
np.where(base_dados['rank_gh35'] == 2, 2,
np.where(base_dados['rank_gh35'] == 3, 1,
np.where(base_dados['rank_gh35'] == 4, 0,
0)))))
base_dados['rank_gh37'] = np.where(base_dados['rank_gh36'] == 0, 1,
np.where(base_dados['rank_gh36'] == 1, 1,
np.where(base_dados['rank_gh36'] == 2, 2,
np.where(base_dados['rank_gh36'] == 3, 3,
np.where(base_dados['rank_gh36'] == 4, 4,
0)))))
base_dados['rank_gh38'] = np.where(base_dados['rank_gh37'] == 1, 0,
np.where(base_dados['rank_gh37'] == 2, 1,
np.where(base_dados['rank_gh37'] == 3, 2,
np.where(base_dados['rank_gh37'] == 4, 3,
0))))
         
         
         
         
         
         
         
base_dados['Desconto_Padrao_gh30'] = np.where(base_dados['Desconto_Padrao'] == 0.0, 0,
np.where(base_dados['Desconto_Padrao'] == 10.0, 1,
np.where(base_dados['Desconto_Padrao'] == 15.0, 2,
np.where(base_dados['Desconto_Padrao'] == 25.0, 3,
np.where(base_dados['Desconto_Padrao'] == 27.0, 4,
np.where(base_dados['Desconto_Padrao'] == 28.5, 5,
np.where(base_dados['Desconto_Padrao'] == 30.0, 6,
np.where(base_dados['Desconto_Padrao'] == 40.0, 7,
np.where(base_dados['Desconto_Padrao'] == 45.0, 8,
np.where(base_dados['Desconto_Padrao'] == 47.5, 9,
np.where(base_dados['Desconto_Padrao'] == 50.0, 10,
np.where(base_dados['Desconto_Padrao'] == 55.0, 11,
np.where(base_dados['Desconto_Padrao'] == 65.0, 12,
np.where(base_dados['Desconto_Padrao'] == 70.0, 13,
np.where(base_dados['Desconto_Padrao'] == 75.0, 14,
np.where(base_dados['Desconto_Padrao'] == 76.0, 15,
np.where(base_dados['Desconto_Padrao'] == 77.0, 16,
np.where(base_dados['Desconto_Padrao'] == 80.0, 17,
np.where(base_dados['Desconto_Padrao'] == 82.0, 18,
np.where(base_dados['Desconto_Padrao'] == 85.0, 19,
np.where(base_dados['Desconto_Padrao'] == 90.0, 20,
np.where(base_dados['Desconto_Padrao'] == 93.0, 21,
0))))))))))))))))))))))
base_dados['Desconto_Padrao_gh31'] = np.where(base_dados['Desconto_Padrao_gh30'] == 0, 0,
np.where(base_dados['Desconto_Padrao_gh30'] == 1, 1,
np.where(base_dados['Desconto_Padrao_gh30'] == 2, 2,
np.where(base_dados['Desconto_Padrao_gh30'] == 3, 3,
np.where(base_dados['Desconto_Padrao_gh30'] == 4, 4,
np.where(base_dados['Desconto_Padrao_gh30'] == 5, 5,
np.where(base_dados['Desconto_Padrao_gh30'] == 6, 6,
np.where(base_dados['Desconto_Padrao_gh30'] == 7, 6,
np.where(base_dados['Desconto_Padrao_gh30'] == 8, 8,
np.where(base_dados['Desconto_Padrao_gh30'] == 9, 9,
np.where(base_dados['Desconto_Padrao_gh30'] == 10, 10,
np.where(base_dados['Desconto_Padrao_gh30'] == 11, 11,
np.where(base_dados['Desconto_Padrao_gh30'] == 12, 12,
np.where(base_dados['Desconto_Padrao_gh30'] == 13, 12,
np.where(base_dados['Desconto_Padrao_gh30'] == 14, 14,
np.where(base_dados['Desconto_Padrao_gh30'] == 15, 15,
np.where(base_dados['Desconto_Padrao_gh30'] == 16, 16,
np.where(base_dados['Desconto_Padrao_gh30'] == 17, 17,
np.where(base_dados['Desconto_Padrao_gh30'] == 18, 18,
np.where(base_dados['Desconto_Padrao_gh30'] == 19, 19,
np.where(base_dados['Desconto_Padrao_gh30'] == 20, 20,
np.where(base_dados['Desconto_Padrao_gh30'] == 21, 21,
0))))))))))))))))))))))
base_dados['Desconto_Padrao_gh32'] = np.where(base_dados['Desconto_Padrao_gh31'] == 0, 0,
np.where(base_dados['Desconto_Padrao_gh31'] == 1, 1,
np.where(base_dados['Desconto_Padrao_gh31'] == 2, 2,
np.where(base_dados['Desconto_Padrao_gh31'] == 3, 3,
np.where(base_dados['Desconto_Padrao_gh31'] == 4, 4,
np.where(base_dados['Desconto_Padrao_gh31'] == 5, 5,
np.where(base_dados['Desconto_Padrao_gh31'] == 6, 6,
np.where(base_dados['Desconto_Padrao_gh31'] == 8, 7,
np.where(base_dados['Desconto_Padrao_gh31'] == 9, 8,
np.where(base_dados['Desconto_Padrao_gh31'] == 10, 9,
np.where(base_dados['Desconto_Padrao_gh31'] == 11, 10,
np.where(base_dados['Desconto_Padrao_gh31'] == 12, 11,
np.where(base_dados['Desconto_Padrao_gh31'] == 14, 12,
np.where(base_dados['Desconto_Padrao_gh31'] == 15, 13,
np.where(base_dados['Desconto_Padrao_gh31'] == 16, 14,
np.where(base_dados['Desconto_Padrao_gh31'] == 17, 15,
np.where(base_dados['Desconto_Padrao_gh31'] == 18, 16,
np.where(base_dados['Desconto_Padrao_gh31'] == 19, 17,
np.where(base_dados['Desconto_Padrao_gh31'] == 20, 18,
np.where(base_dados['Desconto_Padrao_gh31'] == 21, 19,
0))))))))))))))))))))
base_dados['Desconto_Padrao_gh33'] = np.where(base_dados['Desconto_Padrao_gh32'] == 0, 0,
np.where(base_dados['Desconto_Padrao_gh32'] == 1, 1,
np.where(base_dados['Desconto_Padrao_gh32'] == 2, 2,
np.where(base_dados['Desconto_Padrao_gh32'] == 3, 3,
np.where(base_dados['Desconto_Padrao_gh32'] == 4, 4,
np.where(base_dados['Desconto_Padrao_gh32'] == 5, 5,
np.where(base_dados['Desconto_Padrao_gh32'] == 6, 6,
np.where(base_dados['Desconto_Padrao_gh32'] == 7, 7,
np.where(base_dados['Desconto_Padrao_gh32'] == 8, 8,
np.where(base_dados['Desconto_Padrao_gh32'] == 9, 9,
np.where(base_dados['Desconto_Padrao_gh32'] == 10, 10,
np.where(base_dados['Desconto_Padrao_gh32'] == 11, 11,
np.where(base_dados['Desconto_Padrao_gh32'] == 12, 12,
np.where(base_dados['Desconto_Padrao_gh32'] == 13, 13,
np.where(base_dados['Desconto_Padrao_gh32'] == 14, 14,
np.where(base_dados['Desconto_Padrao_gh32'] == 15, 15,
np.where(base_dados['Desconto_Padrao_gh32'] == 16, 16,
np.where(base_dados['Desconto_Padrao_gh32'] == 17, 17,
np.where(base_dados['Desconto_Padrao_gh32'] == 18, 18,
np.where(base_dados['Desconto_Padrao_gh32'] == 19, 19,
0))))))))))))))))))))
base_dados['Desconto_Padrao_gh34'] = np.where(base_dados['Desconto_Padrao_gh33'] == 0, 0,
np.where(base_dados['Desconto_Padrao_gh33'] == 1, 12,
np.where(base_dados['Desconto_Padrao_gh33'] == 2, 2,
np.where(base_dados['Desconto_Padrao_gh33'] == 3, 0,
np.where(base_dados['Desconto_Padrao_gh33'] == 4, 12,
np.where(base_dados['Desconto_Padrao_gh33'] == 5, 19,
np.where(base_dados['Desconto_Padrao_gh33'] == 6, 6,
np.where(base_dados['Desconto_Padrao_gh33'] == 7, 18,
np.where(base_dados['Desconto_Padrao_gh33'] == 8, 2,
np.where(base_dados['Desconto_Padrao_gh33'] == 9, 12,
np.where(base_dados['Desconto_Padrao_gh33'] == 10, 2,
np.where(base_dados['Desconto_Padrao_gh33'] == 11, 0,
np.where(base_dados['Desconto_Padrao_gh33'] == 12, 12,
np.where(base_dados['Desconto_Padrao_gh33'] == 13, 19,
np.where(base_dados['Desconto_Padrao_gh33'] == 14, 2,
np.where(base_dados['Desconto_Padrao_gh33'] == 15, 15,
np.where(base_dados['Desconto_Padrao_gh33'] == 16, 12,
np.where(base_dados['Desconto_Padrao_gh33'] == 17, 0,
np.where(base_dados['Desconto_Padrao_gh33'] == 18, 18,
np.where(base_dados['Desconto_Padrao_gh33'] == 19, 19,
0))))))))))))))))))))
base_dados['Desconto_Padrao_gh35'] = np.where(base_dados['Desconto_Padrao_gh34'] == 0, 0,
np.where(base_dados['Desconto_Padrao_gh34'] == 2, 1,
np.where(base_dados['Desconto_Padrao_gh34'] == 6, 2,
np.where(base_dados['Desconto_Padrao_gh34'] == 12, 3,
np.where(base_dados['Desconto_Padrao_gh34'] == 15, 4,
np.where(base_dados['Desconto_Padrao_gh34'] == 18, 5,
np.where(base_dados['Desconto_Padrao_gh34'] == 19, 6,
0)))))))
base_dados['Desconto_Padrao_gh36'] = np.where(base_dados['Desconto_Padrao_gh35'] == 0, 2,
np.where(base_dados['Desconto_Padrao_gh35'] == 1, 0,
np.where(base_dados['Desconto_Padrao_gh35'] == 2, 4,
np.where(base_dados['Desconto_Padrao_gh35'] == 3, 6,
np.where(base_dados['Desconto_Padrao_gh35'] == 4, 2,
np.where(base_dados['Desconto_Padrao_gh35'] == 5, 5,
np.where(base_dados['Desconto_Padrao_gh35'] == 6, 1,
0)))))))
base_dados['Desconto_Padrao_gh37'] = np.where(base_dados['Desconto_Padrao_gh36'] == 0, 1,
np.where(base_dados['Desconto_Padrao_gh36'] == 1, 1,
np.where(base_dados['Desconto_Padrao_gh36'] == 2, 2,
np.where(base_dados['Desconto_Padrao_gh36'] == 4, 3,
np.where(base_dados['Desconto_Padrao_gh36'] == 5, 4,
np.where(base_dados['Desconto_Padrao_gh36'] == 6, 4,
0))))))
base_dados['Desconto_Padrao_gh38'] = np.where(base_dados['Desconto_Padrao_gh37'] == 1, 0,
np.where(base_dados['Desconto_Padrao_gh37'] == 2, 1,
np.where(base_dados['Desconto_Padrao_gh37'] == 3, 2,
np.where(base_dados['Desconto_Padrao_gh37'] == 4, 3,
0))))
                                                
                                                
                                                
                                                
                                                
                                                
                                                
                                                
base_dados['Indicador_Correntista_gh30'] = np.where(base_dados['Indicador_Correntista'] == 0, 0,
np.where(base_dados['Indicador_Correntista'] == 1, 1,
0))
base_dados['Indicador_Correntista_gh31'] = np.where(base_dados['Indicador_Correntista_gh30'] == 0, 0,
np.where(base_dados['Indicador_Correntista_gh30'] == 1, 1,
0))
base_dados['Indicador_Correntista_gh32'] = np.where(base_dados['Indicador_Correntista_gh31'] == 0, 0,
np.where(base_dados['Indicador_Correntista_gh31'] == 1, 1,
0))
base_dados['Indicador_Correntista_gh33'] = np.where(base_dados['Indicador_Correntista_gh32'] == 0, 0,
np.where(base_dados['Indicador_Correntista_gh32'] == 1, 1,
0))
base_dados['Indicador_Correntista_gh34'] = np.where(base_dados['Indicador_Correntista_gh33'] == 0, 0,
np.where(base_dados['Indicador_Correntista_gh33'] == 1, 1,
0))
base_dados['Indicador_Correntista_gh35'] = np.where(base_dados['Indicador_Correntista_gh34'] == 0, 0,
np.where(base_dados['Indicador_Correntista_gh34'] == 1, 1,
0))
base_dados['Indicador_Correntista_gh36'] = np.where(base_dados['Indicador_Correntista_gh35'] == 0, 1,
np.where(base_dados['Indicador_Correntista_gh35'] == 1, 0,
0))
base_dados['Indicador_Correntista_gh37'] = np.where(base_dados['Indicador_Correntista_gh36'] == 0, 0,
np.where(base_dados['Indicador_Correntista_gh36'] == 1, 1,
0))
base_dados['Indicador_Correntista_gh38'] = np.where(base_dados['Indicador_Correntista_gh37'] == 0, 0,
np.where(base_dados['Indicador_Correntista_gh37'] == 1, 1,
0))





base_dados['genero_gh40'] = np.where(base_dados['genero'] == '-3', 0,
np.where(base_dados['genero'] == 'F', 1,
np.where(base_dados['genero'] == 'M', 2,
0)))
base_dados['genero_gh41'] = np.where(base_dados['genero_gh40'] == 0, 0,
np.where(base_dados['genero_gh40'] == 1, -5,
np.where(base_dados['genero_gh40'] == 2, -5,
0)))
base_dados['genero_gh42'] = np.where(base_dados['genero_gh41'] == -5, 0,
np.where(base_dados['genero_gh41'] == 0, 1,
0))
base_dados['genero_gh43'] = np.where(base_dados['genero_gh42'] == 0, 0,
np.where(base_dados['genero_gh42'] == 1, 1,
0))
base_dados['genero_gh44'] = np.where(base_dados['genero_gh43'] == 0, 0,
np.where(base_dados['genero_gh43'] == 1, 1,
0))
base_dados['genero_gh45'] = np.where(base_dados['genero_gh44'] == 0, 0,
np.where(base_dados['genero_gh44'] == 1, 1,
0))
base_dados['genero_gh46'] = np.where(base_dados['genero_gh45'] == 0, 0,
np.where(base_dados['genero_gh45'] == 1, 1,
0))
base_dados['genero_gh47'] = np.where(base_dados['genero_gh46'] == 0, 0,
np.where(base_dados['genero_gh46'] == 1, 1,
0))
base_dados['genero_gh48'] = np.where(base_dados['genero_gh47'] == 0, 0,
np.where(base_dados['genero_gh47'] == 1, 1,
0))
base_dados['genero_gh49'] = np.where(base_dados['genero_gh48'] == 0, 0,
np.where(base_dados['genero_gh48'] == 1, 1,
0))
base_dados['genero_gh50'] = np.where(base_dados['genero_gh49'] == 0, 0,
np.where(base_dados['genero_gh49'] == 1, 1,
0))
base_dados['genero_gh51'] = np.where(base_dados['genero_gh50'] == 0, 0,
np.where(base_dados['genero_gh50'] == 1, 1,
0))

                                     
                                     
                                     
                                     
base_dados['PRODUTO_TOP_gh40'] = np.where(base_dados['PRODUTO_TOP'] == '1  2  3 DO SANTANDER', 0,
np.where(base_dados['PRODUTO_TOP'] == 'ADIANTAMENTOS A DEPOSITANTES', 1,
np.where(base_dados['PRODUTO_TOP'] == 'CARTAO FLEX INTERNACIONAL MC', 2,
np.where(base_dados['PRODUTO_TOP'] == 'CARTAO FLEX MASTERCARD', 3,
np.where(base_dados['PRODUTO_TOP'] == 'CARTAO FREE GOLD MC', 4,
np.where(base_dados['PRODUTO_TOP'] == 'CARTAO FREE GOLD VISA', 5,
np.where(base_dados['PRODUTO_TOP'] == 'CARTAO MASTERCARD GOLD', 6,
np.where(base_dados['PRODUTO_TOP'] == 'CARTAO SANTANDER FIT MC', 7,
np.where(base_dados['PRODUTO_TOP'] == 'CHEQUE ESPECIAL BANESPA', 8,
np.where(base_dados['PRODUTO_TOP'] == 'CONTA DA TURMA, INDEPENDENTE E UNIVERSIDADE BANESPA', 9,
np.where(base_dados['PRODUTO_TOP'] == 'CREDITO PESSOAL', 10,
np.where(base_dados['PRODUTO_TOP'] == 'CREDITO PESSOAL ELETRONICO', 11,
np.where(base_dados['PRODUTO_TOP'] == 'CREDITO PREVENTIVO', 12,
np.where(base_dados['PRODUTO_TOP'] == 'CREDITO RENOVADO BANESPA', 13,
np.where(base_dados['PRODUTO_TOP'] == 'CREDITO REORGANIZACAO', 14,
np.where(base_dados['PRODUTO_TOP'] == 'CREDITO SOLUCOES', 15,
np.where(base_dados['PRODUTO_TOP'] == 'EMPRESTIMO EM FOLHA', 16,
np.where(base_dados['PRODUTO_TOP'] == 'EMPRESTIMOS EM FOLHA - CARNE', 17,
np.where(base_dados['PRODUTO_TOP'] == 'MICROCREDITO', 18,
np.where(base_dados['PRODUTO_TOP'] == 'OUTROS', 19,
np.where(base_dados['PRODUTO_TOP'] == 'REFIN', 20,
np.where(base_dados['PRODUTO_TOP'] == 'SANTANDER BASICO MASTERCARD', 21,
np.where(base_dados['PRODUTO_TOP'] == 'SANTANDER ELITE PLATINUM MC', 22,
np.where(base_dados['PRODUTO_TOP'] == 'SANTANDER ELITE PLATINUM VISA', 23,
np.where(base_dados['PRODUTO_TOP'] == 'SANTANDER FLEX MASTERCARD', 24,
np.where(base_dados['PRODUTO_TOP'] == 'SANTANDER FLEX NACIONAL MASTERCARD', 25,
np.where(base_dados['PRODUTO_TOP'] == 'SANTANDER STYLE PLATINUM', 26,
np.where(base_dados['PRODUTO_TOP'] == 'SANTANDER STYLE PLATINUM MC', 27,
np.where(base_dados['PRODUTO_TOP'] == 'SANTANDER SX MASTER', 28,
np.where(base_dados['PRODUTO_TOP'] == 'SANTANDER SX VISA', 29,
0))))))))))))))))))))))))))))))
base_dados['PRODUTO_TOP_gh41'] = np.where(base_dados['PRODUTO_TOP_gh40'] == 0, -5,
np.where(base_dados['PRODUTO_TOP_gh40'] == 1, -5,
np.where(base_dados['PRODUTO_TOP_gh40'] == 2, -5,
np.where(base_dados['PRODUTO_TOP_gh40'] == 3, -5,
np.where(base_dados['PRODUTO_TOP_gh40'] == 4, 0,
np.where(base_dados['PRODUTO_TOP_gh40'] == 5, 2,
np.where(base_dados['PRODUTO_TOP_gh40'] == 6, -5,
np.where(base_dados['PRODUTO_TOP_gh40'] == 7, -5,
np.where(base_dados['PRODUTO_TOP_gh40'] == 8, 1,
np.where(base_dados['PRODUTO_TOP_gh40'] == 9, -5,
np.where(base_dados['PRODUTO_TOP_gh40'] == 10, -5,
np.where(base_dados['PRODUTO_TOP_gh40'] == 11, -5,
np.where(base_dados['PRODUTO_TOP_gh40'] == 12, -5,
np.where(base_dados['PRODUTO_TOP_gh40'] == 13, -5,
np.where(base_dados['PRODUTO_TOP_gh40'] == 14, -5,
np.where(base_dados['PRODUTO_TOP_gh40'] == 15, -5,
np.where(base_dados['PRODUTO_TOP_gh40'] == 16, -5,
np.where(base_dados['PRODUTO_TOP_gh40'] == 17, -5,
np.where(base_dados['PRODUTO_TOP_gh40'] == 18, -5,
np.where(base_dados['PRODUTO_TOP_gh40'] == 19, -5,
np.where(base_dados['PRODUTO_TOP_gh40'] == 20, -5,
np.where(base_dados['PRODUTO_TOP_gh40'] == 21, -5,
np.where(base_dados['PRODUTO_TOP_gh40'] == 22, -5,
np.where(base_dados['PRODUTO_TOP_gh40'] == 23, -5,
np.where(base_dados['PRODUTO_TOP_gh40'] == 24, -5,
np.where(base_dados['PRODUTO_TOP_gh40'] == 25, -5,
np.where(base_dados['PRODUTO_TOP_gh40'] == 26, -5,
np.where(base_dados['PRODUTO_TOP_gh40'] == 27, -5,
np.where(base_dados['PRODUTO_TOP_gh40'] == 28, -5,
np.where(base_dados['PRODUTO_TOP_gh40'] == 29, -5,
0))))))))))))))))))))))))))))))
base_dados['PRODUTO_TOP_gh42'] = np.where(base_dados['PRODUTO_TOP_gh41'] == -5, 0,
np.where(base_dados['PRODUTO_TOP_gh41'] == 0, 1,
np.where(base_dados['PRODUTO_TOP_gh41'] == 1, 2,
np.where(base_dados['PRODUTO_TOP_gh41'] == 2, 3,
0))))
base_dados['PRODUTO_TOP_gh43'] = np.where(base_dados['PRODUTO_TOP_gh42'] == 0, 0,
np.where(base_dados['PRODUTO_TOP_gh42'] == 1, 2,
np.where(base_dados['PRODUTO_TOP_gh42'] == 2, 1,
np.where(base_dados['PRODUTO_TOP_gh42'] == 3, 2,
0))))
base_dados['PRODUTO_TOP_gh44'] = np.where(base_dados['PRODUTO_TOP_gh43'] == 0, 0,
np.where(base_dados['PRODUTO_TOP_gh43'] == 1, 1,
np.where(base_dados['PRODUTO_TOP_gh43'] == 2, 2,
0)))
base_dados['PRODUTO_TOP_gh45'] = np.where(base_dados['PRODUTO_TOP_gh44'] == 0, 0,
np.where(base_dados['PRODUTO_TOP_gh44'] == 1, 1,
np.where(base_dados['PRODUTO_TOP_gh44'] == 2, 2,
0)))
base_dados['PRODUTO_TOP_gh46'] = np.where(base_dados['PRODUTO_TOP_gh45'] == 0, 0,
np.where(base_dados['PRODUTO_TOP_gh45'] == 1, 1,
np.where(base_dados['PRODUTO_TOP_gh45'] == 2, 2,
0)))
base_dados['PRODUTO_TOP_gh47'] = np.where(base_dados['PRODUTO_TOP_gh46'] == 0, 0,
np.where(base_dados['PRODUTO_TOP_gh46'] == 1, 1,
np.where(base_dados['PRODUTO_TOP_gh46'] == 2, 2,
0)))
base_dados['PRODUTO_TOP_gh48'] = np.where(base_dados['PRODUTO_TOP_gh47'] == 0, 0,
np.where(base_dados['PRODUTO_TOP_gh47'] == 1, 1,
np.where(base_dados['PRODUTO_TOP_gh47'] == 2, 2,
0)))
base_dados['PRODUTO_TOP_gh49'] = np.where(base_dados['PRODUTO_TOP_gh48'] == 0, 0,
np.where(base_dados['PRODUTO_TOP_gh48'] == 1, 1,
np.where(base_dados['PRODUTO_TOP_gh48'] == 2, 2,
0)))
base_dados['PRODUTO_TOP_gh50'] = np.where(base_dados['PRODUTO_TOP_gh49'] == 0, 0,
np.where(base_dados['PRODUTO_TOP_gh49'] == 1, 1,
np.where(base_dados['PRODUTO_TOP_gh49'] == 2, 2,
0)))
base_dados['PRODUTO_TOP_gh51'] = np.where(base_dados['PRODUTO_TOP_gh50'] == 0, 0,
np.where(base_dados['PRODUTO_TOP_gh50'] == 1, 1,
np.where(base_dados['PRODUTO_TOP_gh50'] == 2, 2,
0)))

         
         
         
         
         
         
base_dados['Qtde_Parcelas_Padrao_gh40'] = np.where(base_dados['Qtde_Parcelas_Padrao'] == 1, 0,
np.where(base_dados['Qtde_Parcelas_Padrao'] == 4, 1,
np.where(base_dados['Qtde_Parcelas_Padrao'] == 12, 2,
np.where(base_dados['Qtde_Parcelas_Padrao'] == 24, 3,
np.where(base_dados['Qtde_Parcelas_Padrao'] == 48, 4,
np.where(base_dados['Qtde_Parcelas_Padrao'] == 60, 5,
np.where(base_dados['Qtde_Parcelas_Padrao'] == 120, 6,
0)))))))
base_dados['Qtde_Parcelas_Padrao_gh41'] = np.where(base_dados['Qtde_Parcelas_Padrao_gh40'] == 0, 0,
np.where(base_dados['Qtde_Parcelas_Padrao_gh40'] == 1, -5,
np.where(base_dados['Qtde_Parcelas_Padrao_gh40'] == 2, -5,
np.where(base_dados['Qtde_Parcelas_Padrao_gh40'] == 3, -5,
np.where(base_dados['Qtde_Parcelas_Padrao_gh40'] == 4, -5,
np.where(base_dados['Qtde_Parcelas_Padrao_gh40'] == 5, -5,
np.where(base_dados['Qtde_Parcelas_Padrao_gh40'] == 6, -5,
0)))))))
base_dados['Qtde_Parcelas_Padrao_gh42'] = np.where(base_dados['Qtde_Parcelas_Padrao_gh41'] == -5, 0,
np.where(base_dados['Qtde_Parcelas_Padrao_gh41'] == 0, 1,
0))
base_dados['Qtde_Parcelas_Padrao_gh43'] = np.where(base_dados['Qtde_Parcelas_Padrao_gh42'] == 0, 0,
np.where(base_dados['Qtde_Parcelas_Padrao_gh42'] == 1, 1,
0))
base_dados['Qtde_Parcelas_Padrao_gh44'] = np.where(base_dados['Qtde_Parcelas_Padrao_gh43'] == 0, 0,
np.where(base_dados['Qtde_Parcelas_Padrao_gh43'] == 1, 1,
0))
base_dados['Qtde_Parcelas_Padrao_gh45'] = np.where(base_dados['Qtde_Parcelas_Padrao_gh44'] == 0, 0,
np.where(base_dados['Qtde_Parcelas_Padrao_gh44'] == 1, 1,
0))
base_dados['Qtde_Parcelas_Padrao_gh46'] = np.where(base_dados['Qtde_Parcelas_Padrao_gh45'] == 0, 0,
np.where(base_dados['Qtde_Parcelas_Padrao_gh45'] == 1, 1,
0))
base_dados['Qtde_Parcelas_Padrao_gh47'] = np.where(base_dados['Qtde_Parcelas_Padrao_gh46'] == 0, 0,
np.where(base_dados['Qtde_Parcelas_Padrao_gh46'] == 1, 1,
0))
base_dados['Qtde_Parcelas_Padrao_gh48'] = np.where(base_dados['Qtde_Parcelas_Padrao_gh47'] == 0, 0,
np.where(base_dados['Qtde_Parcelas_Padrao_gh47'] == 1, 1,
0))
base_dados['Qtde_Parcelas_Padrao_gh49'] = np.where(base_dados['Qtde_Parcelas_Padrao_gh48'] == 0, 0,
np.where(base_dados['Qtde_Parcelas_Padrao_gh48'] == 1, 1,
0))
base_dados['Qtde_Parcelas_Padrao_gh50'] = np.where(base_dados['Qtde_Parcelas_Padrao_gh49'] == 0, 0,
np.where(base_dados['Qtde_Parcelas_Padrao_gh49'] == 1, 1,
0))
base_dados['Qtde_Parcelas_Padrao_gh51'] = np.where(base_dados['Qtde_Parcelas_Padrao_gh50'] == 0, 0,
np.where(base_dados['Qtde_Parcelas_Padrao_gh50'] == 1, 1,
0))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Gerando variáveis numéricas contínuas

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parte 1 de 2

# COMMAND ----------

base_dados['MOB_ENTRADA__p_4'] = np.where(base_dados['MOB_ENTRADA'] <= 5.626382089093015, 0.0,
np.where(np.bitwise_and(base_dados['MOB_ENTRADA'] > 5.626382089093015, base_dados['MOB_ENTRADA'] <= 9.733242599575776), 1.0,
np.where(np.bitwise_and(base_dados['MOB_ENTRADA'] > 9.733242599575776, base_dados['MOB_ENTRADA'] <= 17.125591518444747), 2.0,
np.where(base_dados['MOB_ENTRADA'] > 17.125591518444747, 3.0,
 0))))
base_dados['MOB_ENTRADA__p_4_g_1_1'] = np.where(base_dados['MOB_ENTRADA__p_4'] == 0.0, 1,
np.where(base_dados['MOB_ENTRADA__p_4'] == 1.0, 1,
np.where(base_dados['MOB_ENTRADA__p_4'] == 2.0, 1,
np.where(base_dados['MOB_ENTRADA__p_4'] == 3.0, 0,
 0))))
base_dados['MOB_ENTRADA__p_4_g_1_2'] = np.where(base_dados['MOB_ENTRADA__p_4_g_1_1'] == 0, 0,
np.where(base_dados['MOB_ENTRADA__p_4_g_1_1'] == 1, 1,
 0))
base_dados['MOB_ENTRADA__p_4_g_1'] = np.where(base_dados['MOB_ENTRADA__p_4_g_1_2'] == 0, 0,
np.where(base_dados['MOB_ENTRADA__p_4_g_1_2'] == 1, 1,
 0))
                                              
                                              
                                              
                                              
                                              
                                              
                                              
base_dados['MOB_ENTRADA__S'] = np.sin(base_dados['MOB_ENTRADA'])
np.where(base_dados['MOB_ENTRADA__S'] == 0, -1, base_dados['MOB_ENTRADA__S'])
base_dados['MOB_ENTRADA__S'] = base_dados['MOB_ENTRADA__S'].fillna(-2)
base_dados['MOB_ENTRADA__S__p_7'] = np.where(base_dados['MOB_ENTRADA__S'] <= -0.8756918378462696, 0.0,
np.where(np.bitwise_and(base_dados['MOB_ENTRADA__S'] > -0.8756918378462696, base_dados['MOB_ENTRADA__S'] <= -0.5615485894205193), 1.0,
np.where(np.bitwise_and(base_dados['MOB_ENTRADA__S'] > -0.5615485894205193, base_dados['MOB_ENTRADA__S'] <= -0.2960572421239163), 2.0,
np.where(np.bitwise_and(base_dados['MOB_ENTRADA__S'] > -0.2960572421239163, base_dados['MOB_ENTRADA__S'] <= 0.2309153960342717), 3.0,
np.where(np.bitwise_and(base_dados['MOB_ENTRADA__S'] > 0.2309153960342717, base_dados['MOB_ENTRADA__S'] <= 0.7114443840309153), 4.0,
np.where(np.bitwise_and(base_dados['MOB_ENTRADA__S'] > 0.7114443840309153, base_dados['MOB_ENTRADA__S'] <= 0.7754177078626303), 5.0,
np.where(base_dados['MOB_ENTRADA__S'] > 0.7754177078626303, 6.0,
 0)))))))
base_dados['MOB_ENTRADA__S__p_7_g_1_1'] = np.where(base_dados['MOB_ENTRADA__S__p_7'] == 0.0, 0,
np.where(base_dados['MOB_ENTRADA__S__p_7'] == 1.0, 0,
np.where(base_dados['MOB_ENTRADA__S__p_7'] == 2.0, 0,
np.where(base_dados['MOB_ENTRADA__S__p_7'] == 3.0, 0,
np.where(base_dados['MOB_ENTRADA__S__p_7'] == 4.0, 1,
np.where(base_dados['MOB_ENTRADA__S__p_7'] == 5.0, 3,
np.where(base_dados['MOB_ENTRADA__S__p_7'] == 6.0, 2,
 0)))))))
base_dados['MOB_ENTRADA__S__p_7_g_1_2'] = np.where(base_dados['MOB_ENTRADA__S__p_7_g_1_1'] == 0, 1,
np.where(base_dados['MOB_ENTRADA__S__p_7_g_1_1'] == 1, 3,
np.where(base_dados['MOB_ENTRADA__S__p_7_g_1_1'] == 2, 1,
np.where(base_dados['MOB_ENTRADA__S__p_7_g_1_1'] == 3, 0,
 0))))
base_dados['MOB_ENTRADA__S__p_7_g_1'] = np.where(base_dados['MOB_ENTRADA__S__p_7_g_1_2'] == 0, 0,
np.where(base_dados['MOB_ENTRADA__S__p_7_g_1_2'] == 1, 1,
np.where(base_dados['MOB_ENTRADA__S__p_7_g_1_2'] == 3, 2,
 0)))
         
         
         
         
         
         
         
         
base_dados['Contrato_Altair__T'] = np.tan(base_dados['Contrato_Altair'])
np.where(base_dados['Contrato_Altair__T'] == 0, -1, base_dados['Contrato_Altair__T'])
base_dados['Contrato_Altair__T'] = base_dados['Contrato_Altair__T'].fillna(-2)
base_dados['Contrato_Altair__T__p_10'] = np.where(base_dados['Contrato_Altair__T'] <= 0.267021851171049, 0.0,
np.where(np.bitwise_and(base_dados['Contrato_Altair__T'] > 0.267021851171049, base_dados['Contrato_Altair__T'] <= 1.2547421643537249), 1.0,
np.where(np.bitwise_and(base_dados['Contrato_Altair__T'] > 1.2547421643537249, base_dados['Contrato_Altair__T'] <= 1.3931645473056549), 2.0,
np.where(base_dados['Contrato_Altair__T'] > 1.3931645473056549, 3.0,
 0))))
base_dados['Contrato_Altair__T__p_10_g_1_1'] = np.where(base_dados['Contrato_Altair__T__p_10'] == 0.0, 1,
np.where(base_dados['Contrato_Altair__T__p_10'] == 1.0, 1,
np.where(base_dados['Contrato_Altair__T__p_10'] == 2.0, 0,
np.where(base_dados['Contrato_Altair__T__p_10'] == 3.0, 1,
 0))))
base_dados['Contrato_Altair__T__p_10_g_1_2'] = np.where(base_dados['Contrato_Altair__T__p_10_g_1_1'] == 0, 1,
np.where(base_dados['Contrato_Altair__T__p_10_g_1_1'] == 1, 0,
 0))
base_dados['Contrato_Altair__T__p_10_g_1'] = np.where(base_dados['Contrato_Altair__T__p_10_g_1_2'] == 0, 0,
np.where(base_dados['Contrato_Altair__T__p_10_g_1_2'] == 1, 1,
 0))
                                                      
                                                      
                                                      
                                                      
                                                      
                                                      
base_dados['Contrato_Altair__T'] = np.tan(base_dados['Contrato_Altair'])
np.where(base_dados['Contrato_Altair__T'] == 0, -1, base_dados['Contrato_Altair__T'])
base_dados['Contrato_Altair__T'] = base_dados['Contrato_Altair__T'].fillna(-2)
base_dados['Contrato_Altair__T__pe_3'] = np.where(np.bitwise_and(base_dados['Contrato_Altair__T'] >= -1447.8353666065166, base_dados['Contrato_Altair__T'] <= 2.755483045647408), 0.0,
np.where(np.bitwise_and(base_dados['Contrato_Altair__T'] > 2.755483045647408, base_dados['Contrato_Altair__T'] <= 5.638258966281598), 1.0,
np.where(base_dados['Contrato_Altair__T'] > 5.638258966281598, 2.0,
 -2)))
base_dados['Contrato_Altair__T__pe_3_g_1_1'] = np.where(base_dados['Contrato_Altair__T__pe_3'] == -2.0, 1,
np.where(base_dados['Contrato_Altair__T__pe_3'] == 0.0, 0,
np.where(base_dados['Contrato_Altair__T__pe_3'] == 1.0, 1,
np.where(base_dados['Contrato_Altair__T__pe_3'] == 2.0, 1,
 0))))
base_dados['Contrato_Altair__T__pe_3_g_1_2'] = np.where(base_dados['Contrato_Altair__T__pe_3_g_1_1'] == 0, 1,
np.where(base_dados['Contrato_Altair__T__pe_3_g_1_1'] == 1, 0,
 0))
base_dados['Contrato_Altair__T__pe_3_g_1'] = np.where(base_dados['Contrato_Altair__T__pe_3_g_1_2'] == 0, 0,
np.where(base_dados['Contrato_Altair__T__pe_3_g_1_2'] == 1, 1,
 0))
         
         
         
         
         
         
         
base_dados['Cep1__pe_6'] = np.where(np.bitwise_and(base_dados['Cep1'] >= -3.0, base_dados['Cep1'] <= 15870.0), 0.0,
np.where(np.bitwise_and(base_dados['Cep1'] > 15870.0, base_dados['Cep1'] <= 31749.0), 1.0,
np.where(np.bitwise_and(base_dados['Cep1'] > 31749.0, base_dados['Cep1'] <= 47600.0), 2.0,
np.where(np.bitwise_and(base_dados['Cep1'] > 47600.0, base_dados['Cep1'] <= 63515.0), 3.0,
np.where(np.bitwise_and(base_dados['Cep1'] > 63515.0, base_dados['Cep1'] <= 79390.0), 4.0,
np.where(base_dados['Cep1'] > 79390.0, 5.0,
 -2))))))
base_dados['Cep1__pe_6_g_1_1'] = np.where(base_dados['Cep1__pe_6'] == -2.0, 4,
np.where(base_dados['Cep1__pe_6'] == 0.0, 0,
np.where(base_dados['Cep1__pe_6'] == 1.0, 2,
np.where(base_dados['Cep1__pe_6'] == 2.0, 1,
np.where(base_dados['Cep1__pe_6'] == 3.0, 5,
np.where(base_dados['Cep1__pe_6'] == 4.0, 3,
np.where(base_dados['Cep1__pe_6'] == 5.0, 4,
 0)))))))
base_dados['Cep1__pe_6_g_1_2'] = np.where(base_dados['Cep1__pe_6_g_1_1'] == 0, 5,
np.where(base_dados['Cep1__pe_6_g_1_1'] == 1, 1,
np.where(base_dados['Cep1__pe_6_g_1_1'] == 2, 4,
np.where(base_dados['Cep1__pe_6_g_1_1'] == 3, 1,
np.where(base_dados['Cep1__pe_6_g_1_1'] == 4, 0,
np.where(base_dados['Cep1__pe_6_g_1_1'] == 5, 1,
 0))))))
base_dados['Cep1__pe_6_g_1'] = np.where(base_dados['Cep1__pe_6_g_1_2'] == 0, 0,
np.where(base_dados['Cep1__pe_6_g_1_2'] == 1, 1,
np.where(base_dados['Cep1__pe_6_g_1_2'] == 4, 2,
np.where(base_dados['Cep1__pe_6_g_1_2'] == 5, 3,
 0))))
         
         
         
         
         
         
         
base_dados['Cep1__L'] = np.log(base_dados['Cep1'])
np.where(base_dados['Cep1__L'] == 0, -1, base_dados['Cep1__L'])
base_dados['Cep1__L'] = base_dados['Cep1__L'].fillna(-2)
base_dados['Cep1__L__p_7'] = np.where(base_dados['Cep1__L'] <= 9.515543058145953, 0.0,
np.where(np.bitwise_and(base_dados['Cep1__L'] > 9.515543058145953, base_dados['Cep1__L'] <= 10.163541457870435), 1.0,
np.where(np.bitwise_and(base_dados['Cep1__L'] > 10.163541457870435, base_dados['Cep1__L'] <= 10.62570798058178), 2.0,
np.where(np.bitwise_and(base_dados['Cep1__L'] > 10.62570798058178, base_dados['Cep1__L'] <= 11.007684219498138), 3.0,
np.where(np.bitwise_and(base_dados['Cep1__L'] > 11.007684219498138, base_dados['Cep1__L'] <= 11.202452304217632), 4.0,
np.where(np.bitwise_and(base_dados['Cep1__L'] > 11.202452304217632, base_dados['Cep1__L'] <= 11.381802598088962), 5.0,
np.where(base_dados['Cep1__L'] > 11.381802598088962, 6.0,
 0)))))))
base_dados['Cep1__L__p_7_g_1_1'] = np.where(base_dados['Cep1__L__p_7'] == 0.0, 0,
np.where(base_dados['Cep1__L__p_7'] == 1.0, 1,
np.where(base_dados['Cep1__L__p_7'] == 2.0, 1,
np.where(base_dados['Cep1__L__p_7'] == 3.0, 1,
np.where(base_dados['Cep1__L__p_7'] == 4.0, 2,
np.where(base_dados['Cep1__L__p_7'] == 5.0, 1,
np.where(base_dados['Cep1__L__p_7'] == 6.0, 2,
 0)))))))
base_dados['Cep1__L__p_7_g_1_2'] = np.where(base_dados['Cep1__L__p_7_g_1_1'] == 0, 2,
np.where(base_dados['Cep1__L__p_7_g_1_1'] == 1, 1,
np.where(base_dados['Cep1__L__p_7_g_1_1'] == 2, 0,
 0)))
base_dados['Cep1__L__p_7_g_1'] = np.where(base_dados['Cep1__L__p_7_g_1_2'] == 0, 0,
np.where(base_dados['Cep1__L__p_7_g_1_2'] == 1, 1,
np.where(base_dados['Cep1__L__p_7_g_1_2'] == 2, 2,
 0)))


# COMMAND ----------

# MAGIC %md
# MAGIC ### Parte 2 de 2

# COMMAND ----------

base_dados['MOB_ENTRADA__S__p_7_g_1_c1_30_1'] = np.where(np.bitwise_and(base_dados['MOB_ENTRADA__p_4_g_1'] == 0, base_dados['MOB_ENTRADA__S__p_7_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['MOB_ENTRADA__p_4_g_1'] == 0, base_dados['MOB_ENTRADA__S__p_7_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['MOB_ENTRADA__p_4_g_1'] == 0, base_dados['MOB_ENTRADA__S__p_7_g_1'] == 2), 2,
np.where(np.bitwise_and(base_dados['MOB_ENTRADA__p_4_g_1'] == 1, base_dados['MOB_ENTRADA__S__p_7_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['MOB_ENTRADA__p_4_g_1'] == 1, base_dados['MOB_ENTRADA__S__p_7_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['MOB_ENTRADA__p_4_g_1'] == 1, base_dados['MOB_ENTRADA__S__p_7_g_1'] == 2), 3,
 0))))))
base_dados['MOB_ENTRADA__S__p_7_g_1_c1_30_2'] = np.where(base_dados['MOB_ENTRADA__S__p_7_g_1_c1_30_1'] == 0, 0,
np.where(base_dados['MOB_ENTRADA__S__p_7_g_1_c1_30_1'] == 1, 1,
np.where(base_dados['MOB_ENTRADA__S__p_7_g_1_c1_30_1'] == 2, 2,
np.where(base_dados['MOB_ENTRADA__S__p_7_g_1_c1_30_1'] == 3, 3,
0))))
base_dados['MOB_ENTRADA__S__p_7_g_1_c1_30'] = np.where(base_dados['MOB_ENTRADA__S__p_7_g_1_c1_30_2'] == 0, 0,
np.where(base_dados['MOB_ENTRADA__S__p_7_g_1_c1_30_2'] == 1, 1,
np.where(base_dados['MOB_ENTRADA__S__p_7_g_1_c1_30_2'] == 2, 2,
np.where(base_dados['MOB_ENTRADA__S__p_7_g_1_c1_30_2'] == 3, 3,
 0))))
         
         
         
         
         
base_dados['Contrato_Altair__T__p_10_g_1_c1_26_1'] = np.where(np.bitwise_and(base_dados['Contrato_Altair__T__p_10_g_1'] == 0, base_dados['Contrato_Altair__T__pe_3_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['Contrato_Altair__T__p_10_g_1'] == 0, base_dados['Contrato_Altair__T__pe_3_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['Contrato_Altair__T__p_10_g_1'] == 1, base_dados['Contrato_Altair__T__pe_3_g_1'] == 0), 2,
np.where(np.bitwise_and(base_dados['Contrato_Altair__T__p_10_g_1'] == 1, base_dados['Contrato_Altair__T__pe_3_g_1'] == 1), 2,
 0))))
base_dados['Contrato_Altair__T__p_10_g_1_c1_26_2'] = np.where(base_dados['Contrato_Altair__T__p_10_g_1_c1_26_1'] == 0, 0,
np.where(base_dados['Contrato_Altair__T__p_10_g_1_c1_26_1'] == 1, 1,
np.where(base_dados['Contrato_Altair__T__p_10_g_1_c1_26_1'] == 2, 2,
0)))
base_dados['Contrato_Altair__T__p_10_g_1_c1_26'] = np.where(base_dados['Contrato_Altair__T__p_10_g_1_c1_26_2'] == 0, 0,
np.where(base_dados['Contrato_Altair__T__p_10_g_1_c1_26_2'] == 1, 1,
np.where(base_dados['Contrato_Altair__T__p_10_g_1_c1_26_2'] == 2, 2,
 0)))
         
         
         
         
         
         
         
base_dados['Cep1__L__p_7_g_1_c1_21_1'] = np.where(np.bitwise_and(base_dados['Cep1__pe_6_g_1'] == 0, base_dados['Cep1__L__p_7_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['Cep1__pe_6_g_1'] == 0, base_dados['Cep1__L__p_7_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['Cep1__pe_6_g_1'] == 0, base_dados['Cep1__L__p_7_g_1'] == 2), 3,
np.where(np.bitwise_and(base_dados['Cep1__pe_6_g_1'] == 1, base_dados['Cep1__L__p_7_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['Cep1__pe_6_g_1'] == 1, base_dados['Cep1__L__p_7_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['Cep1__pe_6_g_1'] == 1, base_dados['Cep1__L__p_7_g_1'] == 2), 2,
np.where(np.bitwise_and(base_dados['Cep1__pe_6_g_1'] == 2, base_dados['Cep1__L__p_7_g_1'] == 0), 2,
np.where(np.bitwise_and(base_dados['Cep1__pe_6_g_1'] == 2, base_dados['Cep1__L__p_7_g_1'] == 1), 3,
np.where(np.bitwise_and(base_dados['Cep1__pe_6_g_1'] == 2, base_dados['Cep1__L__p_7_g_1'] == 2), 3,
np.where(np.bitwise_and(base_dados['Cep1__pe_6_g_1'] == 3, base_dados['Cep1__L__p_7_g_1'] == 0), 3,
np.where(np.bitwise_and(base_dados['Cep1__pe_6_g_1'] == 3, base_dados['Cep1__L__p_7_g_1'] == 1), 3,
np.where(np.bitwise_and(base_dados['Cep1__pe_6_g_1'] == 3, base_dados['Cep1__L__p_7_g_1'] == 2), 4,
 0))))))))))))
base_dados['Cep1__L__p_7_g_1_c1_21_2'] = np.where(base_dados['Cep1__L__p_7_g_1_c1_21_1'] == 0, 0,
np.where(base_dados['Cep1__L__p_7_g_1_c1_21_1'] == 1, 1,
np.where(base_dados['Cep1__L__p_7_g_1_c1_21_1'] == 2, 2,
np.where(base_dados['Cep1__L__p_7_g_1_c1_21_1'] == 3, 3,
np.where(base_dados['Cep1__L__p_7_g_1_c1_21_1'] == 4, 4,
0)))))
base_dados['Cep1__L__p_7_g_1_c1_21'] = np.where(base_dados['Cep1__L__p_7_g_1_c1_21_2'] == 0, 0,
np.where(base_dados['Cep1__L__p_7_g_1_c1_21_2'] == 1, 1,
np.where(base_dados['Cep1__L__p_7_g_1_c1_21_2'] == 2, 2,
np.where(base_dados['Cep1__L__p_7_g_1_c1_21_2'] == 3, 3,
np.where(base_dados['Cep1__L__p_7_g_1_c1_21_2'] == 4, 4,
 0)))))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Mantendo apenas as variáveis do modelo

# COMMAND ----------

import pickle
modelo=pickle.load(open(caminho_pickle_model, 'rb'))

base_teste_c0 = base_dados[[chave,'Telefone1_gh38','Telefone2_gh38','Telefone3_gh38','rank_gh38','Desconto_Padrao_gh38','Indicador_Correntista_gh38','genero_gh51','PRODUTO_TOP_gh51','Qtde_Parcelas_Padrao_gh51','Cep1__L__p_7_g_1_c1_21','Contrato_Altair__T__p_10_g_1_c1_26','MOB_ENTRADA__S__p_7_g_1_c1_30']]

var_fin_c0=['Telefone1_gh38','Telefone2_gh38','Telefone3_gh38','rank_gh38','Desconto_Padrao_gh38','Indicador_Correntista_gh38','genero_gh51','PRODUTO_TOP_gh51','Qtde_Parcelas_Padrao_gh51','Cep1__L__p_7_g_1_c1_21','Contrato_Altair__T__p_10_g_1_c1_26','MOB_ENTRADA__S__p_7_g_1_c1_30']

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

#Create PySpark DataFrame from Pandas
sparkDF=spark.createDataFrame(x_teste2) 

sparkDF.write.option("sep",";").option("header","true").mode('overwrite').csv(sink)

# COMMAND ----------

# MAGIC %md
# MAGIC # Modelo de Grupo Homogêneo

# COMMAND ----------


x_teste2['P_1_R'] = np.sqrt(x_teste2['P_1']) 
x_teste2['P_1_R'] = np.where(x_teste2['P_1'] == 0, -1, x_teste2['P_1_R'])        
x_teste2['P_1_R'] = x_teste2['P_1_R'].fillna(-2)
        
x_teste2['P_1_p_40_g_1'] = np.where(x_teste2['P_1'] <= 0.075068263, 0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.075068263, x_teste2['P_1'] <= 0.193315398), 1,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.193315398, x_teste2['P_1'] <= 0.36086929), 2,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.36086929, x_teste2['P_1'] <= 0.585671237), 3,4))))

x_teste2['P_1_R_p_17_g_1'] = np.where(x_teste2['P_1_R'] <= 0.175297944, 0,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.175297944, x_teste2['P_1_R'] <= 0.270192547), 1,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.270192547, x_teste2['P_1_R'] <= 0.426593441), 2,3)))

x_teste2['GH'] = np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 0, x_teste2['P_1_R_p_17_g_1'] == 0), 0,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 0, x_teste2['P_1_R_p_17_g_1'] == 1), 1,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 0, x_teste2['P_1_R_p_17_g_1'] == 2), 2,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 0, x_teste2['P_1_R_p_17_g_1'] == 3), 2,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 1, x_teste2['P_1_R_p_17_g_1'] == 0), 1,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 1, x_teste2['P_1_R_p_17_g_1'] == 1), 2,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 1, x_teste2['P_1_R_p_17_g_1'] == 2), 2,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 1, x_teste2['P_1_R_p_17_g_1'] == 3), 2,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 2, x_teste2['P_1_R_p_17_g_1'] == 0), 2,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 2, x_teste2['P_1_R_p_17_g_1'] == 1), 3,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 2, x_teste2['P_1_R_p_17_g_1'] == 2), 3,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 2, x_teste2['P_1_R_p_17_g_1'] == 3), 3,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 3, x_teste2['P_1_R_p_17_g_1'] == 0), 3,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 3, x_teste2['P_1_R_p_17_g_1'] == 1), 4,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 3, x_teste2['P_1_R_p_17_g_1'] == 2), 4,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 3, x_teste2['P_1_R_p_17_g_1'] == 3), 4,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 4, x_teste2['P_1_R_p_17_g_1'] == 0), 4,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 4, x_teste2['P_1_R_p_17_g_1'] == 1), 5,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 4, x_teste2['P_1_R_p_17_g_1'] == 2), 5,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 4, x_teste2['P_1_R_p_17_g_1'] == 3), 5,
             2))))))))))))))))))))

del x_teste2['P_1_R']
del x_teste2['P_1_p_40_g_1']
del x_teste2['P_1_R_p_17_g_1']

x_teste2


# COMMAND ----------

x_teste2.groupby(['GH'])['P_1'].count()

# COMMAND ----------

sparkDF=spark.createDataFrame(x_teste2) 

sparkDF.write.mode('overwrite').option("sep",";").option("header","true").option("mode","overwrite").csv(sinkHomomgeneous)

# COMMAND ----------

dbutils.notebook.exit('SUCCESS')