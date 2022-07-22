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
base_dados = base_dados[[chave,'Cep1', 'Cep2', 'Qtde_Parcelas_Padrao', 'genero', 'DATA_ENTRADA_DIVIDA','Desconto_Padrao', 'Contrato_Altair', 'Indicador_Correntista', 'Descricao_Produto', 'Codigo_Politica','Saldo_Devedor_Contrato', 'Forma_Pgto_Padrao','Telefone1_skip:hot', 'Telefone1_skip:alto', 'Telefone1_skip:medio','Telefone1_skip:baixo', 'Telefone1_skip:nhot','Telefone1_sem_tags', 'Telefone2_skip:hot', 'Telefone2_skip:alto', 'Telefone2_skip:medio', 'Telefone2_skip:baixo','Telefone2_skip:nhot', 'Telefone2_sem_tags', 'Telefone3_skip:hot', 'Telefone3_skip:alto', 'Telefone3_skip:medio','Telefone3_skip:baixo', 'Telefone3_skip:nhot', 'Telefone3_sem_tags','rank:a','rank:b','rank:c','rank:d','rank:e','sem_skip']]

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
base_dados['Cep2'] = base_dados['Cep2'].replace(np.nan, '')

base_dados['Cep1'] = base_dados['Cep1'].str[:5]
base_dados['Cep2'] = base_dados['Cep2'].str[:5]

base_dados['DATA_ENTRADA_DIVIDA'] = pd.to_datetime(base_dados['DATA_ENTRADA_DIVIDA'])
base_dados['MOB_ENTRADA'] = ((datetime.today()) - base_dados.DATA_ENTRADA_DIVIDA)/np.timedelta64(1, 'M')

del base_dados['DATA_ENTRADA_DIVIDA']

base_dados = base_dados.apply(pd.to_numeric, errors='ignore')

base_dados['Cep1'] = base_dados['Cep1'].replace(np.nan, -3)
base_dados['Cep2'] = base_dados['Cep2'].replace(np.nan, -3)

base_dados.fillna(-3)

print("shape da Base de Dados:",base_dados.shape)

base_dados.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gerando variáveis Categóricas

# COMMAND ----------

base_dados['genero_gh30'] = np.where(base_dados['genero'] == '-3', 0,
np.where(base_dados['genero'] == 'F', 1,
np.where(base_dados['genero'] == 'M', 2,
0)))
base_dados['genero_gh31'] = np.where(base_dados['genero_gh30'] == 0, 0,
np.where(base_dados['genero_gh30'] == 1, 0,
np.where(base_dados['genero_gh30'] == 2, 2,
0)))
base_dados['genero_gh32'] = np.where(base_dados['genero_gh31'] == 0, 0,
np.where(base_dados['genero_gh31'] == 2, 1,
0))
base_dados['genero_gh33'] = np.where(base_dados['genero_gh32'] == 0, 0,
np.where(base_dados['genero_gh32'] == 1, 1,
0))
base_dados['genero_gh34'] = np.where(base_dados['genero_gh33'] == 0, 0,
np.where(base_dados['genero_gh33'] == 1, 1,
0))
base_dados['genero_gh35'] = np.where(base_dados['genero_gh34'] == 0, 0,
np.where(base_dados['genero_gh34'] == 1, 1,
0))
base_dados['genero_gh36'] = np.where(base_dados['genero_gh35'] == 0, 1,
np.where(base_dados['genero_gh35'] == 1, 0,
0))
base_dados['genero_gh37'] = np.where(base_dados['genero_gh36'] == 0, 0,
np.where(base_dados['genero_gh36'] == 1, 1,
0))
base_dados['genero_gh38'] = np.where(base_dados['genero_gh37'] == 0, 0,
np.where(base_dados['genero_gh37'] == 1, 1,
0))
                                     
                                     
                                     
                                     
                                     
                                     
                                     
base_dados['Forma_Pgto_Padrao_gh30'] = np.where(base_dados['Forma_Pgto_Padrao'] == 'B', 0,
np.where(base_dados['Forma_Pgto_Padrao'] == 'D', 1,
0))
base_dados['Forma_Pgto_Padrao_gh31'] = np.where(base_dados['Forma_Pgto_Padrao_gh30'] == 0, 0,
np.where(base_dados['Forma_Pgto_Padrao_gh30'] == 1, 1,
0))
base_dados['Forma_Pgto_Padrao_gh32'] = np.where(base_dados['Forma_Pgto_Padrao_gh31'] == 0, 0,
np.where(base_dados['Forma_Pgto_Padrao_gh31'] == 1, 1,
0))
base_dados['Forma_Pgto_Padrao_gh33'] = np.where(base_dados['Forma_Pgto_Padrao_gh32'] == 0, 0,
np.where(base_dados['Forma_Pgto_Padrao_gh32'] == 1, 1,
0))
base_dados['Forma_Pgto_Padrao_gh34'] = np.where(base_dados['Forma_Pgto_Padrao_gh33'] == 0, 0,
np.where(base_dados['Forma_Pgto_Padrao_gh33'] == 1, 1,
0))
base_dados['Forma_Pgto_Padrao_gh35'] = np.where(base_dados['Forma_Pgto_Padrao_gh34'] == 0, 0,
np.where(base_dados['Forma_Pgto_Padrao_gh34'] == 1, 1,
0))
base_dados['Forma_Pgto_Padrao_gh36'] = np.where(base_dados['Forma_Pgto_Padrao_gh35'] == 0, 1,
np.where(base_dados['Forma_Pgto_Padrao_gh35'] == 1, 0,
0))
base_dados['Forma_Pgto_Padrao_gh37'] = np.where(base_dados['Forma_Pgto_Padrao_gh36'] == 0, 0,
np.where(base_dados['Forma_Pgto_Padrao_gh36'] == 1, 1,
0))
base_dados['Forma_Pgto_Padrao_gh38'] = np.where(base_dados['Forma_Pgto_Padrao_gh37'] == 0, 0,
np.where(base_dados['Forma_Pgto_Padrao_gh37'] == 1, 1,
0))
         
         
         
         
         
         
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
np.where(base_dados['Telefone2_gh30'] == 1, 1,
np.where(base_dados['Telefone2_gh30'] == 2, 1,
np.where(base_dados['Telefone2_gh30'] == 3, 3,
np.where(base_dados['Telefone2_gh30'] == 4, 4,
np.where(base_dados['Telefone2_gh30'] == 5, 5,
0))))))
base_dados['Telefone2_gh32'] = np.where(base_dados['Telefone2_gh31'] == 0, 0,
np.where(base_dados['Telefone2_gh31'] == 1, 1,
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
np.where(base_dados['Telefone3_gh33'] == 1, 1,
np.where(base_dados['Telefone3_gh33'] == 2, 2,
np.where(base_dados['Telefone3_gh33'] == 3, 4,
np.where(base_dados['Telefone3_gh33'] == 4, 4,
np.where(base_dados['Telefone3_gh33'] == 5, 1,
0))))))
base_dados['Telefone3_gh35'] = np.where(base_dados['Telefone3_gh34'] == 0, 0,
np.where(base_dados['Telefone3_gh34'] == 1, 1,
np.where(base_dados['Telefone3_gh34'] == 2, 2,
np.where(base_dados['Telefone3_gh34'] == 4, 3,
0))))
base_dados['Telefone3_gh36'] = np.where(base_dados['Telefone3_gh35'] == 0, 1,
np.where(base_dados['Telefone3_gh35'] == 1, 0,
np.where(base_dados['Telefone3_gh35'] == 2, 1,
np.where(base_dados['Telefone3_gh35'] == 3, 3,
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
base_dados['rank_gh37'] = np.where(base_dados['rank_gh36'] == 0, 0,
np.where(base_dados['rank_gh36'] == 1, 1,
np.where(base_dados['rank_gh36'] == 2, 2,
np.where(base_dados['rank_gh36'] == 3, 3,
np.where(base_dados['rank_gh36'] == 4, 4,
0)))))
base_dados['rank_gh38'] = np.where(base_dados['rank_gh37'] == 0, 0,
np.where(base_dados['rank_gh37'] == 1, 1,
np.where(base_dados['rank_gh37'] == 2, 2,
np.where(base_dados['rank_gh37'] == 3, 3,
np.where(base_dados['rank_gh37'] == 4, 4,
0)))))
                                                
                                                
                                                
                                                
                                                
                                                
base_dados['Desconto_Padrao_gh30'] = np.where(base_dados['Desconto_Padrao'] == 0, 0,
np.where(base_dados['Desconto_Padrao'] == 10, 1,
np.where(base_dados['Desconto_Padrao'] == 25, 2,
np.where(base_dados['Desconto_Padrao'] == 27, 3,
np.where(base_dados['Desconto_Padrao'] == 28, 4,
np.where(base_dados['Desconto_Padrao'] == 30, 5,
np.where(base_dados['Desconto_Padrao'] == 40, 6,
np.where(base_dados['Desconto_Padrao'] == 45, 7,
np.where(base_dados['Desconto_Padrao'] == 50, 8,
np.where(base_dados['Desconto_Padrao'] == 54, 9,
np.where(base_dados['Desconto_Padrao'] == 55, 10,
np.where(base_dados['Desconto_Padrao'] == 65, 11,
np.where(base_dados['Desconto_Padrao'] == 70, 12,
np.where(base_dados['Desconto_Padrao'] == 75, 13,
np.where(base_dados['Desconto_Padrao'] == 76, 14,
np.where(base_dados['Desconto_Padrao'] == 77, 15,
np.where(base_dados['Desconto_Padrao'] == 80, 16,
np.where(base_dados['Desconto_Padrao'] == 82, 17,
np.where(base_dados['Desconto_Padrao'] == 83, 18,
np.where(base_dados['Desconto_Padrao'] == 85, 19,
np.where(base_dados['Desconto_Padrao'] == 90, 20,
np.where(base_dados['Desconto_Padrao'] == 93, 21,
0))))))))))))))))))))))

base_dados['Desconto_Padrao_gh31'] = np.where(base_dados['Desconto_Padrao_gh30'] == 0, 0,
np.where(base_dados['Desconto_Padrao_gh30'] == 1, 1,
np.where(base_dados['Desconto_Padrao_gh30'] == 2, 2,
np.where(base_dados['Desconto_Padrao_gh30'] == 3, 3,
np.where(base_dados['Desconto_Padrao_gh30'] == 4, 4,
np.where(base_dados['Desconto_Padrao_gh30'] == 5, 5,
np.where(base_dados['Desconto_Padrao_gh30'] == 6, 5,
np.where(base_dados['Desconto_Padrao_gh30'] == 7, 5,
np.where(base_dados['Desconto_Padrao_gh30'] == 8, 8,
np.where(base_dados['Desconto_Padrao_gh30'] == 9, 9,
np.where(base_dados['Desconto_Padrao_gh30'] == 10, 10,
np.where(base_dados['Desconto_Padrao_gh30'] == 11, 11,
np.where(base_dados['Desconto_Padrao_gh30'] == 12, 11,
np.where(base_dados['Desconto_Padrao_gh30'] == 13, 13,
np.where(base_dados['Desconto_Padrao_gh30'] == 14, 14,
np.where(base_dados['Desconto_Padrao_gh30'] == 15, 15,
np.where(base_dados['Desconto_Padrao_gh30'] == 16, 16,
np.where(base_dados['Desconto_Padrao_gh30'] == 17, 17,
np.where(base_dados['Desconto_Padrao_gh30'] == 18, 17,
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
np.where(base_dados['Desconto_Padrao_gh31'] == 8, 6,
np.where(base_dados['Desconto_Padrao_gh31'] == 9, 7,
np.where(base_dados['Desconto_Padrao_gh31'] == 10, 8,
np.where(base_dados['Desconto_Padrao_gh31'] == 11, 9,
np.where(base_dados['Desconto_Padrao_gh31'] == 13, 10,
np.where(base_dados['Desconto_Padrao_gh31'] == 14, 11,
np.where(base_dados['Desconto_Padrao_gh31'] == 15, 12,
np.where(base_dados['Desconto_Padrao_gh31'] == 16, 13,
np.where(base_dados['Desconto_Padrao_gh31'] == 17, 14,
np.where(base_dados['Desconto_Padrao_gh31'] == 19, 15,
np.where(base_dados['Desconto_Padrao_gh31'] == 20, 16,
np.where(base_dados['Desconto_Padrao_gh31'] == 21, 17,
0))))))))))))))))))
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
0))))))))))))))))))
base_dados['Desconto_Padrao_gh34'] = np.where(base_dados['Desconto_Padrao_gh33'] == 0, 0,
np.where(base_dados['Desconto_Padrao_gh33'] == 1, 10,
np.where(base_dados['Desconto_Padrao_gh33'] == 2, 0,
np.where(base_dados['Desconto_Padrao_gh33'] == 3, 10,
np.where(base_dados['Desconto_Padrao_gh33'] == 4, 8,
np.where(base_dados['Desconto_Padrao_gh33'] == 5, 5,
np.where(base_dados['Desconto_Padrao_gh33'] == 6, 10,
np.where(base_dados['Desconto_Padrao_gh33'] == 7, 10,
np.where(base_dados['Desconto_Padrao_gh33'] == 8, 8,
np.where(base_dados['Desconto_Padrao_gh33'] == 9, 0,
np.where(base_dados['Desconto_Padrao_gh33'] == 10, 10,
np.where(base_dados['Desconto_Padrao_gh33'] == 11, 17,
np.where(base_dados['Desconto_Padrao_gh33'] == 12, 10,
np.where(base_dados['Desconto_Padrao_gh33'] == 13, 13,
np.where(base_dados['Desconto_Padrao_gh33'] == 14, 10,
np.where(base_dados['Desconto_Padrao_gh33'] == 15, 0,
np.where(base_dados['Desconto_Padrao_gh33'] == 16, 16,
np.where(base_dados['Desconto_Padrao_gh33'] == 17, 17,
0))))))))))))))))))
base_dados['Desconto_Padrao_gh35'] = np.where(base_dados['Desconto_Padrao_gh34'] == 0, 0,
np.where(base_dados['Desconto_Padrao_gh34'] == 5, 1,
np.where(base_dados['Desconto_Padrao_gh34'] == 8, 2,
np.where(base_dados['Desconto_Padrao_gh34'] == 10, 3,
np.where(base_dados['Desconto_Padrao_gh34'] == 13, 4,
np.where(base_dados['Desconto_Padrao_gh34'] == 16, 5,
np.where(base_dados['Desconto_Padrao_gh34'] == 17, 6,
0)))))))
base_dados['Desconto_Padrao_gh36'] = np.where(base_dados['Desconto_Padrao_gh35'] == 0, 2,
np.where(base_dados['Desconto_Padrao_gh35'] == 1, 4,
np.where(base_dados['Desconto_Padrao_gh35'] == 2, 0,
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




base_dados['PRODUTO_gh40'] = np.where(base_dados['PRODUTO'] == 'AD', 0,
np.where(base_dados['PRODUTO'] == 'CARTAO', 1,
np.where(base_dados['PRODUTO'] == 'CHEQUE ESPECIAL', 2,
np.where(base_dados['PRODUTO'] == 'EMPRESTIMO', 3,
np.where(base_dados['PRODUTO'] == 'EMPRESTIMO FOLHA', 4,
np.where(base_dados['PRODUTO'] == 'OUTROS', 5,
np.where(base_dados['PRODUTO'] == 'REFIN', 6,
0)))))))
base_dados['PRODUTO_gh41'] = np.where(base_dados['PRODUTO_gh40'] == 0, -5,
np.where(base_dados['PRODUTO_gh40'] == 1, 0,
np.where(base_dados['PRODUTO_gh40'] == 2, -5,
np.where(base_dados['PRODUTO_gh40'] == 3, -5,
np.where(base_dados['PRODUTO_gh40'] == 4, -5,
np.where(base_dados['PRODUTO_gh40'] == 5, -5,
np.where(base_dados['PRODUTO_gh40'] == 6, -5,
0)))))))
base_dados['PRODUTO_gh42'] = np.where(base_dados['PRODUTO_gh41'] == -5, 0,
np.where(base_dados['PRODUTO_gh41'] == 0, 1,
0))
base_dados['PRODUTO_gh43'] = np.where(base_dados['PRODUTO_gh42'] == 0, 0,
np.where(base_dados['PRODUTO_gh42'] == 1, 1,
0))
base_dados['PRODUTO_gh44'] = np.where(base_dados['PRODUTO_gh43'] == 0, 0,
np.where(base_dados['PRODUTO_gh43'] == 1, 1,
0))
base_dados['PRODUTO_gh45'] = np.where(base_dados['PRODUTO_gh44'] == 0, 0,
np.where(base_dados['PRODUTO_gh44'] == 1, 1,
0))
base_dados['PRODUTO_gh46'] = np.where(base_dados['PRODUTO_gh45'] == 0, 0,
np.where(base_dados['PRODUTO_gh45'] == 1, 1,
0))
base_dados['PRODUTO_gh47'] = np.where(base_dados['PRODUTO_gh46'] == 0, 0,
np.where(base_dados['PRODUTO_gh46'] == 1, 1,
0))
base_dados['PRODUTO_gh48'] = np.where(base_dados['PRODUTO_gh47'] == 0, 0,
np.where(base_dados['PRODUTO_gh47'] == 1, 1,
0))
base_dados['PRODUTO_gh49'] = np.where(base_dados['PRODUTO_gh48'] == 0, 0,
np.where(base_dados['PRODUTO_gh48'] == 1, 1,
0))
base_dados['PRODUTO_gh50'] = np.where(base_dados['PRODUTO_gh49'] == 0, 0,
np.where(base_dados['PRODUTO_gh49'] == 1, 1,
0))
base_dados['PRODUTO_gh51'] = np.where(base_dados['PRODUTO_gh50'] == 0, 0,
np.where(base_dados['PRODUTO_gh50'] == 1, 1,
0))





                                      
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


base_dados['Saldo_Devedor_Contrato__pk_40'] = np.where(base_dados['Saldo_Devedor_Contrato'] == 0 , -1.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato'] > 0.0, base_dados['Saldo_Devedor_Contrato'] <= 523.37), 0.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato'] > 523.37, base_dados['Saldo_Devedor_Contrato'] <= 1187.7), 1.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato'] > 1187.7, base_dados['Saldo_Devedor_Contrato'] <= 2054.39), 2.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato'] > 2054.39, base_dados['Saldo_Devedor_Contrato'] <= 3005.81), 3.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato'] > 3005.81, base_dados['Saldo_Devedor_Contrato'] <= 3959.52), 4.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato'] > 3959.52, base_dados['Saldo_Devedor_Contrato'] <= 4896.7), 5.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato'] > 4896.7, base_dados['Saldo_Devedor_Contrato'] <= 5802.34), 6.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato'] > 5802.34, base_dados['Saldo_Devedor_Contrato'] <= 6710.95), 7.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato'] > 6710.95, base_dados['Saldo_Devedor_Contrato'] <= 7593.78), 8.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato'] > 7593.78, base_dados['Saldo_Devedor_Contrato'] <= 8398.37), 9.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato'] > 8398.37, base_dados['Saldo_Devedor_Contrato'] <= 9214.9), 10.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato'] > 9214.9, base_dados['Saldo_Devedor_Contrato'] <= 10100.97), 11.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato'] > 10100.97, base_dados['Saldo_Devedor_Contrato'] <= 11146.04), 12.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato'] > 11146.04, base_dados['Saldo_Devedor_Contrato'] <= 12334.29), 13.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato'] > 12334.29, base_dados['Saldo_Devedor_Contrato'] <= 13705.94), 14.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato'] > 13705.94, base_dados['Saldo_Devedor_Contrato'] <= 15446.63), 15.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato'] > 15446.63, base_dados['Saldo_Devedor_Contrato'] <= 17770.39), 16.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato'] > 17770.39, base_dados['Saldo_Devedor_Contrato'] <= 20587.04), 17.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato'] > 20587.04, base_dados['Saldo_Devedor_Contrato'] <= 24009.71), 18.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato'] > 24009.71, base_dados['Saldo_Devedor_Contrato'] <= 28597.88), 19.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato'] > 28597.88, base_dados['Saldo_Devedor_Contrato'] <= 33966.72), 20.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato'] > 33966.72, base_dados['Saldo_Devedor_Contrato'] <= 41212.56), 21.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato'] > 41212.56, base_dados['Saldo_Devedor_Contrato'] <= 51825.84), 22.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato'] > 51825.84, base_dados['Saldo_Devedor_Contrato'] <= 64341.49), 23.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato'] > 64341.49, base_dados['Saldo_Devedor_Contrato'] <= 76625.6), 24.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato'] > 76625.6, base_dados['Saldo_Devedor_Contrato'] <= 90631.26), 25.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato'] > 90631.26, base_dados['Saldo_Devedor_Contrato'] <= 105281.02), 26.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato'] > 105281.02, base_dados['Saldo_Devedor_Contrato'] <= 116872.92), 27.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato'] > 116872.92, base_dados['Saldo_Devedor_Contrato'] <= 125623.14), 28.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato'] > 125623.14, base_dados['Saldo_Devedor_Contrato'] <= 139300.47), 29.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato'] > 139300.47, base_dados['Saldo_Devedor_Contrato'] <= 150013.36), 30.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato'] > 150013.36, base_dados['Saldo_Devedor_Contrato'] <= 164175.06), 31.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato'] > 164175.06, base_dados['Saldo_Devedor_Contrato'] <= 171979.99), 32.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato'] > 171979.99, base_dados['Saldo_Devedor_Contrato'] <= 194481.34), 33.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato'] > 194481.34, base_dados['Saldo_Devedor_Contrato'] <= 216983.59), 34.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato'] > 216983.59, base_dados['Saldo_Devedor_Contrato'] <= 244148.26), 35.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato'] > 244148.26, base_dados['Saldo_Devedor_Contrato'] <= 252081.25), 36.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato'] > 252081.25, base_dados['Saldo_Devedor_Contrato'] <= 262579.87), 37.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato'] > 262579.87, base_dados['Saldo_Devedor_Contrato'] <= 418271.66), 38.0,
np.where(base_dados['Saldo_Devedor_Contrato'] > 418271.66, 39.0,
 0)))))))))))))))))))))))))))))))))))))))))
base_dados['Saldo_Devedor_Contrato__pk_40_g_1_1'] = np.where(base_dados['Saldo_Devedor_Contrato__pk_40'] == -1.0, 4,
np.where(base_dados['Saldo_Devedor_Contrato__pk_40'] == 0.0, 0,
np.where(base_dados['Saldo_Devedor_Contrato__pk_40'] == 1.0, 1,
np.where(base_dados['Saldo_Devedor_Contrato__pk_40'] == 2.0, 2,
np.where(base_dados['Saldo_Devedor_Contrato__pk_40'] == 3.0, 1,
np.where(base_dados['Saldo_Devedor_Contrato__pk_40'] == 4.0, 3,
np.where(base_dados['Saldo_Devedor_Contrato__pk_40'] == 5.0, 3,
np.where(base_dados['Saldo_Devedor_Contrato__pk_40'] == 6.0, 3,
np.where(base_dados['Saldo_Devedor_Contrato__pk_40'] == 7.0, 3,
np.where(base_dados['Saldo_Devedor_Contrato__pk_40'] == 8.0, 4,
np.where(base_dados['Saldo_Devedor_Contrato__pk_40'] == 9.0, 3,
np.where(base_dados['Saldo_Devedor_Contrato__pk_40'] == 10.0, 4,
np.where(base_dados['Saldo_Devedor_Contrato__pk_40'] == 11.0, 2,
np.where(base_dados['Saldo_Devedor_Contrato__pk_40'] == 12.0, 4,
np.where(base_dados['Saldo_Devedor_Contrato__pk_40'] == 13.0, 4,
np.where(base_dados['Saldo_Devedor_Contrato__pk_40'] == 14.0, 4,
np.where(base_dados['Saldo_Devedor_Contrato__pk_40'] == 15.0, 4,
np.where(base_dados['Saldo_Devedor_Contrato__pk_40'] == 16.0, 3,
np.where(base_dados['Saldo_Devedor_Contrato__pk_40'] == 17.0, 4,
np.where(base_dados['Saldo_Devedor_Contrato__pk_40'] == 18.0, 4,
np.where(base_dados['Saldo_Devedor_Contrato__pk_40'] == 19.0, 4,
np.where(base_dados['Saldo_Devedor_Contrato__pk_40'] == 20.0, 4,
np.where(base_dados['Saldo_Devedor_Contrato__pk_40'] == 21.0, 4,
np.where(base_dados['Saldo_Devedor_Contrato__pk_40'] == 22.0, 3,
np.where(base_dados['Saldo_Devedor_Contrato__pk_40'] == 23.0, 4,
np.where(base_dados['Saldo_Devedor_Contrato__pk_40'] == 24.0, 4,
np.where(base_dados['Saldo_Devedor_Contrato__pk_40'] == 25.0, 4,
np.where(base_dados['Saldo_Devedor_Contrato__pk_40'] == 26.0, 4,
np.where(base_dados['Saldo_Devedor_Contrato__pk_40'] == 27.0, 4,
np.where(base_dados['Saldo_Devedor_Contrato__pk_40'] == 28.0, 4,
np.where(base_dados['Saldo_Devedor_Contrato__pk_40'] == 29.0, 4,
np.where(base_dados['Saldo_Devedor_Contrato__pk_40'] == 30.0, 4,
np.where(base_dados['Saldo_Devedor_Contrato__pk_40'] == 31.0, 4,
np.where(base_dados['Saldo_Devedor_Contrato__pk_40'] == 32.0, 4,
np.where(base_dados['Saldo_Devedor_Contrato__pk_40'] == 33.0, 4,
np.where(base_dados['Saldo_Devedor_Contrato__pk_40'] == 34.0, 4,
np.where(base_dados['Saldo_Devedor_Contrato__pk_40'] == 35.0, 4,
np.where(base_dados['Saldo_Devedor_Contrato__pk_40'] == 36.0, 4,
np.where(base_dados['Saldo_Devedor_Contrato__pk_40'] == 37.0, 4,
np.where(base_dados['Saldo_Devedor_Contrato__pk_40'] == 38.0, 4,
np.where(base_dados['Saldo_Devedor_Contrato__pk_40'] == 39.0, 4,
 0)))))))))))))))))))))))))))))))))))))))))
base_dados['Saldo_Devedor_Contrato__pk_40_g_1_2'] = np.where(base_dados['Saldo_Devedor_Contrato__pk_40_g_1_1'] == 0, 4,
np.where(base_dados['Saldo_Devedor_Contrato__pk_40_g_1_1'] == 1, 3,
np.where(base_dados['Saldo_Devedor_Contrato__pk_40_g_1_1'] == 2, 2,
np.where(base_dados['Saldo_Devedor_Contrato__pk_40_g_1_1'] == 3, 1,
np.where(base_dados['Saldo_Devedor_Contrato__pk_40_g_1_1'] == 4, 0,
 0)))))
base_dados['Saldo_Devedor_Contrato__pk_40_g_1'] = np.where(base_dados['Saldo_Devedor_Contrato__pk_40_g_1_2'] == 0, 0,
np.where(base_dados['Saldo_Devedor_Contrato__pk_40_g_1_2'] == 1, 1,
np.where(base_dados['Saldo_Devedor_Contrato__pk_40_g_1_2'] == 2, 2,
np.where(base_dados['Saldo_Devedor_Contrato__pk_40_g_1_2'] == 3, 3,
np.where(base_dados['Saldo_Devedor_Contrato__pk_40_g_1_2'] == 4, 4,
 0)))))
         
         
         
         
         
         
base_dados['Saldo_Devedor_Contrato__R'] = np.sqrt(base_dados['Saldo_Devedor_Contrato'])
np.where(base_dados['Saldo_Devedor_Contrato__R'] == 0, -1, base_dados['Saldo_Devedor_Contrato__R'])
base_dados['Saldo_Devedor_Contrato__R'] = base_dados['Saldo_Devedor_Contrato__R'].fillna(-2)
base_dados['Saldo_Devedor_Contrato__R__pk_17'] = np.where(base_dados['Saldo_Devedor_Contrato__R'] <= 16.110245187457576, 0.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R'] > 16.110245187457576, base_dados['Saldo_Devedor_Contrato__R'] <= 28.675250652784186), 1.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R'] > 28.675250652784186, base_dados['Saldo_Devedor_Contrato__R'] <= 42.915381857790805), 2.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R'] > 42.915381857790805, base_dados['Saldo_Devedor_Contrato__R'] <= 59.47444829504516), 3.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R'] > 59.47444829504516, base_dados['Saldo_Devedor_Contrato__R'] <= 78.13289960061638), 4.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R'] > 78.13289960061638, base_dados['Saldo_Devedor_Contrato__R'] <= 99.44214398332329), 5.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R'] > 99.44214398332329, base_dados['Saldo_Devedor_Contrato__R'] <= 123.63118538621232), 6.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R'] > 123.63118538621232, base_dados['Saldo_Devedor_Contrato__R'] <= 152.555235898346), 7.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R'] > 152.555235898346, base_dados['Saldo_Devedor_Contrato__R'] <= 187.40168088893972), 8.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R'] > 187.40168088893972, base_dados['Saldo_Devedor_Contrato__R'] <= 223.3785128431112), 9.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R'] > 223.3785128431112, base_dados['Saldo_Devedor_Contrato__R'] <= 259.5399005933385), 10.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R'] > 259.5399005933385, base_dados['Saldo_Devedor_Contrato__R'] <= 301.0502615843408), 11.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R'] > 301.0502615843408, base_dados['Saldo_Devedor_Contrato__R'] <= 354.4335480735423), 12.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R'] > 354.4335480735423, base_dados['Saldo_Devedor_Contrato__R'] <= 414.70470216769905), 13.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R'] > 414.70470216769905, base_dados['Saldo_Devedor_Contrato__R'] <= 465.814973997187), 14.0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R'] > 465.814973997187, base_dados['Saldo_Devedor_Contrato__R'] <= 512.4254775086813), 15.0,
np.where(base_dados['Saldo_Devedor_Contrato__R'] > 512.4254775086813, 16.0,
 0)))))))))))))))))
base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1_1'] = np.where(base_dados['Saldo_Devedor_Contrato__R__pk_17'] == 0.0, 1,
np.where(base_dados['Saldo_Devedor_Contrato__R__pk_17'] == 1.0, 0,
np.where(base_dados['Saldo_Devedor_Contrato__R__pk_17'] == 2.0, 3,
np.where(base_dados['Saldo_Devedor_Contrato__R__pk_17'] == 3.0, 2,
np.where(base_dados['Saldo_Devedor_Contrato__R__pk_17'] == 4.0, 4,
np.where(base_dados['Saldo_Devedor_Contrato__R__pk_17'] == 5.0, 4,
np.where(base_dados['Saldo_Devedor_Contrato__R__pk_17'] == 6.0, 4,
np.where(base_dados['Saldo_Devedor_Contrato__R__pk_17'] == 7.0, 4,
np.where(base_dados['Saldo_Devedor_Contrato__R__pk_17'] == 8.0, 4,
np.where(base_dados['Saldo_Devedor_Contrato__R__pk_17'] == 9.0, 4,
np.where(base_dados['Saldo_Devedor_Contrato__R__pk_17'] == 10.0, 4,
np.where(base_dados['Saldo_Devedor_Contrato__R__pk_17'] == 11.0, 4,
np.where(base_dados['Saldo_Devedor_Contrato__R__pk_17'] == 12.0, 4,
np.where(base_dados['Saldo_Devedor_Contrato__R__pk_17'] == 13.0, 4,
np.where(base_dados['Saldo_Devedor_Contrato__R__pk_17'] == 14.0, 4,
np.where(base_dados['Saldo_Devedor_Contrato__R__pk_17'] == 15.0, 4,
np.where(base_dados['Saldo_Devedor_Contrato__R__pk_17'] == 16.0, 4,
 0)))))))))))))))))
base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1_2'] = np.where(base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1_1'] == 0, 3,
np.where(base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1_1'] == 1, 4,
np.where(base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1_1'] == 2, 1,
np.where(base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1_1'] == 3, 2,
np.where(base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1_1'] == 4, 0,
 0)))))
base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1'] = np.where(base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1_2'] == 0, 0,
np.where(base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1_2'] == 1, 1,
np.where(base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1_2'] == 2, 2,
np.where(base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1_2'] == 3, 3,
np.where(base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1_2'] == 4, 4,
 0)))))
         
         
         
         
         
         
base_dados['Cep1__pu_8'] = np.where(base_dados['Cep1'] <= 12490.0, 0.0,
np.where(base_dados['Cep1'] == 0 , -1.0,
np.where(np.bitwise_and(base_dados['Cep1'] > 0.0, base_dados['Cep1'] <= 24944.0), 1.0,
np.where(np.bitwise_and(base_dados['Cep1'] > 24944.0, base_dados['Cep1'] <= 37490.0), 2.0,
np.where(np.bitwise_and(base_dados['Cep1'] > 37490.0, base_dados['Cep1'] <= 49960.0), 3.0,
np.where(np.bitwise_and(base_dados['Cep1'] > 49960.0, base_dados['Cep1'] <= 62430.0), 4.0,
np.where(np.bitwise_and(base_dados['Cep1'] > 62430.0, base_dados['Cep1'] <= 74989.0), 5.0,
np.where(np.bitwise_and(base_dados['Cep1'] > 74989.0, base_dados['Cep1'] <= 87450.0), 6.0,
np.where(base_dados['Cep1'] > 87450.0, 7.0,
 0)))))))))
base_dados['Cep1__pu_8_g_1_1'] = np.where(base_dados['Cep1__pu_8'] == -1.0, 6,
np.where(base_dados['Cep1__pu_8'] == 0.0, 2,
np.where(base_dados['Cep1__pu_8'] == 1.0, 4,
np.where(base_dados['Cep1__pu_8'] == 2.0, 1,
np.where(base_dados['Cep1__pu_8'] == 3.0, 3,
np.where(base_dados['Cep1__pu_8'] == 4.0, 6,
np.where(base_dados['Cep1__pu_8'] == 5.0, 0,
np.where(base_dados['Cep1__pu_8'] == 6.0, 6,
np.where(base_dados['Cep1__pu_8'] == 7.0, 5,
 0)))))))))
base_dados['Cep1__pu_8_g_1_2'] = np.where(base_dados['Cep1__pu_8_g_1_1'] == 0, 1,
np.where(base_dados['Cep1__pu_8_g_1_1'] == 1, 4,
np.where(base_dados['Cep1__pu_8_g_1_1'] == 2, 6,
np.where(base_dados['Cep1__pu_8_g_1_1'] == 3, 1,
np.where(base_dados['Cep1__pu_8_g_1_1'] == 4, 4,
np.where(base_dados['Cep1__pu_8_g_1_1'] == 5, 0,
np.where(base_dados['Cep1__pu_8_g_1_1'] == 6, 1,
 0)))))))
base_dados['Cep1__pu_8_g_1'] = np.where(base_dados['Cep1__pu_8_g_1_2'] == 0, 0,
np.where(base_dados['Cep1__pu_8_g_1_2'] == 1, 1,
np.where(base_dados['Cep1__pu_8_g_1_2'] == 4, 2,
np.where(base_dados['Cep1__pu_8_g_1_2'] == 6, 3,
 0))))
         
         
         
         
         
         
         
base_dados['Cep1__pe_6'] = np.where(base_dados['Cep1'] == 0 , -1.0,
np.where(np.bitwise_and(base_dados['Cep1'] > 0.0, base_dados['Cep1'] <= 15870.0), 0.0,
np.where(np.bitwise_and(base_dados['Cep1'] > 15870.0, base_dados['Cep1'] <= 31748.0), 1.0,
np.where(np.bitwise_and(base_dados['Cep1'] > 31748.0, base_dados['Cep1'] <= 47600.0), 2.0,
np.where(np.bitwise_and(base_dados['Cep1'] > 47600.0, base_dados['Cep1'] <= 63500.0), 3.0,
np.where(np.bitwise_and(base_dados['Cep1'] > 63500.0, base_dados['Cep1'] <= 79370.0), 4.0,
np.where(base_dados['Cep1'] > 79370.0, 5.0,
 -2)))))))
base_dados['Cep1__pe_6_g_1_1'] = np.where(base_dados['Cep1__pe_6'] == -2.0, 3,
np.where(base_dados['Cep1__pe_6'] == -1.0, 3,
np.where(base_dados['Cep1__pe_6'] == 0.0, 2,
np.where(base_dados['Cep1__pe_6'] == 1.0, 3,
np.where(base_dados['Cep1__pe_6'] == 2.0, 1,
np.where(base_dados['Cep1__pe_6'] == 3.0, 3,
np.where(base_dados['Cep1__pe_6'] == 4.0, 0,
np.where(base_dados['Cep1__pe_6'] == 5.0, 3,
 0))))))))
base_dados['Cep1__pe_6_g_1_2'] = np.where(base_dados['Cep1__pe_6_g_1_1'] == 0, 0,
np.where(base_dados['Cep1__pe_6_g_1_1'] == 1, 1,
np.where(base_dados['Cep1__pe_6_g_1_1'] == 2, 3,
np.where(base_dados['Cep1__pe_6_g_1_1'] == 3, 1,
 0))))
base_dados['Cep1__pe_6_g_1'] = np.where(base_dados['Cep1__pe_6_g_1_2'] == 0, 0,
np.where(base_dados['Cep1__pe_6_g_1_2'] == 1, 1,
np.where(base_dados['Cep1__pe_6_g_1_2'] == 3, 2,
 0)))
         
         
         
         
         
         
base_dados['Codigo_Politica__L'] = np.log(base_dados['Codigo_Politica'])
np.where(base_dados['Codigo_Politica__L'] == 0, -1, base_dados['Codigo_Politica__L'])
base_dados['Codigo_Politica__L'] = base_dados['Codigo_Politica__L'].fillna(-2)
base_dados['Codigo_Politica__L__pu_3'] = np.where(base_dados['Codigo_Politica__L'] <= 4.736198448394496, 0.0,
np.where(np.bitwise_and(base_dados['Codigo_Politica__L'] > 4.736198448394496, base_dados['Codigo_Politica__L'] <= 5.602118820879701), 1.0,
np.where(base_dados['Codigo_Politica__L'] > 5.602118820879701, 2.0,
 0)))
base_dados['Codigo_Politica__L__pu_3_g_1_1'] = np.where(base_dados['Codigo_Politica__L__pu_3'] == 0.0, 2,
np.where(base_dados['Codigo_Politica__L__pu_3'] == 1.0, 1,
np.where(base_dados['Codigo_Politica__L__pu_3'] == 2.0, 0,
 0)))
base_dados['Codigo_Politica__L__pu_3_g_1_2'] = np.where(base_dados['Codigo_Politica__L__pu_3_g_1_1'] == 0, 1,
np.where(base_dados['Codigo_Politica__L__pu_3_g_1_1'] == 1, 0,
np.where(base_dados['Codigo_Politica__L__pu_3_g_1_1'] == 2, 1,
 0)))
base_dados['Codigo_Politica__L__pu_3_g_1'] = np.where(base_dados['Codigo_Politica__L__pu_3_g_1_2'] == 0, 0,
np.where(base_dados['Codigo_Politica__L__pu_3_g_1_2'] == 1, 1,
 0))
                                                      
                                                      
                                                      
                                                      
                                                      
                                                      
base_dados['Codigo_Politica__S'] = np.sin(base_dados['Codigo_Politica'])
np.where(base_dados['Codigo_Politica__S'] == 0, -1, base_dados['Codigo_Politica__S'])
base_dados['Codigo_Politica__S'] = base_dados['Codigo_Politica__S'].fillna(-2)
base_dados['Codigo_Politica__S__p_7'] = np.where(base_dados['Codigo_Politica__S'] <= -0.9300948780045254, 0.0,
np.where(np.bitwise_and(base_dados['Codigo_Politica__S'] > -0.9300948780045254, base_dados['Codigo_Politica__S'] <= -0.643491986364181), 1.0,
np.where(np.bitwise_and(base_dados['Codigo_Politica__S'] > -0.643491986364181, base_dados['Codigo_Politica__S'] <= -0.3714041014380902), 2.0,
np.where(np.bitwise_and(base_dados['Codigo_Politica__S'] > -0.3714041014380902, base_dados['Codigo_Politica__S'] <= 0.2021203593127912), 3.0,
np.where(np.bitwise_and(base_dados['Codigo_Politica__S'] > 0.2021203593127912, base_dados['Codigo_Politica__S'] <= 0.6090440218832924), 4.0,
np.where(np.bitwise_and(base_dados['Codigo_Politica__S'] > 0.6090440218832924, base_dados['Codigo_Politica__S'] <= 0.9589413745467228), 5.0,
np.where(base_dados['Codigo_Politica__S'] > 0.9589413745467228, 6.0,
 0)))))))
base_dados['Codigo_Politica__S__p_7_g_1_1'] = np.where(base_dados['Codigo_Politica__S__p_7'] == 0.0, 0,
np.where(base_dados['Codigo_Politica__S__p_7'] == 1.0, 2,
np.where(base_dados['Codigo_Politica__S__p_7'] == 2.0, 0,
np.where(base_dados['Codigo_Politica__S__p_7'] == 3.0, 2,
np.where(base_dados['Codigo_Politica__S__p_7'] == 4.0, 0,
np.where(base_dados['Codigo_Politica__S__p_7'] == 5.0, 1,
np.where(base_dados['Codigo_Politica__S__p_7'] == 6.0, 0,
 0)))))))
base_dados['Codigo_Politica__S__p_7_g_1_2'] = np.where(base_dados['Codigo_Politica__S__p_7_g_1_1'] == 0, 2,
np.where(base_dados['Codigo_Politica__S__p_7_g_1_1'] == 1, 0,
np.where(base_dados['Codigo_Politica__S__p_7_g_1_1'] == 2, 1,
 0)))
base_dados['Codigo_Politica__S__p_7_g_1'] = np.where(base_dados['Codigo_Politica__S__p_7_g_1_2'] == 0, 0,
np.where(base_dados['Codigo_Politica__S__p_7_g_1_2'] == 1, 1,
np.where(base_dados['Codigo_Politica__S__p_7_g_1_2'] == 2, 2,
 0)))
         
         
         
         
         
         
base_dados['Contrato_Altair__p_2'] = np.where(base_dados['Contrato_Altair'] <= 698100000000.0, 0.0,
np.where(base_dados['Contrato_Altair'] > 698100000000.0, 1.0,
 0))
base_dados['Contrato_Altair__p_2_g_1_1'] = np.where(base_dados['Contrato_Altair__p_2'] == 0.0, 1,
np.where(base_dados['Contrato_Altair__p_2'] == 1.0, 0,
 0))
base_dados['Contrato_Altair__p_2_g_1_2'] = np.where(base_dados['Contrato_Altair__p_2_g_1_1'] == 0, 1,
np.where(base_dados['Contrato_Altair__p_2_g_1_1'] == 1, 0,
 0))
base_dados['Contrato_Altair__p_2_g_1'] = np.where(base_dados['Contrato_Altair__p_2_g_1_2'] == 0, 0,
np.where(base_dados['Contrato_Altair__p_2_g_1_2'] == 1, 1,
 0))
                                                  
                                                  
                                                  
                                                  
                                                  
                                                  
                                                  
base_dados['Contrato_Altair__pk_40'] = np.where(base_dados['Contrato_Altair'] <= 30023984.0, 0.0,
np.where(np.bitwise_and(base_dados['Contrato_Altair'] > 30023984.0, base_dados['Contrato_Altair'] <= 33500074701.0), 1.0,
np.where(np.bitwise_and(base_dados['Contrato_Altair'] > 33500074701.0, base_dados['Contrato_Altair'] <= 63310045800.0), 2.0,
np.where(np.bitwise_and(base_dados['Contrato_Altair'] > 63310045800.0, base_dados['Contrato_Altair'] <= 69936495205.0), 3.0,
np.where(np.bitwise_and(base_dados['Contrato_Altair'] > 69936495205.0, base_dados['Contrato_Altair'] <= 69982662583.0), 4.0,
np.where(np.bitwise_and(base_dados['Contrato_Altair'] > 69982662583.0, base_dados['Contrato_Altair'] <= 69996083747.0), 5.0,
np.where(np.bitwise_and(base_dados['Contrato_Altair'] > 69996083747.0, base_dados['Contrato_Altair'] <= 69997768735.0), 6.0,
np.where(np.bitwise_and(base_dados['Contrato_Altair'] > 69997768735.0, base_dados['Contrato_Altair'] <= 69998059218.0), 7.0,
np.where(np.bitwise_and(base_dados['Contrato_Altair'] > 69998059218.0, base_dados['Contrato_Altair'] <= 69998082517.0), 8.0,
np.where(np.bitwise_and(base_dados['Contrato_Altair'] > 69998082517.0, base_dados['Contrato_Altair'] <= 69998340383.0), 10.0,
np.where(np.bitwise_and(base_dados['Contrato_Altair'] > 69998340383.0, base_dados['Contrato_Altair'] <= 69998402427.0), 11.0,
np.where(np.bitwise_and(base_dados['Contrato_Altair'] > 69998402427.0, base_dados['Contrato_Altair'] <= 69998552670.0), 12.0,
np.where(np.bitwise_and(base_dados['Contrato_Altair'] > 69998552670.0, base_dados['Contrato_Altair'] <= 69998706271.0), 13.0,
np.where(np.bitwise_and(base_dados['Contrato_Altair'] > 69998706271.0, base_dados['Contrato_Altair'] <= 69998848290.0), 15.0,
np.where(np.bitwise_and(base_dados['Contrato_Altair'] > 69998848290.0, base_dados['Contrato_Altair'] <= 69998990029.0), 17.0,
np.where(np.bitwise_and(base_dados['Contrato_Altair'] > 69998990029.0, base_dados['Contrato_Altair'] <= 69999178871.0), 24.0,
np.where(np.bitwise_and(base_dados['Contrato_Altair'] > 69999178871.0, base_dados['Contrato_Altair'] <= 69999270313.0), 28.0,
np.where(np.bitwise_and(base_dados['Contrato_Altair'] > 69999270313.0, base_dados['Contrato_Altair'] <= 69999991397.0), 29.0,
np.where(np.bitwise_and(base_dados['Contrato_Altair'] > 69999991397.0, base_dados['Contrato_Altair'] <= 320010000000.0), 30.0,
np.where(np.bitwise_and(base_dados['Contrato_Altair'] > 320010000000.0, base_dados['Contrato_Altair'] <= 512158000000.0), 31.0,
np.where(np.bitwise_and(base_dados['Contrato_Altair'] > 512158000000.0, base_dados['Contrato_Altair'] <= 660015000000.0), 32.0,
np.where(np.bitwise_and(base_dados['Contrato_Altair'] > 660015000000.0, base_dados['Contrato_Altair'] <= 660060000000.0), 33.0,
np.where(np.bitwise_and(base_dados['Contrato_Altair'] > 660060000000.0, base_dados['Contrato_Altair'] <= 660095000000.0), 34.0,
np.where(np.bitwise_and(base_dados['Contrato_Altair'] > 660095000000.0, base_dados['Contrato_Altair'] <= 660143000000.0), 35.0,
np.where(np.bitwise_and(base_dados['Contrato_Altair'] > 660143000000.0, base_dados['Contrato_Altair'] <= 669995000000.0), 36.0,
np.where(np.bitwise_and(base_dados['Contrato_Altair'] > 669995000000.0, base_dados['Contrato_Altair'] <= 698100000000.0), 37.0,
np.where(np.bitwise_and(base_dados['Contrato_Altair'] > 698100000000.0, base_dados['Contrato_Altair'] <= 698215000000.0), 38.0,
np.where(base_dados['Contrato_Altair'] > 698215000000.0, 39.0,
 0))))))))))))))))))))))))))))
base_dados['Contrato_Altair__pk_40_g_1_1'] = np.where(base_dados['Contrato_Altair__pk_40'] == 0.0, 3,
np.where(base_dados['Contrato_Altair__pk_40'] == 1.0, 3,
np.where(base_dados['Contrato_Altair__pk_40'] == 2.0, 2,
np.where(base_dados['Contrato_Altair__pk_40'] == 3.0, 2,
np.where(base_dados['Contrato_Altair__pk_40'] == 4.0, 3,
np.where(base_dados['Contrato_Altair__pk_40'] == 5.0, 3,
np.where(base_dados['Contrato_Altair__pk_40'] == 6.0, 3,
np.where(base_dados['Contrato_Altair__pk_40'] == 7.0, 3,
np.where(base_dados['Contrato_Altair__pk_40'] == 8.0, 3,
np.where(base_dados['Contrato_Altair__pk_40'] == 10.0, 3,
np.where(base_dados['Contrato_Altair__pk_40'] == 11.0, 3,
np.where(base_dados['Contrato_Altair__pk_40'] == 12.0, 3,
np.where(base_dados['Contrato_Altair__pk_40'] == 13.0, 3,
np.where(base_dados['Contrato_Altair__pk_40'] == 15.0, 3,
np.where(base_dados['Contrato_Altair__pk_40'] == 17.0, 3,
np.where(base_dados['Contrato_Altair__pk_40'] == 24.0, 3,
np.where(base_dados['Contrato_Altair__pk_40'] == 28.0, 3,
np.where(base_dados['Contrato_Altair__pk_40'] == 29.0, 3,
np.where(base_dados['Contrato_Altair__pk_40'] == 30.0, 2,
np.where(base_dados['Contrato_Altair__pk_40'] == 31.0, 3,
np.where(base_dados['Contrato_Altair__pk_40'] == 32.0, 2,
np.where(base_dados['Contrato_Altair__pk_40'] == 33.0, 3,
np.where(base_dados['Contrato_Altair__pk_40'] == 34.0, 3,
np.where(base_dados['Contrato_Altair__pk_40'] == 35.0, 3,
np.where(base_dados['Contrato_Altair__pk_40'] == 36.0, 3,
np.where(base_dados['Contrato_Altair__pk_40'] == 37.0, 0,
np.where(base_dados['Contrato_Altair__pk_40'] == 38.0, 1,
np.where(base_dados['Contrato_Altair__pk_40'] == 39.0, 0,
 0))))))))))))))))))))))))))))
base_dados['Contrato_Altair__pk_40_g_1_2'] = np.where(base_dados['Contrato_Altair__pk_40_g_1_1'] == 0, 3,
np.where(base_dados['Contrato_Altair__pk_40_g_1_1'] == 1, 1,
np.where(base_dados['Contrato_Altair__pk_40_g_1_1'] == 2, 0,
np.where(base_dados['Contrato_Altair__pk_40_g_1_1'] == 3, 1,
 0))))
base_dados['Contrato_Altair__pk_40_g_1'] = np.where(base_dados['Contrato_Altair__pk_40_g_1_2'] == 0, 0,
np.where(base_dados['Contrato_Altair__pk_40_g_1_2'] == 1, 1,
np.where(base_dados['Contrato_Altair__pk_40_g_1_2'] == 3, 2,
 0)))
         
         
         
         
         
         
base_dados['Cep2__p_6'] = np.where(base_dados['Cep2'] <= 45070.0, 0.0,
np.where(base_dados['Cep2'] > 45070.0, 1.0,
 0))
base_dados['Cep2__p_6_g_1_1'] = np.where(base_dados['Cep2__p_6'] == 0.0, 0,
np.where(base_dados['Cep2__p_6'] == 1.0, 1,
 0))
base_dados['Cep2__p_6_g_1_2'] = np.where(base_dados['Cep2__p_6_g_1_1'] == 0, 1,
np.where(base_dados['Cep2__p_6_g_1_1'] == 1, 0,
 0))
base_dados['Cep2__p_6_g_1'] = np.where(base_dados['Cep2__p_6_g_1_2'] == 0, 0,
np.where(base_dados['Cep2__p_6_g_1_2'] == 1, 1,
 0))
                                       
                                       
                                       
                                       
                                       
                                       
base_dados['Cep2__L'] = np.log(base_dados['Cep2'])
np.where(base_dados['Cep2__L'] == 0, -1, base_dados['Cep2__L'])
base_dados['Cep2__L'] = base_dados['Cep2__L'].fillna(-2)
base_dados['Cep2__L__p_15'] = np.where(base_dados['Cep2__L'] <= 9.80477157959697, 0.0,
np.where(np.bitwise_and(base_dados['Cep2__L'] > 9.80477157959697, base_dados['Cep2__L'] <= 10.394732724323175), 1.0,
np.where(np.bitwise_and(base_dados['Cep2__L'] > 10.394732724323175, base_dados['Cep2__L'] <= 10.97164063371953), 2.0,
np.where(np.bitwise_and(base_dados['Cep2__L'] > 10.97164063371953, base_dados['Cep2__L'] <= 11.240460358323308), 3.0,
np.where(base_dados['Cep2__L'] > 11.240460358323308, 4.0,
 0)))))
base_dados['Cep2__L__p_15_g_1_1'] = np.where(base_dados['Cep2__L__p_15'] == 0.0, 0,
np.where(base_dados['Cep2__L__p_15'] == 1.0, 1,
np.where(base_dados['Cep2__L__p_15'] == 2.0, 0,
np.where(base_dados['Cep2__L__p_15'] == 3.0, 1,
np.where(base_dados['Cep2__L__p_15'] == 4.0, 0,
 0)))))
base_dados['Cep2__L__p_15_g_1_2'] = np.where(base_dados['Cep2__L__p_15_g_1_1'] == 0, 1,
np.where(base_dados['Cep2__L__p_15_g_1_1'] == 1, 0,
 0))
base_dados['Cep2__L__p_15_g_1'] = np.where(base_dados['Cep2__L__p_15_g_1_2'] == 0, 0,
np.where(base_dados['Cep2__L__p_15_g_1_2'] == 1, 1,
 0))
         
         
         
         
         
         
         
         
base_dados['MOB_ENTRADA__p_7'] = np.where(base_dados['MOB_ENTRADA'] <= 4.377509421838459, 0.0,
np.where(np.bitwise_and(base_dados['MOB_ENTRADA'] > 4.377509421838459, base_dados['MOB_ENTRADA'] <= 6.250237814618598), 1.0,
np.where(np.bitwise_and(base_dados['MOB_ENTRADA'] > 6.250237814618598, base_dados['MOB_ENTRADA'] <= 8.615789468656669), 2.0,
np.where(np.bitwise_and(base_dados['MOB_ENTRADA'] > 8.615789468656669, base_dados['MOB_ENTRADA'] <= 10.948486238610878), 3.0,
np.where(np.bitwise_and(base_dados['MOB_ENTRADA'] > 10.948486238610878, base_dados['MOB_ENTRADA'] <= 14.29968441516481), 4.0,
np.where(np.bitwise_and(base_dados['MOB_ENTRADA'] > 14.29968441516481, base_dados['MOB_ENTRADA'] <= 19.62217563675047), 5.0,
np.where(base_dados['MOB_ENTRADA'] > 19.62217563675047, 6.0,
 0)))))))
base_dados['MOB_ENTRADA__p_7_g_1_1'] = np.where(base_dados['MOB_ENTRADA__p_7'] == 0.0, 2,
np.where(base_dados['MOB_ENTRADA__p_7'] == 1.0, 0,
np.where(base_dados['MOB_ENTRADA__p_7'] == 2.0, 0,
np.where(base_dados['MOB_ENTRADA__p_7'] == 3.0, 0,
np.where(base_dados['MOB_ENTRADA__p_7'] == 4.0, 0,
np.where(base_dados['MOB_ENTRADA__p_7'] == 5.0, 2,
np.where(base_dados['MOB_ENTRADA__p_7'] == 6.0, 1,
 0)))))))
base_dados['MOB_ENTRADA__p_7_g_1_2'] = np.where(base_dados['MOB_ENTRADA__p_7_g_1_1'] == 0, 1,
np.where(base_dados['MOB_ENTRADA__p_7_g_1_1'] == 1, 0,
np.where(base_dados['MOB_ENTRADA__p_7_g_1_1'] == 2, 1,
 0)))
base_dados['MOB_ENTRADA__p_7_g_1'] = np.where(base_dados['MOB_ENTRADA__p_7_g_1_2'] == 0, 0,
np.where(base_dados['MOB_ENTRADA__p_7_g_1_2'] == 1, 1,
 0))
                                              
                                              
                                              
                                              
                                              
                                              
base_dados['MOB_ENTRADA__C'] = np.cos(base_dados['MOB_ENTRADA'])
np.where(base_dados['MOB_ENTRADA__C'] == 0, -1, base_dados['MOB_ENTRADA__C'])
base_dados['MOB_ENTRADA__C'] = base_dados['MOB_ENTRADA__C'].fillna(-2)
base_dados['MOB_ENTRADA__C__p_5'] = np.where(base_dados['MOB_ENTRADA__C'] <= -0.8186879065763143, 0.0,
np.where(np.bitwise_and(base_dados['MOB_ENTRADA__C'] > -0.8186879065763143, base_dados['MOB_ENTRADA__C'] <= -0.18637373497223053), 1.0,
np.where(np.bitwise_and(base_dados['MOB_ENTRADA__C'] > -0.18637373497223053, base_dados['MOB_ENTRADA__C'] <= 0.6499467800374286), 2.0,
np.where(np.bitwise_and(base_dados['MOB_ENTRADA__C'] > 0.6499467800374286, base_dados['MOB_ENTRADA__C'] <= 0.7498903163479008), 3.0,
np.where(base_dados['MOB_ENTRADA__C'] > 0.7498903163479008, 4.0,
 0)))))
base_dados['MOB_ENTRADA__C__p_5_g_1_1'] = np.where(base_dados['MOB_ENTRADA__C__p_5'] == 0.0, 0,
np.where(base_dados['MOB_ENTRADA__C__p_5'] == 1.0, 0,
np.where(base_dados['MOB_ENTRADA__C__p_5'] == 2.0, 0,
np.where(base_dados['MOB_ENTRADA__C__p_5'] == 3.0, 1,
np.where(base_dados['MOB_ENTRADA__C__p_5'] == 4.0, 0,
 0)))))
base_dados['MOB_ENTRADA__C__p_5_g_1_2'] = np.where(base_dados['MOB_ENTRADA__C__p_5_g_1_1'] == 0, 1,
np.where(base_dados['MOB_ENTRADA__C__p_5_g_1_1'] == 1, 0,
 0))
base_dados['MOB_ENTRADA__C__p_5_g_1'] = np.where(base_dados['MOB_ENTRADA__C__p_5_g_1_2'] == 0, 0,
np.where(base_dados['MOB_ENTRADA__C__p_5_g_1_2'] == 1, 1,
 0))


# COMMAND ----------

# MAGIC %md
# MAGIC ### Parte 2 de 2

# COMMAND ----------

base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1_c1_76_1'] = np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__pk_40_g_1'] == 0, base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__pk_40_g_1'] == 0, base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__pk_40_g_1'] == 0, base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1'] == 2), 2,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__pk_40_g_1'] == 0, base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1'] == 3), 3,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__pk_40_g_1'] == 0, base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1'] == 4), 4,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__pk_40_g_1'] == 1, base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__pk_40_g_1'] == 1, base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__pk_40_g_1'] == 1, base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1'] == 2), 2,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__pk_40_g_1'] == 1, base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1'] == 3), 3,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__pk_40_g_1'] == 1, base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1'] == 4), 4,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__pk_40_g_1'] == 2, base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__pk_40_g_1'] == 2, base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__pk_40_g_1'] == 2, base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1'] == 2), 3,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__pk_40_g_1'] == 2, base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1'] == 3), 4,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__pk_40_g_1'] == 2, base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1'] == 4), 5,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__pk_40_g_1'] == 3, base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1'] == 0), 2,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__pk_40_g_1'] == 3, base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__pk_40_g_1'] == 3, base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1'] == 2), 3,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__pk_40_g_1'] == 3, base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1'] == 3), 4,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__pk_40_g_1'] == 3, base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1'] == 4), 5,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__pk_40_g_1'] == 4, base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1'] == 0), 2,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__pk_40_g_1'] == 4, base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1'] == 1), 3,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__pk_40_g_1'] == 4, base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1'] == 2), 4,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__pk_40_g_1'] == 4, base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1'] == 3), 4,
np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__pk_40_g_1'] == 4, base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1'] == 4), 5,
 0)))))))))))))))))))))))))
base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1_c1_76_2'] = np.where(base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1_c1_76_1'] == 0, 0,
np.where(base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1_c1_76_1'] == 1, 1,
np.where(base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1_c1_76_1'] == 2, 2,
np.where(base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1_c1_76_1'] == 3, 3,
np.where(base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1_c1_76_1'] == 4, 4,
np.where(base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1_c1_76_1'] == 5, 5,
0))))))
base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1_c1_76'] = np.where(base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1_c1_76_2'] == 0, 0,
np.where(base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1_c1_76_2'] == 1, 1,
np.where(base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1_c1_76_2'] == 2, 2,
np.where(base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1_c1_76_2'] == 3, 3,
np.where(base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1_c1_76_2'] == 4, 4,
np.where(base_dados['Saldo_Devedor_Contrato__R__pk_17_g_1_c1_76_2'] == 5, 5,
 0))))))
         
         
         
         
         
         
         
base_dados['Cep1__pu_8_g_1_c1_20_1'] = np.where(np.bitwise_and(base_dados['Cep1__pu_8_g_1'] == 0, base_dados['Cep1__pe_6_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['Cep1__pu_8_g_1'] == 0, base_dados['Cep1__pe_6_g_1'] == 1), 0,
np.where(np.bitwise_and(base_dados['Cep1__pu_8_g_1'] == 0, base_dados['Cep1__pe_6_g_1'] == 2), 1,
np.where(np.bitwise_and(base_dados['Cep1__pu_8_g_1'] == 1, base_dados['Cep1__pe_6_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['Cep1__pu_8_g_1'] == 1, base_dados['Cep1__pe_6_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['Cep1__pu_8_g_1'] == 1, base_dados['Cep1__pe_6_g_1'] == 2), 3,
np.where(np.bitwise_and(base_dados['Cep1__pu_8_g_1'] == 2, base_dados['Cep1__pe_6_g_1'] == 0), 2,
np.where(np.bitwise_and(base_dados['Cep1__pu_8_g_1'] == 2, base_dados['Cep1__pe_6_g_1'] == 1), 3,
np.where(np.bitwise_and(base_dados['Cep1__pu_8_g_1'] == 2, base_dados['Cep1__pe_6_g_1'] == 2), 4,
np.where(np.bitwise_and(base_dados['Cep1__pu_8_g_1'] == 3, base_dados['Cep1__pe_6_g_1'] == 0), 3,
np.where(np.bitwise_and(base_dados['Cep1__pu_8_g_1'] == 3, base_dados['Cep1__pe_6_g_1'] == 1), 4,
np.where(np.bitwise_and(base_dados['Cep1__pu_8_g_1'] == 3, base_dados['Cep1__pe_6_g_1'] == 2), 4,
 0))))))))))))
base_dados['Cep1__pu_8_g_1_c1_20_2'] = np.where(base_dados['Cep1__pu_8_g_1_c1_20_1'] == 0, 0,
np.where(base_dados['Cep1__pu_8_g_1_c1_20_1'] == 1, 1,
np.where(base_dados['Cep1__pu_8_g_1_c1_20_1'] == 2, 2,
np.where(base_dados['Cep1__pu_8_g_1_c1_20_1'] == 3, 3,
np.where(base_dados['Cep1__pu_8_g_1_c1_20_1'] == 4, 4,
0)))))
base_dados['Cep1__pu_8_g_1_c1_20'] = np.where(base_dados['Cep1__pu_8_g_1_c1_20_2'] == 0, 0,
np.where(base_dados['Cep1__pu_8_g_1_c1_20_2'] == 1, 1,
np.where(base_dados['Cep1__pu_8_g_1_c1_20_2'] == 2, 2,
np.where(base_dados['Cep1__pu_8_g_1_c1_20_2'] == 3, 3,
np.where(base_dados['Cep1__pu_8_g_1_c1_20_2'] == 4, 4,
 0)))))
         
         
         
         
         
         
base_dados['Codigo_Politica__S__p_7_g_1_c1_8_1'] = np.where(np.bitwise_and(base_dados['Codigo_Politica__L__pu_3_g_1'] == 0, base_dados['Codigo_Politica__S__p_7_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['Codigo_Politica__L__pu_3_g_1'] == 0, base_dados['Codigo_Politica__S__p_7_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['Codigo_Politica__L__pu_3_g_1'] == 0, base_dados['Codigo_Politica__S__p_7_g_1'] == 2), 3,
np.where(np.bitwise_and(base_dados['Codigo_Politica__L__pu_3_g_1'] == 1, base_dados['Codigo_Politica__S__p_7_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['Codigo_Politica__L__pu_3_g_1'] == 1, base_dados['Codigo_Politica__S__p_7_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['Codigo_Politica__L__pu_3_g_1'] == 1, base_dados['Codigo_Politica__S__p_7_g_1'] == 2), 4,
 0))))))
base_dados['Codigo_Politica__S__p_7_g_1_c1_8_2'] = np.where(base_dados['Codigo_Politica__S__p_7_g_1_c1_8_1'] == 0, 1,
np.where(base_dados['Codigo_Politica__S__p_7_g_1_c1_8_1'] == 1, 0,
np.where(base_dados['Codigo_Politica__S__p_7_g_1_c1_8_1'] == 2, 3,
np.where(base_dados['Codigo_Politica__S__p_7_g_1_c1_8_1'] == 3, 2,
np.where(base_dados['Codigo_Politica__S__p_7_g_1_c1_8_1'] == 4, 3,
0)))))
base_dados['Codigo_Politica__S__p_7_g_1_c1_8'] = np.where(base_dados['Codigo_Politica__S__p_7_g_1_c1_8_2'] == 0, 0,
np.where(base_dados['Codigo_Politica__S__p_7_g_1_c1_8_2'] == 1, 1,
np.where(base_dados['Codigo_Politica__S__p_7_g_1_c1_8_2'] == 2, 2,
np.where(base_dados['Codigo_Politica__S__p_7_g_1_c1_8_2'] == 3, 3,
 0))))
         
         
         
         
         
         
base_dados['Contrato_Altair__pk_40_g_1_c1_4_1'] = np.where(np.bitwise_and(base_dados['Contrato_Altair__p_2_g_1'] == 0, base_dados['Contrato_Altair__pk_40_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['Contrato_Altair__p_2_g_1'] == 0, base_dados['Contrato_Altair__pk_40_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['Contrato_Altair__p_2_g_1'] == 0, base_dados['Contrato_Altair__pk_40_g_1'] == 2), 2,
np.where(np.bitwise_and(base_dados['Contrato_Altair__p_2_g_1'] == 1, base_dados['Contrato_Altair__pk_40_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['Contrato_Altair__p_2_g_1'] == 1, base_dados['Contrato_Altair__pk_40_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['Contrato_Altair__p_2_g_1'] == 1, base_dados['Contrato_Altair__pk_40_g_1'] == 2), 3,
 0))))))
base_dados['Contrato_Altair__pk_40_g_1_c1_4_2'] = np.where(base_dados['Contrato_Altair__pk_40_g_1_c1_4_1'] == 0, 0,
np.where(base_dados['Contrato_Altair__pk_40_g_1_c1_4_1'] == 1, 1,
np.where(base_dados['Contrato_Altair__pk_40_g_1_c1_4_1'] == 2, 2,
np.where(base_dados['Contrato_Altair__pk_40_g_1_c1_4_1'] == 3, 3,
0))))
base_dados['Contrato_Altair__pk_40_g_1_c1_4'] = np.where(base_dados['Contrato_Altair__pk_40_g_1_c1_4_2'] == 0, 0,
np.where(base_dados['Contrato_Altair__pk_40_g_1_c1_4_2'] == 1, 1,
np.where(base_dados['Contrato_Altair__pk_40_g_1_c1_4_2'] == 2, 2,
np.where(base_dados['Contrato_Altair__pk_40_g_1_c1_4_2'] == 3, 3,
 0))))
         
         
         
         
         
         
base_dados['Cep2__L__p_15_g_1_c1_7_1'] = np.where(np.bitwise_and(base_dados['Cep2__p_6_g_1'] == 0, base_dados['Cep2__L__p_15_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['Cep2__p_6_g_1'] == 0, base_dados['Cep2__L__p_15_g_1'] == 1), 0,
np.where(np.bitwise_and(base_dados['Cep2__p_6_g_1'] == 1, base_dados['Cep2__L__p_15_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['Cep2__p_6_g_1'] == 1, base_dados['Cep2__L__p_15_g_1'] == 1), 1,
 0))))
base_dados['Cep2__L__p_15_g_1_c1_7_2'] = np.where(base_dados['Cep2__L__p_15_g_1_c1_7_1'] == 0, 0,
np.where(base_dados['Cep2__L__p_15_g_1_c1_7_1'] == 1, 1,
0))
base_dados['Cep2__L__p_15_g_1_c1_7'] = np.where(base_dados['Cep2__L__p_15_g_1_c1_7_2'] == 0, 0,
np.where(base_dados['Cep2__L__p_15_g_1_c1_7_2'] == 1, 1,
 0))
                                                
                                                
                                                
                                                
                                                
base_dados['MOB_ENTRADA__C__p_5_g_1_c1_30_1'] = np.where(np.bitwise_and(base_dados['MOB_ENTRADA__p_7_g_1'] == 0, base_dados['MOB_ENTRADA__C__p_5_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['MOB_ENTRADA__p_7_g_1'] == 0, base_dados['MOB_ENTRADA__C__p_5_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['MOB_ENTRADA__p_7_g_1'] == 1, base_dados['MOB_ENTRADA__C__p_5_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['MOB_ENTRADA__p_7_g_1'] == 1, base_dados['MOB_ENTRADA__C__p_5_g_1'] == 1), 2,
 0))))
base_dados['MOB_ENTRADA__C__p_5_g_1_c1_30_2'] = np.where(base_dados['MOB_ENTRADA__C__p_5_g_1_c1_30_1'] == 0, 0,
np.where(base_dados['MOB_ENTRADA__C__p_5_g_1_c1_30_1'] == 1, 1,
np.where(base_dados['MOB_ENTRADA__C__p_5_g_1_c1_30_1'] == 2, 2,
0)))
base_dados['MOB_ENTRADA__C__p_5_g_1_c1_30'] = np.where(base_dados['MOB_ENTRADA__C__p_5_g_1_c1_30_2'] == 0, 0,
np.where(base_dados['MOB_ENTRADA__C__p_5_g_1_c1_30_2'] == 1, 1,
np.where(base_dados['MOB_ENTRADA__C__p_5_g_1_c1_30_2'] == 2, 2,
 0)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Mantendo apenas as variáveis do modelo

# COMMAND ----------

import pickle
modelo=pickle.load(open(caminho_pickle_model, 'rb'))

base_teste_c0 = base_dados[[chave,'Telefone1_gh38','Qtde_Parcelas_Padrao_gh51','genero_gh38','Telefone2_gh38','Telefone3_gh38','Desconto_Padrao_gh38','rank_gh38','Indicador_Correntista_gh38','PRODUTO_gh51','Forma_Pgto_Padrao_gh38','Cep1__pu_8_g_1_c1_20','Cep2__L__p_15_g_1_c1_7','MOB_ENTRADA__C__p_5_g_1_c1_30','Contrato_Altair__pk_40_g_1_c1_4','Codigo_Politica__S__p_7_g_1_c1_8','Saldo_Devedor_Contrato__R__pk_17_g_1_c1_76']]

var_fin_c0=['Telefone1_gh38','Qtde_Parcelas_Padrao_gh51','genero_gh38','Telefone2_gh38','Telefone3_gh38','Desconto_Padrao_gh38','rank_gh38','Indicador_Correntista_gh38','PRODUTO_gh51','Forma_Pgto_Padrao_gh38','Cep1__pu_8_g_1_c1_20','Cep2__L__p_15_g_1_c1_7','MOB_ENTRADA__C__p_5_g_1_c1_30','Contrato_Altair__pk_40_g_1_c1_4','Codigo_Politica__S__p_7_g_1_c1_8','Saldo_Devedor_Contrato__R__pk_17_g_1_c1_76']

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

sparkDF.write.mode('overwrite').option("sep",";").option("header","true").csv(sink)

# COMMAND ----------

# MAGIC %md
# MAGIC # Modelo de Grupo Homogêneo

# COMMAND ----------


x_teste2['P_1_p_17_g_1'] = np.where(x_teste2['P_1'] <= 0.013223768, 1,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.013223768, x_teste2['P_1'] <= 0.02258316), 0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.02258316, x_teste2['P_1'] <= 0.044979021), 2,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.044979021, x_teste2['P_1'] <= 0.150817399), 3,4))))

x_teste2['P_1_pk_10_g_1'] = np.where(x_teste2['P_1'] <= 0.042628809, 0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.042628809, x_teste2['P_1'] <= 0.098146556), 1,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.098146556, x_teste2['P_1'] <= 0.169306338), 2,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.169306338, x_teste2['P_1'] <= 0.34179749), 3,4))))

x_teste2['GH'] = np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 0, x_teste2['P_1_pk_10_g_1'] == 0), 0,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 0, x_teste2['P_1_pk_10_g_1'] == 1), 0,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 0, x_teste2['P_1_pk_10_g_1'] == 2), 0,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 0, x_teste2['P_1_pk_10_g_1'] == 3), 1,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 0, x_teste2['P_1_pk_10_g_1'] == 4), 1,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 1, x_teste2['P_1_pk_10_g_1'] == 0), 1,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 1, x_teste2['P_1_pk_10_g_1'] == 1), 1,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 1, x_teste2['P_1_pk_10_g_1'] == 2), 1,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 1, x_teste2['P_1_pk_10_g_1'] == 3), 2,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 1, x_teste2['P_1_pk_10_g_1'] == 4), 2,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 2, x_teste2['P_1_pk_10_g_1'] == 0), 2,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 2, x_teste2['P_1_pk_10_g_1'] == 1), 2,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 2, x_teste2['P_1_pk_10_g_1'] == 2), 2,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 2, x_teste2['P_1_pk_10_g_1'] == 3), 3,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 2, x_teste2['P_1_pk_10_g_1'] == 4), 3,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 3, x_teste2['P_1_pk_10_g_1'] == 0), 3,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 3, x_teste2['P_1_pk_10_g_1'] == 1), 3,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 3, x_teste2['P_1_pk_10_g_1'] == 2), 4,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 3, x_teste2['P_1_pk_10_g_1'] == 3), 4,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 3, x_teste2['P_1_pk_10_g_1'] == 4), 5,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 4, x_teste2['P_1_pk_10_g_1'] == 0), 4,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 4, x_teste2['P_1_pk_10_g_1'] == 1), 4,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 4, x_teste2['P_1_pk_10_g_1'] == 2), 4,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 4, x_teste2['P_1_pk_10_g_1'] == 3), 5,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 4, x_teste2['P_1_pk_10_g_1'] == 4), 6,
             2)))))))))))))))))))))))))

del x_teste2['P_1_p_17_g_1']
del x_teste2['P_1_pk_10_g_1']

x_teste2


# COMMAND ----------

x_teste2.groupby(['GH'])['P_1'].count()

# COMMAND ----------

sparkDF=spark.createDataFrame(x_teste2) 

sparkDF.write.mode('overwrite').option("sep",";").option("header","true").option("mode","overwrite").csv(sinkHomomgeneous)

# COMMAND ----------

dbutils.notebook.exit('SUCCESS')