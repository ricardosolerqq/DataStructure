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
chave = 'ID_DEVEDOR'

#Nome da Base de Dados
N_Base = "amostra_santander_aleatoria_100.csv"

#Caminho da base de dados
caminho_base = "Base_Dados_Ferramenta/Santander/"

caminho_model_pickle = ""

#Separador
separador_ = ";"

#Decimal
decimal_ = "."


# COMMAND ----------

# MAGIC %md
# MAGIC ## Importação da Base de Dados

# COMMAND ----------

base_dados = pd.read_csv(caminho_base+N_Base, sep=separador_, decimal=decimal_)

base_dados['PRODUTO'] = np.where(base_dados['Descricao_Produto'].str.contains('CARTAO')==True,'CARTAO',
                                    np.where(base_dados['Descricao_Produto'].str.contains('MC')==True,'CARTAO',
                                    np.where(base_dados['Descricao_Produto'].str.contains('VS')==True,'CARTAO',
                                    np.where(base_dados['Descricao_Produto'].str.contains('MASTERCARD')==True,'CARTAO',
                                    np.where(base_dados['Descricao_Produto'].str.contains('VISA')==True,'CARTAO',
                                    np.where(base_dados['Descricao_Produto'].str.contains('CHEQUE')==True,'CHEQUE ESPECIAL',
                                    np.where(base_dados['Descricao_Produto'].str.contains('CREDITO')==True,'EMPRESTIMO',
                                    np.where(base_dados['Descricao_Produto'].str.contains('REFIN')==True,'REFIN','OUTROS'))))))))

base_dados['Telefone1'] = np.where(base_dados['Telefone1_skip:hot']==True,'skip_hot',
                                    np.where(base_dados['Telefone1_skip:alto']==True,'skip_alto',
                                    np.where(base_dados['Telefone1_skip:medio']==True,'skip_medio',
                                    np.where(base_dados['Telefone1_skip:baixo']==True,'skip_baixo',
                                    np.where(base_dados['Telefone1_skip:nhot']==True,'skip_nhot','sem_tags')))))

base_dados['Telefone3'] = np.where(base_dados['Telefone3_skip:hot']==True,'skip_hot',
                                    np.where(base_dados['Telefone3_skip:alto']==True,'skip_alto',
                                    np.where(base_dados['Telefone3_skip:medio']==True,'skip_medio',
                                    np.where(base_dados['Telefone3_skip:baixo']==True,'skip_baixo',
                                    np.where(base_dados['Telefone3_skip:nhot']==True,'skip_nhot','sem_tags')))))

base_dados['Telefone5'] = np.where(base_dados['Telefone5_skip:hot']==True,'skip_hot',
                                    np.where(base_dados['Telefone5_skip:alto']==True,'skip_alto',
                                    np.where(base_dados['Telefone5_skip:medio']==True,'skip_medio',
                                    np.where(base_dados['Telefone5_skip:baixo']==True,'skip_baixo',
                                    np.where(base_dados['Telefone5_skip:nhot']==True,'skip_nhot','sem_tags')))))

base_dados['ACORDO'] = base_dados['ACORDO'].map({True:1,False:0},na_action=None)
base_dados['Indicador_Correntista'] = base_dados['Indicador_Correntista'].map({True:1,False:0},na_action=None)
base_dados['rank:a'] = base_dados['rank:a'].map({True:1,False:0},na_action=None)
base_dados['rank:b'] = base_dados['rank:b'].map({True:1,False:0},na_action=None)
base_dados['rank:c'] = base_dados['rank:c'].map({True:1,False:0},na_action=None)
base_dados['rank:d'] = base_dados['rank:d'].map({True:1,False:0},na_action=None)

base_dados['DATA_ENTRADA_DIVIDA'] = pd.to_datetime(base_dados['DATA_ENTRADA_DIVIDA'])
base_dados['Contrato_Altair'] = base_dados['Contrato_Altair'].astype(np.int64)
base_dados['MOB_ENTRADA'] = ((datetime.today()) - base_dados.DATA_ENTRADA_DIVIDA)/np.timedelta64(1, 'M')

base_dados = base_dados.apply(pd.to_numeric, errors='ignore')

base_dados.fillna(-3)

list_v = ['ID_DEVEDOR','Telefone1','Telefone3','Telefone5','rank:a','rank:b','rank:c','rank:d','Indicador_Correntista','Dias_Atraso','Qtde_Max_Parcelas','Codigo_Politica','Contrato_Altair','Saldo_Devedor_Contrato','PRODUTO','MOB_ENTRADA']
base_dados = base_dados[list_v]

print("shape da Base de Dados:",base_dados.shape)

base_dados.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gerando variáveis Categóricas

# COMMAND ----------

base_dados['PRODUTO_gh30'] = np.where(base_dados['PRODUTO'] == 'CARTAO', 0,
    np.where(base_dados['PRODUTO'] == 'CHEQUE ESPECIAL', 1,
    np.where(base_dados['PRODUTO'] == 'EMPRESTIMO', 2,
    np.where(base_dados['PRODUTO'] == 'OUTROS', 3,
    np.where(base_dados['PRODUTO'] == 'REFIN', 4,0)))))

base_dados['PRODUTO_gh31'] = np.where(base_dados['PRODUTO_gh30'] == 0, 0,
    np.where(base_dados['PRODUTO_gh30'] == 1, 1,
    np.where(base_dados['PRODUTO_gh30'] == 2, 1,
    np.where(base_dados['PRODUTO_gh30'] == 3, 3,
    np.where(base_dados['PRODUTO_gh30'] == 4, 4,0)))))

base_dados['PRODUTO_gh32'] = np.where(base_dados['PRODUTO_gh31'] == 0, 0,
    np.where(base_dados['PRODUTO_gh31'] == 1, 1,
    np.where(base_dados['PRODUTO_gh31'] == 3, 2,
    np.where(base_dados['PRODUTO_gh31'] == 4, 3,0))))
             
base_dados['PRODUTO_gh33'] = np.where(base_dados['PRODUTO_gh32'] == 0, 0,
    np.where(base_dados['PRODUTO_gh32'] == 1, 1,
    np.where(base_dados['PRODUTO_gh32'] == 2, 2,
    np.where(base_dados['PRODUTO_gh32'] == 3, 3,0))))

base_dados['PRODUTO_gh34'] = np.where(base_dados['PRODUTO_gh33'] == 0, 0,
    np.where(base_dados['PRODUTO_gh33'] == 1, 1,
    np.where(base_dados['PRODUTO_gh33'] == 2, 2,
    np.where(base_dados['PRODUTO_gh33'] == 3, 3,0))))

base_dados['PRODUTO_gh35'] = np.where(base_dados['PRODUTO_gh34'] == 0, 0,
    np.where(base_dados['PRODUTO_gh34'] == 1, 1,
    np.where(base_dados['PRODUTO_gh34'] == 2, 2,
    np.where(base_dados['PRODUTO_gh34'] == 3, 3,0))))

base_dados['PRODUTO_gh36'] = np.where(base_dados['PRODUTO_gh35'] == 0, 2,
    np.where(base_dados['PRODUTO_gh35'] == 1, 1,
    np.where(base_dados['PRODUTO_gh35'] == 2, 0,
    np.where(base_dados['PRODUTO_gh35'] == 3, 2,0))))
             
base_dados['PRODUTO_gh37'] = np.where(base_dados['PRODUTO_gh36'] == 0, 1,
    np.where(base_dados['PRODUTO_gh36'] == 1, 1,
    np.where(base_dados['PRODUTO_gh36'] == 2, 2,0)))

base_dados['PRODUTO_gh38'] = np.where(base_dados['PRODUTO_gh37'] == 1, 0,
    np.where(base_dados['PRODUTO_gh37'] == 2, 1,1))
                                      
                                      
                                      
                                      
                                      
                                      
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

base_dados['Telefone1_gh34'] = np.where(base_dados['Telefone1_gh33'] == 0, 4,
    np.where(base_dados['Telefone1_gh33'] == 1, 1,
    np.where(base_dados['Telefone1_gh33'] == 2, 2,
    np.where(base_dados['Telefone1_gh33'] == 3, 3,
    np.where(base_dados['Telefone1_gh33'] == 4, 4,
    np.where(base_dados['Telefone1_gh33'] == 5, 5,
    0))))))

base_dados['Telefone1_gh35'] = np.where(base_dados['Telefone1_gh34'] == 1, 0,
    np.where(base_dados['Telefone1_gh34'] == 2, 1,
    np.where(base_dados['Telefone1_gh34'] == 3, 2,
    np.where(base_dados['Telefone1_gh34'] == 4, 3,
    np.where(base_dados['Telefone1_gh34'] == 5, 4,
    0)))))

base_dados['Telefone1_gh36'] = np.where(base_dados['Telefone1_gh35'] == 0, 3,
    np.where(base_dados['Telefone1_gh35'] == 1, 2,
    np.where(base_dados['Telefone1_gh35'] == 2, 4,
    np.where(base_dados['Telefone1_gh35'] == 3, 1,
    np.where(base_dados['Telefone1_gh35'] == 4, 0,
    0)))))

base_dados['Telefone1_gh37'] = np.where(base_dados['Telefone1_gh36'] == 0, 1,
    np.where(base_dados['Telefone1_gh36'] == 1, 1,
    np.where(base_dados['Telefone1_gh36'] == 2, 2,
    np.where(base_dados['Telefone1_gh36'] == 3, 3,
    np.where(base_dados['Telefone1_gh36'] == 4, 4,
    0)))))

base_dados['Telefone1_gh38'] = np.where(base_dados['Telefone1_gh37'] == 1, 0,
    np.where(base_dados['Telefone1_gh37'] == 2, 1,
    np.where(base_dados['Telefone1_gh37'] == 3, 2,
    np.where(base_dados['Telefone1_gh37'] == 4, 3,
    0))))
                                        
                                        
                                        
                                        
                                        
                                        
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
    np.where(base_dados['Telefone3_gh30'] == 5, 4,
    0))))))

base_dados['Telefone3_gh32'] = np.where(base_dados['Telefone3_gh31'] == 0, 0,
    np.where(base_dados['Telefone3_gh31'] == 1, 1,
    np.where(base_dados['Telefone3_gh31'] == 2, 2,
    np.where(base_dados['Telefone3_gh31'] == 3, 3,
    np.where(base_dados['Telefone3_gh31'] == 4, 4,
    0)))))

base_dados['Telefone3_gh33'] = np.where(base_dados['Telefone3_gh32'] == 0, 0,
    np.where(base_dados['Telefone3_gh32'] == 1, 1,
    np.where(base_dados['Telefone3_gh32'] == 2, 2,
    np.where(base_dados['Telefone3_gh32'] == 3, 3,
    np.where(base_dados['Telefone3_gh32'] == 4, 4,
    0)))))

base_dados['Telefone3_gh34'] = np.where(base_dados['Telefone3_gh33'] == 0, 0,
    np.where(base_dados['Telefone3_gh33'] == 1, 2,
    np.where(base_dados['Telefone3_gh33'] == 2, 2,
    np.where(base_dados['Telefone3_gh33'] == 3, 2,
    np.where(base_dados['Telefone3_gh33'] == 4, 4,
    0)))))

base_dados['Telefone3_gh35'] = np.where(base_dados['Telefone3_gh34'] == 0, 0,
    np.where(base_dados['Telefone3_gh34'] == 2, 1,
    np.where(base_dados['Telefone3_gh34'] == 4, 2,
    0)))

base_dados['Telefone3_gh36'] = np.where(base_dados['Telefone3_gh35'] == 0, 0,
    np.where(base_dados['Telefone3_gh35'] == 1, 2,
    np.where(base_dados['Telefone3_gh35'] == 2, 0,
    0)))

base_dados['Telefone3_gh37'] = np.where(base_dados['Telefone3_gh36'] == 0, 0,
    np.where(base_dados['Telefone3_gh36'] == 2, 1,
    0))
                                        
base_dados['Telefone3_gh38'] = np.where(base_dados['Telefone3_gh37'] == 0, 0,
    np.where(base_dados['Telefone3_gh37'] == 1, 1,
    0))
                                        
                                        
                                        
                                        
                                        
                                        
base_dados['Telefone5_gh30'] = np.where(base_dados['Telefone5'] == 'sem_tags', 0,
    np.where(base_dados['Telefone5'] == 'skip_alto', 1,
    np.where(base_dados['Telefone5'] == 'skip_baixo', 2,
    np.where(base_dados['Telefone5'] == 'skip_hot', 3,
    np.where(base_dados['Telefone5'] == 'skip_medio', 4,
    np.where(base_dados['Telefone5'] == 'skip_nhot', 5,
    0))))))

base_dados['Telefone5_gh31'] = np.where(base_dados['Telefone5_gh30'] == 0, 0,
    np.where(base_dados['Telefone5_gh30'] == 1, 1,
    np.where(base_dados['Telefone5_gh30'] == 2, 2,
    np.where(base_dados['Telefone5_gh30'] == 3, 3,
    np.where(base_dados['Telefone5_gh30'] == 4, 4,
    np.where(base_dados['Telefone5_gh30'] == 5, 4,
    0))))))

base_dados['Telefone5_gh32'] = np.where(base_dados['Telefone5_gh31'] == 0, 0,
    np.where(base_dados['Telefone5_gh31'] == 1, 1,
    np.where(base_dados['Telefone5_gh31'] == 2, 2,
    np.where(base_dados['Telefone5_gh31'] == 3, 3,
    np.where(base_dados['Telefone5_gh31'] == 4, 4,
    0)))))

base_dados['Telefone5_gh33'] = np.where(base_dados['Telefone5_gh32'] == 0, 0,
    np.where(base_dados['Telefone5_gh32'] == 1, 1,
    np.where(base_dados['Telefone5_gh32'] == 2, 2,
    np.where(base_dados['Telefone5_gh32'] == 3, 3,
    np.where(base_dados['Telefone5_gh32'] == 4, 4,
    0)))))

base_dados['Telefone5_gh34'] = np.where(base_dados['Telefone5_gh33'] == 0, 0,
    np.where(base_dados['Telefone5_gh33'] == 1, 2,
    np.where(base_dados['Telefone5_gh33'] == 2, 2,
    np.where(base_dados['Telefone5_gh33'] == 3, 0,
    np.where(base_dados['Telefone5_gh33'] == 4, 4,
    0)))))

base_dados['Telefone5_gh35'] = np.where(base_dados['Telefone5_gh34'] == 0, 0,
    np.where(base_dados['Telefone5_gh34'] == 2, 1,
    np.where(base_dados['Telefone5_gh34'] == 4, 2,
    0)))

base_dados['Telefone5_gh36'] = np.where(base_dados['Telefone5_gh35'] == 0, 2,
    np.where(base_dados['Telefone5_gh35'] == 1, 0,
    np.where(base_dados['Telefone5_gh35'] == 2, 0,
    0)))

base_dados['Telefone5_gh37'] = np.where(base_dados['Telefone5_gh36'] == 0, 0,
    np.where(base_dados['Telefone5_gh36'] == 2, 1,
    0))

base_dados['Telefone5_gh38'] = np.where(base_dados['Telefone5_gh37'] == 0, 0,
    np.where(base_dados['Telefone5_gh37'] == 1, 1,
    0))
                                        
                                        
                                        
                                        
                                        
                                        
                                        
base_dados['Qtde_Max_Parcelas_gh30'] = np.where(base_dados['Qtde_Max_Parcelas'] == 72, 0,
    np.where(base_dados['Qtde_Max_Parcelas'] == 96, 1,
    0))
                                                
base_dados['Qtde_Max_Parcelas_gh31'] = np.where(base_dados['Qtde_Max_Parcelas_gh30'] == 0, 0,
    np.where(base_dados['Qtde_Max_Parcelas_gh30'] == 1, 1,
    0))
                                                
base_dados['Qtde_Max_Parcelas_gh32'] = np.where(base_dados['Qtde_Max_Parcelas_gh31'] == 0, 0,
    np.where(base_dados['Qtde_Max_Parcelas_gh31'] == 1, 1,
    0))
                                                
base_dados['Qtde_Max_Parcelas_gh33'] = np.where(base_dados['Qtde_Max_Parcelas_gh32'] == 0, 0,
    np.where(base_dados['Qtde_Max_Parcelas_gh32'] == 1, 1,
    0))
                                                
base_dados['Qtde_Max_Parcelas_gh34'] = np.where(base_dados['Qtde_Max_Parcelas_gh33'] == 0, 0,
    np.where(base_dados['Qtde_Max_Parcelas_gh33'] == 1, 1,
    0))
                                                
base_dados['Qtde_Max_Parcelas_gh35'] = np.where(base_dados['Qtde_Max_Parcelas_gh34'] == 0, 0,
    np.where(base_dados['Qtde_Max_Parcelas_gh34'] == 1, 1,
    0))
                                                
base_dados['Qtde_Max_Parcelas_gh36'] = np.where(base_dados['Qtde_Max_Parcelas_gh35'] == 0, 1,
    np.where(base_dados['Qtde_Max_Parcelas_gh35'] == 1, 0,
    0))
                                                
base_dados['Qtde_Max_Parcelas_gh37'] = np.where(base_dados['Qtde_Max_Parcelas_gh36'] == 0, 0,
    np.where(base_dados['Qtde_Max_Parcelas_gh36'] == 1, 1,
    0))
                                                
base_dados['Qtde_Max_Parcelas_gh38'] = np.where(base_dados['Qtde_Max_Parcelas_gh37'] == 0, 0,
    np.where(base_dados['Qtde_Max_Parcelas_gh37'] == 1, 1,
    0))
                                                
                                                
                                                
                                                
                                                
                                                
base_dados['rank:c_gh30'] = np.where(base_dados['rank:c'] == 0, 0,
    np.where(base_dados['rank:c'] == 1, 1,
    0))
                                     
base_dados['rank:c_gh31'] = np.where(base_dados['rank:c_gh30'] == 0, 0,
    np.where(base_dados['rank:c_gh30'] == 1, 1,
    0))
                                     
base_dados['rank:c_gh32'] = np.where(base_dados['rank:c_gh31'] == 0, 0,
    np.where(base_dados['rank:c_gh31'] == 1, 1,
    0))
                                     
base_dados['rank:c_gh33'] = np.where(base_dados['rank:c_gh32'] == 0, 0,
    np.where(base_dados['rank:c_gh32'] == 1, 1,
    0))
                                     
base_dados['rank:c_gh34'] = np.where(base_dados['rank:c_gh33'] == 0, 0,
    np.where(base_dados['rank:c_gh33'] == 1, 1,
    0))
                                     
base_dados['rank:c_gh35'] = np.where(base_dados['rank:c_gh34'] == 0, 0,
    np.where(base_dados['rank:c_gh34'] == 1, 1,
    0))
                                     
base_dados['rank:c_gh36'] = np.where(base_dados['rank:c_gh35'] == 0, 1,
    np.where(base_dados['rank:c_gh35'] == 1, 0,
    0))
                                     
base_dados['rank:c_gh37'] = np.where(base_dados['rank:c_gh36'] == 0, 0,
    np.where(base_dados['rank:c_gh36'] == 1, 1,
    0))
                                     
base_dados['rank:c_gh38'] = np.where(base_dados['rank:c_gh37'] == 0, 0,
    np.where(base_dados['rank:c_gh37'] == 1, 1,
    0))
                                     
                                     
                                     
                                     
                                     
                                     
base_dados['rank:a_gh30'] = np.where(base_dados['rank:a'] == 0, 0,
    np.where(base_dados['rank:a'] == 1, 1,
    0))

base_dados['rank:a_gh31'] = np.where(base_dados['rank:a_gh30'] == 0, 0,
    np.where(base_dados['rank:a_gh30'] == 1, 1,
    0))

base_dados['rank:a_gh32'] = np.where(base_dados['rank:a_gh31'] == 0, 0,
    np.where(base_dados['rank:a_gh31'] == 1, 1,
    0))

base_dados['rank:a_gh33'] = np.where(base_dados['rank:a_gh32'] == 0, 0,
    np.where(base_dados['rank:a_gh32'] == 1, 1,
    0))

base_dados['rank:a_gh34'] = np.where(base_dados['rank:a_gh33'] == 0, 0,
    np.where(base_dados['rank:a_gh33'] == 1, 1,
    0))

base_dados['rank:a_gh35'] = np.where(base_dados['rank:a_gh34'] == 0, 0,
    np.where(base_dados['rank:a_gh34'] == 1, 1,
    0))

base_dados['rank:a_gh36'] = np.where(base_dados['rank:a_gh35'] == 0, 0,
    np.where(base_dados['rank:a_gh35'] == 1, 1,
    0))

base_dados['rank:a_gh37'] = np.where(base_dados['rank:a_gh36'] == 0, 0,
    np.where(base_dados['rank:a_gh36'] == 1, 1,
    0))

base_dados['rank:a_gh38'] = np.where(base_dados['rank:a_gh37'] == 0, 0,
    np.where(base_dados['rank:a_gh37'] == 1, 1,
    0))
                                     
                                     
                                     
                                     
                                     
                                     
base_dados['rank:b_gh30'] = np.where(base_dados['rank:b'] == 0, 0,
    np.where(base_dados['rank:b'] == 1, 1,
    0))

base_dados['rank:b_gh31'] = np.where(base_dados['rank:b_gh30'] == 0, 0,
    np.where(base_dados['rank:b_gh30'] == 1, 1,
    0))

base_dados['rank:b_gh32'] = np.where(base_dados['rank:b_gh31'] == 0, 0,
    np.where(base_dados['rank:b_gh31'] == 1, 1,
    0))

base_dados['rank:b_gh33'] = np.where(base_dados['rank:b_gh32'] == 0, 0,
    np.where(base_dados['rank:b_gh32'] == 1, 1,
    0))

base_dados['rank:b_gh34'] = np.where(base_dados['rank:b_gh33'] == 0, 0,
    np.where(base_dados['rank:b_gh33'] == 1, 1,
    0))

base_dados['rank:b_gh35'] = np.where(base_dados['rank:b_gh34'] == 0, 0,
    np.where(base_dados['rank:b_gh34'] == 1, 1,
    0))

base_dados['rank:b_gh36'] = np.where(base_dados['rank:b_gh35'] == 0, 0,
    np.where(base_dados['rank:b_gh35'] == 1, 1,
    0))

base_dados['rank:b_gh37'] = np.where(base_dados['rank:b_gh36'] == 0, 0,
    np.where(base_dados['rank:b_gh36'] == 1, 1,
    0))

base_dados['rank:b_gh38'] = np.where(base_dados['rank:b_gh37'] == 0, 0,
    np.where(base_dados['rank:b_gh37'] == 1, 1,
    0))

                                     
                                     
                                     
                                     
                                     
                                     
base_dados['rank:d_gh30'] = np.where(base_dados['rank:d'] == 0, 0,
    np.where(base_dados['rank:d'] == 1, 1,
    0))

base_dados['rank:d_gh31'] = np.where(base_dados['rank:d_gh30'] == 0, 0,
    np.where(base_dados['rank:d_gh30'] == 1, 1,
    0))

base_dados['rank:d_gh32'] = np.where(base_dados['rank:d_gh31'] == 0, 0,
    np.where(base_dados['rank:d_gh31'] == 1, 1,
    0))

base_dados['rank:d_gh33'] = np.where(base_dados['rank:d_gh32'] == 0, 0,
    np.where(base_dados['rank:d_gh32'] == 1, 1,
    0))

base_dados['rank:d_gh34'] = np.where(base_dados['rank:d_gh33'] == 0, 0,
    np.where(base_dados['rank:d_gh33'] == 1, 1,
    0))

base_dados['rank:d_gh35'] = np.where(base_dados['rank:d_gh34'] == 0, 0,
    np.where(base_dados['rank:d_gh34'] == 1, 1,
    0))

base_dados['rank:d_gh36'] = np.where(base_dados['rank:d_gh35'] == 0, 1,
    np.where(base_dados['rank:d_gh35'] == 1, 0,
    0))

base_dados['rank:d_gh37'] = np.where(base_dados['rank:d_gh36'] == 0, 0,
    np.where(base_dados['rank:d_gh36'] == 1, 1,
    0))

base_dados['rank:d_gh38'] = np.where(base_dados['rank:d_gh37'] == 0, 0,
    np.where(base_dados['rank:d_gh37'] == 1, 1,
    0))
                                     
                                     
                                     
                                     
                                     
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


# COMMAND ----------

# MAGIC %md
# MAGIC ## Gerando variáveis numéricas contínuas

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parte 1 de 2

# COMMAND ----------



base_dados['Codigo_Politica__L'] = np.log(base_dados['Codigo_Politica'])
np.where(base_dados['Codigo_Politica__L'] == 0, -1, base_dados['Codigo_Politica__L'])
base_dados['Codigo_Politica__L'] = base_dados['Codigo_Politica__L'].fillna(-2)

base_dados['Codigo_Politica__L__p_6'] = np.where(base_dados['Codigo_Politica__L'] <= 4.955827057601261, 0.0,
    np.where(np.bitwise_and(base_dados['Codigo_Politica__L'] > 4.955827057601261, base_dados['Codigo_Politica__L'] <= 5.308267697401205), 1.0,
    np.where(np.bitwise_and(base_dados['Codigo_Politica__L'] > 5.308267697401205, base_dados['Codigo_Politica__L'] <= 5.575949103146316), 3.0,
    np.where(np.bitwise_and(base_dados['Codigo_Politica__L'] > 5.575949103146316, base_dados['Codigo_Politica__L'] <= 5.831882477283517), 4.0,
    np.where(base_dados['Codigo_Politica__L'] > 5.831882477283517, 5.0,
     0)))))
         
base_dados['Codigo_Politica__L__p_6_g_1_1'] = np.where(base_dados['Codigo_Politica__L__p_6'] == 0, 1,
    np.where(base_dados['Codigo_Politica__L__p_6'] == 1, 2,
    np.where(base_dados['Codigo_Politica__L__p_6'] == 3, 0,
    np.where(base_dados['Codigo_Politica__L__p_6'] == 4, 2,
    np.where(base_dados['Codigo_Politica__L__p_6'] == 5, 3,
     0)))))
         
base_dados['Codigo_Politica__L__p_6_g_1_2'] = np.where(base_dados['Codigo_Politica__L__p_6_g_1_1'] == 0, 0,
    np.where(base_dados['Codigo_Politica__L__p_6_g_1_1'] == 1, 2,
    np.where(base_dados['Codigo_Politica__L__p_6_g_1_1'] == 2, 1,
    np.where(base_dados['Codigo_Politica__L__p_6_g_1_1'] == 3, 3,
     0))))
         
base_dados['Codigo_Politica__L__p_6_g_1'] = np.where(base_dados['Codigo_Politica__L__p_6_g_1_2'] == 0, 0,
    np.where(base_dados['Codigo_Politica__L__p_6_g_1_2'] == 1, 1,
    np.where(base_dados['Codigo_Politica__L__p_6_g_1_2'] == 2, 2,
    np.where(base_dados['Codigo_Politica__L__p_6_g_1_2'] == 3, 3,
     0))))
         
         
         
         
              
base_dados['Codigo_Politica__L'] = np.log(base_dados['Codigo_Politica'])
np.where(base_dados['Codigo_Politica__L'] == 0, -1, base_dados['Codigo_Politica__L'])
base_dados['Codigo_Politica__L'] = base_dados['Codigo_Politica__L'].fillna(-2)

base_dados['Codigo_Politica__L__p_4'] = np.where(base_dados['Codigo_Politica__L'] <= 5.288267030694535, 0.0,
    np.where(np.bitwise_and(base_dados['Codigo_Politica__L'] > 5.288267030694535, base_dados['Codigo_Politica__L'] <= 5.3471075307174685), 1.0,
    np.where(np.bitwise_and(base_dados['Codigo_Politica__L'] > 5.3471075307174685, base_dados['Codigo_Politica__L'] <= 5.8289456176102075), 2.0,
    np.where(base_dados['Codigo_Politica__L'] > 5.8289456176102075, 3.0,
     0))))

base_dados['Codigo_Politica__L__p_4_g_1_1'] = np.where(base_dados['Codigo_Politica__L__p_4'] == 0, 1,
    np.where(base_dados['Codigo_Politica__L__p_4'] == 1, 0,
    np.where(base_dados['Codigo_Politica__L__p_4'] == 2, 2,
    np.where(base_dados['Codigo_Politica__L__p_4'] == 3, 3,
     0))))

base_dados['Codigo_Politica__L__p_4_g_1_2'] = np.where(base_dados['Codigo_Politica__L__p_4_g_1_1'] == 0, 0,
    np.where(base_dados['Codigo_Politica__L__p_4_g_1_1'] == 1, 2,
    np.where(base_dados['Codigo_Politica__L__p_4_g_1_1'] == 2, 1,
    np.where(base_dados['Codigo_Politica__L__p_4_g_1_1'] == 3, 2,
     0))))

base_dados['Codigo_Politica__L__p_4_g_1'] = np.where(base_dados['Codigo_Politica__L__p_4_g_1_2'] == 0, 0,
    np.where(base_dados['Codigo_Politica__L__p_4_g_1_2'] == 1, 1,
    np.where(base_dados['Codigo_Politica__L__p_4_g_1_2'] == 2, 2,
     0)))
         
         
         
               
base_dados['Saldo_Devedor_Contrato__R'] = np.sqrt(base_dados['Saldo_Devedor_Contrato'])
np.where(base_dados['Saldo_Devedor_Contrato__R'] == 0, -1, base_dados['Saldo_Devedor_Contrato__R'])
base_dados['Saldo_Devedor_Contrato__R'] = base_dados['Saldo_Devedor_Contrato__R'].fillna(-2)

base_dados['Saldo_Devedor_Contrato__R__pe_20'] = np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R'] >= -1.0, base_dados['Saldo_Devedor_Contrato__R'] <= 11.095043938624128), 0.0,
    np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R'] > 11.095043938624128, base_dados['Saldo_Devedor_Contrato__R'] <= 22.20022522408275), 1.0,
    np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R'] > 22.20022522408275, base_dados['Saldo_Devedor_Contrato__R'] <= 33.297297187609686), 2.0,
    np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R'] > 33.297297187609686, base_dados['Saldo_Devedor_Contrato__R'] <= 44.40304043643859), 3.0,
    np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R'] > 44.40304043643859, base_dados['Saldo_Devedor_Contrato__R'] <= 55.499009000161436), 4.0,
    np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R'] > 55.499009000161436, base_dados['Saldo_Devedor_Contrato__R'] <= 66.59707200770917), 5.0,
    np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R'] > 66.59707200770917, base_dados['Saldo_Devedor_Contrato__R'] <= 77.69144097003222), 6.0,
    np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R'] > 77.69144097003222, base_dados['Saldo_Devedor_Contrato__R'] <= 88.80011261254121), 7.0,
    np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R'] > 88.80011261254121, base_dados['Saldo_Devedor_Contrato__R'] <= 99.90030029984895), 8.0,
    np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R'] > 99.90030029984895, base_dados['Saldo_Devedor_Contrato__R'] <= 110.93123996422288), 9.0,
    np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R'] > 110.93123996422288, base_dados['Saldo_Devedor_Contrato__R'] <= 121.99610649524844), 10.0,
    np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R'] > 121.99610649524844, base_dados['Saldo_Devedor_Contrato__R'] <= 133.19260489982167), 11.0,
    np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R'] > 133.19260489982167, base_dados['Saldo_Devedor_Contrato__R'] <= 144.30027720001095), 12.0,
    np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R'] > 144.30027720001095, base_dados['Saldo_Devedor_Contrato__R'] <= 155.36029093690576), 13.0,
    np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R'] > 155.36029093690576, base_dados['Saldo_Devedor_Contrato__R'] <= 166.38846113838542), 14.0,
    np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R'] > 166.38846113838542, base_dados['Saldo_Devedor_Contrato__R'] <= 177.3372493301957), 15.0,
    np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R'] > 177.3372493301957, base_dados['Saldo_Devedor_Contrato__R'] <= 188.64967532439593), 16.0,
    np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R'] > 188.64967532439593, base_dados['Saldo_Devedor_Contrato__R'] <= 199.70140209823265), 17.0,
    np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R'] > 199.70140209823265, base_dados['Saldo_Devedor_Contrato__R'] <= 210.88506348245718), 18.0,
    np.where(base_dados['Saldo_Devedor_Contrato__R'] > 210.88506348245718, 19.0,
     -2))))))))))))))))))))
             
base_dados['Saldo_Devedor_Contrato__R__pe_20_g_1_1'] = np.where(base_dados['Saldo_Devedor_Contrato__R__pe_20'] == -2.0, 5,
    np.where(base_dados['Saldo_Devedor_Contrato__R__pe_20'] == 0.0, 3,
    np.where(base_dados['Saldo_Devedor_Contrato__R__pe_20'] == 1.0, 2,
    np.where(base_dados['Saldo_Devedor_Contrato__R__pe_20'] == 2.0, 0,
    np.where(base_dados['Saldo_Devedor_Contrato__R__pe_20'] == 3.0, 1,
    np.where(base_dados['Saldo_Devedor_Contrato__R__pe_20'] == 4.0, 1,
    np.where(base_dados['Saldo_Devedor_Contrato__R__pe_20'] == 5.0, 3,
    np.where(base_dados['Saldo_Devedor_Contrato__R__pe_20'] == 6.0, 3,
    np.where(base_dados['Saldo_Devedor_Contrato__R__pe_20'] == 7.0, 3,
    np.where(base_dados['Saldo_Devedor_Contrato__R__pe_20'] == 8.0, 4,
    np.where(base_dados['Saldo_Devedor_Contrato__R__pe_20'] == 9.0, 4,
    np.where(base_dados['Saldo_Devedor_Contrato__R__pe_20'] == 10.0, 3,
    np.where(base_dados['Saldo_Devedor_Contrato__R__pe_20'] == 11.0, 5,
    np.where(base_dados['Saldo_Devedor_Contrato__R__pe_20'] == 12.0, 5,
    np.where(base_dados['Saldo_Devedor_Contrato__R__pe_20'] == 13.0, 3,
    np.where(base_dados['Saldo_Devedor_Contrato__R__pe_20'] == 14.0, 5,
    np.where(base_dados['Saldo_Devedor_Contrato__R__pe_20'] == 15.0, 5,
    np.where(base_dados['Saldo_Devedor_Contrato__R__pe_20'] == 16.0, 5,
    np.where(base_dados['Saldo_Devedor_Contrato__R__pe_20'] == 17.0, 5,
    np.where(base_dados['Saldo_Devedor_Contrato__R__pe_20'] == 18.0, 5,
    np.where(base_dados['Saldo_Devedor_Contrato__R__pe_20'] == 19.0, 5,
     0)))))))))))))))))))))
         
base_dados['Saldo_Devedor_Contrato__R__pe_20_g_1_2'] = np.where(base_dados['Saldo_Devedor_Contrato__R__pe_20_g_1_1'] == 0, 4,
    np.where(base_dados['Saldo_Devedor_Contrato__R__pe_20_g_1_1'] == 1, 3,
    np.where(base_dados['Saldo_Devedor_Contrato__R__pe_20_g_1_1'] == 2, 4,
    np.where(base_dados['Saldo_Devedor_Contrato__R__pe_20_g_1_1'] == 3, 2,
    np.where(base_dados['Saldo_Devedor_Contrato__R__pe_20_g_1_1'] == 4, 1,
    np.where(base_dados['Saldo_Devedor_Contrato__R__pe_20_g_1_1'] == 5, 0,
     0))))))
         
base_dados['Saldo_Devedor_Contrato__R__pe_20_g_1'] = np.where(base_dados['Saldo_Devedor_Contrato__R__pe_20_g_1_2'] == 0, 0,
    np.where(base_dados['Saldo_Devedor_Contrato__R__pe_20_g_1_2'] == 1, 1,
    np.where(base_dados['Saldo_Devedor_Contrato__R__pe_20_g_1_2'] == 2, 2,
    np.where(base_dados['Saldo_Devedor_Contrato__R__pe_20_g_1_2'] == 3, 3,
    np.where(base_dados['Saldo_Devedor_Contrato__R__pe_20_g_1_2'] == 4, 4,
     0)))))
         
         
         
         
         
base_dados['Saldo_Devedor_Contrato__R'] = np.sqrt(base_dados['Saldo_Devedor_Contrato'])
np.where(base_dados['Saldo_Devedor_Contrato__R'] == 0, -1, base_dados['Saldo_Devedor_Contrato__R'])
base_dados['Saldo_Devedor_Contrato__R'] = base_dados['Saldo_Devedor_Contrato__R'].fillna(-2)

base_dados['Saldo_Devedor_Contrato__R__pe_5'] = np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R'] >= -1.0, base_dados['Saldo_Devedor_Contrato__R'] <= 44.40304043643859), 0.0,
    np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R'] > 44.40304043643859, base_dados['Saldo_Devedor_Contrato__R'] <= 88.80011261254121), 1.0,
    np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R'] > 88.80011261254121, base_dados['Saldo_Devedor_Contrato__R'] <= 133.19260489982167), 2.0,
    np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R'] > 133.19260489982167, base_dados['Saldo_Devedor_Contrato__R'] <= 177.3372493301957), 3.0,
    np.where(base_dados['Saldo_Devedor_Contrato__R'] > 177.3372493301957, 4.0,
     -2)))))

base_dados['Saldo_Devedor_Contrato__R__pe_5_g_1_1'] = np.where(base_dados['Saldo_Devedor_Contrato__R__pe_5'] == -2.0, 2,
    np.where(base_dados['Saldo_Devedor_Contrato__R__pe_5'] == 0.0, 0,
    np.where(base_dados['Saldo_Devedor_Contrato__R__pe_5'] == 1.0, 1,
    np.where(base_dados['Saldo_Devedor_Contrato__R__pe_5'] == 2.0, 2,
    np.where(base_dados['Saldo_Devedor_Contrato__R__pe_5'] == 3.0, 2,
    np.where(base_dados['Saldo_Devedor_Contrato__R__pe_5'] == 4.0, 2,
     0))))))

base_dados['Saldo_Devedor_Contrato__R__pe_5_g_1_2'] = np.where(base_dados['Saldo_Devedor_Contrato__R__pe_5_g_1_1'] == 0, 2,
    np.where(base_dados['Saldo_Devedor_Contrato__R__pe_5_g_1_1'] == 1, 1,
    np.where(base_dados['Saldo_Devedor_Contrato__R__pe_5_g_1_1'] == 2, 0,
     0)))
             
base_dados['Saldo_Devedor_Contrato__R__pe_5_g_1'] = np.where(base_dados['Saldo_Devedor_Contrato__R__pe_5_g_1_2'] == 0, 0,
    np.where(base_dados['Saldo_Devedor_Contrato__R__pe_5_g_1_2'] == 1, 1,
    np.where(base_dados['Saldo_Devedor_Contrato__R__pe_5_g_1_2'] == 2, 2,
     0)))
         
         
         
         
         
base_dados['MOB_ENTRADA__pe_4'] = np.where(np.bitwise_and(base_dados['MOB_ENTRADA'] >= -1.2156222400087309, base_dados['MOB_ENTRADA'] <= 5.158225272260515), 0.0,
    np.where(np.bitwise_and(base_dados['MOB_ENTRADA'] > 5.158225272260515, base_dados['MOB_ENTRADA'] <= 10.349296957510726), 1.0,
    np.where(np.bitwise_and(base_dados['MOB_ENTRADA'] > 10.349296957510726, base_dados['MOB_ENTRADA'] <= 15.474658874593212), 2.0,
    np.where(base_dados['MOB_ENTRADA'] > 15.474658874593212, 3.0,
     -2))))
             
base_dados['MOB_ENTRADA__pe_4_g_1_1'] = np.where(base_dados['MOB_ENTRADA__pe_4'] == -2.0, 0,
    np.where(base_dados['MOB_ENTRADA__pe_4'] == 0.0, 1,
    np.where(base_dados['MOB_ENTRADA__pe_4'] == 1.0, 1,
    np.where(base_dados['MOB_ENTRADA__pe_4'] == 2.0, 0,
    np.where(base_dados['MOB_ENTRADA__pe_4'] == 3.0, 1,
     0)))))
             
base_dados['MOB_ENTRADA__pe_4_g_1_2'] = np.where(base_dados['MOB_ENTRADA__pe_4_g_1_1'] == 0, 0,
    np.where(base_dados['MOB_ENTRADA__pe_4_g_1_1'] == 1, 1,
     0))

base_dados['MOB_ENTRADA__pe_4_g_1'] = np.where(base_dados['MOB_ENTRADA__pe_4_g_1_2'] == 0, 0,
    np.where(base_dados['MOB_ENTRADA__pe_4_g_1_2'] == 1, 1,
     0))
                                               
                                               
                                               
                                               
                                               
                                               
base_dados['MOB_ENTRADA__L'] = np.log(base_dados['MOB_ENTRADA'])
np.where(base_dados['MOB_ENTRADA__L'] == 0, -1, base_dados['MOB_ENTRADA__L'])
base_dados['MOB_ENTRADA__L'] = base_dados['MOB_ENTRADA__L'].fillna(-2)

base_dados['MOB_ENTRADA__L__pe_15'] = np.where(np.bitwise_and(base_dados['MOB_ENTRADA__L'] >= -2.0, base_dados['MOB_ENTRADA__L'] <= 0.59168300635466), 2.0,
    np.where(np.bitwise_and(base_dados['MOB_ENTRADA__L'] > 0.59168300635466, base_dados['MOB_ENTRADA__L'] <= 0.6447925891929438), 3.0,
    np.where(np.bitwise_and(base_dados['MOB_ENTRADA__L'] > 0.6447925891929438, base_dados['MOB_ENTRADA__L'] <= 1.0386954275634461), 4.0,
    np.where(np.bitwise_and(base_dados['MOB_ENTRADA__L'] > 1.0386954275634461, base_dados['MOB_ENTRADA__L'] <= 1.072984399983988), 5.0,
    np.where(np.bitwise_and(base_dados['MOB_ENTRADA__L'] > 1.072984399983988, base_dados['MOB_ENTRADA__L'] <= 1.4671490091184891), 6.0,
    np.where(np.bitwise_and(base_dados['MOB_ENTRADA__L'] > 1.4671490091184891, base_dados['MOB_ENTRADA__L'] <= 1.6842131332281869), 7.0,
    np.where(np.bitwise_and(base_dados['MOB_ENTRADA__L'] > 1.6842131332281869, base_dados['MOB_ENTRADA__L'] <= 1.8776512536189742), 8.0,
    np.where(np.bitwise_and(base_dados['MOB_ENTRADA__L'] > 1.8776512536189742, base_dados['MOB_ENTRADA__L'] <= 2.101799065189532), 9.0,
    np.where(np.bitwise_and(base_dados['MOB_ENTRADA__L'] > 2.101799065189532, base_dados['MOB_ENTRADA__L'] <= 2.314445753344331), 10.0,
    np.where(np.bitwise_and(base_dados['MOB_ENTRADA__L'] > 2.314445753344331, base_dados['MOB_ENTRADA__L'] <= 2.524517064478683), 11.0,
    np.where(np.bitwise_and(base_dados['MOB_ENTRADA__L'] > 2.524517064478683, base_dados['MOB_ENTRADA__L'] <= 2.7370783765350724), 12.0,
    np.where(np.bitwise_and(base_dados['MOB_ENTRADA__L'] > 2.7370783765350724, base_dados['MOB_ENTRADA__L'] <= 2.9473736813300326), 13.0,
    np.where(base_dados['MOB_ENTRADA__L'] > 2.9473736813300326, 14.0,
     -2)))))))))))))
             
base_dados['MOB_ENTRADA__L__pe_15_g_1_1'] = np.where(base_dados['MOB_ENTRADA__L__pe_15'] == -2.0, 2,
    np.where(base_dados['MOB_ENTRADA__L__pe_15'] == 2.0, 2,
    np.where(base_dados['MOB_ENTRADA__L__pe_15'] == 3.0, 2,
    np.where(base_dados['MOB_ENTRADA__L__pe_15'] == 4.0, 2,
    np.where(base_dados['MOB_ENTRADA__L__pe_15'] == 5.0, 2,
    np.where(base_dados['MOB_ENTRADA__L__pe_15'] == 6.0, 2,
    np.where(base_dados['MOB_ENTRADA__L__pe_15'] == 7.0, 2,
    np.where(base_dados['MOB_ENTRADA__L__pe_15'] == 8.0, 2,
    np.where(base_dados['MOB_ENTRADA__L__pe_15'] == 9.0, 2,
    np.where(base_dados['MOB_ENTRADA__L__pe_15'] == 10.0, 2,
    np.where(base_dados['MOB_ENTRADA__L__pe_15'] == 11.0, 2,
    np.where(base_dados['MOB_ENTRADA__L__pe_15'] == 12.0, 1,
    np.where(base_dados['MOB_ENTRADA__L__pe_15'] == 13.0, 0,
    np.where(base_dados['MOB_ENTRADA__L__pe_15'] == 14.0, 1,
     0))))))))))))))
             
base_dados['MOB_ENTRADA__L__pe_15_g_1_2'] = np.where(base_dados['MOB_ENTRADA__L__pe_15_g_1_1'] == 0, 0,
    np.where(base_dados['MOB_ENTRADA__L__pe_15_g_1_1'] == 1, 1,
    np.where(base_dados['MOB_ENTRADA__L__pe_15_g_1_1'] == 2, 2,
     0)))

base_dados['MOB_ENTRADA__L__pe_15_g_1'] = np.where(base_dados['MOB_ENTRADA__L__pe_15_g_1_2'] == 0, 0,
    np.where(base_dados['MOB_ENTRADA__L__pe_15_g_1_2'] == 1, 1,
    np.where(base_dados['MOB_ENTRADA__L__pe_15_g_1_2'] == 2, 2,
     0)))
         
         
         
         
                 
base_dados['Contrato_Altair__p_6'] = np.where(base_dados['Contrato_Altair'] <= 69900132840.0, 0.0,
    np.where(np.bitwise_and(base_dados['Contrato_Altair'] > 69900132840.0, base_dados['Contrato_Altair'] <= 669992000000.0), 1.0,
    np.where(np.bitwise_and(base_dados['Contrato_Altair'] > 669992000000.0, base_dados['Contrato_Altair'] <= 698200000000.0), 2.0,
     -2)))
             
base_dados['Contrato_Altair__p_6_g_1_1'] = np.where(base_dados['Contrato_Altair__p_6'] == -2.0, 2,
    np.where(base_dados['Contrato_Altair__p_6'] == 0.0, 3,
    np.where(base_dados['Contrato_Altair__p_6'] == 1.0, 1,
    np.where(base_dados['Contrato_Altair__p_6'] == 2.0, 0,
     0))))
             
base_dados['Contrato_Altair__p_6_g_1_2'] = np.where(base_dados['Contrato_Altair__p_6_g_1_1'] == 0, 2,
    np.where(base_dados['Contrato_Altair__p_6_g_1_1'] == 1, 1,
    np.where(base_dados['Contrato_Altair__p_6_g_1_1'] == 2, 3,
    np.where(base_dados['Contrato_Altair__p_6_g_1_1'] == 3, 0,
     0))))
             
base_dados['Contrato_Altair__p_6_g_1'] = np.where(base_dados['Contrato_Altair__p_6_g_1_2'] == 0, 0,
    np.where(base_dados['Contrato_Altair__p_6_g_1_2'] == 1, 1,
    np.where(base_dados['Contrato_Altair__p_6_g_1_2'] == 2, 2,
    np.where(base_dados['Contrato_Altair__p_6_g_1_2'] == 3, 3,
     0))))
         
         
         
         
              
base_dados['Contrato_Altair__C'] = np.cos(base_dados['Contrato_Altair'])
np.where(base_dados['Contrato_Altair__C'] == 0, -1, base_dados['Contrato_Altair__C'])
base_dados['Contrato_Altair__C'] = base_dados['Contrato_Altair__C'].fillna(-2)

base_dados['Contrato_Altair__C__pe_3'] = np.where(np.bitwise_and(base_dados['Contrato_Altair__C'] >= -0.9999999920637928, base_dados['Contrato_Altair__C'] <= 0.6591330859869091), 0.0,
    np.where(base_dados['Contrato_Altair__C'] > 0.6591330859869091, 1.0,
     -2))

base_dados['Contrato_Altair__C__pe_3_g_1_1'] = np.where(base_dados['Contrato_Altair__C__pe_3'] == -2.0, 0,
    np.where(base_dados['Contrato_Altair__C__pe_3'] == 0.0, 2,
    np.where(base_dados['Contrato_Altair__C__pe_3'] == 1.0, 1,
     0)))

base_dados['Contrato_Altair__C__pe_3_g_1_2'] = np.where(base_dados['Contrato_Altair__C__pe_3_g_1_1'] == 0, 1,
    np.where(base_dados['Contrato_Altair__C__pe_3_g_1_1'] == 1, 0,
    np.where(base_dados['Contrato_Altair__C__pe_3_g_1_1'] == 2, 1,
     0)))

base_dados['Contrato_Altair__C__pe_3_g_1'] = np.where(base_dados['Contrato_Altair__C__pe_3_g_1_2'] == 0, 0,
    np.where(base_dados['Contrato_Altair__C__pe_3_g_1_2'] == 1, 1,
     0))
                                                      
                                                      
                                                      
                                                      
                                                      
                                                      
base_dados['Dias_Atraso__pe_7'] = np.where(base_dados['Dias_Atraso'] == 0 , -1.0,
    np.where(np.bitwise_and(base_dados['Dias_Atraso'] > 0.0, base_dados['Dias_Atraso'] <= 582.0), 0.0,
    np.where(np.bitwise_and(base_dados['Dias_Atraso'] > 582.0, base_dados['Dias_Atraso'] <= 1164.0), 1.0,
    np.where(np.bitwise_and(base_dados['Dias_Atraso'] > 1164.0, base_dados['Dias_Atraso'] <= 1745.0), 2.0,
    np.where(np.bitwise_and(base_dados['Dias_Atraso'] > 1745.0, base_dados['Dias_Atraso'] <= 2327.0), 3.0,
    np.where(np.bitwise_and(base_dados['Dias_Atraso'] > 2327.0, base_dados['Dias_Atraso'] <= 2910.0), 4.0,
    np.where(np.bitwise_and(base_dados['Dias_Atraso'] > 2910.0, base_dados['Dias_Atraso'] <= 3493.0), 5.0,
    np.where(np.bitwise_and(base_dados['Dias_Atraso'] > 3493.0, base_dados['Dias_Atraso'] <= 4076.0), 6.0,
     -2))))))))
             
base_dados['Dias_Atraso__pe_7_g_1_1'] = np.where(base_dados['Dias_Atraso__pe_7'] == -2.0, 3,
    np.where(base_dados['Dias_Atraso__pe_7'] == -1.0, 3,
    np.where(base_dados['Dias_Atraso__pe_7'] == 0.0, 2,
    np.where(base_dados['Dias_Atraso__pe_7'] == 1.0, 0,
    np.where(base_dados['Dias_Atraso__pe_7'] == 2.0, 1,
    np.where(base_dados['Dias_Atraso__pe_7'] == 3.0, 3,
    np.where(base_dados['Dias_Atraso__pe_7'] == 4.0, 3,
    np.where(base_dados['Dias_Atraso__pe_7'] == 5.0, 3,
    np.where(base_dados['Dias_Atraso__pe_7'] == 6.0, 3,
     0)))))))))
             
base_dados['Dias_Atraso__pe_7_g_1_2'] = np.where(base_dados['Dias_Atraso__pe_7_g_1_1'] == 0, 2,
    np.where(base_dados['Dias_Atraso__pe_7_g_1_1'] == 1, 1,
    np.where(base_dados['Dias_Atraso__pe_7_g_1_1'] == 2, 3,
    np.where(base_dados['Dias_Atraso__pe_7_g_1_1'] == 3, 0,
     0))))

base_dados['Dias_Atraso__pe_7_g_1'] = np.where(base_dados['Dias_Atraso__pe_7_g_1_2'] == 0, 0,
    np.where(base_dados['Dias_Atraso__pe_7_g_1_2'] == 1, 1,
    np.where(base_dados['Dias_Atraso__pe_7_g_1_2'] == 2, 2,
    np.where(base_dados['Dias_Atraso__pe_7_g_1_2'] == 3, 3,
     0))))
         
         
         
         
         
        
base_dados['Dias_Atraso__L'] = np.log(base_dados['Dias_Atraso'])
np.where(base_dados['Dias_Atraso__L'] == 0, -1, base_dados['Dias_Atraso__L'])
base_dados['Dias_Atraso__L'] = base_dados['Dias_Atraso__L'].fillna(-2)

base_dados['Dias_Atraso__L__p_3'] = np.where(np.bitwise_and(base_dados['Dias_Atraso__L'] >= -1.0, base_dados['Dias_Atraso__L'] <= 6.089044875446846), 0.0,
    np.where(np.bitwise_and(base_dados['Dias_Atraso__L'] > 6.089044875446846, base_dados['Dias_Atraso__L'] <= 6.841615476477592), 1.0,
    np.where(base_dados['Dias_Atraso__L'] > 6.841615476477592, 2.0,
     -2)))

base_dados['Dias_Atraso__L__p_3_g_1_1'] = np.where(base_dados['Dias_Atraso__L__p_3'] == -2.0, 2,
    np.where(base_dados['Dias_Atraso__L__p_3'] == 0.0, 2,
    np.where(base_dados['Dias_Atraso__L__p_3'] == 1.0, 0,
    np.where(base_dados['Dias_Atraso__L__p_3'] == 2.0, 1,
     0))))

base_dados['Dias_Atraso__L__p_3_g_1_2'] = np.where(base_dados['Dias_Atraso__L__p_3_g_1_1'] == 0, 1,
    np.where(base_dados['Dias_Atraso__L__p_3_g_1_1'] == 1, 0,
    np.where(base_dados['Dias_Atraso__L__p_3_g_1_1'] == 2, 2,
     0)))

base_dados['Dias_Atraso__L__p_3_g_1'] = np.where(base_dados['Dias_Atraso__L__p_3_g_1_2'] == 0, 0,
    np.where(base_dados['Dias_Atraso__L__p_3_g_1_2'] == 1, 1,
    np.where(base_dados['Dias_Atraso__L__p_3_g_1_2'] == 2, 2,
     0)))
         


# COMMAND ----------

# MAGIC %md
# MAGIC ### Parte 2 de 2

# COMMAND ----------


base_dados['Codigo_Politica__L__p_4_g_1_c1_6_1'] = np.where(np.bitwise_and(base_dados['Codigo_Politica__L__p_6_g_1'] == 0, base_dados['Codigo_Politica__L__p_4_g_1'] == 0), 0,
    np.where(np.bitwise_and(base_dados['Codigo_Politica__L__p_6_g_1'] == 0, base_dados['Codigo_Politica__L__p_4_g_1'] == 1), 2,
    np.where(np.bitwise_and(base_dados['Codigo_Politica__L__p_6_g_1'] == 1, base_dados['Codigo_Politica__L__p_4_g_1'] == 0), 1,
    np.where(np.bitwise_and(base_dados['Codigo_Politica__L__p_6_g_1'] == 1, base_dados['Codigo_Politica__L__p_4_g_1'] == 1), 2,
    np.where(np.bitwise_and(base_dados['Codigo_Politica__L__p_6_g_1'] == 1, base_dados['Codigo_Politica__L__p_4_g_1'] == 2), 3,
    np.where(np.bitwise_and(base_dados['Codigo_Politica__L__p_6_g_1'] == 2, base_dados['Codigo_Politica__L__p_4_g_1'] == 2), 3,
    np.where(np.bitwise_and(base_dados['Codigo_Politica__L__p_6_g_1'] == 3, base_dados['Codigo_Politica__L__p_4_g_1'] == 2), 4,
     0)))))))

base_dados['Codigo_Politica__L__p_4_g_1_c1_6_2'] = np.where(base_dados['Codigo_Politica__L__p_4_g_1_c1_6_1'] == 0, 0,
    np.where(base_dados['Codigo_Politica__L__p_4_g_1_c1_6_1'] == 1, 1,
    np.where(base_dados['Codigo_Politica__L__p_4_g_1_c1_6_1'] == 2, 2,
    np.where(base_dados['Codigo_Politica__L__p_4_g_1_c1_6_1'] == 3, 3,
    np.where(base_dados['Codigo_Politica__L__p_4_g_1_c1_6_1'] == 4, 4,
    0)))))

base_dados['Codigo_Politica__L__p_4_g_1_c1_6'] = np.where(base_dados['Codigo_Politica__L__p_4_g_1_c1_6_2'] == 0, 0,
    np.where(base_dados['Codigo_Politica__L__p_4_g_1_c1_6_2'] == 1, 1,
    np.where(base_dados['Codigo_Politica__L__p_4_g_1_c1_6_2'] == 2, 2,
    np.where(base_dados['Codigo_Politica__L__p_4_g_1_c1_6_2'] == 3, 3,
    np.where(base_dados['Codigo_Politica__L__p_4_g_1_c1_6_2'] == 4, 4,
    0)))))
         
         
         
         
         
base_dados['Saldo_Devedor_Contrato__R__pe_20_g_1_c1_26_1'] = np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R__pe_20_g_1'] == 0, base_dados['Saldo_Devedor_Contrato__R__pe_5_g_1'] == 0), 0,
    np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R__pe_20_g_1'] == 1, base_dados['Saldo_Devedor_Contrato__R__pe_5_g_1'] == 0), 0,
    np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R__pe_20_g_1'] == 2, base_dados['Saldo_Devedor_Contrato__R__pe_5_g_1'] == 0), 0,
    np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R__pe_20_g_1'] == 2, base_dados['Saldo_Devedor_Contrato__R__pe_5_g_1'] == 1), 1,
    np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R__pe_20_g_1'] == 2, base_dados['Saldo_Devedor_Contrato__R__pe_5_g_1'] == 2), 2,
    np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R__pe_20_g_1'] == 3, base_dados['Saldo_Devedor_Contrato__R__pe_5_g_1'] == 1), 2,
    np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R__pe_20_g_1'] == 3, base_dados['Saldo_Devedor_Contrato__R__pe_5_g_1'] == 2), 3,
    np.where(np.bitwise_and(base_dados['Saldo_Devedor_Contrato__R__pe_20_g_1'] == 4, base_dados['Saldo_Devedor_Contrato__R__pe_5_g_1'] == 2), 4,
    0))))))))
             
base_dados['Saldo_Devedor_Contrato__R__pe_20_g_1_c1_26_2'] = np.where(base_dados['Saldo_Devedor_Contrato__R__pe_20_g_1_c1_26_1'] == 0, 0,
    np.where(base_dados['Saldo_Devedor_Contrato__R__pe_20_g_1_c1_26_1'] == 1, 1,
    np.where(base_dados['Saldo_Devedor_Contrato__R__pe_20_g_1_c1_26_1'] == 2, 2,
    np.where(base_dados['Saldo_Devedor_Contrato__R__pe_20_g_1_c1_26_1'] == 3, 3,
    np.where(base_dados['Saldo_Devedor_Contrato__R__pe_20_g_1_c1_26_1'] == 4, 4,
    0)))))
             
base_dados['Saldo_Devedor_Contrato__R__pe_20_g_1_c1_26'] = np.where(base_dados['Saldo_Devedor_Contrato__R__pe_20_g_1_c1_26_2'] == 0, 0,
    np.where(base_dados['Saldo_Devedor_Contrato__R__pe_20_g_1_c1_26_2'] == 1, 1,
    np.where(base_dados['Saldo_Devedor_Contrato__R__pe_20_g_1_c1_26_2'] == 2, 2,
    np.where(base_dados['Saldo_Devedor_Contrato__R__pe_20_g_1_c1_26_2'] == 3, 3,
    np.where(base_dados['Saldo_Devedor_Contrato__R__pe_20_g_1_c1_26_2'] == 4, 4,
    0)))))
         
         
         
         
         
base_dados['MOB_ENTRADA__L__pe_15_g_1_c1_36_1'] = np.where(np.bitwise_and(base_dados['MOB_ENTRADA__pe_4_g_1'] == 0, base_dados['MOB_ENTRADA__L__pe_15_g_1'] == 0), 0,
    np.where(np.bitwise_and(base_dados['MOB_ENTRADA__pe_4_g_1'] == 0, base_dados['MOB_ENTRADA__L__pe_15_g_1'] == 1), 2,
    np.where(np.bitwise_and(base_dados['MOB_ENTRADA__pe_4_g_1'] == 0, base_dados['MOB_ENTRADA__L__pe_15_g_1'] == 2), 3,
    np.where(np.bitwise_and(base_dados['MOB_ENTRADA__pe_4_g_1'] == 1, base_dados['MOB_ENTRADA__L__pe_15_g_1'] == 0), 1,
    np.where(np.bitwise_and(base_dados['MOB_ENTRADA__pe_4_g_1'] == 1, base_dados['MOB_ENTRADA__L__pe_15_g_1'] == 1), 3,
    np.where(np.bitwise_and(base_dados['MOB_ENTRADA__pe_4_g_1'] == 1, base_dados['MOB_ENTRADA__L__pe_15_g_1'] == 2), 3,
    0))))))

base_dados['MOB_ENTRADA__L__pe_15_g_1_c1_36_2'] = np.where(base_dados['MOB_ENTRADA__L__pe_15_g_1_c1_36_1'] == 0, 0,
    np.where(base_dados['MOB_ENTRADA__L__pe_15_g_1_c1_36_1'] == 1, 1,
    np.where(base_dados['MOB_ENTRADA__L__pe_15_g_1_c1_36_1'] == 2, 2,
    np.where(base_dados['MOB_ENTRADA__L__pe_15_g_1_c1_36_1'] == 3, 3,
    0))))

base_dados['MOB_ENTRADA__L__pe_15_g_1_c1_36'] = np.where(base_dados['MOB_ENTRADA__L__pe_15_g_1_c1_36_2'] == 0, 0,
    np.where(base_dados['MOB_ENTRADA__L__pe_15_g_1_c1_36_2'] == 1, 1,
    np.where(base_dados['MOB_ENTRADA__L__pe_15_g_1_c1_36_2'] == 2, 2,
    np.where(base_dados['MOB_ENTRADA__L__pe_15_g_1_c1_36_2'] == 3, 3,
    0))))
         
         
         
         
         
base_dados['Contrato_Altair__C__pe_3_g_1_c1_16_1'] = np.where(np.bitwise_and(base_dados['Contrato_Altair__p_6_g_1'] == 0, base_dados['Contrato_Altair__C__pe_3_g_1'] == 0), 0,
    np.where(np.bitwise_and(base_dados['Contrato_Altair__p_6_g_1'] == 0, base_dados['Contrato_Altair__C__pe_3_g_1'] == 1), 1,
    np.where(np.bitwise_and(base_dados['Contrato_Altair__p_6_g_1'] == 1, base_dados['Contrato_Altair__C__pe_3_g_1'] == 0), 0,
    np.where(np.bitwise_and(base_dados['Contrato_Altair__p_6_g_1'] == 1, base_dados['Contrato_Altair__C__pe_3_g_1'] == 1), 2,
    np.where(np.bitwise_and(base_dados['Contrato_Altair__p_6_g_1'] == 2, base_dados['Contrato_Altair__C__pe_3_g_1'] == 0), 0,
    np.where(np.bitwise_and(base_dados['Contrato_Altair__p_6_g_1'] == 2, base_dados['Contrato_Altair__C__pe_3_g_1'] == 1), 3,
    np.where(np.bitwise_and(base_dados['Contrato_Altair__p_6_g_1'] == 3, base_dados['Contrato_Altair__C__pe_3_g_1'] == 0), 0,
    np.where(np.bitwise_and(base_dados['Contrato_Altair__p_6_g_1'] == 3, base_dados['Contrato_Altair__C__pe_3_g_1'] == 1), 4,
    0))))))))

base_dados['Contrato_Altair__C__pe_3_g_1_c1_16_2'] = np.where(base_dados['Contrato_Altair__C__pe_3_g_1_c1_16_1'] == 0, 2,
    np.where(base_dados['Contrato_Altair__C__pe_3_g_1_c1_16_1'] == 1, 0,
    np.where(base_dados['Contrato_Altair__C__pe_3_g_1_c1_16_1'] == 2, 1,
    np.where(base_dados['Contrato_Altair__C__pe_3_g_1_c1_16_1'] == 3, 3,
    np.where(base_dados['Contrato_Altair__C__pe_3_g_1_c1_16_1'] == 4, 4,
    0)))))

base_dados['Contrato_Altair__C__pe_3_g_1_c1_16'] = np.where(base_dados['Contrato_Altair__C__pe_3_g_1_c1_16_2'] == 0, 0,
    np.where(base_dados['Contrato_Altair__C__pe_3_g_1_c1_16_2'] == 1, 1,
    np.where(base_dados['Contrato_Altair__C__pe_3_g_1_c1_16_2'] == 2, 2,
    np.where(base_dados['Contrato_Altair__C__pe_3_g_1_c1_16_2'] == 3, 3,
    np.where(base_dados['Contrato_Altair__C__pe_3_g_1_c1_16_2'] == 4, 4,
    0)))))
         
         
         
         
                
base_dados['Dias_Atraso__L__p_3_g_1_c1_44_1'] = np.where(np.bitwise_and(base_dados['Dias_Atraso__pe_7_g_1'] == 0, base_dados['Dias_Atraso__L__p_3_g_1'] == 0), 0,
    np.where(np.bitwise_and(base_dados['Dias_Atraso__pe_7_g_1'] == 0, base_dados['Dias_Atraso__L__p_3_g_1'] == 2), 3,
    np.where(np.bitwise_and(base_dados['Dias_Atraso__pe_7_g_1'] == 1, base_dados['Dias_Atraso__L__p_3_g_1'] == 0), 1,
    np.where(np.bitwise_and(base_dados['Dias_Atraso__pe_7_g_1'] == 2, base_dados['Dias_Atraso__L__p_3_g_1'] == 0), 2,
    np.where(np.bitwise_and(base_dados['Dias_Atraso__pe_7_g_1'] == 2, base_dados['Dias_Atraso__L__p_3_g_1'] == 1), 3,
    np.where(np.bitwise_and(base_dados['Dias_Atraso__pe_7_g_1'] == 3, base_dados['Dias_Atraso__L__p_3_g_1'] == 1), 3,
    np.where(np.bitwise_and(base_dados['Dias_Atraso__pe_7_g_1'] == 3, base_dados['Dias_Atraso__L__p_3_g_1'] == 2), 4,
    0)))))))

base_dados['Dias_Atraso__L__p_3_g_1_c1_44_2'] = np.where(base_dados['Dias_Atraso__L__p_3_g_1_c1_44_1'] == 0, 0,
    np.where(base_dados['Dias_Atraso__L__p_3_g_1_c1_44_1'] == 1, 1,
    np.where(base_dados['Dias_Atraso__L__p_3_g_1_c1_44_1'] == 2, 2,
    np.where(base_dados['Dias_Atraso__L__p_3_g_1_c1_44_1'] == 3, 3,
    np.where(base_dados['Dias_Atraso__L__p_3_g_1_c1_44_1'] == 4, 4,
    0)))))

base_dados['Dias_Atraso__L__p_3_g_1_c1_44'] = np.where(base_dados['Dias_Atraso__L__p_3_g_1_c1_44_2'] == 0, 0,
    np.where(base_dados['Dias_Atraso__L__p_3_g_1_c1_44_2'] == 1, 1,
    np.where(base_dados['Dias_Atraso__L__p_3_g_1_c1_44_2'] == 2, 2,
    np.where(base_dados['Dias_Atraso__L__p_3_g_1_c1_44_2'] == 3, 3,
    np.where(base_dados['Dias_Atraso__L__p_3_g_1_c1_44_2'] == 4, 4,
    0)))))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Mantendo apenas as variáveis do modelo

# COMMAND ----------

import pickle
modelo=pickle.load(open(caminho_base + 'model_fit_santander.sav', 'rb'))

base_teste_c0 = base_dados[[chave,'Telefone1_gh38','rank:d_gh38','rank:c_gh38','rank:a_gh38','rank:b_gh38','MOB_ENTRADA__L__pe_15_g_1_c1_36','Indicador_Correntista_gh38','Telefone5_gh38','Dias_Atraso__L__p_3_g_1_c1_44','Qtde_Max_Parcelas_gh38','PRODUTO_gh38','Codigo_Politica__L__p_4_g_1_c1_6','Telefone3_gh38','Contrato_Altair__C__pe_3_g_1_c1_16','Saldo_Devedor_Contrato__R__pe_20_g_1_c1_26']]

var_fin_c0=['Telefone1_gh38','rank:d_gh38','rank:c_gh38','rank:a_gh38','rank:b_gh38','MOB_ENTRADA__L__pe_15_g_1_c1_36','Indicador_Correntista_gh38','Telefone5_gh38','Dias_Atraso__L__p_3_g_1_c1_44','Qtde_Max_Parcelas_gh38','PRODUTO_gh38','Codigo_Politica__L__p_4_g_1_c1_6','Telefone3_gh38','Contrato_Altair__C__pe_3_g_1_c1_16','Saldo_Devedor_Contrato__R__pe_20_g_1_c1_26']

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

x_teste2['P_1_R_p_8_g_1'] = np.where(x_teste2['P_1_R'] <= 0.168088741, 0,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.168088741, x_teste2['P_1_R'] <= 0.238287946), 1,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.238287946, x_teste2['P_1_R'] <= 0.308712555), 2,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.308712555, x_teste2['P_1_R'] <= 0.389777009), 3,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.389777009, x_teste2['P_1_R'] <= 0.465852516), 4,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.465852516, x_teste2['P_1_R'] <= 0.566455277), 5,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.566455277, x_teste2['P_1_R'] <= 0.684075197), 6,
    np.where(x_teste2['P_1_R'] > 0.684075197,7,0))))))))

x_teste2['P_1_p_17_g_1'] = np.where(x_teste2['P_1'] <= 0.040745566, 0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.040745566, x_teste2['P_1'] <= 0.071708545), 1,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.071708545, x_teste2['P_1'] <= 0.116065584), 2,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.116065584, x_teste2['P_1'] <= 0.204724701), 3,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.204724701, x_teste2['P_1'] <= 0.293440542), 4,
    np.where(x_teste2['P_1'] > 0.293440542, 5,0))))))

x_teste2['GH'] = np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 0, x_teste2['P_1_R_p_8_g_1'] == 0), 0,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 0, x_teste2['P_1_R_p_8_g_1'] == 1), 0,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 1, x_teste2['P_1_R_p_8_g_1'] == 1), 1,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 1, x_teste2['P_1_R_p_8_g_1'] == 2), 1,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 2, x_teste2['P_1_R_p_8_g_1'] == 2), 2,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 2, x_teste2['P_1_R_p_8_g_1'] == 3), 2,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 3, x_teste2['P_1_R_p_8_g_1'] == 3), 3,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 3, x_teste2['P_1_R_p_8_g_1'] == 4), 3,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 4, x_teste2['P_1_R_p_8_g_1'] == 4), 3,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 4, x_teste2['P_1_R_p_8_g_1'] == 5), 4,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 5, x_teste2['P_1_R_p_8_g_1'] == 5), 4,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 5, x_teste2['P_1_R_p_8_g_1'] == 6), 5,
    np.where(np.bitwise_and(x_teste2['P_1_p_17_g_1'] == 5, x_teste2['P_1_R_p_8_g_1'] == 7), 6,
             2)))))))))))))

del x_teste2['P_1_R']
del x_teste2['P_1_R_p_8_g_1']
del x_teste2['P_1_p_17_g_1']

x_teste2


# COMMAND ----------

