# Databricks notebook source
# MAGIC %md
# MAGIC # <font color='blue'>IA - Feature Selection</font>
# MAGIC 
# MAGIC # <font color='blue'>Ferramenta de Criação de Variáveis</font>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importando os pacotes Python

# COMMAND ----------

# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

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
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.preprocessing import KBinsDiscretizer
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
chave = 'documento'

#Nome da Base de Dados
N_Base = "base_aleatoria_f.csv"

#Caminho da base de dados
caminho_base = '/mnt/ml-prd/ml-data/propensaodeal/havan/trusted/'

#Separador
separador_ = ";"

#Decimal
decimal_ = "."

caminho_pickle = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/havan/pickle_model'

model = 'v1_model_20220113'

caminho_output = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/havan/output'

# COMMAND ----------

folder_a_processar = max(os.listdir('/dbfs'+caminho_base))
folder_a_processar

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importação da Base de Dados

# COMMAND ----------

caminho_arquivo = '/dbfs'+os.path.join(caminho_base, folder_a_processar, 'havan_full.csv')

# COMMAND ----------

base_dados = pd.read_csv(caminho_arquivo, sep=separador_, decimal=decimal_)
base_dados = base_dados[[chave,'codigo', 'dataMenorVencimento', 'valorVencido', 'celularPrincipal', 'emailPrincipal','diaVencimentoCartao']]

base_dados.fillna(-3)

#string

base_dados['P_emailPrincipal'] = np.where(base_dados['emailPrincipal'] != '-3',1,0)

#numericas
base_dados['celularPrincipal'] = base_dados['celularPrincipal'].replace(np.nan, '-3')
base_dados['codigo'] = base_dados['codigo'].replace(np.nan, '-3')
base_dados['diaVencimentoCartao'] = base_dados['diaVencimentoCartao'].replace(np.nan, '-3')
base_dados['valorVencido'] = base_dados['valorVencido'].replace(np.nan, '-3')

base_dados = base_dados.apply(pd.to_numeric, errors='ignore')

base_dados['celularPrincipal'] = base_dados['celularPrincipal'].astype(np.int64)
base_dados['codigo'] = base_dados['codigo'].astype(np.int64)
base_dados['diaVencimentoCartao'] = base_dados['diaVencimentoCartao'].astype(int)
base_dados['valorVencido'] = base_dados['valorVencido'].astype(float)

base_dados['dataMenorVencimento'] = pd.to_datetime(base_dados['dataMenorVencimento'])

base_dados['mob_dataMenorVencimento'] = ((datetime.today()) - base_dados.dataMenorVencimento)/np.timedelta64(1, 'M')

del base_dados['dataMenorVencimento']
del base_dados['emailPrincipal']
base_dados.drop_duplicates(keep=False, inplace=True)

print("shape da Base de Dados:",base_dados.shape)

base_dados.head()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Gerando variáveis Categóricas

# COMMAND ----------

base_dados['diaVencimentoCartao_gh30'] = np.where(base_dados['diaVencimentoCartao'] == 1, 0,
np.where(base_dados['diaVencimentoCartao'] == 5, 1,
np.where(base_dados['diaVencimentoCartao'] == 8, 2,
np.where(base_dados['diaVencimentoCartao'] == 10, 3,
np.where(base_dados['diaVencimentoCartao'] == 13, 4,
np.where(base_dados['diaVencimentoCartao'] == 15, 5,
np.where(base_dados['diaVencimentoCartao'] == 20, 6,
np.where(base_dados['diaVencimentoCartao'] == 25, 7,
np.where(base_dados['diaVencimentoCartao'] == 28, 8,
np.where(base_dados['diaVencimentoCartao'] == 30, 9,
0))))))))))
base_dados['diaVencimentoCartao_gh31'] = np.where(base_dados['diaVencimentoCartao_gh30'] == 0, 0,
np.where(base_dados['diaVencimentoCartao_gh30'] == 1, 0,
np.where(base_dados['diaVencimentoCartao_gh30'] == 2, 2,
np.where(base_dados['diaVencimentoCartao_gh30'] == 3, 3,
np.where(base_dados['diaVencimentoCartao_gh30'] == 4, 4,
np.where(base_dados['diaVencimentoCartao_gh30'] == 5, 5,
np.where(base_dados['diaVencimentoCartao_gh30'] == 6, 5,
np.where(base_dados['diaVencimentoCartao_gh30'] == 7, 7,
np.where(base_dados['diaVencimentoCartao_gh30'] == 8, 7,
np.where(base_dados['diaVencimentoCartao_gh30'] == 9, 9,
0))))))))))
base_dados['diaVencimentoCartao_gh32'] = np.where(base_dados['diaVencimentoCartao_gh31'] == 0, 0,
np.where(base_dados['diaVencimentoCartao_gh31'] == 2, 1,
np.where(base_dados['diaVencimentoCartao_gh31'] == 3, 2,
np.where(base_dados['diaVencimentoCartao_gh31'] == 4, 3,
np.where(base_dados['diaVencimentoCartao_gh31'] == 5, 4,
np.where(base_dados['diaVencimentoCartao_gh31'] == 7, 5,
np.where(base_dados['diaVencimentoCartao_gh31'] == 9, 6,
0)))))))
base_dados['diaVencimentoCartao_gh33'] = np.where(base_dados['diaVencimentoCartao_gh32'] == 0, 0,
np.where(base_dados['diaVencimentoCartao_gh32'] == 1, 1,
np.where(base_dados['diaVencimentoCartao_gh32'] == 2, 2,
np.where(base_dados['diaVencimentoCartao_gh32'] == 3, 3,
np.where(base_dados['diaVencimentoCartao_gh32'] == 4, 4,
np.where(base_dados['diaVencimentoCartao_gh32'] == 5, 5,
np.where(base_dados['diaVencimentoCartao_gh32'] == 6, 6,
0)))))))
base_dados['diaVencimentoCartao_gh34'] = np.where(base_dados['diaVencimentoCartao_gh33'] == 0, 0,
np.where(base_dados['diaVencimentoCartao_gh33'] == 1, 1,
np.where(base_dados['diaVencimentoCartao_gh33'] == 2, 2,
np.where(base_dados['diaVencimentoCartao_gh33'] == 3, 1,
np.where(base_dados['diaVencimentoCartao_gh33'] == 4, 4,
np.where(base_dados['diaVencimentoCartao_gh33'] == 5, 5,
np.where(base_dados['diaVencimentoCartao_gh33'] == 6, 4,
0)))))))
base_dados['diaVencimentoCartao_gh35'] = np.where(base_dados['diaVencimentoCartao_gh34'] == 0, 0,
np.where(base_dados['diaVencimentoCartao_gh34'] == 1, 1,
np.where(base_dados['diaVencimentoCartao_gh34'] == 2, 2,
np.where(base_dados['diaVencimentoCartao_gh34'] == 4, 3,
np.where(base_dados['diaVencimentoCartao_gh34'] == 5, 4,
0)))))
base_dados['diaVencimentoCartao_gh36'] = np.where(base_dados['diaVencimentoCartao_gh35'] == 0, 1,
np.where(base_dados['diaVencimentoCartao_gh35'] == 1, 0,
np.where(base_dados['diaVencimentoCartao_gh35'] == 2, 1,
np.where(base_dados['diaVencimentoCartao_gh35'] == 3, 4,
np.where(base_dados['diaVencimentoCartao_gh35'] == 4, 1,
0)))))
base_dados['diaVencimentoCartao_gh37'] = np.where(base_dados['diaVencimentoCartao_gh36'] == 0, 1,
np.where(base_dados['diaVencimentoCartao_gh36'] == 1, 1,
np.where(base_dados['diaVencimentoCartao_gh36'] == 4, 2,
0)))
base_dados['diaVencimentoCartao_gh38'] = np.where(base_dados['diaVencimentoCartao_gh37'] == 1, 0,
np.where(base_dados['diaVencimentoCartao_gh37'] == 2, 1,
0))





base_dados['P_emailPrincipal_gh30'] = np.where(base_dados['P_emailPrincipal'] == 0, 0,
np.where(base_dados['P_emailPrincipal'] == 1, 1,
0))
base_dados['P_emailPrincipal_gh31'] = np.where(base_dados['P_emailPrincipal_gh30'] == 0, 0,
np.where(base_dados['P_emailPrincipal_gh30'] == 1, 1,
0))
base_dados['P_emailPrincipal_gh32'] = np.where(base_dados['P_emailPrincipal_gh31'] == 0, 0,
np.where(base_dados['P_emailPrincipal_gh31'] == 1, 1,
0))
base_dados['P_emailPrincipal_gh33'] = np.where(base_dados['P_emailPrincipal_gh32'] == 0, 0,
np.where(base_dados['P_emailPrincipal_gh32'] == 1, 1,
0))
base_dados['P_emailPrincipal_gh34'] = np.where(base_dados['P_emailPrincipal_gh33'] == 0, 0,
np.where(base_dados['P_emailPrincipal_gh33'] == 1, 1,
0))
base_dados['P_emailPrincipal_gh35'] = np.where(base_dados['P_emailPrincipal_gh34'] == 0, 0,
np.where(base_dados['P_emailPrincipal_gh34'] == 1, 1,
0))
base_dados['P_emailPrincipal_gh36'] = np.where(base_dados['P_emailPrincipal_gh35'] == 0, 0,
np.where(base_dados['P_emailPrincipal_gh35'] == 1, 1,
0))
base_dados['P_emailPrincipal_gh37'] = np.where(base_dados['P_emailPrincipal_gh36'] == 0, 0,
np.where(base_dados['P_emailPrincipal_gh36'] == 1, 1,
0))
base_dados['P_emailPrincipal_gh38'] = np.where(base_dados['P_emailPrincipal_gh37'] == 0, 0,
np.where(base_dados['P_emailPrincipal_gh37'] == 1, 1,
0))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Gerando variáveis numéricas contínuas

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parte 1 de 2

# COMMAND ----------


base_dados['celularPrincipal__p_7'] = np.where(base_dados['celularPrincipal'] == 0 , -1.0,
np.where(base_dados['celularPrincipal'] <= 16997246368.0, 0.0,
np.where(np.bitwise_and(base_dados['celularPrincipal'] > 0.0, base_dados['celularPrincipal'] <= 35984182966.0), 1.0,
np.where(np.bitwise_and(base_dados['celularPrincipal'] > 35984182966.0, base_dados['celularPrincipal'] <= 42999410392.0), 2.0,
np.where(np.bitwise_and(base_dados['celularPrincipal'] > 42999410392.0, base_dados['celularPrincipal'] <= 47992325955.0), 3.0,
np.where(np.bitwise_and(base_dados['celularPrincipal'] > 47992325955.0, base_dados['celularPrincipal'] <= 49988121428.0), 4.0,
np.where(np.bitwise_and(base_dados['celularPrincipal'] > 49988121428.0, base_dados['celularPrincipal'] <= 66992928526.0), 5.0,
np.where(base_dados['celularPrincipal'] > 66992928526.0, 6.0,
 0))))))))
base_dados['celularPrincipal__p_7_g_1_1'] = np.where(base_dados['celularPrincipal__p_7'] == -1.0, 2,
np.where(base_dados['celularPrincipal__p_7'] == 0.0, 2,
np.where(base_dados['celularPrincipal__p_7'] == 1.0, 0,
np.where(base_dados['celularPrincipal__p_7'] == 2.0, 1,
np.where(base_dados['celularPrincipal__p_7'] == 3.0, 2,
np.where(base_dados['celularPrincipal__p_7'] == 4.0, 1,
np.where(base_dados['celularPrincipal__p_7'] == 5.0, 2,
np.where(base_dados['celularPrincipal__p_7'] == 6.0, 0,
 0))))))))
base_dados['celularPrincipal__p_7_g_1_2'] = np.where(base_dados['celularPrincipal__p_7_g_1_1'] == 0, 1,
np.where(base_dados['celularPrincipal__p_7_g_1_1'] == 1, 0,
np.where(base_dados['celularPrincipal__p_7_g_1_1'] == 2, 1,
 0)))
base_dados['celularPrincipal__p_7_g_1'] = np.where(base_dados['celularPrincipal__p_7_g_1_2'] == 0, 0,
np.where(base_dados['celularPrincipal__p_7_g_1_2'] == 1, 1,
 0))
                                                   
                                                   
                                                   
                                                   
                                                   
                                                   
                                                   
base_dados['celularPrincipal__p_3'] = np.where(base_dados['celularPrincipal'] == 0 , -1.0,
np.where(base_dados['celularPrincipal'] <= 41995075100.0, 0.0,
np.where(np.bitwise_and(base_dados['celularPrincipal'] > 0.0, base_dados['celularPrincipal'] <= 48994252089.0), 1.0,
np.where(base_dados['celularPrincipal'] > 48994252089.0, 2.0,
 0))))
base_dados['celularPrincipal__p_3_g_1_1'] = np.where(base_dados['celularPrincipal__p_3'] == -1.0, 1,
np.where(base_dados['celularPrincipal__p_3'] == 0.0, 0,
np.where(base_dados['celularPrincipal__p_3'] == 1.0, 1,
np.where(base_dados['celularPrincipal__p_3'] == 2.0, 0,
 0))))
base_dados['celularPrincipal__p_3_g_1_2'] = np.where(base_dados['celularPrincipal__p_3_g_1_1'] == 0, 1,
np.where(base_dados['celularPrincipal__p_3_g_1_1'] == 1, 0,
 0))
base_dados['celularPrincipal__p_3_g_1'] = np.where(base_dados['celularPrincipal__p_3_g_1_2'] == 0, 0,
np.where(base_dados['celularPrincipal__p_3_g_1_2'] == 1, 1,
 0))
         
         
         
         
         
        
base_dados['codigo__pu_8'] = np.where(base_dados['codigo'] <= 4382344.0, 0.0,
np.where(np.bitwise_and(base_dados['codigo'] > 4382344.0, base_dados['codigo'] <= 5683598.0), 1.0,
np.where(np.bitwise_and(base_dados['codigo'] > 5683598.0, base_dados['codigo'] <= 13170521.0), 2.0,
np.where(np.bitwise_and(base_dados['codigo'] > 13170521.0, base_dados['codigo'] <= 17558826.0), 3.0,
np.where(np.bitwise_and(base_dados['codigo'] > 17558826.0, base_dados['codigo'] <= 21934525.0), 4.0,
np.where(np.bitwise_and(base_dados['codigo'] > 21934525.0, base_dados['codigo'] <= 24264764.0), 5.0,
np.where(base_dados['codigo'] > 24264764.0, 7.0,
 0)))))))
base_dados['codigo__pu_8_g_1_1'] = np.where(base_dados['codigo__pu_8'] == 0.0, 1,
np.where(base_dados['codigo__pu_8'] == 1.0, 0,
np.where(base_dados['codigo__pu_8'] == 2.0, 2,
np.where(base_dados['codigo__pu_8'] == 3.0, 2,
np.where(base_dados['codigo__pu_8'] == 4.0, 0,
np.where(base_dados['codigo__pu_8'] == 5.0, 2,
np.where(base_dados['codigo__pu_8'] == 7.0, 2,
 0)))))))
base_dados['codigo__pu_8_g_1_2'] = np.where(base_dados['codigo__pu_8_g_1_1'] == 0, 1,
np.where(base_dados['codigo__pu_8_g_1_1'] == 1, 0,
np.where(base_dados['codigo__pu_8_g_1_1'] == 2, 1,
 0)))
base_dados['codigo__pu_8_g_1'] = np.where(base_dados['codigo__pu_8_g_1_2'] == 0, 0,
np.where(base_dados['codigo__pu_8_g_1_2'] == 1, 1,
 0))
                                          
                                          
                                          
                                          
                                          
                                          
base_dados['codigo__L'] = np.log(base_dados['codigo'])
np.where(base_dados['codigo__L'] == 0, -1, base_dados['codigo__L'])
base_dados['codigo__L'] = base_dados['codigo__L'].fillna(-2)
base_dados['codigo__L__pu_13'] = np.where(base_dados['codigo__L'] <= 9.172742341560864, 0.0,
np.where(np.bitwise_and(base_dados['codigo__L'] > 9.172742341560864, base_dados['codigo__L'] <= 10.83281296268074), 2.0,
np.where(np.bitwise_and(base_dados['codigo__L'] > 10.83281296268074, base_dados['codigo__L'] <= 11.683064785395203), 3.0,
np.where(np.bitwise_and(base_dados['codigo__L'] > 11.683064785395203, base_dados['codigo__L'] <= 12.21719061126402), 4.0,
np.where(np.bitwise_and(base_dados['codigo__L'] > 12.21719061126402, base_dados['codigo__L'] <= 12.50185680723493), 5.0,
np.where(np.bitwise_and(base_dados['codigo__L'] > 12.50185680723493, base_dados['codigo__L'] <= 13.572682022433666), 6.0,
np.where(np.bitwise_and(base_dados['codigo__L'] > 13.572682022433666, base_dados['codigo__L'] <= 14.218783264515524), 7.0,
np.where(np.bitwise_and(base_dados['codigo__L'] > 14.218783264515524, base_dados['codigo__L'] <= 14.850388624182111), 8.0,
np.where(np.bitwise_and(base_dados['codigo__L'] > 14.850388624182111, base_dados['codigo__L'] <= 15.481123678828032), 9.0,
np.where(np.bitwise_and(base_dados['codigo__L'] > 15.481123678828032, base_dados['codigo__L'] <= 15.553095040856348), 10.0,
np.where(np.bitwise_and(base_dados['codigo__L'] > 15.553095040856348, base_dados['codigo__L'] <= 16.742803646004656), 11.0,
np.where(base_dados['codigo__L'] > 16.742803646004656, 12.0,
 0))))))))))))
base_dados['codigo__L__pu_13_g_1_1'] = np.where(base_dados['codigo__L__pu_13'] == 0.0, 1,
np.where(base_dados['codigo__L__pu_13'] == 2.0, 1,
np.where(base_dados['codigo__L__pu_13'] == 3.0, 1,
np.where(base_dados['codigo__L__pu_13'] == 4.0, 1,
np.where(base_dados['codigo__L__pu_13'] == 5.0, 1,
np.where(base_dados['codigo__L__pu_13'] == 6.0, 1,
np.where(base_dados['codigo__L__pu_13'] == 7.0, 0,
np.where(base_dados['codigo__L__pu_13'] == 8.0, 1,
np.where(base_dados['codigo__L__pu_13'] == 9.0, 0,
np.where(base_dados['codigo__L__pu_13'] == 10.0, 0,
np.where(base_dados['codigo__L__pu_13'] == 11.0, 0,
np.where(base_dados['codigo__L__pu_13'] == 12.0, 0,
 0))))))))))))
base_dados['codigo__L__pu_13_g_1_2'] = np.where(base_dados['codigo__L__pu_13_g_1_1'] == 0, 1,
np.where(base_dados['codigo__L__pu_13_g_1_1'] == 1, 0,
 0))
base_dados['codigo__L__pu_13_g_1'] = np.where(base_dados['codigo__L__pu_13_g_1_2'] == 0, 0,
np.where(base_dados['codigo__L__pu_13_g_1_2'] == 1, 1,
 0))
                                            
                                            
                                            
                                            
                                            
                                            
                                            
base_dados['valorVencido__R'] = np.sqrt(base_dados['valorVencido'])
np.where(base_dados['valorVencido__R'] == 0, -1, base_dados['valorVencido__R'])
base_dados['valorVencido__R'] = base_dados['valorVencido__R'].fillna(-2)
base_dados['valorVencido__R__pe_13'] = np.where(base_dados['valorVencido__R'] <= 16.375591592366977, 0.0,
np.where(np.bitwise_and(base_dados['valorVencido__R'] > 16.375591592366977, base_dados['valorVencido__R'] <= 32.75759453928203), 1.0,
np.where(np.bitwise_and(base_dados['valorVencido__R'] > 32.75759453928203, base_dados['valorVencido__R'] <= 49.106007779089516), 2.0,
np.where(np.bitwise_and(base_dados['valorVencido__R'] > 49.106007779089516, base_dados['valorVencido__R'] <= 65.49206058752465), 3.0,
np.where(np.bitwise_and(base_dados['valorVencido__R'] > 65.49206058752465, base_dados['valorVencido__R'] <= 81.89322316284785), 4.0,
np.where(np.bitwise_and(base_dados['valorVencido__R'] > 81.89322316284785, base_dados['valorVencido__R'] <= 98.24896946024421), 5.0,
np.where(np.bitwise_and(base_dados['valorVencido__R'] > 98.24896946024421, base_dados['valorVencido__R'] <= 114.64815742086743), 6.0,
np.where(np.bitwise_and(base_dados['valorVencido__R'] > 114.64815742086743, base_dados['valorVencido__R'] <= 130.96468989769724), 7.0,
np.where(np.bitwise_and(base_dados['valorVencido__R'] > 130.96468989769724, base_dados['valorVencido__R'] <= 147.40111939873455), 8.0,
np.where(np.bitwise_and(base_dados['valorVencido__R'] > 147.40111939873455, base_dados['valorVencido__R'] <= 163.60788489556364), 9.0,
np.where(np.bitwise_and(base_dados['valorVencido__R'] > 163.60788489556364, base_dados['valorVencido__R'] <= 180.1582360037975), 10.0,
np.where(np.bitwise_and(base_dados['valorVencido__R'] > 180.1582360037975, base_dados['valorVencido__R'] <= 196.53223654148954), 11.0,
np.where(base_dados['valorVencido__R'] > 196.53223654148954, 12.0,
 -2)))))))))))))
base_dados['valorVencido__R__pe_13_g_1_1'] = np.where(base_dados['valorVencido__R__pe_13'] == -2.0, 4,
np.where(base_dados['valorVencido__R__pe_13'] == 0.0, 0,
np.where(base_dados['valorVencido__R__pe_13'] == 1.0, 0,
np.where(base_dados['valorVencido__R__pe_13'] == 2.0, 2,
np.where(base_dados['valorVencido__R__pe_13'] == 3.0, 1,
np.where(base_dados['valorVencido__R__pe_13'] == 4.0, 2,
np.where(base_dados['valorVencido__R__pe_13'] == 5.0, 3,
np.where(base_dados['valorVencido__R__pe_13'] == 6.0, 3,
np.where(base_dados['valorVencido__R__pe_13'] == 7.0, 3,
np.where(base_dados['valorVencido__R__pe_13'] == 8.0, 4,
np.where(base_dados['valorVencido__R__pe_13'] == 9.0, 4,
np.where(base_dados['valorVencido__R__pe_13'] == 10.0, 4,
np.where(base_dados['valorVencido__R__pe_13'] == 11.0, 4,
np.where(base_dados['valorVencido__R__pe_13'] == 12.0, 4,
 0))))))))))))))
base_dados['valorVencido__R__pe_13_g_1_2'] = np.where(base_dados['valorVencido__R__pe_13_g_1_1'] == 0, 4,
np.where(base_dados['valorVencido__R__pe_13_g_1_1'] == 1, 2,
np.where(base_dados['valorVencido__R__pe_13_g_1_1'] == 2, 3,
np.where(base_dados['valorVencido__R__pe_13_g_1_1'] == 3, 1,
np.where(base_dados['valorVencido__R__pe_13_g_1_1'] == 4, 0,
 0)))))
base_dados['valorVencido__R__pe_13_g_1'] = np.where(base_dados['valorVencido__R__pe_13_g_1_2'] == 0, 0,
np.where(base_dados['valorVencido__R__pe_13_g_1_2'] == 1, 1,
np.where(base_dados['valorVencido__R__pe_13_g_1_2'] == 2, 2,
np.where(base_dados['valorVencido__R__pe_13_g_1_2'] == 3, 3,
np.where(base_dados['valorVencido__R__pe_13_g_1_2'] == 4, 4,
 0)))))
         
         
         
         
         
        
base_dados['valorVencido__L'] = np.log(base_dados['valorVencido'])
np.where(base_dados['valorVencido__L'] == 0, -1, base_dados['valorVencido__L'])
base_dados['valorVencido__L'] = base_dados['valorVencido__L'].fillna(-2)
base_dados['valorVencido__L__pu_7'] = np.where(base_dados['valorVencido__L'] <= 4.25063580654847, 0.0,
np.where(np.bitwise_and(base_dados['valorVencido__L'] > 4.25063580654847, base_dados['valorVencido__L'] <= 5.513307771105859), 1.0,
np.where(np.bitwise_and(base_dados['valorVencido__L'] > 5.513307771105859, base_dados['valorVencido__L'] <= 6.748313961714909), 2.0,
np.where(np.bitwise_and(base_dados['valorVencido__L'] > 6.748313961714909, base_dados['valorVencido__L'] <= 7.98411514204711), 3.0,
np.where(np.bitwise_and(base_dados['valorVencido__L'] > 7.98411514204711, base_dados['valorVencido__L'] <= 9.219637023881278), 4.0,
np.where(np.bitwise_and(base_dados['valorVencido__L'] > 9.219637023881278, base_dados['valorVencido__L'] <= 10.454439056591072), 5.0,
np.where(base_dados['valorVencido__L'] > 10.454439056591072, 6.0,
 0)))))))
base_dados['valorVencido__L__pu_7_g_1_1'] = np.where(base_dados['valorVencido__L__pu_7'] == 0.0, 3,
np.where(base_dados['valorVencido__L__pu_7'] == 1.0, 2,
np.where(base_dados['valorVencido__L__pu_7'] == 2.0, 0,
np.where(base_dados['valorVencido__L__pu_7'] == 3.0, 2,
np.where(base_dados['valorVencido__L__pu_7'] == 4.0, 1,
np.where(base_dados['valorVencido__L__pu_7'] == 5.0, 3,
np.where(base_dados['valorVencido__L__pu_7'] == 6.0, 3,
 0)))))))
base_dados['valorVencido__L__pu_7_g_1_2'] = np.where(base_dados['valorVencido__L__pu_7_g_1_1'] == 0, 3,
np.where(base_dados['valorVencido__L__pu_7_g_1_1'] == 1, 1,
np.where(base_dados['valorVencido__L__pu_7_g_1_1'] == 2, 2,
np.where(base_dados['valorVencido__L__pu_7_g_1_1'] == 3, 0,
 0))))
base_dados['valorVencido__L__pu_7_g_1'] = np.where(base_dados['valorVencido__L__pu_7_g_1_2'] == 0, 0,
np.where(base_dados['valorVencido__L__pu_7_g_1_2'] == 1, 1,
np.where(base_dados['valorVencido__L__pu_7_g_1_2'] == 2, 2,
np.where(base_dados['valorVencido__L__pu_7_g_1_2'] == 3, 3,
 0))))
         
         
         
         
               
base_dados['mob_dataMenorVencimento__L'] = np.log(base_dados['mob_dataMenorVencimento'])
np.where(base_dados['mob_dataMenorVencimento__L'] == 0, -1, base_dados['mob_dataMenorVencimento__L'])
base_dados['mob_dataMenorVencimento__L'] = base_dados['mob_dataMenorVencimento__L'].fillna(-2)
base_dados['mob_dataMenorVencimento__L__pe_5'] = np.where(base_dados['mob_dataMenorVencimento__L'] <= 0.661738184581481, 0.0,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__L'] > 0.661738184581481, base_dados['mob_dataMenorVencimento__L'] <= 1.7200934599448887), 1.0,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__L'] > 1.7200934599448887, base_dados['mob_dataMenorVencimento__L'] <= 2.5931373883820727), 2.0,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__L'] > 2.5931373883820727, base_dados['mob_dataMenorVencimento__L'] <= 3.458534819411297), 3.0,
np.where(base_dados['mob_dataMenorVencimento__L'] > 3.458534819411297, 4.0,
 -2)))))
base_dados['mob_dataMenorVencimento__L__pe_5_g_1_1'] = np.where(base_dados['mob_dataMenorVencimento__L__pe_5'] == -2.0, 3,
np.where(base_dados['mob_dataMenorVencimento__L__pe_5'] == 0.0, 2,
np.where(base_dados['mob_dataMenorVencimento__L__pe_5'] == 1.0, 1,
np.where(base_dados['mob_dataMenorVencimento__L__pe_5'] == 2.0, 0,
np.where(base_dados['mob_dataMenorVencimento__L__pe_5'] == 3.0, 2,
np.where(base_dados['mob_dataMenorVencimento__L__pe_5'] == 4.0, 3,
 0))))))
base_dados['mob_dataMenorVencimento__L__pe_5_g_1_2'] = np.where(base_dados['mob_dataMenorVencimento__L__pe_5_g_1_1'] == 0, 2,
np.where(base_dados['mob_dataMenorVencimento__L__pe_5_g_1_1'] == 1, 3,
np.where(base_dados['mob_dataMenorVencimento__L__pe_5_g_1_1'] == 2, 1,
np.where(base_dados['mob_dataMenorVencimento__L__pe_5_g_1_1'] == 3, 0,
 0))))
base_dados['mob_dataMenorVencimento__L__pe_5_g_1'] = np.where(base_dados['mob_dataMenorVencimento__L__pe_5_g_1_2'] == 0, 0,
np.where(base_dados['mob_dataMenorVencimento__L__pe_5_g_1_2'] == 1, 1,
np.where(base_dados['mob_dataMenorVencimento__L__pe_5_g_1_2'] == 2, 2,
np.where(base_dados['mob_dataMenorVencimento__L__pe_5_g_1_2'] == 3, 3,
 0))))
         
         
         
         
         
base_dados['mob_dataMenorVencimento__T'] = np.tan(base_dados['mob_dataMenorVencimento'])
np.where(base_dados['mob_dataMenorVencimento__T'] == 0, -1, base_dados['mob_dataMenorVencimento__T'])
base_dados['mob_dataMenorVencimento__T'] = base_dados['mob_dataMenorVencimento__T'].fillna(-2)
base_dados['mob_dataMenorVencimento__T__pk_40'] = np.where(base_dados['mob_dataMenorVencimento__T'] <= -380.29124485627545, 0.0,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__T'] > -380.29124485627545, base_dados['mob_dataMenorVencimento__T'] <= -258.0863401524025), 1.0,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__T'] > -258.0863401524025, base_dados['mob_dataMenorVencimento__T'] <= -103.15847207761904), 2.0,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__T'] > -103.15847207761904, base_dados['mob_dataMenorVencimento__T'] <= -64.45944299755152), 3.0,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__T'] > -64.45944299755152, base_dados['mob_dataMenorVencimento__T'] <= -56.82533965429011), 4.0,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__T'] > -56.82533965429011, base_dados['mob_dataMenorVencimento__T'] <= -54.240111430672236), 5.0,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__T'] > -54.240111430672236, base_dados['mob_dataMenorVencimento__T'] <= -45.0991417322798), 6.0,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__T'] > -45.0991417322798, base_dados['mob_dataMenorVencimento__T'] <= -41.22338620938211), 7.0,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__T'] > -41.22338620938211, base_dados['mob_dataMenorVencimento__T'] <= -31.920812767892304), 8.0,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__T'] > -31.920812767892304, base_dados['mob_dataMenorVencimento__T'] <= -24.95031565393704), 9.0,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__T'] > -24.95031565393704, base_dados['mob_dataMenorVencimento__T'] <= -23.488459810354325), 10.0,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__T'] > -23.488459810354325, base_dados['mob_dataMenorVencimento__T'] <= -19.32664141194029), 11.0,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__T'] > -19.32664141194029, base_dados['mob_dataMenorVencimento__T'] <= -17.117536443286735), 12.0,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__T'] > -17.117536443286735, base_dados['mob_dataMenorVencimento__T'] <= -13.850477850991659), 13.0,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__T'] > -13.850477850991659, base_dados['mob_dataMenorVencimento__T'] <= -11.085666287078604), 14.0,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__T'] > -11.085666287078604, base_dados['mob_dataMenorVencimento__T'] <= -9.025942099751465), 15.0,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__T'] > -9.025942099751465, base_dados['mob_dataMenorVencimento__T'] <= -8.101150386858755), 16.0,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__T'] > -8.101150386858755, base_dados['mob_dataMenorVencimento__T'] <= -7.6559575117484755), 17.0,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__T'] > -7.6559575117484755, base_dados['mob_dataMenorVencimento__T'] <= -7.415216798975123), 18.0,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__T'] > -7.415216798975123, base_dados['mob_dataMenorVencimento__T'] <= -6.090549362987054), 19.0,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__T'] > -6.090549362987054, base_dados['mob_dataMenorVencimento__T'] <= -3.7394976746632196), 20.0,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__T'] > -3.7394976746632196, base_dados['mob_dataMenorVencimento__T'] <= -1.5088961093352076), 21.0,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__T'] > -1.5088961093352076, base_dados['mob_dataMenorVencimento__T'] <= -0.10127221151730681), 22.0,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__T'] > -0.10127221151730681, base_dados['mob_dataMenorVencimento__T'] <= 1.031109683999919), 23.0,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__T'] > 1.031109683999919, base_dados['mob_dataMenorVencimento__T'] <= 2.6498318191070998), 24.0,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__T'] > 2.6498318191070998, base_dados['mob_dataMenorVencimento__T'] <= 4.414208793072406), 25.0,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__T'] > 4.414208793072406, base_dados['mob_dataMenorVencimento__T'] <= 5.930186973444349), 26.0,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__T'] > 5.930186973444349, base_dados['mob_dataMenorVencimento__T'] <= 8.090959434413993), 27.0,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__T'] > 8.090959434413993, base_dados['mob_dataMenorVencimento__T'] <= 11.49344601339165), 28.0,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__T'] > 11.49344601339165, base_dados['mob_dataMenorVencimento__T'] <= 15.036082865440456), 29.0,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__T'] > 15.036082865440456, base_dados['mob_dataMenorVencimento__T'] <= 18.52365681375331), 30.0,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__T'] > 18.52365681375331, base_dados['mob_dataMenorVencimento__T'] <= 21.320573043347302), 31.0,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__T'] > 21.320573043347302, base_dados['mob_dataMenorVencimento__T'] <= 24.107878622423588), 32.0,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__T'] > 24.107878622423588, base_dados['mob_dataMenorVencimento__T'] <= 27.403075989681863), 33.0,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__T'] > 27.403075989681863, base_dados['mob_dataMenorVencimento__T'] <= 33.074761168450046), 34.0,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__T'] > 33.074761168450046, base_dados['mob_dataMenorVencimento__T'] <= 50.4151962161202), 35.0,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__T'] > 50.4151962161202, base_dados['mob_dataMenorVencimento__T'] <= 56.33494261222131), 36.0,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__T'] > 56.33494261222131, base_dados['mob_dataMenorVencimento__T'] <= 93.58430355631006), 37.0,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__T'] > 93.58430355631006, base_dados['mob_dataMenorVencimento__T'] <= 116.25503923641622), 38.0,
np.where(base_dados['mob_dataMenorVencimento__T'] > 116.25503923641622, 39.0,
 0))))))))))))))))))))))))))))))))))))))))
base_dados['mob_dataMenorVencimento__T__pk_40_g_1_1'] = np.where(base_dados['mob_dataMenorVencimento__T__pk_40'] == 0.0, 2,
np.where(base_dados['mob_dataMenorVencimento__T__pk_40'] == 1.0, 2,
np.where(base_dados['mob_dataMenorVencimento__T__pk_40'] == 2.0, 2,
np.where(base_dados['mob_dataMenorVencimento__T__pk_40'] == 3.0, 2,
np.where(base_dados['mob_dataMenorVencimento__T__pk_40'] == 4.0, 2,
np.where(base_dados['mob_dataMenorVencimento__T__pk_40'] == 5.0, 1,
np.where(base_dados['mob_dataMenorVencimento__T__pk_40'] == 6.0, 2,
np.where(base_dados['mob_dataMenorVencimento__T__pk_40'] == 7.0, 2,
np.where(base_dados['mob_dataMenorVencimento__T__pk_40'] == 8.0, 2,
np.where(base_dados['mob_dataMenorVencimento__T__pk_40'] == 9.0, 1,
np.where(base_dados['mob_dataMenorVencimento__T__pk_40'] == 10.0, 2,
np.where(base_dados['mob_dataMenorVencimento__T__pk_40'] == 11.0, 2,
np.where(base_dados['mob_dataMenorVencimento__T__pk_40'] == 12.0, 2,
np.where(base_dados['mob_dataMenorVencimento__T__pk_40'] == 13.0, 1,
np.where(base_dados['mob_dataMenorVencimento__T__pk_40'] == 14.0, 2,
np.where(base_dados['mob_dataMenorVencimento__T__pk_40'] == 15.0, 2,
np.where(base_dados['mob_dataMenorVencimento__T__pk_40'] == 16.0, 2,
np.where(base_dados['mob_dataMenorVencimento__T__pk_40'] == 17.0, 1,
np.where(base_dados['mob_dataMenorVencimento__T__pk_40'] == 18.0, 2,
np.where(base_dados['mob_dataMenorVencimento__T__pk_40'] == 19.0, 1,
np.where(base_dados['mob_dataMenorVencimento__T__pk_40'] == 20.0, 0,
np.where(base_dados['mob_dataMenorVencimento__T__pk_40'] == 21.0, 1,
np.where(base_dados['mob_dataMenorVencimento__T__pk_40'] == 22.0, 0,
np.where(base_dados['mob_dataMenorVencimento__T__pk_40'] == 23.0, 0,
np.where(base_dados['mob_dataMenorVencimento__T__pk_40'] == 24.0, 1,
np.where(base_dados['mob_dataMenorVencimento__T__pk_40'] == 25.0, 1,
np.where(base_dados['mob_dataMenorVencimento__T__pk_40'] == 26.0, 1,
np.where(base_dados['mob_dataMenorVencimento__T__pk_40'] == 27.0, 0,
np.where(base_dados['mob_dataMenorVencimento__T__pk_40'] == 28.0, 1,
np.where(base_dados['mob_dataMenorVencimento__T__pk_40'] == 29.0, 2,
np.where(base_dados['mob_dataMenorVencimento__T__pk_40'] == 30.0, 2,
np.where(base_dados['mob_dataMenorVencimento__T__pk_40'] == 31.0, 2,
np.where(base_dados['mob_dataMenorVencimento__T__pk_40'] == 32.0, 2,
np.where(base_dados['mob_dataMenorVencimento__T__pk_40'] == 33.0, 2,
np.where(base_dados['mob_dataMenorVencimento__T__pk_40'] == 34.0, 2,
np.where(base_dados['mob_dataMenorVencimento__T__pk_40'] == 35.0, 2,
np.where(base_dados['mob_dataMenorVencimento__T__pk_40'] == 36.0, 2,
np.where(base_dados['mob_dataMenorVencimento__T__pk_40'] == 37.0, 2,
np.where(base_dados['mob_dataMenorVencimento__T__pk_40'] == 38.0, 2,
np.where(base_dados['mob_dataMenorVencimento__T__pk_40'] == 39.0, 2,
 0))))))))))))))))))))))))))))))))))))))))
base_dados['mob_dataMenorVencimento__T__pk_40_g_1_2'] = np.where(base_dados['mob_dataMenorVencimento__T__pk_40_g_1_1'] == 0, 2,
np.where(base_dados['mob_dataMenorVencimento__T__pk_40_g_1_1'] == 1, 1,
np.where(base_dados['mob_dataMenorVencimento__T__pk_40_g_1_1'] == 2, 0,
 0)))
base_dados['mob_dataMenorVencimento__T__pk_40_g_1'] = np.where(base_dados['mob_dataMenorVencimento__T__pk_40_g_1_2'] == 0, 0,
np.where(base_dados['mob_dataMenorVencimento__T__pk_40_g_1_2'] == 1, 1,
np.where(base_dados['mob_dataMenorVencimento__T__pk_40_g_1_2'] == 2, 2,
 0)))


# COMMAND ----------

# MAGIC %md
# MAGIC ### Parte 2 de 2

# COMMAND ----------

base_dados['celularPrincipal__p_7_g_1_c1_3_1'] = np.where(np.bitwise_and(base_dados['celularPrincipal__p_7_g_1'] == 0, base_dados['celularPrincipal__p_3_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['celularPrincipal__p_7_g_1'] == 0, base_dados['celularPrincipal__p_3_g_1'] == 1), 0,
np.where(np.bitwise_and(base_dados['celularPrincipal__p_7_g_1'] == 1, base_dados['celularPrincipal__p_3_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['celularPrincipal__p_7_g_1'] == 1, base_dados['celularPrincipal__p_3_g_1'] == 1), 2,
 0))))
base_dados['celularPrincipal__p_7_g_1_c1_3_2'] = np.where(base_dados['celularPrincipal__p_7_g_1_c1_3_1'] == 0, 0,
np.where(base_dados['celularPrincipal__p_7_g_1_c1_3_1'] == 1, 1,
np.where(base_dados['celularPrincipal__p_7_g_1_c1_3_1'] == 2, 2,
0)))
base_dados['celularPrincipal__p_7_g_1_c1_3'] = np.where(base_dados['celularPrincipal__p_7_g_1_c1_3_2'] == 0, 0,
np.where(base_dados['celularPrincipal__p_7_g_1_c1_3_2'] == 1, 1,
np.where(base_dados['celularPrincipal__p_7_g_1_c1_3_2'] == 2, 2,
 0)))
         
         
         
         
                
base_dados['codigo__L__pu_13_g_1_c1_17_1'] = np.where(np.bitwise_and(base_dados['codigo__pu_8_g_1'] == 0, base_dados['codigo__L__pu_13_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['codigo__pu_8_g_1'] == 0, base_dados['codigo__L__pu_13_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['codigo__pu_8_g_1'] == 1, base_dados['codigo__L__pu_13_g_1'] == 0), 2,
np.where(np.bitwise_and(base_dados['codigo__pu_8_g_1'] == 1, base_dados['codigo__L__pu_13_g_1'] == 1), 2,
 0))))
base_dados['codigo__L__pu_13_g_1_c1_17_2'] = np.where(base_dados['codigo__L__pu_13_g_1_c1_17_1'] == 0, 0,
np.where(base_dados['codigo__L__pu_13_g_1_c1_17_1'] == 1, 1,
np.where(base_dados['codigo__L__pu_13_g_1_c1_17_1'] == 2, 2,
0)))
base_dados['codigo__L__pu_13_g_1_c1_17'] = np.where(base_dados['codigo__L__pu_13_g_1_c1_17_2'] == 0, 0,
np.where(base_dados['codigo__L__pu_13_g_1_c1_17_2'] == 1, 1,
np.where(base_dados['codigo__L__pu_13_g_1_c1_17_2'] == 2, 2,
 0)))
         
         
         
         
         
         
         
base_dados['valorVencido__R__pe_13_g_1_c1_19_1'] = np.where(np.bitwise_and(base_dados['valorVencido__R__pe_13_g_1'] == 0, base_dados['valorVencido__L__pu_7_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['valorVencido__R__pe_13_g_1'] == 0, base_dados['valorVencido__L__pu_7_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['valorVencido__R__pe_13_g_1'] == 0, base_dados['valorVencido__L__pu_7_g_1'] == 2), 1,
np.where(np.bitwise_and(base_dados['valorVencido__R__pe_13_g_1'] == 0, base_dados['valorVencido__L__pu_7_g_1'] == 3), 2,
np.where(np.bitwise_and(base_dados['valorVencido__R__pe_13_g_1'] == 1, base_dados['valorVencido__L__pu_7_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['valorVencido__R__pe_13_g_1'] == 1, base_dados['valorVencido__L__pu_7_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['valorVencido__R__pe_13_g_1'] == 1, base_dados['valorVencido__L__pu_7_g_1'] == 2), 2,
np.where(np.bitwise_and(base_dados['valorVencido__R__pe_13_g_1'] == 1, base_dados['valorVencido__L__pu_7_g_1'] == 3), 2,
np.where(np.bitwise_and(base_dados['valorVencido__R__pe_13_g_1'] == 2, base_dados['valorVencido__L__pu_7_g_1'] == 0), 2,
np.where(np.bitwise_and(base_dados['valorVencido__R__pe_13_g_1'] == 2, base_dados['valorVencido__L__pu_7_g_1'] == 1), 3,
np.where(np.bitwise_and(base_dados['valorVencido__R__pe_13_g_1'] == 2, base_dados['valorVencido__L__pu_7_g_1'] == 2), 3,
np.where(np.bitwise_and(base_dados['valorVencido__R__pe_13_g_1'] == 2, base_dados['valorVencido__L__pu_7_g_1'] == 3), 3,
np.where(np.bitwise_and(base_dados['valorVencido__R__pe_13_g_1'] == 3, base_dados['valorVencido__L__pu_7_g_1'] == 0), 4,
np.where(np.bitwise_and(base_dados['valorVencido__R__pe_13_g_1'] == 3, base_dados['valorVencido__L__pu_7_g_1'] == 1), 4,
np.where(np.bitwise_and(base_dados['valorVencido__R__pe_13_g_1'] == 3, base_dados['valorVencido__L__pu_7_g_1'] == 2), 5,
np.where(np.bitwise_and(base_dados['valorVencido__R__pe_13_g_1'] == 3, base_dados['valorVencido__L__pu_7_g_1'] == 3), 5,
np.where(np.bitwise_and(base_dados['valorVencido__R__pe_13_g_1'] == 4, base_dados['valorVencido__L__pu_7_g_1'] == 0), 5,
np.where(np.bitwise_and(base_dados['valorVencido__R__pe_13_g_1'] == 4, base_dados['valorVencido__L__pu_7_g_1'] == 1), 5,
np.where(np.bitwise_and(base_dados['valorVencido__R__pe_13_g_1'] == 4, base_dados['valorVencido__L__pu_7_g_1'] == 2), 6,
np.where(np.bitwise_and(base_dados['valorVencido__R__pe_13_g_1'] == 4, base_dados['valorVencido__L__pu_7_g_1'] == 3), 6,
 0))))))))))))))))))))
base_dados['valorVencido__R__pe_13_g_1_c1_19_2'] = np.where(base_dados['valorVencido__R__pe_13_g_1_c1_19_1'] == 0, 0,
np.where(base_dados['valorVencido__R__pe_13_g_1_c1_19_1'] == 1, 1,
np.where(base_dados['valorVencido__R__pe_13_g_1_c1_19_1'] == 2, 2,
np.where(base_dados['valorVencido__R__pe_13_g_1_c1_19_1'] == 3, 4,
np.where(base_dados['valorVencido__R__pe_13_g_1_c1_19_1'] == 4, 3,
np.where(base_dados['valorVencido__R__pe_13_g_1_c1_19_1'] == 5, 5,
np.where(base_dados['valorVencido__R__pe_13_g_1_c1_19_1'] == 6, 6,
0)))))))
base_dados['valorVencido__R__pe_13_g_1_c1_19'] = np.where(base_dados['valorVencido__R__pe_13_g_1_c1_19_2'] == 0, 0,
np.where(base_dados['valorVencido__R__pe_13_g_1_c1_19_2'] == 1, 1,
np.where(base_dados['valorVencido__R__pe_13_g_1_c1_19_2'] == 2, 2,
np.where(base_dados['valorVencido__R__pe_13_g_1_c1_19_2'] == 3, 3,
np.where(base_dados['valorVencido__R__pe_13_g_1_c1_19_2'] == 4, 4,
np.where(base_dados['valorVencido__R__pe_13_g_1_c1_19_2'] == 5, 5,
np.where(base_dados['valorVencido__R__pe_13_g_1_c1_19_2'] == 6, 6,
 0)))))))
         
         
         
         
                
base_dados['mob_dataMenorVencimento__L__pe_5_g_1_c1_50_1'] = np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__L__pe_5_g_1'] == 0, base_dados['mob_dataMenorVencimento__T__pk_40_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__L__pe_5_g_1'] == 0, base_dados['mob_dataMenorVencimento__T__pk_40_g_1'] == 1), 0,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__L__pe_5_g_1'] == 0, base_dados['mob_dataMenorVencimento__T__pk_40_g_1'] == 2), 1,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__L__pe_5_g_1'] == 1, base_dados['mob_dataMenorVencimento__T__pk_40_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__L__pe_5_g_1'] == 1, base_dados['mob_dataMenorVencimento__T__pk_40_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__L__pe_5_g_1'] == 1, base_dados['mob_dataMenorVencimento__T__pk_40_g_1'] == 2), 3,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__L__pe_5_g_1'] == 2, base_dados['mob_dataMenorVencimento__T__pk_40_g_1'] == 0), 3,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__L__pe_5_g_1'] == 2, base_dados['mob_dataMenorVencimento__T__pk_40_g_1'] == 1), 4,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__L__pe_5_g_1'] == 2, base_dados['mob_dataMenorVencimento__T__pk_40_g_1'] == 2), 4,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__L__pe_5_g_1'] == 3, base_dados['mob_dataMenorVencimento__T__pk_40_g_1'] == 0), 5,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__L__pe_5_g_1'] == 3, base_dados['mob_dataMenorVencimento__T__pk_40_g_1'] == 1), 5,
np.where(np.bitwise_and(base_dados['mob_dataMenorVencimento__L__pe_5_g_1'] == 3, base_dados['mob_dataMenorVencimento__T__pk_40_g_1'] == 2), 5,
 0))))))))))))
base_dados['mob_dataMenorVencimento__L__pe_5_g_1_c1_50_2'] = np.where(base_dados['mob_dataMenorVencimento__L__pe_5_g_1_c1_50_1'] == 0, 0,
np.where(base_dados['mob_dataMenorVencimento__L__pe_5_g_1_c1_50_1'] == 1, 1,
np.where(base_dados['mob_dataMenorVencimento__L__pe_5_g_1_c1_50_1'] == 2, 3,
np.where(base_dados['mob_dataMenorVencimento__L__pe_5_g_1_c1_50_1'] == 3, 2,
np.where(base_dados['mob_dataMenorVencimento__L__pe_5_g_1_c1_50_1'] == 4, 4,
np.where(base_dados['mob_dataMenorVencimento__L__pe_5_g_1_c1_50_1'] == 5, 5,
0))))))
base_dados['mob_dataMenorVencimento__L__pe_5_g_1_c1_50'] = np.where(base_dados['mob_dataMenorVencimento__L__pe_5_g_1_c1_50_2'] == 0, 0,
np.where(base_dados['mob_dataMenorVencimento__L__pe_5_g_1_c1_50_2'] == 1, 1,
np.where(base_dados['mob_dataMenorVencimento__L__pe_5_g_1_c1_50_2'] == 2, 2,
np.where(base_dados['mob_dataMenorVencimento__L__pe_5_g_1_c1_50_2'] == 3, 3,
np.where(base_dados['mob_dataMenorVencimento__L__pe_5_g_1_c1_50_2'] == 4, 4,
np.where(base_dados['mob_dataMenorVencimento__L__pe_5_g_1_c1_50_2'] == 5, 5,
 0))))))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Mantendo apenas as variáveis do modelo

# COMMAND ----------

caminho_model_completo = os.path.join(caminho_pickle, model)
model_file = os.listdir(caminho_model_completo)[0]
caminho_model_completo = os.path.join(caminho_model_completo, model_file)
caminho_model_completo

# COMMAND ----------

import pickle
modelo=pickle.load(open(caminho_model_completo, 'rb'))

base_teste_c0 = base_dados[[chave,'codigo__L__pu_13_g_1_c1_17','mob_dataMenorVencimento__L__pe_5_g_1_c1_50','valorVencido__R__pe_13_g_1_c1_19','P_emailPrincipal_gh38','celularPrincipal__p_7_g_1_c1_3','diaVencimentoCartao_gh38']]

var_fin_c0=['codigo__L__pu_13_g_1_c1_17','mob_dataMenorVencimento__L__pe_5_g_1_c1_50','valorVencido__R__pe_13_g_1_c1_19','P_emailPrincipal_gh38','celularPrincipal__p_7_g_1_c1_3','diaVencimentoCartao_gh38']

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

x_teste2['P_1_R'] = np.sqrt(x_teste2['P_1'])
x_teste2['P_1_R'] = np.where(x_teste2['P_1'] == 0, -1, x_teste2['P_1_R'])
x_teste2['P_1_R'] = x_teste2['P_1_R'].fillna(-2)
x_teste2['P_1_R'] = x_teste2['P_1_R'].fillna(-2)

x_teste2['P_1_R_p_25_g_1'] = np.where(x_teste2['P_1_R'] <= 0.208265675, 1,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.208265675, x_teste2['P_1_R'] <= 0.250345973), 0,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.250345973, x_teste2['P_1_R'] <= 0.310910629), 2,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.310910629, x_teste2['P_1_R'] <= 0.559476406), 3,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.559476406, x_teste2['P_1_R'] <= 0.747117426), 4,5)))))

x_teste2['P_1_p_40_g_1'] = np.where(x_teste2['P_1'] <= 0.089773939, 0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.089773939, x_teste2['P_1'] <= 0.178999226), 1,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.178999226, x_teste2['P_1'] <= 0.352827553), 2,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.352827553, x_teste2['P_1'] <= 0.584120216), 3,4))))

x_teste2['GH'] = np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 0, x_teste2['P_1_R_p_25_g_1'] == 0), 0,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 0, x_teste2['P_1_R_p_25_g_1'] == 1), 1,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 0, x_teste2['P_1_R_p_25_g_1'] == 2), 2,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 0, x_teste2['P_1_R_p_25_g_1'] == 3), 2,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 0, x_teste2['P_1_R_p_25_g_1'] == 4), 3,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 0, x_teste2['P_1_R_p_25_g_1'] == 5), 3,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 1, x_teste2['P_1_R_p_25_g_1'] == 0), 2,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 1, x_teste2['P_1_R_p_25_g_1'] == 1), 2,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 1, x_teste2['P_1_R_p_25_g_1'] == 2), 3,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 1, x_teste2['P_1_R_p_25_g_1'] == 3), 3,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 1, x_teste2['P_1_R_p_25_g_1'] == 4), 3,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 1, x_teste2['P_1_R_p_25_g_1'] == 5), 4,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 2, x_teste2['P_1_R_p_25_g_1'] == 0), 3,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 2, x_teste2['P_1_R_p_25_g_1'] == 1), 3,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 2, x_teste2['P_1_R_p_25_g_1'] == 2), 4,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 2, x_teste2['P_1_R_p_25_g_1'] == 3), 4,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 2, x_teste2['P_1_R_p_25_g_1'] == 4), 4,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 2, x_teste2['P_1_R_p_25_g_1'] == 5), 5,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 3, x_teste2['P_1_R_p_25_g_1'] == 0), 3,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 3, x_teste2['P_1_R_p_25_g_1'] == 1), 3,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 3, x_teste2['P_1_R_p_25_g_1'] == 2), 4,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 3, x_teste2['P_1_R_p_25_g_1'] == 3), 4,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 3, x_teste2['P_1_R_p_25_g_1'] == 4), 5,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 3, x_teste2['P_1_R_p_25_g_1'] == 5), 5,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 4, x_teste2['P_1_R_p_25_g_1'] == 0), 4,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 4, x_teste2['P_1_R_p_25_g_1'] == 1), 4,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 4, x_teste2['P_1_R_p_25_g_1'] == 2), 5,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 4, x_teste2['P_1_R_p_25_g_1'] == 3), 5,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 4, x_teste2['P_1_R_p_25_g_1'] == 4), 6,
    np.where(np.bitwise_and(x_teste2['P_1_p_40_g_1'] == 4, x_teste2['P_1_R_p_25_g_1'] == 5), 6,0))))))))))))))))))))))))))))))

del x_teste2['P_1_R']
del x_teste2['P_1_R_p_25_g_1']
del x_teste2['P_1_p_40_g_1']
x_teste2

# COMMAND ----------

x_teste2.groupby(['GH'])['P_1'].count()

# COMMAND ----------

import datetime
data_escrita = datetime.datetime.today()
nome_escrito = 'pre_output:'+str(data_escrita.year)+str(data_escrita.month).zfill(2)+str(data_escrita.day).zfill(2)+'.csv'
nome_escrito

# COMMAND ----------

try:
  dbutils.fs.mkdirs(caminho_output.replace('/dbfs',''))
except:
  pass
x_teste2.to_csv(open(os.path.join(caminho_output, nome_escrito),'wb'))