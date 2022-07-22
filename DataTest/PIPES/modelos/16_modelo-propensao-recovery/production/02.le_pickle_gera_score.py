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
chave = 'document'

caminho_base = "/dbfs/mnt/ml-prd/ml-data/propensaodeal/recovery/trusted/"
list_base = os.listdir(caminho_base)

#Nome da Base de Dados
N_Base = max(list_base)
dt_max = N_Base.split('.')[0]
dt_max = (dt_max[::-1][0:8])[::-1]

#Separador
separador_ = ";"

#Decimal
decimal_ = "."

caminho_trusted = '/mnt/ml-prd/ml-data/propensaodeal/recovery/trusted'
caminho_trusted_dbfs = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/recovery/trusted'

pickle_path = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/recovery/pickle_models/'

outputpath = 'mnt/ml-prd/ml-data/propensaodeal/recovery/output/'
outputpath_dbfs = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/recovery/output/'

N_Base, dt_max

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importação da Base de Dados

# COMMAND ----------

#carregar o arquivo em formato tabela
base_dados = pd.read_csv(caminho_base+N_Base, sep=separador_, decimal=decimal_)

del base_dados['contrato']


del base_dados['Class_Produto']
del base_dados['Class_Portfolio']

base_dados['Data_Mora'] = pd.to_datetime(base_dados['Data_Mora'])
base_dados['MOB_MORA'] = ((datetime.today()) - base_dados.Data_Mora)/np.timedelta64(1, 'M')

del base_dados['Data_Mora']

base_dados = base_dados.apply(pd.to_numeric, errors='ignore')

base_dados.fillna(-3)

base_dados['document'] = base_dados['document'].replace(np.nan, '-3')

base_dados['document'] = base_dados['document'].astype(np.int64)
base_dados['IdContatoSIR'] = base_dados['IdContatoSIR'].astype(np.int64)

print("shape da Base de Dados:",base_dados.shape)

base_dados.head()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Gerando variáveis Categóricas

# COMMAND ----------

base_dados['Class_Carteira_gh30'] = np.where(base_dados['Class_Carteira'] == 'a', 0,
np.where(base_dados['Class_Carteira'] == 'b', 1,
np.where(base_dados['Class_Carteira'] == 'c', 2,
np.where(base_dados['Class_Carteira'] == 'd', 3,
np.where(base_dados['Class_Carteira'] == 'e', 4,
np.where(base_dados['Class_Carteira'] == 'f', 5,
np.where(base_dados['Class_Carteira'] == 'g', 6,
np.where(base_dados['Class_Carteira'] == 'h', 7,
np.where(base_dados['Class_Carteira'] == 'i', 8,
np.where(base_dados['Class_Carteira'] == 'j', 9,
np.where(base_dados['Class_Carteira'] == 'k', 10,
0)))))))))))
base_dados['Class_Carteira_gh31'] = np.where(base_dados['Class_Carteira_gh30'] == 0, 0,
np.where(base_dados['Class_Carteira_gh30'] == 1, 1,
np.where(base_dados['Class_Carteira_gh30'] == 2, 2,
np.where(base_dados['Class_Carteira_gh30'] == 3, 3,
np.where(base_dados['Class_Carteira_gh30'] == 4, 4,
np.where(base_dados['Class_Carteira_gh30'] == 5, 5,
np.where(base_dados['Class_Carteira_gh30'] == 6, 6,
np.where(base_dados['Class_Carteira_gh30'] == 7, 7,
np.where(base_dados['Class_Carteira_gh30'] == 8, 8,
np.where(base_dados['Class_Carteira_gh30'] == 9, 9,
np.where(base_dados['Class_Carteira_gh30'] == 10, 10,
0)))))))))))
base_dados['Class_Carteira_gh32'] = np.where(base_dados['Class_Carteira_gh31'] == 0, 0,
np.where(base_dados['Class_Carteira_gh31'] == 1, 1,
np.where(base_dados['Class_Carteira_gh31'] == 2, 2,
np.where(base_dados['Class_Carteira_gh31'] == 3, 3,
np.where(base_dados['Class_Carteira_gh31'] == 4, 4,
np.where(base_dados['Class_Carteira_gh31'] == 5, 5,
np.where(base_dados['Class_Carteira_gh31'] == 6, 6,
np.where(base_dados['Class_Carteira_gh31'] == 7, 7,
np.where(base_dados['Class_Carteira_gh31'] == 8, 8,
np.where(base_dados['Class_Carteira_gh31'] == 9, 9,
np.where(base_dados['Class_Carteira_gh31'] == 10, 10,
0)))))))))))
base_dados['Class_Carteira_gh33'] = np.where(base_dados['Class_Carteira_gh32'] == 0, 0,
np.where(base_dados['Class_Carteira_gh32'] == 1, 1,
np.where(base_dados['Class_Carteira_gh32'] == 2, 2,
np.where(base_dados['Class_Carteira_gh32'] == 3, 3,
np.where(base_dados['Class_Carteira_gh32'] == 4, 4,
np.where(base_dados['Class_Carteira_gh32'] == 5, 5,
np.where(base_dados['Class_Carteira_gh32'] == 6, 6,
np.where(base_dados['Class_Carteira_gh32'] == 7, 7,
np.where(base_dados['Class_Carteira_gh32'] == 8, 8,
np.where(base_dados['Class_Carteira_gh32'] == 9, 9,
np.where(base_dados['Class_Carteira_gh32'] == 10, 10,
0)))))))))))
base_dados['Class_Carteira_gh34'] = np.where(base_dados['Class_Carteira_gh33'] == 0, 0,
np.where(base_dados['Class_Carteira_gh33'] == 1, 1,
np.where(base_dados['Class_Carteira_gh33'] == 2, 2,
np.where(base_dados['Class_Carteira_gh33'] == 3, 3,
np.where(base_dados['Class_Carteira_gh33'] == 4, 0,
np.where(base_dados['Class_Carteira_gh33'] == 5, 0,
np.where(base_dados['Class_Carteira_gh33'] == 6, 0,
np.where(base_dados['Class_Carteira_gh33'] == 7, 1,
np.where(base_dados['Class_Carteira_gh33'] == 8, 3,
np.where(base_dados['Class_Carteira_gh33'] == 9, 0,
np.where(base_dados['Class_Carteira_gh33'] == 10, 10,
0)))))))))))
base_dados['Class_Carteira_gh35'] = np.where(base_dados['Class_Carteira_gh34'] == 0, 0,
np.where(base_dados['Class_Carteira_gh34'] == 1, 1,
np.where(base_dados['Class_Carteira_gh34'] == 2, 2,
np.where(base_dados['Class_Carteira_gh34'] == 3, 3,
np.where(base_dados['Class_Carteira_gh34'] == 10, 4,
0)))))
base_dados['Class_Carteira_gh36'] = np.where(base_dados['Class_Carteira_gh35'] == 0, 0,
np.where(base_dados['Class_Carteira_gh35'] == 1, 1,
np.where(base_dados['Class_Carteira_gh35'] == 2, 2,
np.where(base_dados['Class_Carteira_gh35'] == 3, 4,
np.where(base_dados['Class_Carteira_gh35'] == 4, 2,
0)))))
base_dados['Class_Carteira_gh37'] = np.where(base_dados['Class_Carteira_gh36'] == 0, 0,
np.where(base_dados['Class_Carteira_gh36'] == 1, 1,
np.where(base_dados['Class_Carteira_gh36'] == 2, 2,
np.where(base_dados['Class_Carteira_gh36'] == 4, 2,
0))))
base_dados['Class_Carteira_gh38'] = np.where(base_dados['Class_Carteira_gh37'] == 0, 0,
np.where(base_dados['Class_Carteira_gh37'] == 1, 1,
np.where(base_dados['Class_Carteira_gh37'] == 2, 2,
0)))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Gerando variáveis numéricas contínuas

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parte 1 de 2

# COMMAND ----------

base_dados['VlDividaAtualizado__R'] = np.sqrt(base_dados['VlDividaAtualizado'])
np.where(base_dados['VlDividaAtualizado__R'] == 0, -1, base_dados['VlDividaAtualizado__R'])
base_dados['VlDividaAtualizado__R'] = base_dados['VlDividaAtualizado__R'].fillna(-2)
base_dados['VlDividaAtualizado__R__p_7'] = np.where(base_dados['VlDividaAtualizado__R'] <= 20.50695247958604, 0.0,
np.where(np.bitwise_and(base_dados['VlDividaAtualizado__R'] > 20.50695247958604, base_dados['VlDividaAtualizado__R'] <= 28.62374888095548), 1.0,
np.where(np.bitwise_and(base_dados['VlDividaAtualizado__R'] > 28.62374888095548, base_dados['VlDividaAtualizado__R'] <= 37.15439005016769), 2.0,
np.where(np.bitwise_and(base_dados['VlDividaAtualizado__R'] > 37.15439005016769, base_dados['VlDividaAtualizado__R'] <= 47.4622407814886), 3.0,
np.where(np.bitwise_and(base_dados['VlDividaAtualizado__R'] > 47.4622407814886, base_dados['VlDividaAtualizado__R'] <= 61.33517750850648), 4.0,
np.where(np.bitwise_and(base_dados['VlDividaAtualizado__R'] > 61.33517750850648, base_dados['VlDividaAtualizado__R'] <= 90.50707320425293), 5.0,
np.where(base_dados['VlDividaAtualizado__R'] > 90.50707320425293, 6.0,
 0)))))))
base_dados['VlDividaAtualizado__R__p_7_g_1_1'] = np.where(base_dados['VlDividaAtualizado__R__p_7'] == 0.0, 0,
np.where(base_dados['VlDividaAtualizado__R__p_7'] == 1.0, 4,
np.where(base_dados['VlDividaAtualizado__R__p_7'] == 2.0, 2,
np.where(base_dados['VlDividaAtualizado__R__p_7'] == 3.0, 4,
np.where(base_dados['VlDividaAtualizado__R__p_7'] == 4.0, 5,
np.where(base_dados['VlDividaAtualizado__R__p_7'] == 5.0, 1,
np.where(base_dados['VlDividaAtualizado__R__p_7'] == 6.0, 3,
 0)))))))
base_dados['VlDividaAtualizado__R__p_7_g_1_2'] = np.where(base_dados['VlDividaAtualizado__R__p_7_g_1_1'] == 0, 3,
np.where(base_dados['VlDividaAtualizado__R__p_7_g_1_1'] == 1, 1,
np.where(base_dados['VlDividaAtualizado__R__p_7_g_1_1'] == 2, 3,
np.where(base_dados['VlDividaAtualizado__R__p_7_g_1_1'] == 3, 0,
np.where(base_dados['VlDividaAtualizado__R__p_7_g_1_1'] == 4, 3,
np.where(base_dados['VlDividaAtualizado__R__p_7_g_1_1'] == 5, 2,
 0))))))
base_dados['VlDividaAtualizado__R__p_7_g_1'] = np.where(base_dados['VlDividaAtualizado__R__p_7_g_1_2'] == 0, 0,
np.where(base_dados['VlDividaAtualizado__R__p_7_g_1_2'] == 1, 1,
np.where(base_dados['VlDividaAtualizado__R__p_7_g_1_2'] == 2, 2,
np.where(base_dados['VlDividaAtualizado__R__p_7_g_1_2'] == 3, 3,
 0))))
         
         
         
         
         
         
         
base_dados['VlDividaAtualizado__L'] = np.log(base_dados['VlDividaAtualizado'])
np.where(base_dados['VlDividaAtualizado__L'] == 0, -1, base_dados['VlDividaAtualizado__L'])
base_dados['VlDividaAtualizado__L'] = base_dados['VlDividaAtualizado__L'].fillna(-2)
base_dados['VlDividaAtualizado__L__p_2'] = np.where(base_dados['VlDividaAtualizado__L'] <= 7.495050545390012, 0.0,
np.where(base_dados['VlDividaAtualizado__L'] > 7.495050545390012, 1.0,
 0))
base_dados['VlDividaAtualizado__L__p_2_g_1_1'] = np.where(base_dados['VlDividaAtualizado__L__p_2'] == 0.0, 0,
np.where(base_dados['VlDividaAtualizado__L__p_2'] == 1.0, 1,
 0))
base_dados['VlDividaAtualizado__L__p_2_g_1_2'] = np.where(base_dados['VlDividaAtualizado__L__p_2_g_1_1'] == 0, 1,
np.where(base_dados['VlDividaAtualizado__L__p_2_g_1_1'] == 1, 0,
 0))
base_dados['VlDividaAtualizado__L__p_2_g_1'] = np.where(base_dados['VlDividaAtualizado__L__p_2_g_1_2'] == 0, 0,
np.where(base_dados['VlDividaAtualizado__L__p_2_g_1_2'] == 1, 1,
 0))
                                                        
                                                        
                                                        
                                                        
                                                        
                                                        
                                                        
base_dados['IdContatoSIR__L'] = np.log(base_dados['IdContatoSIR'])
np.where(base_dados['IdContatoSIR__L'] == 0, -1, base_dados['IdContatoSIR__L'])
base_dados['IdContatoSIR__L'] = base_dados['IdContatoSIR__L'].fillna(-2)
base_dados['IdContatoSIR__L__pu_10'] = np.where(base_dados['IdContatoSIR__L'] <= 13.589862623298329, 0.0,
np.where(np.bitwise_and(base_dados['IdContatoSIR__L'] > 13.589862623298329, base_dados['IdContatoSIR__L'] <= 14.03823402105378), 1.0,
np.where(np.bitwise_and(base_dados['IdContatoSIR__L'] > 14.03823402105378, base_dados['IdContatoSIR__L'] <= 14.470350807744628), 2.0,
np.where(np.bitwise_and(base_dados['IdContatoSIR__L'] > 14.470350807744628, base_dados['IdContatoSIR__L'] <= 14.82063783353084), 3.0,
np.where(np.bitwise_and(base_dados['IdContatoSIR__L'] > 14.82063783353084, base_dados['IdContatoSIR__L'] <= 15.329320440084928), 4.0,
np.where(np.bitwise_and(base_dados['IdContatoSIR__L'] > 15.329320440084928, base_dados['IdContatoSIR__L'] <= 15.760008996739655), 5.0,
np.where(np.bitwise_and(base_dados['IdContatoSIR__L'] > 15.760008996739655, base_dados['IdContatoSIR__L'] <= 16.19160983256761), 6.0,
np.where(np.bitwise_and(base_dados['IdContatoSIR__L'] > 16.19160983256761, base_dados['IdContatoSIR__L'] <= 16.621781970431318), 7.0,
np.where(np.bitwise_and(base_dados['IdContatoSIR__L'] > 16.621781970431318, base_dados['IdContatoSIR__L'] <= 17.05246968980692), 8.0,
np.where(base_dados['IdContatoSIR__L'] > 17.05246968980692, 9.0,
 0))))))))))
base_dados['IdContatoSIR__L__pu_10_g_1_1'] = np.where(base_dados['IdContatoSIR__L__pu_10'] == 0.0, 4,
np.where(base_dados['IdContatoSIR__L__pu_10'] == 1.0, 3,
np.where(base_dados['IdContatoSIR__L__pu_10'] == 2.0, 3,
np.where(base_dados['IdContatoSIR__L__pu_10'] == 3.0, 3,
np.where(base_dados['IdContatoSIR__L__pu_10'] == 4.0, 4,
np.where(base_dados['IdContatoSIR__L__pu_10'] == 5.0, 4,
np.where(base_dados['IdContatoSIR__L__pu_10'] == 6.0, 0,
np.where(base_dados['IdContatoSIR__L__pu_10'] == 7.0, 2,
np.where(base_dados['IdContatoSIR__L__pu_10'] == 8.0, 3,
np.where(base_dados['IdContatoSIR__L__pu_10'] == 9.0, 1,
 0))))))))))
base_dados['IdContatoSIR__L__pu_10_g_1_2'] = np.where(base_dados['IdContatoSIR__L__pu_10_g_1_1'] == 0, 3,
np.where(base_dados['IdContatoSIR__L__pu_10_g_1_1'] == 1, 0,
np.where(base_dados['IdContatoSIR__L__pu_10_g_1_1'] == 2, 2,
np.where(base_dados['IdContatoSIR__L__pu_10_g_1_1'] == 3, 3,
np.where(base_dados['IdContatoSIR__L__pu_10_g_1_1'] == 4, 0,
 0)))))
base_dados['IdContatoSIR__L__pu_10_g_1'] = np.where(base_dados['IdContatoSIR__L__pu_10_g_1_2'] == 0, 0,
np.where(base_dados['IdContatoSIR__L__pu_10_g_1_2'] == 2, 1,
np.where(base_dados['IdContatoSIR__L__pu_10_g_1_2'] == 3, 2,
 0)))
         
         
         
         
         
         
         
base_dados['IdContatoSIR__L'] = np.log(base_dados['IdContatoSIR'])
np.where(base_dados['IdContatoSIR__L'] == 0, -1, base_dados['IdContatoSIR__L'])
base_dados['IdContatoSIR__L'] = base_dados['IdContatoSIR__L'].fillna(-2)
base_dados['IdContatoSIR__L__pk_10'] = np.where(base_dados['IdContatoSIR__L'] <= 13.589862623298329, 0.0,
np.where(np.bitwise_and(base_dados['IdContatoSIR__L'] > 13.589862623298329, base_dados['IdContatoSIR__L'] <= 14.049764954592986), 1.0,
np.where(np.bitwise_and(base_dados['IdContatoSIR__L'] > 14.049764954592986, base_dados['IdContatoSIR__L'] <= 14.44225508096349), 2.0,
np.where(np.bitwise_and(base_dados['IdContatoSIR__L'] > 14.44225508096349, base_dados['IdContatoSIR__L'] <= 14.82063783353084), 3.0,
np.where(np.bitwise_and(base_dados['IdContatoSIR__L'] > 14.82063783353084, base_dados['IdContatoSIR__L'] <= 15.37357727824003), 4.0,
np.where(np.bitwise_and(base_dados['IdContatoSIR__L'] > 15.37357727824003, base_dados['IdContatoSIR__L'] <= 15.836914402288864), 5.0,
np.where(np.bitwise_and(base_dados['IdContatoSIR__L'] > 15.836914402288864, base_dados['IdContatoSIR__L'] <= 16.291014053004286), 6.0,
np.where(np.bitwise_and(base_dados['IdContatoSIR__L'] > 16.291014053004286, base_dados['IdContatoSIR__L'] <= 16.73126737158355), 7.0,
np.where(np.bitwise_and(base_dados['IdContatoSIR__L'] > 16.73126737158355, base_dados['IdContatoSIR__L'] <= 17.103873377553764), 8.0,
np.where(base_dados['IdContatoSIR__L'] > 17.103873377553764, 9.0,
 0))))))))))
base_dados['IdContatoSIR__L__pk_10_g_1_1'] = np.where(base_dados['IdContatoSIR__L__pk_10'] == 0.0, 3,
np.where(base_dados['IdContatoSIR__L__pk_10'] == 1.0, 2,
np.where(base_dados['IdContatoSIR__L__pk_10'] == 2.0, 2,
np.where(base_dados['IdContatoSIR__L__pk_10'] == 3.0, 2,
np.where(base_dados['IdContatoSIR__L__pk_10'] == 4.0, 3,
np.where(base_dados['IdContatoSIR__L__pk_10'] == 5.0, 3,
np.where(base_dados['IdContatoSIR__L__pk_10'] == 6.0, 0,
np.where(base_dados['IdContatoSIR__L__pk_10'] == 7.0, 2,
np.where(base_dados['IdContatoSIR__L__pk_10'] == 8.0, 2,
np.where(base_dados['IdContatoSIR__L__pk_10'] == 9.0, 1,
 0))))))))))
base_dados['IdContatoSIR__L__pk_10_g_1_2'] = np.where(base_dados['IdContatoSIR__L__pk_10_g_1_1'] == 0, 2,
np.where(base_dados['IdContatoSIR__L__pk_10_g_1_1'] == 1, 1,
np.where(base_dados['IdContatoSIR__L__pk_10_g_1_1'] == 2, 2,
np.where(base_dados['IdContatoSIR__L__pk_10_g_1_1'] == 3, 0,
 0))))
base_dados['IdContatoSIR__L__pk_10_g_1'] = np.where(base_dados['IdContatoSIR__L__pk_10_g_1_2'] == 0, 0,
np.where(base_dados['IdContatoSIR__L__pk_10_g_1_2'] == 1, 1,
np.where(base_dados['IdContatoSIR__L__pk_10_g_1_2'] == 2, 2,
 0)))
         
         
         
         
         
         
         
base_dados['document__pe_13'] = np.where(np.bitwise_and(base_dados['document'] >= -3.0, base_dados['document'] <= 7299207710.0), 0.0,
np.where(np.bitwise_and(base_dados['document'] > 7299207710.0, base_dados['document'] <= 14600660811.0), 1.0,
np.where(np.bitwise_and(base_dados['document'] > 14600660811.0, base_dados['document'] <= 21887381600.0), 2.0,
np.where(np.bitwise_and(base_dados['document'] > 21887381600.0, base_dados['document'] <= 29179145809.0), 3.0,
np.where(np.bitwise_and(base_dados['document'] > 29179145809.0, base_dados['document'] <= 36468259878.0), 4.0,
np.where(np.bitwise_and(base_dados['document'] > 36468259878.0, base_dados['document'] <= 43839402859.0), 5.0,
np.where(np.bitwise_and(base_dados['document'] > 43839402859.0, base_dados['document'] <= 50946889791.0), 6.0,
np.where(np.bitwise_and(base_dados['document'] > 50946889791.0, base_dados['document'] <= 58403280572.0), 7.0,
np.where(np.bitwise_and(base_dados['document'] > 58403280572.0, base_dados['document'] <= 65657128087.0), 8.0,
np.where(np.bitwise_and(base_dados['document'] > 65657128087.0, base_dados['document'] <= 73026034649.0), 9.0,
np.where(np.bitwise_and(base_dados['document'] > 73026034649.0, base_dados['document'] <= 80362800278.0), 10.0,
np.where(np.bitwise_and(base_dados['document'] > 80362800278.0, base_dados['document'] <= 87731096904.0), 11.0,
np.where(base_dados['document'] > 87731096904.0, 12.0,
 -2)))))))))))))
base_dados['document__pe_13_g_1_1'] = np.where(base_dados['document__pe_13'] == -2.0, 4,
np.where(base_dados['document__pe_13'] == 0.0, 1,
np.where(base_dados['document__pe_13'] == 1.0, 0,
np.where(base_dados['document__pe_13'] == 2.0, 3,
np.where(base_dados['document__pe_13'] == 3.0, 1,
np.where(base_dados['document__pe_13'] == 4.0, 1,
np.where(base_dados['document__pe_13'] == 5.0, 1,
np.where(base_dados['document__pe_13'] == 6.0, 3,
np.where(base_dados['document__pe_13'] == 7.0, 2,
np.where(base_dados['document__pe_13'] == 8.0, 2,
np.where(base_dados['document__pe_13'] == 9.0, 3,
np.where(base_dados['document__pe_13'] == 10.0, 2,
np.where(base_dados['document__pe_13'] == 11.0, 2,
np.where(base_dados['document__pe_13'] == 12.0, 1,
 0))))))))))))))
base_dados['document__pe_13_g_1_2'] = np.where(base_dados['document__pe_13_g_1_1'] == 0, 1,
np.where(base_dados['document__pe_13_g_1_1'] == 1, 3,
np.where(base_dados['document__pe_13_g_1_1'] == 2, 1,
np.where(base_dados['document__pe_13_g_1_1'] == 3, 3,
np.where(base_dados['document__pe_13_g_1_1'] == 4, 0,
 0)))))
base_dados['document__pe_13_g_1'] = np.where(base_dados['document__pe_13_g_1_2'] == 0, 0,
np.where(base_dados['document__pe_13_g_1_2'] == 1, 1,
np.where(base_dados['document__pe_13_g_1_2'] == 3, 2,
 0)))
         
         
         
         
         
         
         
base_dados['document__L'] = np.log(base_dados['document'])
np.where(base_dados['document__L'] == 0, -1, base_dados['document__L'])
base_dados['document__L'] = base_dados['document__L'].fillna(-2)
base_dados['document__L__pk_8'] = np.where(base_dados['document__L'] <= -2.0, 0.0,
np.where(np.bitwise_and(base_dados['document__L'] > -2.0, base_dados['document__L'] <= 15.74079866981307), 1.0,
np.where(np.bitwise_and(base_dados['document__L'] > 15.74079866981307, base_dados['document__L'] <= 19.67350973026851), 2.0,
np.where(np.bitwise_and(base_dados['document__L'] > 19.67350973026851, base_dados['document__L'] <= 20.92460117076028), 3.0,
np.where(np.bitwise_and(base_dados['document__L'] > 20.92460117076028, base_dados['document__L'] <= 21.96654977620284), 4.0,
np.where(np.bitwise_and(base_dados['document__L'] > 21.96654977620284, base_dados['document__L'] <= 22.947007689077235), 5.0,
np.where(np.bitwise_and(base_dados['document__L'] > 22.947007689077235, base_dados['document__L'] <= 24.064318533177154), 6.0,
np.where(base_dados['document__L'] > 24.064318533177154, 7.0,
 0))))))))
base_dados['document__L__pk_8_g_1_1'] = np.where(base_dados['document__L__pk_8'] == 0.0, 2,
np.where(base_dados['document__L__pk_8'] == 1.0, 1,
np.where(base_dados['document__L__pk_8'] == 2.0, 1,
np.where(base_dados['document__L__pk_8'] == 3.0, 0,
np.where(base_dados['document__L__pk_8'] == 4.0, 0,
np.where(base_dados['document__L__pk_8'] == 5.0, 0,
np.where(base_dados['document__L__pk_8'] == 6.0, 1,
np.where(base_dados['document__L__pk_8'] == 7.0, 1,
 0))))))))
base_dados['document__L__pk_8_g_1_2'] = np.where(base_dados['document__L__pk_8_g_1_1'] == 0, 2,
np.where(base_dados['document__L__pk_8_g_1_1'] == 1, 1,
np.where(base_dados['document__L__pk_8_g_1_1'] == 2, 0,
 0)))
base_dados['document__L__pk_8_g_1'] = np.where(base_dados['document__L__pk_8_g_1_2'] == 0, 0,
np.where(base_dados['document__L__pk_8_g_1_2'] == 1, 1,
np.where(base_dados['document__L__pk_8_g_1_2'] == 2, 2,
 0)))
         
         
         
         
         
         
base_dados['MOB_MORA__pe_5'] = np.where(base_dados['MOB_MORA'] <= 52.279498267632306, 0.0,
np.where(np.bitwise_and(base_dados['MOB_MORA'] > 52.279498267632306, base_dados['MOB_MORA'] <= 104.58447372914075), 1.0,
np.where(np.bitwise_and(base_dados['MOB_MORA'] > 104.58447372914075, base_dados['MOB_MORA'] <= 157.02086872698467), 2.0,
np.where(np.bitwise_and(base_dados['MOB_MORA'] > 157.02086872698467, base_dados['MOB_MORA'] <= 208.73445627498361), 3.0,
np.where(np.bitwise_and(base_dados['MOB_MORA'] > 208.73445627498361, base_dados['MOB_MORA'] <= 261.40083546141454), 4.0,
 -2)))))
base_dados['MOB_MORA__pe_5_g_1_1'] = np.where(base_dados['MOB_MORA__pe_5'] == -2.0, 0,
np.where(base_dados['MOB_MORA__pe_5'] == 0.0, 1,
np.where(base_dados['MOB_MORA__pe_5'] == 1.0, 0,
np.where(base_dados['MOB_MORA__pe_5'] == 2.0, 0,
np.where(base_dados['MOB_MORA__pe_5'] == 3.0, 0,
np.where(base_dados['MOB_MORA__pe_5'] == 4.0, 1,
 0))))))
base_dados['MOB_MORA__pe_5_g_1_2'] = np.where(base_dados['MOB_MORA__pe_5_g_1_1'] == 0, 0,
np.where(base_dados['MOB_MORA__pe_5_g_1_1'] == 1, 1,
 0))
base_dados['MOB_MORA__pe_5_g_1'] = np.where(base_dados['MOB_MORA__pe_5_g_1_2'] == 0, 0,
np.where(base_dados['MOB_MORA__pe_5_g_1_2'] == 1, 1,
 0))
                                            
                                            
                                            
                                            
                                            
                                            
base_dados['MOB_MORA__L'] = np.log(base_dados['MOB_MORA'])
np.where(base_dados['MOB_MORA__L'] == 0, -1, base_dados['MOB_MORA__L'])
base_dados['MOB_MORA__L'] = base_dados['MOB_MORA__L'].fillna(-2)
base_dados['MOB_MORA__L__p_4'] = np.where(base_dados['MOB_MORA__L'] <= 4.399239214194365, 0.0,
np.where(np.bitwise_and(base_dados['MOB_MORA__L'] > 4.399239214194365, base_dados['MOB_MORA__L'] <= 4.772263003084575), 1.0,
np.where(np.bitwise_and(base_dados['MOB_MORA__L'] > 4.772263003084575, base_dados['MOB_MORA__L'] <= 5.063467657153408), 2.0,
np.where(base_dados['MOB_MORA__L'] > 5.063467657153408, 3.0,
 0))))
base_dados['MOB_MORA__L__p_4_g_1_1'] = np.where(base_dados['MOB_MORA__L__p_4'] == 0.0, 1,
np.where(base_dados['MOB_MORA__L__p_4'] == 1.0, 0,
np.where(base_dados['MOB_MORA__L__p_4'] == 2.0, 0,
np.where(base_dados['MOB_MORA__L__p_4'] == 3.0, 0,
 0))))
base_dados['MOB_MORA__L__p_4_g_1_2'] = np.where(base_dados['MOB_MORA__L__p_4_g_1_1'] == 0, 0,
np.where(base_dados['MOB_MORA__L__p_4_g_1_1'] == 1, 1,
 0))
base_dados['MOB_MORA__L__p_4_g_1'] = np.where(base_dados['MOB_MORA__L__p_4_g_1_2'] == 0, 0,
np.where(base_dados['MOB_MORA__L__p_4_g_1_2'] == 1, 1,
 0))


# COMMAND ----------

# MAGIC %md
# MAGIC ### Parte 2 de 2

# COMMAND ----------

base_dados['VlDividaAtualizado__L__p_2_g_1_c1_24_1'] = np.where(np.bitwise_and(base_dados['VlDividaAtualizado__R__p_7_g_1'] == 0, base_dados['VlDividaAtualizado__L__p_2_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['VlDividaAtualizado__R__p_7_g_1'] == 0, base_dados['VlDividaAtualizado__L__p_2_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['VlDividaAtualizado__R__p_7_g_1'] == 1, base_dados['VlDividaAtualizado__L__p_2_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['VlDividaAtualizado__R__p_7_g_1'] == 1, base_dados['VlDividaAtualizado__L__p_2_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['VlDividaAtualizado__R__p_7_g_1'] == 2, base_dados['VlDividaAtualizado__L__p_2_g_1'] == 0), 2,
np.where(np.bitwise_and(base_dados['VlDividaAtualizado__R__p_7_g_1'] == 2, base_dados['VlDividaAtualizado__L__p_2_g_1'] == 1), 3,
np.where(np.bitwise_and(base_dados['VlDividaAtualizado__R__p_7_g_1'] == 3, base_dados['VlDividaAtualizado__L__p_2_g_1'] == 0), 2,
np.where(np.bitwise_and(base_dados['VlDividaAtualizado__R__p_7_g_1'] == 3, base_dados['VlDividaAtualizado__L__p_2_g_1'] == 1), 3,
 0))))))))
base_dados['VlDividaAtualizado__L__p_2_g_1_c1_24_2'] = np.where(base_dados['VlDividaAtualizado__L__p_2_g_1_c1_24_1'] == 0, 0,
np.where(base_dados['VlDividaAtualizado__L__p_2_g_1_c1_24_1'] == 1, 1,
np.where(base_dados['VlDividaAtualizado__L__p_2_g_1_c1_24_1'] == 2, 2,
np.where(base_dados['VlDividaAtualizado__L__p_2_g_1_c1_24_1'] == 3, 3,
0))))
base_dados['VlDividaAtualizado__L__p_2_g_1_c1_24'] = np.where(base_dados['VlDividaAtualizado__L__p_2_g_1_c1_24_2'] == 0, 0,
np.where(base_dados['VlDividaAtualizado__L__p_2_g_1_c1_24_2'] == 1, 1,
np.where(base_dados['VlDividaAtualizado__L__p_2_g_1_c1_24_2'] == 2, 2,
np.where(base_dados['VlDividaAtualizado__L__p_2_g_1_c1_24_2'] == 3, 3,
 0))))
         
         
         
         
         
         
         
base_dados['IdContatoSIR__L__pk_10_g_1_c1_5_1'] = np.where(np.bitwise_and(base_dados['IdContatoSIR__L__pu_10_g_1'] == 0, base_dados['IdContatoSIR__L__pk_10_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['IdContatoSIR__L__pu_10_g_1'] == 0, base_dados['IdContatoSIR__L__pk_10_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['IdContatoSIR__L__pu_10_g_1'] == 0, base_dados['IdContatoSIR__L__pk_10_g_1'] == 2), 1,
np.where(np.bitwise_and(base_dados['IdContatoSIR__L__pu_10_g_1'] == 1, base_dados['IdContatoSIR__L__pk_10_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['IdContatoSIR__L__pu_10_g_1'] == 1, base_dados['IdContatoSIR__L__pk_10_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['IdContatoSIR__L__pu_10_g_1'] == 1, base_dados['IdContatoSIR__L__pk_10_g_1'] == 2), 2,
np.where(np.bitwise_and(base_dados['IdContatoSIR__L__pu_10_g_1'] == 2, base_dados['IdContatoSIR__L__pk_10_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['IdContatoSIR__L__pu_10_g_1'] == 2, base_dados['IdContatoSIR__L__pk_10_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['IdContatoSIR__L__pu_10_g_1'] == 2, base_dados['IdContatoSIR__L__pk_10_g_1'] == 2), 3,
 0)))))))))
base_dados['IdContatoSIR__L__pk_10_g_1_c1_5_2'] = np.where(base_dados['IdContatoSIR__L__pk_10_g_1_c1_5_1'] == 0, 0,
np.where(base_dados['IdContatoSIR__L__pk_10_g_1_c1_5_1'] == 1, 1,
np.where(base_dados['IdContatoSIR__L__pk_10_g_1_c1_5_1'] == 2, 2,
np.where(base_dados['IdContatoSIR__L__pk_10_g_1_c1_5_1'] == 3, 3,
0))))
base_dados['IdContatoSIR__L__pk_10_g_1_c1_5'] = np.where(base_dados['IdContatoSIR__L__pk_10_g_1_c1_5_2'] == 0, 0,
np.where(base_dados['IdContatoSIR__L__pk_10_g_1_c1_5_2'] == 1, 1,
np.where(base_dados['IdContatoSIR__L__pk_10_g_1_c1_5_2'] == 2, 2,
np.where(base_dados['IdContatoSIR__L__pk_10_g_1_c1_5_2'] == 3, 3,
 0))))
         
         
         
         
         
         
base_dados['document__L__pk_8_g_1_c1_62_1'] = np.where(np.bitwise_and(base_dados['document__pe_13_g_1'] == 0, base_dados['document__L__pk_8_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['document__pe_13_g_1'] == 0, base_dados['document__L__pk_8_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['document__pe_13_g_1'] == 1, base_dados['document__L__pk_8_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['document__pe_13_g_1'] == 1, base_dados['document__L__pk_8_g_1'] == 2), 2,
np.where(np.bitwise_and(base_dados['document__pe_13_g_1'] == 2, base_dados['document__L__pk_8_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['document__pe_13_g_1'] == 2, base_dados['document__L__pk_8_g_1'] == 2), 3,
 0))))))
base_dados['document__L__pk_8_g_1_c1_62_2'] = np.where(base_dados['document__L__pk_8_g_1_c1_62_1'] == 0, 0,
np.where(base_dados['document__L__pk_8_g_1_c1_62_1'] == 1, 1,
np.where(base_dados['document__L__pk_8_g_1_c1_62_1'] == 2, 2,
np.where(base_dados['document__L__pk_8_g_1_c1_62_1'] == 3, 3,
0))))
base_dados['document__L__pk_8_g_1_c1_62'] = np.where(base_dados['document__L__pk_8_g_1_c1_62_2'] == 0, 0,
np.where(base_dados['document__L__pk_8_g_1_c1_62_2'] == 1, 1,
np.where(base_dados['document__L__pk_8_g_1_c1_62_2'] == 2, 2,
np.where(base_dados['document__L__pk_8_g_1_c1_62_2'] == 3, 3,
 0))))
         
         
         
         
         
         
base_dados['MOB_MORA__L__p_4_g_1_c1_17_1'] = np.where(np.bitwise_and(base_dados['MOB_MORA__pe_5_g_1'] == 0, base_dados['MOB_MORA__L__p_4_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['MOB_MORA__pe_5_g_1'] == 0, base_dados['MOB_MORA__L__p_4_g_1'] == 1), 0,
np.where(np.bitwise_and(base_dados['MOB_MORA__pe_5_g_1'] == 1, base_dados['MOB_MORA__L__p_4_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['MOB_MORA__pe_5_g_1'] == 1, base_dados['MOB_MORA__L__p_4_g_1'] == 1), 1,
 0))))
base_dados['MOB_MORA__L__p_4_g_1_c1_17_2'] = np.where(base_dados['MOB_MORA__L__p_4_g_1_c1_17_1'] == 0, 0,
np.where(base_dados['MOB_MORA__L__p_4_g_1_c1_17_1'] == 1, 1,
0))
base_dados['MOB_MORA__L__p_4_g_1_c1_17'] = np.where(base_dados['MOB_MORA__L__p_4_g_1_c1_17_2'] == 0, 0,
np.where(base_dados['MOB_MORA__L__p_4_g_1_c1_17_2'] == 1, 1,
 0))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Gerando variáveis compostas gerais

# COMMAND ----------

# MAGIC %md
# MAGIC ## Mantendo apenas as variáveis do modelo

# COMMAND ----------

import pickle
modelo=pickle.load(open(pickle_path + 'model_fit_recovery_r_forest.sav', 'rb'))

base_teste_c0 = base_dados[[chave,'VlDividaAtualizado__L__p_2_g_1_c1_24','MOB_MORA__L__p_4_g_1_c1_17','IdContatoSIR__L__pk_10_g_1_c1_5','Class_Carteira_gh38','document__L__pk_8_g_1_c1_62']]

var_fin_c0=['VlDividaAtualizado__L__p_2_g_1_c1_24','MOB_MORA__L__p_4_g_1_c1_17','IdContatoSIR__L__pk_10_g_1_c1_5','Class_Carteira_gh38','document__L__pk_8_g_1_c1_62']

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


x_teste2['P_1' + '_C'] = np.cos(x_teste2['P_1'])
x_teste2['P_1' + '_C'] = np.where(x_teste2['P_1'] == 0, -1, x_teste2['P_1' + '_C'])
x_teste2['P_1' + '_C'] = np.where(x_teste2['P_1'] == np.nan, -2, x_teste2['P_1' + '_C'])
x_teste2['P_1' + '_C'] = x_teste2['P_1' + '_C'].fillna(-2)


x_teste2['P_1_p_25_g_1'] = np.where(x_teste2['P_1'] <= 0.104040418, 0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.104040418, x_teste2['P_1'] <= 0.298714195), 1,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.298714195, x_teste2['P_1'] <= 0.465638889), 2,3)))

x_teste2['P_1_C_p_6_g_1'] = np.where(x_teste2['P_1_C'] <= 0.876268473, 1,
    np.where(np.bitwise_and(x_teste2['P_1_C'] > 0.876268473, x_teste2['P_1_C'] <= 0.932703911), 3,
    np.where(np.bitwise_and(x_teste2['P_1_C'] > 0.932703911, x_teste2['P_1_C'] <= 0.983475973), 2,0)))

x_teste2['GH'] = np.where(np.bitwise_and(x_teste2['P_1_p_25_g_1'] == 0, x_teste2['P_1_C_p_6_g_1'] == 0), 1,
    np.where(np.bitwise_and(x_teste2['P_1_p_25_g_1'] == 0, x_teste2['P_1_C_p_6_g_1'] == 1), 0,
    np.where(np.bitwise_and(x_teste2['P_1_p_25_g_1'] == 0, x_teste2['P_1_C_p_6_g_1'] == 2), 0,
    np.where(np.bitwise_and(x_teste2['P_1_p_25_g_1'] == 0, x_teste2['P_1_C_p_6_g_1'] == 3), 0,
    np.where(np.bitwise_and(x_teste2['P_1_p_25_g_1'] == 1, x_teste2['P_1_C_p_6_g_1'] == 0), 2,
    np.where(np.bitwise_and(x_teste2['P_1_p_25_g_1'] == 1, x_teste2['P_1_C_p_6_g_1'] == 1), 2,
    np.where(np.bitwise_and(x_teste2['P_1_p_25_g_1'] == 1, x_teste2['P_1_C_p_6_g_1'] == 2), 2,
    np.where(np.bitwise_and(x_teste2['P_1_p_25_g_1'] == 1, x_teste2['P_1_C_p_6_g_1'] == 3), 2,
    np.where(np.bitwise_and(x_teste2['P_1_p_25_g_1'] == 2, x_teste2['P_1_C_p_6_g_1'] == 0), 4,
    np.where(np.bitwise_and(x_teste2['P_1_p_25_g_1'] == 2, x_teste2['P_1_C_p_6_g_1'] == 1), 4,
    np.where(np.bitwise_and(x_teste2['P_1_p_25_g_1'] == 2, x_teste2['P_1_C_p_6_g_1'] == 2), 3,
    np.where(np.bitwise_and(x_teste2['P_1_p_25_g_1'] == 2, x_teste2['P_1_C_p_6_g_1'] == 3), 3,
    np.where(np.bitwise_and(x_teste2['P_1_p_25_g_1'] == 3, x_teste2['P_1_C_p_6_g_1'] == 0), 5,
    np.where(np.bitwise_and(x_teste2['P_1_p_25_g_1'] == 3, x_teste2['P_1_C_p_6_g_1'] == 1), 5,
    np.where(np.bitwise_and(x_teste2['P_1_p_25_g_1'] == 3, x_teste2['P_1_C_p_6_g_1'] == 2), 4,
    np.where(np.bitwise_and(x_teste2['P_1_p_25_g_1'] == 3, x_teste2['P_1_C_p_6_g_1'] == 3), 4,
             2))))))))))))))))

del x_teste2['P_1_C']
del x_teste2['P_1_p_25_g_1']
del x_teste2['P_1_C_p_6_g_1']

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