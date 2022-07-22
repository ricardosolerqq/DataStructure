# Databricks notebook source
# MAGIC %run "/pipe_modelos/extrator_recupera/v2/00.le_regras_disponibiliza_variaveis_v2"

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

dbutils.widgets.text('credor', 'fort_brasil')
credor = dbutils.widgets.get('credor')
credor = Credor(credor)

dbutils.widgets.text('modelo_escolhido', 'v2_model_20220307')
modelo_escolhido = dbutils.widgets.get('modelo_escolhido')


# COMMAND ----------

# MAGIC %md
# MAGIC ## Inserindo hiperparâmetros do Algoritmo

# COMMAND ----------

## Parâmetros do Algoritmo

#Variável chave-primaria
chave = 'DOCUMENTO'

#Caminho da base de dados
caminho_base = credor.caminho_joined_trusted_dbfs
list_base = os.listdir(caminho_base)

#Nome da Base de Dados
N_Base = max(list_base)

#Separador
separador_ = ";"

#Decimal
decimal_ = "."

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importação da Base de Dados

# COMMAND ----------

#carregar o arquivo em formato tabela
############ BLOCO TRY QUE GERA ARQUIVO DE TRANSFORMAÇÃO PREVENTIVA #############
try:
  base_dados = pd.read_csv(os.path.join(caminho_base,N_Base), sep=separador_, decimal=decimal_)
except:
  preemptive_transform(credor, modelo_escolhido)
  base_dados = pd.read_csv(os.path.join(caminho_base,N_Base), sep=separador_, decimal=decimal_)
#################################################################################

#carregar o arquivo em formato tabela
base_dados = base_dados[[chave,'dat_expir_prazo','dat_inici_contr','dat_refer','dat_cadas_clien','ind_estad_civil','ind_sexo','des_estad_resid','val_compr','dat_nasci','des_cep_resid','des_fones_resid','des_fones_celul']]

base_dados.fillna(-3)


base_dados['ind_estad_resid'] = np.where(base_dados['des_estad_resid'] == 'PE', 3,
    np.where(base_dados['des_estad_resid'] == 'CE', 3,
    np.where(base_dados['des_estad_resid'] == 'BA', 3,
    np.where(base_dados['des_estad_resid'] == 'GO', 3,
    np.where(base_dados['des_estad_resid'] == 'DF', 3,
    np.where(base_dados['des_estad_resid'] == 'RN', 4,
    np.where(base_dados['des_estad_resid'] == 'PI', 4,
    np.where(base_dados['des_estad_resid'] == '', 1,
    2))))))))

base_dados['ind_sexo_estad_civil'] = np.where(np.bitwise_and(base_dados['ind_estad_civil'] == 'S', base_dados['ind_sexo'] <= 'F'), 3,
    np.where(np.bitwise_and(base_dados['ind_estad_civil'] == 'C', base_dados['ind_sexo'] <= 'M'), 3,
    np.where(np.bitwise_and(base_dados['ind_estad_civil'] == 'O', base_dados['ind_sexo'] <= 'F'), 3,
    np.where(np.bitwise_and(base_dados['ind_estad_civil'] == 'V', base_dados['ind_sexo'] <= 'M'), 3,
    np.where(np.bitwise_and(base_dados['ind_estad_civil'] == 'S', base_dados['ind_sexo'] <= 'M'), 2,
    np.where(np.bitwise_and(base_dados['ind_estad_civil'] == 'C', base_dados['ind_sexo'] <= 'F'), 2,         
    np.where(np.bitwise_and(base_dados['ind_estad_civil'] == 'O', base_dados['ind_sexo'] <= 'M'), 2,1)))))))

del base_dados['ind_sexo']
del base_dados['ind_estad_civil']
del base_dados['des_estad_resid']


base_dados['val_compr'] = base_dados['val_compr'].replace(np.nan, '-3')
base_dados['des_cep_resid'] = base_dados['des_cep_resid'].replace(np.nan, '-3')
base_dados['des_fones_resid'] = base_dados['des_fones_resid'].replace(np.nan, '-3')
base_dados['des_fones_celul'] = base_dados['des_fones_celul'].replace(np.nan, '-3')


base_dados = base_dados.apply(pd.to_numeric, errors='ignore')

base_dados['des_cep_resid'] = base_dados['des_cep_resid'] / 1000000


base_dados['val_compr'] = base_dados['val_compr'].astype(float)
base_dados['des_cep_resid'] = base_dados['des_cep_resid'].astype(int)
base_dados['des_fones_resid'] = base_dados['des_fones_resid'].astype(np.int64)
base_dados['des_fones_celul'] = base_dados['des_fones_celul'].astype(np.int64)


base_dados['dat_inici_contr'] = pd.to_datetime(base_dados['dat_inici_contr'])
base_dados['dat_nasci'] = pd.to_datetime(base_dados['dat_nasci'])
base_dados['dat_cadas_clien'] = pd.to_datetime(base_dados['dat_cadas_clien'])
base_dados['dat_refer'] = pd.to_datetime(base_dados['dat_refer'])
base_dados['dat_expir_prazo'] = pd.to_datetime(base_dados['dat_expir_prazo'])

base_dados['mob_contrato'] = ((datetime.today()) - base_dados.dat_inici_contr)/np.timedelta64(1, 'M')
base_dados['idade'] = ((datetime.today()) - base_dados.dat_nasci)/np.timedelta64(1, 'Y')
base_dados['mob_cliente'] = ((datetime.today()) - base_dados.dat_cadas_clien)/np.timedelta64(1, 'M')
base_dados['mob_refer'] = ((datetime.today()) - base_dados.dat_refer)/np.timedelta64(1, 'M')
base_dados['mob_expir_prazo'] = (base_dados.dat_expir_prazo - (datetime.today()))/np.timedelta64(1, 'M')

base_dados['mob_contrato_cliente'] = base_dados['mob_contrato']/base_dados['mob_cliente']

del base_dados['dat_inici_contr']
del base_dados['dat_nasci']
del base_dados['dat_cadas_clien']
del base_dados['dat_refer']
del base_dados['dat_expir_prazo']

print("shape da Base de Dados:",base_dados.shape)

base_dados.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gerando variáveis Categóricas

# COMMAND ----------

base_dados['ind_estad_resid_gh30'] = np.where(base_dados['ind_estad_resid'] == 2, 0,
np.where(base_dados['ind_estad_resid'] == 3, 1,
np.where(base_dados['ind_estad_resid'] == 4, 2,
0)))

base_dados['ind_estad_resid_gh31'] = np.where(base_dados['ind_estad_resid_gh30'] == 0, 0,
np.where(base_dados['ind_estad_resid_gh30'] == 1, 1,
np.where(base_dados['ind_estad_resid_gh30'] == 2, 2,
0)))

base_dados['ind_estad_resid_gh32'] = np.where(base_dados['ind_estad_resid_gh31'] == 0, 0,
np.where(base_dados['ind_estad_resid_gh31'] == 1, 1,
np.where(base_dados['ind_estad_resid_gh31'] == 2, 2,
0)))

base_dados['ind_estad_resid_gh33'] = np.where(base_dados['ind_estad_resid_gh32'] == 0, 0,
np.where(base_dados['ind_estad_resid_gh32'] == 1, 1,
np.where(base_dados['ind_estad_resid_gh32'] == 2, 2,
0)))

base_dados['ind_estad_resid_gh34'] = np.where(base_dados['ind_estad_resid_gh33'] == 0, 0,
np.where(base_dados['ind_estad_resid_gh33'] == 1, 1,
np.where(base_dados['ind_estad_resid_gh33'] == 2, 2,
0)))

base_dados['ind_estad_resid_gh35'] = np.where(base_dados['ind_estad_resid_gh34'] == 0, 0,
np.where(base_dados['ind_estad_resid_gh34'] == 1, 1,
np.where(base_dados['ind_estad_resid_gh34'] == 2, 2,
0)))

base_dados['ind_estad_resid_gh36'] = np.where(base_dados['ind_estad_resid_gh35'] == 0, 0,
np.where(base_dados['ind_estad_resid_gh35'] == 1, 1,
np.where(base_dados['ind_estad_resid_gh35'] == 2, 2,
0)))

base_dados['ind_estad_resid_gh37'] = np.where(base_dados['ind_estad_resid_gh36'] == 0, 0,
np.where(base_dados['ind_estad_resid_gh36'] == 1, 1,
np.where(base_dados['ind_estad_resid_gh36'] == 2, 2,
0)))

base_dados['ind_estad_resid_gh38'] = np.where(base_dados['ind_estad_resid_gh37'] == 0, 0,
np.where(base_dados['ind_estad_resid_gh37'] == 1, 1,
np.where(base_dados['ind_estad_resid_gh37'] == 2, 2,
0)))
         


         
                
base_dados['ind_sexo_estad_civil_gh40'] = np.where(base_dados['ind_sexo_estad_civil'] == 1, 0,
np.where(base_dados['ind_sexo_estad_civil'] == 2, 1,
np.where(base_dados['ind_sexo_estad_civil'] == 3, 2,
0)))
         
base_dados['ind_sexo_estad_civil_gh41'] = np.where(base_dados['ind_sexo_estad_civil_gh40'] == 0, -5,
np.where(base_dados['ind_sexo_estad_civil_gh40'] == 1, -5,
np.where(base_dados['ind_sexo_estad_civil_gh40'] == 2, 0,
0)))

base_dados['ind_sexo_estad_civil_gh42'] = np.where(base_dados['ind_sexo_estad_civil_gh41'] == -5, 0,
np.where(base_dados['ind_sexo_estad_civil_gh41'] == 0, 1,
0))

base_dados['ind_sexo_estad_civil_gh43'] = np.where(base_dados['ind_sexo_estad_civil_gh42'] == 0, 0,
np.where(base_dados['ind_sexo_estad_civil_gh42'] == 1, 1,
0))

base_dados['ind_sexo_estad_civil_gh44'] = np.where(base_dados['ind_sexo_estad_civil_gh43'] == 0, 0,
np.where(base_dados['ind_sexo_estad_civil_gh43'] == 1, 1,
0))

base_dados['ind_sexo_estad_civil_gh45'] = np.where(base_dados['ind_sexo_estad_civil_gh44'] == 0, 0,
np.where(base_dados['ind_sexo_estad_civil_gh44'] == 1, 1,
0))

base_dados['ind_sexo_estad_civil_gh46'] = np.where(base_dados['ind_sexo_estad_civil_gh45'] == 0, 0,
np.where(base_dados['ind_sexo_estad_civil_gh45'] == 1, 1,
0))

base_dados['ind_sexo_estad_civil_gh47'] = np.where(base_dados['ind_sexo_estad_civil_gh46'] == 0, 0,
np.where(base_dados['ind_sexo_estad_civil_gh46'] == 1, 1,
0))

base_dados['ind_sexo_estad_civil_gh48'] = np.where(base_dados['ind_sexo_estad_civil_gh47'] == 0, 0,
np.where(base_dados['ind_sexo_estad_civil_gh47'] == 1, 1,
0))

base_dados['ind_sexo_estad_civil_gh49'] = np.where(base_dados['ind_sexo_estad_civil_gh48'] == 0, 0,
np.where(base_dados['ind_sexo_estad_civil_gh48'] == 1, 1,
0))

base_dados['ind_sexo_estad_civil_gh50'] = np.where(base_dados['ind_sexo_estad_civil_gh49'] == 0, 0,
np.where(base_dados['ind_sexo_estad_civil_gh49'] == 1, 1,
0))

base_dados['ind_sexo_estad_civil_gh51'] = np.where(base_dados['ind_sexo_estad_civil_gh50'] == 0, 0,
np.where(base_dados['ind_sexo_estad_civil_gh50'] == 1, 1,
0))
         
         
         
       


# COMMAND ----------

# MAGIC %md
# MAGIC ## Gerando variáveis numéricas contínuas

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parte 1 de 2

# COMMAND ----------

base_dados['val_compr__R'] = np.sqrt(base_dados['val_compr'])
np.where(base_dados['val_compr__R'] == 0, -1, base_dados['val_compr__R'])
base_dados['val_compr__R'] = base_dados['val_compr__R'].fillna(-2)
base_dados['val_compr__R__pe_13'] = np.where(np.bitwise_and(base_dados['val_compr__R'] >= -2.0, base_dados['val_compr__R'] <= 8.61278120005379), 1.0,
np.where(np.bitwise_and(base_dados['val_compr__R'] > 8.61278120005379, base_dados['val_compr__R'] <= 12.936382802004585), 2.0,
np.where(np.bitwise_and(base_dados['val_compr__R'] > 12.936382802004585, base_dados['val_compr__R'] <= 17.227594144279113), 3.0,
np.where(np.bitwise_and(base_dados['val_compr__R'] > 17.227594144279113, base_dados['val_compr__R'] <= 21.566640906733713), 4.0,
np.where(np.bitwise_and(base_dados['val_compr__R'] > 21.566640906733713, base_dados['val_compr__R'] <= 25.878755766071908), 5.0,
np.where(np.bitwise_and(base_dados['val_compr__R'] > 25.878755766071908, base_dados['val_compr__R'] <= 30.19453592953533), 6.0,
np.where(np.bitwise_and(base_dados['val_compr__R'] > 30.19453592953533, base_dados['val_compr__R'] <= 34.48260430999956), 7.0,
np.where(np.bitwise_and(base_dados['val_compr__R'] > 34.48260430999956, base_dados['val_compr__R'] <= 38.808633060183915), 8.0,
np.where(np.bitwise_and(base_dados['val_compr__R'] > 38.808633060183915, base_dados['val_compr__R'] <= 43.11554244121254), 9.0,
np.where(np.bitwise_and(base_dados['val_compr__R'] > 43.11554244121254, base_dados['val_compr__R'] <= 47.43195125651063), 10.0,
np.where(np.bitwise_and(base_dados['val_compr__R'] > 47.43195125651063, base_dados['val_compr__R'] <= 51.655783800074126), 11.0,
np.where(base_dados['val_compr__R'] > 51.655783800074126, 12.0,
 -2))))))))))))
base_dados['val_compr__R__pe_13_g_1_1'] = np.where(base_dados['val_compr__R__pe_13'] == -2.0, 1,
np.where(base_dados['val_compr__R__pe_13'] == 1.0, 1,
np.where(base_dados['val_compr__R__pe_13'] == 2.0, 1,
np.where(base_dados['val_compr__R__pe_13'] == 3.0, 0,
np.where(base_dados['val_compr__R__pe_13'] == 4.0, 0,
np.where(base_dados['val_compr__R__pe_13'] == 5.0, 0,
np.where(base_dados['val_compr__R__pe_13'] == 6.0, 0,
np.where(base_dados['val_compr__R__pe_13'] == 7.0, 0,
np.where(base_dados['val_compr__R__pe_13'] == 8.0, 0,
np.where(base_dados['val_compr__R__pe_13'] == 9.0, 0,
np.where(base_dados['val_compr__R__pe_13'] == 10.0, 0,
np.where(base_dados['val_compr__R__pe_13'] == 11.0, 1,
np.where(base_dados['val_compr__R__pe_13'] == 12.0, 0,
 0)))))))))))))
base_dados['val_compr__R__pe_13_g_1_2'] = np.where(base_dados['val_compr__R__pe_13_g_1_1'] == 0, 1,
np.where(base_dados['val_compr__R__pe_13_g_1_1'] == 1, 0,
 0))
base_dados['val_compr__R__pe_13_g_1'] = np.where(base_dados['val_compr__R__pe_13_g_1_2'] == 0, 0,
np.where(base_dados['val_compr__R__pe_13_g_1_2'] == 1, 1,
 0))
                                                 
                                                 
                                                 
                                                 
base_dados['val_compr__L'] = np.log(base_dados['val_compr'])
np.where(base_dados['val_compr__L'] == 0, -1, base_dados['val_compr__L'])
base_dados['val_compr__L'] = base_dados['val_compr__L'].fillna(-2)
base_dados['val_compr__L__pu_20'] = np.where(base_dados['val_compr__L'] <= -2.0, 0.0,
np.where(np.bitwise_and(base_dados['val_compr__L'] > -2.0, base_dados['val_compr__L'] <= 4.345881188037528), 10.0,
np.where(np.bitwise_and(base_dados['val_compr__L'] > 4.345881188037528, base_dados['val_compr__L'] <= 4.922896379788249), 11.0,
np.where(np.bitwise_and(base_dados['val_compr__L'] > 4.922896379788249, base_dados['val_compr__L'] <= 5.501217393385187), 12.0,
np.where(np.bitwise_and(base_dados['val_compr__L'] > 5.501217393385187, base_dados['val_compr__L'] <= 6.078857101440297), 13.0,
np.where(np.bitwise_and(base_dados['val_compr__L'] > 6.078857101440297, base_dados['val_compr__L'] <= 6.6560064706575925), 14.0,
np.where(np.bitwise_and(base_dados['val_compr__L'] > 6.6560064706575925, base_dados['val_compr__L'] <= 7.2325452572578), 15.0,
np.where(np.bitwise_and(base_dados['val_compr__L'] > 7.2325452572578, base_dados['val_compr__L'] <= 7.80840025179941), 16.0,
np.where(np.bitwise_and(base_dados['val_compr__L'] > 7.80840025179941, base_dados['val_compr__L'] <= 8.383151954557514), 17.0,
np.where(np.bitwise_and(base_dados['val_compr__L'] > 8.383151954557514, base_dados['val_compr__L'] <= 8.90918527794222), 18.0,
np.where(base_dados['val_compr__L'] > 8.90918527794222, 19.0,
 0)))))))))))
base_dados['val_compr__L__pu_20_g_1_1'] = np.where(base_dados['val_compr__L__pu_20'] == 0.0, 1,
np.where(base_dados['val_compr__L__pu_20'] == 10.0, 1,
np.where(base_dados['val_compr__L__pu_20'] == 11.0, 1,
np.where(base_dados['val_compr__L__pu_20'] == 12.0, 1,
np.where(base_dados['val_compr__L__pu_20'] == 13.0, 0,
np.where(base_dados['val_compr__L__pu_20'] == 14.0, 0,
np.where(base_dados['val_compr__L__pu_20'] == 15.0, 0,
np.where(base_dados['val_compr__L__pu_20'] == 16.0, 0,
np.where(base_dados['val_compr__L__pu_20'] == 17.0, 0,
np.where(base_dados['val_compr__L__pu_20'] == 18.0, 1,
np.where(base_dados['val_compr__L__pu_20'] == 19.0, 1,
 0)))))))))))
base_dados['val_compr__L__pu_20_g_1_2'] = np.where(base_dados['val_compr__L__pu_20_g_1_1'] == 0, 1,
np.where(base_dados['val_compr__L__pu_20_g_1_1'] == 1, 0,
 0))
base_dados['val_compr__L__pu_20_g_1'] = np.where(base_dados['val_compr__L__pu_20_g_1_2'] == 0, 0,
np.where(base_dados['val_compr__L__pu_20_g_1_2'] == 1, 1,
 0))
                                                 
                                                 
                                                 
                                                 
                                                 
                                                 
                                                 
base_dados['des_cep_resid__pk_2'] = np.where(base_dados['des_cep_resid'] == 0 , -1.0,
np.where(np.bitwise_and(base_dados['des_cep_resid'] > 0.0, base_dados['des_cep_resid'] <= 29.0), 0.0,
np.where(base_dados['des_cep_resid'] > 29.0, 1.0,
 0)))
base_dados['des_cep_resid__pk_2_g_1_1'] = np.where(base_dados['des_cep_resid__pk_2'] == -1.0, 1,
np.where(base_dados['des_cep_resid__pk_2'] == 0.0, 1,
np.where(base_dados['des_cep_resid__pk_2'] == 1.0, 0,
 0)))
base_dados['des_cep_resid__pk_2_g_1_2'] = np.where(base_dados['des_cep_resid__pk_2_g_1_1'] == 0, 1,
np.where(base_dados['des_cep_resid__pk_2_g_1_1'] == 1, 0,
 0))
base_dados['des_cep_resid__pk_2_g_1'] = np.where(base_dados['des_cep_resid__pk_2_g_1_2'] == 0, 0,
np.where(base_dados['des_cep_resid__pk_2_g_1_2'] == 1, 1,
 0))
                                                 
                                                 
                                                 
                                                 
                                                 
                                                 
                                                 
                                                 
base_dados['des_cep_resid__C'] = np.cos(base_dados['des_cep_resid'])
np.where(base_dados['des_cep_resid__C'] == 0, -1, base_dados['des_cep_resid__C'])
base_dados['des_cep_resid__C'] = base_dados['des_cep_resid__C'].fillna(-2)
base_dados['des_cep_resid__C__pu_10'] = np.where(base_dados['des_cep_resid__C'] <= -0.8293098328631502, 0.0,
np.where(np.bitwise_and(base_dados['des_cep_resid__C'] > -0.8293098328631502, base_dados['des_cep_resid__C'] <= -0.6401443394691997), 1.0,
np.where(np.bitwise_and(base_dados['des_cep_resid__C'] > -0.6401443394691997, base_dados['des_cep_resid__C'] <= -0.5177697997895051), 2.0,
np.where(np.bitwise_and(base_dados['des_cep_resid__C'] > -0.5177697997895051, base_dados['des_cep_resid__C'] <= -0.25810163593826746), 3.0,
np.where(np.bitwise_and(base_dados['des_cep_resid__C'] > -0.25810163593826746, base_dados['des_cep_resid__C'] <= -0.013276747223059479), 4.0,
np.where(np.bitwise_and(base_dados['des_cep_resid__C'] > -0.013276747223059479, base_dados['des_cep_resid__C'] <= 0.17171734183077755), 5.0,
np.where(np.bitwise_and(base_dados['des_cep_resid__C'] > 0.17171734183077755, base_dados['des_cep_resid__C'] <= 0.39185723042955), 6.0,
np.where(np.bitwise_and(base_dados['des_cep_resid__C'] > 0.39185723042955, base_dados['des_cep_resid__C'] <= 0.5551133015206257), 7.0,
np.where(np.bitwise_and(base_dados['des_cep_resid__C'] > 0.5551133015206257, base_dados['des_cep_resid__C'] <= 0.7654140519453434), 8.0,
np.where(base_dados['des_cep_resid__C'] > 0.7654140519453434, 9.0,
 0))))))))))
base_dados['des_cep_resid__C__pu_10_g_1_1'] = np.where(base_dados['des_cep_resid__C__pu_10'] == 0.0, 1,
np.where(base_dados['des_cep_resid__C__pu_10'] == 1.0, 0,
np.where(base_dados['des_cep_resid__C__pu_10'] == 2.0, 2,
np.where(base_dados['des_cep_resid__C__pu_10'] == 3.0, 2,
np.where(base_dados['des_cep_resid__C__pu_10'] == 4.0, 2,
np.where(base_dados['des_cep_resid__C__pu_10'] == 5.0, 1,
np.where(base_dados['des_cep_resid__C__pu_10'] == 6.0, 2,
np.where(base_dados['des_cep_resid__C__pu_10'] == 7.0, 2,
np.where(base_dados['des_cep_resid__C__pu_10'] == 8.0, 2,
np.where(base_dados['des_cep_resid__C__pu_10'] == 9.0, 2,
 0))))))))))
base_dados['des_cep_resid__C__pu_10_g_1_2'] = np.where(base_dados['des_cep_resid__C__pu_10_g_1_1'] == 0, 1,
np.where(base_dados['des_cep_resid__C__pu_10_g_1_1'] == 1, 0,
np.where(base_dados['des_cep_resid__C__pu_10_g_1_1'] == 2, 1,
 0)))
base_dados['des_cep_resid__C__pu_10_g_1'] = np.where(base_dados['des_cep_resid__C__pu_10_g_1_2'] == 0, 0,
np.where(base_dados['des_cep_resid__C__pu_10_g_1_2'] == 1, 1,
 0))
                                                     
                                                     
                                                     
                                                     
                                                     
                                                     
                                                     
                                                     
                                                     
base_dados['des_fones_resid__p_10'] = np.where(base_dados['des_fones_resid'] <= 8136032070.0, 0.0,
np.where(base_dados['des_fones_resid'] > 8136032070.0, 1.0,
 0))
base_dados['des_fones_resid__p_10_g_1_1'] = np.where(base_dados['des_fones_resid__p_10'] == 0.0, 0,
np.where(base_dados['des_fones_resid__p_10'] == 1.0, 1,
 0))
base_dados['des_fones_resid__p_10_g_1_2'] = np.where(base_dados['des_fones_resid__p_10_g_1_1'] == 0, 0,
np.where(base_dados['des_fones_resid__p_10_g_1_1'] == 1, 1,
 0))
base_dados['des_fones_resid__p_10_g_1'] = np.where(base_dados['des_fones_resid__p_10_g_1_2'] == 0, 0,
np.where(base_dados['des_fones_resid__p_10_g_1_2'] == 1, 1,
 0))
                                                   
                                                   
                                                   
                                                   
                                                   
                                                   
                                                   
base_dados['des_fones_resid__C'] = np.cos(base_dados['des_fones_resid'])
np.where(base_dados['des_fones_resid__C'] == 0, -1, base_dados['des_fones_resid__C'])
base_dados['des_fones_resid__C'] = base_dados['des_fones_resid__C'].fillna(-2)
base_dados['des_fones_resid__C__p_10'] = np.where(base_dados['des_fones_resid__C'] <= -0.9904569013230891, 0.0,
np.where(np.bitwise_and(base_dados['des_fones_resid__C'] > -0.9904569013230891, base_dados['des_fones_resid__C'] <= -0.4551511974879847), 1.0,
np.where(base_dados['des_fones_resid__C'] > -0.4551511974879847, 2.0,
 0)))
base_dados['des_fones_resid__C__p_10_g_1_1'] = np.where(base_dados['des_fones_resid__C__p_10'] == 0.0, 0,
np.where(base_dados['des_fones_resid__C__p_10'] == 1.0, 0,
np.where(base_dados['des_fones_resid__C__p_10'] == 2.0, 1,
 0)))
base_dados['des_fones_resid__C__p_10_g_1_2'] = np.where(base_dados['des_fones_resid__C__p_10_g_1_1'] == 0, 0,
np.where(base_dados['des_fones_resid__C__p_10_g_1_1'] == 1, 1,
 0))
base_dados['des_fones_resid__C__p_10_g_1'] = np.where(base_dados['des_fones_resid__C__p_10_g_1_2'] == 0, 0,
np.where(base_dados['des_fones_resid__C__p_10_g_1_2'] == 1, 1,
 0))
                                                      
                                                      
                                                      
                                                      
                                                      
                                                      
                                                      
base_dados['des_fones_celul__C'] = np.cos(base_dados['des_fones_celul'])
np.where(base_dados['des_fones_celul__C'] == 0, -1, base_dados['des_fones_celul__C'])
base_dados['des_fones_celul__C'] = base_dados['des_fones_celul__C'].fillna(-2)
base_dados['des_fones_celul__C__p_6'] = np.where(base_dados['des_fones_celul__C'] <= -0.9900331291588672, 0.0,
np.where(np.bitwise_and(base_dados['des_fones_celul__C'] > -0.9900331291588672, base_dados['des_fones_celul__C'] <= -0.7777945610901995), 1.0,
np.where(np.bitwise_and(base_dados['des_fones_celul__C'] > -0.7777945610901995, base_dados['des_fones_celul__C'] <= -0.030037805718202873), 2.0,
np.where(np.bitwise_and(base_dados['des_fones_celul__C'] > -0.030037805718202873, base_dados['des_fones_celul__C'] <= 0.6763721681299887), 3.0,
np.where(base_dados['des_fones_celul__C'] > 0.6763721681299887, 4.0,
 0)))))
base_dados['des_fones_celul__C__p_6_g_1_1'] = np.where(base_dados['des_fones_celul__C__p_6'] == 0.0, 0,
np.where(base_dados['des_fones_celul__C__p_6'] == 1.0, 1,
np.where(base_dados['des_fones_celul__C__p_6'] == 2.0, 2,
np.where(base_dados['des_fones_celul__C__p_6'] == 3.0, 0,
np.where(base_dados['des_fones_celul__C__p_6'] == 4.0, 0,
 0)))))
base_dados['des_fones_celul__C__p_6_g_1_2'] = np.where(base_dados['des_fones_celul__C__p_6_g_1_1'] == 0, 1,
np.where(base_dados['des_fones_celul__C__p_6_g_1_1'] == 1, 0,
np.where(base_dados['des_fones_celul__C__p_6_g_1_1'] == 2, 1,
 0)))
base_dados['des_fones_celul__C__p_6_g_1'] = np.where(base_dados['des_fones_celul__C__p_6_g_1_2'] == 0, 0,
np.where(base_dados['des_fones_celul__C__p_6_g_1_2'] == 1, 1,
 0))
                                                     
                                                     
                                                     
                                                     
                                                     
                                                     
                                                     
base_dados['des_fones_celul__C'] = np.cos(base_dados['des_fones_celul'])
np.where(base_dados['des_fones_celul__C'] == 0, -1, base_dados['des_fones_celul__C'])
base_dados['des_fones_celul__C'] = base_dados['des_fones_celul__C'].fillna(-2)
base_dados['des_fones_celul__C__pu_5'] = np.where(base_dados['des_fones_celul__C'] <= -0.6002203325874924, 0.0,
np.where(np.bitwise_and(base_dados['des_fones_celul__C'] > -0.6002203325874924, base_dados['des_fones_celul__C'] <= -0.20045281124787379), 1.0,
np.where(np.bitwise_and(base_dados['des_fones_celul__C'] > -0.20045281124787379, base_dados['des_fones_celul__C'] <= 0.1994711415012865), 2.0,
np.where(np.bitwise_and(base_dados['des_fones_celul__C'] > 0.1994711415012865, base_dados['des_fones_celul__C'] <= 0.5999279933520578), 3.0,
np.where(base_dados['des_fones_celul__C'] > 0.5999279933520578, 4.0,
 0)))))
base_dados['des_fones_celul__C__pu_5_g_1_1'] = np.where(base_dados['des_fones_celul__C__pu_5'] == 0.0, 0,
np.where(base_dados['des_fones_celul__C__pu_5'] == 1.0, 1,
np.where(base_dados['des_fones_celul__C__pu_5'] == 2.0, 0,
np.where(base_dados['des_fones_celul__C__pu_5'] == 3.0, 1,
np.where(base_dados['des_fones_celul__C__pu_5'] == 4.0, 0,
 0)))))
base_dados['des_fones_celul__C__pu_5_g_1_2'] = np.where(base_dados['des_fones_celul__C__pu_5_g_1_1'] == 0, 0,
np.where(base_dados['des_fones_celul__C__pu_5_g_1_1'] == 1, 1,
 0))
base_dados['des_fones_celul__C__pu_5_g_1'] = np.where(base_dados['des_fones_celul__C__pu_5_g_1_2'] == 0, 0,
np.where(base_dados['des_fones_celul__C__pu_5_g_1_2'] == 1, 1,
 0))
                                                      
                                                      
                                                      
                                                      
                                                      
                                                      
base_dados['mob_contrato__L10'] = np.log10(base_dados['mob_contrato'])
base_dados['mob_contrato__L10'] = np.where(base_dados['mob_contrato'] == 0, -1, base_dados['mob_contrato__L10'])
base_dados['mob_contrato__L10'] = np.where(base_dados['mob_contrato'] == np.nan, -2, base_dados['mob_contrato__L10'])
base_dados['mob_contrato__L10'] = base_dados['mob_contrato__L10'].fillna(-2)

base_dados['mob_contrato__L10__pe_17'] = np.where(np.bitwise_and(base_dados['mob_contrato'] >= -2.0, base_dados['mob_contrato__L10'] <= 0.44138123971927684), 4.0,
np.where(np.bitwise_and(base_dados['mob_contrato__L10'] > 0.44138123971927684, base_dados['mob_contrato__L10'] <= 0.4515887367363552), 5.0,
np.where(np.bitwise_and(base_dados['mob_contrato__L10'] > 0.4515887367363552, base_dados['mob_contrato__L10'] <= 0.5814215141109987), 6.0,
np.where(np.bitwise_and(base_dados['mob_contrato__L10'] > 0.5814215141109987, base_dados['mob_contrato__L10'] <= 0.6842042104408885), 7.0,
np.where(np.bitwise_and(base_dados['mob_contrato__L10'] > 0.6842042104408885, base_dados['mob_contrato__L10'] <= 0.7648115221058317), 8.0,
np.where(np.bitwise_and(base_dados['mob_contrato__L10'] > 0.7648115221058317, base_dados['mob_contrato__L10'] <= 0.8915262808349335), 9.0,
np.where(np.bitwise_and(base_dados['mob_contrato__L10'] > 0.8915262808349335, base_dados['mob_contrato__L10'] <= 0.9761389078553248), 10.0,
np.where(np.bitwise_and(base_dados['mob_contrato__L10'] > 0.9761389078553248, base_dados['mob_contrato__L10'] <= 1.0706008091258556), 11.0,
np.where(np.bitwise_and(base_dados['mob_contrato__L10'] > 1.0706008091258556, base_dados['mob_contrato__L10'] <= 1.1601484997054374), 12.0,
np.where(np.bitwise_and(base_dados['mob_contrato__L10'] > 1.1601484997054374, base_dados['mob_contrato__L10'] <= 1.2490718283549807), 13.0,
np.where(np.bitwise_and(base_dados['mob_contrato__L10'] > 1.2490718283549807, base_dados['mob_contrato__L10'] <= 1.33883155856557), 14.0,
np.where(np.bitwise_and(base_dados['mob_contrato__L10'] > 1.33883155856557, base_dados['mob_contrato__L10'] <= 1.4278093162134406), 15.0,
np.where(base_dados['mob_contrato__L10'] > 1.4278093162134406, 16.0,
 -2)))))))))))))
base_dados['mob_contrato__L10__pe_17_g_1_1'] = np.where(base_dados['mob_contrato__L10__pe_17'] == -2.0, 0,
np.where(base_dados['mob_contrato__L10__pe_17'] == 4.0, 2,
np.where(base_dados['mob_contrato__L10__pe_17'] == 5.0, 2,
np.where(base_dados['mob_contrato__L10__pe_17'] == 6.0, 2,
np.where(base_dados['mob_contrato__L10__pe_17'] == 7.0, 2,
np.where(base_dados['mob_contrato__L10__pe_17'] == 8.0, 2,
np.where(base_dados['mob_contrato__L10__pe_17'] == 9.0, 2,
np.where(base_dados['mob_contrato__L10__pe_17'] == 10.0, 2,
np.where(base_dados['mob_contrato__L10__pe_17'] == 11.0, 2,
np.where(base_dados['mob_contrato__L10__pe_17'] == 12.0, 2,
np.where(base_dados['mob_contrato__L10__pe_17'] == 13.0, 0,
np.where(base_dados['mob_contrato__L10__pe_17'] == 14.0, 0,
np.where(base_dados['mob_contrato__L10__pe_17'] == 15.0, 2,
np.where(base_dados['mob_contrato__L10__pe_17'] == 16.0, 1,
 0))))))))))))))
base_dados['mob_contrato__L10__pe_17_g_1_2'] = np.where(base_dados['mob_contrato__L10__pe_17_g_1_1'] == 0, 0,
np.where(base_dados['mob_contrato__L10__pe_17_g_1_1'] == 1, 2,
np.where(base_dados['mob_contrato__L10__pe_17_g_1_1'] == 2, 0,
 0)))
base_dados['mob_contrato__L10__pe_17_g_1'] = np.where(base_dados['mob_contrato__L10__pe_17_g_1_2'] == 0, 0,
np.where(base_dados['mob_contrato__L10__pe_17_g_1_2'] == 2, 1,
 0))
                                                      
                                                      
                                                      
                                                      
                                                      
                                                      
                                                      
                                                      
base_dados['mob_contrato__T'] = np.tan(base_dados['mob_contrato'])
np.where(base_dados['mob_contrato__T'] == 0, -1, base_dados['mob_contrato__T'])
base_dados['mob_contrato__T'] = base_dados['mob_contrato__T'].fillna(-2)
base_dados['mob_contrato__T__pu_17'] = np.where(base_dados['mob_contrato__T'] <= -654.5969961410223, 0.0,
np.where(np.bitwise_and(base_dados['mob_contrato__T'] > -654.5969961410223, base_dados['mob_contrato__T'] <= -360.6516320631411), 3.0,
np.where(np.bitwise_and(base_dados['mob_contrato__T'] > -360.6516320631411, base_dados['mob_contrato__T'] <= -190.00601800209452), 5.0,
np.where(np.bitwise_and(base_dados['mob_contrato__T'] > -190.00601800209452, base_dados['mob_contrato__T'] <= -101.65668698084056), 6.0,
np.where(np.bitwise_and(base_dados['mob_contrato__T'] > -101.65668698084056, base_dados['mob_contrato__T'] <= -9.117726175171567), 7.0,
np.where(np.bitwise_and(base_dados['mob_contrato__T'] > -9.117726175171567, base_dados['mob_contrato__T'] <= 56.79322762226827), 8.0,
np.where(np.bitwise_and(base_dados['mob_contrato__T'] > 56.79322762226827, base_dados['mob_contrato__T'] <= 103.05271733595247), 9.0,
np.where(np.bitwise_and(base_dados['mob_contrato__T'] > 103.05271733595247, base_dados['mob_contrato__T'] <= 194.9416378213113), 10.0,
np.where(np.bitwise_and(base_dados['mob_contrato__T'] > 194.9416378213113, base_dados['mob_contrato__T'] <= 555.2404520277414), 14.0,
np.where(base_dados['mob_contrato__T'] > 555.2404520277414, 16.0,
0))))))))))
base_dados['mob_contrato__T__pu_17_g_1_1'] = np.where(base_dados['mob_contrato__T__pu_17'] == 0.0, 1,
np.where(base_dados['mob_contrato__T__pu_17'] == 3.0, 1,
np.where(base_dados['mob_contrato__T__pu_17'] == 5.0, 1,
np.where(base_dados['mob_contrato__T__pu_17'] == 6.0, 1,
np.where(base_dados['mob_contrato__T__pu_17'] == 7.0, 0,
np.where(base_dados['mob_contrato__T__pu_17'] == 8.0, 0,
np.where(base_dados['mob_contrato__T__pu_17'] == 9.0, 1,
np.where(base_dados['mob_contrato__T__pu_17'] == 10.0, 1,
np.where(base_dados['mob_contrato__T__pu_17'] == 14.0, 1,
np.where(base_dados['mob_contrato__T__pu_17'] == 16.0, 1,
 0))))))))))
base_dados['mob_contrato__T__pu_17_g_1_2'] = np.where(base_dados['mob_contrato__T__pu_17_g_1_1'] == 0, 0,
np.where(base_dados['mob_contrato__T__pu_17_g_1_1'] == 1, 1,
 0))
base_dados['mob_contrato__T__pu_17_g_1'] = np.where(base_dados['mob_contrato__T__pu_17_g_1_2'] == 0, 0,
np.where(base_dados['mob_contrato__T__pu_17_g_1_2'] == 1, 1,
 0))
                                                    
                                                    
                                                    
                                                    
                                                    
                                                    
                                                    
                                                    
                                                    
                                                    
base_dados['idade__pe_3'] = np.where(np.bitwise_and(base_dados['idade'] >= -3.0, base_dados['idade'] <= 26.136326288294416), 0.0,
np.where(np.bitwise_and(base_dados['idade'] > 26.136326288294416, base_dados['idade'] <= 52.27786239102069), 1.0,
np.where(base_dados['idade'] > 52.27786239102069, 2.0,
 -2)))
base_dados['idade__pe_3_g_1_1'] = np.where(base_dados['idade__pe_3'] == -2.0, 1,
np.where(base_dados['idade__pe_3'] == 0.0, 1,
np.where(base_dados['idade__pe_3'] == 1.0, 0,
np.where(base_dados['idade__pe_3'] == 2.0, 1,
 0))))
base_dados['idade__pe_3_g_1_2'] = np.where(base_dados['idade__pe_3_g_1_1'] == 0, 1,
np.where(base_dados['idade__pe_3_g_1_1'] == 1, 0,
 0))
base_dados['idade__pe_3_g_1'] = np.where(base_dados['idade__pe_3_g_1_2'] == 0, 0,
np.where(base_dados['idade__pe_3_g_1_2'] == 1, 1,
 0))
                                                 
                                                 
                                                 
                                                 
                                                 
                                                 
base_dados['idade__R'] = np.sqrt(base_dados['idade'])
np.where(base_dados['idade__R'] == 0, -1, base_dados['idade__R'])
base_dados['idade__R'] = base_dados['idade__R'].fillna(-2)
base_dados['idade__R__pk_5'] = np.where(base_dados['idade__R'] <= -2.0, 0.0,
np.where(np.bitwise_and(base_dados['idade__R'] > -2.0, base_dados['idade__R'] <= 5.680601903315235), 1.0,
np.where(np.bitwise_and(base_dados['idade__R'] > 5.680601903315235, base_dados['idade__R'] <= 6.669560790796078), 2.0,
np.where(np.bitwise_and(base_dados['idade__R'] > 6.669560790796078, base_dados['idade__R'] <= 7.643052394213251), 3.0,
np.where(base_dados['idade__R'] > 7.643052394213251, 4.0,
 0)))))
base_dados['idade__R__pk_5_g_1_1'] = np.where(base_dados['idade__R__pk_5'] == 0.0, 1,
np.where(base_dados['idade__R__pk_5'] == 1.0, 0,
np.where(base_dados['idade__R__pk_5'] == 2.0, 0,
np.where(base_dados['idade__R__pk_5'] == 3.0, 0,
np.where(base_dados['idade__R__pk_5'] == 4.0, 2,
 0)))))
base_dados['idade__R__pk_5_g_1_2'] = np.where(base_dados['idade__R__pk_5_g_1_1'] == 0, 2,
np.where(base_dados['idade__R__pk_5_g_1_1'] == 1, 0,
np.where(base_dados['idade__R__pk_5_g_1_1'] == 2, 1,
 0)))
base_dados['idade__R__pk_5_g_1'] = np.where(base_dados['idade__R__pk_5_g_1_2'] == 0, 0,
np.where(base_dados['idade__R__pk_5_g_1_2'] == 1, 1,
np.where(base_dados['idade__R__pk_5_g_1_2'] == 2, 2,
 0)))
         
         
         
         
         
         
         
base_dados['mob_cliente__p_6'] = np.where(base_dados['mob_cliente'] <= 10.385335365868034, 0.0,
np.where(np.bitwise_and(base_dados['mob_cliente'] > 10.385335365868034, base_dados['mob_cliente'] <= 16.364924269130935), 1.0,
np.where(np.bitwise_and(base_dados['mob_cliente'] > 16.364924269130935, base_dados['mob_cliente'] <= 25.07146855135439), 2.0,
np.where(np.bitwise_and(base_dados['mob_cliente'] > 25.07146855135439, base_dados['mob_cliente'] <= 37.52346961913812), 3.0,
np.where(base_dados['mob_cliente'] > 37.52346961913812, 4.0,
0)))))
base_dados['mob_cliente__p_6_g_1_1'] = np.where(base_dados['mob_cliente__p_6'] == 0.0, 2,
np.where(base_dados['mob_cliente__p_6'] == 1.0, 3,
np.where(base_dados['mob_cliente__p_6'] == 2.0, 0,
np.where(base_dados['mob_cliente__p_6'] == 3.0, 1,
np.where(base_dados['mob_cliente__p_6'] == 4.0, 0,
 0)))))
base_dados['mob_cliente__p_6_g_1_2'] = np.where(base_dados['mob_cliente__p_6_g_1_1'] == 0, 1,
np.where(base_dados['mob_cliente__p_6_g_1_1'] == 1, 3,
np.where(base_dados['mob_cliente__p_6_g_1_1'] == 2, 0,
np.where(base_dados['mob_cliente__p_6_g_1_1'] == 3, 1,
 0))))
base_dados['mob_cliente__p_6_g_1'] = np.where(base_dados['mob_cliente__p_6_g_1_2'] == 0, 0,
np.where(base_dados['mob_cliente__p_6_g_1_2'] == 1, 1,
np.where(base_dados['mob_cliente__p_6_g_1_2'] == 3, 2,
 0)))
         
         
         
         
         
         
         
base_dados['mob_cliente__L'] = np.log(base_dados['mob_cliente'])
np.where(base_dados['mob_cliente__L'] == 0, -1, base_dados['mob_cliente__L'])
base_dados['mob_cliente__L'] = base_dados['mob_cliente__L'].fillna(-2)
base_dados['mob_cliente__L__p_3'] = np.where(base_dados['mob_cliente__L'] <= 2.3403947501016065, 0.0,
np.where(np.bitwise_and(base_dados['mob_cliente__L'] > 2.3403947501016065, base_dados['mob_cliente__L'] <= 3.2217304884902513), 1.0,
np.where(base_dados['mob_cliente__L'] > 3.2217304884902513, 2.0,
 0)))
base_dados['mob_cliente__L__p_3_g_1_1'] = np.where(base_dados['mob_cliente__L__p_3'] == 0.0, 1,
np.where(base_dados['mob_cliente__L__p_3'] == 1.0, 0,
np.where(base_dados['mob_cliente__L__p_3'] == 2.0, 0,
 0)))
base_dados['mob_cliente__L__p_3_g_1_2'] = np.where(base_dados['mob_cliente__L__p_3_g_1_1'] == 0, 1,
np.where(base_dados['mob_cliente__L__p_3_g_1_1'] == 1, 0,
 0))
base_dados['mob_cliente__L__p_3_g_1'] = np.where(base_dados['mob_cliente__L__p_3_g_1_2'] == 0, 0,
np.where(base_dados['mob_cliente__L__p_3_g_1_2'] == 1, 1,
 0))
                                                 
                                                 
                                                 
                                                 
                                                 
                                                 
                                                 
base_dados['mob_refer__R'] = np.sqrt(base_dados['mob_refer'])
np.where(base_dados['mob_refer__R'] == 0, -1, base_dados['mob_refer__R'])
base_dados['mob_refer__R'] = base_dados['mob_refer__R'].fillna(-2)
base_dados['mob_refer__R__pu_6'] = np.where(base_dados['mob_refer__R'] <= -2.0, 0.0,
np.where(np.bitwise_and(base_dados['mob_refer__R'] > -2.0, base_dados['mob_refer__R'] <= 1.9361427586008055), 2.0,
np.where(np.bitwise_and(base_dados['mob_refer__R'] > 1.9361427586008055, base_dados['mob_refer__R'] <= 3.367213537192805), 3.0,
np.where(np.bitwise_and(base_dados['mob_refer__R'] > 3.367213537192805, base_dados['mob_refer__R'] <= 4.7095900833011966), 4.0,
np.where(base_dados['mob_refer__R'] > 4.7095900833011966, 5.0,
 0)))))
base_dados['mob_refer__R__pu_6_g_1_1'] = np.where(base_dados['mob_refer__R__pu_6'] == 0.0, 1,
np.where(base_dados['mob_refer__R__pu_6'] == 2.0, 1,
np.where(base_dados['mob_refer__R__pu_6'] == 3.0, 0,
np.where(base_dados['mob_refer__R__pu_6'] == 4.0, 0,
np.where(base_dados['mob_refer__R__pu_6'] == 5.0, 1,
 0)))))
base_dados['mob_refer__R__pu_6_g_1_2'] = np.where(base_dados['mob_refer__R__pu_6_g_1_1'] == 0, 1,
np.where(base_dados['mob_refer__R__pu_6_g_1_1'] == 1, 0,
 0))
base_dados['mob_refer__R__pu_6_g_1'] = np.where(base_dados['mob_refer__R__pu_6_g_1_2'] == 0, 0,
np.where(base_dados['mob_refer__R__pu_6_g_1_2'] == 1, 1,
 0))
                                                
                                                
                                                
                                                
                                                
                                                
                                                
base_dados['mob_refer__R'] = np.sqrt(base_dados['mob_refer'])
np.where(base_dados['mob_refer__R'] == 0, -1, base_dados['mob_refer__R'])
base_dados['mob_refer__R'] = base_dados['mob_refer__R'].fillna(-2)
base_dados['mob_refer__R__pe_10'] = np.where(np.bitwise_and(base_dados['mob_refer__R'] >= -2.0, base_dados['mob_refer__R'] <= 1.6120570209354146), 1.0,
np.where(np.bitwise_and(base_dados['mob_refer__R'] > 1.6120570209354146, base_dados['mob_refer__R'] <= 2.418958735292835), 2.0,
np.where(np.bitwise_and(base_dados['mob_refer__R'] > 2.418958735292835, base_dados['mob_refer__R'] <= 3.2226286423698403), 3.0,
np.where(np.bitwise_and(base_dados['mob_refer__R'] > 3.2226286423698403, base_dados['mob_refer__R'] <= 4.0127403600550196), 4.0,
np.where(np.bitwise_and(base_dados['mob_refer__R'] > 4.0127403600550196, base_dados['mob_refer__R'] <= 4.830130401517556), 5.0,
np.where(np.bitwise_and(base_dados['mob_refer__R'] > 4.830130401517556, base_dados['mob_refer__R'] <= 5.598816965406754), 6.0,
np.where(base_dados['mob_refer__R'] > 5.598816965406754, 7.0,
 -2)))))))
base_dados['mob_refer__R__pe_10_g_1_1'] = np.where(base_dados['mob_refer__R__pe_10'] == -2.0, 0,
np.where(base_dados['mob_refer__R__pe_10'] == 1.0, 1,
np.where(base_dados['mob_refer__R__pe_10'] == 2.0, 1,
np.where(base_dados['mob_refer__R__pe_10'] == 3.0, 0,
np.where(base_dados['mob_refer__R__pe_10'] == 4.0, 0,
np.where(base_dados['mob_refer__R__pe_10'] == 5.0, 1,
np.where(base_dados['mob_refer__R__pe_10'] == 6.0, 1,
np.where(base_dados['mob_refer__R__pe_10'] == 7.0, 1,
 0))))))))
base_dados['mob_refer__R__pe_10_g_1_2'] = np.where(base_dados['mob_refer__R__pe_10_g_1_1'] == 0, 0,
np.where(base_dados['mob_refer__R__pe_10_g_1_1'] == 1, 1,
 0))
base_dados['mob_refer__R__pe_10_g_1'] = np.where(base_dados['mob_refer__R__pe_10_g_1_2'] == 0, 0,
np.where(base_dados['mob_refer__R__pe_10_g_1_2'] == 1, 1,
 0))
                                                 
                                                 
                                                 
                                                 
                                                 
                                                 
                                                 
base_dados['mob_expir_prazo__L'] = np.log(base_dados['mob_expir_prazo'])
np.where(base_dados['mob_expir_prazo__L'] == 0, -1, base_dados['mob_expir_prazo__L'])
base_dados['mob_expir_prazo__L'] = base_dados['mob_expir_prazo__L'].fillna(-2)
base_dados['mob_expir_prazo__L__pe_13'] = np.where(np.bitwise_and(base_dados['mob_expir_prazo__L'] >= -2.0, base_dados['mob_expir_prazo__L'] <= 0.2955447684863883), 0.0,
np.where(np.bitwise_and(base_dados['mob_expir_prazo__L'] > 0.2955447684863883, base_dados['mob_expir_prazo__L'] <= 1.1371988236787351), 2.0,
np.where(np.bitwise_and(base_dados['mob_expir_prazo__L'] > 1.1371988236787351, base_dados['mob_expir_prazo__L'] <= 1.510894553089111), 3.0,
np.where(np.bitwise_and(base_dados['mob_expir_prazo__L'] > 1.510894553089111, base_dados['mob_expir_prazo__L'] <= 1.8821766097790396), 4.0,
np.where(np.bitwise_and(base_dados['mob_expir_prazo__L'] > 1.8821766097790396, base_dados['mob_expir_prazo__L'] <= 2.2743763093763456), 5.0,
np.where(np.bitwise_and(base_dados['mob_expir_prazo__L'] > 2.2743763093763456, base_dados['mob_expir_prazo__L'] <= 2.6548584610212975), 6.0,
np.where(np.bitwise_and(base_dados['mob_expir_prazo__L'] > 2.6548584610212975, base_dados['mob_expir_prazo__L'] <= 3.0106763705836173), 7.0,
np.where(base_dados['mob_expir_prazo__L'] > 3.0106763705836173, 8.0,
 -2))))))))
base_dados['mob_expir_prazo__L__pe_13_g_1_1'] = np.where(base_dados['mob_expir_prazo__L__pe_13'] == -2.0, 2,
np.where(base_dados['mob_expir_prazo__L__pe_13'] == 0.0, 2,
np.where(base_dados['mob_expir_prazo__L__pe_13'] == 2.0, 2,
np.where(base_dados['mob_expir_prazo__L__pe_13'] == 3.0, 2,
np.where(base_dados['mob_expir_prazo__L__pe_13'] == 4.0, 2,
np.where(base_dados['mob_expir_prazo__L__pe_13'] == 5.0, 2,
np.where(base_dados['mob_expir_prazo__L__pe_13'] == 6.0, 0,
np.where(base_dados['mob_expir_prazo__L__pe_13'] == 7.0, 1,
np.where(base_dados['mob_expir_prazo__L__pe_13'] == 8.0, 0,
 0)))))))))
base_dados['mob_expir_prazo__L__pe_13_g_1_2'] = np.where(base_dados['mob_expir_prazo__L__pe_13_g_1_1'] == 0, 1,
np.where(base_dados['mob_expir_prazo__L__pe_13_g_1_1'] == 1, 2,
np.where(base_dados['mob_expir_prazo__L__pe_13_g_1_1'] == 2, 0,
 0)))
base_dados['mob_expir_prazo__L__pe_13_g_1'] = np.where(base_dados['mob_expir_prazo__L__pe_13_g_1_2'] == 0, 0,
np.where(base_dados['mob_expir_prazo__L__pe_13_g_1_2'] == 1, 1,
np.where(base_dados['mob_expir_prazo__L__pe_13_g_1_2'] == 2, 2,
 0)))
         
         
         
         
         
         
         
base_dados['mob_expir_prazo__C'] = np.cos(base_dados['mob_expir_prazo'])
np.where(base_dados['mob_expir_prazo__C'] == 0, -1, base_dados['mob_expir_prazo__C'])
base_dados['mob_expir_prazo__C'] = base_dados['mob_expir_prazo__C'].fillna(-2)
base_dados['mob_expir_prazo__C__p_5'] = np.where(base_dados['mob_expir_prazo__C'] <= -0.9911948580615881, 0.0,
np.where(np.bitwise_and(base_dados['mob_expir_prazo__C'] > -0.9911948580615881, base_dados['mob_expir_prazo__C'] <= -0.7287643793408249), 1.0,
np.where(np.bitwise_and(base_dados['mob_expir_prazo__C'] > -0.7287643793408249, base_dados['mob_expir_prazo__C'] <= -0.06594444243037606), 2.0,
np.where(np.bitwise_and(base_dados['mob_expir_prazo__C'] > -0.06594444243037606, base_dados['mob_expir_prazo__C'] <= 0.7849430491742455), 3.0,
np.where(base_dados['mob_expir_prazo__C'] > 0.7849430491742455, 4.0,
 0)))))
base_dados['mob_expir_prazo__C__p_5_g_1_1'] = np.where(base_dados['mob_expir_prazo__C__p_5'] == 0.0, 2,
np.where(base_dados['mob_expir_prazo__C__p_5'] == 1.0, 1,
np.where(base_dados['mob_expir_prazo__C__p_5'] == 2.0, 2,
np.where(base_dados['mob_expir_prazo__C__p_5'] == 3.0, 0,
np.where(base_dados['mob_expir_prazo__C__p_5'] == 4.0, 2,
 0)))))
base_dados['mob_expir_prazo__C__p_5_g_1_2'] = np.where(base_dados['mob_expir_prazo__C__p_5_g_1_1'] == 0, 1,
np.where(base_dados['mob_expir_prazo__C__p_5_g_1_1'] == 1, 0,
np.where(base_dados['mob_expir_prazo__C__p_5_g_1_1'] == 2, 1,
 0)))
base_dados['mob_expir_prazo__C__p_5_g_1'] = np.where(base_dados['mob_expir_prazo__C__p_5_g_1_2'] == 0, 0,
np.where(base_dados['mob_expir_prazo__C__p_5_g_1_2'] == 1, 1,
 0))
                                                     
                                                     
                                                     
                                                     
                                                     
                                                     
                                                     
                                                     
base_dados['mob_contrato_cliente__pu_17'] = np.where(base_dados['mob_contrato_cliente'] <= -47.56398452347995, 0.0,
np.where(np.bitwise_and(base_dados['mob_contrato_cliente'] > -47.56398452347995, base_dados['mob_contrato_cliente'] <= -41.65010538838478), 1.0,
np.where(np.bitwise_and(base_dados['mob_contrato_cliente'] > -41.65010538838478, base_dados['mob_contrato_cliente'] <= -34.73962810274579), 2.0,
np.where(np.bitwise_and(base_dados['mob_contrato_cliente'] > -34.73962810274579, base_dados['mob_contrato_cliente'] <= -28.234361054141093), 3.0,
np.where(np.bitwise_and(base_dados['mob_contrato_cliente'] > -28.234361054141093, base_dados['mob_contrato_cliente'] <= -21.904320053983668), 4.0,
np.where(np.bitwise_and(base_dados['mob_contrato_cliente'] > -21.904320053983668, base_dados['mob_contrato_cliente'] <= -15.355246493267158), 5.0,
np.where(np.bitwise_and(base_dados['mob_contrato_cliente'] > -15.355246493267158, base_dados['mob_contrato_cliente'] <= -8.806172932550647), 6.0,
np.where(np.bitwise_and(base_dados['mob_contrato_cliente'] > -8.806172932550647, base_dados['mob_contrato_cliente'] <= -2.5856482126727576), 7.0,
np.where(np.bitwise_and(base_dados['mob_contrato_cliente'] > -2.5856482126727576, base_dados['mob_contrato_cliente'] <= 4.048182151621529), 8.0,
np.where(np.bitwise_and(base_dados['mob_contrato_cliente'] > 4.048182151621529, base_dados['mob_contrato_cliente'] <= 10.414283685179797), 9.0,
np.where(np.bitwise_and(base_dados['mob_contrato_cliente'] > 10.414283685179797, base_dados['mob_contrato_cliente'] <= 12.888350676445288), 10.0,
np.where(np.bitwise_and(base_dados['mob_contrato_cliente'] > 12.888350676445288, base_dados['mob_contrato_cliente'] <= 23.428254361202463), 11.0,
np.where(np.bitwise_and(base_dados['mob_contrato_cliente'] > 23.428254361202463, base_dados['mob_contrato_cliente'] <= 37.64684656776856), 14.0,
np.where(base_dados['mob_contrato_cliente'] > 37.64684656776856, 16.0,
 0))))))))))))))
base_dados['mob_contrato_cliente__pu_17_g_1_1'] = np.where(base_dados['mob_contrato_cliente__pu_17'] == 0.0, 1,
np.where(base_dados['mob_contrato_cliente__pu_17'] == 1.0, 1,
np.where(base_dados['mob_contrato_cliente__pu_17'] == 2.0, 1,
np.where(base_dados['mob_contrato_cliente__pu_17'] == 3.0, 1,
np.where(base_dados['mob_contrato_cliente__pu_17'] == 4.0, 1,
np.where(base_dados['mob_contrato_cliente__pu_17'] == 5.0, 0,
np.where(base_dados['mob_contrato_cliente__pu_17'] == 6.0, 0,
np.where(base_dados['mob_contrato_cliente__pu_17'] == 7.0, 1,
np.where(base_dados['mob_contrato_cliente__pu_17'] == 8.0, 0,
np.where(base_dados['mob_contrato_cliente__pu_17'] == 9.0, 1,
np.where(base_dados['mob_contrato_cliente__pu_17'] == 10.0, 0,
np.where(base_dados['mob_contrato_cliente__pu_17'] == 11.0, 1,
np.where(base_dados['mob_contrato_cliente__pu_17'] == 14.0, 1,
np.where(base_dados['mob_contrato_cliente__pu_17'] == 16.0, 1,
 0))))))))))))))
base_dados['mob_contrato_cliente__pu_17_g_1_2'] = np.where(base_dados['mob_contrato_cliente__pu_17_g_1_1'] == 0, 1,
np.where(base_dados['mob_contrato_cliente__pu_17_g_1_1'] == 1, 0,
 0))
base_dados['mob_contrato_cliente__pu_17_g_1'] = np.where(base_dados['mob_contrato_cliente__pu_17_g_1_2'] == 0, 0,
np.where(base_dados['mob_contrato_cliente__pu_17_g_1_2'] == 1, 1,
 0))
                                                         
                                                         
                                                         
                                                         
                                                         
                                                         
                                                         
base_dados['mob_contrato_cliente__S'] = np.sin(base_dados['mob_contrato_cliente'])
np.where(base_dados['mob_contrato_cliente__S'] == 0, -1, base_dados['mob_contrato_cliente__S'])
base_dados['mob_contrato_cliente__S'] = base_dados['mob_contrato_cliente__S'].fillna(-2)
base_dados['mob_contrato_cliente__S__pk_8'] = np.where(base_dados['mob_contrato_cliente__S'] <= -0.8081801760218511, 0.0,
np.where(np.bitwise_and(base_dados['mob_contrato_cliente__S'] > -0.8081801760218511, base_dados['mob_contrato_cliente__S'] <= -0.5277457458267376), 1.0,
np.where(np.bitwise_and(base_dados['mob_contrato_cliente__S'] > -0.5277457458267376, base_dados['mob_contrato_cliente__S'] <= -0.2087979049653195), 2.0,
np.where(np.bitwise_and(base_dados['mob_contrato_cliente__S'] > -0.2087979049653195, base_dados['mob_contrato_cliente__S'] <= 0.14139351292094235), 3.0,
np.where(np.bitwise_and(base_dados['mob_contrato_cliente__S'] > 0.14139351292094235, base_dados['mob_contrato_cliente__S'] <= 0.4489645058912951), 4.0,
np.where(np.bitwise_and(base_dados['mob_contrato_cliente__S'] > 0.4489645058912951, base_dados['mob_contrato_cliente__S'] <= 0.668631597641303), 5.0,
np.where(np.bitwise_and(base_dados['mob_contrato_cliente__S'] > 0.668631597641303, base_dados['mob_contrato_cliente__S'] <= 0.8362528814940743), 6.0,
np.where(base_dados['mob_contrato_cliente__S'] > 0.8362528814940743, 7.0,
 0))))))))
base_dados['mob_contrato_cliente__S__pk_8_g_1_1'] = np.where(base_dados['mob_contrato_cliente__S__pk_8'] == 0.0, 1,
np.where(base_dados['mob_contrato_cliente__S__pk_8'] == 1.0, 1,
np.where(base_dados['mob_contrato_cliente__S__pk_8'] == 2.0, 0,
np.where(base_dados['mob_contrato_cliente__S__pk_8'] == 3.0, 1,
np.where(base_dados['mob_contrato_cliente__S__pk_8'] == 4.0, 0,
np.where(base_dados['mob_contrato_cliente__S__pk_8'] == 5.0, 0,
np.where(base_dados['mob_contrato_cliente__S__pk_8'] == 6.0, 0,
np.where(base_dados['mob_contrato_cliente__S__pk_8'] == 7.0, 0,
 0))))))))
base_dados['mob_contrato_cliente__S__pk_8_g_1_2'] = np.where(base_dados['mob_contrato_cliente__S__pk_8_g_1_1'] == 0, 1,
np.where(base_dados['mob_contrato_cliente__S__pk_8_g_1_1'] == 1, 0,
 0))
base_dados['mob_contrato_cliente__S__pk_8_g_1'] = np.where(base_dados['mob_contrato_cliente__S__pk_8_g_1_2'] == 0, 0,
np.where(base_dados['mob_contrato_cliente__S__pk_8_g_1_2'] == 1, 1,
 0))
                                                           
                                                           
                                                           
                                                           
                                                           
                                                           
                                                           



# COMMAND ----------

# MAGIC %md
# MAGIC ### Parte 2 de 2

# COMMAND ----------

base_dados['val_compr__L__pu_20_g_1_c1_3_1'] = np.where(np.bitwise_and(base_dados['val_compr__R__pe_13_g_1'] == 0, base_dados['val_compr__L__pu_20_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['val_compr__R__pe_13_g_1'] == 0, base_dados['val_compr__L__pu_20_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['val_compr__R__pe_13_g_1'] == 1, base_dados['val_compr__L__pu_20_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['val_compr__R__pe_13_g_1'] == 1, base_dados['val_compr__L__pu_20_g_1'] == 1), 1,
 0))))
base_dados['val_compr__L__pu_20_g_1_c1_3_2'] = np.where(base_dados['val_compr__L__pu_20_g_1_c1_3_1'] == 0, 0,
np.where(base_dados['val_compr__L__pu_20_g_1_c1_3_1'] == 1, 1,
0))
base_dados['val_compr__L__pu_20_g_1_c1_3'] = np.where(base_dados['val_compr__L__pu_20_g_1_c1_3_2'] == 0, 0,
np.where(base_dados['val_compr__L__pu_20_g_1_c1_3_2'] == 1, 1,
 0))
                                                      
                                                      
                                                      
                                                      
                                                      
                                                      
                                                      
base_dados['des_cep_resid__C__pu_10_g_1_c1_26_1'] = np.where(np.bitwise_and(base_dados['des_cep_resid__pk_2_g_1'] == 0, base_dados['des_cep_resid__C__pu_10_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['des_cep_resid__pk_2_g_1'] == 0, base_dados['des_cep_resid__C__pu_10_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['des_cep_resid__pk_2_g_1'] == 1, base_dados['des_cep_resid__C__pu_10_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['des_cep_resid__pk_2_g_1'] == 1, base_dados['des_cep_resid__C__pu_10_g_1'] == 1), 2,
 0))))
base_dados['des_cep_resid__C__pu_10_g_1_c1_26_2'] = np.where(base_dados['des_cep_resid__C__pu_10_g_1_c1_26_1'] == 0, 0,
np.where(base_dados['des_cep_resid__C__pu_10_g_1_c1_26_1'] == 1, 1,
np.where(base_dados['des_cep_resid__C__pu_10_g_1_c1_26_1'] == 2, 2,
0)))
base_dados['des_cep_resid__C__pu_10_g_1_c1_26'] = np.where(base_dados['des_cep_resid__C__pu_10_g_1_c1_26_2'] == 0, 0,
np.where(base_dados['des_cep_resid__C__pu_10_g_1_c1_26_2'] == 1, 1,
np.where(base_dados['des_cep_resid__C__pu_10_g_1_c1_26_2'] == 2, 2,
 0)))
         
         
         
         
         
         
                
base_dados['des_fones_resid__C__p_10_g_1_c1_5_1'] = np.where(np.bitwise_and(base_dados['des_fones_resid__p_10_g_1'] == 0, base_dados['des_fones_resid__C__p_10_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['des_fones_resid__p_10_g_1'] == 0, base_dados['des_fones_resid__C__p_10_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['des_fones_resid__p_10_g_1'] == 1, base_dados['des_fones_resid__C__p_10_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['des_fones_resid__p_10_g_1'] == 1, base_dados['des_fones_resid__C__p_10_g_1'] == 1), 1,
 0))))
base_dados['des_fones_resid__C__p_10_g_1_c1_5_2'] = np.where(base_dados['des_fones_resid__C__p_10_g_1_c1_5_1'] == 0, 0,
np.where(base_dados['des_fones_resid__C__p_10_g_1_c1_5_1'] == 1, 1,
0))
base_dados['des_fones_resid__C__p_10_g_1_c1_5'] = np.where(base_dados['des_fones_resid__C__p_10_g_1_c1_5_2'] == 0, 0,
np.where(base_dados['des_fones_resid__C__p_10_g_1_c1_5_2'] == 1, 1,
 0))
                                                           
                                                           
                                                           
                                                           
                                                           
                                                           
                                                           
                                                           
base_dados['des_fones_celul__C__p_6_g_1_c1_3_1'] = np.where(np.bitwise_and(base_dados['des_fones_celul__C__p_6_g_1'] == 0, base_dados['des_fones_celul__C__pu_5_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['des_fones_celul__C__p_6_g_1'] == 0, base_dados['des_fones_celul__C__pu_5_g_1'] == 1), 0,
np.where(np.bitwise_and(base_dados['des_fones_celul__C__p_6_g_1'] == 1, base_dados['des_fones_celul__C__pu_5_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['des_fones_celul__C__p_6_g_1'] == 1, base_dados['des_fones_celul__C__pu_5_g_1'] == 1), 1,
 0))))
base_dados['des_fones_celul__C__p_6_g_1_c1_3_2'] = np.where(base_dados['des_fones_celul__C__p_6_g_1_c1_3_1'] == 0, 0,
np.where(base_dados['des_fones_celul__C__p_6_g_1_c1_3_1'] == 1, 1,
0))
base_dados['des_fones_celul__C__p_6_g_1_c1_3'] = np.where(base_dados['des_fones_celul__C__p_6_g_1_c1_3_2'] == 0, 0,
np.where(base_dados['des_fones_celul__C__p_6_g_1_c1_3_2'] == 1, 1,
 0))
                                                          
                                                          
                                                          
                                                          
                                                          
                                                          
                                                          
                                                          
base_dados['mob_contrato__L10__pe_17_g_1_c1_3_1'] = np.where(np.bitwise_and(base_dados['mob_contrato__L10__pe_17_g_1'] == 0, base_dados['mob_contrato__T__pu_17_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['mob_contrato__L10__pe_17_g_1'] == 0, base_dados['mob_contrato__T__pu_17_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['mob_contrato__L10__pe_17_g_1'] == 1, base_dados['mob_contrato__T__pu_17_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['mob_contrato__L10__pe_17_g_1'] == 1, base_dados['mob_contrato__T__pu_17_g_1'] == 1), 1,
 0))))
base_dados['mob_contrato__L10__pe_17_g_1_c1_3_2'] = np.where(base_dados['mob_contrato__L10__pe_17_g_1_c1_3_1'] == 0, 0,
np.where(base_dados['mob_contrato__L10__pe_17_g_1_c1_3_1'] == 1, 1,
0))
base_dados['mob_contrato__L10__pe_17_g_1_c1_3'] = np.where(base_dados['mob_contrato__L10__pe_17_g_1_c1_3_2'] == 0, 0,
np.where(base_dados['mob_contrato__L10__pe_17_g_1_c1_3_2'] == 1, 1,
 0))
                                                           
                                                           
                                                           
                                                           
                                                           
                                                           
                                                           
                                                           
base_dados['idade__R__pk_5_g_1_c1_29_1'] = np.where(np.bitwise_and(base_dados['idade__pe_3_g_1'] == 0, base_dados['idade__R__pk_5_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['idade__pe_3_g_1'] == 0, base_dados['idade__R__pk_5_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['idade__pe_3_g_1'] == 0, base_dados['idade__R__pk_5_g_1'] == 2), 2,
np.where(np.bitwise_and(base_dados['idade__pe_3_g_1'] == 1, base_dados['idade__R__pk_5_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['idade__pe_3_g_1'] == 1, base_dados['idade__R__pk_5_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['idade__pe_3_g_1'] == 1, base_dados['idade__R__pk_5_g_1'] == 2), 3,
 0))))))
base_dados['idade__R__pk_5_g_1_c1_29_2'] = np.where(base_dados['idade__R__pk_5_g_1_c1_29_1'] == 0, 0,
np.where(base_dados['idade__R__pk_5_g_1_c1_29_1'] == 1, 1,
np.where(base_dados['idade__R__pk_5_g_1_c1_29_1'] == 2, 2,
np.where(base_dados['idade__R__pk_5_g_1_c1_29_1'] == 3, 3,
0))))
base_dados['idade__R__pk_5_g_1_c1_29'] = np.where(base_dados['idade__R__pk_5_g_1_c1_29_2'] == 0, 0,
np.where(base_dados['idade__R__pk_5_g_1_c1_29_2'] == 1, 1,
np.where(base_dados['idade__R__pk_5_g_1_c1_29_2'] == 2, 2,
np.where(base_dados['idade__R__pk_5_g_1_c1_29_2'] == 3, 3,
 0))))
         
         
         
         
         
         
         
         
         
base_dados['mob_cliente__L__p_3_g_1_c1_23_1'] = np.where(np.bitwise_and(base_dados['mob_cliente__p_6_g_1'] == 0, base_dados['mob_cliente__L__p_3_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['mob_cliente__p_6_g_1'] == 0, base_dados['mob_cliente__L__p_3_g_1'] == 1), 0,
np.where(np.bitwise_and(base_dados['mob_cliente__p_6_g_1'] == 1, base_dados['mob_cliente__L__p_3_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['mob_cliente__p_6_g_1'] == 1, base_dados['mob_cliente__L__p_3_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['mob_cliente__p_6_g_1'] == 2, base_dados['mob_cliente__L__p_3_g_1'] == 0), 2,    
np.where(np.bitwise_and(base_dados['mob_cliente__p_6_g_1'] == 2, base_dados['mob_cliente__L__p_3_g_1'] == 1), 2,
 0))))))
base_dados['mob_cliente__L__p_3_g_1_c1_23_2'] = np.where(base_dados['mob_cliente__L__p_3_g_1_c1_23_1'] == 0, 0,
np.where(base_dados['mob_cliente__L__p_3_g_1_c1_23_1'] == 1, 1,
np.where(base_dados['mob_cliente__L__p_3_g_1_c1_23_1'] == 2, 2,
0)))
base_dados['mob_cliente__L__p_3_g_1_c1_23'] = np.where(base_dados['mob_cliente__L__p_3_g_1_c1_23_2'] == 0, 0,
np.where(base_dados['mob_cliente__L__p_3_g_1_c1_23_2'] == 1, 1,
np.where(base_dados['mob_cliente__L__p_3_g_1_c1_23_2'] == 2, 2,
 0)))
         
         
         
         
         
         
         
base_dados['mob_refer__R__pe_10_g_1_c1_9_1'] = np.where(np.bitwise_and(base_dados['mob_refer__R__pu_6_g_1'] == 0, base_dados['mob_refer__R__pe_10_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['mob_refer__R__pu_6_g_1'] == 0, base_dados['mob_refer__R__pe_10_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['mob_refer__R__pu_6_g_1'] == 1, base_dados['mob_refer__R__pe_10_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['mob_refer__R__pu_6_g_1'] == 1, base_dados['mob_refer__R__pe_10_g_1'] == 1), 2,
 0))))
base_dados['mob_refer__R__pe_10_g_1_c1_9_2'] = np.where(base_dados['mob_refer__R__pe_10_g_1_c1_9_1'] == 0, 0,
np.where(base_dados['mob_refer__R__pe_10_g_1_c1_9_1'] == 1, 1,
np.where(base_dados['mob_refer__R__pe_10_g_1_c1_9_1'] == 2, 2,
0)))
base_dados['mob_refer__R__pe_10_g_1_c1_9'] = np.where(base_dados['mob_refer__R__pe_10_g_1_c1_9_2'] == 0, 0,
np.where(base_dados['mob_refer__R__pe_10_g_1_c1_9_2'] == 1, 1,
np.where(base_dados['mob_refer__R__pe_10_g_1_c1_9_2'] == 2, 2,
 0)))
         
         
         
         
         
         
         
         
base_dados['mob_expir_prazo__C__p_5_g_1_c1_13_1'] = np.where(np.bitwise_and(base_dados['mob_expir_prazo__L__pe_13_g_1'] == 0, base_dados['mob_expir_prazo__C__p_5_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['mob_expir_prazo__L__pe_13_g_1'] == 0, base_dados['mob_expir_prazo__C__p_5_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['mob_expir_prazo__L__pe_13_g_1'] == 1, base_dados['mob_expir_prazo__C__p_5_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['mob_expir_prazo__L__pe_13_g_1'] == 1, base_dados['mob_expir_prazo__C__p_5_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['mob_expir_prazo__L__pe_13_g_1'] == 2, base_dados['mob_expir_prazo__C__p_5_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['mob_expir_prazo__L__pe_13_g_1'] == 2, base_dados['mob_expir_prazo__C__p_5_g_1'] == 1), 3,
 0))))))
base_dados['mob_expir_prazo__C__p_5_g_1_c1_13_2'] = np.where(base_dados['mob_expir_prazo__C__p_5_g_1_c1_13_1'] == 0, 1,
np.where(base_dados['mob_expir_prazo__C__p_5_g_1_c1_13_1'] == 1, 0,
np.where(base_dados['mob_expir_prazo__C__p_5_g_1_c1_13_1'] == 2, 2,
np.where(base_dados['mob_expir_prazo__C__p_5_g_1_c1_13_1'] == 3, 3,
0))))
base_dados['mob_expir_prazo__C__p_5_g_1_c1_13'] = np.where(base_dados['mob_expir_prazo__C__p_5_g_1_c1_13_2'] == 0, 0,
np.where(base_dados['mob_expir_prazo__C__p_5_g_1_c1_13_2'] == 1, 1,
np.where(base_dados['mob_expir_prazo__C__p_5_g_1_c1_13_2'] == 2, 2,
np.where(base_dados['mob_expir_prazo__C__p_5_g_1_c1_13_2'] == 3, 3,
 0))))
         
         
         
         
         
         
         
base_dados['mob_contrato_cliente__S__pk_8_g_1_c1_52_1'] = np.where(np.bitwise_and(base_dados['mob_contrato_cliente__pu_17_g_1'] == 0, base_dados['mob_contrato_cliente__S__pk_8_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['mob_contrato_cliente__pu_17_g_1'] == 0, base_dados['mob_contrato_cliente__S__pk_8_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['mob_contrato_cliente__pu_17_g_1'] == 1, base_dados['mob_contrato_cliente__S__pk_8_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['mob_contrato_cliente__pu_17_g_1'] == 1, base_dados['mob_contrato_cliente__S__pk_8_g_1'] == 1), 2,
 0))))
base_dados['mob_contrato_cliente__S__pk_8_g_1_c1_52_2'] = np.where(base_dados['mob_contrato_cliente__S__pk_8_g_1_c1_52_1'] == 0, 0,
np.where(base_dados['mob_contrato_cliente__S__pk_8_g_1_c1_52_1'] == 1, 1,
np.where(base_dados['mob_contrato_cliente__S__pk_8_g_1_c1_52_1'] == 2, 2,
0)))
base_dados['mob_contrato_cliente__S__pk_8_g_1_c1_52'] = np.where(base_dados['mob_contrato_cliente__S__pk_8_g_1_c1_52_2'] == 0, 0,
np.where(base_dados['mob_contrato_cliente__S__pk_8_g_1_c1_52_2'] == 1, 1,
np.where(base_dados['mob_contrato_cliente__S__pk_8_g_1_c1_52_2'] == 2, 2,
 0)))
         
    


# COMMAND ----------

# MAGIC %md
# MAGIC ## Mantendo apenas as variáveis do modelo

# COMMAND ----------

import pickle
for file in os.listdir(os.path.join(credor.caminho_pickle_dbfs, modelo_escolhido)):
  if file.split('.')[-1]=='sav':
    break
modelo=pickle.load(open(os.path.join(credor.caminho_pickle_dbfs, modelo_escolhido, file), 'rb'))

base_teste_c0 = base_dados[[chave,'ind_sexo_estad_civil_gh51','ind_estad_resid_gh38','mob_contrato_cliente__S__pk_8_g_1_c1_52','val_compr__L__pu_20_g_1_c1_3','mob_refer__R__pe_10_g_1_c1_9','idade__R__pk_5_g_1_c1_29','mob_contrato__L10__pe_17_g_1_c1_3','mob_cliente__L__p_3_g_1_c1_23','des_cep_resid__C__pu_10_g_1_c1_26','des_fones_resid__C__p_10_g_1_c1_5','mob_expir_prazo__C__p_5_g_1_c1_13','des_fones_celul__C__p_6_g_1_c1_3']]

var_fin_c0=['ind_sexo_estad_civil_gh51','ind_estad_resid_gh38','mob_contrato_cliente__S__pk_8_g_1_c1_52','val_compr__L__pu_20_g_1_c1_3','mob_refer__R__pe_10_g_1_c1_9','idade__R__pk_5_g_1_c1_29','mob_contrato__L10__pe_17_g_1_c1_3','mob_cliente__L__p_3_g_1_c1_23','des_cep_resid__C__pu_10_g_1_c1_26','des_fones_resid__C__p_10_g_1_c1_5','mob_expir_prazo__C__p_5_g_1_c1_13','des_fones_celul__C__p_6_g_1_c1_3']

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
x_teste2['P_1_L'] = np.log(x_teste2['P_1']) 
       
#marcando os valores igual a 0 como -1
x_teste2['P_1_L'] = np.where(x_teste2['P_1_L'] == 0, -1, x_teste2['P_1_L'])        
        
#marcando os valores igual a missing como -2
x_teste2['P_1_L'] = x_teste2['P_1_L'].fillna(-2)

x_teste2['GH'] = np.where(np.bitwise_and(x_teste2['P_1_L'] > -1.811509187, x_teste2['P_1_L'] <= -1.417333913), 0,
    np.where(np.bitwise_and(x_teste2['P_1_L'] > -1.417333913, x_teste2['P_1_L'] <= -1.268703147), 2,1))

del x_teste2['P_1_L']

x_teste2


# COMMAND ----------

x_teste2.groupby(['GH'])['P_1'].count()

# COMMAND ----------

data = datetime.today().date()
data = str(data.year)+str(data.month).zfill(2)+str(data.day).zfill(2)
print (data)

try:
  dbutils.fs.mkdirs(credor.caminho_pre_output)
except:
  pass

x_teste2.to_parquet(os.path.join(credor.caminho_pre_output_dbfs, 'pre_output_'+data+'.PARQUET'))

# COMMAND ----------

dbutils.notebook.exit('OK')