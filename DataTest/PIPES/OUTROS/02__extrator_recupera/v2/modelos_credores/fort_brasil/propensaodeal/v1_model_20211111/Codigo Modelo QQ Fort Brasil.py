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
chave = 'DOCUMENTO'

#Nome da Base de Dados
N_Base = "amostra_aleatoria.csv"

#Caminho da base de dados
caminho_base = "Base_Dados_Ferramenta/Fort_Brasil/"

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
base_dados = base_dados[[chave,'des_cep_resid','dat_expir_prazo','dat_cadas_clien','des_estad_resid']]

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

base_dados['des_cep_resid'] = base_dados['des_cep_resid'].replace(np.nan, '-3')

base_dados = base_dados.apply(pd.to_numeric, errors='ignore')

base_dados['des_cep_resid'] = base_dados['des_cep_resid'] / 1000000

base_dados['DOCUMENTO'] = base_dados['DOCUMENTO'].astype(np.int64)
base_dados['des_cep_resid'] = base_dados['des_cep_resid'].astype(int)

base_dados['dat_cadas_clien'] = pd.to_datetime(base_dados['dat_cadas_clien'])
base_dados['dat_expir_prazo'] = pd.to_datetime(base_dados['dat_expir_prazo'])

base_dados['mob_cliente'] = ((datetime.today()) - base_dados.dat_cadas_clien)/np.timedelta64(1, 'M')
base_dados['year_cliente'] = ((datetime.today()) - base_dados.dat_cadas_clien)/np.timedelta64(1, 'Y')
base_dados['mob_expir_prazo'] = (base_dados.dat_expir_prazo - (datetime.today()))/np.timedelta64(1, 'M')

base_dados['mob_cliente'] = base_dados['mob_cliente'].replace(np.nan, -3)
base_dados['year_cliente'] = base_dados['year_cliente'].replace(np.nan, -3)
base_dados['mob_expir_prazo'] = base_dados['mob_expir_prazo'].replace(np.nan, -3)

del base_dados['dat_cadas_clien']
del base_dados['dat_expir_prazo']
del base_dados['des_estad_resid']

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


# COMMAND ----------

# MAGIC %md
# MAGIC ## Gerando variáveis numéricas contínuas

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parte 1 de 2

# COMMAND ----------


base_dados['des_cep_resid__pe_4'] = np.where(base_dados['des_cep_resid'] == 0 , -1.0,
    np.where(np.bitwise_and(base_dados['des_cep_resid'] > 0.0, base_dados['des_cep_resid'] <= 18.0), 0.0,
    np.where(np.bitwise_and(base_dados['des_cep_resid'] > 18.0, base_dados['des_cep_resid'] <= 37.0), 1.0,
    np.where(np.bitwise_and(base_dados['des_cep_resid'] > 37.0, base_dados['des_cep_resid'] <= 55.0), 2.0,
    np.where(base_dados['des_cep_resid'] > 55.0, 3.0,
     -2)))))
         
base_dados['des_cep_resid__pe_4_g_1_1'] = np.where(base_dados['des_cep_resid__pe_4'] == -2.0, 1,
    np.where(base_dados['des_cep_resid__pe_4'] == -1.0, 1,
    np.where(base_dados['des_cep_resid__pe_4'] == 0.0, 1,
    np.where(base_dados['des_cep_resid__pe_4'] == 1.0, 1,
    np.where(base_dados['des_cep_resid__pe_4'] == 2.0, 0,
    np.where(base_dados['des_cep_resid__pe_4'] == 3.0, 0,
     0))))))
         
base_dados['des_cep_resid__pe_4_g_1_2'] = np.where(base_dados['des_cep_resid__pe_4_g_1_1'] == 0, 1,
    np.where(base_dados['des_cep_resid__pe_4_g_1_1'] == 1, 0,
     0))
                                                   
base_dados['des_cep_resid__pe_4_g_1'] = np.where(base_dados['des_cep_resid__pe_4_g_1_2'] == 0, 0,
    np.where(base_dados['des_cep_resid__pe_4_g_1_2'] == 1, 1,
     0))
                                                 
                                                 
                                                 
                                                 
                                                 
base_dados['des_cep_resid__S'] = np.sin(base_dados['des_cep_resid'])
np.where(base_dados['des_cep_resid__S'] == 0, -1, base_dados['des_cep_resid__S'])
base_dados['des_cep_resid__S'] = base_dados['des_cep_resid__S'].fillna(-2)

base_dados['des_cep_resid__S__p_4'] = np.where(np.bitwise_and(base_dados['des_cep_resid__S'] >= -1.0, base_dados['des_cep_resid__S'] <= 0.6367380071391379), 2.0,
np.where(base_dados['des_cep_resid__S'] > 0.6367380071391379, 3.0,
 -2))
                                               
base_dados['des_cep_resid__S__p_4_g_1_1'] = np.where(base_dados['des_cep_resid__S__p_4'] == -2.0, 1,
np.where(base_dados['des_cep_resid__S__p_4'] == 2.0, 0,
np.where(base_dados['des_cep_resid__S__p_4'] == 3.0, 1,
 0)))
         
base_dados['des_cep_resid__S__p_4_g_1_2'] = np.where(base_dados['des_cep_resid__S__p_4_g_1_1'] == 0, 1,
np.where(base_dados['des_cep_resid__S__p_4_g_1_1'] == 1, 0,
 0))
                                                     
base_dados['des_cep_resid__S__p_4_g_1'] = np.where(base_dados['des_cep_resid__S__p_4_g_1_2'] == 0, 0,
np.where(base_dados['des_cep_resid__S__p_4_g_1_2'] == 1, 1,
 0))
                                                   
                                                   
                                                   
                                                   
                                                   
base_dados['mob_cliente__L'] = np.log(base_dados['mob_cliente'])
np.where(base_dados['mob_cliente__L'] == 0, -1, base_dados['mob_cliente__L'])
base_dados['mob_cliente__L'] = base_dados['mob_cliente__L'].fillna(-2)

base_dados['mob_cliente__L__pe_4'] = np.where(np.bitwise_and(base_dados['mob_cliente__L'] >= -2.0, base_dados['mob_cliente__L'] <= 1.416740914403913), 0.0,
np.where(np.bitwise_and(base_dados['mob_cliente__L'] > 1.416740914403913, base_dados['mob_cliente__L'] <= 2.8158325674537856), 1.0,
np.where(base_dados['mob_cliente__L'] > 2.8158325674537856, 2.0,
 -2)))
         
base_dados['mob_cliente__L__pe_4_g_1_1'] = np.where(base_dados['mob_cliente__L__pe_4'] == -2.0, 1,
np.where(base_dados['mob_cliente__L__pe_4'] == 0.0, 1,
np.where(base_dados['mob_cliente__L__pe_4'] == 1.0, 0,
np.where(base_dados['mob_cliente__L__pe_4'] == 2.0, 0,
 0))))
         
base_dados['mob_cliente__L__pe_4_g_1_2'] = np.where(base_dados['mob_cliente__L__pe_4_g_1_1'] == 0, 1,
np.where(base_dados['mob_cliente__L__pe_4_g_1_1'] == 1, 0,
 0))
                                                    
base_dados['mob_cliente__L__pe_4_g_1'] = np.where(base_dados['mob_cliente__L__pe_4_g_1_2'] == 0, 0,
np.where(base_dados['mob_cliente__L__pe_4_g_1_2'] == 1, 1,
 0))
                                                  
                                                  
                                                  
                                                  
                                                  
                                                  
base_dados['mob_cliente__S'] = np.sin(base_dados['mob_cliente'])
np.where(base_dados['mob_cliente__S'] == 0, -1, base_dados['mob_cliente__S'])
base_dados['mob_cliente__S'] = base_dados['mob_cliente__S'].fillna(-2)

base_dados['mob_cliente__S__p_10'] = np.where(np.bitwise_and(base_dados['mob_cliente__S'] >= -0.999580928886329, base_dados['mob_cliente__S'] <= 0.43180987312250607), 7.0,
np.where(np.bitwise_and(base_dados['mob_cliente__S'] > 0.43180987312250607, base_dados['mob_cliente__S'] <= 0.8483886142734363), 8.0,
np.where(base_dados['mob_cliente__S'] > 0.8483886142734363, 9.0,
 -2)))
         
base_dados['mob_cliente__S__p_10_g_1_1'] = np.where(base_dados['mob_cliente__S__p_10'] == -2.0, 0,
np.where(base_dados['mob_cliente__S__p_10'] == 7.0, 0,
np.where(base_dados['mob_cliente__S__p_10'] == 8.0, 1,
np.where(base_dados['mob_cliente__S__p_10'] == 9.0, 1,
 0))))
         
base_dados['mob_cliente__S__p_10_g_1_2'] = np.where(base_dados['mob_cliente__S__p_10_g_1_1'] == 0, 0,
np.where(base_dados['mob_cliente__S__p_10_g_1_1'] == 1, 1,
 0))
                                                    
base_dados['mob_cliente__S__p_10_g_1'] = np.where(base_dados['mob_cliente__S__p_10_g_1_2'] == 0, 0,
np.where(base_dados['mob_cliente__S__p_10_g_1_2'] == 1, 1,
 0))
                                                  
                                                  
                                                  
                                                  
                                                  
                                                  
base_dados['year_cliente__T'] = np.tan(base_dados['year_cliente'])
np.where(base_dados['year_cliente__T'] == 0, -1, base_dados['year_cliente__T'])
base_dados['year_cliente__T'] = base_dados['year_cliente__T'].fillna(-2)

base_dados['year_cliente__T__p_10'] = np.where(np.bitwise_and(base_dados['year_cliente__T'] >= -130.9218378563028, base_dados['year_cliente__T'] <= 0.1425465430742778), 2.0,
np.where(np.bitwise_and(base_dados['year_cliente__T'] > 0.1425465430742778, base_dados['year_cliente__T'] <= 0.284001233952946), 4.0,
np.where(np.bitwise_and(base_dados['year_cliente__T'] > 0.284001233952946, base_dados['year_cliente__T'] <= 0.5746912548800013), 6.0,
np.where(np.bitwise_and(base_dados['year_cliente__T'] > 0.5746912548800013, base_dados['year_cliente__T'] <= 1.1706187030660358), 7.0,
np.where(np.bitwise_and(base_dados['year_cliente__T'] > 1.1706187030660358, base_dados['year_cliente__T'] <= 1.8353932199417753), 8.0,
np.where(base_dados['year_cliente__T'] > 1.8353932199417753, 9.0,
 -2))))))
         
base_dados['year_cliente__T__p_10_g_1_1'] = np.where(base_dados['year_cliente__T__p_10'] == -2.0, 0,
np.where(base_dados['year_cliente__T__p_10'] == 2.0, 1,
np.where(base_dados['year_cliente__T__p_10'] == 4.0, 0,
np.where(base_dados['year_cliente__T__p_10'] == 6.0, 1,
np.where(base_dados['year_cliente__T__p_10'] == 7.0, 1,
np.where(base_dados['year_cliente__T__p_10'] == 8.0, 0,
np.where(base_dados['year_cliente__T__p_10'] == 9.0, 1,
 0)))))))
         
base_dados['year_cliente__T__p_10_g_1_2'] = np.where(base_dados['year_cliente__T__p_10_g_1_1'] == 0, 1,
np.where(base_dados['year_cliente__T__p_10_g_1_1'] == 1, 0,
 0))
                                                     
base_dados['year_cliente__T__p_10_g_1'] = np.where(base_dados['year_cliente__T__p_10_g_1_2'] == 0, 0,
np.where(base_dados['year_cliente__T__p_10_g_1_2'] == 1, 1,
 0))
                                                   
                                                   
                                                   
                                                   
                                                   
                                                   
base_dados['year_cliente__T'] = np.tan(base_dados['year_cliente'])
np.where(base_dados['year_cliente__T'] == 0, -1, base_dados['year_cliente__T'])
base_dados['year_cliente__T'] = base_dados['year_cliente__T'].fillna(-2)

base_dados['year_cliente__T__pe_3'] = np.where(np.bitwise_and(base_dados['year_cliente__T'] >= -130.9218378563028, base_dados['year_cliente__T'] <= 21.206637181236626), 0.0,
np.where(np.bitwise_and(base_dados['year_cliente__T'] > 21.206637181236626, base_dados['year_cliente__T'] <= 35.76272030544593), 1.0,
np.where(base_dados['year_cliente__T'] > 35.76272030544593, 2.0,
 -2)))
         
base_dados['year_cliente__T__pe_3_g_1_1'] = np.where(base_dados['year_cliente__T__pe_3'] == -2.0, 1,
np.where(base_dados['year_cliente__T__pe_3'] == 0.0, 0,
np.where(base_dados['year_cliente__T__pe_3'] == 1.0, 1,
np.where(base_dados['year_cliente__T__pe_3'] == 2.0, 1,
 0))))
         
base_dados['year_cliente__T__pe_3_g_1_2'] = np.where(base_dados['year_cliente__T__pe_3_g_1_1'] == 0, 0,
np.where(base_dados['year_cliente__T__pe_3_g_1_1'] == 1, 1,
 0))
                                                     
base_dados['year_cliente__T__pe_3_g_1'] = np.where(base_dados['year_cliente__T__pe_3_g_1_2'] == 0, 0,
np.where(base_dados['year_cliente__T__pe_3_g_1_2'] == 1, 1,
 0))
                                                   
                                                   
                                                   
                                                     
base_dados['mob_expir_prazo__p_2'] = np.where(np.bitwise_and(base_dados['mob_expir_prazo'] >= -9.938973644148522, base_dados['mob_expir_prazo'] <= 15.359287100425288), 0.0,
np.where(base_dados['mob_expir_prazo'] > 15.359287100425288, 1.0,
 -2))
                                              
base_dados['mob_expir_prazo__p_2_g_1_1'] = np.where(base_dados['mob_expir_prazo__p_2'] == -2.0, 1,
np.where(base_dados['mob_expir_prazo__p_2'] == 0.0, 1,
np.where(base_dados['mob_expir_prazo__p_2'] == 1.0, 0,
 0)))
         
base_dados['mob_expir_prazo__p_2_g_1_2'] = np.where(base_dados['mob_expir_prazo__p_2_g_1_1'] == 0, 1,
np.where(base_dados['mob_expir_prazo__p_2_g_1_1'] == 1, 0,
 0))
                                                    
base_dados['mob_expir_prazo__p_2_g_1'] = np.where(base_dados['mob_expir_prazo__p_2_g_1_2'] == 0, 0,
np.where(base_dados['mob_expir_prazo__p_2_g_1_2'] == 1, 1,
 0))
                                                  
                                                  
                                                  
                                                  
                                                  
                                                  
base_dados['mob_expir_prazo__L'] = np.log(base_dados['mob_expir_prazo'])
np.where(base_dados['mob_expir_prazo__L'] == 0, -1, base_dados['mob_expir_prazo__L'])
base_dados['mob_expir_prazo__L'] = base_dados['mob_expir_prazo__L'].fillna(-2)

base_dados['mob_expir_prazo__L__p_13'] = np.where(np.bitwise_and(base_dados['mob_expir_prazo__L'] >= -2.0, base_dados['mob_expir_prazo__L'] <= 1.9290149406783226), 2.0,
np.where(np.bitwise_and(base_dados['mob_expir_prazo__L'] > 1.9290149406783226, base_dados['mob_expir_prazo__L'] <= 2.5045518392387467), 3.0,
np.where(np.bitwise_and(base_dados['mob_expir_prazo__L'] > 2.5045518392387467, base_dados['mob_expir_prazo__L'] <= 2.6041266962813405), 4.0,
np.where(np.bitwise_and(base_dados['mob_expir_prazo__L'] > 2.6041266962813405, base_dados['mob_expir_prazo__L'] <= 2.7209672608538296), 5.0,
np.where(np.bitwise_and(base_dados['mob_expir_prazo__L'] > 2.7209672608538296, base_dados['mob_expir_prazo__L'] <= 2.7797473774625474), 6.0,
np.where(np.bitwise_and(base_dados['mob_expir_prazo__L'] > 2.7797473774625474, base_dados['mob_expir_prazo__L'] <= 2.8505750050765584), 7.0,
np.where(np.bitwise_and(base_dados['mob_expir_prazo__L'] > 2.8505750050765584, base_dados['mob_expir_prazo__L'] <= 2.876818866436721), 8.0,
np.where(np.bitwise_and(base_dados['mob_expir_prazo__L'] > 2.876818866436721, base_dados['mob_expir_prazo__L'] <= 2.905992001089515), 9.0,
np.where(np.bitwise_and(base_dados['mob_expir_prazo__L'] > 2.905992001089515, base_dados['mob_expir_prazo__L'] <= 3.0355872979098786), 10.0,
np.where(base_dados['mob_expir_prazo__L'] > 3.0355872979098786, 11.0,
 -2))))))))))
         
base_dados['mob_expir_prazo__L__p_13_g_1_1'] = np.where(base_dados['mob_expir_prazo__L__p_13'] == -2.0, 1,
np.where(base_dados['mob_expir_prazo__L__p_13'] == 2.0, 3,
np.where(base_dados['mob_expir_prazo__L__p_13'] == 3.0, 3,
np.where(base_dados['mob_expir_prazo__L__p_13'] == 4.0, 3,
np.where(base_dados['mob_expir_prazo__L__p_13'] == 5.0, 2,
np.where(base_dados['mob_expir_prazo__L__p_13'] == 6.0, 2,
np.where(base_dados['mob_expir_prazo__L__p_13'] == 7.0, 0,
np.where(base_dados['mob_expir_prazo__L__p_13'] == 8.0, 0,
np.where(base_dados['mob_expir_prazo__L__p_13'] == 9.0, 2,
np.where(base_dados['mob_expir_prazo__L__p_13'] == 10.0, 2,
np.where(base_dados['mob_expir_prazo__L__p_13'] == 11.0, 2,
 0)))))))))))
         
base_dados['mob_expir_prazo__L__p_13_g_1_2'] = np.where(base_dados['mob_expir_prazo__L__p_13_g_1_1'] == 0, 2,
np.where(base_dados['mob_expir_prazo__L__p_13_g_1_1'] == 1, 0,
np.where(base_dados['mob_expir_prazo__L__p_13_g_1_1'] == 2, 2,
np.where(base_dados['mob_expir_prazo__L__p_13_g_1_1'] == 3, 0,
 0))))
         
base_dados['mob_expir_prazo__L__p_13_g_1'] = np.where(base_dados['mob_expir_prazo__L__p_13_g_1_2'] == 0, 0,
np.where(base_dados['mob_expir_prazo__L__p_13_g_1_2'] == 2, 1,
 0))
                                                      
                                                      
                                                      
                                                      
                                                      
                                            

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parte 2 de 2

# COMMAND ----------

base_dados['des_cep_resid__S__p_4_g_1_c1_12_1'] = np.where(np.bitwise_and(base_dados['des_cep_resid__pe_4_g_1'] == 0, base_dados['des_cep_resid__S__p_4_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['des_cep_resid__pe_4_g_1'] == 0, base_dados['des_cep_resid__S__p_4_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['des_cep_resid__pe_4_g_1'] == 1, base_dados['des_cep_resid__S__p_4_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['des_cep_resid__pe_4_g_1'] == 1, base_dados['des_cep_resid__S__p_4_g_1'] == 1), 2,
 0))))
         
base_dados['des_cep_resid__S__p_4_g_1_c1_12_2'] = np.where(base_dados['des_cep_resid__S__p_4_g_1_c1_12_1'] == 0, 0,
np.where(base_dados['des_cep_resid__S__p_4_g_1_c1_12_1'] == 1, 1,
np.where(base_dados['des_cep_resid__S__p_4_g_1_c1_12_1'] == 2, 2,
 0)))
         
base_dados['des_cep_resid__S__p_4_g_1_c1_12'] = np.where(base_dados['des_cep_resid__S__p_4_g_1_c1_12_2'] == 0, 0,
np.where(base_dados['des_cep_resid__S__p_4_g_1_c1_12_2'] == 1, 1,
np.where(base_dados['des_cep_resid__S__p_4_g_1_c1_12_2'] == 2, 2,
 0)))
         
         
         
         
                
base_dados['mob_cliente__L__pe_4_g_1_c1_8_1'] = np.where(np.bitwise_and(base_dados['mob_cliente__L__pe_4_g_1'] == 0, base_dados['mob_cliente__S__p_10_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['mob_cliente__L__pe_4_g_1'] == 0, base_dados['mob_cliente__S__p_10_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['mob_cliente__L__pe_4_g_1'] == 1, base_dados['mob_cliente__S__p_10_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['mob_cliente__L__pe_4_g_1'] == 1, base_dados['mob_cliente__S__p_10_g_1'] == 1), 1,
 0))))

base_dados['mob_cliente__L__pe_4_g_1_c1_8_2'] = np.where(base_dados['mob_cliente__L__pe_4_g_1_c1_8_1'] == 0, 0,
np.where(base_dados['mob_cliente__L__pe_4_g_1_c1_8_1'] == 1, 1,
 0))

base_dados['mob_cliente__L__pe_4_g_1_c1_8'] = np.where(base_dados['mob_cliente__L__pe_4_g_1_c1_8_2'] == 0, 0,
np.where(base_dados['mob_cliente__L__pe_4_g_1_c1_8_2'] == 1, 1,
 0))
                                                       
                                                       
                                                       
                                                       
                                                       
                                                       
base_dados['year_cliente__T__p_10_g_1_c1_3_1'] = np.where(np.bitwise_and(base_dados['year_cliente__T__p_10_g_1'] == 0, base_dados['year_cliente__T__pe_3_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['year_cliente__T__p_10_g_1'] == 0, base_dados['year_cliente__T__pe_3_g_1'] == 1), 0,
np.where(np.bitwise_and(base_dados['year_cliente__T__p_10_g_1'] == 1, base_dados['year_cliente__T__pe_3_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['year_cliente__T__p_10_g_1'] == 1, base_dados['year_cliente__T__pe_3_g_1'] == 1), 1,
 0))))

base_dados['year_cliente__T__p_10_g_1_c1_3_2'] = np.where(base_dados['year_cliente__T__p_10_g_1_c1_3_1'] == 0, 0,
np.where(base_dados['year_cliente__T__p_10_g_1_c1_3_1'] == 1, 1,
 0))

base_dados['year_cliente__T__p_10_g_1_c1_3'] = np.where(base_dados['year_cliente__T__p_10_g_1_c1_3_2'] == 0, 0,
np.where(base_dados['year_cliente__T__p_10_g_1_c1_3_2'] == 1, 1,
 0))

         
         
               
base_dados['mob_expir_prazo__L__p_13_g_1_c1_18_1'] = np.where(np.bitwise_and(base_dados['mob_expir_prazo__p_2_g_1'] == 0, base_dados['mob_expir_prazo__L__p_13_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['mob_expir_prazo__p_2_g_1'] == 0, base_dados['mob_expir_prazo__L__p_13_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['mob_expir_prazo__p_2_g_1'] == 1, base_dados['mob_expir_prazo__L__p_13_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['mob_expir_prazo__p_2_g_1'] == 1, base_dados['mob_expir_prazo__L__p_13_g_1'] == 1), 2,
 0))))

base_dados['mob_expir_prazo__L__p_13_g_1_c1_18_2'] = np.where(base_dados['mob_expir_prazo__L__p_13_g_1_c1_18_1'] == 0, 0,
np.where(base_dados['mob_expir_prazo__L__p_13_g_1_c1_18_1'] == 1, 1,
np.where(base_dados['mob_expir_prazo__L__p_13_g_1_c1_18_1'] == 2, 2,
 0)))

base_dados['mob_expir_prazo__L__p_13_g_1_c1_18'] = np.where(base_dados['mob_expir_prazo__L__p_13_g_1_c1_18_2'] == 0, 0,
np.where(base_dados['mob_expir_prazo__L__p_13_g_1_c1_18_2'] == 1, 1,
np.where(base_dados['mob_expir_prazo__L__p_13_g_1_c1_18_2'] == 2, 2,
 0)))
         
         

# COMMAND ----------

# MAGIC %md
# MAGIC ## Mantendo apenas as variáveis do modelo

# COMMAND ----------

import pickle
modelo=pickle.load(open(caminho_base + 'model_fit_fort_brasil.sav', 'rb'))

base_teste_c0 = base_dados[[chave,'ind_estad_resid_gh38', 'mob_expir_prazo__L__p_13_g_1_c1_18', 'mob_cliente__L__pe_4_g_1_c1_8', 'year_cliente__T__p_10_g_1_c1_3', 'des_cep_resid__S__p_4_g_1_c1_12']]

var_fin_c0=['ind_estad_resid_gh38', 'mob_expir_prazo__L__p_13_g_1_c1_18', 'mob_cliente__L__pe_4_g_1_c1_8', 'year_cliente__T__p_10_g_1_c1_3', 'des_cep_resid__S__p_4_g_1_c1_12']

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



x_teste2['P_1_p_7_g_1'] = np.where(x_teste2['P_1'] <= 0.241467231, 0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.241467231, x_teste2['P_1'] <= 0.271693918), 1,
    np.where(x_teste2['P_1'] > 0.271693918,2,0)))

x_teste2['P_1_p_8_g_1'] = np.where(x_teste2['P_1'] <= 0.263975567, 0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.263975567, x_teste2['P_1'] <= 0.271693918), 1,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.271693918, x_teste2['P_1'] <= 0.354025866), 2,
    np.where(x_teste2['P_1'] > 0.354025866,3,0))))

x_teste2['GH'] = np.where(np.bitwise_and(x_teste2['P_1_p_7_g_1'] == 0, x_teste2['P_1_p_8_g_1'] == 0), 0,
    np.where(np.bitwise_and(x_teste2['P_1_p_7_g_1'] == 1, x_teste2['P_1_p_8_g_1'] == 0), 1,
    np.where(np.bitwise_and(x_teste2['P_1_p_7_g_1'] == 1, x_teste2['P_1_p_8_g_1'] == 1), 1,
    np.where(np.bitwise_and(x_teste2['P_1_p_7_g_1'] == 2, x_teste2['P_1_p_8_g_1'] == 2), 2,
    np.where(np.bitwise_and(x_teste2['P_1_p_7_g_1'] == 2, x_teste2['P_1_p_8_g_1'] == 3), 3,
             1)))))

del x_teste2['P_1_p_7_g_1']
del x_teste2['P_1_p_8_g_1']

x_teste2


# COMMAND ----------

