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
N_Base = "df_aleatorio_tribanco.csv"

#Caminho da base de dados
caminho_base = "Base_Dados_Ferramenta/Tribanco/"

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
base_dados = base_dados[[chave,'ind_sexo','val_compr','des_contr','des_cep_resid','des_fones_celul','cod_produt','ind_estad_civil','cod_credor','des_fones_refer','des_fones_resid','dat_nasci','cod_profi_clien','dat_cadas_clien']]

base_dados['var_tmp'] = np.where(base_dados['ind_sexo'] == 'M', 1, 0)
base_dados['des_cep_resid'] = base_dados['des_cep_resid'].replace(np.nan, 0)
base_dados['des_fones_celul'] = base_dados['des_fones_celul'].replace(np.nan, '-3')
base_dados['des_fones_refer'] = base_dados['des_fones_refer'].replace(np.nan, '-3')
base_dados['des_fones_resid'] = base_dados['des_fones_resid'].replace(np.nan, '-3')
base_dados['ind_estad_civil'] = base_dados['ind_estad_civil'].replace(np.nan, '-3')
base_dados['cod_profi_clien'] = base_dados['cod_profi_clien'].replace(np.nan, '-3')

base_dados = base_dados.apply(pd.to_numeric, errors='ignore')

base_dados['dat_nasci'] = pd.to_datetime(base_dados['dat_nasci'])
base_dados['dat_cadas_clien'] = pd.to_datetime(base_dados['dat_cadas_clien'])

base_dados['DOCUMENTO'] = base_dados['DOCUMENTO'].astype(np.int64)
base_dados['des_contr'] = base_dados['des_contr'].astype(np.int64)
base_dados['des_cep_resid'] = base_dados['des_cep_resid'].astype(np.int64)
base_dados['cod_produt'] = base_dados['cod_produt'].astype(int)
base_dados['cod_credor'] = base_dados['cod_credor'].astype(int)
base_dados['des_fones_celul'] = base_dados['des_fones_celul'].astype(np.int64)
base_dados['des_fones_refer'] = base_dados['des_fones_refer'].astype(np.int64)
base_dados['des_fones_resid'] = base_dados['des_fones_resid'].astype(np.int64)
base_dados['cod_profi_clien'] = base_dados['cod_profi_clien'].astype(int)
base_dados['cod_credor'] = base_dados['cod_credor'].astype(int)
base_dados['val_compr'] = base_dados['val_compr'].astype(float)

base_dados['idade'] = ((datetime.today()) - base_dados.dat_nasci)/np.timedelta64(1, 'Y')
base_dados['mob_cliente'] = ((datetime.today()) - base_dados.dat_cadas_clien)/np.timedelta64(1, 'M')

del base_dados['dat_nasci']
del base_dados['dat_cadas_clien']
del base_dados['ind_sexo']
               
base_dados

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gerando variáveis Categóricas

# COMMAND ----------


         
base_dados['ind_estad_civil_gh30'] = np.where(base_dados['ind_estad_civil'] == '-3', 0,
    np.where(base_dados['ind_estad_civil'] == 'C', 1,
    np.where(base_dados['ind_estad_civil'] == 'O', 2,
    np.where(base_dados['ind_estad_civil'] == 'S', 3,
    np.where(base_dados['ind_estad_civil'] == 'V', 4,
    0)))))

base_dados['ind_estad_civil_gh31'] = np.where(base_dados['ind_estad_civil_gh30'] == 0, 0,
    np.where(base_dados['ind_estad_civil_gh30'] == 1, 1,
    np.where(base_dados['ind_estad_civil_gh30'] == 2, 2,
    np.where(base_dados['ind_estad_civil_gh30'] == 3, 3,
    np.where(base_dados['ind_estad_civil_gh30'] == 4, 4,
    0)))))

base_dados['ind_estad_civil_gh32'] = np.where(base_dados['ind_estad_civil_gh31'] == 0, 0,
    np.where(base_dados['ind_estad_civil_gh31'] == 1, 1,
    np.where(base_dados['ind_estad_civil_gh31'] == 2, 2,
    np.where(base_dados['ind_estad_civil_gh31'] == 3, 3,
    np.where(base_dados['ind_estad_civil_gh31'] == 4, 4,
    0)))))

base_dados['ind_estad_civil_gh33'] = np.where(base_dados['ind_estad_civil_gh32'] == 0, 0,
    np.where(base_dados['ind_estad_civil_gh32'] == 1, 1,
    np.where(base_dados['ind_estad_civil_gh32'] == 2, 2,
    np.where(base_dados['ind_estad_civil_gh32'] == 3, 3,
    np.where(base_dados['ind_estad_civil_gh32'] == 4, 4,
    0)))))

base_dados['ind_estad_civil_gh34'] = np.where(base_dados['ind_estad_civil_gh33'] == 0, 1,
    np.where(base_dados['ind_estad_civil_gh33'] == 1, 1,
    np.where(base_dados['ind_estad_civil_gh33'] == 2, 2,
    np.where(base_dados['ind_estad_civil_gh33'] == 3, 3,
    np.where(base_dados['ind_estad_civil_gh33'] == 4, 5,
    1)))))
base_dados['ind_estad_civil_gh35'] = np.where(base_dados['ind_estad_civil_gh34'] == 1, 0,
    np.where(base_dados['ind_estad_civil_gh34'] == 2, 1,
    np.where(base_dados['ind_estad_civil_gh34'] == 3, 2,
    np.where(base_dados['ind_estad_civil_gh34'] == 5, 3,
    0))))

base_dados['ind_estad_civil_gh36'] = np.where(base_dados['ind_estad_civil_gh35'] == 0, 1,
    np.where(base_dados['ind_estad_civil_gh35'] == 1, 3,
    np.where(base_dados['ind_estad_civil_gh35'] == 2, 1,
    np.where(base_dados['ind_estad_civil_gh35'] == 3, 0,
    0))))

base_dados['ind_estad_civil_gh37'] = np.where(base_dados['ind_estad_civil_gh36'] == 0, 1,
    np.where(base_dados['ind_estad_civil_gh36'] == 1, 1,
    np.where(base_dados['ind_estad_civil_gh36'] == 3, 2,
    1)))
base_dados['ind_estad_civil_gh38'] = np.where(base_dados['ind_estad_civil_gh37'] == 1, 0,
    np.where(base_dados['ind_estad_civil_gh37'] == 2, 1,
    0))
                                              
                                              
                                              
                                              
                                              
                                              
                                              
                                              
base_dados['cod_credor_gh30'] = np.where(base_dados['cod_credor'] == 1, 0,
    np.where(base_dados['cod_credor'] == 2, 1,
    0))
base_dados['cod_credor_gh31'] = np.where(base_dados['cod_credor_gh30'] == 0, 0,
    np.where(base_dados['cod_credor_gh30'] == 1, 1,
    0))
base_dados['cod_credor_gh32'] = np.where(base_dados['cod_credor_gh31'] == 0, 0,
    np.where(base_dados['cod_credor_gh31'] == 1, 1,
    0))
base_dados['cod_credor_gh33'] = np.where(base_dados['cod_credor_gh32'] == 0, 0,
    np.where(base_dados['cod_credor_gh32'] == 1, 1,
    0))
base_dados['cod_credor_gh34'] = np.where(base_dados['cod_credor_gh33'] == 0, 0,
    np.where(base_dados['cod_credor_gh33'] == 1, 1,
    0))
base_dados['cod_credor_gh35'] = np.where(base_dados['cod_credor_gh34'] == 0, 0,
    np.where(base_dados['cod_credor_gh34'] == 1, 1,
    0))
base_dados['cod_credor_gh36'] = np.where(base_dados['cod_credor_gh35'] == 0, 0,
    np.where(base_dados['cod_credor_gh35'] == 1, 1,
    0))
base_dados['cod_credor_gh37'] = np.where(base_dados['cod_credor_gh36'] == 0, 0,
    np.where(base_dados['cod_credor_gh36'] == 1, 1,
    0))
base_dados['cod_credor_gh38'] = np.where(base_dados['cod_credor_gh37'] == 0, 0,
    np.where(base_dados['cod_credor_gh37'] == 1, 1,
    0))
                                         
                                         
                                         
                                         
                                         
                                         
                                         
base_dados['cod_produt_gh30'] = np.where(base_dados['cod_produt'] == 201001, 0,
np.where(base_dados['cod_produt'] == 201002, 1,
np.where(base_dados['cod_produt'] == 201003, 2,
np.where(base_dados['cod_produt'] == 201006, 3,
np.where(base_dados['cod_produt'] == 201013, 4,
np.where(base_dados['cod_produt'] == 203008, 5,
np.where(base_dados['cod_produt'] == 601001, 6,
np.where(base_dados['cod_produt'] == 601002, 7,
np.where(base_dados['cod_produt'] == 601003, 8,
np.where(base_dados['cod_produt'] == 601004, 9,
np.where(base_dados['cod_produt'] == 601005, 10,
np.where(base_dados['cod_produt'] == 601006, 11,
np.where(base_dados['cod_produt'] == 601013, 12,
0)))))))))))))

base_dados['cod_produt_gh31'] = np.where(base_dados['cod_produt_gh30'] == 0, 0,
np.where(base_dados['cod_produt_gh30'] == 1, 1,
np.where(base_dados['cod_produt_gh30'] == 2, 2,
np.where(base_dados['cod_produt_gh30'] == 3, 3,
np.where(base_dados['cod_produt_gh30'] == 4, 3,
np.where(base_dados['cod_produt_gh30'] == 5, 5,
np.where(base_dados['cod_produt_gh30'] == 6, 6,
np.where(base_dados['cod_produt_gh30'] == 7, 7,
np.where(base_dados['cod_produt_gh30'] == 8, 8,
np.where(base_dados['cod_produt_gh30'] == 9, 9,
np.where(base_dados['cod_produt_gh30'] == 10, 10,
np.where(base_dados['cod_produt_gh30'] == 11, 11,
np.where(base_dados['cod_produt_gh30'] == 12, 12,
0)))))))))))))

base_dados['cod_produt_gh32'] = np.where(base_dados['cod_produt_gh31'] == 0, 0,
np.where(base_dados['cod_produt_gh31'] == 1, 1,
np.where(base_dados['cod_produt_gh31'] == 2, 2,
np.where(base_dados['cod_produt_gh31'] == 3, 3,
np.where(base_dados['cod_produt_gh31'] == 5, 4,
np.where(base_dados['cod_produt_gh31'] == 6, 5,
np.where(base_dados['cod_produt_gh31'] == 7, 6,
np.where(base_dados['cod_produt_gh31'] == 8, 7,
np.where(base_dados['cod_produt_gh31'] == 9, 8,
np.where(base_dados['cod_produt_gh31'] == 10, 9,
np.where(base_dados['cod_produt_gh31'] == 11, 10,
np.where(base_dados['cod_produt_gh31'] == 12, 11,
0))))))))))))

base_dados['cod_produt_gh33'] = np.where(base_dados['cod_produt_gh32'] == 0, 0,
np.where(base_dados['cod_produt_gh32'] == 1, 1,
np.where(base_dados['cod_produt_gh32'] == 2, 2,
np.where(base_dados['cod_produt_gh32'] == 3, 3,
np.where(base_dados['cod_produt_gh32'] == 4, 4,
np.where(base_dados['cod_produt_gh32'] == 5, 5,
np.where(base_dados['cod_produt_gh32'] == 6, 6,
np.where(base_dados['cod_produt_gh32'] == 7, 7,
np.where(base_dados['cod_produt_gh32'] == 8, 8,
np.where(base_dados['cod_produt_gh32'] == 9, 9,
np.where(base_dados['cod_produt_gh32'] == 10, 10,
np.where(base_dados['cod_produt_gh32'] == 11, 11,
0))))))))))))

base_dados['cod_produt_gh34'] = np.where(base_dados['cod_produt_gh33'] == 0, 0,
np.where(base_dados['cod_produt_gh33'] == 1, 5,
np.where(base_dados['cod_produt_gh33'] == 2, 5,
np.where(base_dados['cod_produt_gh33'] == 3, 3,
np.where(base_dados['cod_produt_gh33'] == 4, 4,
np.where(base_dados['cod_produt_gh33'] == 5, 5,
np.where(base_dados['cod_produt_gh33'] == 6, 6,
np.where(base_dados['cod_produt_gh33'] == 7, 4,
np.where(base_dados['cod_produt_gh33'] == 8, 4,
np.where(base_dados['cod_produt_gh33'] == 9, 4,
np.where(base_dados['cod_produt_gh33'] == 10, 0,
np.where(base_dados['cod_produt_gh33'] == 11, 5,
0))))))))))))

base_dados['cod_produt_gh35'] = np.where(base_dados['cod_produt_gh34'] == 0, 0,
np.where(base_dados['cod_produt_gh34'] == 3, 1,
np.where(base_dados['cod_produt_gh34'] == 4, 2,
np.where(base_dados['cod_produt_gh34'] == 5, 3,
np.where(base_dados['cod_produt_gh34'] == 6, 4,
0)))))

base_dados['cod_produt_gh36'] = np.where(base_dados['cod_produt_gh35'] == 0, 3,
np.where(base_dados['cod_produt_gh35'] == 1, 4,
np.where(base_dados['cod_produt_gh35'] == 2, 0,
np.where(base_dados['cod_produt_gh35'] == 3, 2,
np.where(base_dados['cod_produt_gh35'] == 4, 1,
0)))))

base_dados['cod_produt_gh37'] = np.where(base_dados['cod_produt_gh36'] == 0, 0,
np.where(base_dados['cod_produt_gh36'] == 1, 0,
np.where(base_dados['cod_produt_gh36'] == 2, 2,
np.where(base_dados['cod_produt_gh36'] == 3, 3,
np.where(base_dados['cod_produt_gh36'] == 4, 3,
0)))))

base_dados['cod_produt_gh38'] = np.where(base_dados['cod_produt_gh37'] == 0, 0,
np.where(base_dados['cod_produt_gh37'] == 2, 1,
np.where(base_dados['cod_produt_gh37'] == 3, 2,
0)))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Gerando variáveis numéricas contínuas

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parte 1 de 2

# COMMAND ----------



base_dados['des_cep_resid__pe_4'] = np.where(base_dados['des_cep_resid'] == 0 , -1.0,
np.where(np.bitwise_and(base_dados['des_cep_resid'] > 0.0, base_dados['des_cep_resid'] <= 22783117.0), 0.0,
np.where(np.bitwise_and(base_dados['des_cep_resid'] > 22783117.0, base_dados['des_cep_resid'] <= 45700000.0), 1.0,
np.where(np.bitwise_and(base_dados['des_cep_resid'] > 45700000.0, base_dados['des_cep_resid'] <= 68557800.0), 2.0,
np.where(base_dados['des_cep_resid'] > 68557800.0, 3.0,
 -2)))))
base_dados['des_cep_resid__pe_4_g_1_1'] = np.where(base_dados['des_cep_resid__pe_4'] == -2.0, 1,
np.where(base_dados['des_cep_resid__pe_4'] == -1.0, 1,
np.where(base_dados['des_cep_resid__pe_4'] == 0.0, 1,
np.where(base_dados['des_cep_resid__pe_4'] == 1.0, 0,
np.where(base_dados['des_cep_resid__pe_4'] == 2.0, 0,
np.where(base_dados['des_cep_resid__pe_4'] == 3.0, 1,
 0))))))
base_dados['des_cep_resid__pe_4_g_1_2'] = np.where(base_dados['des_cep_resid__pe_4_g_1_1'] == 0, 0,
np.where(base_dados['des_cep_resid__pe_4_g_1_1'] == 1, 1,
 0))
base_dados['des_cep_resid__pe_4_g_1'] = np.where(base_dados['des_cep_resid__pe_4_g_1_2'] == 0, 0,
np.where(base_dados['des_cep_resid__pe_4_g_1_2'] == 1, 1,
 0))
                                                 
                                                 
                                                 
                                                 
                                                 
                                                 
                                                 
base_dados['des_cep_resid__L'] = np.log(base_dados['des_cep_resid'])
np.where(base_dados['des_cep_resid__L'] == 0, -1, base_dados['des_cep_resid__L'])
base_dados['des_cep_resid__L'] = base_dados['des_cep_resid__L'].fillna(-2)
base_dados['des_cep_resid__L__p_34'] = np.where(np.bitwise_and(base_dados['des_cep_resid__L'] >= -2.0, base_dados['des_cep_resid__L'] <= 15.710976675965972), 0.0,
np.where(np.bitwise_and(base_dados['des_cep_resid__L'] > 15.710976675965972, base_dados['des_cep_resid__L'] <= 15.952457980888045), 1.0,
np.where(np.bitwise_and(base_dados['des_cep_resid__L'] > 15.952457980888045, base_dados['des_cep_resid__L'] <= 16.42854741520201), 2.0,
np.where(np.bitwise_and(base_dados['des_cep_resid__L'] > 16.42854741520201, base_dados['des_cep_resid__L'] <= 16.67083067780152), 3.0,
np.where(np.bitwise_and(base_dados['des_cep_resid__L'] > 16.67083067780152, base_dados['des_cep_resid__L'] <= 17.030378361434938), 4.0,
np.where(base_dados['des_cep_resid__L'] > 17.030378361434938, 6.0,
 -2))))))
base_dados['des_cep_resid__L__p_34_g_1_1'] = np.where(base_dados['des_cep_resid__L__p_34'] == -2.0, 0,
np.where(base_dados['des_cep_resid__L__p_34'] == 0.0, 1,
np.where(base_dados['des_cep_resid__L__p_34'] == 1.0, 1,
np.where(base_dados['des_cep_resid__L__p_34'] == 2.0, 1,
np.where(base_dados['des_cep_resid__L__p_34'] == 3.0, 0,
np.where(base_dados['des_cep_resid__L__p_34'] == 4.0, 1,
np.where(base_dados['des_cep_resid__L__p_34'] == 6.0, 0,
 0)))))))
base_dados['des_cep_resid__L__p_34_g_1_2'] = np.where(base_dados['des_cep_resid__L__p_34_g_1_1'] == 0, 0,
np.where(base_dados['des_cep_resid__L__p_34_g_1_1'] == 1, 1,
 0))
base_dados['des_cep_resid__L__p_34_g_1'] = np.where(base_dados['des_cep_resid__L__p_34_g_1_2'] == 0, 0,
np.where(base_dados['des_cep_resid__L__p_34_g_1_2'] == 1, 1,
 0))
                                                    
                                                    
                                                    
                                                    
                                                    
                                                    
base_dados['des_contr__p_5'] = np.where(base_dados['des_contr'] <= 5076417255344000.0, 0.0,
np.where(np.bitwise_and(base_dados['des_contr'] > 5076417255344000.0, base_dados['des_contr'] <= 5182770860425000.0), 1.0,
np.where(base_dados['des_contr'] > 5182770860425000.0, 2.0,
 -2)))
base_dados['des_contr__p_5_g_1_1'] = np.where(base_dados['des_contr__p_5'] == -2.0, 0,
np.where(base_dados['des_contr__p_5'] == 0.0, 2,
np.where(base_dados['des_contr__p_5'] == 1.0, 1,
np.where(base_dados['des_contr__p_5'] == 2.0, 3,
 0))))
base_dados['des_contr__p_5_g_1_2'] = np.where(base_dados['des_contr__p_5_g_1_1'] == 0, 0,
np.where(base_dados['des_contr__p_5_g_1_1'] == 1, 3,
np.where(base_dados['des_contr__p_5_g_1_1'] == 2, 2,
np.where(base_dados['des_contr__p_5_g_1_1'] == 3, 0,
0))))
base_dados['des_contr__p_5_g_1'] = np.where(base_dados['des_contr__p_5_g_1_2'] == 0, 0,
np.where(base_dados['des_contr__p_5_g_1_2'] == 2, 1,
np.where(base_dados['des_contr__p_5_g_1_2'] == 3, 2,
 0)))
         
         
         
         
                
base_dados['des_contr__p_4'] = np.where(base_dados['des_contr'] <= 5076417377777000.0, 0.0,
np.where(np.bitwise_and(base_dados['des_contr'] > 5076417377777000.0, base_dados['des_contr'] <= 6363750762277000.0), 1.0,
 -2))
base_dados['des_contr__p_4_g_1_1'] = np.where(base_dados['des_contr__p_4'] == -2.0, 0,
np.where(base_dados['des_contr__p_4'] == 0.0, 1,
np.where(base_dados['des_contr__p_4'] == 1.0, 1,
 0)))
base_dados['des_contr__p_4_g_1_2'] = np.where(base_dados['des_contr__p_4_g_1_1'] == 0, 0,
np.where(base_dados['des_contr__p_4_g_1_1'] == 1, 1,
 0))
base_dados['des_contr__p_4_g_1'] = np.where(base_dados['des_contr__p_4_g_1_2'] == 0, 0,
np.where(base_dados['des_contr__p_4_g_1_2'] == 1, 1,
 0))
                                            
                                            
                                            
                                            
                                            
                                            
base_dados['des_fones_resid__S'] = np.sin(base_dados['des_fones_resid'])
np.where(base_dados['des_fones_resid__S'] == 0, -1, base_dados['des_fones_resid__S'])
base_dados['des_fones_resid__S'] = base_dados['des_fones_resid__S'].fillna(-2)
base_dados['des_fones_resid__S__p_13'] = np.where(np.bitwise_and(base_dados['des_fones_resid__S'] >= -0.9999996041594625, base_dados['des_fones_resid__S'] <= 0.11885700682493316), 10.0,
np.where(np.bitwise_and(base_dados['des_fones_resid__S'] > 0.11885700682493316, base_dados['des_fones_resid__S'] <= 0.8871788210937707), 11.0,
np.where(base_dados['des_fones_resid__S'] > 0.8871788210937707, 12.0,
 -2)))
base_dados['des_fones_resid__S__p_13_g_1_1'] = np.where(base_dados['des_fones_resid__S__p_13'] == -2.0, 0,
np.where(base_dados['des_fones_resid__S__p_13'] == 10.0, 1,
np.where(base_dados['des_fones_resid__S__p_13'] == 11.0, 1,
np.where(base_dados['des_fones_resid__S__p_13'] == 12.0, 1,
 0))))
base_dados['des_fones_resid__S__p_13_g_1_2'] = np.where(base_dados['des_fones_resid__S__p_13_g_1_1'] == 0, 1,
np.where(base_dados['des_fones_resid__S__p_13_g_1_1'] == 1, 0,
 0))
base_dados['des_fones_resid__S__p_13_g_1'] = np.where(base_dados['des_fones_resid__S__p_13_g_1_2'] == 0, 0,
np.where(base_dados['des_fones_resid__S__p_13_g_1_2'] == 1, 1,
 0))
                                                      
                                                      
                                                      
                                                      
                                                      
                                                      
                                                      
base_dados['des_fones_resid__T'] = np.tan(base_dados['des_fones_resid'])
np.where(base_dados['des_fones_resid__T'] == 0, -1, base_dados['des_fones_resid__T'])
base_dados['des_fones_resid__T'] = base_dados['des_fones_resid__T'].fillna(-2)
base_dados['des_fones_resid__T__p_13'] = np.where(np.bitwise_and(base_dados['des_fones_resid__T'] >= -1123.892412893752, base_dados['des_fones_resid__T'] <= 0.1425465430742778), 1.0,
np.where(np.bitwise_and(base_dados['des_fones_resid__T'] > 0.1425465430742778, base_dados['des_fones_resid__T'] <= 0.17034415206186926), 10.0,
np.where(np.bitwise_and(base_dados['des_fones_resid__T'] > 0.17034415206186926, base_dados['des_fones_resid__T'] <= 2.1060712622030913), 11.0,
np.where(base_dados['des_fones_resid__T'] > 2.1060712622030913, 12.0,
 -2))))
base_dados['des_fones_resid__T__p_13_g_1_1'] = np.where(base_dados['des_fones_resid__T__p_13'] == -2.0, 1,
np.where(base_dados['des_fones_resid__T__p_13'] == 1.0, 0,
np.where(base_dados['des_fones_resid__T__p_13'] == 10.0, 1,
np.where(base_dados['des_fones_resid__T__p_13'] == 11.0, 1,
np.where(base_dados['des_fones_resid__T__p_13'] == 12.0, 1,
 0)))))
base_dados['des_fones_resid__T__p_13_g_1_2'] = np.where(base_dados['des_fones_resid__T__p_13_g_1_1'] == 0, 1,
np.where(base_dados['des_fones_resid__T__p_13_g_1_1'] == 1, 0,
0))
base_dados['des_fones_resid__T__p_13_g_1'] = np.where(base_dados['des_fones_resid__T__p_13_g_1_2'] == 0, 0,
np.where(base_dados['des_fones_resid__T__p_13_g_1_2'] == 1, 1,
0))
                                                      
                                                      
                                                      
                                                      
                                                      
                                                      
base_dados['mob_cliente__p_7'] = np.where(np.bitwise_and(base_dados['mob_cliente'] >= -1.1475735386774997, base_dados['mob_cliente'] <= 14.984174546498787), 0.0,
np.where(np.bitwise_and(base_dados['mob_cliente'] > 14.984174546498787, base_dados['mob_cliente'] <= 27.928998875540447), 1.0,
np.where(np.bitwise_and(base_dados['mob_cliente'] > 27.928998875540447, base_dados['mob_cliente'] <= 46.45915349883867), 2.0,
np.where(np.bitwise_and(base_dados['mob_cliente'] > 46.45915349883867, base_dados['mob_cliente'] <= 87.26491953099539), 3.0,
np.where(np.bitwise_and(base_dados['mob_cliente'] > 87.26491953099539, base_dados['mob_cliente'] <= 113.2531328413303), 4.0,
np.where(np.bitwise_and(base_dados['mob_cliente'] > 113.2531328413303, base_dados['mob_cliente'] <= 158.95427660198246), 5.0,
np.where(base_dados['mob_cliente'] > 158.95427660198246, 6.0,
 -2)))))))
base_dados['mob_cliente__p_7_g_1_1'] = np.where(base_dados['mob_cliente__p_7'] == -2.0, 6,
np.where(base_dados['mob_cliente__p_7'] == 0.0, 4,
np.where(base_dados['mob_cliente__p_7'] == 1.0, 1,
np.where(base_dados['mob_cliente__p_7'] == 2.0, 3,
np.where(base_dados['mob_cliente__p_7'] == 3.0, 0,
np.where(base_dados['mob_cliente__p_7'] == 4.0, 6,
np.where(base_dados['mob_cliente__p_7'] == 5.0, 2,
np.where(base_dados['mob_cliente__p_7'] == 6.0, 5,
 0))))))))
base_dados['mob_cliente__p_7_g_1_2'] = np.where(base_dados['mob_cliente__p_7_g_1_1'] == 0, 3,
np.where(base_dados['mob_cliente__p_7_g_1_1'] == 1, 5,
np.where(base_dados['mob_cliente__p_7_g_1_1'] == 2, 1,
np.where(base_dados['mob_cliente__p_7_g_1_1'] == 3, 4,
np.where(base_dados['mob_cliente__p_7_g_1_1'] == 4, 6,
np.where(base_dados['mob_cliente__p_7_g_1_1'] == 5, 0,
np.where(base_dados['mob_cliente__p_7_g_1_1'] == 6, 2,
 0)))))))
base_dados['mob_cliente__p_7_g_1'] = np.where(base_dados['mob_cliente__p_7_g_1_2'] == 0, 0,
np.where(base_dados['mob_cliente__p_7_g_1_2'] == 1, 1,
np.where(base_dados['mob_cliente__p_7_g_1_2'] == 2, 2,
np.where(base_dados['mob_cliente__p_7_g_1_2'] == 3, 3,
np.where(base_dados['mob_cliente__p_7_g_1_2'] == 4, 4,
np.where(base_dados['mob_cliente__p_7_g_1_2'] == 5, 5,
np.where(base_dados['mob_cliente__p_7_g_1_2'] == 6, 6,
 0)))))))
         
         
         
         
         
         
         
base_dados['mob_cliente__L'] = np.log(base_dados['mob_cliente'])
np.where(base_dados['mob_cliente__L'] == 0, -1, base_dados['mob_cliente__L'])
base_dados['mob_cliente__L'] = base_dados['mob_cliente__L'].fillna(-2)
base_dados['mob_cliente__L__pe_8'] = np.where(np.bitwise_and(base_dados['mob_cliente__L'] >= -6.054445132517176, base_dados['mob_cliente__L'] <= 0.020634448744025854), 0.0,
np.where(np.bitwise_and(base_dados['mob_cliente__L'] > 0.020634448744025854, base_dados['mob_cliente__L'] <= 1.1069100137873433), 1.0,
np.where(np.bitwise_and(base_dados['mob_cliente__L'] > 1.1069100137873433, base_dados['mob_cliente__L'] <= 1.7322549171242048), 2.0,
np.where(np.bitwise_and(base_dados['mob_cliente__L'] > 1.7322549171242048, base_dados['mob_cliente__L'] <= 2.3146768622104394), 3.0,
np.where(np.bitwise_and(base_dados['mob_cliente__L'] > 2.3146768622104394, base_dados['mob_cliente__L'] <= 2.894393307856653), 4.0,
np.where(np.bitwise_and(base_dados['mob_cliente__L'] > 2.894393307856653, base_dados['mob_cliente__L'] <= 3.47197060802131), 5.0,
np.where(np.bitwise_and(base_dados['mob_cliente__L'] > 3.47197060802131, base_dados['mob_cliente__L'] <= 4.051757026561621), 6.0,
np.where(base_dados['mob_cliente__L'] > 4.051757026561621, 7.0,
 -2))))))))
base_dados['mob_cliente__L__pe_8_g_1_1'] = np.where(base_dados['mob_cliente__L__pe_8'] == -2.0, 1,
np.where(base_dados['mob_cliente__L__pe_8'] == 0.0, 3,
np.where(base_dados['mob_cliente__L__pe_8'] == 1.0, 3,
np.where(base_dados['mob_cliente__L__pe_8'] == 2.0, 3,
np.where(base_dados['mob_cliente__L__pe_8'] == 3.0, 3,
np.where(base_dados['mob_cliente__L__pe_8'] == 4.0, 3,
np.where(base_dados['mob_cliente__L__pe_8'] == 5.0, 0,
np.where(base_dados['mob_cliente__L__pe_8'] == 6.0, 0,
np.where(base_dados['mob_cliente__L__pe_8'] == 7.0, 2,
 0)))))))))
base_dados['mob_cliente__L__pe_8_g_1_2'] = np.where(base_dados['mob_cliente__L__pe_8_g_1_1'] == 0, 2,
np.where(base_dados['mob_cliente__L__pe_8_g_1_1'] == 1, 0,
np.where(base_dados['mob_cliente__L__pe_8_g_1_1'] == 2, 1,
np.where(base_dados['mob_cliente__L__pe_8_g_1_1'] == 3, 3,
 0))))
base_dados['mob_cliente__L__pe_8_g_1'] = np.where(base_dados['mob_cliente__L__pe_8_g_1_2'] == 0, 0,
np.where(base_dados['mob_cliente__L__pe_8_g_1_2'] == 1, 1,
np.where(base_dados['mob_cliente__L__pe_8_g_1_2'] == 2, 2,
np.where(base_dados['mob_cliente__L__pe_8_g_1_2'] == 3, 3,
 0))))
         
         
         
         
         
         
         
base_dados['idade__pe_13'] = np.where(base_dados['idade'] <= 18.99853233848307, 2.0,
np.where(np.bitwise_and(base_dados['idade'] > 18.99853233848307, base_dados['idade'] <= 25.95281613623388), 3.0,
np.where(np.bitwise_and(base_dados['idade'] > 25.95281613623388, base_dados['idade'] <= 32.44439364980363), 4.0,
np.where(np.bitwise_and(base_dados['idade'] > 32.44439364980363, base_dados['idade'] <= 38.93323325636639), 5.0,
np.where(np.bitwise_and(base_dados['idade'] > 38.93323325636639, base_dados['idade'] <= 45.41933495592217), 6.0,
np.where(np.bitwise_and(base_dados['idade'] > 45.41933495592217, base_dados['idade'] <= 51.90817456248493), 7.0,
np.where(np.bitwise_and(base_dados['idade'] > 51.90817456248493, base_dados['idade'] <= 58.397014169047694), 8.0,
np.where(np.bitwise_and(base_dados['idade'] > 58.397014169047694, base_dados['idade'] <= 64.88311586860347), 9.0,
np.where(np.bitwise_and(base_dados['idade'] > 64.88311586860347, base_dados['idade'] <= 71.37195547516623), 10.0,
np.where(np.bitwise_and(base_dados['idade'] > 71.37195547516623, base_dados['idade'] <= 77.62807298613497), 11.0,
np.where(np.bitwise_and(base_dados['idade'] > 77.62807298613497, base_dados['idade'] <= 86.50984331680569), 12.0,
 -2)))))))))))
base_dados['idade__pe_13_g_1_1'] = np.where(base_dados['idade__pe_13'] == -2.0, 1,
np.where(base_dados['idade__pe_13'] == 2.0, 1,
np.where(base_dados['idade__pe_13'] == 3.0, 0,
np.where(base_dados['idade__pe_13'] == 4.0, 0,
np.where(base_dados['idade__pe_13'] == 5.0, 0,
np.where(base_dados['idade__pe_13'] == 6.0, 0,
np.where(base_dados['idade__pe_13'] == 7.0, 0,
np.where(base_dados['idade__pe_13'] == 8.0, 0,
np.where(base_dados['idade__pe_13'] == 9.0, 0,
np.where(base_dados['idade__pe_13'] == 10.0, 0,
np.where(base_dados['idade__pe_13'] == 11.0, 1,
np.where(base_dados['idade__pe_13'] == 12.0, 1,
 0))))))))))))

base_dados['idade__pe_13_g_1_2'] = np.where(base_dados['idade__pe_13_g_1_1'] == 0, 1,
np.where(base_dados['idade__pe_13_g_1_1'] == 1, 0,
 0))
base_dados['idade__pe_13_g_1'] = np.where(base_dados['idade__pe_13_g_1_2'] == 0, 0,
np.where(base_dados['idade__pe_13_g_1_2'] == 1, 1,
 0))






base_dados['idade__pe_3'] = np.where(base_dados['idade'] <= 28.54013825783802, 0.0,
np.where(np.bitwise_and(base_dados['idade'] > 28.54013825783802, base_dados['idade'] <= 57.09377043372117), 1.0,
np.where(np.bitwise_and(base_dados['idade'] > 57.09377043372117, base_dados['idade'] <= 86.50984331680569), 2.0,
 -2)))
base_dados['idade__pe_3_g_1_1'] = np.where(base_dados['idade__pe_3'] == -2.0, 2,
np.where(base_dados['idade__pe_3'] == 0.0, 2,
np.where(base_dados['idade__pe_3'] == 1.0, 0,
np.where(base_dados['idade__pe_3'] == 2.0, 1,
 0))))
base_dados['idade__pe_3_g_1_2'] = np.where(base_dados['idade__pe_3_g_1_1'] == 0, 1,
np.where(base_dados['idade__pe_3_g_1_1'] == 1, 0,
np.where(base_dados['idade__pe_3_g_1_1'] == 2, 1,
 0)))
base_dados['idade__pe_3_g_1'] = np.where(base_dados['idade__pe_3_g_1_2'] == 0, 0,
np.where(base_dados['idade__pe_3_g_1_2'] == 1, 1,
 0))
                                         
                                         
                                         
                                         
                                         
                                         
base_dados['des_fones_refer__R'] = np.sqrt(base_dados['des_fones_refer'])
np.where(base_dados['des_fones_refer__R'] == 0, -1, base_dados['des_fones_refer__R'])
base_dados['des_fones_refer__R'] = base_dados['des_fones_refer__R'].fillna(-2)
base_dados['des_fones_refer__R__p_10'] = np.where(np.bitwise_and(base_dados['des_fones_refer__R'] >= -2.0, base_dados['des_fones_refer__R'] <= 33790.78750192129), 7.0,
np.where(np.bitwise_and(base_dados['des_fones_refer__R'] > 33790.78750192129, base_dados['des_fones_refer__R'] <= 80850.74088095916), 8.0,
np.where(base_dados['des_fones_refer__R'] > 80850.74088095916, 9.0,
 -2)))
base_dados['des_fones_refer__R__p_10_g_1_1'] = np.where(base_dados['des_fones_refer__R__p_10'] == -2.0, 1,
np.where(base_dados['des_fones_refer__R__p_10'] == 7.0, 2,
np.where(base_dados['des_fones_refer__R__p_10'] == 8.0, 2,
np.where(base_dados['des_fones_refer__R__p_10'] == 9.0, 0,
 0))))
base_dados['des_fones_refer__R__p_10_g_1_2'] = np.where(base_dados['des_fones_refer__R__p_10_g_1_1'] == 0, 0,
np.where(base_dados['des_fones_refer__R__p_10_g_1_1'] == 1, 2,
np.where(base_dados['des_fones_refer__R__p_10_g_1_1'] == 2, 1,
 0)))
base_dados['des_fones_refer__R__p_10_g_1'] = np.where(base_dados['des_fones_refer__R__p_10_g_1_2'] == 0, 0,
np.where(base_dados['des_fones_refer__R__p_10_g_1_2'] == 1, 1,
np.where(base_dados['des_fones_refer__R__p_10_g_1_2'] == 2, 2,
 0)))
         
         
         
         
         
         
         
base_dados['des_fones_refer__T'] = np.tan(base_dados['des_fones_refer'])
np.where(base_dados['des_fones_refer__T'] == 0, -1, base_dados['des_fones_refer__T'])
base_dados['des_fones_refer__T'] = base_dados['des_fones_refer__T'].fillna(-2)
base_dados['des_fones_refer__T__pe_8'] = np.where(np.bitwise_and(base_dados['des_fones_refer__T'] >= -575.0937641192395, base_dados['des_fones_refer__T'] <= 1.8074218227187393), 0.0,
np.where(np.bitwise_and(base_dados['des_fones_refer__T'] > 1.8074218227187393, base_dados['des_fones_refer__T'] <= 3.6196578377984525), 1.0,
np.where(np.bitwise_and(base_dados['des_fones_refer__T'] > 3.6196578377984525, base_dados['des_fones_refer__T'] <= 5.303481019235197), 2.0,
np.where(np.bitwise_and(base_dados['des_fones_refer__T'] > 5.303481019235197, base_dados['des_fones_refer__T'] <= 6.912121311628884), 3.0,
np.where(np.bitwise_and(base_dados['des_fones_refer__T'] > 6.912121311628884, base_dados['des_fones_refer__T'] <= 8.922076443501409), 4.0,
np.where(np.bitwise_and(base_dados['des_fones_refer__T'] > 8.922076443501409, base_dados['des_fones_refer__T'] <= 10.835176831155751), 5.0,
np.where(np.bitwise_and(base_dados['des_fones_refer__T'] > 10.835176831155751, base_dados['des_fones_refer__T'] <= 12.624676803073912), 6.0,
np.where(base_dados['des_fones_refer__T'] > 12.624676803073912, 7.0,
 -2))))))))

base_dados['des_fones_refer__T__pe_8_g_1_1'] = np.where(base_dados['des_fones_refer__T__pe_8'] == -2.0, 1,
np.where(base_dados['des_fones_refer__T__pe_8'] == 0.0, 0,
np.where(base_dados['des_fones_refer__T__pe_8'] == 1.0, 0,
np.where(base_dados['des_fones_refer__T__pe_8'] == 2.0, 1,
np.where(base_dados['des_fones_refer__T__pe_8'] == 3.0, 1,
np.where(base_dados['des_fones_refer__T__pe_8'] == 4.0, 1,
np.where(base_dados['des_fones_refer__T__pe_8'] == 5.0, 1,
np.where(base_dados['des_fones_refer__T__pe_8'] == 6.0, 1,
np.where(base_dados['des_fones_refer__T__pe_8'] == 7.0, 1,
 0)))))))))
base_dados['des_fones_refer__T__pe_8_g_1_2'] = np.where(base_dados['des_fones_refer__T__pe_8_g_1_1'] == 0, 1,
np.where(base_dados['des_fones_refer__T__pe_8_g_1_1'] == 1, 0,
 0))
base_dados['des_fones_refer__T__pe_8_g_1'] = np.where(base_dados['des_fones_refer__T__pe_8_g_1_2'] == 0, 0,
np.where(base_dados['des_fones_refer__T__pe_8_g_1_2'] == 1, 1,
 0))
                                                      
                                                      
                                                      
                                                      
                                                      
                                                      
                                                      
base_dados['des_fones_celul__pe_7'] = np.where(np.bitwise_and(base_dados['des_fones_celul'] >= -3.0, base_dados['des_fones_celul'] <= 13997986479.0), 0.0,
np.where(np.bitwise_and(base_dados['des_fones_celul'] > 13997986479.0, base_dados['des_fones_celul'] <= 27999949311.0), 1.0,
np.where(np.bitwise_and(base_dados['des_fones_celul'] > 27999949311.0, base_dados['des_fones_celul'] <= 41999914893.0), 2.0,
np.where(np.bitwise_and(base_dados['des_fones_celul'] > 41999914893.0, base_dados['des_fones_celul'] <= 55999969856.0), 3.0,
np.where(np.bitwise_and(base_dados['des_fones_celul'] > 55999969856.0, base_dados['des_fones_celul'] <= 69999748924.0), 4.0,
np.where(np.bitwise_and(base_dados['des_fones_celul'] > 69999748924.0, base_dados['des_fones_celul'] <= 83999948405.0), 5.0,
np.where(base_dados['des_fones_celul'] > 83999948405.0, 6.0,
 -2)))))))
base_dados['des_fones_celul__pe_7_g_1_1'] = np.where(base_dados['des_fones_celul__pe_7'] == -2.0, 0,
np.where(base_dados['des_fones_celul__pe_7'] == 0.0, 2,
np.where(base_dados['des_fones_celul__pe_7'] == 1.0, 2,
np.where(base_dados['des_fones_celul__pe_7'] == 2.0, 2,
np.where(base_dados['des_fones_celul__pe_7'] == 3.0, 2,
np.where(base_dados['des_fones_celul__pe_7'] == 4.0, 1,
np.where(base_dados['des_fones_celul__pe_7'] == 5.0, 1,
np.where(base_dados['des_fones_celul__pe_7'] == 6.0, 2,
 0))))))))
base_dados['des_fones_celul__pe_7_g_1_2'] = np.where(base_dados['des_fones_celul__pe_7_g_1_1'] == 0, 0,
np.where(base_dados['des_fones_celul__pe_7_g_1_1'] == 1, 1,
np.where(base_dados['des_fones_celul__pe_7_g_1_1'] == 2, 2,
 0)))
base_dados['des_fones_celul__pe_7_g_1'] = np.where(base_dados['des_fones_celul__pe_7_g_1_2'] == 0, 0,
np.where(base_dados['des_fones_celul__pe_7_g_1_2'] == 1, 1,
np.where(base_dados['des_fones_celul__pe_7_g_1_2'] == 2, 2,
 0)))
         
         
         
         
         
         
         
base_dados['des_fones_celul__L'] = np.log(base_dados['des_fones_celul'])
np.where(base_dados['des_fones_celul__L'] == 0, -1, base_dados['des_fones_celul__L'])
base_dados['des_fones_celul__L'] = base_dados['des_fones_celul__L'].fillna(-2)
base_dados['des_fones_celul__L__pe_15'] = np.where(np.bitwise_and(base_dados['des_fones_celul__L'] >= -2.0, base_dados['des_fones_celul__L'] <= 17.324758045092704), 9.0,
np.where(np.bitwise_and(base_dados['des_fones_celul__L'] > 17.324758045092704, base_dados['des_fones_celul__L'] <= 19.253345380481363), 10.0,
np.where(np.bitwise_and(base_dados['des_fones_celul__L'] > 19.253345380481363, base_dados['des_fones_celul__L'] <= 21.556169448996766), 11.0,
np.where(np.bitwise_and(base_dados['des_fones_celul__L'] > 21.556169448996766, base_dados['des_fones_celul__L'] <= 23.66765717071467), 12.0,
np.where(base_dados['des_fones_celul__L'] > 23.66765717071467, 13.0,
 -2)))))
base_dados['des_fones_celul__L__pe_15_g_1_1'] = np.where(base_dados['des_fones_celul__L__pe_15'] == -2.0, 2,
np.where(base_dados['des_fones_celul__L__pe_15'] == 9.0, 2,
np.where(base_dados['des_fones_celul__L__pe_15'] == 10.0, 2,
np.where(base_dados['des_fones_celul__L__pe_15'] == 11.0, 2,
np.where(base_dados['des_fones_celul__L__pe_15'] == 12.0, 1,
np.where(base_dados['des_fones_celul__L__pe_15'] == 13.0, 0,
 0))))))
base_dados['des_fones_celul__L__pe_15_g_1_2'] = np.where(base_dados['des_fones_celul__L__pe_15_g_1_1'] == 0, 1,
np.where(base_dados['des_fones_celul__L__pe_15_g_1_1'] == 1, 2,
np.where(base_dados['des_fones_celul__L__pe_15_g_1_1'] == 2, 0,
 0)))
base_dados['des_fones_celul__L__pe_15_g_1'] = np.where(base_dados['des_fones_celul__L__pe_15_g_1_2'] == 0, 0,
np.where(base_dados['des_fones_celul__L__pe_15_g_1_2'] == 1, 1,
np.where(base_dados['des_fones_celul__L__pe_15_g_1_2'] == 2, 2,
 0)))
         
         
         
         
                 
base_dados['val_compr__L'] = np.log(base_dados['val_compr'])
np.where(base_dados['val_compr__L'] == 0, -1, base_dados['val_compr__L'])
base_dados['val_compr__L'] = base_dados['val_compr__L'].fillna(-2)
base_dados['val_compr__L__p_6'] = np.where(base_dados['val_compr__L'] == 0 , -1.0,
np.where(np.bitwise_and(base_dados['val_compr__L'] > 0.0, base_dados['val_compr__L'] <= 1.5912739418064292), 1.0,
np.where(np.bitwise_and(base_dados['val_compr__L'] > 1.5912739418064292, base_dados['val_compr__L'] <= 3.2519236789144013), 2.0,
np.where(np.bitwise_and(base_dados['val_compr__L'] > 3.2519236789144013, base_dados['val_compr__L'] <= 4.874815570953721), 3.0,
np.where(np.bitwise_and(base_dados['val_compr__L'] > 4.874815570953721, base_dados['val_compr__L'] <= 5.7183754769591015), 4.0,
np.where(base_dados['val_compr__L'] > 5.7183754769591015, 5.0,
 -2))))))
base_dados['val_compr__L__p_6_g_1_1'] = np.where(base_dados['val_compr__L__p_6'] == -2.0, 3,
np.where(base_dados['val_compr__L__p_6'] == -1.0, 3,
np.where(base_dados['val_compr__L__p_6'] == 1.0, 2,
np.where(base_dados['val_compr__L__p_6'] == 2.0, 1,
np.where(base_dados['val_compr__L__p_6'] == 3.0, 1,
np.where(base_dados['val_compr__L__p_6'] == 4.0, 1,
np.where(base_dados['val_compr__L__p_6'] == 5.0, 0,
 0)))))))
base_dados['val_compr__L__p_6_g_1_2'] = np.where(base_dados['val_compr__L__p_6_g_1_1'] == 0, 3,
np.where(base_dados['val_compr__L__p_6_g_1_1'] == 1, 2,
np.where(base_dados['val_compr__L__p_6_g_1_1'] == 2, 1,
np.where(base_dados['val_compr__L__p_6_g_1_1'] == 3, 0,
 0))))
base_dados['val_compr__L__p_6_g_1'] = np.where(base_dados['val_compr__L__p_6_g_1_2'] == 0, 0,
np.where(base_dados['val_compr__L__p_6_g_1_2'] == 1, 1,
np.where(base_dados['val_compr__L__p_6_g_1_2'] == 2, 2,
np.where(base_dados['val_compr__L__p_6_g_1_2'] == 3, 3,
 0))))
         
         
         
         
                
base_dados['val_compr__T'] = np.tan(base_dados['val_compr'])
np.where(base_dados['val_compr__T'] == 0, -1, base_dados['val_compr__T'])
base_dados['val_compr__T'] = base_dados['val_compr__T'].fillna(-2)
base_dados['val_compr__T__p_5'] = np.where(np.bitwise_and(base_dados['val_compr__T'] >= -2615.112645309714, base_dados['val_compr__T'] <= 0.17165682217014272), 2.0,
np.where(np.bitwise_and(base_dados['val_compr__T'] > 0.17165682217014272, base_dados['val_compr__T'] <= 0.9762853124056229), 3.0,
np.where(base_dados['val_compr__T'] > 0.9762853124056229, 4.0,
 -2)))
base_dados['val_compr__T__p_5_g_1_1'] = np.where(base_dados['val_compr__T__p_5'] == -2.0, 0,
np.where(base_dados['val_compr__T__p_5'] == 2.0, 1,
np.where(base_dados['val_compr__T__p_5'] == 3.0, 0,
np.where(base_dados['val_compr__T__p_5'] == 4.0, 0,
 0))))
base_dados['val_compr__T__p_5_g_1_2'] = np.where(base_dados['val_compr__T__p_5_g_1_1'] == 0, 1,
np.where(base_dados['val_compr__T__p_5_g_1_1'] == 1, 0,
 0))
base_dados['val_compr__T__p_5_g_1'] = np.where(base_dados['val_compr__T__p_5_g_1_2'] == 0, 0,
np.where(base_dados['val_compr__T__p_5_g_1_2'] == 1, 1,
 0))
 
        
         
         
         
         
         
base_dados['cod_profi_clien__p_4'] = np.where(np.bitwise_and(base_dados['cod_profi_clien'] >= -3.0, base_dados['cod_profi_clien'] <= 303.0), 1.0,
np.where(np.bitwise_and(base_dados['cod_profi_clien'] > 303.0, base_dados['cod_profi_clien'] <= 902.0), 2.0,
np.where(base_dados['cod_profi_clien'] > 902.0, 3.0,
 -2)))
base_dados['cod_profi_clien__p_4_g_1_1'] = np.where(base_dados['cod_profi_clien__p_4'] == -2.0, 1,
np.where(base_dados['cod_profi_clien__p_4'] == 1.0, 0,
np.where(base_dados['cod_profi_clien__p_4'] == 2.0, 0,
np.where(base_dados['cod_profi_clien__p_4'] == 3.0, 0,
 0))))
base_dados['cod_profi_clien__p_4_g_1_2'] = np.where(base_dados['cod_profi_clien__p_4_g_1_1'] == 0, 1,
np.where(base_dados['cod_profi_clien__p_4_g_1_1'] == 1, 0,
 0))
base_dados['cod_profi_clien__p_4_g_1'] = np.where(base_dados['cod_profi_clien__p_4_g_1_2'] == 0, 0,
np.where(base_dados['cod_profi_clien__p_4_g_1_2'] == 1, 1,
 0))
                                                  
                                                  
                                                  
                                                  
                                                  
                                                  
base_dados['cod_profi_clien__L'] = np.log(base_dados['cod_profi_clien'])
np.where(base_dados['cod_profi_clien__L'] == 0, -1, base_dados['cod_profi_clien__L'])
base_dados['cod_profi_clien__L'] = base_dados['cod_profi_clien__L'].fillna(-2)
base_dados['cod_profi_clien__L__p_20'] = np.where(np.bitwise_and(base_dados['cod_profi_clien__L'] >= -2.0, base_dados['cod_profi_clien__L'] <= 5.236441962829949), 8.0,
np.where(np.bitwise_and(base_dados['cod_profi_clien__L'] > 5.236441962829949, base_dados['cod_profi_clien__L'] <= 5.713732805509369), 9.0,
np.where(np.bitwise_and(base_dados['cod_profi_clien__L'] > 5.713732805509369, base_dados['cod_profi_clien__L'] <= 6.016157159698354), 10.0,
np.where(np.bitwise_and(base_dados['cod_profi_clien__L'] > 6.016157159698354, base_dados['cod_profi_clien__L'] <= 6.263398262591624), 11.0,
np.where(np.bitwise_and(base_dados['cod_profi_clien__L'] > 6.263398262591624, base_dados['cod_profi_clien__L'] <= 6.606650186198215), 12.0,
np.where(np.bitwise_and(base_dados['cod_profi_clien__L'] > 6.606650186198215, base_dados['cod_profi_clien__L'] <= 6.688354713946762), 13.0,
np.where(np.bitwise_and(base_dados['cod_profi_clien__L'] > 6.688354713946762, base_dados['cod_profi_clien__L'] <= 6.804614520062624), 14.0,
np.where(base_dados['cod_profi_clien__L'] > 6.804614520062624, 18.0,
 -2))))))))
base_dados['cod_profi_clien__L__p_20_g_1_1'] = np.where(base_dados['cod_profi_clien__L__p_20'] == -2.0, 1,
np.where(base_dados['cod_profi_clien__L__p_20'] == 8.0, 1,
np.where(base_dados['cod_profi_clien__L__p_20'] == 9.0, 0,
np.where(base_dados['cod_profi_clien__L__p_20'] == 10.0, 1,
np.where(base_dados['cod_profi_clien__L__p_20'] == 11.0, 0,
np.where(base_dados['cod_profi_clien__L__p_20'] == 12.0, 1,
np.where(base_dados['cod_profi_clien__L__p_20'] == 13.0, 1,
np.where(base_dados['cod_profi_clien__L__p_20'] == 14.0, 0,
np.where(base_dados['cod_profi_clien__L__p_20'] == 18.0, 1,
 0)))))))))
base_dados['cod_profi_clien__L__p_20_g_1_2'] = np.where(base_dados['cod_profi_clien__L__p_20_g_1_1'] == 0, 1,
np.where(base_dados['cod_profi_clien__L__p_20_g_1_1'] == 1, 0,
 0))
base_dados['cod_profi_clien__L__p_20_g_1'] = np.where(base_dados['cod_profi_clien__L__p_20_g_1_2'] == 0, 0,
np.where(base_dados['cod_profi_clien__L__p_20_g_1_2'] == 1, 1,
 0))


# COMMAND ----------

# MAGIC %md
# MAGIC ### Parte 2 de 2

# COMMAND ----------



base_dados['des_cep_resid__L__p_34_g_1_c1_4_1'] = np.where(np.bitwise_and(base_dados['des_cep_resid__pe_4_g_1'] == 0, base_dados['des_cep_resid__L__p_34_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['des_cep_resid__pe_4_g_1'] == 0, base_dados['des_cep_resid__L__p_34_g_1'] == 1), 0,
np.where(np.bitwise_and(base_dados['des_cep_resid__pe_4_g_1'] == 1, base_dados['des_cep_resid__L__p_34_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['des_cep_resid__pe_4_g_1'] == 1, base_dados['des_cep_resid__L__p_34_g_1'] == 1), 1,
 0))))
base_dados['des_cep_resid__L__p_34_g_1_c1_4_2'] = np.where(base_dados['des_cep_resid__L__p_34_g_1_c1_4_1'] == 0, 0,
np.where(base_dados['des_cep_resid__L__p_34_g_1_c1_4_1'] == 1, 1,
 0))
base_dados['des_cep_resid__L__p_34_g_1_c1_4'] = np.where(base_dados['des_cep_resid__L__p_34_g_1_c1_4_2'] == 0, 0,
np.where(base_dados['des_cep_resid__L__p_34_g_1_c1_4_2'] == 1, 1,
 0))
                                                         
                                                         
                                                         
                                                         
                                                         
                                                         
base_dados['des_contr__p_5_g_1_c1_5_1'] = np.where(np.bitwise_and(base_dados['des_contr__p_5_g_1'] == 0, base_dados['des_contr__p_4_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['des_contr__p_5_g_1'] == 0, base_dados['des_contr__p_4_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['des_contr__p_5_g_1'] == 1, base_dados['des_contr__p_4_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['des_contr__p_5_g_1'] == 2, base_dados['des_contr__p_4_g_1'] == 1), 2,
 0))))
base_dados['des_contr__p_5_g_1_c1_5_2'] = np.where(base_dados['des_contr__p_5_g_1_c1_5_1'] == 0, 0,
np.where(base_dados['des_contr__p_5_g_1_c1_5_1'] == 1, 1,
np.where(base_dados['des_contr__p_5_g_1_c1_5_1'] == 2, 2,
 0)))
base_dados['des_contr__p_5_g_1_c1_5'] = np.where(base_dados['des_contr__p_5_g_1_c1_5_2'] == 0, 0,
np.where(base_dados['des_contr__p_5_g_1_c1_5_2'] == 1, 1,
np.where(base_dados['des_contr__p_5_g_1_c1_5_2'] == 2, 2,
 0)))
         
         
         
         
               
base_dados['des_fones_resid__S__p_13_g_1_c1_5_1'] = np.where(np.bitwise_and(base_dados['des_fones_resid__S__p_13_g_1'] == 0, base_dados['des_fones_resid__T__p_13_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['des_fones_resid__S__p_13_g_1'] == 0, base_dados['des_fones_resid__T__p_13_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['des_fones_resid__S__p_13_g_1'] == 1, base_dados['des_fones_resid__T__p_13_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['des_fones_resid__S__p_13_g_1'] == 1, base_dados['des_fones_resid__T__p_13_g_1'] == 1), 2,
 0))))
base_dados['des_fones_resid__S__p_13_g_1_c1_5_2'] = np.where(base_dados['des_fones_resid__S__p_13_g_1_c1_5_1'] == 0, 0,
np.where(base_dados['des_fones_resid__S__p_13_g_1_c1_5_1'] == 1, 1,
np.where(base_dados['des_fones_resid__S__p_13_g_1_c1_5_1'] == 2, 2,
 0)))
base_dados['des_fones_resid__S__p_13_g_1_c1_5'] = np.where(base_dados['des_fones_resid__S__p_13_g_1_c1_5_2'] == 0, 0,
np.where(base_dados['des_fones_resid__S__p_13_g_1_c1_5_2'] == 1, 1,
np.where(base_dados['des_fones_resid__S__p_13_g_1_c1_5_2'] == 2, 2,
 0)))
         
         
         
         
              
base_dados['mob_cliente__p_7_g_1_c1_9_1'] = np.where(np.bitwise_and(base_dados['mob_cliente__p_7_g_1'] == 0, base_dados['mob_cliente__L__pe_8_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['mob_cliente__p_7_g_1'] == 1, base_dados['mob_cliente__L__pe_8_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['mob_cliente__p_7_g_1'] == 2, base_dados['mob_cliente__L__pe_8_g_1'] == 0), 2,
np.where(np.bitwise_and(base_dados['mob_cliente__p_7_g_1'] == 2, base_dados['mob_cliente__L__pe_8_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['mob_cliente__p_7_g_1'] == 3, base_dados['mob_cliente__L__pe_8_g_1'] == 1), 3,
np.where(np.bitwise_and(base_dados['mob_cliente__p_7_g_1'] == 3, base_dados['mob_cliente__L__pe_8_g_1'] == 2), 4,
np.where(np.bitwise_and(base_dados['mob_cliente__p_7_g_1'] == 4, base_dados['mob_cliente__L__pe_8_g_1'] == 2), 4,
np.where(np.bitwise_and(base_dados['mob_cliente__p_7_g_1'] == 5, base_dados['mob_cliente__L__pe_8_g_1'] == 2), 5,
np.where(np.bitwise_and(base_dados['mob_cliente__p_7_g_1'] == 5, base_dados['mob_cliente__L__pe_8_g_1'] == 3), 6,
np.where(np.bitwise_and(base_dados['mob_cliente__p_7_g_1'] == 6, base_dados['mob_cliente__L__pe_8_g_1'] == 0), 6,
np.where(np.bitwise_and(base_dados['mob_cliente__p_7_g_1'] == 6, base_dados['mob_cliente__L__pe_8_g_1'] == 3), 6,
 0)))))))))))
base_dados['mob_cliente__p_7_g_1_c1_9_2'] = np.where(base_dados['mob_cliente__p_7_g_1_c1_9_1'] == 0, 0,
np.where(base_dados['mob_cliente__p_7_g_1_c1_9_1'] == 1, 1,
np.where(base_dados['mob_cliente__p_7_g_1_c1_9_1'] == 2, 2,
np.where(base_dados['mob_cliente__p_7_g_1_c1_9_1'] == 3, 3,
np.where(base_dados['mob_cliente__p_7_g_1_c1_9_1'] == 4, 4,
np.where(base_dados['mob_cliente__p_7_g_1_c1_9_1'] == 5, 5,
np.where(base_dados['mob_cliente__p_7_g_1_c1_9_1'] == 6, 6,
 0)))))))
base_dados['mob_cliente__p_7_g_1_c1_9'] = np.where(base_dados['mob_cliente__p_7_g_1_c1_9_2'] == 0, 0,
np.where(base_dados['mob_cliente__p_7_g_1_c1_9_2'] == 1, 1,
np.where(base_dados['mob_cliente__p_7_g_1_c1_9_2'] == 2, 2,
np.where(base_dados['mob_cliente__p_7_g_1_c1_9_2'] == 3, 3,
np.where(base_dados['mob_cliente__p_7_g_1_c1_9_2'] == 4, 4,
np.where(base_dados['mob_cliente__p_7_g_1_c1_9_2'] == 5, 5,
np.where(base_dados['mob_cliente__p_7_g_1_c1_9_2'] == 6, 6,
 0)))))))
         
         
         
         
         
        
base_dados['idade__pe_13_g_1_c1_7_1'] = np.where(np.bitwise_and(base_dados['idade__pe_13_g_1'] == 0, base_dados['idade__pe_3_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['idade__pe_13_g_1'] == 0, base_dados['idade__pe_3_g_1'] == 1), 0,
np.where(np.bitwise_and(base_dados['idade__pe_13_g_1'] == 1, base_dados['idade__pe_3_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['idade__pe_13_g_1'] == 1, base_dados['idade__pe_3_g_1'] == 1), 2,
 0))))
base_dados['idade__pe_13_g_1_c1_7_2'] = np.where(base_dados['idade__pe_13_g_1_c1_7_1'] == 0, 0,
np.where(base_dados['idade__pe_13_g_1_c1_7_1'] == 1, 1,
np.where(base_dados['idade__pe_13_g_1_c1_7_1'] == 2, 2,
 0)))
base_dados['idade__pe_13_g_1_c1_7'] = np.where(base_dados['idade__pe_13_g_1_c1_7_2'] == 0, 0,
np.where(base_dados['idade__pe_13_g_1_c1_7_2'] == 1, 1,
np.where(base_dados['idade__pe_13_g_1_c1_7_2'] == 2, 2,
 0)))
         
         
         
         
         
         
         
base_dados['des_fones_refer__R__p_10_g_1_c1_10_1'] = np.where(np.bitwise_and(base_dados['des_fones_refer__R__p_10_g_1'] == 0, base_dados['des_fones_refer__T__pe_8_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['des_fones_refer__R__p_10_g_1'] == 0, base_dados['des_fones_refer__T__pe_8_g_1'] == 1), 0,
np.where(np.bitwise_and(base_dados['des_fones_refer__R__p_10_g_1'] == 1, base_dados['des_fones_refer__T__pe_8_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['des_fones_refer__R__p_10_g_1'] == 1, base_dados['des_fones_refer__T__pe_8_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['des_fones_refer__R__p_10_g_1'] == 2, base_dados['des_fones_refer__T__pe_8_g_1'] == 1), 2,
 0)))))
base_dados['des_fones_refer__R__p_10_g_1_c1_10_2'] = np.where(base_dados['des_fones_refer__R__p_10_g_1_c1_10_1'] == 0, 0,
np.where(base_dados['des_fones_refer__R__p_10_g_1_c1_10_1'] == 1, 1,
np.where(base_dados['des_fones_refer__R__p_10_g_1_c1_10_1'] == 2, 2,
 0)))
base_dados['des_fones_refer__R__p_10_g_1_c1_10'] = np.where(base_dados['des_fones_refer__R__p_10_g_1_c1_10_2'] == 0, 0,
np.where(base_dados['des_fones_refer__R__p_10_g_1_c1_10_2'] == 1, 1,
np.where(base_dados['des_fones_refer__R__p_10_g_1_c1_10_2'] == 2, 2,
 0)))
         
         
         
         
         
        
base_dados['des_fones_celul__L__pe_15_g_1_c1_38_1'] = np.where(np.bitwise_and(base_dados['des_fones_celul__pe_7_g_1'] == 0, base_dados['des_fones_celul__L__pe_15_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['des_fones_celul__pe_7_g_1'] == 0, base_dados['des_fones_celul__L__pe_15_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['des_fones_celul__pe_7_g_1'] == 1, base_dados['des_fones_celul__L__pe_15_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['des_fones_celul__pe_7_g_1'] == 2, base_dados['des_fones_celul__L__pe_15_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['des_fones_celul__pe_7_g_1'] == 2, base_dados['des_fones_celul__L__pe_15_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['des_fones_celul__pe_7_g_1'] == 2, base_dados['des_fones_celul__L__pe_15_g_1'] == 2), 2,
 0))))))
base_dados['des_fones_celul__L__pe_15_g_1_c1_38_2'] = np.where(base_dados['des_fones_celul__L__pe_15_g_1_c1_38_1'] == 0, 0,
np.where(base_dados['des_fones_celul__L__pe_15_g_1_c1_38_1'] == 1, 1,
np.where(base_dados['des_fones_celul__L__pe_15_g_1_c1_38_1'] == 2, 2,
 0)))
base_dados['des_fones_celul__L__pe_15_g_1_c1_38'] = np.where(base_dados['des_fones_celul__L__pe_15_g_1_c1_38_2'] == 0, 0,
np.where(base_dados['des_fones_celul__L__pe_15_g_1_c1_38_2'] == 1, 1,
np.where(base_dados['des_fones_celul__L__pe_15_g_1_c1_38_2'] == 2, 2,
 0)))
         
         
         
         
         
         
         
base_dados['val_compr__T__p_5_g_1_c1_11_1'] = np.where(np.bitwise_and(base_dados['val_compr__L__p_6_g_1'] == 0, base_dados['val_compr__T__p_5_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['val_compr__L__p_6_g_1'] == 0, base_dados['val_compr__T__p_5_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['val_compr__L__p_6_g_1'] == 1, base_dados['val_compr__T__p_5_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['val_compr__L__p_6_g_1'] == 1, base_dados['val_compr__T__p_5_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['val_compr__L__p_6_g_1'] == 2, base_dados['val_compr__T__p_5_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['val_compr__L__p_6_g_1'] == 2, base_dados['val_compr__T__p_5_g_1'] == 1), 3,
np.where(np.bitwise_and(base_dados['val_compr__L__p_6_g_1'] == 3, base_dados['val_compr__T__p_5_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['val_compr__L__p_6_g_1'] == 3, base_dados['val_compr__T__p_5_g_1'] == 1), 4,
 0))))))))
base_dados['val_compr__T__p_5_g_1_c1_11_2'] = np.where(base_dados['val_compr__T__p_5_g_1_c1_11_1'] == 0, 0,
np.where(base_dados['val_compr__T__p_5_g_1_c1_11_1'] == 1, 1,
np.where(base_dados['val_compr__T__p_5_g_1_c1_11_1'] == 2, 2,
np.where(base_dados['val_compr__T__p_5_g_1_c1_11_1'] == 3, 3,
np.where(base_dados['val_compr__T__p_5_g_1_c1_11_1'] == 4, 4,
 0)))))
base_dados['val_compr__T__p_5_g_1_c1_11'] = np.where(base_dados['val_compr__T__p_5_g_1_c1_11_2'] == 0, 0,
np.where(base_dados['val_compr__T__p_5_g_1_c1_11_2'] == 1, 1,
np.where(base_dados['val_compr__T__p_5_g_1_c1_11_2'] == 2, 2,
np.where(base_dados['val_compr__T__p_5_g_1_c1_11_2'] == 3, 3,
np.where(base_dados['val_compr__T__p_5_g_1_c1_11_2'] == 4, 4,
 0)))))
         
         
         
         
                
base_dados['cod_profi_clien__L__p_20_g_1_c1_4_1'] = np.where(np.bitwise_and(base_dados['cod_profi_clien__p_4_g_1'] == 0, base_dados['cod_profi_clien__L__p_20_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['cod_profi_clien__p_4_g_1'] == 1, base_dados['cod_profi_clien__L__p_20_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['cod_profi_clien__p_4_g_1'] == 1, base_dados['cod_profi_clien__L__p_20_g_1'] == 1), 2,
 0)))
base_dados['cod_profi_clien__L__p_20_g_1_c1_4_2'] = np.where(base_dados['cod_profi_clien__L__p_20_g_1_c1_4_1'] == 0, 0,
np.where(base_dados['cod_profi_clien__L__p_20_g_1_c1_4_1'] == 1, 1,
np.where(base_dados['cod_profi_clien__L__p_20_g_1_c1_4_1'] == 2, 2,
 0)))
base_dados['cod_profi_clien__L__p_20_g_1_c1_4'] = np.where(base_dados['cod_profi_clien__L__p_20_g_1_c1_4_2'] == 0, 0,
np.where(base_dados['cod_profi_clien__L__p_20_g_1_c1_4_2'] == 1, 1,
np.where(base_dados['cod_profi_clien__L__p_20_g_1_c1_4_2'] == 2, 2,
 0)))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Mantendo apenas as variáveis do modelo

# COMMAND ----------

import pickle
modelo=pickle.load(open(caminho_base + 'model_fit_tribanco.sav', 'rb'))

base_teste_c0 = base_dados[[chave,'mob_cliente__p_7_g_1_c1_9','des_fones_refer__R__p_10_g_1_c1_10','idade__pe_13_g_1_c1_7','cod_produt_gh38','des_fones_resid__S__p_13_g_1_c1_5','cod_credor_gh38','des_contr__p_5_g_1_c1_5','cod_profi_clien__L__p_20_g_1_c1_4','ind_estad_civil_gh38','des_fones_celul__L__pe_15_g_1_c1_38','des_cep_resid__L__p_34_g_1_c1_4','val_compr__T__p_5_g_1_c1_11']]

var_fin_c0=['mob_cliente__p_7_g_1_c1_9','des_fones_refer__R__p_10_g_1_c1_10','idade__pe_13_g_1_c1_7','cod_produt_gh38','des_fones_resid__S__p_13_g_1_c1_5','cod_credor_gh38','des_contr__p_5_g_1_c1_5','cod_profi_clien__L__p_20_g_1_c1_4','ind_estad_civil_gh38','des_fones_celul__L__pe_15_g_1_c1_38','des_cep_resid__L__p_34_g_1_c1_4','val_compr__T__p_5_g_1_c1_11']

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

x_teste2['P_1_R_p_8_g_1'] = np.where(x_teste2['P_1_R'] <= 0.176140123, 0,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.176140123, x_teste2['P_1_R'] <= 0.224400752), 1,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.224400752, x_teste2['P_1_R'] <= 0.282280343), 2,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.282280343, x_teste2['P_1_R'] <= 0.365863898), 3,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.365863898, x_teste2['P_1_R'] <= 0.49995784), 4,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.49995784, x_teste2['P_1_R'] <= 0.586220367), 5,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.586220367, x_teste2['P_1_R'] <= 0.689826683), 6,
    np.where(x_teste2['P_1_R'] > 0.689826683,7,0))))))))

x_teste2['P_1_p_20_g_1'] = np.where(x_teste2['P_1'] <= 0.04145281, 0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.04145281, x_teste2['P_1'] <= 0.237969921), 1,
    np.where(x_teste2['P_1'] > 0.237969921,2,0)))

x_teste2['GH'] = np.where(np.bitwise_and(x_teste2['P_1_p_20_g_1'] == 0, x_teste2['P_1_R_p_8_g_1'] == 0), 0,
    np.where(np.bitwise_and(x_teste2['P_1_p_20_g_1'] == 0, x_teste2['P_1_R_p_8_g_1'] == 1), 0,
    np.where(np.bitwise_and(x_teste2['P_1_p_20_g_1'] == 1, x_teste2['P_1_R_p_8_g_1'] == 1), 1,
    np.where(np.bitwise_and(x_teste2['P_1_p_20_g_1'] == 1, x_teste2['P_1_R_p_8_g_1'] == 2), 1,
    np.where(np.bitwise_and(x_teste2['P_1_p_20_g_1'] == 1, x_teste2['P_1_R_p_8_g_1'] == 3), 2,
    np.where(np.bitwise_and(x_teste2['P_1_p_20_g_1'] == 1, x_teste2['P_1_R_p_8_g_1'] == 4), 2,
    np.where(np.bitwise_and(x_teste2['P_1_p_20_g_1'] == 2, x_teste2['P_1_R_p_8_g_1'] == 4), 3,
    np.where(np.bitwise_and(x_teste2['P_1_p_20_g_1'] == 2, x_teste2['P_1_R_p_8_g_1'] == 5), 3,
    np.where(np.bitwise_and(x_teste2['P_1_p_20_g_1'] == 2, x_teste2['P_1_R_p_8_g_1'] == 6), 4,
    np.where(np.bitwise_and(x_teste2['P_1_p_20_g_1'] == 2, x_teste2['P_1_R_p_8_g_1'] == 7), 5,
             2))))))))))

del x_teste2['P_1_R']
del x_teste2['P_1_R_p_8_g_1']
del x_teste2['P_1_p_20_g_1']

x_teste2


# COMMAND ----------

