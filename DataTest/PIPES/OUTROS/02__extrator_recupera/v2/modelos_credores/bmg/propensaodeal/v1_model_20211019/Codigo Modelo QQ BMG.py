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

%matplotlib inline

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inserindo hiperparâmetros do Algoritmo

# COMMAND ----------

## Parâmetros do Algoritmo

#Variável chave-primaria
chave = 'DOCUMENTO'

#Variável resposta ou target
target = 'VARIAVEL_RESPOSTA'

#Nome da Base de Dados
N_Base = "20210907_amostra_aleatoria.csv"

#Caminho da base de dados
caminho_base = "Base_Dados_Ferramenta/BMG/"

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
base_dados[target] = base_dados[target].map({True:1,False:0},na_action=None)
print("shape da Base de Dados:",base_dados.shape)

list_var = [chave,target, 'ind_sexo', 'cod_credor', 'des_estad_comer', 'val_renda', 'des_cep_resid', 'tip_ender_corre', 'des_fones_resid', 'des_uf_rg', 'qtd_prest', 'val_compr', 'cod_filia', 'cod_profi_clien']
base_dados = base_dados[list_var]
base_dados['raiz_cep'] = base_dados['des_cep_resid']/1000

base_dados['cod_credor'] = base_dados['cod_credor'].replace(np.nan, '-3')
base_dados['des_estad_comer'] = base_dados['des_estad_comer'].replace(np.nan, '-3')
base_dados['tip_ender_corre'] = base_dados['tip_ender_corre'].replace(np.nan, '-3')
base_dados['des_uf_rg'] = base_dados['des_uf_rg'].replace(np.nan, '-3')
base_dados['ind_sexo'] = base_dados['ind_sexo'].replace(np.nan, '-3')
base_dados['cod_filia'] = base_dados['cod_filia'].replace(np.nan, '-3')

base_dados.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gerando variáveis Categóricas

# COMMAND ----------


base_dados['cod_credor_gh30'] = np.where(base_dados['cod_credor'] == 'ATIVO_CS', 0,
    np.where(base_dados['cod_credor'] == 'BMG_AT', 1,
    np.where(base_dados['cod_credor'] == 'BMG_CONT', 2,
    np.where(base_dados['cod_credor'] == 'BMG_CS', 3,
    np.where(base_dados['cod_credor'] == 'CARDCONT', 4,
    np.where(base_dados['cod_credor'] == 'CP_LENDC', 5,
    np.where(base_dados['cod_credor'] == 'EXFUN_CS', 6,
    0)))))))
         
base_dados['cod_credor_gh31'] = np.where(base_dados['cod_credor_gh30'] == 0, 0,
    np.where(base_dados['cod_credor_gh30'] == 1, 1,
    np.where(base_dados['cod_credor_gh30'] == 2, 2,
    np.where(base_dados['cod_credor_gh30'] == 3, 3,
    np.where(base_dados['cod_credor_gh30'] == 4, 4,
    np.where(base_dados['cod_credor_gh30'] == 5, 5,
    np.where(base_dados['cod_credor_gh30'] == 6, 6,
    0)))))))
         
base_dados['cod_credor_gh32'] = np.where(base_dados['cod_credor_gh31'] == 0, 0,
    np.where(base_dados['cod_credor_gh31'] == 1, 1,
    np.where(base_dados['cod_credor_gh31'] == 2, 2,
    np.where(base_dados['cod_credor_gh31'] == 3, 3,
    np.where(base_dados['cod_credor_gh31'] == 4, 4,
    np.where(base_dados['cod_credor_gh31'] == 5, 5,
    np.where(base_dados['cod_credor_gh31'] == 6, 6,
    0)))))))
         
base_dados['cod_credor_gh33'] = np.where(base_dados['cod_credor_gh32'] == 0, 0,
    np.where(base_dados['cod_credor_gh32'] == 1, 1,
    np.where(base_dados['cod_credor_gh32'] == 2, 2,
    np.where(base_dados['cod_credor_gh32'] == 3, 3,
    np.where(base_dados['cod_credor_gh32'] == 4, 4,
    np.where(base_dados['cod_credor_gh32'] == 5, 5,
    np.where(base_dados['cod_credor_gh32'] == 6, 6,
    0)))))))
         
base_dados['cod_credor_gh34'] = np.where(base_dados['cod_credor_gh33'] == 0, 0,
    np.where(base_dados['cod_credor_gh33'] == 1, 3,
    np.where(base_dados['cod_credor_gh33'] == 2, 2,
    np.where(base_dados['cod_credor_gh33'] == 3, 3,
    np.where(base_dados['cod_credor_gh33'] == 4, 4,
    np.where(base_dados['cod_credor_gh33'] == 5, 2,
    np.where(base_dados['cod_credor_gh33'] == 6, 6,
    0)))))))
         
base_dados['cod_credor_gh35'] = np.where(base_dados['cod_credor_gh34'] == 0, 0,
    np.where(base_dados['cod_credor_gh34'] == 2, 1,
    np.where(base_dados['cod_credor_gh34'] == 3, 2,
    np.where(base_dados['cod_credor_gh34'] == 4, 3,
    np.where(base_dados['cod_credor_gh34'] == 6, 4,
    0)))))
         
base_dados['cod_credor_gh36'] = np.where(base_dados['cod_credor_gh35'] == 0, 1,
    np.where(base_dados['cod_credor_gh35'] == 1, 3,
    np.where(base_dados['cod_credor_gh35'] == 2, 2,
    np.where(base_dados['cod_credor_gh35'] == 3, 4,
    np.where(base_dados['cod_credor_gh35'] == 4, 0,
    0)))))
         
base_dados['cod_credor_gh37'] = np.where(base_dados['cod_credor_gh36'] == 0, 0,
    np.where(base_dados['cod_credor_gh36'] == 1, 1,
    np.where(base_dados['cod_credor_gh36'] == 2, 2,
    np.where(base_dados['cod_credor_gh36'] == 3, 3,
    np.where(base_dados['cod_credor_gh36'] == 4, 4,
    0)))))
         
base_dados['cod_credor_gh38'] = np.where(base_dados['cod_credor_gh37'] == 0, 0,
    np.where(base_dados['cod_credor_gh37'] == 1, 1,
    np.where(base_dados['cod_credor_gh37'] == 2, 2,
    np.where(base_dados['cod_credor_gh37'] == 3, 3,
    np.where(base_dados['cod_credor_gh37'] == 4, 4,
    0)))))
         
        
                                            
                                            
                                            
                                            
                                            
                                            
   
         
         
         
         
            
base_dados['des_estad_comer_gh30'] = np.where(base_dados['des_estad_comer'] == '-3', 0,
    np.where(base_dados['des_estad_comer'] == 'AC', 1,
    np.where(base_dados['des_estad_comer'] == 'AL', 2,
    np.where(base_dados['des_estad_comer'] == 'AM', 3,
    np.where(base_dados['des_estad_comer'] == 'AP', 4,
    np.where(base_dados['des_estad_comer'] == 'BA', 5,
    np.where(base_dados['des_estad_comer'] == 'CE', 6,
    np.where(base_dados['des_estad_comer'] == 'DF', 7,
    np.where(base_dados['des_estad_comer'] == 'ES', 8,
    np.where(base_dados['des_estad_comer'] == 'GO', 9,
    np.where(base_dados['des_estad_comer'] == 'MA', 10,
    np.where(base_dados['des_estad_comer'] == 'MG', 11,
    np.where(base_dados['des_estad_comer'] == 'MS', 12,
    np.where(base_dados['des_estad_comer'] == 'MT', 13,
    np.where(base_dados['des_estad_comer'] == 'PA', 14,
    np.where(base_dados['des_estad_comer'] == 'PB', 15,
    np.where(base_dados['des_estad_comer'] == 'PE', 16,
    np.where(base_dados['des_estad_comer'] == 'PI', 17,
    np.where(base_dados['des_estad_comer'] == 'PR', 18,
    np.where(base_dados['des_estad_comer'] == 'RJ', 19,
    np.where(base_dados['des_estad_comer'] == 'RN', 20,
    np.where(base_dados['des_estad_comer'] == 'RO', 21,
    np.where(base_dados['des_estad_comer'] == 'RR', 22,
    np.where(base_dados['des_estad_comer'] == 'RS', 23,
    np.where(base_dados['des_estad_comer'] == 'SC', 24,
    np.where(base_dados['des_estad_comer'] == 'SE', 25,
    np.where(base_dados['des_estad_comer'] == 'SP', 26,
    np.where(base_dados['des_estad_comer'] == 'TO', 27,
    0))))))))))))))))))))))))))))
         
base_dados['des_estad_comer_gh31'] = np.where(base_dados['des_estad_comer_gh30'] == 0, 0,
    np.where(base_dados['des_estad_comer_gh30'] == 1, 1,
    np.where(base_dados['des_estad_comer_gh30'] == 2, 1,
    np.where(base_dados['des_estad_comer_gh30'] == 3, 3,
    np.where(base_dados['des_estad_comer_gh30'] == 4, 4,
    np.where(base_dados['des_estad_comer_gh30'] == 5, 5,
    np.where(base_dados['des_estad_comer_gh30'] == 6, 6,
    np.where(base_dados['des_estad_comer_gh30'] == 7, 7,
    np.where(base_dados['des_estad_comer_gh30'] == 8, 8,
    np.where(base_dados['des_estad_comer_gh30'] == 9, 9,
    np.where(base_dados['des_estad_comer_gh30'] == 10, 10,
    np.where(base_dados['des_estad_comer_gh30'] == 11, 11,
    np.where(base_dados['des_estad_comer_gh30'] == 12, 12,
    np.where(base_dados['des_estad_comer_gh30'] == 13, 13,
    np.where(base_dados['des_estad_comer_gh30'] == 14, 14,
    np.where(base_dados['des_estad_comer_gh30'] == 15, 15,
    np.where(base_dados['des_estad_comer_gh30'] == 16, 16,
    np.where(base_dados['des_estad_comer_gh30'] == 17, 17,
    np.where(base_dados['des_estad_comer_gh30'] == 18, 18,
    np.where(base_dados['des_estad_comer_gh30'] == 19, 19,
    np.where(base_dados['des_estad_comer_gh30'] == 20, 19,
    np.where(base_dados['des_estad_comer_gh30'] == 21, 21,
    np.where(base_dados['des_estad_comer_gh30'] == 22, 22,
    np.where(base_dados['des_estad_comer_gh30'] == 23, 23,
    np.where(base_dados['des_estad_comer_gh30'] == 24, 24,
    np.where(base_dados['des_estad_comer_gh30'] == 25, 25,
    np.where(base_dados['des_estad_comer_gh30'] == 26, 26,
    np.where(base_dados['des_estad_comer_gh30'] == 27, 27,
    0))))))))))))))))))))))))))))
         
base_dados['des_estad_comer_gh32'] = np.where(base_dados['des_estad_comer_gh31'] == 0, 0,
    np.where(base_dados['des_estad_comer_gh31'] == 1, 1,
    np.where(base_dados['des_estad_comer_gh31'] == 3, 2,
    np.where(base_dados['des_estad_comer_gh31'] == 4, 3,
    np.where(base_dados['des_estad_comer_gh31'] == 5, 4,
    np.where(base_dados['des_estad_comer_gh31'] == 6, 5,
    np.where(base_dados['des_estad_comer_gh31'] == 7, 6,
    np.where(base_dados['des_estad_comer_gh31'] == 8, 7,
    np.where(base_dados['des_estad_comer_gh31'] == 9, 8,
    np.where(base_dados['des_estad_comer_gh31'] == 10, 9,
    np.where(base_dados['des_estad_comer_gh31'] == 11, 10,
    np.where(base_dados['des_estad_comer_gh31'] == 12, 11,
    np.where(base_dados['des_estad_comer_gh31'] == 13, 12,
    np.where(base_dados['des_estad_comer_gh31'] == 14, 13,
    np.where(base_dados['des_estad_comer_gh31'] == 15, 14,
    np.where(base_dados['des_estad_comer_gh31'] == 16, 15,
    np.where(base_dados['des_estad_comer_gh31'] == 17, 16,
    np.where(base_dados['des_estad_comer_gh31'] == 18, 17,
    np.where(base_dados['des_estad_comer_gh31'] == 19, 18,
    np.where(base_dados['des_estad_comer_gh31'] == 21, 19,
    np.where(base_dados['des_estad_comer_gh31'] == 22, 20,
    np.where(base_dados['des_estad_comer_gh31'] == 23, 21,
    np.where(base_dados['des_estad_comer_gh31'] == 24, 22,
    np.where(base_dados['des_estad_comer_gh31'] == 25, 23,
    np.where(base_dados['des_estad_comer_gh31'] == 26, 24,
    np.where(base_dados['des_estad_comer_gh31'] == 27, 25,
    0))))))))))))))))))))))))))
         
base_dados['des_estad_comer_gh33'] = np.where(base_dados['des_estad_comer_gh32'] == 0, 0,
    np.where(base_dados['des_estad_comer_gh32'] == 1, 1,
    np.where(base_dados['des_estad_comer_gh32'] == 2, 2,
    np.where(base_dados['des_estad_comer_gh32'] == 3, 3,
    np.where(base_dados['des_estad_comer_gh32'] == 4, 4,
    np.where(base_dados['des_estad_comer_gh32'] == 5, 5,
    np.where(base_dados['des_estad_comer_gh32'] == 6, 6,
    np.where(base_dados['des_estad_comer_gh32'] == 7, 7,
    np.where(base_dados['des_estad_comer_gh32'] == 8, 8,
    np.where(base_dados['des_estad_comer_gh32'] == 9, 9,
    np.where(base_dados['des_estad_comer_gh32'] == 10, 10,
    np.where(base_dados['des_estad_comer_gh32'] == 11, 11,
    np.where(base_dados['des_estad_comer_gh32'] == 12, 12,
    np.where(base_dados['des_estad_comer_gh32'] == 13, 13,
    np.where(base_dados['des_estad_comer_gh32'] == 14, 14,
    np.where(base_dados['des_estad_comer_gh32'] == 15, 15,
    np.where(base_dados['des_estad_comer_gh32'] == 16, 16,
    np.where(base_dados['des_estad_comer_gh32'] == 17, 17,
    np.where(base_dados['des_estad_comer_gh32'] == 18, 18,
    np.where(base_dados['des_estad_comer_gh32'] == 19, 19,
    np.where(base_dados['des_estad_comer_gh32'] == 20, 20,
    np.where(base_dados['des_estad_comer_gh32'] == 21, 21,
    np.where(base_dados['des_estad_comer_gh32'] == 22, 22,
    np.where(base_dados['des_estad_comer_gh32'] == 23, 23,
    np.where(base_dados['des_estad_comer_gh32'] == 24, 24,
    np.where(base_dados['des_estad_comer_gh32'] == 25, 25,
    0))))))))))))))))))))))))))
         
base_dados['des_estad_comer_gh34'] = np.where(base_dados['des_estad_comer_gh33'] == 0, 0,
    np.where(base_dados['des_estad_comer_gh33'] == 1, 25,
    np.where(base_dados['des_estad_comer_gh33'] == 2, 25,
    np.where(base_dados['des_estad_comer_gh33'] == 3, 25,
    np.where(base_dados['des_estad_comer_gh33'] == 4, 25,
    np.where(base_dados['des_estad_comer_gh33'] == 5, 0,
    np.where(base_dados['des_estad_comer_gh33'] == 6, 25,
    np.where(base_dados['des_estad_comer_gh33'] == 7, 25,
    np.where(base_dados['des_estad_comer_gh33'] == 8, 25,
    np.where(base_dados['des_estad_comer_gh33'] == 9, 25,
    np.where(base_dados['des_estad_comer_gh33'] == 10, 25,
    np.where(base_dados['des_estad_comer_gh33'] == 11, 25,
    np.where(base_dados['des_estad_comer_gh33'] == 12, 25,
    np.where(base_dados['des_estad_comer_gh33'] == 13, 25,
    np.where(base_dados['des_estad_comer_gh33'] == 14, 25,
    np.where(base_dados['des_estad_comer_gh33'] == 15, 25,
    np.where(base_dados['des_estad_comer_gh33'] == 16, 25,
    np.where(base_dados['des_estad_comer_gh33'] == 17, 25,
    np.where(base_dados['des_estad_comer_gh33'] == 18, 25,
    np.where(base_dados['des_estad_comer_gh33'] == 19, 25,
    np.where(base_dados['des_estad_comer_gh33'] == 20, 0,
    np.where(base_dados['des_estad_comer_gh33'] == 21, 25,
    np.where(base_dados['des_estad_comer_gh33'] == 22, 0,
    np.where(base_dados['des_estad_comer_gh33'] == 23, 0,
    np.where(base_dados['des_estad_comer_gh33'] == 24, 25,
    np.where(base_dados['des_estad_comer_gh33'] == 25, 25,
    0))))))))))))))))))))))))))
         
base_dados['des_estad_comer_gh35'] = np.where(base_dados['des_estad_comer_gh34'] == 0, 0,
    np.where(base_dados['des_estad_comer_gh34'] == 25, 1,
    0))
                                              
base_dados['des_estad_comer_gh36'] = np.where(base_dados['des_estad_comer_gh35'] == 0, 1,
    np.where(base_dados['des_estad_comer_gh35'] == 1, 0,
    0))
                                              
base_dados['des_estad_comer_gh37'] = np.where(base_dados['des_estad_comer_gh36'] == 0, 0,
    np.where(base_dados['des_estad_comer_gh36'] == 1, 1,
    0))
                                              
base_dados['des_estad_comer_gh38'] = np.where(base_dados['des_estad_comer_gh37'] == 0, 0,
    np.where(base_dados['des_estad_comer_gh37'] == 1, 1,
    0))
                                              
                                              
                                              
                                              
                                              
                                              
base_dados['ind_sexo_gh30'] = np.where(base_dados['ind_sexo'] == '-3', 0,
    np.where(base_dados['ind_sexo'] == 'F', 1,
    np.where(base_dados['ind_sexo'] == 'M', 2,
    0)))
         
base_dados['ind_sexo_gh31'] = np.where(base_dados['ind_sexo_gh30'] == 0, 0,
    np.where(base_dados['ind_sexo_gh30'] == 1, 0,
    np.where(base_dados['ind_sexo_gh30'] == 2, 2,
    0)))
         
base_dados['ind_sexo_gh32'] = np.where(base_dados['ind_sexo_gh31'] == 0, 0,
    np.where(base_dados['ind_sexo_gh31'] == 2, 1,
    0))
                                       
base_dados['ind_sexo_gh33'] = np.where(base_dados['ind_sexo_gh32'] == 0, 0,
    np.where(base_dados['ind_sexo_gh32'] == 1, 1,
    0))
                                       
base_dados['ind_sexo_gh34'] = np.where(base_dados['ind_sexo_gh33'] == 0, 0,
    np.where(base_dados['ind_sexo_gh33'] == 1, 1,
    0))
                                       
base_dados['ind_sexo_gh35'] = np.where(base_dados['ind_sexo_gh34'] == 0, 0,
    np.where(base_dados['ind_sexo_gh34'] == 1, 1,
    0))
                                       
base_dados['ind_sexo_gh36'] = np.where(base_dados['ind_sexo_gh35'] == 0, 1,
    np.where(base_dados['ind_sexo_gh35'] == 1, 0,
    0))
                                       
base_dados['ind_sexo_gh37'] = np.where(base_dados['ind_sexo_gh36'] == 0, 0,
    np.where(base_dados['ind_sexo_gh36'] == 1, 1,
    0))
                                       
base_dados['ind_sexo_gh38'] = np.where(base_dados['ind_sexo_gh37'] == 0, 0,
    np.where(base_dados['ind_sexo_gh37'] == 1, 1,
    0))
                                       
                                       
                                       
                                       
                                       
                                       
                                       
                                       
base_dados['des_uf_rg_gh30'] = np.where(base_dados['des_uf_rg'] == '-3', 0,
    np.where(base_dados['des_uf_rg'] == 'AC', 1,
    np.where(base_dados['des_uf_rg'] == 'AL', 2,
    np.where(base_dados['des_uf_rg'] == 'AM', 3,
    np.where(base_dados['des_uf_rg'] == 'AP', 4,
    np.where(base_dados['des_uf_rg'] == 'BA', 5,
    np.where(base_dados['des_uf_rg'] == 'CE', 6,
    np.where(base_dados['des_uf_rg'] == 'DF', 7,
    np.where(base_dados['des_uf_rg'] == 'ES', 8,
    np.where(base_dados['des_uf_rg'] == 'GO', 9,
    np.where(base_dados['des_uf_rg'] == 'M', 10,
    np.where(base_dados['des_uf_rg'] == 'MA', 11,
    np.where(base_dados['des_uf_rg'] == 'MG', 12,
    np.where(base_dados['des_uf_rg'] == 'MS', 13,
    np.where(base_dados['des_uf_rg'] == 'MT', 14,
    np.where(base_dados['des_uf_rg'] == 'NI', 15,
    np.where(base_dados['des_uf_rg'] == 'P', 16,
    np.where(base_dados['des_uf_rg'] == 'PA', 17,
    np.where(base_dados['des_uf_rg'] == 'PB', 18,
    np.where(base_dados['des_uf_rg'] == 'PE', 19,
    np.where(base_dados['des_uf_rg'] == 'PI', 20,
    np.where(base_dados['des_uf_rg'] == 'PR', 21,
    np.where(base_dados['des_uf_rg'] == 'RI', 22,
    np.where(base_dados['des_uf_rg'] == 'RJ', 23,
    np.where(base_dados['des_uf_rg'] == 'RN', 24,
    np.where(base_dados['des_uf_rg'] == 'RO', 25,
    np.where(base_dados['des_uf_rg'] == 'RR', 26,
    np.where(base_dados['des_uf_rg'] == 'RS', 27,
    np.where(base_dados['des_uf_rg'] == 'SC', 28,
    np.where(base_dados['des_uf_rg'] == 'SE', 29,
    np.where(base_dados['des_uf_rg'] == 'SP', 30,
    np.where(base_dados['des_uf_rg'] == 'TO', 31,
    0))))))))))))))))))))))))))))))))
         
base_dados['des_uf_rg_gh31'] = np.where(base_dados['des_uf_rg_gh30'] == 0, 0,
    np.where(base_dados['des_uf_rg_gh30'] == 1, 1,
    np.where(base_dados['des_uf_rg_gh30'] == 2, 2,
    np.where(base_dados['des_uf_rg_gh30'] == 3, 3,
    np.where(base_dados['des_uf_rg_gh30'] == 4, 3,
    np.where(base_dados['des_uf_rg_gh30'] == 5, 5,
    np.where(base_dados['des_uf_rg_gh30'] == 6, 5,
    np.where(base_dados['des_uf_rg_gh30'] == 7, 7,
    np.where(base_dados['des_uf_rg_gh30'] == 8, 8,
    np.where(base_dados['des_uf_rg_gh30'] == 9, 9,
    np.where(base_dados['des_uf_rg_gh30'] == 10, 10,
    np.where(base_dados['des_uf_rg_gh30'] == 11, 11,
    np.where(base_dados['des_uf_rg_gh30'] == 12, 12,
    np.where(base_dados['des_uf_rg_gh30'] == 13, 12,
    np.where(base_dados['des_uf_rg_gh30'] == 14, 14,
    np.where(base_dados['des_uf_rg_gh30'] == 15, 15,
    np.where(base_dados['des_uf_rg_gh30'] == 16, 15,
    np.where(base_dados['des_uf_rg_gh30'] == 17, 17,
    np.where(base_dados['des_uf_rg_gh30'] == 18, 18,
    np.where(base_dados['des_uf_rg_gh30'] == 19, 18,
    np.where(base_dados['des_uf_rg_gh30'] == 20, 20,
    np.where(base_dados['des_uf_rg_gh30'] == 21, 21,
    np.where(base_dados['des_uf_rg_gh30'] == 22, 22,
    np.where(base_dados['des_uf_rg_gh30'] == 23, 23,
    np.where(base_dados['des_uf_rg_gh30'] == 24, 24,
    np.where(base_dados['des_uf_rg_gh30'] == 25, 25,
    np.where(base_dados['des_uf_rg_gh30'] == 26, 25,
    np.where(base_dados['des_uf_rg_gh30'] == 27, 27,
    np.where(base_dados['des_uf_rg_gh30'] == 28, 27,
    np.where(base_dados['des_uf_rg_gh30'] == 29, 29,
    np.where(base_dados['des_uf_rg_gh30'] == 30, 30,
    np.where(base_dados['des_uf_rg_gh30'] == 31, 31,
    0))))))))))))))))))))))))))))))))
         
base_dados['des_uf_rg_gh32'] = np.where(base_dados['des_uf_rg_gh31'] == 0, 0,
    np.where(base_dados['des_uf_rg_gh31'] == 1, 1,
    np.where(base_dados['des_uf_rg_gh31'] == 2, 2,
    np.where(base_dados['des_uf_rg_gh31'] == 3, 3,
    np.where(base_dados['des_uf_rg_gh31'] == 5, 4,
    np.where(base_dados['des_uf_rg_gh31'] == 7, 5,
    np.where(base_dados['des_uf_rg_gh31'] == 8, 6,
    np.where(base_dados['des_uf_rg_gh31'] == 9, 7,
    np.where(base_dados['des_uf_rg_gh31'] == 10, 8,
    np.where(base_dados['des_uf_rg_gh31'] == 11, 9,
    np.where(base_dados['des_uf_rg_gh31'] == 12, 10,
    np.where(base_dados['des_uf_rg_gh31'] == 14, 11,
    np.where(base_dados['des_uf_rg_gh31'] == 15, 12,
    np.where(base_dados['des_uf_rg_gh31'] == 17, 13,
    np.where(base_dados['des_uf_rg_gh31'] == 18, 14,
    np.where(base_dados['des_uf_rg_gh31'] == 20, 15,
    np.where(base_dados['des_uf_rg_gh31'] == 21, 16,
    np.where(base_dados['des_uf_rg_gh31'] == 22, 17,
    np.where(base_dados['des_uf_rg_gh31'] == 23, 18,
    np.where(base_dados['des_uf_rg_gh31'] == 24, 19,
    np.where(base_dados['des_uf_rg_gh31'] == 25, 20,
    np.where(base_dados['des_uf_rg_gh31'] == 27, 21,
    np.where(base_dados['des_uf_rg_gh31'] == 29, 22,
    np.where(base_dados['des_uf_rg_gh31'] == 30, 23,
    np.where(base_dados['des_uf_rg_gh31'] == 31, 24,
    0)))))))))))))))))))))))))
         
base_dados['des_uf_rg_gh33'] = np.where(base_dados['des_uf_rg_gh32'] == 0, 0,
    np.where(base_dados['des_uf_rg_gh32'] == 1, 1,
    np.where(base_dados['des_uf_rg_gh32'] == 2, 2,
    np.where(base_dados['des_uf_rg_gh32'] == 3, 3,
    np.where(base_dados['des_uf_rg_gh32'] == 4, 4,
    np.where(base_dados['des_uf_rg_gh32'] == 5, 5,
    np.where(base_dados['des_uf_rg_gh32'] == 6, 6,
    np.where(base_dados['des_uf_rg_gh32'] == 7, 7,
    np.where(base_dados['des_uf_rg_gh32'] == 8, 8,
    np.where(base_dados['des_uf_rg_gh32'] == 9, 9,
    np.where(base_dados['des_uf_rg_gh32'] == 10, 10,
    np.where(base_dados['des_uf_rg_gh32'] == 11, 11,
    np.where(base_dados['des_uf_rg_gh32'] == 12, 12,
    np.where(base_dados['des_uf_rg_gh32'] == 13, 13,
    np.where(base_dados['des_uf_rg_gh32'] == 14, 14,
    np.where(base_dados['des_uf_rg_gh32'] == 15, 15,
    np.where(base_dados['des_uf_rg_gh32'] == 16, 16,
    np.where(base_dados['des_uf_rg_gh32'] == 17, 17,
    np.where(base_dados['des_uf_rg_gh32'] == 18, 18,
    np.where(base_dados['des_uf_rg_gh32'] == 19, 19,
    np.where(base_dados['des_uf_rg_gh32'] == 20, 20,
    np.where(base_dados['des_uf_rg_gh32'] == 21, 21,
    np.where(base_dados['des_uf_rg_gh32'] == 22, 22,
    np.where(base_dados['des_uf_rg_gh32'] == 23, 23,
    np.where(base_dados['des_uf_rg_gh32'] == 24, 24,
    0)))))))))))))))))))))))))
         
base_dados['des_uf_rg_gh34'] = np.where(base_dados['des_uf_rg_gh33'] == 0, 0,
    np.where(base_dados['des_uf_rg_gh33'] == 1, 21,
    np.where(base_dados['des_uf_rg_gh33'] == 2, 12,
    np.where(base_dados['des_uf_rg_gh33'] == 3, 12,
    np.where(base_dados['des_uf_rg_gh33'] == 4, 4,
    np.where(base_dados['des_uf_rg_gh33'] == 5, 12,
    np.where(base_dados['des_uf_rg_gh33'] == 6, 18,
    np.where(base_dados['des_uf_rg_gh33'] == 7, 12,
    np.where(base_dados['des_uf_rg_gh33'] == 8, 12,
    np.where(base_dados['des_uf_rg_gh33'] == 9, 12,
    np.where(base_dados['des_uf_rg_gh33'] == 10, 10,
    np.where(base_dados['des_uf_rg_gh33'] == 11, 12,
    np.where(base_dados['des_uf_rg_gh33'] == 12, 12,
    np.where(base_dados['des_uf_rg_gh33'] == 13, 12,
    np.where(base_dados['des_uf_rg_gh33'] == 14, 12,
    np.where(base_dados['des_uf_rg_gh33'] == 15, 12,
    np.where(base_dados['des_uf_rg_gh33'] == 16, 21,
    np.where(base_dados['des_uf_rg_gh33'] == 17, 12,
    np.where(base_dados['des_uf_rg_gh33'] == 18, 18,
    np.where(base_dados['des_uf_rg_gh33'] == 19, 12,
    np.where(base_dados['des_uf_rg_gh33'] == 20, 18,
    np.where(base_dados['des_uf_rg_gh33'] == 21, 21,
    np.where(base_dados['des_uf_rg_gh33'] == 22, 18,
    np.where(base_dados['des_uf_rg_gh33'] == 23, 23,
    np.where(base_dados['des_uf_rg_gh33'] == 24, 12,
    0)))))))))))))))))))))))))
         
base_dados['des_uf_rg_gh35'] = np.where(base_dados['des_uf_rg_gh34'] == 0, 0,
    np.where(base_dados['des_uf_rg_gh34'] == 4, 1,
    np.where(base_dados['des_uf_rg_gh34'] == 10, 2,
    np.where(base_dados['des_uf_rg_gh34'] == 12, 3,
    np.where(base_dados['des_uf_rg_gh34'] == 18, 4,
    np.where(base_dados['des_uf_rg_gh34'] == 21, 5,
    np.where(base_dados['des_uf_rg_gh34'] == 23, 6,
    0)))))))
         
base_dados['des_uf_rg_gh36'] = np.where(base_dados['des_uf_rg_gh35'] == 0, 6,
    np.where(base_dados['des_uf_rg_gh35'] == 1, 1,
    np.where(base_dados['des_uf_rg_gh35'] == 2, 3,
    np.where(base_dados['des_uf_rg_gh35'] == 3, 0,
    np.where(base_dados['des_uf_rg_gh35'] == 4, 1,
    np.where(base_dados['des_uf_rg_gh35'] == 5, 3,
    np.where(base_dados['des_uf_rg_gh35'] == 6, 5,
    0)))))))
         
base_dados['des_uf_rg_gh37'] = np.where(base_dados['des_uf_rg_gh36'] == 0, 0,
    np.where(base_dados['des_uf_rg_gh36'] == 1, 1,
    np.where(base_dados['des_uf_rg_gh36'] == 3, 2,
    np.where(base_dados['des_uf_rg_gh36'] == 5, 3,
    np.where(base_dados['des_uf_rg_gh36'] == 6, 4,
    0)))))
         
base_dados['des_uf_rg_gh38'] = np.where(base_dados['des_uf_rg_gh37'] == 0, 0,
    np.where(base_dados['des_uf_rg_gh37'] == 1, 1,
    np.where(base_dados['des_uf_rg_gh37'] == 2, 2,
    np.where(base_dados['des_uf_rg_gh37'] == 3, 3,
    np.where(base_dados['des_uf_rg_gh37'] == 4, 4,
    0)))))
         
         
         
         
         
         
         
         
         
base_dados['tip_ender_corre_gh30'] = np.where(base_dados['tip_ender_corre'] == '-3', 0,
    np.where(base_dados['tip_ender_corre'] == 'C', 1,
    np.where(base_dados['tip_ender_corre'] == 'R', 2,
    0)))
         
base_dados['tip_ender_corre_gh31'] = np.where(base_dados['tip_ender_corre_gh30'] == 0, 0,
    np.where(base_dados['tip_ender_corre_gh30'] == 1, 1,
    np.where(base_dados['tip_ender_corre_gh30'] == 2, 2,
    0)))
         
base_dados['tip_ender_corre_gh32'] = np.where(base_dados['tip_ender_corre_gh31'] == 0, 0,
    np.where(base_dados['tip_ender_corre_gh31'] == 1, 1,
    np.where(base_dados['tip_ender_corre_gh31'] == 2, 2,
    0)))
         
base_dados['tip_ender_corre_gh33'] = np.where(base_dados['tip_ender_corre_gh32'] == 0, 0,
    np.where(base_dados['tip_ender_corre_gh32'] == 1, 1,
    np.where(base_dados['tip_ender_corre_gh32'] == 2, 2,
    0)))
         
base_dados['tip_ender_corre_gh34'] = np.where(base_dados['tip_ender_corre_gh33'] == 0, 0,
    np.where(base_dados['tip_ender_corre_gh33'] == 1, 0,
    np.where(base_dados['tip_ender_corre_gh33'] == 2, 2,
    0)))
         
base_dados['tip_ender_corre_gh35'] = np.where(base_dados['tip_ender_corre_gh34'] == 0, 0,
    np.where(base_dados['tip_ender_corre_gh34'] == 2, 1,
    0))
                                              
base_dados['tip_ender_corre_gh36'] = np.where(base_dados['tip_ender_corre_gh35'] == 0, 1,
    np.where(base_dados['tip_ender_corre_gh35'] == 1, 0,
    0))
                                              
base_dados['tip_ender_corre_gh37'] = np.where(base_dados['tip_ender_corre_gh36'] == 0, 0,
    np.where(base_dados['tip_ender_corre_gh36'] == 1, 1,
    0))
                                              
base_dados['tip_ender_corre_gh38'] = np.where(base_dados['tip_ender_corre_gh37'] == 0, 0,
    np.where(base_dados['tip_ender_corre_gh37'] == 1, 1,
    0))
                                              
                                              
                                              
                                              
                                              
                                              
                                              
base_dados['cod_filia_gh30'] = np.where(base_dados['cod_filia'] == -3.0, 0,
    np.where(base_dados['cod_filia'] == 1.0, 1,
    np.where(base_dados['cod_filia'] == 2.0, 2,
    np.where(base_dados['cod_filia'] == 4.0, 3,
    np.where(base_dados['cod_filia'] == 5.0, 4,
    np.where(base_dados['cod_filia'] == 6.0, 5,
    np.where(base_dados['cod_filia'] == 8.0, 6,
    np.where(base_dados['cod_filia'] == 27.0, 7,
    np.where(base_dados['cod_filia'] == 30.0, 8,
    np.where(base_dados['cod_filia'] == 31.0, 9,
    np.where(base_dados['cod_filia'] == 32.0, 10,
    np.where(base_dados['cod_filia'] == 33.0, 11,
    np.where(base_dados['cod_filia'] == 35.0, 12,
    np.where(base_dados['cod_filia'] == 38.0, 13,
    np.where(base_dados['cod_filia'] == 39.0, 14,
    np.where(base_dados['cod_filia'] == 41.0, 15,
    np.where(base_dados['cod_filia'] == 42.0, 16,
    np.where(base_dados['cod_filia'] == 43.0, 17,
    np.where(base_dados['cod_filia'] == 46.0, 18,
    np.where(base_dados['cod_filia'] == 47.0, 19,
    np.where(base_dados['cod_filia'] == 51.0, 20,
    np.where(base_dados['cod_filia'] == 56.0, 21,
    np.where(base_dados['cod_filia'] == 57.0, 22,
    np.where(base_dados['cod_filia'] == 58.0, 23,
    np.where(base_dados['cod_filia'] == 61.0, 24,
    np.where(base_dados['cod_filia'] == 88.0, 25,
    0))))))))))))))))))))))))))
         
base_dados['cod_filia_gh31'] = np.where(base_dados['cod_filia_gh30'] == 0, 0,
    np.where(base_dados['cod_filia_gh30'] == 1, 1,
    np.where(base_dados['cod_filia_gh30'] == 2, 2,
    np.where(base_dados['cod_filia_gh30'] == 3, 3,
    np.where(base_dados['cod_filia_gh30'] == 4, 4,
    np.where(base_dados['cod_filia_gh30'] == 5, 5,
    np.where(base_dados['cod_filia_gh30'] == 6, 6,
    np.where(base_dados['cod_filia_gh30'] == 7, 7,
    np.where(base_dados['cod_filia_gh30'] == 8, 8,
    np.where(base_dados['cod_filia_gh30'] == 9, 8,
    np.where(base_dados['cod_filia_gh30'] == 10, 8,
    np.where(base_dados['cod_filia_gh30'] == 11, 11,
    np.where(base_dados['cod_filia_gh30'] == 12, 12,
    np.where(base_dados['cod_filia_gh30'] == 13, 13,
    np.where(base_dados['cod_filia_gh30'] == 14, 13,
    np.where(base_dados['cod_filia_gh30'] == 15, 13,
    np.where(base_dados['cod_filia_gh30'] == 16, 16,
    np.where(base_dados['cod_filia_gh30'] == 17, 17,
    np.where(base_dados['cod_filia_gh30'] == 18, 18,
    np.where(base_dados['cod_filia_gh30'] == 19, 19,
    np.where(base_dados['cod_filia_gh30'] == 20, 20,
    np.where(base_dados['cod_filia_gh30'] == 21, 21,
    np.where(base_dados['cod_filia_gh30'] == 22, 21,
    np.where(base_dados['cod_filia_gh30'] == 23, 21,
    np.where(base_dados['cod_filia_gh30'] == 24, 21,
    np.where(base_dados['cod_filia_gh30'] == 25, 21,
    0))))))))))))))))))))))))))
         
base_dados['cod_filia_gh32'] = np.where(base_dados['cod_filia_gh31'] == 0, 0,
    np.where(base_dados['cod_filia_gh31'] == 1, 1,
    np.where(base_dados['cod_filia_gh31'] == 2, 2,
    np.where(base_dados['cod_filia_gh31'] == 3, 3,
    np.where(base_dados['cod_filia_gh31'] == 4, 4,
    np.where(base_dados['cod_filia_gh31'] == 5, 5,
    np.where(base_dados['cod_filia_gh31'] == 6, 6,
    np.where(base_dados['cod_filia_gh31'] == 7, 7,
    np.where(base_dados['cod_filia_gh31'] == 8, 8,
    np.where(base_dados['cod_filia_gh31'] == 11, 9,
    np.where(base_dados['cod_filia_gh31'] == 12, 10,
    np.where(base_dados['cod_filia_gh31'] == 13, 11,
    np.where(base_dados['cod_filia_gh31'] == 16, 12,
    np.where(base_dados['cod_filia_gh31'] == 17, 13,
    np.where(base_dados['cod_filia_gh31'] == 18, 14,
    np.where(base_dados['cod_filia_gh31'] == 19, 15,
    np.where(base_dados['cod_filia_gh31'] == 20, 16,
    np.where(base_dados['cod_filia_gh31'] == 21, 17,
    0))))))))))))))))))
         
base_dados['cod_filia_gh33'] = np.where(base_dados['cod_filia_gh32'] == 0, 0,
    np.where(base_dados['cod_filia_gh32'] == 1, 1,
    np.where(base_dados['cod_filia_gh32'] == 2, 2,
    np.where(base_dados['cod_filia_gh32'] == 3, 3,
    np.where(base_dados['cod_filia_gh32'] == 4, 4,
    np.where(base_dados['cod_filia_gh32'] == 5, 5,
    np.where(base_dados['cod_filia_gh32'] == 6, 6,
    np.where(base_dados['cod_filia_gh32'] == 7, 7,
    np.where(base_dados['cod_filia_gh32'] == 8, 8,
    np.where(base_dados['cod_filia_gh32'] == 9, 9,
    np.where(base_dados['cod_filia_gh32'] == 10, 10,
    np.where(base_dados['cod_filia_gh32'] == 11, 11,
    np.where(base_dados['cod_filia_gh32'] == 12, 12,
    np.where(base_dados['cod_filia_gh32'] == 13, 13,
    np.where(base_dados['cod_filia_gh32'] == 14, 14,
    np.where(base_dados['cod_filia_gh32'] == 15, 15,
    np.where(base_dados['cod_filia_gh32'] == 16, 16,
    np.where(base_dados['cod_filia_gh32'] == 17, 17,
    0))))))))))))))))))
         
base_dados['cod_filia_gh34'] = np.where(base_dados['cod_filia_gh33'] == 0, 0,
    np.where(base_dados['cod_filia_gh33'] == 1, 1,
    np.where(base_dados['cod_filia_gh33'] == 2, 4,
    np.where(base_dados['cod_filia_gh33'] == 3, 4,
    np.where(base_dados['cod_filia_gh33'] == 4, 4,
    np.where(base_dados['cod_filia_gh33'] == 5, 4,
    np.where(base_dados['cod_filia_gh33'] == 6, 4,
    np.where(base_dados['cod_filia_gh33'] == 7, 4,
    np.where(base_dados['cod_filia_gh33'] == 8, 17,
    np.where(base_dados['cod_filia_gh33'] == 9, 4,
    np.where(base_dados['cod_filia_gh33'] == 10, 4,
    np.where(base_dados['cod_filia_gh33'] == 11, 17,
    np.where(base_dados['cod_filia_gh33'] == 12, 4,
    np.where(base_dados['cod_filia_gh33'] == 13, 4,
    np.where(base_dados['cod_filia_gh33'] == 14, 0,
    np.where(base_dados['cod_filia_gh33'] == 15, 4,
    np.where(base_dados['cod_filia_gh33'] == 16, 4,
    np.where(base_dados['cod_filia_gh33'] == 17, 17,
    0))))))))))))))))))
         
base_dados['cod_filia_gh35'] = np.where(base_dados['cod_filia_gh34'] == 0, 0,
    np.where(base_dados['cod_filia_gh34'] == 1, 1,
    np.where(base_dados['cod_filia_gh34'] == 4, 2,
    np.where(base_dados['cod_filia_gh34'] == 17, 3,
    0))))
         
base_dados['cod_filia_gh36'] = np.where(base_dados['cod_filia_gh35'] == 0, 3,
    np.where(base_dados['cod_filia_gh35'] == 1, 2,
    np.where(base_dados['cod_filia_gh35'] == 2, 1,
    np.where(base_dados['cod_filia_gh35'] == 3, 0,
    0))))
         
base_dados['cod_filia_gh37'] = np.where(base_dados['cod_filia_gh36'] == 0, 1,
    np.where(base_dados['cod_filia_gh36'] == 1, 1,
    np.where(base_dados['cod_filia_gh36'] == 2, 2,
    np.where(base_dados['cod_filia_gh36'] == 3, 3,
    1))))
         
base_dados['cod_filia_gh38'] = np.where(base_dados['cod_filia_gh37'] == 1, 0,
    np.where(base_dados['cod_filia_gh37'] == 2, 1,
    np.where(base_dados['cod_filia_gh37'] == 3, 2,
    0)))
         


# COMMAND ----------

# MAGIC %md
# MAGIC ## Gerando variáveis numéricas contínuas

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parte 1 de 2

# COMMAND ----------


  
              
         
base_dados['val_compr__T'] = np.tan(base_dados['val_compr'])
np.where(base_dados['val_compr__T'] == 0, -1, base_dados['val_compr__T'])
base_dados['val_compr__T'] = base_dados['val_compr__T'].fillna(-2)

base_dados['val_compr__T__p_13'] = np.where(np.bitwise_and(base_dados['val_compr__T'] >= -2177.9343142453763, base_dados['val_compr__T'] <= 0.399536683231318), 10.0,
    np.where(np.bitwise_and(base_dados['val_compr__T'] > 0.399536683231318, base_dados['val_compr__T'] <= 2.3636955544197176), 11.0,
    np.where(base_dados['val_compr__T'] > 2.3636955544197176, 12.0,
     -2)))

base_dados['val_compr__T__p_13_g_1_1'] = np.where(base_dados['val_compr__T__p_13'] == -2.0, 1,
    np.where(base_dados['val_compr__T__p_13'] == 10.0, 1,
    np.where(base_dados['val_compr__T__p_13'] == 11.0, 0,
    np.where(base_dados['val_compr__T__p_13'] == 12.0, 0,
     0))))

base_dados['val_compr__T__p_13_g_1_2'] = np.where(base_dados['val_compr__T__p_13_g_1_1'] == 0, 1,
    np.where(base_dados['val_compr__T__p_13_g_1_1'] == 1, 0,
     0))

base_dados['val_compr__T__p_13_g_1'] = np.where(base_dados['val_compr__T__p_13_g_1_2'] == 0, 0,
    np.where(base_dados['val_compr__T__p_13_g_1_2'] == 1, 1,
     0))
                                                
                                                
                                                
                                                
                                                
                                                
base_dados['val_compr__T'] = np.tan(base_dados['val_compr'])
np.where(base_dados['val_compr__T'] == 0, -1, base_dados['val_compr__T'])
base_dados['val_compr__T'] = base_dados['val_compr__T'].fillna(-2)

base_dados['val_compr__T__pe_10'] = np.where(np.bitwise_and(base_dados['val_compr__T'] >= -2177.9343142453763, base_dados['val_compr__T'] <= 2.0077700366381706), 0.0,
    np.where(np.bitwise_and(base_dados['val_compr__T'] > 2.0077700366381706, base_dados['val_compr__T'] <= 4.003671335861064), 1.0,
    np.where(np.bitwise_and(base_dados['val_compr__T'] > 4.003671335861064, base_dados['val_compr__T'] <= 5.969092544272484), 2.0,
    np.where(np.bitwise_and(base_dados['val_compr__T'] > 5.969092544272484, base_dados['val_compr__T'] <= 8.031337491254321), 3.0,
    np.where(np.bitwise_and(base_dados['val_compr__T'] > 8.031337491254321, base_dados['val_compr__T'] <= 10.020703019688455), 4.0,
    np.where(np.bitwise_and(base_dados['val_compr__T'] > 10.020703019688455, base_dados['val_compr__T'] <= 11.92079585548991), 5.0,
    np.where(np.bitwise_and(base_dados['val_compr__T'] > 11.92079585548991, base_dados['val_compr__T'] <= 14.042480329550676), 6.0,
    np.where(np.bitwise_and(base_dados['val_compr__T'] > 14.042480329550676, base_dados['val_compr__T'] <= 15.888646604432106), 7.0,
    np.where(np.bitwise_and(base_dados['val_compr__T'] > 15.888646604432106, base_dados['val_compr__T'] <= 18.019932891392386), 8.0,
    np.where(base_dados['val_compr__T'] > 18.019932891392386, 9.0,
     -2))))))))))

base_dados['val_compr__T__pe_10_g_1_1'] = np.where(base_dados['val_compr__T__pe_10'] == -2.0, 1,
    np.where(base_dados['val_compr__T__pe_10'] == 0.0, 0,
    np.where(base_dados['val_compr__T__pe_10'] == 1.0, 1,
    np.where(base_dados['val_compr__T__pe_10'] == 2.0, 1,
    np.where(base_dados['val_compr__T__pe_10'] == 3.0, 1,
    np.where(base_dados['val_compr__T__pe_10'] == 4.0, 1,
    np.where(base_dados['val_compr__T__pe_10'] == 5.0, 1,
    np.where(base_dados['val_compr__T__pe_10'] == 6.0, 1,
    np.where(base_dados['val_compr__T__pe_10'] == 7.0, 1,
    np.where(base_dados['val_compr__T__pe_10'] == 8.0, 1,
    np.where(base_dados['val_compr__T__pe_10'] == 9.0, 1,
     0)))))))))))

base_dados['val_compr__T__pe_10_g_1_2'] = np.where(base_dados['val_compr__T__pe_10_g_1_1'] == 0, 1,
    np.where(base_dados['val_compr__T__pe_10_g_1_1'] == 1, 0,
     0))

base_dados['val_compr__T__pe_10_g_1'] = np.where(base_dados['val_compr__T__pe_10_g_1_2'] == 0, 0,
    np.where(base_dados['val_compr__T__pe_10_g_1_2'] == 1, 1,
     0))
                                                 
                                                 
                                                 
                                                 
                                                 
base_dados['qtd_prest__pe_6'] = np.where(base_dados['qtd_prest'] <= 13.0, 0.0,
    np.where(np.bitwise_and(base_dados['qtd_prest'] > 13.0, base_dados['qtd_prest'] <= 27.0), 1.0,
    np.where(np.bitwise_and(base_dados['qtd_prest'] > 27.0, base_dados['qtd_prest'] <= 41.0), 2.0,
    np.where(np.bitwise_and(base_dados['qtd_prest'] > 41.0, base_dados['qtd_prest'] <= 55.0), 3.0,
    np.where(np.bitwise_and(base_dados['qtd_prest'] > 55.0, base_dados['qtd_prest'] <= 69.0), 4.0,
    np.where(base_dados['qtd_prest'] > 69.0, 5.0,
     -2))))))

base_dados['qtd_prest__pe_6_g_1_1'] = np.where(base_dados['qtd_prest__pe_6'] == -2.0, 1,
    np.where(base_dados['qtd_prest__pe_6'] == 0.0, 0,
    np.where(base_dados['qtd_prest__pe_6'] == 1.0, 1,
    np.where(base_dados['qtd_prest__pe_6'] == 2.0, 1,
    np.where(base_dados['qtd_prest__pe_6'] == 3.0, 2,
    np.where(base_dados['qtd_prest__pe_6'] == 4.0, 2,
    np.where(base_dados['qtd_prest__pe_6'] == 5.0, 1,
     0)))))))

base_dados['qtd_prest__pe_6_g_1_2'] = np.where(base_dados['qtd_prest__pe_6_g_1_1'] == 0, 2,
    np.where(base_dados['qtd_prest__pe_6_g_1_1'] == 1, 1,
    np.where(base_dados['qtd_prest__pe_6_g_1_1'] == 2, 0,
     0)))

base_dados['qtd_prest__pe_6_g_1'] = np.where(base_dados['qtd_prest__pe_6_g_1_2'] == 0, 0,
    np.where(base_dados['qtd_prest__pe_6_g_1_2'] == 1, 1,
    np.where(base_dados['qtd_prest__pe_6_g_1_2'] == 2, 2,
     0)))
         
         
         
         
         
         
         
base_dados['qtd_prest__S'] = np.sin(base_dados['qtd_prest'])
np.where(base_dados['qtd_prest__S'] == 0, -1, base_dados['qtd_prest__S'])
base_dados['qtd_prest__S'] = base_dados['qtd_prest__S'].fillna(-2)

base_dados['qtd_prest__S__p_2'] = np.where(np.bitwise_and(base_dados['qtd_prest__S'] >= -2.0, base_dados['qtd_prest__S'] <= 0.6502878401571168), 0.0,
    np.where(base_dados['qtd_prest__S'] > 0.6502878401571168, 1.0,
     -2))
                                           
base_dados['qtd_prest__S__p_2_g_1_1'] = np.where(base_dados['qtd_prest__S__p_2'] == -2.0, 2,
    np.where(base_dados['qtd_prest__S__p_2'] == 0.0, 1,
    np.where(base_dados['qtd_prest__S__p_2'] == 1.0, 0,
     0)))
         
base_dados['qtd_prest__S__p_2_g_1_2'] = np.where(base_dados['qtd_prest__S__p_2_g_1_1'] == 0, 2,
    np.where(base_dados['qtd_prest__S__p_2_g_1_1'] == 1, 1,
    np.where(base_dados['qtd_prest__S__p_2_g_1_1'] == 2, 0,
     0)))
                                           
base_dados['qtd_prest__S__p_2_g_1'] = np.where(base_dados['qtd_prest__S__p_2_g_1_2'] == 0, 0,
    np.where(base_dados['qtd_prest__S__p_2_g_1_2'] == 1, 1,
    np.where(base_dados['qtd_prest__S__p_2_g_1_2'] == 2, 2,
     0)))
         
         
         
         
         
         
base_dados['raiz_cep__pe_4'] = np.where(base_dados['raiz_cep'] == 0 , -1.0,
    np.where(np.bitwise_and(base_dados['raiz_cep'] > 0.0, base_dados['raiz_cep'] <= 24350.0), 0.0,
    np.where(np.bitwise_and(base_dados['raiz_cep'] > 24350.0, base_dados['raiz_cep'] <= 48700.0), 1.0,
    np.where(np.bitwise_and(base_dados['raiz_cep'] > 48700.0, base_dados['raiz_cep'] <= 73062.0), 2.0,
    np.where(base_dados['raiz_cep'] > 73062.0, 3.0,
     -2)))))
                                           
base_dados['raiz_cep__pe_4_g_1_1'] = np.where(base_dados['raiz_cep__pe_4'] == -2.0, 1,
    np.where(base_dados['raiz_cep__pe_4'] == -1.0, 1,
    np.where(base_dados['raiz_cep__pe_4'] == 0.0, 1,
    np.where(base_dados['raiz_cep__pe_4'] == 1.0, 1,
    np.where(base_dados['raiz_cep__pe_4'] == 2.0, 0,
    np.where(base_dados['raiz_cep__pe_4'] == 3.0, 1,
     0))))))
                                           
base_dados['raiz_cep__pe_4_g_1_2'] = np.where(base_dados['raiz_cep__pe_4_g_1_1'] == 0, 0,
    np.where(base_dados['raiz_cep__pe_4_g_1_1'] == 1, 1,
     0))
                                           
base_dados['raiz_cep__pe_4_g_1'] = np.where(base_dados['raiz_cep__pe_4_g_1_2'] == 0, 0,
    np.where(base_dados['raiz_cep__pe_4_g_1_2'] == 1, 1,
     0))
                                            
                                            
                                            
                                            
                                            
                                            
                                            
base_dados['raiz_cep__L'] = np.log(base_dados['raiz_cep'])
np.where(base_dados['raiz_cep__L'] == 0, -1, base_dados['raiz_cep__L'])
base_dados['raiz_cep__L'] = base_dados['raiz_cep__L'].fillna(-2)
                                           
base_dados['raiz_cep__L__p_13'] = np.where(np.bitwise_and(base_dados['raiz_cep__L'] >= -1.0, base_dados['raiz_cep__L'] <= 8.872767529910936), 0.0,
    np.where(np.bitwise_and(base_dados['raiz_cep__L'] > 8.872767529910936, base_dados['raiz_cep__L'] <= 9.503606822721864), 1.0,
    np.where(np.bitwise_and(base_dados['raiz_cep__L'] > 9.503606822721864, base_dados['raiz_cep__L'] <= 9.876630098366245), 2.0,
    np.where(np.bitwise_and(base_dados['raiz_cep__L'] > 9.876630098366245, base_dados['raiz_cep__L'] <= 10.10528488583074), 3.0,
    np.where(np.bitwise_and(base_dados['raiz_cep__L'] > 10.10528488583074, base_dados['raiz_cep__L'] <= 10.323512811975622), 4.0,
    np.where(np.bitwise_and(base_dados['raiz_cep__L'] > 10.323512811975622, base_dados['raiz_cep__L'] <= 10.521911186900137), 5.0,
    np.where(np.bitwise_and(base_dados['raiz_cep__L'] > 10.521911186900137, base_dados['raiz_cep__L'] <= 10.762572810639504), 6.0,
    np.where(np.bitwise_and(base_dados['raiz_cep__L'] > 10.762572810639504, base_dados['raiz_cep__L'] <= 10.956195509013263), 7.0,
    np.where(np.bitwise_and(base_dados['raiz_cep__L'] > 10.956195509013263, base_dados['raiz_cep__L'] <= 11.085214747914744), 8.0,
    np.where(np.bitwise_and(base_dados['raiz_cep__L'] > 11.085214747914744, base_dados['raiz_cep__L'] <= 11.311176410510175), 10.0,
    np.where(base_dados['raiz_cep__L'] > 11.311176410510175, 11.0,
     -2)))))))))))
                                           
base_dados['raiz_cep__L__p_13_g_1_1'] = np.where(base_dados['raiz_cep__L__p_13'] == -2.0, 2,
    np.where(base_dados['raiz_cep__L__p_13'] == 0.0, 2,
    np.where(base_dados['raiz_cep__L__p_13'] == 1.0, 2,
    np.where(base_dados['raiz_cep__L__p_13'] == 2.0, 2,
    np.where(base_dados['raiz_cep__L__p_13'] == 3.0, 0,
    np.where(base_dados['raiz_cep__L__p_13'] == 4.0, 2,
    np.where(base_dados['raiz_cep__L__p_13'] == 5.0, 2,
    np.where(base_dados['raiz_cep__L__p_13'] == 6.0, 0,
    np.where(base_dados['raiz_cep__L__p_13'] == 7.0, 2,
    np.where(base_dados['raiz_cep__L__p_13'] == 8.0, 0,
    np.where(base_dados['raiz_cep__L__p_13'] == 10.0, 1,
    np.where(base_dados['raiz_cep__L__p_13'] == 11.0, 0,
     0))))))))))))
                                           
base_dados['raiz_cep__L__p_13_g_1_2'] = np.where(base_dados['raiz_cep__L__p_13_g_1_1'] == 0, 1,
    np.where(base_dados['raiz_cep__L__p_13_g_1_1'] == 1, 0,
    np.where(base_dados['raiz_cep__L__p_13_g_1_1'] == 2, 2,
     0)))
                                           
base_dados['raiz_cep__L__p_13_g_1'] = np.where(base_dados['raiz_cep__L__p_13_g_1_2'] == 0, 0,
    np.where(base_dados['raiz_cep__L__p_13_g_1_2'] == 1, 1,
    np.where(base_dados['raiz_cep__L__p_13_g_1_2'] == 2, 2,
     0)))
         
         
         
         
         
         
         
base_dados['des_fones_resid__L'] = np.log(base_dados['des_fones_resid'])
np.where(base_dados['des_fones_resid__L'] == 0, -1, base_dados['des_fones_resid__L'])
base_dados['des_fones_resid__L'] = base_dados['des_fones_resid__L'].fillna(-2)
                                           
base_dados['des_fones_resid__L__p_3'] = np.where(np.bitwise_and(base_dados['des_fones_resid__L'] >= -2.0, base_dados['des_fones_resid__L'] <= 21.476815190582904), 1.0,
    np.where(base_dados['des_fones_resid__L'] > 21.476815190582904, 2.0,
     -2))
                                           
base_dados['des_fones_resid__L__p_3_g_1_1'] = np.where(base_dados['des_fones_resid__L__p_3'] == -2.0, 0,
    np.where(base_dados['des_fones_resid__L__p_3'] == 1.0, 0,
    np.where(base_dados['des_fones_resid__L__p_3'] == 2.0, 1,
     0)))
                                           
base_dados['des_fones_resid__L__p_3_g_1_2'] = np.where(base_dados['des_fones_resid__L__p_3_g_1_1'] == 0, 1,
    np.where(base_dados['des_fones_resid__L__p_3_g_1_1'] == 1, 0,
     0))
                                           
base_dados['des_fones_resid__L__p_3_g_1'] = np.where(base_dados['des_fones_resid__L__p_3_g_1_2'] == 0, 0,
    np.where(base_dados['des_fones_resid__L__p_3_g_1_2'] == 1, 1,
     0))
                                                     
                                                     
                                                     
                                                     
                                                     
                                                     
base_dados['des_fones_resid__L'] = np.log(base_dados['des_fones_resid'])
np.where(base_dados['des_fones_resid__L'] == 0, -1, base_dados['des_fones_resid__L'])
base_dados['des_fones_resid__L'] = base_dados['des_fones_resid__L'].fillna(-2)
                                           
base_dados['des_fones_resid__L__pe_6'] = np.where(np.bitwise_and(base_dados['des_fones_resid__L'] >= -2.0, base_dados['des_fones_resid__L'] <= 22.631388226981954), 4.0,
    np.where(base_dados['des_fones_resid__L'] > 22.631388226981954, 5.0,
     -2))
                                           
base_dados['des_fones_resid__L__pe_6_g_1_1'] = np.where(base_dados['des_fones_resid__L__pe_6'] == -2.0, 0,
    np.where(base_dados['des_fones_resid__L__pe_6'] == 4.0, 1,
    np.where(base_dados['des_fones_resid__L__pe_6'] == 5.0, 2,
     0)))
                                           
base_dados['des_fones_resid__L__pe_6_g_1_2'] = np.where(base_dados['des_fones_resid__L__pe_6_g_1_1'] == 0, 2,
    np.where(base_dados['des_fones_resid__L__pe_6_g_1_1'] == 1, 1,
    np.where(base_dados['des_fones_resid__L__pe_6_g_1_1'] == 2, 0,
     0)))
                                           
base_dados['des_fones_resid__L__pe_6_g_1'] = np.where(base_dados['des_fones_resid__L__pe_6_g_1_2'] == 0, 0,
    np.where(base_dados['des_fones_resid__L__pe_6_g_1_2'] == 1, 1,
    np.where(base_dados['des_fones_resid__L__pe_6_g_1_2'] == 2, 2,
     0)))
         
         
         
         
         
         
base_dados['cod_profi_clien__p_2'] = np.where(base_dados['cod_profi_clien'] <= 21.0, 0.0,
    np.where(base_dados['cod_profi_clien'] > 21.0, 1.0,
     -2))
                                           
base_dados['cod_profi_clien__p_2_g_1_1'] = np.where(base_dados['cod_profi_clien__p_2'] == -2.0, 0,
    np.where(base_dados['cod_profi_clien__p_2'] == 0.0, 1,
    np.where(base_dados['cod_profi_clien__p_2'] == 1.0, 1,
     0)))
                                           
base_dados['cod_profi_clien__p_2_g_1_2'] = np.where(base_dados['cod_profi_clien__p_2_g_1_1'] == 0, 1,
    np.where(base_dados['cod_profi_clien__p_2_g_1_1'] == 1, 0,
     0))
                                           
base_dados['cod_profi_clien__p_2_g_1'] = np.where(base_dados['cod_profi_clien__p_2_g_1_2'] == 0, 0,
    np.where(base_dados['cod_profi_clien__p_2_g_1_2'] == 1, 1,
     0))
                                                  
                                                  
                                                  
                                                  
                                                  
                                                  
base_dados['cod_profi_clien__T'] = np.tan(base_dados['cod_profi_clien'])
np.where(base_dados['cod_profi_clien__T'] == 0, -1, base_dados['cod_profi_clien__T'])
base_dados['cod_profi_clien__T'] = base_dados['cod_profi_clien__T'].fillna(-2)
                                           
base_dados['cod_profi_clien__T__p_34'] = np.where(np.bitwise_and(base_dados['cod_profi_clien__T'] >= -32.268575775934416, base_dados['cod_profi_clien__T'] <= 0.053158536832187686), 26.0,
    np.where(np.bitwise_and(base_dados['cod_profi_clien__T'] > 0.053158536832187686, base_dados['cod_profi_clien__T'] <= 0.5067526002248183), 32.0,
    np.where(base_dados['cod_profi_clien__T'] > 0.5067526002248183, 33.0,
     -2)))
                                           
base_dados['cod_profi_clien__T__p_34_g_1_1'] = np.where(base_dados['cod_profi_clien__T__p_34'] == -2.0, 0,
    np.where(base_dados['cod_profi_clien__T__p_34'] == 26.0, 1,
    np.where(base_dados['cod_profi_clien__T__p_34'] == 32.0, 1,
    np.where(base_dados['cod_profi_clien__T__p_34'] == 33.0, 1,
     0))))
                                           
base_dados['cod_profi_clien__T__p_34_g_1_2'] = np.where(base_dados['cod_profi_clien__T__p_34_g_1_1'] == 0, 0,
    np.where(base_dados['cod_profi_clien__T__p_34_g_1_1'] == 1, 1,
     0))
                                           
base_dados['cod_profi_clien__T__p_34_g_1'] = np.where(base_dados['cod_profi_clien__T__p_34_g_1_2'] == 0, 0,
    np.where(base_dados['cod_profi_clien__T__p_34_g_1_2'] == 1, 1,
     0))
                                                      
                                                      
                                                      
                                                      
                                                      
                                                      
base_dados['val_renda__p_3'] = np.where(base_dados['val_renda'] == 0 , -1.0,
    np.where(np.bitwise_and(base_dados['val_renda'] > 0.0, base_dados['val_renda'] <= 966.57), 1.0,
    np.where(base_dados['val_renda'] > 966.57, 2.0,
     -2)))
                                        
base_dados['val_renda__p_3_g_1_1'] = np.where(base_dados['val_renda__p_3'] == -2.0, 1,
    np.where(base_dados['val_renda__p_3'] == -1.0, 0,
    np.where(base_dados['val_renda__p_3'] == 1.0, 1,
    np.where(base_dados['val_renda__p_3'] == 2.0, 0,
     0))))
                                        
base_dados['val_renda__p_3_g_1_2'] = np.where(base_dados['val_renda__p_3_g_1_1'] == 0, 1,
    np.where(base_dados['val_renda__p_3_g_1_1'] == 1, 0,
     0))
                                        
base_dados['val_renda__p_3_g_1'] = np.where(base_dados['val_renda__p_3_g_1_2'] == 0, 0,
    np.where(base_dados['val_renda__p_3_g_1_2'] == 1, 1,
     0))
                                            
                                            
                                            
                                            
                                            
                                            
base_dados['val_renda__C'] = np.cos(base_dados['val_renda'])
np.where(base_dados['val_renda__C'] == 0, -1, base_dados['val_renda__C'])
base_dados['val_renda__C'] = base_dados['val_renda__C'].fillna(-2)

base_dados['val_renda__C__pe_5'] = np.where(np.bitwise_and(base_dados['val_renda__C'] >= -2.0, base_dados['val_renda__C'] <= 0.5991340617178956), 0.0,
    np.where(base_dados['val_renda__C'] > 0.5991340617178956, 1.0,
     -2))
                                            
base_dados['val_renda__C__pe_5_g_1_1'] = np.where(base_dados['val_renda__C__pe_5'] == -2.0, 0,
    np.where(base_dados['val_renda__C__pe_5'] == 0.0, 2,
    np.where(base_dados['val_renda__C__pe_5'] == 1.0, 1,
     0)))
                                        
base_dados['val_renda__C__pe_5_g_1_2'] = np.where(base_dados['val_renda__C__pe_5_g_1_1'] == 0, 2,
    np.where(base_dados['val_renda__C__pe_5_g_1_1'] == 1, 0,
    np.where(base_dados['val_renda__C__pe_5_g_1_1'] == 2, 1,
     0)))
                                        
base_dados['val_renda__C__pe_5_g_1'] = np.where(base_dados['val_renda__C__pe_5_g_1_2'] == 0, 0,
    np.where(base_dados['val_renda__C__pe_5_g_1_2'] == 1, 1,
    np.where(base_dados['val_renda__C__pe_5_g_1_2'] == 2, 2,
     0)))
         
         
         
         


# COMMAND ----------

# MAGIC %md
# MAGIC ### Parte 2 de 2

# COMMAND ----------




base_dados['val_compr__T__p_13_g_1_c1_6_1'] = np.where(np.bitwise_and(base_dados['val_compr__T__p_13_g_1'] == 0, base_dados['val_compr__T__pe_10_g_1'] == 0), 0,
    np.where(np.bitwise_and(base_dados['val_compr__T__p_13_g_1'] == 0, base_dados['val_compr__T__pe_10_g_1'] == 1), 1,
    np.where(np.bitwise_and(base_dados['val_compr__T__p_13_g_1'] == 1, base_dados['val_compr__T__pe_10_g_1'] == 0), 1,
    np.where(np.bitwise_and(base_dados['val_compr__T__p_13_g_1'] == 1, base_dados['val_compr__T__pe_10_g_1'] == 1), 1,
     0))))

base_dados['val_compr__T__p_13_g_1_c1_6_2'] = np.where(base_dados['val_compr__T__p_13_g_1_c1_6_1'] == 0, 0,
    np.where(base_dados['val_compr__T__p_13_g_1_c1_6_1'] == 1, 1,
    0))

base_dados['val_compr__T__p_13_g_1_c1_6'] = np.where(base_dados['val_compr__T__p_13_g_1_c1_6_2'] == 0, 0,
    np.where(base_dados['val_compr__T__p_13_g_1_c1_6_2'] == 1, 1,
     0))





base_dados['qtd_prest__pe_6_g_1_c1_11_1'] = np.where(np.bitwise_and(base_dados['qtd_prest__pe_6_g_1'] == 0, base_dados['qtd_prest__S__p_2_g_1'] == 0), 0,
    np.where(np.bitwise_and(base_dados['qtd_prest__pe_6_g_1'] == 0, base_dados['qtd_prest__S__p_2_g_1'] == 1), 0,
    np.where(np.bitwise_and(base_dados['qtd_prest__pe_6_g_1'] == 0, base_dados['qtd_prest__S__p_2_g_1'] == 2), 1,
    np.where(np.bitwise_and(base_dados['qtd_prest__pe_6_g_1'] == 1, base_dados['qtd_prest__S__p_2_g_1'] == 0), 1,
    np.where(np.bitwise_and(base_dados['qtd_prest__pe_6_g_1'] == 1, base_dados['qtd_prest__S__p_2_g_1'] == 1), 2,
    np.where(np.bitwise_and(base_dados['qtd_prest__pe_6_g_1'] == 1, base_dados['qtd_prest__S__p_2_g_1'] == 2), 2,
    np.where(np.bitwise_and(base_dados['qtd_prest__pe_6_g_1'] == 2, base_dados['qtd_prest__S__p_2_g_1'] == 0), 3,
    np.where(np.bitwise_and(base_dados['qtd_prest__pe_6_g_1'] == 2, base_dados['qtd_prest__S__p_2_g_1'] == 1), 3,
    np.where(np.bitwise_and(base_dados['qtd_prest__pe_6_g_1'] == 2, base_dados['qtd_prest__S__p_2_g_1'] == 2), 3,
     0)))))))))

base_dados['qtd_prest__pe_6_g_1_c1_11_2'] = np.where(base_dados['qtd_prest__pe_6_g_1_c1_11_1'] == 0, 0,
    np.where(base_dados['qtd_prest__pe_6_g_1_c1_11_1'] == 1, 1,
    np.where(base_dados['qtd_prest__pe_6_g_1_c1_11_1'] == 2, 2,
    np.where(base_dados['qtd_prest__pe_6_g_1_c1_11_1'] == 3, 3,
    0))))

base_dados['qtd_prest__pe_6_g_1_c1_11'] = np.where(base_dados['qtd_prest__pe_6_g_1_c1_11_2'] == 0, 0,
    np.where(base_dados['qtd_prest__pe_6_g_1_c1_11_2'] == 1, 1,
    np.where(base_dados['qtd_prest__pe_6_g_1_c1_11_2'] == 2, 2,
    np.where(base_dados['qtd_prest__pe_6_g_1_c1_11_2'] == 3, 3,
     0))))






base_dados['raiz_cep__L__p_13_g_1_c1_25_1'] = np.where(np.bitwise_and(base_dados['raiz_cep__pe_4_g_1'] == 0, base_dados['raiz_cep__L__p_13_g_1'] == 0), 0,
    np.where(np.bitwise_and(base_dados['raiz_cep__pe_4_g_1'] == 0, base_dados['raiz_cep__L__p_13_g_1'] == 1), 1,
    np.where(np.bitwise_and(base_dados['raiz_cep__pe_4_g_1'] == 0, base_dados['raiz_cep__L__p_13_g_1'] == 2), 2,
    np.where(np.bitwise_and(base_dados['raiz_cep__pe_4_g_1'] == 1, base_dados['raiz_cep__L__p_13_g_1'] == 0), 1,
    np.where(np.bitwise_and(base_dados['raiz_cep__pe_4_g_1'] == 1, base_dados['raiz_cep__L__p_13_g_1'] == 1), 2,
    np.where(np.bitwise_and(base_dados['raiz_cep__pe_4_g_1'] == 1, base_dados['raiz_cep__L__p_13_g_1'] == 2), 3,
     0))))))

base_dados['raiz_cep__L__p_13_g_1_c1_25_2'] = np.where(base_dados['raiz_cep__L__p_13_g_1_c1_25_1'] == 0, 0,
    np.where(base_dados['raiz_cep__L__p_13_g_1_c1_25_1'] == 1, 1,
    np.where(base_dados['raiz_cep__L__p_13_g_1_c1_25_1'] == 2, 2,
    np.where(base_dados['raiz_cep__L__p_13_g_1_c1_25_1'] == 3, 3,
    0))))

base_dados['raiz_cep__L__p_13_g_1_c1_25'] = np.where(base_dados['raiz_cep__L__p_13_g_1_c1_25_2'] == 0, 0,
    np.where(base_dados['raiz_cep__L__p_13_g_1_c1_25_2'] == 1, 1,
    np.where(base_dados['raiz_cep__L__p_13_g_1_c1_25_2'] == 2, 2,
    np.where(base_dados['raiz_cep__L__p_13_g_1_c1_25_2'] == 3, 3,
     0))))






base_dados['des_fones_resid__L__p_3_g_1_c1_3_1'] = np.where(np.bitwise_and(base_dados['des_fones_resid__L__p_3_g_1'] == 0, base_dados['des_fones_resid__L__pe_6_g_1'] == 0), 0,
    np.where(np.bitwise_and(base_dados['des_fones_resid__L__p_3_g_1'] == 0, base_dados['des_fones_resid__L__pe_6_g_1'] == 1), 1,
    np.where(np.bitwise_and(base_dados['des_fones_resid__L__p_3_g_1'] == 1, base_dados['des_fones_resid__L__pe_6_g_1'] == 1), 1,
    np.where(np.bitwise_and(base_dados['des_fones_resid__L__p_3_g_1'] == 1, base_dados['des_fones_resid__L__pe_6_g_1'] == 2), 2,
     0))))

base_dados['des_fones_resid__L__p_3_g_1_c1_3_2'] = np.where(base_dados['des_fones_resid__L__p_3_g_1_c1_3_1'] == 0, 0,
    np.where(base_dados['des_fones_resid__L__p_3_g_1_c1_3_1'] == 1, 1,
    np.where(base_dados['des_fones_resid__L__p_3_g_1_c1_3_1'] == 2, 2,
    0)))

base_dados['des_fones_resid__L__p_3_g_1_c1_3'] = np.where(base_dados['des_fones_resid__L__p_3_g_1_c1_3_2'] == 0, 0,
    np.where(base_dados['des_fones_resid__L__p_3_g_1_c1_3_2'] == 1, 1,
    np.where(base_dados['des_fones_resid__L__p_3_g_1_c1_3_2'] == 2, 2,
     0)))






base_dados['cod_profi_clien__T__p_34_g_1_c1_4_1'] = np.where(np.bitwise_and(base_dados['cod_profi_clien__p_2_g_1'] == 0, base_dados['cod_profi_clien__T__p_34_g_1'] == 0), 0,
    np.where(np.bitwise_and(base_dados['cod_profi_clien__p_2_g_1'] == 0, base_dados['cod_profi_clien__T__p_34_g_1'] == 1), 1,
    np.where(np.bitwise_and(base_dados['cod_profi_clien__p_2_g_1'] == 1, base_dados['cod_profi_clien__T__p_34_g_1'] == 0), 1,
     0)))

base_dados['cod_profi_clien__T__p_34_g_1_c1_4_2'] = np.where(base_dados['cod_profi_clien__T__p_34_g_1_c1_4_1'] == 0, 0,
    np.where(base_dados['cod_profi_clien__T__p_34_g_1_c1_4_1'] == 1, 1,
    0))

base_dados['cod_profi_clien__T__p_34_g_1_c1_4'] = np.where(base_dados['cod_profi_clien__T__p_34_g_1_c1_4_2'] == 0, 0,
    np.where(base_dados['cod_profi_clien__T__p_34_g_1_c1_4_2'] == 1, 1,
     0))





base_dados['val_renda__C__pe_5_g_1_c1_24_1'] = np.where(np.bitwise_and(base_dados['val_renda__p_3_g_1'] == 0, base_dados['val_renda__C__pe_5_g_1'] == 0), 0,
    np.where(np.bitwise_and(base_dados['val_renda__p_3_g_1'] == 0, base_dados['val_renda__C__pe_5_g_1'] == 1), 2,
    np.where(np.bitwise_and(base_dados['val_renda__p_3_g_1'] == 0, base_dados['val_renda__C__pe_5_g_1'] == 2), 3,
    np.where(np.bitwise_and(base_dados['val_renda__p_3_g_1'] == 1, base_dados['val_renda__C__pe_5_g_1'] == 0), 1,
    np.where(np.bitwise_and(base_dados['val_renda__p_3_g_1'] == 1, base_dados['val_renda__C__pe_5_g_1'] == 1), 3,
    np.where(np.bitwise_and(base_dados['val_renda__p_3_g_1'] == 1, base_dados['val_renda__C__pe_5_g_1'] == 2), 3,
     0))))))

base_dados['val_renda__C__pe_5_g_1_c1_24_2'] = np.where(base_dados['val_renda__C__pe_5_g_1_c1_24_1'] == 0, 0,
    np.where(base_dados['val_renda__C__pe_5_g_1_c1_24_1'] == 1, 2,
    np.where(base_dados['val_renda__C__pe_5_g_1_c1_24_1'] == 2, 1,
    np.where(base_dados['val_renda__C__pe_5_g_1_c1_24_1'] == 3, 2,
    0))))

base_dados['val_renda__C__pe_5_g_1_c1_24'] = np.where(base_dados['val_renda__C__pe_5_g_1_c1_24_2'] == 0, 0,
    np.where(base_dados['val_renda__C__pe_5_g_1_c1_24_2'] == 1, 1,
    np.where(base_dados['val_renda__C__pe_5_g_1_c1_24_2'] == 2, 2,
     0)))



# COMMAND ----------

# MAGIC %md
# MAGIC ## Mantendo apenas as variáveis do modelo

# COMMAND ----------

varvar=[]
varvar= [chave,target,'cod_credor_gh38','des_estad_comer_gh38','val_renda__C__pe_5_g_1_c1_24','raiz_cep__L__p_13_g_1_c1_25','tip_ender_corre_gh38','des_fones_resid__L__p_3_g_1_c1_3','des_uf_rg_gh38','qtd_prest__pe_6_g_1_c1_11','ind_sexo_gh38','val_compr__T__p_13_g_1_c1_6','cod_filia_gh38','cod_profi_clien__T__p_34_g_1_c1_4']
base_teste_c0 = base_dados[varvar]
base_teste_c0



# COMMAND ----------

# MAGIC %md
# MAGIC ## Gerando Amostra de treinamento e teste

# COMMAND ----------

base_treino_c0 = pd.read_csv(caminho_base + 'base_treino_final.csv', sep=",", decimal=".")

var_fin_c0=list(base_teste_c0.columns)
var_fin_c0.remove(target)
var_fin_c0.remove(chave)

print(var_fin_c0)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Rodando Regressão Logística

# COMMAND ----------

# Datasets de treino e de teste
x_treino = base_treino_c0[var_fin_c0]
y_treino = base_treino_c0[target]
x_teste = base_teste_c0[var_fin_c0]
y_teste = base_teste_c0[target]
z_teste = base_teste_c0[chave]

# Criando o objeto logistic regression 
modelo = LogisticRegression()

# Treinando o modelo com dados de treino e checando o score
modelo.fit(x_treino, y_treino)
modelo.score(x_treino, y_treino)

# Coletando os coeficientes
print('Coefficient: \n', modelo.coef_)
print('Intercept: \n', modelo.intercept_)
print("Score:",modelo.score(x_treino, y_treino))

# Previsões
valores_previstos = modelo.predict(x_teste)

# Fazendo as previsões e construindo a Confusion Matrix
matrix = metrics.confusion_matrix(y_teste, valores_previstos)

# Imprimindo a Confusion Matrix
print(matrix)

print("Accuracy:",metrics.accuracy_score(y_teste, valores_previstos))
print("Precision:",metrics.precision_score(y_teste, valores_previstos))
print("Recall:",metrics.recall_score(y_teste, valores_previstos))


probabilidades = modelo.predict_proba(x_teste)
data_prob = pd.DataFrame({'P_0': probabilidades[:, 0], 'P_1': probabilidades[:, 1]})

y_teste1 = y_teste.reset_index(drop=True)
z_teste1 = z_teste.reset_index(drop=True)
x_teste1 = x_teste.reset_index(drop=True)
data_prob1 = data_prob.reset_index(drop=True)


x_teste2 = pd.concat([z_teste1,y_teste1,x_teste1, data_prob1], axis=1)

x_teste2


# COMMAND ----------

import pickle

##Save###
pickle.dump(modelo,open(caminho_base+'/model_fit_bmg.sav','wb'))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Avaliando os dados do modelo

# COMMAND ----------

# Curva Roc

y_pred_proba = modelo.predict_proba(x_teste)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_teste,  y_pred_proba)
auc = metrics.roc_auc_score(y_teste, y_pred_proba)
plt.plot(fpr,tpr,label="Base Teste 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

# Confusion Matrix

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# Sumamary do modelo

logit_model=sm.Logit(y_treino,x_treino) 
result=logit_model.fit()
print(result.summary2())


#P-valor P>|z|
print('P-valor P>|z|' + '\n',result.pvalues)

#Importância Z-Score
print('Importância T - Z-Score' + '\n',result.tvalues)

#Coeficientes
print('Coeficientes do modelo' + '\n',result.params)

#Erro Padrão
print('Erro Padrão' + '\n',result.bse)

#Score ou logito, linha a linha
result.predict()

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

x_teste2['P_1_R_p_15_g_1'] = np.where(x_teste2['P_1_R'] <= 0.158978397, 0.0,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.158978397, x_teste2['P_1_R'] <= 0.257094051), 1.0,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.257094051, x_teste2['P_1_R'] <= 0.285989946), 2.0,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.285989946, x_teste2['P_1_R'] <= 0.37535335), 3.0,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.37535335, x_teste2['P_1_R'] <= 0.440098446), 4.0,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.440098446, x_teste2['P_1_R'] <= 0.553522136), 5.0,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.553522136, x_teste2['P_1_R'] <= 0.607873495), 6.0,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.607873495, x_teste2['P_1_R'] <= 0.654412684), 7.0,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.654412684, x_teste2['P_1_R'] <= 0.711128429), 8.0,
    np.where(x_teste2['P_1_R'] > 0.711128429,9,0))))))))))

x_teste2['P_1_pe_10_g_1'] = np.where(x_teste2['P_1'] <= 0.057506572, 0.0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.057506572, x_teste2['P_1'] <= 0.114748183), 1.0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.114748183, x_teste2['P_1'] <= 0.172133035), 2.0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.172133035, x_teste2['P_1'] <= 0.229995392), 3.0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.229995392, x_teste2['P_1'] <= 0.287502594), 4.0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.287502594, x_teste2['P_1'] <= 0.34464105), 5.0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.34464105, x_teste2['P_1'] <= 0.45967743), 6.0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.45967743, x_teste2['P_1'] <= 0.574152655), 7.0,
    np.where(x_teste2['P_1'] > 0.574152655, 8.0,0)))))))))

x_teste2['GH'] = np.where(np.bitwise_and(x_teste2['P_1_pe_10_g_1'] == 0, x_teste2['P_1_R_p_15_g_1'] == 0), 0,
    np.where(np.bitwise_and(x_teste2['P_1_pe_10_g_1'] == 0, x_teste2['P_1_R_p_15_g_1'] == 1), 1,
    np.where(np.bitwise_and(x_teste2['P_1_pe_10_g_1'] == 1, x_teste2['P_1_R_p_15_g_1'] == 1), 2,
    np.where(np.bitwise_and(x_teste2['P_1_pe_10_g_1'] == 1, x_teste2['P_1_R_p_15_g_1'] == 2), 2,
    np.where(np.bitwise_and(x_teste2['P_1_pe_10_g_1'] == 1, x_teste2['P_1_R_p_15_g_1'] == 3), 3,
    np.where(np.bitwise_and(x_teste2['P_1_pe_10_g_1'] == 2, x_teste2['P_1_R_p_15_g_1'] == 3), 4,
    np.where(np.bitwise_and(x_teste2['P_1_pe_10_g_1'] == 2, x_teste2['P_1_R_p_15_g_1'] == 4), 4,
    np.where(np.bitwise_and(x_teste2['P_1_pe_10_g_1'] == 3, x_teste2['P_1_R_p_15_g_1'] == 4), 4,
    np.where(np.bitwise_and(x_teste2['P_1_pe_10_g_1'] == 3, x_teste2['P_1_R_p_15_g_1'] == 5), 5,
    np.where(np.bitwise_and(x_teste2['P_1_pe_10_g_1'] == 4, x_teste2['P_1_R_p_15_g_1'] == 5), 5,
    np.where(np.bitwise_and(x_teste2['P_1_pe_10_g_1'] == 5, x_teste2['P_1_R_p_15_g_1'] == 5), 6,
    np.where(np.bitwise_and(x_teste2['P_1_pe_10_g_1'] == 5, x_teste2['P_1_R_p_15_g_1'] == 6), 6,
    np.where(np.bitwise_and(x_teste2['P_1_pe_10_g_1'] == 6, x_teste2['P_1_R_p_15_g_1'] == 6), 7,
    np.where(np.bitwise_and(x_teste2['P_1_pe_10_g_1'] == 6, x_teste2['P_1_R_p_15_g_1'] == 7), 7,
    np.where(np.bitwise_and(x_teste2['P_1_pe_10_g_1'] == 6, x_teste2['P_1_R_p_15_g_1'] == 8), 8,
    np.where(np.bitwise_and(x_teste2['P_1_pe_10_g_1'] == 7, x_teste2['P_1_R_p_15_g_1'] == 8), 8,
    np.where(np.bitwise_and(x_teste2['P_1_pe_10_g_1'] == 7, x_teste2['P_1_R_p_15_g_1'] == 9), 8,
    np.where(np.bitwise_and(x_teste2['P_1_pe_10_g_1'] == 8, x_teste2['P_1_R_p_15_g_1'] == 9), 9,
             0))))))))))))))))))

del x_teste2['P_1_R']
del x_teste2['P_1_pe_10_g_1']
del x_teste2['P_1_R_p_15_g_1']

x_teste2


# COMMAND ----------

