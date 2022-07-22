# Databricks notebook source
# MAGIC %md
# MAGIC # <font color='blue'>IA - Feature Selection</font>
# MAGIC 
# MAGIC # <font color='blue'>Ferramenta de Criação de Variáveis</font>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importando os pacotes Python

# COMMAND ----------

# MAGIC %run "/pipe_modelos/extrator_recupera/v2/00.le_regras_disponibiliza_variaveis_v2"

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
import pickle

%matplotlib inline

# COMMAND ----------

dbutils.widgets.text('credor', 'zema')
credor = dbutils.widgets.get('credor')
credor = Credor(credor)

dbutils.widgets.text('modelo_escolhido', 'v1_model_20211124')
modelo_escolhido = dbutils.widgets.get('modelo_escolhido')


# COMMAND ----------

# MAGIC %md
# MAGIC ## Inserindo hiperparâmetros do Algoritmo

# COMMAND ----------

## Parâmetros do Algoritmo

#Variável chave-primaria
chave = 'DOCUMENTO'

#Nome da Base de Dados
N_Base = 'pre_processed_pre_output.csv' # este arquivo será construído ao chamar a função preemtive_transform() 

#Caminho da base de dados
caminho_base = credor.caminho_joined_trusted_dbfs

#Separador
separador_ = ";"

#Decimal
decimal_ = "."

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importação da Base de Dados

# COMMAND ----------

############ BLOCO TRY QUE GERA ARQUIVO DE TRANSFORMAÇÃO PREVENTIVA #############
try:
  base_dados = pd.read_csv(os.path.join(caminho_base,N_Base), sep=separador_, decimal=decimal_)
except:
  preemptive_transform(credor, modelo_escolhido)
  base_dados = pd.read_csv(os.path.join(caminho_base,N_Base), sep=separador_, decimal=decimal_)
#################################################################################
  

#carregar o arquivo em formato tabela
base_dados = pd.read_csv(os.path.join(caminho_base,N_Base), sep=separador_, decimal=decimal_)
base_dados = base_dados[[chave,'cod_produt','val_taxa_contr','mod_venda','val_compr','qtd_prest','val_renda','des_fones_resid','dat_inici_contr','dat_expir_prazo','dat_cadas_clien']]

base_dados.fillna(-3)

base_dados['cod_produt'] = base_dados['cod_produt'].replace(np.nan, '-3')
base_dados['mod_venda'] = base_dados['mod_venda'].replace(np.nan, '-3')

base_dados['val_compr'] = base_dados['val_compr'].replace(np.nan, '-3')
base_dados['qtd_prest'] = base_dados['qtd_prest'].replace(np.nan, '-3')
base_dados['val_taxa_contr'] = base_dados['val_taxa_contr'].replace(np.nan, '-3')
base_dados['des_fones_resid'] = base_dados['des_fones_resid'].replace(np.nan, '-3')
base_dados['val_renda'] = base_dados['val_renda'].replace(np.nan, '-3')

base_dados = base_dados.apply(pd.to_numeric, errors='ignore')

base_dados['DOCUMENTO'] = base_dados['DOCUMENTO'].astype(np.int64)
base_dados['val_compr'] = base_dados['val_compr'].astype(float)
base_dados['qtd_prest'] = base_dados['qtd_prest'].astype(int)
base_dados['des_fones_resid'] = base_dados['des_fones_resid'].astype(np.int64)
base_dados['val_renda'] = base_dados['val_renda'].astype(float)

base_dados['p_des_fones_resid'] = np.where(base_dados['des_fones_resid'] > 0, 1,0) 

del base_dados['des_fones_resid']

base_dados['dat_inici_contr'] = pd.to_datetime(base_dados['dat_inici_contr'])
base_dados['dat_expir_prazo'] = pd.to_datetime(base_dados['dat_expir_prazo'])
base_dados['dat_cadas_clien'] = pd.to_datetime(base_dados['dat_cadas_clien'])

base_dados['mob_contrato'] = ((datetime.today()) - base_dados.dat_inici_contr)/np.timedelta64(1, 'M')
base_dados['mob_cliente'] = ((datetime.today()) - base_dados.dat_cadas_clien)/np.timedelta64(1, 'M')
base_dados['mob_expir_prazo'] = (base_dados.dat_expir_prazo - (datetime.today()))/np.timedelta64(1, 'M')

del base_dados['dat_inici_contr']
del base_dados['dat_cadas_clien']
del base_dados['dat_expir_prazo']

base_dados.drop_duplicates(keep=False, inplace=True)

print("shape da Base de Dados:",base_dados.shape)

base_dados.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gerando variáveis Categóricas

# COMMAND ----------

base_dados['cod_produt_gh30'] = np.where(base_dados['cod_produt'] == 'CDC', 0,
np.where(base_dados['cod_produt'] == 'CDCEM', 1,
np.where(base_dados['cod_produt'] == 'EP', 2,
np.where(base_dados['cod_produt'] == 'EPEM', 3,
np.where(base_dados['cod_produt'] == 'PERDA', 4,
np.where(base_dados['cod_produt'] == 'REFINBPC', 5,
np.where(base_dados['cod_produt'] == 'RENCDC', 6,
np.where(base_dados['cod_produt'] == 'RENCDCEM', 7,
np.where(base_dados['cod_produt'] == 'RENEP', 8,
np.where(base_dados['cod_produt'] == 'RENEPEM', 9,
np.where(base_dados['cod_produt'] == 'RENPERDA', 10,
4)))))))))))

base_dados['cod_produt_gh31'] = np.where(base_dados['cod_produt_gh30'] == 0, 0,
np.where(base_dados['cod_produt_gh30'] == 1, 0,
np.where(base_dados['cod_produt_gh30'] == 2, 2,
np.where(base_dados['cod_produt_gh30'] == 3, 3,
np.where(base_dados['cod_produt_gh30'] == 4, 4,
np.where(base_dados['cod_produt_gh30'] == 5, 5,
np.where(base_dados['cod_produt_gh30'] == 6, 6,
np.where(base_dados['cod_produt_gh30'] == 7, 6,
np.where(base_dados['cod_produt_gh30'] == 8, 8,
np.where(base_dados['cod_produt_gh30'] == 9, 9,
np.where(base_dados['cod_produt_gh30'] == 10, 10,
4)))))))))))

base_dados['cod_produt_gh32'] = np.where(base_dados['cod_produt_gh31'] == 0, 0,
np.where(base_dados['cod_produt_gh31'] == 2, 1,
np.where(base_dados['cod_produt_gh31'] == 3, 2,
np.where(base_dados['cod_produt_gh31'] == 4, 3,
np.where(base_dados['cod_produt_gh31'] == 5, 4,
np.where(base_dados['cod_produt_gh31'] == 6, 5,
np.where(base_dados['cod_produt_gh31'] == 8, 6,
np.where(base_dados['cod_produt_gh31'] == 9, 7,
np.where(base_dados['cod_produt_gh31'] == 10, 8,
4)))))))))

base_dados['cod_produt_gh33'] = np.where(base_dados['cod_produt_gh32'] == 0, 0,
np.where(base_dados['cod_produt_gh32'] == 1, 1,
np.where(base_dados['cod_produt_gh32'] == 2, 2,
np.where(base_dados['cod_produt_gh32'] == 3, 3,
np.where(base_dados['cod_produt_gh32'] == 4, 4,
np.where(base_dados['cod_produt_gh32'] == 5, 5,
np.where(base_dados['cod_produt_gh32'] == 6, 6,
np.where(base_dados['cod_produt_gh32'] == 7, 7,
np.where(base_dados['cod_produt_gh32'] == 8, 8,
4)))))))))

base_dados['cod_produt_gh34'] = np.where(base_dados['cod_produt_gh33'] == 0, 0,
np.where(base_dados['cod_produt_gh33'] == 1, 1,
np.where(base_dados['cod_produt_gh33'] == 2, 2,
np.where(base_dados['cod_produt_gh33'] == 3, 3,
np.where(base_dados['cod_produt_gh33'] == 4, 1,
np.where(base_dados['cod_produt_gh33'] == 5, 5,
np.where(base_dados['cod_produt_gh33'] == 6, 2,
np.where(base_dados['cod_produt_gh33'] == 7, 5,
np.where(base_dados['cod_produt_gh33'] == 8, 2,
2)))))))))

base_dados['cod_produt_gh35'] = np.where(base_dados['cod_produt_gh34'] == 0, 0,
np.where(base_dados['cod_produt_gh34'] == 1, 1,
np.where(base_dados['cod_produt_gh34'] == 2, 2,
np.where(base_dados['cod_produt_gh34'] == 3, 3,
np.where(base_dados['cod_produt_gh34'] == 5, 4,
2)))))

base_dados['cod_produt_gh36'] = np.where(base_dados['cod_produt_gh35'] == 0, 2,
np.where(base_dados['cod_produt_gh35'] == 1, 0,
np.where(base_dados['cod_produt_gh35'] == 2, 2,
np.where(base_dados['cod_produt_gh35'] == 3, 1,
np.where(base_dados['cod_produt_gh35'] == 4, 4,
2)))))

base_dados['cod_produt_gh37'] = np.where(base_dados['cod_produt_gh36'] == 0, 1,
np.where(base_dados['cod_produt_gh36'] == 1, 1,
np.where(base_dados['cod_produt_gh36'] == 2, 2,
np.where(base_dados['cod_produt_gh36'] == 4, 3,
1))))

base_dados['cod_produt_gh38'] = np.where(base_dados['cod_produt_gh37'] == 1, 0,
np.where(base_dados['cod_produt_gh37'] == 2, 1,
np.where(base_dados['cod_produt_gh37'] == 3, 2,
1)))
         
                                        
                                        
                                        
                                        
                                        
base_dados['qtd_prest_gh30'] = np.where(base_dados['qtd_prest'] == 0, 0,
np.where(base_dados['qtd_prest'] == 1, 1,
np.where(base_dados['qtd_prest'] == 2, 2,
np.where(base_dados['qtd_prest'] == 3, 3,
np.where(base_dados['qtd_prest'] == 4, 4,
np.where(base_dados['qtd_prest'] == 5, 5,
np.where(base_dados['qtd_prest'] == 6, 6,
np.where(base_dados['qtd_prest'] == 7, 7,
np.where(base_dados['qtd_prest'] == 8, 8,
np.where(base_dados['qtd_prest'] == 9, 9,
np.where(base_dados['qtd_prest'] == 10, 10,
np.where(base_dados['qtd_prest'] == 11, 11,
np.where(base_dados['qtd_prest'] == 12, 12,
np.where(base_dados['qtd_prest'] == 13, 13,
np.where(base_dados['qtd_prest'] == 14, 14,
np.where(base_dados['qtd_prest'] == 15, 15,
np.where(base_dados['qtd_prest'] == 16, 16,
np.where(base_dados['qtd_prest'] == 17, 17,
np.where(base_dados['qtd_prest'] == 18, 18,
np.where(base_dados['qtd_prest'] == 19, 19,
np.where(base_dados['qtd_prest'] == 20, 20,
np.where(base_dados['qtd_prest'] == 21, 21,
np.where(base_dados['qtd_prest'] == 22, 22,
np.where(base_dados['qtd_prest'] == 23, 23,
np.where(base_dados['qtd_prest'] == 24, 24,
np.where(base_dados['qtd_prest'] == 25, 25,
8))))))))))))))))))))))))))

base_dados['qtd_prest_gh31'] = np.where(base_dados['qtd_prest_gh30'] == 0, 0,
np.where(base_dados['qtd_prest_gh30'] == 1, 1,
np.where(base_dados['qtd_prest_gh30'] == 2, 2,
np.where(base_dados['qtd_prest_gh30'] == 3, 3,
np.where(base_dados['qtd_prest_gh30'] == 4, 3,
np.where(base_dados['qtd_prest_gh30'] == 5, 3,
np.where(base_dados['qtd_prest_gh30'] == 6, 3,
np.where(base_dados['qtd_prest_gh30'] == 7, 7,
np.where(base_dados['qtd_prest_gh30'] == 8, 8,
np.where(base_dados['qtd_prest_gh30'] == 9, 9,
np.where(base_dados['qtd_prest_gh30'] == 10, 9,
np.where(base_dados['qtd_prest_gh30'] == 11, 9,
np.where(base_dados['qtd_prest_gh30'] == 12, 9,
np.where(base_dados['qtd_prest_gh30'] == 13, 13,
np.where(base_dados['qtd_prest_gh30'] == 14, 14,
np.where(base_dados['qtd_prest_gh30'] == 15, 14,
np.where(base_dados['qtd_prest_gh30'] == 16, 16,
np.where(base_dados['qtd_prest_gh30'] == 17, 17,
np.where(base_dados['qtd_prest_gh30'] == 18, 18,
np.where(base_dados['qtd_prest_gh30'] == 19, 19,
np.where(base_dados['qtd_prest_gh30'] == 20, 20,
np.where(base_dados['qtd_prest_gh30'] == 21, 21,
np.where(base_dados['qtd_prest_gh30'] == 22, 22,
np.where(base_dados['qtd_prest_gh30'] == 23, 23,
np.where(base_dados['qtd_prest_gh30'] == 24, 24,
np.where(base_dados['qtd_prest_gh30'] == 25, 25,
9))))))))))))))))))))))))))

base_dados['qtd_prest_gh32'] = np.where(base_dados['qtd_prest_gh31'] == 0, 0,
np.where(base_dados['qtd_prest_gh31'] == 1, 1,
np.where(base_dados['qtd_prest_gh31'] == 2, 2,
np.where(base_dados['qtd_prest_gh31'] == 3, 3,
np.where(base_dados['qtd_prest_gh31'] == 7, 4,
np.where(base_dados['qtd_prest_gh31'] == 8, 5,
np.where(base_dados['qtd_prest_gh31'] == 9, 6,
np.where(base_dados['qtd_prest_gh31'] == 13, 7,
np.where(base_dados['qtd_prest_gh31'] == 14, 8,
np.where(base_dados['qtd_prest_gh31'] == 16, 9,
np.where(base_dados['qtd_prest_gh31'] == 17, 10,
np.where(base_dados['qtd_prest_gh31'] == 18, 11,
np.where(base_dados['qtd_prest_gh31'] == 19, 12,
np.where(base_dados['qtd_prest_gh31'] == 20, 13,
np.where(base_dados['qtd_prest_gh31'] == 21, 14,
np.where(base_dados['qtd_prest_gh31'] == 22, 15,
np.where(base_dados['qtd_prest_gh31'] == 23, 16,
np.where(base_dados['qtd_prest_gh31'] == 24, 17,
np.where(base_dados['qtd_prest_gh31'] == 25, 18,
9)))))))))))))))))))

base_dados['qtd_prest_gh33'] = np.where(base_dados['qtd_prest_gh32'] == 0, 0,
np.where(base_dados['qtd_prest_gh32'] == 1, 1,
np.where(base_dados['qtd_prest_gh32'] == 2, 2,
np.where(base_dados['qtd_prest_gh32'] == 3, 3,
np.where(base_dados['qtd_prest_gh32'] == 4, 4,
np.where(base_dados['qtd_prest_gh32'] == 5, 5,
np.where(base_dados['qtd_prest_gh32'] == 6, 6,
np.where(base_dados['qtd_prest_gh32'] == 7, 7,
np.where(base_dados['qtd_prest_gh32'] == 8, 8,
np.where(base_dados['qtd_prest_gh32'] == 9, 9,
np.where(base_dados['qtd_prest_gh32'] == 10, 10,
np.where(base_dados['qtd_prest_gh32'] ==91, 11,
np.where(base_dados['qtd_prest_gh32'] ==92, 12,
np.where(base_dados['qtd_prest_gh32'] ==93, 13,
np.where(base_dados['qtd_prest_gh32'] ==94, 14,
np.where(base_dados['qtd_prest_gh32'] ==95, 15,
np.where(base_dados['qtd_prest_gh32'] ==96, 16,
np.where(base_dados['qtd_prest_gh32'] ==97, 17,
np.where(base_dados['qtd_prest_gh32'] ==98, 18,
9)))))))))))))))))))

base_dados['qtd_prest_gh34'] = np.where(base_dados['qtd_prest_gh33'] == 0, 12,
np.where(base_dados['qtd_prest_gh33'] == 1, 12,
np.where(base_dados['qtd_prest_gh33'] == 2, 12,
np.where(base_dados['qtd_prest_gh33'] == 3, 3,
np.where(base_dados['qtd_prest_gh33'] == 4, 12,
np.where(base_dados['qtd_prest_gh33'] == 5, 12,
np.where(base_dados['qtd_prest_gh33'] == 6, 6,
np.where(base_dados['qtd_prest_gh33'] == 7, 12,
np.where(base_dados['qtd_prest_gh33'] == 8, 8,
np.where(base_dados['qtd_prest_gh33'] == 9, 12,
np.where(base_dados['qtd_prest_gh33'] == 10, 12,
np.where(base_dados['qtd_prest_gh33'] == 11, 6,
np.where(base_dados['qtd_prest_gh33'] == 12, 12,
np.where(base_dados['qtd_prest_gh33'] == 13, 6,
np.where(base_dados['qtd_prest_gh33'] == 14, 8,
np.where(base_dados['qtd_prest_gh33'] == 15, 6,
np.where(base_dados['qtd_prest_gh33'] == 16, 8,
np.where(base_dados['qtd_prest_gh33'] == 17, 6,
np.where(base_dados['qtd_prest_gh33'] == 18, 6,
8)))))))))))))))))))

base_dados['qtd_prest_gh35'] = np.where(base_dados['qtd_prest_gh34'] == 3, 0,
np.where(base_dados['qtd_prest_gh34'] == 6, 1,
np.where(base_dados['qtd_prest_gh34'] == 8, 2,
np.where(base_dados['qtd_prest_gh34'] == 12, 3,
1))))

base_dados['qtd_prest_gh36'] = np.where(base_dados['qtd_prest_gh35'] == 0, 1,
np.where(base_dados['qtd_prest_gh35'] == 1, 1,
np.where(base_dados['qtd_prest_gh35'] == 2, 1,
np.where(base_dados['qtd_prest_gh35'] == 3, 0,
0))))
base_dados['qtd_prest_gh37'] = np.where(base_dados['qtd_prest_gh36'] == 0, 0,
np.where(base_dados['qtd_prest_gh36'] == 1, 1,
0))
base_dados['qtd_prest_gh38'] = np.where(base_dados['qtd_prest_gh37'] == 0, 0,
np.where(base_dados['qtd_prest_gh37'] == 1, 1,
0))
         
         
         
         
         
base_dados['mod_venda_gh30'] = np.where(base_dados['mod_venda'] == -3, 0,
np.where(base_dados['mod_venda'] == 1, 1,
np.where(base_dados['mod_venda'] == 2, 2,
np.where(base_dados['mod_venda'] == 7, 3,
1))))

base_dados['mod_venda_gh31'] = np.where(base_dados['mod_venda_gh30'] == 0, 0,
np.where(base_dados['mod_venda_gh30'] == 1, 1,
np.where(base_dados['mod_venda_gh30'] == 2, 2,
np.where(base_dados['mod_venda_gh30'] == 3, 3,
1))))
         
base_dados['mod_venda_gh32'] = np.where(base_dados['mod_venda_gh31'] == 0, 0,
np.where(base_dados['mod_venda_gh31'] == 1, 1,
np.where(base_dados['mod_venda_gh31'] == 2, 2,
np.where(base_dados['mod_venda_gh31'] == 3, 3,
1))))

base_dados['mod_venda_gh33'] = np.where(base_dados['mod_venda_gh32'] == 0, 0,
np.where(base_dados['mod_venda_gh32'] == 1, 1,
np.where(base_dados['mod_venda_gh32'] == 2, 2,
np.where(base_dados['mod_venda_gh32'] == 3, 3,
1))))

base_dados['mod_venda_gh34'] = np.where(base_dados['mod_venda_gh33'] == 0, 1,
np.where(base_dados['mod_venda_gh33'] == 1, 1,
np.where(base_dados['mod_venda_gh33'] == 2, 3,
np.where(base_dados['mod_venda_gh33'] == 3, 3,
1))))

base_dados['mod_venda_gh35'] = np.where(base_dados['mod_venda_gh34'] == 1, 0,
np.where(base_dados['mod_venda_gh34'] == 3, 1,
0))

base_dados['mod_venda_gh36'] = np.where(base_dados['mod_venda_gh35'] == 0, 1,
np.where(base_dados['mod_venda_gh35'] == 1, 0,
0))

base_dados['mod_venda_gh37'] = np.where(base_dados['mod_venda_gh36'] == 0, 0,
np.where(base_dados['mod_venda_gh36'] == 1, 1,
0))

base_dados['mod_venda_gh38'] = np.where(base_dados['mod_venda_gh37'] == 0, 0,
np.where(base_dados['mod_venda_gh37'] == 1, 1,
0))
                                                
                                                
                                                
                                                
base_dados['p_des_fones_resid_gh30'] = np.where(base_dados['p_des_fones_resid'] == 0, 0,
np.where(base_dados['p_des_fones_resid'] == 1, 1,
0))

base_dados['p_des_fones_resid_gh31'] = np.where(base_dados['p_des_fones_resid_gh30'] == 0, 0,
np.where(base_dados['p_des_fones_resid_gh30'] == 1, 1,
0))

base_dados['p_des_fones_resid_gh32'] = np.where(base_dados['p_des_fones_resid_gh31'] == 0, 0,
np.where(base_dados['p_des_fones_resid_gh31'] == 1, 1,
0))
                                                
base_dados['p_des_fones_resid_gh33'] = np.where(base_dados['p_des_fones_resid_gh32'] == 0, 0,
np.where(base_dados['p_des_fones_resid_gh32'] == 1, 1,
0))

base_dados['p_des_fones_resid_gh34'] = np.where(base_dados['p_des_fones_resid_gh33'] == 0, 0,
np.where(base_dados['p_des_fones_resid_gh33'] == 1, 1,
0))

base_dados['p_des_fones_resid_gh35'] = np.where(base_dados['p_des_fones_resid_gh34'] == 0, 0,
np.where(base_dados['p_des_fones_resid_gh34'] == 1, 1,
0))

base_dados['p_des_fones_resid_gh36'] = np.where(base_dados['p_des_fones_resid_gh35'] == 0, 1,
np.where(base_dados['p_des_fones_resid_gh35'] == 1, 0,
0))

base_dados['p_des_fones_resid_gh37'] = np.where(base_dados['p_des_fones_resid_gh36'] == 0, 0,
np.where(base_dados['p_des_fones_resid_gh36'] == 1, 1,
0))

base_dados['p_des_fones_resid_gh38'] = np.where(base_dados['p_des_fones_resid_gh37'] == 0, 0,
np.where(base_dados['p_des_fones_resid_gh37'] == 1, 1,
0))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Gerando variáveis numéricas contínuas

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parte 1 de 2

# COMMAND ----------

         
         
base_dados['val_renda__R'] = np.sqrt(base_dados['val_renda'])
np.where(base_dados['val_renda__R'] == 0, -1, base_dados['val_renda__R'])
base_dados['val_renda__R'] = base_dados['val_renda__R'].fillna(-2)

base_dados['val_renda__R__pe_10'] = np.where(np.bitwise_and(base_dados['val_renda__R'] >= -2.0, base_dados['val_renda__R'] <= 0.282842712474619), 0.0,
np.where(np.bitwise_and(base_dados['val_renda__R'] > 0.282842712474619, base_dados['val_renda__R'] <= 0.5744562646538028), 1.0,
np.where(np.bitwise_and(base_dados['val_renda__R'] > 0.5744562646538028, base_dados['val_renda__R'] <= 0.8602325267042626), 2.0,
np.where(np.bitwise_and(base_dados['val_renda__R'] > 0.8602325267042626, base_dados['val_renda__R'] <= 1.140175425099138), 3.0,
np.where(np.bitwise_and(base_dados['val_renda__R'] > 1.140175425099138, base_dados['val_renda__R'] <= 1.4142135623730951), 4.0,
np.where(np.bitwise_and(base_dados['val_renda__R'] > 1.4142135623730951, base_dados['val_renda__R'] <= 1.6613247725836149), 5.0,
np.where(np.bitwise_and(base_dados['val_renda__R'] > 1.6613247725836149, base_dados['val_renda__R'] <= 1.857417562100671), 6.0,
np.where(np.bitwise_and(base_dados['val_renda__R'] > 1.857417562100671, base_dados['val_renda__R'] <= 2.1213203435596424), 7.0,
np.where(base_dados['val_renda__R'] > 2.1213203435596424, 9.0,
 -2)))))))))
         
base_dados['val_renda__R__pe_10_g_1_1'] = np.where(base_dados['val_renda__R__pe_10'] == -2.0, 2,
np.where(base_dados['val_renda__R__pe_10'] == 0.0, 1,
np.where(base_dados['val_renda__R__pe_10'] == 1.0, 0,
np.where(base_dados['val_renda__R__pe_10'] == 2.0, 2,
np.where(base_dados['val_renda__R__pe_10'] == 3.0, 2,
np.where(base_dados['val_renda__R__pe_10'] == 4.0, 2,
np.where(base_dados['val_renda__R__pe_10'] == 5.0, 2,
np.where(base_dados['val_renda__R__pe_10'] == 6.0, 2,
np.where(base_dados['val_renda__R__pe_10'] == 7.0, 2,
np.where(base_dados['val_renda__R__pe_10'] == 9.0, 2,
 0))))))))))
         
base_dados['val_renda__R__pe_10_g_1_2'] = np.where(base_dados['val_renda__R__pe_10_g_1_1'] == 0, 1,
np.where(base_dados['val_renda__R__pe_10_g_1_1'] == 1, 0,
np.where(base_dados['val_renda__R__pe_10_g_1_1'] == 2, 1,
 0)))
         
base_dados['val_renda__R__pe_10_g_1'] = np.where(base_dados['val_renda__R__pe_10_g_1_2'] == 0, 0,
np.where(base_dados['val_renda__R__pe_10_g_1_2'] == 1, 1,
 0))
                                                 
                                                 
                                                 
                                                 
                                                 
                                                 
                                                 
base_dados['val_renda__S'] = np.sin(base_dados['val_renda'])
np.where(base_dados['val_renda__S'] == 0, -1, base_dados['val_renda__S'])
base_dados['val_renda__S'] = base_dados['val_renda__S'].fillna(-2)

base_dados['val_renda__S__p_7'] = np.where(np.bitwise_and(base_dados['val_renda__S'] >= -1.0, base_dados['val_renda__S'] <= 0.06994284733753277), 0.0,
np.where(np.bitwise_and(base_dados['val_renda__S'] > 0.06994284733753277, base_dados['val_renda__S'] <= 0.09983341664682815), 1.0,
np.where(np.bitwise_and(base_dados['val_renda__S'] > 0.09983341664682815, base_dados['val_renda__S'] <= 0.11971220728891936), 2.0,
np.where(np.bitwise_and(base_dados['val_renda__S'] > 0.11971220728891936, base_dados['val_renda__S'] <= 0.14943813247359922), 3.0,
np.where(np.bitwise_and(base_dados['val_renda__S'] > 0.14943813247359922, base_dados['val_renda__S'] <= 0.18885889497650057), 4.0,
np.where(np.bitwise_and(base_dados['val_renda__S'] > 0.18885889497650057, base_dados['val_renda__S'] <= 0.26673143668883115), 5.0,
np.where(base_dados['val_renda__S'] > 0.26673143668883115, 6.0,
 -2)))))))

base_dados['val_renda__S__p_7_g_1_1'] = np.where(base_dados['val_renda__S__p_7'] == -2.0, 2,
np.where(base_dados['val_renda__S__p_7'] == 0.0, 2,
np.where(base_dados['val_renda__S__p_7'] == 1.0, 0,
np.where(base_dados['val_renda__S__p_7'] == 2.0, 2,
np.where(base_dados['val_renda__S__p_7'] == 3.0, 2,
np.where(base_dados['val_renda__S__p_7'] == 4.0, 1,
np.where(base_dados['val_renda__S__p_7'] == 5.0, 1,
np.where(base_dados['val_renda__S__p_7'] == 6.0, 2,
 0))))))))
         
base_dados['val_renda__S__p_7_g_1_2'] = np.where(base_dados['val_renda__S__p_7_g_1_1'] == 0, 0,
np.where(base_dados['val_renda__S__p_7_g_1_1'] == 1, 2,
np.where(base_dados['val_renda__S__p_7_g_1_1'] == 2, 1,
 0)))
         
base_dados['val_renda__S__p_7_g_1'] = np.where(base_dados['val_renda__S__p_7_g_1_2'] == 0, 0,
np.where(base_dados['val_renda__S__p_7_g_1_2'] == 1, 1,
np.where(base_dados['val_renda__S__p_7_g_1_2'] == 2, 2,
 0)))
         
         
         
         
         
         
              
base_dados['mob_contrato__pe_6'] = np.where(np.bitwise_and(base_dados['mob_contrato'] >= -0.782789804219495, base_dados['mob_contrato'] <= 35.85040594928673), 0.0,
np.where(np.bitwise_and(base_dados['mob_contrato'] > 35.85040594928673, base_dados['mob_contrato'] <= 71.79364913703186), 1.0,
np.where(np.bitwise_and(base_dados['mob_contrato'] > 71.79364913703186, base_dados['mob_contrato'] <= 107.50690813618995), 2.0,
np.where(np.bitwise_and(base_dados['mob_contrato'] > 107.50690813618995, base_dados['mob_contrato'] <= 143.31873178759966), 3.0,
np.where(np.bitwise_and(base_dados['mob_contrato'] > 143.31873178759966, base_dados['mob_contrato'] <= 179.3276847435125), 4.0,
np.where(base_dados['mob_contrato'] > 179.3276847435125, 5.0,
 -2))))))
         
base_dados['mob_contrato__pe_6_g_1_1'] = np.where(base_dados['mob_contrato__pe_6'] == -2.0, 2,
np.where(base_dados['mob_contrato__pe_6'] == 0.0, 0,
np.where(base_dados['mob_contrato__pe_6'] == 1.0, 2,
np.where(base_dados['mob_contrato__pe_6'] == 2.0, 1,
np.where(base_dados['mob_contrato__pe_6'] == 3.0, 0,
np.where(base_dados['mob_contrato__pe_6'] == 4.0, 2,
np.where(base_dados['mob_contrato__pe_6'] == 5.0, 2,
 0)))))))
         
base_dados['mob_contrato__pe_6_g_1_2'] = np.where(base_dados['mob_contrato__pe_6_g_1_1'] == 0, 2,
np.where(base_dados['mob_contrato__pe_6_g_1_1'] == 1, 0,
np.where(base_dados['mob_contrato__pe_6_g_1_1'] == 2, 1,
 0)))
         
base_dados['mob_contrato__pe_6_g_1'] = np.where(base_dados['mob_contrato__pe_6_g_1_2'] == 0, 0,
np.where(base_dados['mob_contrato__pe_6_g_1_2'] == 1, 1,
np.where(base_dados['mob_contrato__pe_6_g_1_2'] == 2, 2,
 0)))
         
         
         
         
         
         
         
base_dados['mob_contrato__L'] = np.log(base_dados['mob_contrato'])
np.where(base_dados['mob_contrato__L'] == 0, -1, base_dados['mob_contrato__L'])
base_dados['mob_contrato__L'] = base_dados['mob_contrato__L'].fillna(-2)

base_dados['mob_contrato__L__p_7'] = np.where(np.bitwise_and(base_dados['mob_contrato__L'] >= -2.0, base_dados['mob_contrato__L'] <= 2.7028262324315335), 0.0,
np.where(np.bitwise_and(base_dados['mob_contrato__L'] > 2.7028262324315335, base_dados['mob_contrato__L'] <= 3.228361825766573), 1.0,
np.where(np.bitwise_and(base_dados['mob_contrato__L'] > 3.228361825766573, base_dados['mob_contrato__L'] <= 3.5957162990285223), 2.0,
np.where(np.bitwise_and(base_dados['mob_contrato__L'] > 3.5957162990285223, base_dados['mob_contrato__L'] <= 3.9740167553596835), 3.0,
np.where(np.bitwise_and(base_dados['mob_contrato__L'] > 3.9740167553596835, base_dados['mob_contrato__L'] <= 4.3829397167645885), 4.0,
np.where(np.bitwise_and(base_dados['mob_contrato__L'] > 4.3829397167645885, base_dados['mob_contrato__L'] <= 4.8469490806294555), 5.0,
np.where(base_dados['mob_contrato__L'] > 4.8469490806294555, 6.0,
 -2)))))))
         
base_dados['mob_contrato__L__p_7_g_1_1'] = np.where(base_dados['mob_contrato__L__p_7'] == -2.0, 3,
np.where(base_dados['mob_contrato__L__p_7'] == 0.0, 1,
np.where(base_dados['mob_contrato__L__p_7'] == 1.0, 3,
np.where(base_dados['mob_contrato__L__p_7'] == 2.0, 4,
np.where(base_dados['mob_contrato__L__p_7'] == 3.0, 4,
np.where(base_dados['mob_contrato__L__p_7'] == 4.0, 2,
np.where(base_dados['mob_contrato__L__p_7'] == 5.0, 0,
np.where(base_dados['mob_contrato__L__p_7'] == 6.0, 5,
 0))))))))
         
base_dados['mob_contrato__L__p_7_g_1_2'] = np.where(base_dados['mob_contrato__L__p_7_g_1_1'] == 0, 0,
np.where(base_dados['mob_contrato__L__p_7_g_1_1'] == 1, 4,
np.where(base_dados['mob_contrato__L__p_7_g_1_1'] == 2, 2,
np.where(base_dados['mob_contrato__L__p_7_g_1_1'] == 3, 4,
np.where(base_dados['mob_contrato__L__p_7_g_1_1'] == 4, 3,
np.where(base_dados['mob_contrato__L__p_7_g_1_1'] == 5, 0,
 0))))))
         
base_dados['mob_contrato__L__p_7_g_1'] = np.where(base_dados['mob_contrato__L__p_7_g_1_2'] == 0, 0,
np.where(base_dados['mob_contrato__L__p_7_g_1_2'] == 2, 1,
np.where(base_dados['mob_contrato__L__p_7_g_1_2'] == 3, 2,
np.where(base_dados['mob_contrato__L__p_7_g_1_2'] == 4, 3,
 0))))
         
         
         
         
               
base_dados['val_compr__L'] = np.log(base_dados['val_compr'])
np.where(base_dados['val_compr__L'] == 0, -1, base_dados['val_compr__L'])
base_dados['val_compr__L'] = base_dados['val_compr__L'].fillna(-2)

base_dados['val_compr__L__p_6'] = np.where(np.bitwise_and(base_dados['val_compr__L'] >= -1.0, base_dados['val_compr__L'] <= 4.768139014266231), 0.0,
np.where(np.bitwise_and(base_dados['val_compr__L'] > 4.768139014266231, base_dados['val_compr__L'] <= 6.249955937053834), 1.0,
np.where(np.bitwise_and(base_dados['val_compr__L'] > 6.249955937053834, base_dados['val_compr__L'] <= 6.864148159599793), 2.0,
np.where(np.bitwise_and(base_dados['val_compr__L'] > 6.864148159599793, base_dados['val_compr__L'] <= 7.239774759640453), 3.0,
np.where(np.bitwise_and(base_dados['val_compr__L'] > 7.239774759640453, base_dados['val_compr__L'] <= 7.627344760047607), 4.0,
np.where(base_dados['val_compr__L'] > 7.627344760047607, 5.0,
 -2))))))

base_dados['val_compr__L__p_6_g_1_1'] = np.where(base_dados['val_compr__L__p_6'] == -2.0, 0,
np.where(base_dados['val_compr__L__p_6'] == 0.0, 1,
np.where(base_dados['val_compr__L__p_6'] == 1.0, 0,
np.where(base_dados['val_compr__L__p_6'] == 2.0, 0,
np.where(base_dados['val_compr__L__p_6'] == 3.0, 0,
np.where(base_dados['val_compr__L__p_6'] == 4.0, 1,
np.where(base_dados['val_compr__L__p_6'] == 5.0, 0,
 0)))))))

base_dados['val_compr__L__p_6_g_1_2'] = np.where(base_dados['val_compr__L__p_6_g_1_1'] == 0, 0,
np.where(base_dados['val_compr__L__p_6_g_1_1'] == 1, 1,
 0))
base_dados['val_compr__L__p_6_g_1'] = np.where(base_dados['val_compr__L__p_6_g_1_2'] == 0, 0,
np.where(base_dados['val_compr__L__p_6_g_1_2'] == 1, 1,
 0))
                                               
                                               
                                               
                                               
                                               
                                               
                                               
base_dados['val_compr__L'] = np.log(base_dados['val_compr'])
np.where(base_dados['val_compr__L'] == 0, -1, base_dados['val_compr__L'])
base_dados['val_compr__L'] = base_dados['val_compr__L'].fillna(-2)

base_dados['val_compr__L__p_2'] = np.where(np.bitwise_and(base_dados['val_compr__L'] >= -1.0, base_dados['val_compr__L'] <= 6.838426636844815), 0.0,
np.where(base_dados['val_compr__L'] > 6.838426636844815, 1.0,
 -2))

base_dados['val_compr__L__p_2_g_1_1'] = np.where(base_dados['val_compr__L__p_2'] == -2.0, 2,
np.where(base_dados['val_compr__L__p_2'] == 0.0, 0,
np.where(base_dados['val_compr__L__p_2'] == 1.0, 1,
 0)))

base_dados['val_compr__L__p_2_g_1_2'] = np.where(base_dados['val_compr__L__p_2_g_1_1'] == 0, 1,
np.where(base_dados['val_compr__L__p_2_g_1_1'] == 1, 2,
np.where(base_dados['val_compr__L__p_2_g_1_1'] == 2, 0,
 0)))
         
base_dados['val_compr__L__p_2_g_1'] = np.where(base_dados['val_compr__L__p_2_g_1_2'] == 0, 0,
np.where(base_dados['val_compr__L__p_2_g_1_2'] == 1, 1,
np.where(base_dados['val_compr__L__p_2_g_1_2'] == 2, 2,
 0)))
         
         
         
         
         
         
                  
base_dados['mob_cliente__R'] = np.sqrt(base_dados['mob_cliente'])
np.where(base_dados['mob_cliente__R'] == 0, -1, base_dados['mob_cliente__R'])
base_dados['mob_cliente__R'] = base_dados['mob_cliente__R'].fillna(-2)

base_dados['mob_cliente__R__p_6'] = np.where(np.bitwise_and(base_dados['mob_cliente__R'] >= -2.0, base_dados['mob_cliente__R'] <= 4.843977152216041), 0.0,
np.where(np.bitwise_and(base_dados['mob_cliente__R'] > 4.843977152216041, base_dados['mob_cliente__R'] <= 6.980653301209062), 1.0,
np.where(np.bitwise_and(base_dados['mob_cliente__R'] > 6.980653301209062, base_dados['mob_cliente__R'] <= 9.255182884305526), 2.0,
np.where(np.bitwise_and(base_dados['mob_cliente__R'] > 9.255182884305526, base_dados['mob_cliente__R'] <= 10.65146197274765), 3.0,
np.where(np.bitwise_and(base_dados['mob_cliente__R'] > 10.65146197274765, base_dados['mob_cliente__R'] <= 12.829992076204201), 4.0,
np.where(base_dados['mob_cliente__R'] > 12.829992076204201, 5.0,
 -2))))))
         
base_dados['mob_cliente__R__p_6_g_1_1'] = np.where(base_dados['mob_cliente__R__p_6'] == -2.0, 2,
np.where(base_dados['mob_cliente__R__p_6'] == 0.0, 1,
np.where(base_dados['mob_cliente__R__p_6'] == 1.0, 1,
np.where(base_dados['mob_cliente__R__p_6'] == 2.0, 1,
np.where(base_dados['mob_cliente__R__p_6'] == 3.0, 0,
np.where(base_dados['mob_cliente__R__p_6'] == 4.0, 0,
np.where(base_dados['mob_cliente__R__p_6'] == 5.0, 2,
 0)))))))
         
base_dados['mob_cliente__R__p_6_g_1_2'] = np.where(base_dados['mob_cliente__R__p_6_g_1_1'] == 0, 0,
np.where(base_dados['mob_cliente__R__p_6_g_1_1'] == 1, 2,
np.where(base_dados['mob_cliente__R__p_6_g_1_1'] == 2, 0,
 0)))

base_dados['mob_cliente__R__p_6_g_1'] = np.where(base_dados['mob_cliente__R__p_6_g_1_2'] == 0, 0,
np.where(base_dados['mob_cliente__R__p_6_g_1_2'] == 2, 1,
 0))
                                                 
                                                 
                                                 
                                                 
                                                 
                                                 
base_dados['mob_cliente__L'] = np.log(base_dados['mob_cliente'])
np.where(base_dados['mob_cliente__L'] == 0, -1, base_dados['mob_cliente__L'])
base_dados['mob_cliente__L'] = base_dados['mob_cliente__L'].fillna(-2)

base_dados['mob_cliente__L__pe_13'] = np.where(np.bitwise_and(base_dados['mob_cliente__L'] >= -2.0, base_dados['mob_cliente__L'] <= 0.2775732338779211), 0.0,
np.where(np.bitwise_and(base_dados['mob_cliente__L'] > 0.2775732338779211, base_dados['mob_cliente__L'] <= 1.2110255526844587), 1.0,
np.where(np.bitwise_and(base_dados['mob_cliente__L'] > 1.2110255526844587, base_dados['mob_cliente__L'] <= 2.069865143944902), 2.0,
np.where(np.bitwise_and(base_dados['mob_cliente__L'] > 2.069865143944902, base_dados['mob_cliente__L'] <= 2.783178105326708), 3.0,
np.where(np.bitwise_and(base_dados['mob_cliente__L'] > 2.783178105326708, base_dados['mob_cliente__L'] <= 3.47919177777978), 4.0,
np.where(np.bitwise_and(base_dados['mob_cliente__L'] > 3.47919177777978, base_dados['mob_cliente__L'] <= 4.175285295801343), 5.0,
np.where(np.bitwise_and(base_dados['mob_cliente__L'] > 4.175285295801343, base_dados['mob_cliente__L'] <= 4.87065850661545), 6.0,
np.where(np.bitwise_and(base_dados['mob_cliente__L'] > 4.87065850661545, base_dados['mob_cliente__L'] <= 5.5610084615358675), 7.0,
np.where(np.bitwise_and(base_dados['mob_cliente__L'] > 5.5610084615358675, base_dados['mob_cliente__L'] <= 5.776641112439972), 8.0,
np.where(base_dados['mob_cliente__L'] > 5.776641112439972, 10.0,
 -2))))))))))
         
base_dados['mob_cliente__L__pe_13_g_1_1'] = np.where(base_dados['mob_cliente__L__pe_13'] == -2.0, 2,
np.where(base_dados['mob_cliente__L__pe_13'] == 0.0, 2,
np.where(base_dados['mob_cliente__L__pe_13'] == 1.0, 2,
np.where(base_dados['mob_cliente__L__pe_13'] == 2.0, 2,
np.where(base_dados['mob_cliente__L__pe_13'] == 3.0, 2,
np.where(base_dados['mob_cliente__L__pe_13'] == 4.0, 2,
np.where(base_dados['mob_cliente__L__pe_13'] == 5.0, 2,
np.where(base_dados['mob_cliente__L__pe_13'] == 6.0, 1,
np.where(base_dados['mob_cliente__L__pe_13'] == 7.0, 0,
np.where(base_dados['mob_cliente__L__pe_13'] == 8.0, 2,
np.where(base_dados['mob_cliente__L__pe_13'] == 10.0, 2,
 0)))))))))))
         
base_dados['mob_cliente__L__pe_13_g_1_2'] = np.where(base_dados['mob_cliente__L__pe_13_g_1_1'] == 0, 0,
np.where(base_dados['mob_cliente__L__pe_13_g_1_1'] == 1, 1,
np.where(base_dados['mob_cliente__L__pe_13_g_1_1'] == 2, 2,
 0)))
         
base_dados['mob_cliente__L__pe_13_g_1'] = np.where(base_dados['mob_cliente__L__pe_13_g_1_2'] == 0, 0,
np.where(base_dados['mob_cliente__L__pe_13_g_1_2'] == 1, 1,
np.where(base_dados['mob_cliente__L__pe_13_g_1_2'] == 2, 2,
 0)))
         
         
         
         
                 
base_dados['mob_expir_prazo__p_13'] = np.where(base_dados['mob_expir_prazo'] <= 317.799566325982, 0.0,
np.where(np.bitwise_and(base_dados['mob_expir_prazo'] > 317.799566325982, base_dados['mob_expir_prazo'] <= 321.34789380703916), 8.0,
np.where(base_dados['mob_expir_prazo'] > 321.34789380703916, 12.0,
 -2)))

base_dados['mob_expir_prazo__p_13_g_1_1'] = np.where(base_dados['mob_expir_prazo__p_13'] == -2.0, 2,
np.where(base_dados['mob_expir_prazo__p_13'] == 0.0, 0,
np.where(base_dados['mob_expir_prazo__p_13'] == 8.0, 2,
np.where(base_dados['mob_expir_prazo__p_13'] == 12.0, 1,
 0))))
         
base_dados['mob_expir_prazo__p_13_g_1_2'] = np.where(base_dados['mob_expir_prazo__p_13_g_1_1'] == 0, 0,
np.where(base_dados['mob_expir_prazo__p_13_g_1_1'] == 1, 2,
np.where(base_dados['mob_expir_prazo__p_13_g_1_1'] == 2, 1,
 0)))

base_dados['mob_expir_prazo__p_13_g_1'] = np.where(base_dados['mob_expir_prazo__p_13_g_1_2'] == 0, 0,
np.where(base_dados['mob_expir_prazo__p_13_g_1_2'] == 1, 1,
np.where(base_dados['mob_expir_prazo__p_13_g_1_2'] == 2, 2,
 0)))
         
         
         
         
         
base_dados['mob_expir_prazo__S'] = np.sin(base_dados['mob_expir_prazo'])
np.where(base_dados['mob_expir_prazo__S'] == 0, -1, base_dados['mob_expir_prazo__S'])
base_dados['mob_expir_prazo__S'] = base_dados['mob_expir_prazo__S'].fillna(-2)

base_dados['mob_expir_prazo__S__p_13'] = np.where(np.bitwise_and(base_dados['mob_expir_prazo__S'] >= -2.0, base_dados['mob_expir_prazo__S'] <= 0.16552646403318755), 10.0,
np.where(np.bitwise_and(base_dados['mob_expir_prazo__S'] > 0.16552646403318755, base_dados['mob_expir_prazo__S'] <= 0.8436296408663326), 11.0,
np.where(base_dados['mob_expir_prazo__S'] > 0.8436296408663326, 12.0,
 -2)))

base_dados['mob_expir_prazo__S__p_13_g_1_1'] = np.where(base_dados['mob_expir_prazo__S__p_13'] == -2.0, 0,
np.where(base_dados['mob_expir_prazo__S__p_13'] == 10.0, 1,
np.where(base_dados['mob_expir_prazo__S__p_13'] == 11.0, 1,
np.where(base_dados['mob_expir_prazo__S__p_13'] == 12.0, 1,
 0))))
         
base_dados['mob_expir_prazo__S__p_13_g_1_2'] = np.where(base_dados['mob_expir_prazo__S__p_13_g_1_1'] == 0, 0,
np.where(base_dados['mob_expir_prazo__S__p_13_g_1_1'] == 1, 1,
 0))

base_dados['mob_expir_prazo__S__p_13_g_1'] = np.where(base_dados['mob_expir_prazo__S__p_13_g_1_2'] == 0, 0,
np.where(base_dados['mob_expir_prazo__S__p_13_g_1_2'] == 1, 1,
 0))
                                                      
                                                      
                                                      
                                                      
                                                      
                                                      
base_dados['val_taxa_contr__p_5'] = np.where(base_dados['val_taxa_contr'] == 0 , -1.0,
np.where(np.bitwise_and(base_dados['val_taxa_contr'] > 0.0, base_dados['val_taxa_contr'] <= 2.5), 0.0,
np.where(np.bitwise_and(base_dados['val_taxa_contr'] > 2.5, base_dados['val_taxa_contr'] <= 3.53), 1.0,
np.where(np.bitwise_and(base_dados['val_taxa_contr'] > 3.53, base_dados['val_taxa_contr'] <= 4.64), 2.0,
np.where(np.bitwise_and(base_dados['val_taxa_contr'] > 4.64, base_dados['val_taxa_contr'] <= 6.65), 3.0,
np.where(base_dados['val_taxa_contr'] > 6.65, 4.0,
 -2))))))

base_dados['val_taxa_contr__p_5_g_1_1'] = np.where(base_dados['val_taxa_contr__p_5'] == -2.0, 3,
np.where(base_dados['val_taxa_contr__p_5'] == -1.0, 2,
np.where(base_dados['val_taxa_contr__p_5'] == 0.0, 1,
np.where(base_dados['val_taxa_contr__p_5'] == 1.0, 3,
np.where(base_dados['val_taxa_contr__p_5'] == 2.0, 0,
np.where(base_dados['val_taxa_contr__p_5'] == 3.0, 0,
np.where(base_dados['val_taxa_contr__p_5'] == 4.0, 1,
 0)))))))

base_dados['val_taxa_contr__p_5_g_1_2'] = np.where(base_dados['val_taxa_contr__p_5_g_1_1'] == 0, 1,
np.where(base_dados['val_taxa_contr__p_5_g_1_1'] == 1, 3,
np.where(base_dados['val_taxa_contr__p_5_g_1_1'] == 2, 1,
np.where(base_dados['val_taxa_contr__p_5_g_1_1'] == 3, 0,
 0))))

base_dados['val_taxa_contr__p_5_g_1'] = np.where(base_dados['val_taxa_contr__p_5_g_1_2'] == 0, 0,
np.where(base_dados['val_taxa_contr__p_5_g_1_2'] == 1, 1,
np.where(base_dados['val_taxa_contr__p_5_g_1_2'] == 3, 2,
 0)))
         
         
         
         
         
base_dados['val_taxa_contr__L'] = np.log(base_dados['val_taxa_contr'])
np.where(base_dados['val_taxa_contr__L'] == 0, -1, base_dados['val_taxa_contr__L'])
base_dados['val_taxa_contr__L'] = base_dados['val_taxa_contr__L'].fillna(-2)

base_dados['val_taxa_contr__L__p_4'] = np.where(base_dados['val_taxa_contr__L'] == 0 , -1.0,
np.where(np.bitwise_and(base_dados['val_taxa_contr__L'] > 0.0, base_dados['val_taxa_contr__L'] <= 0.9932517730102834), 0.0,
np.where(np.bitwise_and(base_dados['val_taxa_contr__L'] > 0.9932517730102834, base_dados['val_taxa_contr__L'] <= 1.3837912309017721), 1.0,
np.where(np.bitwise_and(base_dados['val_taxa_contr__L'] > 1.3837912309017721, base_dados['val_taxa_contr__L'] <= 1.8718021769015913), 2.0,
np.where(base_dados['val_taxa_contr__L'] > 1.8718021769015913, 3.0,
 -2)))))
         
base_dados['val_taxa_contr__L__p_4_g_1_1'] = np.where(base_dados['val_taxa_contr__L__p_4'] == -2.0, 2,
np.where(base_dados['val_taxa_contr__L__p_4'] == -1.0, 2,
np.where(base_dados['val_taxa_contr__L__p_4'] == 0.0, 0,
np.where(base_dados['val_taxa_contr__L__p_4'] == 1.0, 1,
np.where(base_dados['val_taxa_contr__L__p_4'] == 2.0, 0,
np.where(base_dados['val_taxa_contr__L__p_4'] == 3.0, 0,
 0))))))
         
base_dados['val_taxa_contr__L__p_4_g_1_2'] = np.where(base_dados['val_taxa_contr__L__p_4_g_1_1'] == 0, 2,
np.where(base_dados['val_taxa_contr__L__p_4_g_1_1'] == 1, 0,
np.where(base_dados['val_taxa_contr__L__p_4_g_1_1'] == 2, 1,
 0)))
         
base_dados['val_taxa_contr__L__p_4_g_1'] = np.where(base_dados['val_taxa_contr__L__p_4_g_1_2'] == 0, 0,
np.where(base_dados['val_taxa_contr__L__p_4_g_1_2'] == 1, 1,
np.where(base_dados['val_taxa_contr__L__p_4_g_1_2'] == 2, 2,
 0)))


# COMMAND ----------

# MAGIC %md
# MAGIC ### Parte 2 de 2

# COMMAND ----------

         
         
         
base_dados['val_renda__R__pe_10_g_1_c1_17_1'] = np.where(np.bitwise_and(base_dados['val_renda__R__pe_10_g_1'] == 0, base_dados['val_renda__S__p_7_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['val_renda__R__pe_10_g_1'] == 0, base_dados['val_renda__S__p_7_g_1'] == 1), 0,
np.where(np.bitwise_and(base_dados['val_renda__R__pe_10_g_1'] == 0, base_dados['val_renda__S__p_7_g_1'] == 2), 0,
np.where(np.bitwise_and(base_dados['val_renda__R__pe_10_g_1'] == 1, base_dados['val_renda__S__p_7_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['val_renda__R__pe_10_g_1'] == 1, base_dados['val_renda__S__p_7_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['val_renda__R__pe_10_g_1'] == 1, base_dados['val_renda__S__p_7_g_1'] == 2), 3,
 0))))))

base_dados['val_renda__R__pe_10_g_1_c1_17_2'] = np.where(base_dados['val_renda__R__pe_10_g_1_c1_17_1'] == 0, 0,
np.where(base_dados['val_renda__R__pe_10_g_1_c1_17_1'] == 1, 1,
np.where(base_dados['val_renda__R__pe_10_g_1_c1_17_1'] == 2, 2,
np.where(base_dados['val_renda__R__pe_10_g_1_c1_17_1'] == 3, 3,
1))))

base_dados['val_renda__R__pe_10_g_1_c1_17'] = np.where(base_dados['val_renda__R__pe_10_g_1_c1_17_2'] == 0, 0,
np.where(base_dados['val_renda__R__pe_10_g_1_c1_17_2'] == 1, 1,
np.where(base_dados['val_renda__R__pe_10_g_1_c1_17_2'] == 2, 2,
np.where(base_dados['val_renda__R__pe_10_g_1_c1_17_2'] == 3, 3,
 0))))
         
         
         
         
         
         
         
base_dados['mob_contrato__L__p_7_g_1_c1_24_1'] = np.where(np.bitwise_and(base_dados['mob_contrato__pe_6_g_1'] == 0, base_dados['mob_contrato__L__p_7_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['mob_contrato__pe_6_g_1'] == 0, base_dados['mob_contrato__L__p_7_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['mob_contrato__pe_6_g_1'] == 0, base_dados['mob_contrato__L__p_7_g_1'] == 2), 1,
np.where(np.bitwise_and(base_dados['mob_contrato__pe_6_g_1'] == 0, base_dados['mob_contrato__L__p_7_g_1'] == 3), 1,
np.where(np.bitwise_and(base_dados['mob_contrato__pe_6_g_1'] == 1, base_dados['mob_contrato__L__p_7_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['mob_contrato__pe_6_g_1'] == 1, base_dados['mob_contrato__L__p_7_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['mob_contrato__pe_6_g_1'] == 1, base_dados['mob_contrato__L__p_7_g_1'] == 2), 3,
np.where(np.bitwise_and(base_dados['mob_contrato__pe_6_g_1'] == 1, base_dados['mob_contrato__L__p_7_g_1'] == 3), 5,
np.where(np.bitwise_and(base_dados['mob_contrato__pe_6_g_1'] == 2, base_dados['mob_contrato__L__p_7_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['mob_contrato__pe_6_g_1'] == 2, base_dados['mob_contrato__L__p_7_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['mob_contrato__pe_6_g_1'] == 2, base_dados['mob_contrato__L__p_7_g_1'] == 2), 4,
np.where(np.bitwise_and(base_dados['mob_contrato__pe_6_g_1'] == 2, base_dados['mob_contrato__L__p_7_g_1'] == 3), 5,
 0))))))))))))
         
base_dados['mob_contrato__L__p_7_g_1_c1_24_2'] = np.where(base_dados['mob_contrato__L__p_7_g_1_c1_24_1'] == 0, 1,
np.where(base_dados['mob_contrato__L__p_7_g_1_c1_24_1'] == 1, 0,
np.where(base_dados['mob_contrato__L__p_7_g_1_c1_24_1'] == 2, 2,
np.where(base_dados['mob_contrato__L__p_7_g_1_c1_24_1'] == 3, 3,
np.where(base_dados['mob_contrato__L__p_7_g_1_c1_24_1'] == 4, 4,
np.where(base_dados['mob_contrato__L__p_7_g_1_c1_24_1'] == 5, 5,
0))))))
         
base_dados['mob_contrato__L__p_7_g_1_c1_24'] = np.where(base_dados['mob_contrato__L__p_7_g_1_c1_24_2'] == 0, 0,
np.where(base_dados['mob_contrato__L__p_7_g_1_c1_24_2'] == 1, 1,
np.where(base_dados['mob_contrato__L__p_7_g_1_c1_24_2'] == 2, 2,
np.where(base_dados['mob_contrato__L__p_7_g_1_c1_24_2'] == 3, 3,
np.where(base_dados['mob_contrato__L__p_7_g_1_c1_24_2'] == 4, 4,
np.where(base_dados['mob_contrato__L__p_7_g_1_c1_24_2'] == 5, 5,
 2))))))
         
         
         
         
         
         
         
base_dados['val_compr__L__p_2_g_1_c1_3_1'] = np.where(np.bitwise_and(base_dados['val_compr__L__p_6_g_1'] == 0, base_dados['val_compr__L__p_2_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['val_compr__L__p_6_g_1'] == 0, base_dados['val_compr__L__p_2_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['val_compr__L__p_6_g_1'] == 0, base_dados['val_compr__L__p_2_g_1'] == 2), 2,
np.where(np.bitwise_and(base_dados['val_compr__L__p_6_g_1'] == 1, base_dados['val_compr__L__p_2_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['val_compr__L__p_6_g_1'] == 1, base_dados['val_compr__L__p_2_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['val_compr__L__p_6_g_1'] == 1, base_dados['val_compr__L__p_2_g_1'] == 2), 2,
 0))))))

base_dados['val_compr__L__p_2_g_1_c1_3_2'] = np.where(base_dados['val_compr__L__p_2_g_1_c1_3_1'] == 0, 0,
np.where(base_dados['val_compr__L__p_2_g_1_c1_3_1'] == 1, 1,
np.where(base_dados['val_compr__L__p_2_g_1_c1_3_1'] == 2, 2,
0)))
         
base_dados['val_compr__L__p_2_g_1_c1_3'] = np.where(base_dados['val_compr__L__p_2_g_1_c1_3_2'] == 0, 0,
np.where(base_dados['val_compr__L__p_2_g_1_c1_3_2'] == 1, 1,
np.where(base_dados['val_compr__L__p_2_g_1_c1_3_2'] == 2, 2,
 1)))
         
         
         
         
         
         
         
base_dados['mob_cliente__R__p_6_g_1_c1_6_1'] = np.where(np.bitwise_and(base_dados['mob_cliente__R__p_6_g_1'] == 0, base_dados['mob_cliente__L__pe_13_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['mob_cliente__R__p_6_g_1'] == 0, base_dados['mob_cliente__L__pe_13_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['mob_cliente__R__p_6_g_1'] == 0, base_dados['mob_cliente__L__pe_13_g_1'] == 2), 1,
np.where(np.bitwise_and(base_dados['mob_cliente__R__p_6_g_1'] == 1, base_dados['mob_cliente__L__pe_13_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['mob_cliente__R__p_6_g_1'] == 1, base_dados['mob_cliente__L__pe_13_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['mob_cliente__R__p_6_g_1'] == 1, base_dados['mob_cliente__L__pe_13_g_1'] == 2), 3,
 0))))))

base_dados['mob_cliente__R__p_6_g_1_c1_6_2'] = np.where(base_dados['mob_cliente__R__p_6_g_1_c1_6_1'] == 0, 0,
np.where(base_dados['mob_cliente__R__p_6_g_1_c1_6_1'] == 1, 1,
np.where(base_dados['mob_cliente__R__p_6_g_1_c1_6_1'] == 2, 2,
np.where(base_dados['mob_cliente__R__p_6_g_1_c1_6_1'] == 3, 3,
 1))))
         
base_dados['mob_cliente__R__p_6_g_1_c1_6'] = np.where(base_dados['mob_cliente__R__p_6_g_1_c1_6_2'] == 0, 0,
np.where(base_dados['mob_cliente__R__p_6_g_1_c1_6_2'] == 1, 1,
np.where(base_dados['mob_cliente__R__p_6_g_1_c1_6_2'] == 2, 2,
np.where(base_dados['mob_cliente__R__p_6_g_1_c1_6_2'] == 3, 3,
 1))))
         
         
            
         
         
base_dados['mob_expir_prazo__S__p_13_g_1_c1_8_1'] = np.where(np.bitwise_and(base_dados['mob_expir_prazo__p_13_g_1'] == 0, base_dados['mob_expir_prazo__S__p_13_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['mob_expir_prazo__p_13_g_1'] == 0, base_dados['mob_expir_prazo__S__p_13_g_1'] == 1), 3,
np.where(np.bitwise_and(base_dados['mob_expir_prazo__p_13_g_1'] == 1, base_dados['mob_expir_prazo__S__p_13_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['mob_expir_prazo__p_13_g_1'] == 1, base_dados['mob_expir_prazo__S__p_13_g_1'] == 1), 3,
np.where(np.bitwise_and(base_dados['mob_expir_prazo__p_13_g_1'] == 2, base_dados['mob_expir_prazo__S__p_13_g_1'] == 0), 2,
np.where(np.bitwise_and(base_dados['mob_expir_prazo__p_13_g_1'] == 2, base_dados['mob_expir_prazo__S__p_13_g_1'] == 1), 3,
 0))))))
                                                        
base_dados['mob_expir_prazo__S__p_13_g_1_c1_8_2'] = np.where(base_dados['mob_expir_prazo__S__p_13_g_1_c1_8_1'] == 0, 0,
np.where(base_dados['mob_expir_prazo__S__p_13_g_1_c1_8_1'] == 1, 1,
np.where(base_dados['mob_expir_prazo__S__p_13_g_1_c1_8_1'] == 2, 3,
np.where(base_dados['mob_expir_prazo__S__p_13_g_1_c1_8_1'] == 3, 2,
0))))

base_dados['mob_expir_prazo__S__p_13_g_1_c1_8'] = np.where(base_dados['mob_expir_prazo__S__p_13_g_1_c1_8_2'] == 0, 0,
np.where(base_dados['mob_expir_prazo__S__p_13_g_1_c1_8_2'] == 1, 1,
np.where(base_dados['mob_expir_prazo__S__p_13_g_1_c1_8_2'] == 2, 2,
np.where(base_dados['mob_expir_prazo__S__p_13_g_1_c1_8_2'] == 3, 3,
0))))
         
         
         
         
                 
base_dados['val_taxa_contr__p_5_g_1_c1_6_1'] = np.where(np.bitwise_and(base_dados['val_taxa_contr__p_5_g_1'] == 0, base_dados['val_taxa_contr__L__p_4_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['val_taxa_contr__p_5_g_1'] == 0, base_dados['val_taxa_contr__L__p_4_g_1'] == 1), 0,
np.where(np.bitwise_and(base_dados['val_taxa_contr__p_5_g_1'] == 0, base_dados['val_taxa_contr__L__p_4_g_1'] == 2), 1,
np.where(np.bitwise_and(base_dados['val_taxa_contr__p_5_g_1'] == 1, base_dados['val_taxa_contr__L__p_4_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['val_taxa_contr__p_5_g_1'] == 1, base_dados['val_taxa_contr__L__p_4_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['val_taxa_contr__p_5_g_1'] == 1, base_dados['val_taxa_contr__L__p_4_g_1'] == 2), 2,
np.where(np.bitwise_and(base_dados['val_taxa_contr__p_5_g_1'] == 2, base_dados['val_taxa_contr__L__p_4_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['val_taxa_contr__p_5_g_1'] == 2, base_dados['val_taxa_contr__L__p_4_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['val_taxa_contr__p_5_g_1'] == 2, base_dados['val_taxa_contr__L__p_4_g_1'] == 2), 3,
 0)))))))))

base_dados['val_taxa_contr__p_5_g_1_c1_6_2'] = np.where(base_dados['val_taxa_contr__p_5_g_1_c1_6_1'] == 0, 1,
np.where(base_dados['val_taxa_contr__p_5_g_1_c1_6_1'] == 1, 0,
np.where(base_dados['val_taxa_contr__p_5_g_1_c1_6_1'] == 2, 2,
np.where(base_dados['val_taxa_contr__p_5_g_1_c1_6_1'] == 3, 3,
0))))

base_dados['val_taxa_contr__p_5_g_1_c1_6'] = np.where(base_dados['val_taxa_contr__p_5_g_1_c1_6_2'] == 0, 0,
np.where(base_dados['val_taxa_contr__p_5_g_1_c1_6_2'] == 1, 1,
np.where(base_dados['val_taxa_contr__p_5_g_1_c1_6_2'] == 2, 2,
np.where(base_dados['val_taxa_contr__p_5_g_1_c1_6_2'] == 3, 3,
0))))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Mantendo apenas as variáveis do modelo

# COMMAND ----------

arquivo_modelo_escolhido = os.listdir(os.path.join(credor.caminho_pickle_dbfs,modelo_escolhido))[0]
print (modelo_escolhido)
modelo=pickle.load(open(os.path.join(credor.caminho_pickle_dbfs, modelo_escolhido, arquivo_modelo_escolhido), 'rb'))

base_teste_c0 = base_dados[[chave,'mob_expir_prazo__S__p_13_g_1_c1_8', 'p_des_fones_resid_gh38', 'cod_produt_gh38', 'val_taxa_contr__p_5_g_1_c1_6', 'mob_contrato__L__p_7_g_1_c1_24', 'val_compr__L__p_2_g_1_c1_3', 'mob_cliente__R__p_6_g_1_c1_6', 'qtd_prest_gh38', 'mod_venda_gh38', 'val_renda__R__pe_10_g_1_c1_17']]

var_fin_c0=['mob_expir_prazo__S__p_13_g_1_c1_8', 'p_des_fones_resid_gh38', 'cod_produt_gh38', 'val_taxa_contr__p_5_g_1_c1_6', 'mob_contrato__L__p_7_g_1_c1_24', 'val_compr__L__p_2_g_1_c1_3', 'mob_cliente__R__p_6_g_1_c1_6', 'qtd_prest_gh38', 'mod_venda_gh38', 'val_renda__R__pe_10_g_1_c1_17']

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

x_teste2['P_1_R_p_4_g_1'] = np.where(x_teste2['P_1_R'] <= 0.488650699, 0.0,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.488650699, x_teste2['P_1_R'] <= 0.618300769), 1.0,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.618300769, x_teste2['P_1_R'] <= 0.677399541), 2.0,
    np.where(x_teste2['P_1_R'] > 0.677399541,3,0))))

x_teste2['P_1_R_p_8_g_1'] = np.where(x_teste2['P_1'] <= 0.446037277, 0.0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.446037277, x_teste2['P_1'] <= 0.488650699), 1.0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.488650699, x_teste2['P_1'] <= 0.546212642), 2.0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.546212642, x_teste2['P_1'] <= 0.618300769), 3.0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.618300769, x_teste2['P_1'] <= 0.653195335), 4.0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.653195335, x_teste2['P_1'] <= 0.677399541), 5.0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.677399541, x_teste2['P_1'] <= 0.708397762), 6.0,
    np.where(x_teste2['P_1'] > 0.708397762, 7,0))))))))

x_teste2['GH'] = np.where(np.bitwise_and(x_teste2['P_1_R_p_4_g_1'] == 0, x_teste2['P_1_R_p_8_g_1'] == 0), 0,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_4_g_1'] == 0, x_teste2['P_1_R_p_8_g_1'] == 1), 1,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_4_g_1'] == 0, x_teste2['P_1_R_p_8_g_1'] == 2), 1,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_4_g_1'] == 0, x_teste2['P_1_R_p_8_g_1'] == 3), 2,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_4_g_1'] == 0, x_teste2['P_1_R_p_8_g_1'] == 4), 2,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_4_g_1'] == 0, x_teste2['P_1_R_p_8_g_1'] == 5), 3,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_4_g_1'] == 0, x_teste2['P_1_R_p_8_g_1'] == 6), 3,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_4_g_1'] == 0, x_teste2['P_1_R_p_8_g_1'] == 7), 4,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_4_g_1'] == 1, x_teste2['P_1_R_p_8_g_1'] == 0), 1,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_4_g_1'] == 1, x_teste2['P_1_R_p_8_g_1'] == 1), 2,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_4_g_1'] == 1, x_teste2['P_1_R_p_8_g_1'] == 2), 2,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_4_g_1'] == 1, x_teste2['P_1_R_p_8_g_1'] == 3), 3,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_4_g_1'] == 1, x_teste2['P_1_R_p_8_g_1'] == 4), 3,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_4_g_1'] == 1, x_teste2['P_1_R_p_8_g_1'] == 5), 4,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_4_g_1'] == 1, x_teste2['P_1_R_p_8_g_1'] == 6), 4,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_4_g_1'] == 1, x_teste2['P_1_R_p_8_g_1'] == 7), 5,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_4_g_1'] == 2, x_teste2['P_1_R_p_8_g_1'] == 0), 2,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_4_g_1'] == 2, x_teste2['P_1_R_p_8_g_1'] == 1), 3,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_4_g_1'] == 2, x_teste2['P_1_R_p_8_g_1'] == 2), 3,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_4_g_1'] == 2, x_teste2['P_1_R_p_8_g_1'] == 3), 4,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_4_g_1'] == 2, x_teste2['P_1_R_p_8_g_1'] == 4), 4,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_4_g_1'] == 2, x_teste2['P_1_R_p_8_g_1'] == 5), 5,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_4_g_1'] == 2, x_teste2['P_1_R_p_8_g_1'] == 6), 5,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_4_g_1'] == 2, x_teste2['P_1_R_p_8_g_1'] == 7), 6,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_4_g_1'] == 3, x_teste2['P_1_R_p_8_g_1'] == 0), 3,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_4_g_1'] == 3, x_teste2['P_1_R_p_8_g_1'] == 1), 3,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_4_g_1'] == 3, x_teste2['P_1_R_p_8_g_1'] == 2), 4,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_4_g_1'] == 3, x_teste2['P_1_R_p_8_g_1'] == 3), 4,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_4_g_1'] == 3, x_teste2['P_1_R_p_8_g_1'] == 4), 5,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_4_g_1'] == 3, x_teste2['P_1_R_p_8_g_1'] == 5), 6,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_4_g_1'] == 3, x_teste2['P_1_R_p_8_g_1'] == 6), 6,
    np.where(np.bitwise_and(x_teste2['P_1_R_p_4_g_1'] == 3, x_teste2['P_1_R_p_8_g_1'] == 7), 7,
             0))))))))))))))))))))))))))))))))

del x_teste2['P_1_R']
del x_teste2['P_1_R_p_4_g_1']
del x_teste2['P_1_R_p_8_g_1']

x_teste2


# COMMAND ----------

# DBTITLE 1,escrevendo
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