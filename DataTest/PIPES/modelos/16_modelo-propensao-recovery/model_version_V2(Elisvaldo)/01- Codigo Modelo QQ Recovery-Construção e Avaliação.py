# Databricks notebook source
# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

blob_account_source_ml = "qqdatastoragemain"
blob_container_source_ml = "ml-prd"

mount_blob_storage_key(dbutils,blob_account_source_ml,blob_container_source_ml,'/mnt/ml-prd')

# COMMAND ----------

# MAGIC %md
# MAGIC # <font color='blue'>IA - Feature Selection</font>
# MAGIC 
# MAGIC # <font color='blue'>Ferramenta de Criação de Variáveis</font>

# COMMAND ----------

# DBTITLE 1,Importando os pacotes Python
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

# DBTITLE 1,Inserindo hiperparâmetros do Algoritmo
## Parâmetros do Algoritmo

#Variável chave-primaria
chave = 'Chave'

#Variável resposta ou target
target = 'Deals_30'

#Lista com a variável Tempo
var_tmp = 'Data de referência'

#Nome da Base de Dados
N_Base = "Amostra formatada para execução do modelo.csv"

#Caminho da base de dados
caminho_base = "/dbfs/mnt/ml-prd/ml-data/propensaodeal/recovery/databases/"

#Separador
separador_ = ";"

#Decimal
decimal_ = "."

# COMMAND ----------

# DBTITLE 1,Importação da Base de Dados
#carregar o arquivo em formato tabela
base_dados = pd.read_csv(caminho_base+N_Base, sep=separador_, decimal=decimal_)
base_dados[target] = base_dados[target].map({True:1,False:0},na_action=None)
print("shape da Base de Dados:",base_dados.shape)

base_dados.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gerando variáveis Categóricas

# COMMAND ----------

base_dados['Class_Portfolio_gh30'] = np.where(base_dados['Class_Portfolio'] == 'a', 0,
    np.where(base_dados['Class_Portfolio'] == 'b', 1,
    np.where(base_dados['Class_Portfolio'] == 'c', 2,
    np.where(base_dados['Class_Portfolio'] == 'd', 3,
    np.where(base_dados['Class_Portfolio'] == 'e', 4,
    np.where(base_dados['Class_Portfolio'] == 'f', 5,
    np.where(base_dados['Class_Portfolio'] == 'g', 6,
    np.where(base_dados['Class_Portfolio'] == 'h', 7,
    np.where(base_dados['Class_Portfolio'] == 'i', 8,
    np.where(base_dados['Class_Portfolio'] == 'j', 9,
    np.where(base_dados['Class_Portfolio'] == 'k', 10,
    0)))))))))))

base_dados['Class_Portfolio_gh31'] = np.where(base_dados['Class_Portfolio_gh30'] == 0, 0,
    np.where(base_dados['Class_Portfolio_gh30'] == 1, 1,
    np.where(base_dados['Class_Portfolio_gh30'] == 2, 1,
    np.where(base_dados['Class_Portfolio_gh30'] == 3, 3,
    np.where(base_dados['Class_Portfolio_gh30'] == 4, 4,
    np.where(base_dados['Class_Portfolio_gh30'] == 5, 5,
    np.where(base_dados['Class_Portfolio_gh30'] == 6, 6,
    np.where(base_dados['Class_Portfolio_gh30'] == 7, 7,
    np.where(base_dados['Class_Portfolio_gh30'] == 8, 7,
    np.where(base_dados['Class_Portfolio_gh30'] == 9, 9,
    np.where(base_dados['Class_Portfolio_gh30'] == 10, 10,
    0)))))))))))

base_dados['Class_Portfolio_gh32'] = np.where(base_dados['Class_Portfolio_gh31'] == 0, 0,
    np.where(base_dados['Class_Portfolio_gh31'] == 1, 1,
    np.where(base_dados['Class_Portfolio_gh31'] == 3, 2,
    np.where(base_dados['Class_Portfolio_gh31'] == 4, 3,
    np.where(base_dados['Class_Portfolio_gh31'] == 5, 4,
    np.where(base_dados['Class_Portfolio_gh31'] == 6, 5,
    np.where(base_dados['Class_Portfolio_gh31'] == 7, 6,
    np.where(base_dados['Class_Portfolio_gh31'] == 9, 7,
    np.where(base_dados['Class_Portfolio_gh31'] == 10, 8,
    0)))))))))

base_dados['Class_Portfolio_gh33'] = np.where(base_dados['Class_Portfolio_gh32'] == 0, 0,
    np.where(base_dados['Class_Portfolio_gh32'] == 1, 1,
    np.where(base_dados['Class_Portfolio_gh32'] == 2, 2,
    np.where(base_dados['Class_Portfolio_gh32'] == 3, 3,
    np.where(base_dados['Class_Portfolio_gh32'] == 4, 4,
    np.where(base_dados['Class_Portfolio_gh32'] == 5, 5,
    np.where(base_dados['Class_Portfolio_gh32'] == 6, 6,
    np.where(base_dados['Class_Portfolio_gh32'] == 7, 7,
    np.where(base_dados['Class_Portfolio_gh32'] == 8, 8,
    0)))))))))

base_dados['Class_Portfolio_gh34'] = np.where(base_dados['Class_Portfolio_gh33'] == 0, 0,
    np.where(base_dados['Class_Portfolio_gh33'] == 1, 1,
    np.where(base_dados['Class_Portfolio_gh33'] == 2, 2,
    np.where(base_dados['Class_Portfolio_gh33'] == 3, 3,
    np.where(base_dados['Class_Portfolio_gh33'] == 4, 0,
    np.where(base_dados['Class_Portfolio_gh33'] == 5, 1,
    np.where(base_dados['Class_Portfolio_gh33'] == 6, 3,
    np.where(base_dados['Class_Portfolio_gh33'] == 7, 2,
    np.where(base_dados['Class_Portfolio_gh33'] == 8, 8,
    0)))))))))

base_dados['Class_Portfolio_gh35'] = np.where(base_dados['Class_Portfolio_gh34'] == 0, 0,
    np.where(base_dados['Class_Portfolio_gh34'] == 1, 1,
    np.where(base_dados['Class_Portfolio_gh34'] == 2, 2,
    np.where(base_dados['Class_Portfolio_gh34'] == 3, 3,
    np.where(base_dados['Class_Portfolio_gh34'] == 8, 4,
    0)))))

base_dados['Class_Portfolio_gh36'] = np.where(base_dados['Class_Portfolio_gh35'] == 0, 0,
    np.where(base_dados['Class_Portfolio_gh35'] == 1, 1,
    np.where(base_dados['Class_Portfolio_gh35'] == 2, 2,
    np.where(base_dados['Class_Portfolio_gh35'] == 3, 3,
    np.where(base_dados['Class_Portfolio_gh35'] == 4, 3,
    0)))))

base_dados['Class_Portfolio_gh37'] = np.where(base_dados['Class_Portfolio_gh36'] == 0, 0,
    np.where(base_dados['Class_Portfolio_gh36'] == 1, 1,
    np.where(base_dados['Class_Portfolio_gh36'] == 2, 1,
    np.where(base_dados['Class_Portfolio_gh36'] == 3, 3,
    0))))

base_dados['Class_Portfolio_gh38'] = np.where(base_dados['Class_Portfolio_gh37'] == 0, 0,
    np.where(base_dados['Class_Portfolio_gh37'] == 1, 1,
    np.where(base_dados['Class_Portfolio_gh37'] == 3, 2,
    0)))





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
    np.where(base_dados['Class_Carteira_gh30'] == 7, 6,
    np.where(base_dados['Class_Carteira_gh30'] == 8, 6,
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
    np.where(base_dados['Class_Carteira_gh31'] == 9, 7,
    np.where(base_dados['Class_Carteira_gh31'] == 10, 8,
    0)))))))))

base_dados['Class_Carteira_gh33'] = np.where(base_dados['Class_Carteira_gh32'] == 0, 0,
    np.where(base_dados['Class_Carteira_gh32'] == 1, 1,
    np.where(base_dados['Class_Carteira_gh32'] == 2, 2,
    np.where(base_dados['Class_Carteira_gh32'] == 3, 3,
    np.where(base_dados['Class_Carteira_gh32'] == 4, 4,
    np.where(base_dados['Class_Carteira_gh32'] == 5, 5,
    np.where(base_dados['Class_Carteira_gh32'] == 6, 6,
    np.where(base_dados['Class_Carteira_gh32'] == 7, 7,
    np.where(base_dados['Class_Carteira_gh32'] == 8, 8,
    0)))))))))

base_dados['Class_Carteira_gh34'] = np.where(base_dados['Class_Carteira_gh33'] == 0, 0,
    np.where(base_dados['Class_Carteira_gh33'] == 1, 1,
    np.where(base_dados['Class_Carteira_gh33'] == 2, 2,
    np.where(base_dados['Class_Carteira_gh33'] == 3, 0,
    np.where(base_dados['Class_Carteira_gh33'] == 4, 0,
    np.where(base_dados['Class_Carteira_gh33'] == 5, 0,
    np.where(base_dados['Class_Carteira_gh33'] == 6, 6,
    np.where(base_dados['Class_Carteira_gh33'] == 7, 8,
    np.where(base_dados['Class_Carteira_gh33'] == 8, 8,
    0)))))))))

base_dados['Class_Carteira_gh35'] = np.where(base_dados['Class_Carteira_gh34'] == 0, 0,
    np.where(base_dados['Class_Carteira_gh34'] == 1, 1,
    np.where(base_dados['Class_Carteira_gh34'] == 2, 2,
    np.where(base_dados['Class_Carteira_gh34'] == 6, 3,
    np.where(base_dados['Class_Carteira_gh34'] == 8, 4,
    0)))))

base_dados['Class_Carteira_gh36'] = np.where(base_dados['Class_Carteira_gh35'] == 0, 0,
    np.where(base_dados['Class_Carteira_gh35'] == 1, 1,
    np.where(base_dados['Class_Carteira_gh35'] == 2, 2,
    np.where(base_dados['Class_Carteira_gh35'] == 3, 2,
    np.where(base_dados['Class_Carteira_gh35'] == 4, 4,
    0)))))

base_dados['Class_Carteira_gh37'] = np.where(base_dados['Class_Carteira_gh36'] == 0, 0,
    np.where(base_dados['Class_Carteira_gh36'] == 1, 1,
    np.where(base_dados['Class_Carteira_gh36'] == 2, 2,
    np.where(base_dados['Class_Carteira_gh36'] == 4, 3,
    0))))

base_dados['Class_Carteira_gh38'] = np.where(base_dados['Class_Carteira_gh37'] == 0, 0,
    np.where(base_dados['Class_Carteira_gh37'] == 1, 1,
    np.where(base_dados['Class_Carteira_gh37'] == 2, 2,
    np.where(base_dados['Class_Carteira_gh37'] == 3, 3,
    0))))




base_dados['Class_Produto_gh30'] = np.where(base_dados['Class_Produto'] == 'a', 0,
    np.where(base_dados['Class_Produto'] == 'b', 1,
    np.where(base_dados['Class_Produto'] == 'c', 2,
    np.where(base_dados['Class_Produto'] == 'd', 3,
    np.where(base_dados['Class_Produto'] == 'e', 4,
    np.where(base_dados['Class_Produto'] == 'f', 5,
    np.where(base_dados['Class_Produto'] == 'g', 6,
    np.where(base_dados['Class_Produto'] == 'h', 7,
    np.where(base_dados['Class_Produto'] == 'i', 8,
    np.where(base_dados['Class_Produto'] == 'j', 9,
    np.where(base_dados['Class_Produto'] == 'k', 10,
    0)))))))))))

base_dados['Class_Produto_gh31'] = np.where(base_dados['Class_Produto_gh30'] == 0, 0,
    np.where(base_dados['Class_Produto_gh30'] == 1, 1,
    np.where(base_dados['Class_Produto_gh30'] == 2, 2,
    np.where(base_dados['Class_Produto_gh30'] == 3, 3,
    np.where(base_dados['Class_Produto_gh30'] == 4, 3,
    np.where(base_dados['Class_Produto_gh30'] == 5, 5,
    np.where(base_dados['Class_Produto_gh30'] == 6, 6,
    np.where(base_dados['Class_Produto_gh30'] == 7, 7,
    np.where(base_dados['Class_Produto_gh30'] == 8, 8,
    np.where(base_dados['Class_Produto_gh30'] == 9, 8,
    np.where(base_dados['Class_Produto_gh30'] == 10, 10,
    0)))))))))))

base_dados['Class_Produto_gh32'] = np.where(base_dados['Class_Produto_gh31'] == 0, 0,
    np.where(base_dados['Class_Produto_gh31'] == 1, 1,
    np.where(base_dados['Class_Produto_gh31'] == 2, 2,
    np.where(base_dados['Class_Produto_gh31'] == 3, 3,
    np.where(base_dados['Class_Produto_gh31'] == 5, 4,
    np.where(base_dados['Class_Produto_gh31'] == 6, 5,
    np.where(base_dados['Class_Produto_gh31'] == 7, 6,
    np.where(base_dados['Class_Produto_gh31'] == 8, 7,
    np.where(base_dados['Class_Produto_gh31'] == 10, 8,
    0)))))))))

base_dados['Class_Produto_gh33'] = np.where(base_dados['Class_Produto_gh32'] == 0, 0,
    np.where(base_dados['Class_Produto_gh32'] == 1, 1,
    np.where(base_dados['Class_Produto_gh32'] == 2, 2,
    np.where(base_dados['Class_Produto_gh32'] == 3, 3,
    np.where(base_dados['Class_Produto_gh32'] == 4, 4,
    np.where(base_dados['Class_Produto_gh32'] == 5, 5,
    np.where(base_dados['Class_Produto_gh32'] == 6, 6,
    np.where(base_dados['Class_Produto_gh32'] == 7, 7,
    np.where(base_dados['Class_Produto_gh32'] == 8, 8,
    0)))))))))

base_dados['Class_Produto_gh34'] = np.where(base_dados['Class_Produto_gh33'] == 0, 0,
    np.where(base_dados['Class_Produto_gh33'] == 1, 1,
    np.where(base_dados['Class_Produto_gh33'] == 2, 2,
    np.where(base_dados['Class_Produto_gh33'] == 3, 3,
    np.where(base_dados['Class_Produto_gh33'] == 4, 4,
    np.where(base_dados['Class_Produto_gh33'] == 5, 5,
    np.where(base_dados['Class_Produto_gh33'] == 6, 6,
    np.where(base_dados['Class_Produto_gh33'] == 7, 7,
    np.where(base_dados['Class_Produto_gh33'] == 8, 8,
    0)))))))))

base_dados['Class_Produto_gh35'] = np.where(base_dados['Class_Produto_gh34'] == 0, 0,
    np.where(base_dados['Class_Produto_gh34'] == 1, 1,
    np.where(base_dados['Class_Produto_gh34'] == 2, 2,
    np.where(base_dados['Class_Produto_gh34'] == 3, 3,
    np.where(base_dados['Class_Produto_gh34'] == 4, 4,
    np.where(base_dados['Class_Produto_gh34'] == 5, 5,
    np.where(base_dados['Class_Produto_gh34'] == 6, 6,
    np.where(base_dados['Class_Produto_gh34'] == 7, 7,
    np.where(base_dados['Class_Produto_gh34'] == 8, 8,
    0)))))))))

base_dados['Class_Produto_gh36'] = np.where(base_dados['Class_Produto_gh35'] == 0, 2,
    np.where(base_dados['Class_Produto_gh35'] == 1, 0,
    np.where(base_dados['Class_Produto_gh35'] == 2, 2,
    np.where(base_dados['Class_Produto_gh35'] == 3, 0,
    np.where(base_dados['Class_Produto_gh35'] == 4, 2,
    np.where(base_dados['Class_Produto_gh35'] == 5, 6,
    np.where(base_dados['Class_Produto_gh35'] == 6, 8,
    np.where(base_dados['Class_Produto_gh35'] == 7, 5,
    np.where(base_dados['Class_Produto_gh35'] == 8, 7,
    0)))))))))

base_dados['Class_Produto_gh37'] = np.where(base_dados['Class_Produto_gh36'] == 0, 0,
    np.where(base_dados['Class_Produto_gh36'] == 2, 1,
    np.where(base_dados['Class_Produto_gh36'] == 5, 1,
    np.where(base_dados['Class_Produto_gh36'] == 6, 1,
    np.where(base_dados['Class_Produto_gh36'] == 7, 4,
    np.where(base_dados['Class_Produto_gh36'] == 8, 5,
    0))))))

base_dados['Class_Produto_gh38'] = np.where(base_dados['Class_Produto_gh37'] == 0, 0,
    np.where(base_dados['Class_Produto_gh37'] == 1, 1,
    np.where(base_dados['Class_Produto_gh37'] == 4, 2,
    np.where(base_dados['Class_Produto_gh37'] == 5, 3,
    0))))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Gerando variáveis numéricas contínuas

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parte 1 de 2

# COMMAND ----------


base_dados['IdContatoSIR__pe_7'] = np.where(base_dados['IdContatoSIR'] <= 5328414.0, 0.0,
np.where(np.bitwise_and(base_dados['IdContatoSIR'] > 5328414.0, base_dados['IdContatoSIR'] <= 10677982.0), 1.0,
np.where(np.bitwise_and(base_dados['IdContatoSIR'] > 10677982.0, base_dados['IdContatoSIR'] <= 16069056.0), 2.0,
np.where(np.bitwise_and(base_dados['IdContatoSIR'] > 16069056.0, base_dados['IdContatoSIR'] <= 21407862.0), 3.0,
np.where(np.bitwise_and(base_dados['IdContatoSIR'] > 21407862.0, base_dados['IdContatoSIR'] <= 26791196.0), 4.0,
np.where(np.bitwise_and(base_dados['IdContatoSIR'] > 26791196.0, base_dados['IdContatoSIR'] <= 32151023.0), 5.0,
np.where(base_dados['IdContatoSIR'] > 32151023.0, 6.0,
 -2)))))))

base_dados['IdContatoSIR__pe_7_g_1_1'] = np.where(base_dados['IdContatoSIR__pe_7'] == -2.0, 1,
np.where(base_dados['IdContatoSIR__pe_7'] == 0.0, 1,
np.where(base_dados['IdContatoSIR__pe_7'] == 1.0, 0,
np.where(base_dados['IdContatoSIR__pe_7'] == 2.0, 0,
np.where(base_dados['IdContatoSIR__pe_7'] == 3.0, 0,
np.where(base_dados['IdContatoSIR__pe_7'] == 4.0, 1,
np.where(base_dados['IdContatoSIR__pe_7'] == 5.0, 1,
np.where(base_dados['IdContatoSIR__pe_7'] == 6.0, 0,
 0))))))))

base_dados['IdContatoSIR__pe_7_g_1_2'] = np.where(base_dados['IdContatoSIR__pe_7_g_1_1'] == 0, 1,
np.where(base_dados['IdContatoSIR__pe_7_g_1_1'] == 1, 0,
 0))

base_dados['IdContatoSIR__pe_7_g_1'] = np.where(base_dados['IdContatoSIR__pe_7_g_1_2'] == 0, 0,
np.where(base_dados['IdContatoSIR__pe_7_g_1_2'] == 1, 1,
 0))



                                                
base_dados['IdContatoSIR__L'] = np.log(base_dados['IdContatoSIR'])                                                
np.where(base_dados['IdContatoSIR__L'] == 0, -1, base_dados['IdContatoSIR__L'])
base_dados['IdContatoSIR__L'] = base_dados['IdContatoSIR__L'].fillna(-2)
base_dados['IdContatoSIR__L__p_8'] = np.where(base_dados['IdContatoSIR__L'] <= 15.292152121848469, 0.0,
np.where(np.bitwise_and(base_dados['IdContatoSIR__L'] > 15.292152121848469, base_dados['IdContatoSIR__L'] <= 15.96208619467597), 1.0,
np.where(np.bitwise_and(base_dados['IdContatoSIR__L'] > 15.96208619467597, base_dados['IdContatoSIR__L'] <= 16.416303260087805), 2.0,
np.where(np.bitwise_and(base_dados['IdContatoSIR__L'] > 16.416303260087805, base_dados['IdContatoSIR__L'] <= 16.65322598770111), 3.0,
np.where(np.bitwise_and(base_dados['IdContatoSIR__L'] > 16.65322598770111, base_dados['IdContatoSIR__L'] <= 16.902009158185486), 4.0,
np.where(np.bitwise_and(base_dados['IdContatoSIR__L'] > 16.902009158185486, base_dados['IdContatoSIR__L'] <= 17.08092552010273), 5.0,
np.where(np.bitwise_and(base_dados['IdContatoSIR__L'] > 17.08092552010273, base_dados['IdContatoSIR__L'] <= 17.256770958197293), 6.0,
np.where(base_dados['IdContatoSIR__L'] > 17.256770958197293, 7.0,
 0))))))))
         
base_dados['IdContatoSIR__L__p_8_g_1_1'] = np.where(base_dados['IdContatoSIR__L__p_8'] == 0, 3,
np.where(base_dados['IdContatoSIR__L__p_8'] == 1, 0,
np.where(base_dados['IdContatoSIR__L__p_8'] == 2, 0,
np.where(base_dados['IdContatoSIR__L__p_8'] == 3, 0,
np.where(base_dados['IdContatoSIR__L__p_8'] == 4, 0,
np.where(base_dados['IdContatoSIR__L__p_8'] == 5, 4,
np.where(base_dados['IdContatoSIR__L__p_8'] == 6, 2,
np.where(base_dados['IdContatoSIR__L__p_8'] == 7, 1,
 0))))))))
         
base_dados['IdContatoSIR__L__p_8_g_1_2'] = np.where(base_dados['IdContatoSIR__L__p_8_g_1_1'] == 0, 2,
np.where(base_dados['IdContatoSIR__L__p_8_g_1_1'] == 1, 4,
np.where(base_dados['IdContatoSIR__L__p_8_g_1_1'] == 2, 0,
np.where(base_dados['IdContatoSIR__L__p_8_g_1_1'] == 3, 3,
np.where(base_dados['IdContatoSIR__L__p_8_g_1_1'] == 4, 1,
 0)))))
         
base_dados['IdContatoSIR__L__p_8_g_1'] = np.where(base_dados['IdContatoSIR__L__p_8_g_1_2'] == 0, 0,
np.where(base_dados['IdContatoSIR__L__p_8_g_1_2'] == 1, 1,
np.where(base_dados['IdContatoSIR__L__p_8_g_1_2'] == 2, 2,
np.where(base_dados['IdContatoSIR__L__p_8_g_1_2'] == 3, 3,
np.where(base_dados['IdContatoSIR__L__p_8_g_1_2'] == 4, 4,
 0)))))
         
         
         
         
         
base_dados['CPF__pe_5'] = np.where(base_dados['CPF'] <= 19329563872.0, 0.0,
np.where(np.bitwise_and(base_dados['CPF'] > 19329563872.0, base_dados['CPF'] <= 38661506875.0), 1.0,
np.where(np.bitwise_and(base_dados['CPF'] > 38661506875.0, base_dados['CPF'] <= 58031693072.0), 2.0,
np.where(np.bitwise_and(base_dados['CPF'] > 58031693072.0, base_dados['CPF'] <= 77287266415.0), 3.0,
np.where(np.bitwise_and(base_dados['CPF'] > 77287266415.0, base_dados['CPF'] <= 96811242353.0), 4.0,
 0)))))
         
base_dados['CPF__pe_5_g_1_1'] = np.where(base_dados['CPF__pe_5'] == -2.0, 2,
np.where(base_dados['CPF__pe_5'] == 0.0, 0,
np.where(base_dados['CPF__pe_5'] == 1.0, 1,
np.where(base_dados['CPF__pe_5'] == 2.0, 2,
np.where(base_dados['CPF__pe_5'] == 3.0, 0,
np.where(base_dados['CPF__pe_5'] == 4.0, 2,
 0))))))
         
base_dados['CPF__pe_5_g_1_2'] = np.where(base_dados['CPF__pe_5_g_1_1'] == 0, 2,
np.where(base_dados['CPF__pe_5_g_1_1'] == 1, 1,
np.where(base_dados['CPF__pe_5_g_1_1'] == 2, 0,
 0)))
         
base_dados['CPF__pe_5_g_1'] = np.where(base_dados['CPF__pe_5_g_1_2'] == 0, 0,
np.where(base_dados['CPF__pe_5_g_1_2'] == 1, 1,
np.where(base_dados['CPF__pe_5_g_1_2'] == 2, 2,
 0)))
         
         
         
         
               
base_dados['CPF__pe_4'] = np.where(base_dados['CPF'] <= 24065862515.0, 0.0,
np.where(np.bitwise_and(base_dados['CPF'] > 24065862515.0, base_dados['CPF'] <= 48276243720.0), 1.0,
np.where(np.bitwise_and(base_dados['CPF'] > 48276243720.0, base_dados['CPF'] <= 72084286204.0), 2.0,
np.where(np.bitwise_and(base_dados['CPF'] > 72084286204.0, base_dados['CPF'] <= 96811242353.0), 3.0,
 0))))
         
base_dados['CPF__pe_4_g_1_1'] = np.where(base_dados['CPF__pe_4'] == -2.0, 1,
np.where(base_dados['CPF__pe_4'] == 0.0, 0,
np.where(base_dados['CPF__pe_4'] == 1.0, 0,
np.where(base_dados['CPF__pe_4'] == 2.0, 0,
np.where(base_dados['CPF__pe_4'] == 3.0, 1,
 0)))))
         
base_dados['CPF__pe_4_g_1_2'] = np.where(base_dados['CPF__pe_4_g_1_1'] == 0, 1,
np.where(base_dados['CPF__pe_4_g_1_1'] == 1, 0,
 0))
                                         
base_dados['CPF__pe_4_g_1'] = np.where(base_dados['CPF__pe_4_g_1_2'] == 0, 0,
np.where(base_dados['CPF__pe_4_g_1_2'] == 1, 1,
 0))
                                       
                                       
                                       
                                       
                                       
                                       
base_dados['Aging__pe_10'] = np.where(base_dados['Aging'] <= 725.0, 0.0,
np.where(np.bitwise_and(base_dados['Aging'] > 725.0, base_dados['Aging'] <= 1460.0), 1.0,
np.where(np.bitwise_and(base_dados['Aging'] > 1460.0, base_dados['Aging'] <= 2184.0), 2.0,
np.where(np.bitwise_and(base_dados['Aging'] > 2184.0, base_dados['Aging'] <= 2900.0), 3.0,
np.where(np.bitwise_and(base_dados['Aging'] > 2900.0, base_dados['Aging'] <= 3652.0), 4.0,
np.where(np.bitwise_and(base_dados['Aging'] > 3652.0, base_dados['Aging'] <= 4380.0), 5.0,
np.where(np.bitwise_and(base_dados['Aging'] > 4380.0, base_dados['Aging'] <= 5110.0), 6.0,
np.where(np.bitwise_and(base_dados['Aging'] > 5110.0, base_dados['Aging'] <= 5778.0), 7.0,
np.where(np.bitwise_and(base_dados['Aging'] > 5778.0, base_dados['Aging'] <= 6573.0), 8.0,
np.where(np.bitwise_and(base_dados['Aging'] > 6573.0, base_dados['Aging'] <= 7303.0), 9.0,
 0))))))))))
         
base_dados['Aging__pe_10_g_1_1'] = np.where(base_dados['Aging__pe_10'] == -2.0, 2,
np.where(base_dados['Aging__pe_10'] == 0.0, 1,
np.where(base_dados['Aging__pe_10'] == 1.0, 1,
np.where(base_dados['Aging__pe_10'] == 2.0, 2,
np.where(base_dados['Aging__pe_10'] == 3.0, 0,
np.where(base_dados['Aging__pe_10'] == 4.0, 1,
np.where(base_dados['Aging__pe_10'] == 5.0, 2,
np.where(base_dados['Aging__pe_10'] == 6.0, 1,
np.where(base_dados['Aging__pe_10'] == 7.0, 1,
np.where(base_dados['Aging__pe_10'] == 8.0, 2,
np.where(base_dados['Aging__pe_10'] == 9.0, 1,
 0)))))))))))
         
base_dados['Aging__pe_10_g_1_2'] = np.where(base_dados['Aging__pe_10_g_1_1'] == 0, 0,
np.where(base_dados['Aging__pe_10_g_1_1'] == 1, 2,
np.where(base_dados['Aging__pe_10_g_1_1'] == 2, 1,
 0)))
         
base_dados['Aging__pe_10_g_1'] = np.where(base_dados['Aging__pe_10_g_1_2'] == 0, 0,
np.where(base_dados['Aging__pe_10_g_1_2'] == 1, 1,
np.where(base_dados['Aging__pe_10_g_1_2'] == 2, 2,
 0)))
         
         
         
         
                  
base_dados['Aging__L'] = np.log(base_dados['Aging'])
np.where(base_dados['Aging__L'] == 0, -1, base_dados['Aging__L'])
base_dados['Aging__L'] = base_dados['Aging__L'].fillna(-2)
base_dados['Aging__L__p_25'] = np.where(base_dados['Aging__L'] <= 6.018593214496234, 0.0,
np.where(np.bitwise_and(base_dados['Aging__L'] > 6.018593214496234, base_dados['Aging__L'] <= 6.408528791059498), 1.0,
np.where(np.bitwise_and(base_dados['Aging__L'] > 6.408528791059498, base_dados['Aging__L'] <= 6.826545223556594), 2.0,
np.where(np.bitwise_and(base_dados['Aging__L'] > 6.826545223556594, base_dados['Aging__L'] <= 7.286191714702382), 3.0,
np.where(np.bitwise_and(base_dados['Aging__L'] > 7.286191714702382, base_dados['Aging__L'] <= 7.542213463193403), 4.0,
np.where(np.bitwise_and(base_dados['Aging__L'] > 7.542213463193403, base_dados['Aging__L'] <= 7.638679823876112), 5.0,
np.where(np.bitwise_and(base_dados['Aging__L'] > 7.638679823876112, base_dados['Aging__L'] <= 7.72356247227797), 6.0,
np.where(np.bitwise_and(base_dados['Aging__L'] > 7.72356247227797, base_dados['Aging__L'] <= 7.8119734296220225), 7.0,
np.where(np.bitwise_and(base_dados['Aging__L'] > 7.8119734296220225, base_dados['Aging__L'] <= 8.552946361122055), 20.0,
np.where(np.bitwise_and(base_dados['Aging__L'] > 8.552946361122055, base_dados['Aging__L'] <= 8.7417757069247), 22.0,
np.where(np.bitwise_and(base_dados['Aging__L'] > 8.7417757069247, base_dados['Aging__L'] <= 8.852950887099581), 23.0,
np.where(base_dados['Aging__L'] > 8.852950887099581, 24.0,
 0))))))))))))
         
base_dados['Aging__L__p_25_g_1_1'] = np.where(base_dados['Aging__L__p_25'] == 0, 1,
np.where(base_dados['Aging__L__p_25'] == 1, 1,
np.where(base_dados['Aging__L__p_25'] == 2, 1,
np.where(base_dados['Aging__L__p_25'] == 3, 2,
np.where(base_dados['Aging__L__p_25'] == 4, 2,
np.where(base_dados['Aging__L__p_25'] == 5, 3,
np.where(base_dados['Aging__L__p_25'] == 6, 2,
np.where(base_dados['Aging__L__p_25'] == 7, 3,
np.where(base_dados['Aging__L__p_25'] == 20, 0,
np.where(base_dados['Aging__L__p_25'] == 22, 2,
np.where(base_dados['Aging__L__p_25'] == 23, 2,
np.where(base_dados['Aging__L__p_25'] == 24, 3,
 0))))))))))))
         
base_dados['Aging__L__p_25_g_1_2'] = np.where(base_dados['Aging__L__p_25_g_1_1'] == 0, 1,
np.where(base_dados['Aging__L__p_25_g_1_1'] == 1, 3,
np.where(base_dados['Aging__L__p_25_g_1_1'] == 2, 2,
np.where(base_dados['Aging__L__p_25_g_1_1'] == 3, 0,
 0))))
         
base_dados['Aging__L__p_25_g_1'] = np.where(base_dados['Aging__L__p_25_g_1_2'] == 0, 0,
np.where(base_dados['Aging__L__p_25_g_1_2'] == 1, 1,
np.where(base_dados['Aging__L__p_25_g_1_2'] == 2, 2,
np.where(base_dados['Aging__L__p_25_g_1_2'] == 3, 3,
 0))))
         
         
         
         
              
base_dados['VlDividaAtualizado__R'] = np.sqrt(base_dados['VlDividaAtualizado'])
np.where(base_dados['VlDividaAtualizado__R'] == 0, -1, base_dados['VlDividaAtualizado__R'])
base_dados['VlDividaAtualizado__R'] = base_dados['VlDividaAtualizado__R'].fillna(-2)
base_dados['VlDividaAtualizado__R__pe_13'] = np.where(base_dados['VlDividaAtualizado__R'] <= 14.98399145755229, 0.0,
np.where(np.bitwise_and(base_dados['VlDividaAtualizado__R'] > 14.98399145755229, base_dados['VlDividaAtualizado__R'] <= 29.983495460002658), 1.0,
np.where(np.bitwise_and(base_dados['VlDividaAtualizado__R'] > 29.983495460002658, base_dados['VlDividaAtualizado__R'] <= 44.94318635788967), 2.0,
np.where(np.bitwise_and(base_dados['VlDividaAtualizado__R'] > 44.94318635788967, base_dados['VlDividaAtualizado__R'] <= 59.89424012373811), 3.0,
np.where(np.bitwise_and(base_dados['VlDividaAtualizado__R'] > 59.89424012373811, base_dados['VlDividaAtualizado__R'] <= 74.83929449159713), 4.0,
np.where(np.bitwise_and(base_dados['VlDividaAtualizado__R'] > 74.83929449159713, base_dados['VlDividaAtualizado__R'] <= 89.87580319529835), 5.0,
np.where(np.bitwise_and(base_dados['VlDividaAtualizado__R'] > 89.87580319529835, base_dados['VlDividaAtualizado__R'] <= 103.22165470481472), 6.0,
np.where(np.bitwise_and(base_dados['VlDividaAtualizado__R'] > 103.22165470481472, base_dados['VlDividaAtualizado__R'] <= 119.59197297477786), 7.0,
np.where(np.bitwise_and(base_dados['VlDividaAtualizado__R'] > 119.59197297477786, base_dados['VlDividaAtualizado__R'] <= 134.7174821617447), 8.0,
np.where(np.bitwise_and(base_dados['VlDividaAtualizado__R'] > 134.7174821617447, base_dados['VlDividaAtualizado__R'] <= 149.40448453778086), 9.0,
np.where(np.bitwise_and(base_dados['VlDividaAtualizado__R'] > 149.40448453778086, base_dados['VlDividaAtualizado__R'] <= 157.32158148200773), 10.0,
np.where(np.bitwise_and(base_dados['VlDividaAtualizado__R'] > 157.32158148200773, base_dados['VlDividaAtualizado__R'] <= 176.440669914847), 11.0,
np.where(np.bitwise_and(base_dados['VlDividaAtualizado__R'] > 176.440669914847, base_dados['VlDividaAtualizado__R'] <= 188.1279351930489), 12.0,
 0)))))))))))))
         
base_dados['VlDividaAtualizado__R__pe_13_g_1_1'] = np.where(base_dados['VlDividaAtualizado__R__pe_13'] == -2.0, 4,
np.where(base_dados['VlDividaAtualizado__R__pe_13'] == 0.0, 4,
np.where(base_dados['VlDividaAtualizado__R__pe_13'] == 1.0, 2,
np.where(base_dados['VlDividaAtualizado__R__pe_13'] == 2.0, 0,
np.where(base_dados['VlDividaAtualizado__R__pe_13'] == 3.0, 1,
np.where(base_dados['VlDividaAtualizado__R__pe_13'] == 4.0, 3,
np.where(base_dados['VlDividaAtualizado__R__pe_13'] == 5.0, 1,
np.where(base_dados['VlDividaAtualizado__R__pe_13'] == 6.0, 3,
np.where(base_dados['VlDividaAtualizado__R__pe_13'] == 7.0, 4,
np.where(base_dados['VlDividaAtualizado__R__pe_13'] == 8.0, 4,
np.where(base_dados['VlDividaAtualizado__R__pe_13'] == 9.0, 4,
np.where(base_dados['VlDividaAtualizado__R__pe_13'] == 10.0, 4,
np.where(base_dados['VlDividaAtualizado__R__pe_13'] == 11.0, 4,
np.where(base_dados['VlDividaAtualizado__R__pe_13'] == 12.0, 4,
 0))))))))))))))
         
base_dados['VlDividaAtualizado__R__pe_13_g_1_2'] = np.where(base_dados['VlDividaAtualizado__R__pe_13_g_1_1'] == 0, 3,
np.where(base_dados['VlDividaAtualizado__R__pe_13_g_1_1'] == 1, 1,
np.where(base_dados['VlDividaAtualizado__R__pe_13_g_1_1'] == 2, 3,
np.where(base_dados['VlDividaAtualizado__R__pe_13_g_1_1'] == 3, 0,
np.where(base_dados['VlDividaAtualizado__R__pe_13_g_1_1'] == 4, 2,
 0)))))
         
base_dados['VlDividaAtualizado__R__pe_13_g_1'] = np.where(base_dados['VlDividaAtualizado__R__pe_13_g_1_2'] == 0, 0,
np.where(base_dados['VlDividaAtualizado__R__pe_13_g_1_2'] == 1, 1,
np.where(base_dados['VlDividaAtualizado__R__pe_13_g_1_2'] == 2, 2,
np.where(base_dados['VlDividaAtualizado__R__pe_13_g_1_2'] == 3, 3,
 0))))
         
         
         
         
               
base_dados['VlDividaAtualizado__L'] = np.log(base_dados['VlDividaAtualizado'])
np.where(base_dados['VlDividaAtualizado__L'] == 0, -1, base_dados['VlDividaAtualizado__L'])
base_dados['VlDividaAtualizado__L'] = base_dados['VlDividaAtualizado__L'].fillna(-2)
base_dados['VlDividaAtualizado__L__p_6'] = np.where(base_dados['VlDividaAtualizado__L'] <= 6.195873700365942, 0.0,
np.where(np.bitwise_and(base_dados['VlDividaAtualizado__L'] > 6.195873700365942, base_dados['VlDividaAtualizado__L'] <= 6.936537080407589), 1.0,
np.where(np.bitwise_and(base_dados['VlDividaAtualizado__L'] > 6.936537080407589, base_dados['VlDividaAtualizado__L'] <= 7.485205371782873), 2.0,
np.where(np.bitwise_and(base_dados['VlDividaAtualizado__L'] > 7.485205371782873, base_dados['VlDividaAtualizado__L'] <= 8.07190871099415), 3.0,
np.where(np.bitwise_and(base_dados['VlDividaAtualizado__L'] > 8.07190871099415, base_dados['VlDividaAtualizado__L'] <= 8.838505378708055), 4.0,
np.where(base_dados['VlDividaAtualizado__L'] > 8.838505378708055, 5.0,
 0))))))
         
base_dados['VlDividaAtualizado__L__p_6_g_1_1'] = np.where(base_dados['VlDividaAtualizado__L__p_6'] == 0, 1,
np.where(base_dados['VlDividaAtualizado__L__p_6'] == 1, 1,
np.where(base_dados['VlDividaAtualizado__L__p_6'] == 2, 1,
np.where(base_dados['VlDividaAtualizado__L__p_6'] == 3, 0,
np.where(base_dados['VlDividaAtualizado__L__p_6'] == 4, 2,
np.where(base_dados['VlDividaAtualizado__L__p_6'] == 5, 0,
 0))))))
         
base_dados['VlDividaAtualizado__L__p_6_g_1_2'] = np.where(base_dados['VlDividaAtualizado__L__p_6_g_1_1'] == 0, 1,
np.where(base_dados['VlDividaAtualizado__L__p_6_g_1_1'] == 1, 2,
np.where(base_dados['VlDividaAtualizado__L__p_6_g_1_1'] == 2, 0,
 0)))
         
base_dados['VlDividaAtualizado__L__p_6_g_1'] = np.where(base_dados['VlDividaAtualizado__L__p_6_g_1_2'] == 0, 0,
np.where(base_dados['VlDividaAtualizado__L__p_6_g_1_2'] == 1, 1,
np.where(base_dados['VlDividaAtualizado__L__p_6_g_1_2'] == 2, 2,
 0)))


# COMMAND ----------

# MAGIC %md
# MAGIC ### Parte 2 de 2

# COMMAND ----------


base_dados['IdContatoSIR__L__p_8_g_1_c1_14_1'] = np.where(np.bitwise_and(base_dados['IdContatoSIR__pe_7_g_1'] == 0, base_dados['IdContatoSIR__L__p_8_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['IdContatoSIR__pe_7_g_1'] == 0, base_dados['IdContatoSIR__L__p_8_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['IdContatoSIR__pe_7_g_1'] == 0, base_dados['IdContatoSIR__L__p_8_g_1'] == 2), 1,
np.where(np.bitwise_and(base_dados['IdContatoSIR__pe_7_g_1'] == 0, base_dados['IdContatoSIR__L__p_8_g_1'] == 3), 3,
np.where(np.bitwise_and(base_dados['IdContatoSIR__pe_7_g_1'] == 0, base_dados['IdContatoSIR__L__p_8_g_1'] == 4), 4,
np.where(np.bitwise_and(base_dados['IdContatoSIR__pe_7_g_1'] == 1, base_dados['IdContatoSIR__L__p_8_g_1'] == 2), 2,
np.where(np.bitwise_and(base_dados['IdContatoSIR__pe_7_g_1'] == 1, base_dados['IdContatoSIR__L__p_8_g_1'] == 4), 4,
 0)))))))

base_dados['IdContatoSIR__L__p_8_g_1_c1_14_2'] = np.where(base_dados['IdContatoSIR__L__p_8_g_1_c1_14_1'] == 0, 0,
np.where(base_dados['IdContatoSIR__L__p_8_g_1_c1_14_1'] == 1, 1,
np.where(base_dados['IdContatoSIR__L__p_8_g_1_c1_14_1'] == 2, 2,
np.where(base_dados['IdContatoSIR__L__p_8_g_1_c1_14_1'] == 3, 3,
np.where(base_dados['IdContatoSIR__L__p_8_g_1_c1_14_1'] == 4, 4,
0)))))

base_dados['IdContatoSIR__L__p_8_g_1_c1_14'] = np.where(base_dados['IdContatoSIR__L__p_8_g_1_c1_14_2'] == 0, 0,
np.where(base_dados['IdContatoSIR__L__p_8_g_1_c1_14_2'] == 1, 1,
np.where(base_dados['IdContatoSIR__L__p_8_g_1_c1_14_2'] == 2, 2,
np.where(base_dados['IdContatoSIR__L__p_8_g_1_c1_14_2'] == 3, 3,
np.where(base_dados['IdContatoSIR__L__p_8_g_1_c1_14_2'] == 4, 4,
0)))))

         
         
            
base_dados['CPF__pe_4_g_1_c1_15_1'] = np.where(np.bitwise_and(base_dados['CPF__pe_5_g_1'] == 0, base_dados['CPF__pe_4_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['CPF__pe_5_g_1'] == 0, base_dados['CPF__pe_4_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['CPF__pe_5_g_1'] == 1, base_dados['CPF__pe_4_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['CPF__pe_5_g_1'] == 2, base_dados['CPF__pe_4_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['CPF__pe_5_g_1'] == 2, base_dados['CPF__pe_4_g_1'] == 1), 3,
0)))))

base_dados['CPF__pe_4_g_1_c1_15_2'] = np.where(base_dados['CPF__pe_4_g_1_c1_15_1'] == 0, 1,
np.where(base_dados['CPF__pe_4_g_1_c1_15_1'] == 1, 0,
np.where(base_dados['CPF__pe_4_g_1_c1_15_1'] == 2, 2,
np.where(base_dados['CPF__pe_4_g_1_c1_15_1'] == 3, 3,
0))))

base_dados['CPF__pe_4_g_1_c1_15'] = np.where(base_dados['CPF__pe_4_g_1_c1_15_2'] == 0, 0,
np.where(base_dados['CPF__pe_4_g_1_c1_15_2'] == 1, 1,
np.where(base_dados['CPF__pe_4_g_1_c1_15_2'] == 2, 2,
np.where(base_dados['CPF__pe_4_g_1_c1_15_2'] == 3, 3,
0))))

         
         
         
         
         
base_dados['Aging__L__p_25_g_1_c1_14_1'] = np.where(np.bitwise_and(base_dados['Aging__pe_10_g_1'] == 0, base_dados['Aging__L__p_25_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['Aging__pe_10_g_1'] == 0, base_dados['Aging__L__p_25_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['Aging__pe_10_g_1'] == 0, base_dados['Aging__L__p_25_g_1'] == 2), 2,
np.where(np.bitwise_and(base_dados['Aging__pe_10_g_1'] == 1, base_dados['Aging__L__p_25_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['Aging__pe_10_g_1'] == 1, base_dados['Aging__L__p_25_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['Aging__pe_10_g_1'] == 1, base_dados['Aging__L__p_25_g_1'] == 2), 3,
np.where(np.bitwise_and(base_dados['Aging__pe_10_g_1'] == 2, base_dados['Aging__L__p_25_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['Aging__pe_10_g_1'] == 2, base_dados['Aging__L__p_25_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['Aging__pe_10_g_1'] == 2, base_dados['Aging__L__p_25_g_1'] == 2), 4,
np.where(np.bitwise_and(base_dados['Aging__pe_10_g_1'] == 2, base_dados['Aging__L__p_25_g_1'] == 3), 5,
0))))))))))

base_dados['Aging__L__p_25_g_1_c1_14_2'] = np.where(base_dados['Aging__L__p_25_g_1_c1_14_1'] == 0, 0,
np.where(base_dados['Aging__L__p_25_g_1_c1_14_1'] == 1, 1,
np.where(base_dados['Aging__L__p_25_g_1_c1_14_1'] == 2, 2,
np.where(base_dados['Aging__L__p_25_g_1_c1_14_1'] == 3, 3,
np.where(base_dados['Aging__L__p_25_g_1_c1_14_1'] == 4, 4,
np.where(base_dados['Aging__L__p_25_g_1_c1_14_1'] == 5, 5,
0))))))

base_dados['Aging__L__p_25_g_1_c1_14'] = np.where(base_dados['Aging__L__p_25_g_1_c1_14_2'] == 0, 0,
np.where(base_dados['Aging__L__p_25_g_1_c1_14_2'] == 1, 1,
np.where(base_dados['Aging__L__p_25_g_1_c1_14_2'] == 2, 2,
np.where(base_dados['Aging__L__p_25_g_1_c1_14_2'] == 3, 3,
np.where(base_dados['Aging__L__p_25_g_1_c1_14_2'] == 4, 4,
np.where(base_dados['Aging__L__p_25_g_1_c1_14_2'] == 5, 5,
0))))))

         
         
         
         
         
base_dados['VlDividaAtualizado__L__p_6_g_1_c1_14_1'] = np.where(np.bitwise_and(base_dados['VlDividaAtualizado__R__pe_13_g_1'] == 0, base_dados['VlDividaAtualizado__L__p_6_g_1'] == 0), 0,
np.where(np.bitwise_and(base_dados['VlDividaAtualizado__R__pe_13_g_1'] == 0, base_dados['VlDividaAtualizado__L__p_6_g_1'] == 1), 1,
np.where(np.bitwise_and(base_dados['VlDividaAtualizado__R__pe_13_g_1'] == 1, base_dados['VlDividaAtualizado__L__p_6_g_1'] == 0), 1,
np.where(np.bitwise_and(base_dados['VlDividaAtualizado__R__pe_13_g_1'] == 1, base_dados['VlDividaAtualizado__L__p_6_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['VlDividaAtualizado__R__pe_13_g_1'] == 2, base_dados['VlDividaAtualizado__L__p_6_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['VlDividaAtualizado__R__pe_13_g_1'] == 2, base_dados['VlDividaAtualizado__L__p_6_g_1'] == 2), 3,
np.where(np.bitwise_and(base_dados['VlDividaAtualizado__R__pe_13_g_1'] == 3, base_dados['VlDividaAtualizado__L__p_6_g_1'] == 1), 2,
np.where(np.bitwise_and(base_dados['VlDividaAtualizado__R__pe_13_g_1'] == 3, base_dados['VlDividaAtualizado__L__p_6_g_1'] == 2), 3,
0))))))))

base_dados['VlDividaAtualizado__L__p_6_g_1_c1_14_2'] = np.where(base_dados['VlDividaAtualizado__L__p_6_g_1_c1_14_1'] == 0, 1,
np.where(base_dados['VlDividaAtualizado__L__p_6_g_1_c1_14_1'] == 1, 0,
np.where(base_dados['VlDividaAtualizado__L__p_6_g_1_c1_14_1'] == 2, 2,
np.where(base_dados['VlDividaAtualizado__L__p_6_g_1_c1_14_1'] == 3, 3,
0))))

base_dados['VlDividaAtualizado__L__p_6_g_1_c1_14'] = np.where(base_dados['VlDividaAtualizado__L__p_6_g_1_c1_14_2'] == 0, 0,
np.where(base_dados['VlDividaAtualizado__L__p_6_g_1_c1_14_2'] == 1, 1,
np.where(base_dados['VlDividaAtualizado__L__p_6_g_1_c1_14_2'] == 2, 2,
np.where(base_dados['VlDividaAtualizado__L__p_6_g_1_c1_14_2'] == 3, 3,
0))))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Mantendo apenas as variáveis do modelo

# COMMAND ----------

varvar=[]
varvar= [chave,var_tmp,target,'CPF__pe_4_g_1_c1_15', 'VlDividaAtualizado__L__p_6_g_1_c1_14', 'Class_Carteira_gh38', 'Aging__L__p_25_g_1_c1_14', 'IdContatoSIR__L__p_8_g_1_c1_14']
base_teste_c0 = base_dados[varvar]
base_teste_c0


# COMMAND ----------

# MAGIC %md
# MAGIC ## Gerando Amostra de treinamento e teste

# COMMAND ----------

base_treino_c0 = pd.read_csv(caminho_base + 'base_treino_final.csv', sep=",", decimal=".")

var_fin_c0=list(base_teste_c0.columns)
var_fin_c0.remove(target)
var_fin_c0.remove(var_tmp)
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
w_teste = base_teste_c0[var_tmp]

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
w_teste1 = w_teste.reset_index(drop=True)
x_teste1 = x_teste.reset_index(drop=True)
data_prob1 = data_prob.reset_index(drop=True)


x_teste2 = pd.concat([z_teste1,w_teste1,y_teste1,x_teste1, data_prob1], axis=1)

x_teste2


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

x_teste2['P_1_R'] = np.sqrt(x_teste2['P_1'])
np.where(x_teste2['P_1_R'] == 0, -1, x_teste2['P_1_R'])
x_teste2['P_1_R'] = x_teste2['P_1_R'].fillna(-2)

x_teste2['P_1_pe_20_g_1'] = np.where(x_teste2['P_1'] <= 0.039482547, 0.0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.039482547, x_teste2['P_1'] <= 0.078855436), 1.0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.078855436, x_teste2['P_1'] <= 0.23682533), 2.0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.23682533, x_teste2['P_1'] <= 0.354488676), 3.0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.354488676, x_teste2['P_1'] <= 0.434850909), 4.0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.434850909, x_teste2['P_1'] <= 0.517960421), 5.0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.517960421, x_teste2['P_1'] <= 0.591608), 6.0,
    np.where(np.bitwise_and(x_teste2['P_1'] > 0.591608, x_teste2['P_1'] <= 0.792068002), 7.0,
    np.where(x_teste2['P_1'] > 0.792068002, 8.0,0)))))))))

x_teste2['P_1_R_p_5_g_1'] = np.where(x_teste2['P_1_R'] <= 0.210253883, 0.0,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.210253883, x_teste2['P_1_R'] <= 0.282139416), 1.0,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.282139416, x_teste2['P_1_R'] <= 0.362194322), 2.0,
    np.where(np.bitwise_and(x_teste2['P_1_R'] > 0.362194322, x_teste2['P_1_R'] <= 0.485356497), 3.0,
    np.where(x_teste2['P_1_R'] > 0.485356497, 4.0,0)))))

x_teste2['GH'] = np.where(np.bitwise_and(x_teste2['P_1_pe_20_g_1'] == 0, x_teste2['P_1_R_p_5_g_1'] == 0), 1,
    np.where(np.bitwise_and(x_teste2['P_1_pe_20_g_1'] == 1, x_teste2['P_1_R_p_5_g_1'] == 0), 0,
    np.where(np.bitwise_and(x_teste2['P_1_pe_20_g_1'] == 1, x_teste2['P_1_R_p_5_g_1'] == 1), 2,
    np.where(np.bitwise_and(x_teste2['P_1_pe_20_g_1'] == 2, x_teste2['P_1_R_p_5_g_1'] == 1), 3,
    np.where(np.bitwise_and(x_teste2['P_1_pe_20_g_1'] == 2, x_teste2['P_1_R_p_5_g_1'] == 2), 3,
    np.where(np.bitwise_and(x_teste2['P_1_pe_20_g_1'] == 2, x_teste2['P_1_R_p_5_g_1'] == 3), 3,
    np.where(np.bitwise_and(x_teste2['P_1_pe_20_g_1'] == 2, x_teste2['P_1_R_p_5_g_1'] == 4), 5,
    np.where(np.bitwise_and(x_teste2['P_1_pe_20_g_1'] == 3, x_teste2['P_1_R_p_5_g_1'] == 2), 5,
    np.where(np.bitwise_and(x_teste2['P_1_pe_20_g_1'] == 3, x_teste2['P_1_R_p_5_g_1'] == 3), 4,
    np.where(np.bitwise_and(x_teste2['P_1_pe_20_g_1'] == 3, x_teste2['P_1_R_p_5_g_1'] == 4), 4,
    np.where(np.bitwise_and(x_teste2['P_1_pe_20_g_1'] == 4, x_teste2['P_1_R_p_5_g_1'] == 3), 5,
    np.where(np.bitwise_and(x_teste2['P_1_pe_20_g_1'] == 4, x_teste2['P_1_R_p_5_g_1'] == 4), 5,
    np.where(np.bitwise_and(x_teste2['P_1_pe_20_g_1'] == 5, x_teste2['P_1_R_p_5_g_1'] == 4), 6,
    np.where(np.bitwise_and(x_teste2['P_1_pe_20_g_1'] == 6, x_teste2['P_1_R_p_5_g_1'] == 4), 6,
    np.where(np.bitwise_and(x_teste2['P_1_pe_20_g_1'] == 7, x_teste2['P_1_R_p_5_g_1'] == 4), 7,
    np.where(np.bitwise_and(x_teste2['P_1_pe_20_g_1'] == 8, x_teste2['P_1_R_p_5_g_1'] == 4), 8,
    np.where(np.bitwise_and(x_teste2['P_1_pe_20_g_1'] == 9, x_teste2['P_1_R_p_5_g_1'] == 4), 8,
             0)))))))))))))))))             

del x_teste2['P_1_R']
del x_teste2['P_1_pe_20_g_1']
del x_teste2['P_1_R_p_5_g_1']

x_teste2


# COMMAND ----------

# DBTITLE 1,Salvando o modelo
import pickle

##Save###
dir_save="/dbfs/mnt/ml-prd/ml-data/propensaodeal/recovery/pickle_models/models_V2 (Elisvaldo)"
pickle.dump(modelo,open(dir_save+'/model_fit_V2_recovery.sav','wb'))