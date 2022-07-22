# Databricks notebook source
import pickle
import os
import pandas as pd
import numpy as np
import time

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pyspark.sql.types import DateType

# COMMAND ----------

# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

blob_account_source_prd = "qqprd"
blob_account_source_ml = "qqdatastoragemain"
blob_container_source_prd = "qq-integrator"
blob_container_source_ml = "ml-prd"

prefix = "etl/bahamas/processed"

mount_blob_storage_key(dbutils,blob_account_source_prd,blob_account_source_prd,'/mnt/qq-integrator')
mount_blob_storage_key(dbutils,blob_account_source_ml,blob_account_source_ml,'/mnt/ml-prd')

caminho_trusted = '/mnt/ml-prd/ml-data/propensaodeal/bahamas/trusted'
caminho_trusted_dbfs = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/bahamas/trusted'

pickle_path = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/bahamas/pickle_models/'

outputpath = 'mnt/ml-prd/ml-data/propensaodeal/bahamas/output/'
outputpath_dbfs = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/bahamas/output/'

# COMMAND ----------

#Caminho da base de dados
list_base = os.listdir(caminho_trusted_dbfs)

#Nome da Base de Dados
N_Base = max(list_base)
dt_max = N_Base.split('_')[1]
dt_max = dt_max[0:4]+'-'+dt_max[4:6]+'-'+dt_max[6:8]
nm_base = "trustedFile_bahamas"

N_Base

# COMMAND ----------

# Puxar o arquivo mais recente:
file = caminho_trusted + '/' + N_Base

# Caminho para puxar o modelo
filename = pickle_path + 'finalized_model.sav'

# COMMAND ----------

df_sp = spark.read.option('delimiter',';').option('header', 'True').csv(file)

# Modificações feitas na base:

df_sp = df_sp.drop('contrato', 'bairro', 'cep', 'cidade', 'cliente', 'complemento', 'ddd', 'email', 'logradouro', 'nome', 'parcela','ramal','produto', 'seguradora', 'seguro', 'tipoEndereco', 'tipoLogradouro','tipoPessoa', 'numeroNossoNumero','codigoDne', 'numero', 'numeroParcelas','numeroParcela','numeroSequencial', 'taxaOperacao', 'telefone', 'valorDesconto', 'valorDespesa','valorMulta','valorMora','valorOutros', 'rg', 'codigo','numeroContrato','saldoPrincipal','saldoAtual','valorOperacao','valorPermanencia','valorPrincipal','valorTotal', 'dataEmissao', 'dataOperacao', 'dataPrevisao', 'dataVencimento') # Após checar quais variáveis são inúteis para modelagem, quais são unárias e quais possuem alta correlação entre si.

df_sp = df_sp.filter(F.col('situacaoContrato') != 'LIQUIDADO') #essas linhas foram removidas, pois chegaram depois que o modelo estava construido e gerou erro na execução do modelo

df_sp = df_sp.withColumn('diasAtraso', F.col('diasAtraso').cast('int'))\
     .withColumn('saldoTotal', F.col('saldoTotal').cast('float'))\
     .withColumn('dataNascimento', F.col('dataNascimento').cast(DateType()))
     
df = df_sp.toPandas()
df['idade'] = ((pd.to_datetime("now") - pd.to_datetime(df['dataNascimento'], format='%Y/%m/%d', errors = 'coerce')).dt.days)/365.25
df = df.drop(columns = ['dataNascimento'])

# COMMAND ----------

variaveis_numericas = ['diasAtraso','saldoTotal','idade']
variaveis_categoricas = ['lp','situacaoContrato', 'tipoTelefone', 'uf']

nonum_feats = df[variaveis_categoricas].astype('category')
ohc_feats = pd.get_dummies(nonum_feats,drop_first=True)

ohc_data = pd.concat([df[variaveis_numericas], ohc_feats],axis=1)

x = ohc_data
x = x.fillna(0)

scaler = StandardScaler()
scaled_numfeats_train = pd.DataFrame(scaler.fit_transform(x[variaveis_numericas]), 
                                     columns=variaveis_numericas, index= x.index)

for col in variaveis_numericas:
    x[col] = scaled_numfeats_train[col]

# COMMAND ----------

modelo=pickle.load(open(filename, 'rb'))
chave = df['cic'].astype(str)
p_1 = modelo.predict_proba(x)[:,1]

dt_gh = pd.DataFrame({'Chave': chave, 'P_1':p_1})

# COMMAND ----------

dt_gh['P_1_L'] = np.log(dt_gh['P_1'])
dt_gh['P_1_L'] = np.where(dt_gh['P_1'] == 0, -1, dt_gh['P_1_L'])
dt_gh['P_1_L'] = np.where(dt_gh['P_1'] == np.nan, -2, dt_gh['P_1_L'])
dt_gh['P_1_L'] = dt_gh['P_1_L'].fillna(-2)

dt_gh['P_1_pu_17_g_1'] = np.where(dt_gh['P_1'] <= 0.0584552, 4,
    np.where(np.bitwise_and(dt_gh['P_1'] > 0.0584552, dt_gh['P_1'] <= 0.116910737), 0,
    np.where(np.bitwise_and(dt_gh['P_1'] > 0.116910737, dt_gh['P_1'] <= 0.175104526), 1,
    np.where(np.bitwise_and(dt_gh['P_1'] > 0.175104526, dt_gh['P_1'] <= 0.233312864), 2,
    np.where(np.bitwise_and(dt_gh['P_1'] > 0.233312864, dt_gh['P_1'] <= 0.40789645), 3,
    np.where(dt_gh['P_1'] > 0.40789645,5,0))))))

dt_gh['P_1_L_p_8_g_1'] = np.where(dt_gh['P_1_L'] <= -2.603911656, 3,
    np.where(np.bitwise_and(dt_gh['P_1_L'] > -2.603911656, dt_gh['P_1_L'] <= -1.698567285), 0,
    np.where(np.bitwise_and(dt_gh['P_1_L'] > -1.698567285, dt_gh['P_1_L'] <= -1.416630199), 1,
    np.where(np.bitwise_and(dt_gh['P_1_L'] > -1.416630199, dt_gh['P_1_L'] <= -1.095076116), 2,
    np.where(np.bitwise_and(dt_gh['P_1_L'] > -1.095076116, dt_gh['P_1_L'] <= -0.584104578), 4,
    np.where(dt_gh['P_1_L'] > -0.584104578,5,0))))))

dt_gh['GH'] = np.where(np.bitwise_and(dt_gh['P_1_pu_17_g_1'] == 0, dt_gh['P_1_L_p_8_g_1'] == 0), 0,
    np.where(np.bitwise_and(dt_gh['P_1_pu_17_g_1'] == 0, dt_gh['P_1_L_p_8_g_1'] == 1), 0,
    np.where(np.bitwise_and(dt_gh['P_1_pu_17_g_1'] == 0, dt_gh['P_1_L_p_8_g_1'] == 2), 0,
    np.where(np.bitwise_and(dt_gh['P_1_pu_17_g_1'] == 0, dt_gh['P_1_L_p_8_g_1'] == 3), 0,
    np.where(np.bitwise_and(dt_gh['P_1_pu_17_g_1'] == 0, dt_gh['P_1_L_p_8_g_1'] == 4), 0,
    np.where(np.bitwise_and(dt_gh['P_1_pu_17_g_1'] == 0, dt_gh['P_1_L_p_8_g_1'] == 5), 1,
    np.where(np.bitwise_and(dt_gh['P_1_pu_17_g_1'] == 1, dt_gh['P_1_L_p_8_g_1'] == 0), 1,
    np.where(np.bitwise_and(dt_gh['P_1_pu_17_g_1'] == 1, dt_gh['P_1_L_p_8_g_1'] == 1), 1,
    np.where(np.bitwise_and(dt_gh['P_1_pu_17_g_1'] == 1, dt_gh['P_1_L_p_8_g_1'] == 2), 1,
    np.where(np.bitwise_and(dt_gh['P_1_pu_17_g_1'] == 1, dt_gh['P_1_L_p_8_g_1'] == 3), 2,
    np.where(np.bitwise_and(dt_gh['P_1_pu_17_g_1'] == 1, dt_gh['P_1_L_p_8_g_1'] == 4), 2,
    np.where(np.bitwise_and(dt_gh['P_1_pu_17_g_1'] == 1, dt_gh['P_1_L_p_8_g_1'] == 5), 2,
    np.where(np.bitwise_and(dt_gh['P_1_pu_17_g_1'] == 2, dt_gh['P_1_L_p_8_g_1'] == 0), 2,
    np.where(np.bitwise_and(dt_gh['P_1_pu_17_g_1'] == 2, dt_gh['P_1_L_p_8_g_1'] == 1), 2,
    np.where(np.bitwise_and(dt_gh['P_1_pu_17_g_1'] == 2, dt_gh['P_1_L_p_8_g_1'] == 2), 2,
    np.where(np.bitwise_and(dt_gh['P_1_pu_17_g_1'] == 2, dt_gh['P_1_L_p_8_g_1'] == 3), 2,
    np.where(np.bitwise_and(dt_gh['P_1_pu_17_g_1'] == 2, dt_gh['P_1_L_p_8_g_1'] == 4), 3,
    np.where(np.bitwise_and(dt_gh['P_1_pu_17_g_1'] == 2, dt_gh['P_1_L_p_8_g_1'] == 5), 3,
    np.where(np.bitwise_and(dt_gh['P_1_pu_17_g_1'] == 3, dt_gh['P_1_L_p_8_g_1'] == 0), 2,
    np.where(np.bitwise_and(dt_gh['P_1_pu_17_g_1'] == 3, dt_gh['P_1_L_p_8_g_1'] == 1), 3,
    np.where(np.bitwise_and(dt_gh['P_1_pu_17_g_1'] == 3, dt_gh['P_1_L_p_8_g_1'] == 2), 3,
    np.where(np.bitwise_and(dt_gh['P_1_pu_17_g_1'] == 3, dt_gh['P_1_L_p_8_g_1'] == 3), 3,
    np.where(np.bitwise_and(dt_gh['P_1_pu_17_g_1'] == 3, dt_gh['P_1_L_p_8_g_1'] == 4), 3,
    np.where(np.bitwise_and(dt_gh['P_1_pu_17_g_1'] == 3, dt_gh['P_1_L_p_8_g_1'] == 5), 4,
    np.where(np.bitwise_and(dt_gh['P_1_pu_17_g_1'] == 4, dt_gh['P_1_L_p_8_g_1'] == 0), 3,
    np.where(np.bitwise_and(dt_gh['P_1_pu_17_g_1'] == 4, dt_gh['P_1_L_p_8_g_1'] == 1), 4,
    np.where(np.bitwise_and(dt_gh['P_1_pu_17_g_1'] == 4, dt_gh['P_1_L_p_8_g_1'] == 2), 4,
    np.where(np.bitwise_and(dt_gh['P_1_pu_17_g_1'] == 4, dt_gh['P_1_L_p_8_g_1'] == 3), 4,
    np.where(np.bitwise_and(dt_gh['P_1_pu_17_g_1'] == 4, dt_gh['P_1_L_p_8_g_1'] == 4), 4,
    np.where(np.bitwise_and(dt_gh['P_1_pu_17_g_1'] == 4, dt_gh['P_1_L_p_8_g_1'] == 5), 5,
    np.where(np.bitwise_and(dt_gh['P_1_pu_17_g_1'] == 5, dt_gh['P_1_L_p_8_g_1'] == 0), 5,
    np.where(np.bitwise_and(dt_gh['P_1_pu_17_g_1'] == 5, dt_gh['P_1_L_p_8_g_1'] == 1), 5,
    np.where(np.bitwise_and(dt_gh['P_1_pu_17_g_1'] == 5, dt_gh['P_1_L_p_8_g_1'] == 2), 5,
    np.where(np.bitwise_and(dt_gh['P_1_pu_17_g_1'] == 5, dt_gh['P_1_L_p_8_g_1'] == 3), 5,
    np.where(np.bitwise_and(dt_gh['P_1_pu_17_g_1'] == 5, dt_gh['P_1_L_p_8_g_1'] == 4), 5,
    np.where(np.bitwise_and(dt_gh['P_1_pu_17_g_1'] == 5, dt_gh['P_1_L_p_8_g_1'] == 5), 6,
             0))))))))))))))))))))))))))))))))))))

# COMMAND ----------

dt_gh

# COMMAND ----------

try:
  dbutils.fs.rm(outputpath, True)
except:
  pass
dbutils.fs.mkdirs(outputpath)

dt_gh.to_csv(open(os.path.join(outputpath_dbfs, 'pre_output:' + nm_base.replace('-','') + '_' + dt_max + '.csv'),'wb'))
os.path.join(outputpath_dbfs, 'pre_output:' + nm_base.replace('-','') + '_' + dt_max + '.csv')