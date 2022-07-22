# Databricks notebook source
# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

# DBTITLE 1,cria widget
trustedpath = '/dbfs/mnt/ml-prd/ml-data/propensaodeal/credz/trusted'

try:
  dbutils.widgets.remove('ARQUIVO_ESCOLHIDO')
except:
  pass
dbutils.widgets.combobox('ARQUIVO_ESCOLHIDO', max(os.listdir(trustedpath)), os.listdir(trustedpath))

# COMMAND ----------

data_arquivo = dbutils.widgets.get('ARQUIVO_ESCOLHIDO')
for file in os.listdir(os.path.join(trustedpath, data_arquivo)):
  print (file)
file = os.path.join(trustedpath, data_arquivo, file)
file

# COMMAND ----------

import numpy as np
import pandas as pd
import datetime
from sklearn.pipeline import Pipeline
from sklearn import set_config # visualização do pipe
set_config(display='diagram') # configuração da visualização do pipe
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC

df = pd.read_parquet(file)

# COMMAND ----------

#dropando colunas vistas no passo anterior
colunas_a_excluir = ['ID_PESSOA:ID_DIVIDA',
                     'ID_CONTRATO',
                     'ID_PRODUTO',
                     'CIDADE',
                     'DOCUMENTO_PESSOA',
                     'ID_DIVIDA',
                     'DATA_CORREÇAO',
                     'DETALHES_CLIENTES_MIGRAR_CEDENTE_LOTE_3',
                     'DETALHES_CLIENTES_TIPO_SEGURO',
                     'DETALHES_CLIENTES_RETIRADA_FIS',
                     'DETALHES_CLIENTES_MIGRAR_CEDENTE_LOTE_2',
                     'DETALHES_CLIENTES_CLIENTES_RETIRADOS_BVS_28092017',
                     'DETALHES_CLIENTES_REVERSAO_ASSESSORIA',
                     'DETALHES_CLIENTES_MARCACLIENTE',
                     'DETALHES_CLIENTES_ACAO_RECOVERY_3',
                     'DETALHES_CLIENTES_ACAO_RECOVERY',
                     'DETALHES_CLIENTES_ACAO_RECOVERY_4',
                     'DETALHES_CLIENTES_EXPIRACAO',
                     'DETALHES_CLIENTES_CAMPANHA',
                     'DETALHES_CLIENTES_TESTE_DISCADOR',
                     'DETALHES_CONTRATOS_HISTORICO_FPD___DESC',
                     'VALOR_CORRECAO',
                     'VALOR_MINIMO',
                     'PLANO',
                     'UF',
                     'DETALHES_CLIENTES_PCT',
                     'DETALHES_CLIENTES_CARGA',
                     'DETALHES_CLIENTES_MIGRAR_CEDENTE',
                     'DETALHES_CLIENTES_CODIGO_ASSESSORIA',
                     'DETALHES_CONTRATOS_COBRADOR',
                     'DETALHES_CONTRATOS_PRODUTO',
                     'DETALHES_CONTRATOS_SALDO_PARCELADO_LOJISTA',
                     'DETALHES_CONTRATOS_BLOQUEIO1',
                     'DETALHES_CONTRATOS_STATUS_CONTA',
                     'DETALHES_DIVIDAS_VALOR_PARCELADO_DIARIO',
                     'DETALHES_DIVIDAS_TAXA_ISF',
                     'DETALHES_DIVIDAS_TAXA_COBRANCA',
                     'DETALHES_DIVIDAS_TAXA_RECUPERACAO',
                     'DDDs', 
                     'DOMINIO_EMAIL', 
                     'DETALHES_CLIENTES_CARTAO', 
                     'DETALHES_CLIENTES_ENRIQUECIMENTO', 
                     'DETALHES_CLIENTES_ENRIQUECIMENTO_SGC',
                     'ID_DIVIDA_INTERNO_DIVIDA_DETALHE',
                     'DETALHES_CONTRATOS_NOME_LOGO',
                     'id_deals',
                     'DETALHES_CONTRATOS_DATA_ATUALIZACAO',
                     'DETALHES_DIVIDAS_VALOR_SALDO_DIARIO']

# COMMAND ----------

df = df.drop(columns=colunas_a_excluir)
df = df.set_index('DOCUMENTO:ID_DIVIDA')

# COMMAND ----------

# MAGIC %md
# MAGIC IDADE_PESSOA - ok,
# MAGIC 
# MAGIC ID_CEDENTE - ok,
# MAGIC 
# MAGIC VALOR_DIVIDA - ok,
# MAGIC 
# MAGIC AGING - ok,
# MAGIC 
# MAGIC NOME_PRODUTO - info importante apesar de gerar 99 features em hot-encoder,
# MAGIC 
# MAGIC DETALHES_CLIENTES_VENCIMENTO_FATURA - verificar campos positivos quando ocorre subtracao da data do arquivo pela data do campo, ja que pode ocorrer caso de divida futura,
# MAGIC 
# MAGIC DETALHES_CLIENTES_SCORE_CARGA - ok,
# MAGIC 
# MAGIC DETALHES_CLIENTES_VALOR_FATURA - ok,
# MAGIC 
# MAGIC DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO - ok,
# MAGIC 
# MAGIC DETALHES_CLIENTES_RENDA_CONSIDERADA - ok,
# MAGIC 
# MAGIC DETALHES_CLIENTES_COLLECT_SCORE - ok,
# MAGIC 
# MAGIC DETALHES_CLIENTES_LIMITE_APROVADO - ok,
# MAGIC 
# MAGIC DETALHES_CONTRATOS_SALDO_ATUAL_ALDIV - ok,
# MAGIC 
# MAGIC DETALHES_CONTRATOS_BLOQUEIO2 - ok,
# MAGIC 
# MAGIC DETALHES_CONTRATOS_CLASSE - verificar variabilidade,
# MAGIC 
# MAGIC DETALHES_CONTRATOS_CODIGO_LOGO - verificar variabilidade,
# MAGIC 
# MAGIC DETALHES_CONTRATOS_HISTORICO_FPD - verificar variabilidade,
# MAGIC 
# MAGIC DETALHES_CONTRATOS_BLOQUEIO2___DESC - ok,
# MAGIC 
# MAGIC DETALHES_CONTRATOS_BLOQUEIO1___DESC - ok,
# MAGIC 
# MAGIC DETALHES_CONTRATOS_CLASSE___DESC - ok,
# MAGIC 
# MAGIC DETALHES_CONTRATOS_STATUS_ACORDO - ok.
# MAGIC 
# MAGIC DETALHES_CONTRATOS_NOME_LOGO - DROP,
# MAGIC 
# MAGIC DETALHES_CONTRATOS_VALOR_PRINCIPAL - ok
# MAGIC 
# MAGIC DETALHES_DIVIDAS_VALOR_JUROS - ok,
# MAGIC 
# MAGIC DETALHES_DIVIDAS_TAXA_SERVICO - ok,
# MAGIC 
# MAGIC DETALHES_DIVIDAS_TAXA_ATRASO - ok
# MAGIC 
# MAGIC DETALHES_DIVIDAS_VALOR_JUROS_DIARIO - verificar variabilidade
# MAGIC 
# MAGIC DETALHES_DIVIDAS_TAXA_SEGURO - ok
# MAGIC 
# MAGIC TIPO_EMAIL_0 - ok
# MAGIC 
# MAGIC TIPO_EMAIL_1 - ok
# MAGIC 
# MAGIC TIPO_EMAIL_2 - ok
# MAGIC 
# MAGIC TIPO_EMAIL_3 - ok
# MAGIC 
# MAGIC TIPO_TELEFONE_0 - ok
# MAGIC 
# MAGIC TIPO_TELEFONE_1 - ok
# MAGIC 
# MAGIC TIPO_TELEFONE_2 - ok
# MAGIC 
# MAGIC TIPO_TELEFONE_3 - ok
# MAGIC 
# MAGIC TIPO_TELEFONE_4 - ok
# MAGIC 
# MAGIC TIPO_ENDERECO_0 - ok
# MAGIC 
# MAGIC TIPO_ENDERECO_1 - ok
# MAGIC 
# MAGIC TIPO_ENDERECO_2 - ok
# MAGIC 
# MAGIC TIPO_ENDERECO_3 - ok

# COMMAND ----------

#### TRANSFORM VENCIMENTO FATURA
df['DETALHES_CLIENTES_VENCIMENTO_FATURA'] = df['DETALHES_CLIENTES_VENCIMENTO_FATURA'].astype('str').str.zfill(8)
df['DETALHES_CLIENTES_VENCIMENTO_FATURA'] = pd.to_datetime(df['DETALHES_CLIENTES_VENCIMENTO_FATURA'], format='%d%m%Y')

# COMMAND ----------

# transformação de vencimento de fatura
df['DETALHES_CLIENTES_VENCIMENTO_FATURA'] = (df['DETALHES_CLIENTES_VENCIMENTO_FATURA']-pd.to_datetime(data_arquivo)).dt.days

# COMMAND ----------

dividir_por_100 = [
    'VALOR_DIVIDA',
    'DETALHES_CLIENTES_VALOR_FATURA',
    'DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO',
    'DETALHES_DIVIDAS_VALOR_JUROS',
    'DETALHES_DIVIDAS_TAXA_SERVICO',
    'DETALHES_DIVIDAS_TAXA_ATRASO',
    'DETALHES_DIVIDAS_TAXA_SEGURO']

for col in dividir_por_100:
    df[col] = round(pd.to_numeric(df[col])/100,2)

# COMMAND ----------

numeric_features = ['IDADE_PESSOA',
 'VALOR_DIVIDA',
 'AGING',
 'IDADE_PESSOA',
 'VALOR_DIVIDA',
 'AGING',
 'DETALHES_CLIENTES_VENCIMENTO_FATURA',
 'DETALHES_CLIENTES_SCORE_CARGA',
 'DETALHES_CLIENTES_VALOR_FATURA',
 'DETALHES_CLIENTES_VALOR_PAGAMENTO_MINIMO',
 'DETALHES_CLIENTES_RENDA_CONSIDERADA',
 'DETALHES_CLIENTES_COLLECT_SCORE',
 'DETALHES_CLIENTES_LIMITE_APROVADO',
 'DETALHES_CONTRATOS_SALDO_ATUAL_ALDIV',
 'DETALHES_CONTRATOS_VALOR_PRINCIPAL',
 'DETALHES_DIVIDAS_VALOR_JUROS',
 'DETALHES_DIVIDAS_TAXA_SERVICO',
 'DETALHES_DIVIDAS_TAXA_ATRASO',
 'DETALHES_DIVIDAS_VALOR_JUROS_DIARIO',
 'DETALHES_DIVIDAS_TAXA_SEGURO']

categorical_features = ['ID_CEDENTE',
 'ID_CEDENTE',
 'NOME_PRODUTO',
 'DETALHES_CONTRATOS_BLOQUEIO2',
 'DETALHES_CONTRATOS_CLASSE',
 'DETALHES_CONTRATOS_CODIGO_LOGO',
 'DETALHES_CONTRATOS_HISTORICO_FPD',
 'DETALHES_CONTRATOS_BLOQUEIO2___DESC',
 'DETALHES_CONTRATOS_BLOQUEIO1___DESC',
 'DETALHES_CONTRATOS_CLASSE___DESC',
 'DETALHES_CONTRATOS_STATUS_ACORDO',
 'TIPO_EMAIL_0',
 'TIPO_EMAIL_1',
 'TIPO_EMAIL_2',
 'TIPO_EMAIL_3',
 'TIPO_TELEFONE_0',
 'TIPO_TELEFONE_1',
 'TIPO_TELEFONE_2',
 'TIPO_TELEFONE_3',
 'TIPO_TELEFONE_4',
 'TIPO_ENDERECO_0',
 'TIPO_ENDERECO_1',
 'TIPO_ENDERECO_2',
 'TIPO_ENDERECO_3']

# COMMAND ----------

#pre-processamento
imputer_preprocess = SimpleImputer(strategy='mean')
standard_scaler_preprocess = StandardScaler()
oneHot_preprocess = OneHotEncoder(sparse=False, handle_unknown='ignore')

pipe_numerical = Pipeline([('SimpleImputer', imputer_preprocess), ('StandardScaler', standard_scaler_preprocess)])
pipe_categorical = Pipeline([('OneHot',oneHot_preprocess)])

preprocess = ColumnTransformer([('PipeNumeric', pipe_numerical, numeric_features), 
                                ('PipeCategorical', pipe_categorical, categorical_features)])
preprocess

# COMMAND ----------

# realizando pre-transformação e smote
RANDOM_STATE = 42
# train_test_split
x = df.drop(columns = 'VARIAVEL_RESPOSTA')
y = df.loc[:,'VARIAVEL_RESPOSTA', ].to_numpy()
#SMOTE
%pip install imblearn
import imblearn # imbalanced learning library
from imblearn.over_sampling import SMOTE

x_pretransform = preprocess.fit_transform(x)

x_pretransform_train, x_pretransform_test, y_pretransform_train, y_pretransform_test = train_test_split(x_pretransform,y, test_size=0.2, random_state = RANDOM_STATE)


oversample = SMOTE()
x_oversample, y_oversample = oversample.fit_resample(x_pretransform_train, y_pretransform_train)

# COMMAND ----------

# modelos

#RandomForestClassifier
randomForest = RandomForestClassifier(random_state = RANDOM_STATE)
pipe_randomForest = Pipeline([('preprocess',preprocess),('randomForest', randomForest)])
#HistGradientBoostingClassifier
histGradient = HistGradientBoostingClassifier(random_state = RANDOM_STATE)
pipe_histGradient = Pipeline([('preprocess', preprocess), ('histGradient', histGradient)])
#BaggingClassifier
bagging = BaggingClassifier(random_state = RANDOM_STATE)
pipe_bagging = Pipeline([('preprocess', preprocess), ('bagging', bagging)])
#ExtraTreesClassifier
extraTrees = ExtraTreesClassifier(random_state = RANDOM_STATE)
pipe_extraTrees = Pipeline([('preprocess', preprocess), ('extraTrees', extraTrees)])

#regressao linear
sgdClassifier = SGDClassifier(random_state = RANDOM_STATE)
pipe_linear_sgd = Pipeline([('preprocess', preprocess), ('linear_regression',sgdClassifier)])

#svr linear
linear_svc = LinearSVC(random_state=RANDOM_STATE)
pipe_linear_svc = Pipeline([('preprocess', preprocess), ('linear_svr',linear_svc)])

#AdaBoostClassifier
adaboost_classifier = AdaBoostClassifier(random_state = RANDOM_STATE)
pipe_adaboost_classifier = Pipeline([('preprocess', preprocess), ('adaBoots', adaboost_classifier)])

modelos = [randomForest,histGradient, bagging, extraTrees, sgdClassifier, linear_svc,adaboost_classifier]

#ver svm
#ver xgboost

# COMMAND ----------

# scoragem em cross_val_score
resultados_cross = {}
resultados_rmse = {}
for model in modelos:
    scores = cross_val_score(model, x_oversample, y_oversample, scoring = 'roc_auc',cv=10)
    model.fit(x_oversample, y_oversample)
    y_pred = model.predict(x_pretransform_test)
    ras = roc_auc_score(y_pretransform_test, y_pred)
    resultados_rmse.update({model:ras})
    resultados_cross.update({model:scores})

result_array = []
for modelo in resultados_cross:
    print (modelo, 'CROSS_VAL_SCORE',round(resultados_cross[modelo].mean(),3), round(resultados_cross[modelo].std(),3))
    print (modelo, "ROC_AUC_SCORE:",resultados_rmse[modelo])
    result_array.append(resultados_cross[modelo])
sns.boxplot(data=result_array)

#classification report dando fit no oversample e predict na variável não oversampleada
#separar em train_test_split
# fit - ajusta o modelo aos dados - predict - prever o resultado


# separar em x-y
# train-test-split
# cross val score - apenas no x de treino
# trazer mais dados limit
# exportar modelo - pickle, joblib