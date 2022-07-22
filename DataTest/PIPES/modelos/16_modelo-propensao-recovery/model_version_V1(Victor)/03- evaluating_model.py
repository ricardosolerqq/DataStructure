# Databricks notebook source
# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

# DBTITLE 1,montando qq-data-studies no cluster
mount_blob_storage_key(dbutils,'qqdatastoragemain','qq-data-studies','/mnt/qq-data-studies')

# COMMAND ----------

# DBTITLE 1,Carregando a base e ajustando os diretórios
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn import set_config
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve,precision_recall_fscore_support,classification_report,precision_recall_curve,f1_score
import os
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from scipy.stats import ks_2samp
import pickle
from itertools import chain
from numpy import argmax,arange,sqrt

def transform_time(X):
    """Convert Datetime objects to seconds for numerical/quantitative parsing"""
    df = pd.DataFrame(X)
    return df.apply(lambda x: pd.to_datetime(x).apply(lambda x: x.timestamp()))

file = '/dbfs/mnt/qq-data-studies/Recovery_model/Tabela completa formatada para execução do modelo.csv'
dir_orig='/dbfs/mnt/qq-data-studies/Recovery_model'
Base_recov = pd.read_csv(file,sep=";",index_col='Chave')
#Base_recov

# COMMAND ----------

# DBTITLE 1,Formatando as variáveis do modelo
###Retirando variáveis descritivas e as que não podem ser extraídas na validação##
Base_recov=Base_recov.drop(columns=['IdContatoSIR','CPF','Nome_Cliente / Empresa','Numero_Contrato','VlDividaAtualizado','SubTipo Produto'])


###Colocando os valores de data no formato correto##
Base_recov['Data de referência']=pd.to_datetime(Base_recov['Data de referência'],format='%Y-%m-%d %H:%M:%S')
Base_recov['Data_Mora']=pd.to_datetime(Base_recov['Data_Mora'],format='%Y-%m-%d %H:%M:%S')

                                                      
Base_recov['Data de referência']=transform_time(Base_recov['Data de referência'])
Base_recov['Data_Mora']=transform_time(Base_recov['Data_Mora'])

# COMMAND ----------

# DBTITLE 1,Pré-processamento das variáveis
TEST_SIZE = 0.2
RANDOM_STATE = 42
#N_SPLITS = 3

impute = SimpleImputer(strategy='mean')
scaler = MinMaxScaler()
ohe = OneHotEncoder(handle_unknown='ignore')

numeric_feat = ['VlSOP','Aging','Data de referência','Data_Mora']
pipe_numeric_transf = Pipeline([('SimpleImputer', impute),
                               ('MinMaxScaler', scaler)])

categ_feat = ['Class_Carteira','Class_Produto','Class_Portfolio']
pipe_categ_feat = Pipeline([('OneHotEncoder', ohe)])

preprocessor = ColumnTransformer([('Pipe_Numeric', pipe_numeric_transf, numeric_feat),
                                 ('Pipe_Categoric', pipe_categ_feat, categ_feat)],
                                 remainder='passthrough')

# COMMAND ----------

# DBTITLE 1,Tabela com os nomes das variáveis
variable_names=[['VlSOP','Aging','Data de referência','Data_Mora']]
variable_names=variable_names +[Base_recov['Class_Carteira'].sort_values().unique()+'_carteira',Base_recov['Class_Produto'].sort_values().unique()+'_produto',Base_recov['Class_Portfolio'].sort_values().unique()+'_portfolio',['Acion_5']]
variable_names=list(chain.from_iterable(variable_names))
print(variable_names)

# COMMAND ----------

# DBTITLE 1,Separando as variáveis de Treino e Teste além do desbalanceamento da base
##Separando input e output
X = Base_recov.drop(columns='Deals_30')
y = Base_recov.loc[:, 'Deals_30']

###pré-processando todas as variáveis###
x_pretransform=preprocessor.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(x_pretransform, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
smt = SMOTE(random_state=RANDOM_STATE)
X_train_SMOTE, y_train_SMOTE = smt.fit_resample(X_train, y_train)

# COMMAND ----------

# DBTITLE 1,Carregando o modelo
modelo_rec=pickle.load(open(dir_orig+'/model_fit_recovery_complete.sav', 'rb'))

# COMMAND ----------

# DBTITLE 1,Montando a tabela com os coeficientes do modelo e suas respectivas variáveis
pd.set_option('display.float_format', lambda x: '%.9f' % x)

###Selecionando as variáveis##
odds=list(np.exp(modelo_rec.coef_[0]))
Tab_coef=pd.DataFrame(data=list(zip(variable_names, odds)),columns=['variable','coefficient'])

###Carregando a tabela das classificações reais##
dir_class=dir_orig+'/Tabela das classificações'
arq=sorted(os.listdir(dir_class))
real_class=[]
for i in range(0,len(arq)):
  aux=pd.read_csv(dir_class+'/'+arq[i],sep=";")
  aux.columns=['real','classification']
  real_class.append(aux)
real_class=pd.concat(real_class,ignore_index=True)
real_class

###Montando a tabela com as variáveis reais##
aux=pd.DataFrame(list(Tab_coef.iloc[4:37,0].str.split("_")))
real_class['var_name']=real_class['real']+'_'+aux.iloc[:,1]
Tab_coef.iloc[4:37,0]=real_class['var_name'].tolist()
Tab_coef.sort_values(by='coefficient',ascending=False)

#del real_class,aux,arq,odds

# COMMAND ----------

# DBTITLE 1,Avaliando a base de treino
#modelo_rec.fit(X_train_SMOTE,y_train_SMOTE)
##Avaliando as predições##
y_pred_train = modelo_rec.predict(X_train_SMOTE)
print(classification_report(y_train_SMOTE, y_pred_train))

###Avaliando as demais métricas##
y_pred_prob_train=modelo_rec.predict_proba(X_train_SMOTE)[:,1]
df_results_train=pd.DataFrame()
fpr_train , tpr_train, thresholds_train = roc_curve(y_train_SMOTE, y_pred_prob_train)
roc_train=roc_auc_score(y_train_SMOTE,y_pred_prob_train)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr_train,tpr_train,label='logisticRegression_train (AUC= '+str(roc_train.round(decimals=4))+')')
plt.legend()
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title('Curvas ROC do modelos de treinamento')
#plt.savefig(dir_retorn+'/Curvas_ROC_modelos.png')
plt.show()
df_results_train.insert(df_results_train.shape[1],'Logistic Regression',[roc_train,2*roc_train-1,accuracy_score(y_train_SMOTE, y_pred_train),precision_recall_fscore_support(y_train_SMOTE,y_pred_train)[1][0],precision_recall_fscore_support(y_train_SMOTE,y_pred_train)[1][1],ks_2samp(y_pred_prob_train[y_train_SMOTE==True],y_pred_prob_train[y_train_SMOTE==False]).statistic])
df_results_train.insert(df_results_train.shape[1],'metrics',['AUC','GINI','accuracy','specificity','precision','KS'])
df_results_train.set_index('metrics',inplace=True,drop=True)
df_results_train

# COMMAND ----------

# DBTITLE 1,Avaliando base de teste
#modelo_rec.fit(X_train_SMOTE,y_train_SMOTE)
##Avaliando as predições##
y_pred = modelo_rec.predict(X_test)
print(classification_report(y_test, y_pred))

###Avaliando as demais métricas##
y_pred_prob=modelo_rec.predict_proba(X_test)[:,1]
df_results=pd.DataFrame()
fpr , tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc=roc_auc_score(y_test,y_pred)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr,label='logisticRegression (AUC= '+str(roc.round(decimals=4))+')')
plt.legend()
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title('Curvas ROC dos modelos')
#plt.savefig(dir_retorn+'/Curvas_ROC_modelos.png')
plt.show()
df_results.insert(df_results.shape[1],'Logistic Regression',[roc,2*roc-1,accuracy_score(y_test, y_pred),precision_recall_fscore_support(y_test,y_pred)[1][0],precision_recall_fscore_support(y_test,y_pred)[1][1],ks_2samp(y_pred_prob[y_test==True],y_pred_prob[y_test==False]).statistic])
df_results.insert(df_results.shape[1],'metrics',['AUC','GINI','accuracy','specificity','precision','KS'])
df_results.set_index('metrics',inplace=True,drop=True)
df_results

# COMMAND ----------

# DBTITLE 1,Avaliando os Threshold
df_best_threshold=pd.DataFrame()
##G-means##
g_means=sqrt(tpr * (1-fpr))
ix=argmax(g_means)
##inserindo na tabela##
df_best_threshold.insert(df_best_threshold.shape[1],'G-mean',[thresholds[ix]])

##Youden’s J statistic##
J=tpr-fpr
ix_J=argmax(J)
##inserindo na tabela##
df_best_threshold.insert(df_best_threshold.shape[1],'Youden’s J statistic',[thresholds[ix_J]])

##Precision-Recall curve##
precision, recall, thresholds_prc = precision_recall_curve(y_test, y_pred_prob)
fscore = (2 * precision * recall) / (precision + recall)
fscore=list(filter(lambda x: str(x) != 'nan', fscore))
ix_prc = argmax(fscore)
##inserindo na tabela##
df_best_threshold.insert(df_best_threshold.shape[1],'Precision-Recall curve',[thresholds_prc[ix_prc]])

# apply threshold to positive probabilities to create labels
def to_labels(pos_probs, threshold):
	return (pos_probs >= threshold).astype('int')

thresholds_teste = arange(0, 1, 0.005)
scores = [f1_score(y_test, to_labels(y_pred_prob, t)) for t in thresholds_teste]
ix_app=argmax(scores)
df_best_threshold.insert(df_best_threshold.shape[1],'apply threshold',[thresholds_teste[ix_app]])
df_best_threshold.to_csv(dir_orig+'/Threshold tables.csv',sep=';',index=False)
df_best_threshold

# COMMAND ----------

# DBTITLE 1,Comparando os métodos de Threshold
df_roc_threshold=pd.DataFrame()
for i in range(0,df_best_threshold.shape[1]):
  y_pred_threshold=(modelo_rec.predict_proba(X_test)[:,1] >=round(df_best_threshold.iloc[0,i],8)).astype(bool)
  roc_threshold=roc_auc_score(y_test,y_pred_threshold)
  df_roc_threshold.insert(df_roc_threshold.shape[1],list(df_best_threshold.columns)[i],[roc_threshold,2*roc_threshold-1,accuracy_score(y_test, y_pred_threshold),precision_recall_fscore_support(y_test,y_pred_threshold)[1][0],precision_recall_fscore_support(y_test,y_pred_threshold)[1][1]])

df_roc_threshold.insert(df_best_threshold.shape[1],'metrics',['AUC','GINI','accuracy','precision','specificity'])  
df_roc_threshold.set_index('metrics',inplace=True,drop=True)

df_roc_threshold