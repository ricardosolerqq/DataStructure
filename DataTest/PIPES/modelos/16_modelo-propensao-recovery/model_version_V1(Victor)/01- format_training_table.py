# Databricks notebook source
# DBTITLE 1,Carregando as funções padrão
# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

# DBTITLE 1,Carregando a base e ajustando diretórios
import pandas as pd
import numpy as np
import os
from string import ascii_lowercase
file = '/dbfs/mnt/qq-data-studies/Recovery_model/Tabela de treino do modelo Recovery (Beta).csv'
dir_orig='/dbfs/mnt/qq-data-studies/Recovery_model'
dir_tab=dir_orig+'/Tabela das contagens'
dir_class=dir_orig+'/Tabela das classificações'
Base_recov = pd.read_csv(file,sep=";")

# COMMAND ----------

# DBTITLE 1,Incluindo a coluna 'Portfolio'
Tab_Port=Base_recov.loc[:,'Carteira'].str.split(" ",n=1,expand=True)
Base_recov['Portfolio']=Tab_Port.loc[:,0]
Base_recov['Portfolio']=Base_recov['Portfolio'].str.lower().str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
Base_recov.dtypes

# COMMAND ----------

# DBTITLE 1,Ajustando a tabela dos portifólios
aux_port=pd.read_csv(dir_tab+'/01- Contagem das carteiras.csv',sep=',')
aux_port.iloc[:,0]=aux_port.iloc[:,0].str.lower().str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
aux_port['Portifolio']=aux_port.iloc[:,0].str.split(" ",n=1,expand=True).iloc[:,0]
aux_port=aux_port.groupby('Portifolio',as_index=False)['Total','Total(%)'].sum().sort_values(by='Total',ascending=False)

if (aux_port.shape[0] <= 10) : 
    max_groups=aux.tab.shape
else:
    max_groups=10

aux_port=aux_port.iloc[0:max_groups,:]
aux_port=aux_port.append(pd.Series(['Outros', 0, 0], index=aux_port.columns), ignore_index=True)
aux_port=list(aux_port.iloc[:,0].unique()) 

Base_recov.loc[-Base_recov['Portfolio'].isin(aux_port),'Portfolio']='Outros'

Tab_class_port=pd.DataFrame()

for j in range(0,len(aux_port)) :
    Tab_class_port=Tab_class_port.append(pd.DataFrame([[aux_port[j],ascii_lowercase[j]]],columns=['Portfolio','Class_Portfolio']))

Tab_class_port.to_csv(dir_class+'/Tabela de classificação Portifolio.csv',sep=';',index=False) 
Base_recov=pd.merge(Base_recov,Tab_class_port,how='left',on=['Portfolio'])
Base_recov=Base_recov.drop('Portfolio',axis=1)

del aux_port,Tab_Port,Tab_class_port,max_groups

# COMMAND ----------

# DBTITLE 1,Ajustando as tabelas de 'produto' e 'carteira'
Tabs=sorted(os.listdir(dir_tab))
columns=['Carteira','Produto']


for i in range(0,len(columns)):
    print("\nExecutando os ajustes da feature "+str(columns[i]))
    aux_tab=pd.read_csv(dir_tab+'/'+Tabs[i],sep=',')
    aux_tab=aux_tab.rename(columns={aux_tab.columns[0] : columns[i]})
    aux_tab.iloc[:,0]=aux_tab.iloc[:,0].str.lower().str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    aux_tab=aux_tab.groupby(columns[i],as_index=False)['Total','Total(%)'].sum().sort_values(by='Total',ascending=False)
    Base_recov[columns[i]]=Base_recov[columns[i]].str.lower().str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    if (aux_tab.shape[0] <= 10) : 
        max_groups=aux.tab.shape
    else:
        max_groups=10
    
    aux_tab=aux_tab.iloc[0:max_groups,:]
    aux_tab=aux_tab.append(pd.Series(['Outros', 0, 0], index=aux_tab.columns), ignore_index=True)
    aux_tab=aux_tab.iloc[:,0].to_list()
    Base_recov.loc[-Base_recov[columns[i]].isin(aux_tab),columns[i]]='Outros'
    print("\nMontando a tabela de classificação...")
    Tab_class=pd.DataFrame()
    for j in range(0,len(aux_tab)) :
        Tab_class=Tab_class.append(pd.DataFrame([[aux_tab[j],ascii_lowercase[j]]],columns=[columns[i],'Class_'+columns[i]]))
    print("\nSalvando a tabela de classificação...")
    Tab_class.to_csv(dir_class+'/Tabela de classificação '+columns[i]+'.csv',sep=';',index=False)    
    print("\nAjustando a tabela principal...")
    Base_recov=pd.merge(Base_recov,Tab_class,how='left',on=[columns[i]])
    Base_recov=Base_recov.drop(columns[i],axis=1)   
display(Base_recov)    

# COMMAND ----------

# DBTITLE 1,Salvando a tabela formatada
Base_recov.to_csv(dir_orig+'/Tabela completa formatada para execução do modelo.csv',sep=';',index=False)