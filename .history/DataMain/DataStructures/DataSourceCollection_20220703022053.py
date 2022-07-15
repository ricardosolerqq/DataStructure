#%%
from . import DataConnection

#%%
# Criadas as instâncias das DATASOURCES e das DATACONNECTIONS, podemos caracterizar cada uma das fontes de dados com seus principais elementos.

# DATASOURCES -- MONGO DB

mdb = DataSource()
mdb.name = "MONGO_DB"
mdb.description = "CONEXÃO COM BANCO MONGO DB UTILIZANDO O NÓ DE ANALYTICS"
mdb.connectionsList = []
#-------------------------------------------------------------------------
mdbCon = DataConnection()
mdbCon.name = "MONGO DB DIRECT ANALYTICS"
mdbCon.description = "CONEXÃO COM BANCO MONGO DB UTILIZANDO O DRIVE PYTHON PYMONGO EM CONEXÃO COM O NÓ DE ANALYTICS"
mdbCon.conectionString = "MDB_ANALYTICS_CON_STR"

mdb.connectionsList.append(mdbCon)

# DATASOURCES -- BLOB DO TIME DE DADOS

DataAzBlob = DataSource()
DataAzBlob.name = "qqdatastoragemain"
DataAzBlob.description = "CONEXÃO COM O BLOB AZURE DE ARMAZENAMENTO DO TIME DE DADOS"
DataAzBlob.connectionsList = []
#-------------------------------------------------------------------------
DataAzBlobCon = DataConnection()
DataAzBlobCon.name = "DATA AZURE BLOB DATABRICKS MOUNT"
DataAzBlobCon.description = "CONEXÃO COM O DATA AZURE BLOB UTILIZANDO MOUNT PARA DATABRICKS "
DataAzBlobCon.key = "DATAAZBLOB_KEY"

DataAzBlob.connectionsList.append(DataAzBlobCon)

# DATASOURCES -- BLOB DO TIME DE TI

ItAzBlob = DataSource()
ItAzBlob.name = "qqprd"
ItAzBlob.description = "CONEXÃO COM O BLOB AZURE DE ARMAZENAMENTO DO TIME DE TI"
ItAzBlob.connectionsList = []
#-------------------------------------------------------------------------
ItAzBlobCon = DataConnection()
ItAzBlobCon.name = "DATA AZURE BLOB DATABRICKS MOUNT"
ItAzBlobCon.description = "CONEXÃO COM O DATA AZURE BLOB UTILIZANDO MOUNT PARA DATABRICKS "
ItAzBlob.key = "ITAZBLOB_KEY"

ItAzBlob.connectionsList.append(ItAzBlobCon)

#COLEÇÃO DE TODAS AS BASES DE DADOS UTILIZADAS PELA EQUIPE

DS_LIST = []

#adicionando à instância criada

DS_LIST.append(mdb)
DS_LIST.append(DataAzBlob)
DS_LIST.append(ItAzBlob)
[DS.name for DS in DS_LIST]
