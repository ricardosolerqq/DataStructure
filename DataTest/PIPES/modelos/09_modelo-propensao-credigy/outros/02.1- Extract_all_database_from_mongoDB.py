# Databricks notebook source
# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

# DBTITLE 1,Selecionando os diretórios
mount_blob_storage_oauth(dbutils,'qqprd','ml-prd',"/mnt/ml-prd")
dir_save='/mnt/ml-prd/ml-data/propensaodeal/credigy/mongoDB_database'

# COMMAND ----------

import pymongo
import pandas as pd
import os
import sys
from datetime import datetime
from itertools import chain

# COMMAND ----------

# DBTITLE 1,Extraindo todos os IDs com dívida ativa
col_person = getPyMongoCollection('col_person')

query_Ids_ativos=[
	{
		"$match" : {
			"debts" : {
				"$elemMatch" : {
					"creditor" : "credigy",
					"status" : "active"
				}
			}
		}
	},
	{
		"$project" : {
			"_id" : 1
		}
	}
  ]

IDs=list(col_person.aggregate(pipeline=query_Ids_ativos,allowDiskUse=True))
ID=list((map(lambda a : list(a.values())[0],IDs)))

# COMMAND ----------

# DBTITLE 1,Selecionando as variáveis do modelo
partes=list(range(0,len(ID),10000))
if(partes[-1]!=len(ID)):
  partes.append(len(ID))
  

Tab_features=[]  
for i in range(0,len(partes)-1):
    print("\nExecutando a parte "+str(i+1)+" de "+str(len(partes)-1)+"...")    
    aux_IDs=ID[partes[i]:(partes[i+1])]
    query_variables=[
      {
          "$match" : {
              "_id" : {"$in" : aux_IDs}
          }
      },
      {
          "$project" : {
              "_id" : 1,
              "document" : 1,
              "name" : {"$cond" : [{"$ifNull" : ["$info.names",False]},
                                          {"$arrayElemAt" : ["$info.names.name",0]},
                                          ""]
                      },
              "birthdate" : {"$ifNull" : ["$info.birthDate",""]},
              "gender" : {"$cond" : [{"$in" : [{"$ifNull" : ["$info.gender","U"]},["M","F"]]},
                                      "$info.gender",
                                      "U"]},
              "emails" : "$info.emails",
              "additionalInfo" : "$info.additionalInfo",
              "debts" : 1
          }
      },
      {
          "$addFields" : {
              "emails" : {
                          "$filter" : {
                              "input" : {"$map" : {
                                                  "input" : {"$ifNull" : ["$emails",[]]},
                                                  "as" : "i",
                                                  "in" : {
                                                      "email" : "$$i.email",
                                                      "dataProvider" : "$$i.dataProvider"
                                                  }

                                              }
                                      },
                              "cond": {
                                      "$in" : ["credigy",{"$ifNull" : ["$$this.dataProvider.provider",[]]}]
                                  }
                          }
                      },
              "PersonId" : {
                          "$filter" : {
                              "input" : {"$map" : {
                                                  "input" : "$debts",
                                                  "as" : "i",
                                                  "in" : {
                                                      "creditor" : "$$i.creditor",
                                                      "creditorPersonId" : "$$i.additionalInfo.creditorPersonId",
                                                      "data_cadastro" : "$$i.lastETL"
                                                  }

                                              }
                                      },
                              "cond": {
                                      "$eq" : ["$$this.creditor","credigy"]
                                  }
                          }
                      },     

          }
      },
      {
          "$project" : {
              "_id" : 0,
              "DOCUMENT" : "$document",
              "ID_PESSOA" : {"$arrayElemAt" : ["$PersonId.creditorPersonId",0]},
              "NAME_PESSOA" : "$name",
              "DT_NASC" : {"$toString" : "$birthdate"},
              "EMAIL" : 	{"$cond" : [{"$ifNull" : ["$emails",False]},
                                      {"$arrayElemAt" : ["$emails.email",-1]},
                                          ""]
                          },
              "GENERO" : "$gender",
              "EST_CIVIL" : {"$ifNull" : ["$additionalInfo.credigy:SitCivil","00"]},
              "TP_PESSOASYS" : {"$ifNull" : ["$additionalInfo.credigy:PessoaSys","1"]},
              "DT_CADASTRO" : {"$ifNull" : ["$additionalInfo.credigy:CadastroData",{"$arrayElemAt" : ["$PersonId.data_cadastro",0]}]}	
          }
      }
    ]
    Tab_features.append(list(col_person.aggregate(pipeline=query_variables,allowDiskUse=True)))  


Tab_features=spark.sparkContext.parallelize(list(chain.from_iterable(Tab_features))).toDF(sampleRatio=0.05)

# COMMAND ----------

# DBTITLE 1,Salvando
Tab_features.write.mode('overwrite').option("header",True).option("delimiter",";").option("emptyValue",'').csv(dir_save+'/data_base_to_score_2022-02-08')

# COMMAND ----------

Tab_features.write.mode('overwrite').option("header",True).option("delimiter",";").option("emptyValue",'').parquet(dir_save+'/parquet_data_base_to_score_2022-02-08')