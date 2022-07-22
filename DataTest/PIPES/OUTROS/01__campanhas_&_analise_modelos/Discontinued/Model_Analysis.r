# Databricks notebook source
# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

# MAGIC %sh
# MAGIC ls -l /usr/bin/java
# MAGIC ls -l /etc/alternatives/java
# MAGIC ln -s /usr/lib/jvm/java-8-openjdk-amd64 /usr/lib/jvm/default-java
# MAGIC R CMD javareconf

# COMMAND ----------

install.packages(c("rJava", "RJDBC"))

# COMMAND ----------

dyn.load('/usr/lib/jvm/java-8-openjdk-amd64/jre/lib/amd64/server/libjvm.so')
library(rJava)

# COMMAND ----------

# MAGIC %sh apt-get install libssl-dev

# COMMAND ----------

# MAGIC %sh apt-get install libsasl2-dev

# COMMAND ----------

# DBTITLE 1,Lendo os pacotes necessários
pacotes <- c("readr","stringr","data.table","rlist","tictoc","beepr",
             "mongolite","dplyr","tidyr","tidyverse")

if(sum(as.numeric(!pacotes %in% installed.packages())) != 0){
  instalador <- pacotes[!pacotes %in% installed.packages()]
  for(i in 1:length(instalador)) {
    install.packages(instalador, dependencies = T)
    break()}
  sapply(pacotes, require, character = T) 
} else {
  sapply(pacotes, require, character = T) 
}

# COMMAND ----------

# DBTITLE 1,Funções a serem utilizadas
data.format.mongo=function(entrada){
  return(as.POSIXlt(as.POSIXct(entrada,tz="GMT",format="%Y-%m-%dT%H:%M:%OS"),tz="America/Sao_Paulo"))
}
###Formata as datas do mongo, ajustando as 3 horas de diferença

# COMMAND ----------

# DBTITLE 1,Selecionando os credores ativos na QQ
Base_creditor=try(mongo(collection = 'col_creditor',db='qq',url = "mongodb://victor-gomes:pg0NXoKaR2PVYo0r@qq-prd-shard-00-00-pri.qxofh.azure.mongodb.net:27017,qq-prd-shard-00-01-qxofh.azure.mongodb.net:27017,qq-prd-shard-00-02-qxofh.azure.mongodb.net:27017,qq-prd-shard-00-03-qxofh.azure.mongodb.net:27017/test?ssl=true&replicaSet=qq-prd-shard-0&authSource=admin&retryWrites=true&readPreference=secondary&readPreferenceTags=nodeType:ANALYTICS&w=majority&sockettimeoutms=3000000"))

query_credores='[
  {
    "$match" : {
        "active" : true,
        "_id" : {"$nin" : ["fake","fake2","pernambucanas"]},
        "_id" : {"$in" : ["recovery"]}
      }
  },
  {
    "$project" : {
        "_id" : 1
      }
  }
]'
credores=Base_creditor$aggregate(options = '{"allowDiskUse" : true}',pipeline = query_credores)[,1]
credores

# COMMAND ----------

# DBTITLE 1,Ajustando diretórios
dir_orig='/dbfs/mnt/qqproductionstore/Models_Campaign/complet_base_to_analysis'
setwd(dir_orig)

# COMMAND ----------

# DBTITLE 1,Atualizando as bases  de campanha
#data_hj=as.Date(data.format.mongo(Sys.time()))-1
ult_data=read.table(file.path(dir_orig,'Pasta_auxiliar','Ultima_data_leitura.txt'))[,1]

###Localizando os .zips com datas onde não houve campanha###

zips_not_add=list.files()[str_detect(list.files(),".zip")]
datas_not_add=str_remove(str_split_fixed(zips_not_add,pattern='_',n=4)[,4],'.zip')
datas_not_add=which(as.Date(datas_not_add,format="%d-%m-%Y")>ult_data)
zips_not_add=zips_not_add[datas_not_add]
zips_not_add

# COMMAND ----------

i=9
datas_unicas=unique(str_split_fixed(str_remove(str_split_fixed(zips_not_add,pattern='_',n=4)[,4],'.zip'),pattern=' ',n=2)[,1])
#zips_aux=zips_not_add[str_detect(zips_not_add,str_split_fixed(datas_unicas[i],pattern=' ',n=2)[,1])]
#zips_aux
datas_unicas

# COMMAND ----------

datas_unicas=unique(sort((as.Date(str_remove(str_split_fixed(zips_not_add,pattern='_',n=4)[,4],'.zip'),format="%d-%m-%Y"))))
for(i in 1:length(datas_unicas)){
  print(paste("Ajustando os arquivos dos dia ",format(datas_unicas[i],"%d/%m/%Y"),"...",sep=""))
  zips_aux=zips_not_add[str_detect(zips_not_add,format(datas_unicas[i],"%d-%m-%Y"))]
}
zips_aux

# COMMAND ----------

setwd(dir_orig)
for(j in 1:length(zips_aux)){
  print(paste("Extraindo o arquivo ",zips_aux[j],'...',sep=""))
  unzip(zips_aux[j],exdir=file.path(dir_orig,'Pasta_auxiliar'))
}

# COMMAND ----------

setwd(file.path(dir_orig,'Pasta_auxiliar'))
aux_pastas=list.files()[!str_detect(list.files(),"[.]")]

for(i in 1:length(aux_pastas)){
  print(paste("Executando os arquivos ",aux_pastas[i],"...",sep=""))
  
}

# COMMAND ----------

####BASES ACIONADAS###
setwd(file.path(dir_orig,'Pasta_auxiliar',aux_pastas[2]))
aux_files_move=list.files()
aux_files_move
arq_acions=c()
aux_files_moves_2=list.files(file.path(dir_orig,'Pasta_auxiliar',aux_pastas[2],aux_files_move[1]))
arq_acions[1]=file.path(dir_orig,'Pasta_auxiliar',aux_pastas[2],aux_files_move[1],aux_files_moves_2[!str_detect(aux_files_moves_2,"NÃO ATIVAR")])
#file.rename(from=arq_acions[1],to=)
setwd(dir_orig)
if(aux_pastas[2]%in%list.files()){
  setwd(file.path(dir_orig,aux_pastas[2]))
  teste_max=list.files()[max(order(list.files()))]
  file.rename(from=arq_acions[1],to=file.path(dir_orig,aux_pastas[2],teste_max,'Base_Acionada',aux_files_moves_2[!str_detect(aux_files_moves_2,"NÃO ATIVAR")]))
}else{
  file.rename(from=arq_acions[1],to=file.path(dir_orig,aux_pastas[2],'TESTE_1/Base_Acionada',aux_files_moves_2[!str_detect(aux_files_moves_2,"NÃO ATIVAR")]))
}

###BASES CONTROLE####
setwd(file.path(dir_orig,'Pasta_auxiliar',aux_pastas[2]))
arq_contr=c()
arq_contr[1]=file.path(dir_orig,'Pasta_auxiliar',aux_pastas[2],aux_files_move[1],aux_files_moves_2[str_detect(aux_files_moves_2,"NÃO ATIVAR")])
#file.rename(from=arq_acions[1],to=)
setwd(dir_orig)
if(aux_pastas[2]%in%list.files()){
  setwd(file.path(dir_orig,aux_pastas[2]))
  teste_max=list.files()[max(order(list.files()))]
  file.rename(from=arq_contr[1],to=file.path(dir_orig,aux_pastas[2],teste_max,'Base_Controle',aux_files_moves_2[str_detect(aux_files_moves_2,"NÃO ATIVAR")]))
}else{
  file.rename(from=arq_acions[1],to=file.path(dir_orig,aux_pastas[2],'TESTE_1/Base_Controle',aux_files_moves_2[str_detect(aux_files_moves_2,"NÃO ATIVAR")]))
}

# COMMAND ----------

arq_contr=c()
arq_contr[1]=file.path(dir_orig,'Pasta_auxiliar',aux_pastas[2],aux_files_move[1],aux_files_moves_2[str_detect(aux_files_moves_2,"NÃO ATIVAR")])
file.path(dir_orig,aux_pastas[2],teste_max,'Base_Controle',aux_files_moves_2[!str_detect(aux_files_moves_2,"NÃO ATIVAR")])

# COMMAND ----------

setwd(file.path(dir_orig,aux_pastas[2]))
teste_max=list.files()[max(order(list.files()))]
file.rename(from=arq_acions[1],to=file.path(dir_orig,aux_pastas[2],teste_max,'Base_Acionada',aux_files_moves_2[!str_detect(aux_files_moves_2,"NÃO ATIVAR")]))

# COMMAND ----------

# DBTITLE 1,Query de localização das informações
print("Construindo a query das informações de telefones...")

query_infos_tels_p1='[
	{
		"$match" : {
			"document" : {"$in" :['
query_infos_tels_p2=']}			
		}
	},
	{
		"$project" : {
			"_id" : 1,
			"document" : 1,
			"phones" : "$info.phones",
			"scores" : "$info.scores",
			"pos_score" : {
							"$map" : {
										"input" : "$info.scores.provider",
										"as" : "provider",
										"in" : {"$cond" : [{"$eq" : ["$$provider","qq_credz_propensity_v1"]},
															1,
															0
														]
												}
										}
						}
		}
	},
	{
		"$unwind" : "$phones"
	},
	{
		"$match" : {
			"phones.phone" : {"$in" : ['
query_infos_tels_p3=']}
		}
	},
	{
		"$addFields" : {
			"pos_tag_contato" : {
								"$map" : {
											"input" : "$phones.tags",
											"as" : "phtags",
											"in" : {"$cond" : [{"$in" : ["$$phtags",['
query_infos_tels_p4=']]},
																1,
																0
															]
													}
											}
							}
		}
	},
	{
		"$project" : {
			"_id" : 1,
			"Documento" : "$document",
			"score_model" : {"$arrayElemAt" : ["$scores.score",{"$indexOfArray" :["$pos_score",{"$max" :  "$pos_score"}]}]},
			"tag_phones" : {"$arrayElemAt" : ["$phones.tags",{"$indexOfArray" :["$pos_tag_contato",{"$max" :  "$pos_tag_contato"}]}]},
			"telefone" : {"$concat" : ["$phones.areaCode","$phones.number"]},
			"chave" : {"$concat" : ["$document",":","$phones.phone"]}
		}
	}
]'

print("Carregando a col_person...")
Base_person=try(mongo(collection = 'col_person',db='qq',url = "mongodb://victor-gomes:pg0NXoKaR2PVYo0r@qq-prd-shard-00-00-pri.qxofh.azure.mongodb.net:27017,qq-prd-shard-00-01-qxofh.azure.mongodb.net:27017,qq-prd-shard-00-02-qxofh.azure.mongodb.net:27017,qq-prd-shard-00-03-qxofh.azure.mongodb.net:27017/test?ssl=true&replicaSet=qq-prd-shard-0&authSource=admin&retryWrites=true&readPreference=secondary&readPreferenceTags=nodeType:ANALYTICS&w=majority&sockettimeoutms=3000000"))
mongo_options(date_as_char = TRUE)

# COMMAND ----------

# DBTITLE 1,Query de localização dos acordos
print("Construindo a query de seleção dos acordos...")
query_deals_p1='[
	{
		"$match" : {
			"documentType" : "cpf",
			"deals" : {
				"$elemMatch" : {
					"creditor" : "credz",
					"status" : {"$ne" : "error"},
					"createdAt" : {"$gte" : {"$date": "'
query_deals_p2='T03:00:00.000Z"}}
				}
			}
		}
	},
	{
		"$unwind" : "$deals"
	},
	{
		"$match" : {
			"deals.creditor" : "credz",
			"deals.status" : {"$ne" : "error"},
			"deals.createdAt" : {"$gte" : {"$date": "'
query_deals_p3='T03:00:00.000Z"}} 
		}
	},
	{
		"$project" : {
			"_id" : 0,
			"ID_acordo" : "$deals._id",
			"CPF" : "$document",
			"total_acordo" : "$deals.totalAmount",
			"data_acordo" : "$deals.createdAt",
			"total_parcelas" : "$deals.totalInstallments",
			"valor_parcelas" : "$deals.installmentValue",
			"valor_entrada" : "$deals.upfront",
			"canal" : {
                        "$cond": [
                            { "$ifNull": ["$deals.tracking.channel", false] },
                            "$deals.tracking.channel",
                            {
                                "$cond": [
                                    { "$ifNull": ["$deals.tracking.utms.source", false] },
                                    "$deals.tracking.utms.source",
                                    {
                                        "$cond": [
                                            { "$ifNull": ["$deals.offer.tokenData.channel", false] },
                                            "$deals.offer.tokenData.channel",
                                            { "$cond": [{ "$ifNull": ["$deals.simulationID", false] }, "web", "unknown"] }
                                        ]
                                    }
                                ]
                            }
                        ]
                    }  
		}
	}
]'

# COMMAND ----------

# DBTITLE 1,Query de seleção das simulações do período
print("Construindo as querys das simulações...")

query_sim_p1='[
	{
		"$match" : {
			"personID" : {"$in" : ['
query_sim_p2=']},
			"createdAt" : {"$gte" : {"$date": "'
query_sim_p3='T03:00:00.000Z"}},
			"creditor" : "credz"
		}
	},
	{
		"$project" : {
			"_id" : 0,
			"ID_doc" : "$personID",
			"sim_date" : "$createdAt" 
		}
	}
]'

Base_sim=try(mongo(collection = 'col_simulation',db='qq',url = "mongodb://victor-gomes:pg0NXoKaR2PVYo0r@qq-prd-shard-00-00-pri.qxofh.azure.mongodb.net:27017,qq-prd-shard-00-01-qxofh.azure.mongodb.net:27017,qq-prd-shard-00-02-qxofh.azure.mongodb.net:27017,qq-prd-shard-00-03-qxofh.azure.mongodb.net:27017/test?ssl=true&replicaSet=qq-prd-shard-0&authSource=admin&retryWrites=true&readPreference=secondary&readPreferenceTags=nodeType:ANALYTICS&w=majority&sockettimeoutms=3000000"))

# COMMAND ----------

# DBTITLE 1,Query de pagamentos
print("Construindo as querys dos pagamentos...")

query_payments_p1='[
	{
		"$match" : {
			"document" : {"$in" : ['
query_payments_p2=']}
		}
	},
	{
		"$unwind" : "$installments"
	},
	{
		"$match" : {
			"installments.dealID" : {"$in" : ['
query_payments_p3=']},
			"installments.status" : "paid"
		}
	},
	{
		"$group" : {
			"_id" : "$installments.dealID",
			"tot_pay" : {"$sum" : {"$cond" : [{"$eq" : ["$installments.payment.paidAmount",0]},
												"$installments.installmentAmount",
												"$installments.payment.paidAmount"]}},
			"tot_parc_pags" : {"$sum" : 1.0}
		}
	},
	{
		"$project" : {
			"_id" : 0,
			"ID_acordo" : "$_id",
			"total_pago" : "$tot_pay",
			"total_parcelas_pagas" : "$tot_parc_pags"
		}
	}
]'

# COMMAND ----------

# DBTITLE 1,Montando as bases acionadas
