# Databricks notebook source
# DBTITLE 1,Executando o primeiro código (seleção da base atual recovery)
# MAGIC %run "/pipe_modelos/modelo-propensao-recovery/model_version_V2(Elisvaldo)/02- select_current_base"

# COMMAND ----------

# DBTITLE 1,Executando o segundo código  (formatação da base selecionada)
# MAGIC %run "/pipe_modelos/modelo-propensao-recovery/model_version_V2(Elisvaldo)/03- formating_current_base"

# COMMAND ----------

# DBTITLE 1,Executando o terceiro código (execução e inserção da base escorada no mongoDB)
# MAGIC %run "/pipe_modelos/modelo-propensao-recovery/model_version_V2(Elisvaldo)/04- scoring_current_base"

# COMMAND ----------

# DBTITLE 1,Sucesso!!!
print("FIM!!!")