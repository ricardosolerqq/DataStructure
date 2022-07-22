# Databricks notebook source
# MAGIC %run "/Shared/common-notebooks/dataprep-funcoes/initial-func"

# COMMAND ----------

# DBTITLE 1,Executando o 1º notebook
# MAGIC %run "/pipe_modelos/modelo-propensao-credigy/Production/Monthly_execution/01-Monthly_extraction"

# COMMAND ----------

# DBTITLE 1,Executando o 2º notebook
# MAGIC %run "/pipe_modelos/modelo-propensao-credigy/Production/Monthly_execution/02-read_pickle_generate_score"

# COMMAND ----------

# DBTITLE 1,Executando o 3º notebook
# MAGIC %run "/pipe_modelos/modelo-propensao-credigy/Production/Monthly_execution/03-read_pre_output_write_output"