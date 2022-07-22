# Databricks notebook source
# MAGIC %run "/pipe_modelos/extrator_recupera/v2/00.le_regras_disponibiliza_variaveis_v2"

# COMMAND ----------

dbutils.widgets.text('credor', '', '')
credor = dbutils.widgets.get('credor')
dbutils.notebook.exit(credor+'weeee')