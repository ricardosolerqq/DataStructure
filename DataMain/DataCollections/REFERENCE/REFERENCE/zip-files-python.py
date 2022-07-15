# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Reading Zip Files to Spark with Python

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Retrieve a sample zip file

# COMMAND ----------

import urllib 
urllib.request.urlretrieve("https://resources.lendingclub.com/LoanStats3a.csv.zip", "/tmp/LoanStats3a.csv.zip")

# COMMAND ----------

# MAGIC %md ### Unzip file and clean up
# MAGIC 1. Unzip the file.
# MAGIC 1. Remove first comment line.
# MAGIC 1. Remove unzipped file.

# COMMAND ----------

# MAGIC %sh
# MAGIC unzip /tmp/LoanStats3a.csv.zip
# MAGIC tail -n +2 LoanStats3a.csv > temp.csv
# MAGIC rm LoanStats3a.csv

# COMMAND ----------

# MAGIC %md ### Move temp file to DBFS

# COMMAND ----------

dbutils.fs.mv("file:/databricks/driver/temp.csv", "dbfs:/tmp/LoanStats3a.csv")  

# COMMAND ----------

# MAGIC %md ### Load file into DataFrame

# COMMAND ----------

df = spark.read.format("csv").option("inferSchema", "true").option("header","true").load("dbfs:/tmp/LoanStats3a.csv")
display(df)