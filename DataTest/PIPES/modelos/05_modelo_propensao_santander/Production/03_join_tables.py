# Databricks notebook source
import datetime
date = datetime.datetime.today()
date = str(date.year)+'-'+str(date.month).zfill(2)+'-'+str(date.day).zfill(2)
#date = '2022-06-21'
date

# COMMAND ----------

from pyspark.sql.functions import col

# COMMAND ----------

#a) Débitos: Tab_vals1.to_csv(dir_debts_datas+'/Base_UFPEL_QQ_Model_std_P'+str(j).zfill(2)+'.csv',index=False,sep=";")

path = "{0}{1}{2}".format("/mnt/etlsantander/Base_UFPEL/Skips_divids/Dados=",date,"/Base_UFPEL_QQ_Model_std_*.csv")
df_debitos = (spark.read
               .format("com.databricks.spark.csv")
               .option("header", "true")
               .option("inferSchema", "true")
               .option("delimiter", ";")
               .load(path))

# COMMAND ----------

#b) Contatos: tab_training.to_csv(dir_phones_datas+'/Base_UFPEL_QQ_Model_std_phones_P'+str(j).zfill(2)+'.csv',index=False,sep=";")

path = "{0}{1}{2}".format("/mnt/etlsantander/Base_UFPEL/Tags_contatos_phones/Dados=",date,"/Base_UFPEL_QQ_Model_std_phones_*.csv")

df_contatos = (spark.read
                 .format("com.databricks.spark.csv")
                 .option("header", "true")
                 .option("inferSchema", "true")
                 .option("delimiter", ";")
                 .load(path))

# COMMAND ----------

df_join = (df_debitos.alias("debt").join(df_contatos.alias("contact"), \
            col("debt.ID_DEVEDOR") == col("contact.ID_DEVEDOR"),"left") \
            .select(col("debt.*"),col("contact.*")) \
            .drop(col("contact.ID_DEVEDOR")) )

# COMMAND ----------

qt_lines = df_join.count()
chunkies = round(qt_lines / 100000)
print(chunkies)

# COMMAND ----------

pathSink = "{0}{1}/".format("/mnt/etlsantander/Base_UFPEL/Join_Skips_Divids_Contact/repart/Data_Arquivo=",date)
print(pathSink)

# COMMAND ----------

df_join.repartition(chunkies).write.mode('overwrite').option("sep",";").option("header","true").csv(pathSink)