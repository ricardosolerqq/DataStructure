# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## Overview
# MAGIC 
# MAGIC This notebook shows you how to load data from JDBC databases using Spark SQL.
# MAGIC 
# MAGIC *For production, you should control the level of parallelism used to read data from the external database, using the parameters described in the documentation.*

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Step 1: Connection Information
# MAGIC 
# MAGIC This is a **Python** notebook so the default cell type is Python. However, you can use different languages by using the `%LANGUAGE` magic command. Python, Scala, SQL, and R are all supported.
# MAGIC 
# MAGIC First we'll define some variables to let us programmatically create these connections.

# COMMAND ----------

driver = "org.mongodb:mongo-java-driver:3.12.11"
url = "jdbc:mongodb://testfed-qxofh.a.query.mongodb.net/?ssl=true&authSource=admin"
table = "TestFed"
user = "ricardo-montesanti"
password = "lVQUdOpWCGRe5La3"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Step 2: Reading the data
# MAGIC 
# MAGIC Now that we specified our file metadata, we can create a DataFrame. You'll notice that we use an *option* to specify that we'd like to infer the schema from the file. We can also explicitly set this to a particular schema if we have one already.
# MAGIC 
# MAGIC First, let's create a DataFrame in Python, notice how we will programmatically reference the variables we defined above.

# COMMAND ----------

from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

spark = SparkSession.\
        builder.\
        appName("streamingExampleRead").\
        config('spark.jars.packages', 'mongo_spark_connector_10_0_2.jar').\
        getOrCreate()

# COMMAND ----------

query=spark.read.format("jdbc")\
        .option("spark.mongodb.connection.uri","mongodb://ricardo-montesanti:lVQUdOpWCGRe5La3@qq-prd-shard-00-00-pri.qxofh.azure.mongodb.net:27017,qq-prd-shard-00-01-pri.qxofh.azure.mongodb.net:27017,qq-prd-shard-00-02-pri.qxofh.azure.mongodb.net:27017,qq-prd-shard-00-03-pri.qxofh.azure.mongodb.net:27017/test?ssl=true&replicaSet=qq-prd-shard-0&authSource=admin&retryWrites=true&readPreference=secondary&readPreferenceTags=nodeType:ANALYTICS&w=majority")\
    	.option('spark.mongodb.database', 'DailyUpdate') \
    	.option('spark.mongodb.collection', 'ColSimulationAws').load()

# COMMAND ----------

query.table("

# COMMAND ----------



query=spark.readStream.format("mongodb")
.option("spark.mongodb.connection.uri","mongodb://ricardo-montesanti:lVQUdOpWCGRe5La3@qq-prd-shard-00-00-pri.qxofh.azure.mongodb.net:27017,qq-prd-shard-00-01-pri.qxofh.azure.mongodb.net:27017,qq-prd-shard-00-02-pri.qxofh.azure.mongodb.net:27017,qq-prd-shard-00-03-pri.qxofh.azure.mongodb.net:27017/test?ssl=true&replicaSet=qq-prd-shard-0&authSource=admin&retryWrites=true&readPreference=secondary&readPreferenceTags=nodeType:ANALYTICS&w=majority")
    	.option('spark.mongodb.database', 'DailyUpdate') \
    	.option('spark.mongodb.collection', 'ColSimulationAws') \
.option('spark.mongodb.change.stream.publish.full.document.only','true') \
    	.option("forceDeleteTempCheckpointLocation", "true") \
    	.load()

query.printSchema()

# COMMAND ----------

# MAGIC %fs ls "dbfs:/FileStore/jars/87b3170d_6b5d_4293_a0a9_c3adcd0b5b81-mongodb_jdbc_2_0_0-81290.jar"

# COMMAND ----------

driver = "org.mongodb:mongo-java-driver:3.12.11"
url = "jdbc:mongodb://testfed-qxofh.a.query.mongodb.net/?ssl=true&authSource=admin"
table = "TestFed"
user = "ricardo-montesanti"
password = "lVQUdOpWCGRe5La3"

# COMMAND ----------


df_sel = spark.read\
  .format("com.microsoft.sqlserver.jdbc.spark")\
  .option("url", url)\
  .option("dbtable", table)\
  .option("user", user )\
  .option("password", password )\
  .load()
df_sel

# COMMAND ----------

# MAGIC %sh $JAVA_HOME=/usr/lib/jvm/java-8-openjdk python

# COMMAND ----------

import jaydebeapi

user = "ricardo-montesanti" 
password = "lVQUdOpWCGRe5La3"

url = "jdbc:mongodb://testfed-qxofh.a.query.mongodb.net/?ssl=true&authSource=admin"


driver1 ="mongodb_jdbc_2_0_0-81290" 
driver2 = "mongo_spark_connector_10_0_2.jar"
path1 = "dbfs:/FileStore/jars/87b3170d_6b5d_4293_a0a9_c3adcd0b5b81-mongodb_jdbc_2_0_0-81290.jar"
path2 = "dbfs:/FileStore/jars/82d153ce_ebed_4109_969f_248a1f3764f6-mongo_spark_connector_10_0_2-6a0af.jar"

conn = jaydebeapi.connect( driver2,\
                          url,\
                          [user, password],\
                          path2,)\


# COMMAND ----------


curs = conn.cursor()
curs.execute('create table CUSTOMER'
             '("CUST_ID" INTEGER not null,'
             ' "NAME" VARCHAR(50) not null,'
             ' primary key ("CUST_ID"))'
            )
curs.execute("insert into CUSTOMER values (1, 'John')")
curs.execute("select * from CUSTOMER")
curs.fetchall()
[(1, u'John')]
curs.close()
conn.close()

# COMMAND ----------

from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

spark = SparkSession.\
        builder.\
        appName("streamingExampleRead").\
        config('spark.jars.packages', 'org.mongodb.spark:mongo-spark-connector:10.0.0').\
        getOrCreate()
uri = "mongodb://ricardo-montesanti:lVQUdOpWCGRe5La3@qq-prd-shard-00-00-pri.qxofh.azure.mongodb.net:27017,qq-prd-shard-00-01-pri.qxofh.azure.mongodb.net:27017,qq-prd-shard-00-02-pri.qxofh.azure.mongodb.net:27017,qq-prd-shard-00-03-pri.qxofh.azure.mongodb.net:27017/test?ssl=true&replicaSet=qq-prd-shard-0&authSource=admin&retryWrites=true&readPreference=secondary&readPreferenceTags=nodeType:ANALYTICS&w=majority"
query=spark.readStream.format("mongodb")\
        .option('spark.mongodb.connection.uri', uri)\
    	.option('spark.mongodb.database', 'DailyUpdate') \
    	.option('spark.mongodb.collection', 'ColSimulationAws') \
        .option('spark.mongodb.change.stream.publish.full.document.only','true') \
    	.option("forceDeleteTempCheckpointLocation", "true") \
    	.load()



# COMMAND ----------

remote_table = spark.read.format("jdbc")\
  .option("driver", 'mongo_spark_connector_10_0_2.jar')\
  .option("url", url)\
  .option("dbtable", table)\
  .option("user", user)\
  .option("password", password)\
  .load()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Step 3: Querying the data
# MAGIC 
# MAGIC Now that we created our DataFrame. We can query it. For instance, you can select some particular columns to select and display within Databricks.

# COMMAND ----------

display(remote_table.select("EXAMPLE_COLUMN"))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Step 4: (Optional) Create a view or table
# MAGIC 
# MAGIC If you'd like to be able to use query this data as a table, it is simple to register it as a *view* or a table.

# COMMAND ----------

remote_table.createOrReplaceTempView("YOUR_TEMP_VIEW_NAME")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC We can query this using Spark SQL. For instance, we can perform a simple aggregation. Notice how we can use `%sql` in order to query the view from SQL.

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT EXAMPLE_GROUP, SUM(EXAMPLE_AGG) FROM YOUR_TEMP_VIEW_NAME GROUP BY EXAMPLE_GROUP

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Since this table is registered as a temp view, it will be available only to this notebook. If you'd like other users to be able to query this table, you can also create a table from the DataFrame.

# COMMAND ----------

remote_table.write.format("parquet").saveAsTable("MY_PERMANENT_TABLE_NAME")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC This table will persist across cluster restarts as well as allow various users across different notebooks to query this data. However, this will not connect back to the original database when doing so.