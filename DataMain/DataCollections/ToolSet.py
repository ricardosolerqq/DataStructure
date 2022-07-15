##########################################
#GENERAL IMPORTS
##########################################

import json
import pandas as pd
from datetime import datetime
from datetime import timedelta
import concurrent.futures
import sys
import os.path
from pymongo import *
import bson as b
import random as re
import json
import string as s
import pandas as pd
import os.path,inspect, re 
import random as r
from types import SimpleNamespace
import sys
import pathlib
import time
import os
import zipfile
import concurrent.futures
import databricks.koalas as ks
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pymongo import *


##########################################
#SPARK CONF
##########################################

from pyspark.sql.types import *
from pyspark.sql import types as T 
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from datetime import datetime, timedelta 
import pyspark
from pyspark.sql import SQLContext  

spark.conf.set('spark.Builder.enableHiveSupport',True)
spark.conf.set("spark.sql.session.timeZone", "America/Sao_Paulo")
spark.conf.set('spark.sql.execution.arrow.pyspark.enabled' , False)
spark.conf.set('spark.sql.caseSensitive', True)
spark.conf.set("spark.sql.shuffle.partitions", "2") 


from pyspark import SparkContext, SparkConf
 
  

class SparkSet:
    def __init__(self,
                 CREDITOR=None,
                 DATABASE=None,
                 COLLECTION=None,
                 DIR=None):
        self.CREDITOR = CREDITOR
        self.DATABASE = DATABASE
        self.COLLECTION = COLLECTION
        self.SCHEMA = self.getSchema()
        self.URI = "mongodb://ricardo-montesanti:lVQUdOpWCGRe5La3@federateddatabase-qxofh.a.query.mongodb.net/?ssl=true&authSource=admin"
        self.INTER_LIST = []
        self.DIR = DIR


    def getSchema(self, sampleSize=50000, mode="local", col="col_person"):
        
        DATA = ""
      
        DATA_PATH = "/dbfs/SCHEMAS/DATA_SCHEMA.json"

        if mode == "local":
            with open(DATA_PATH) as f:
                PersonSchema = T.StructType.fromJson(json.load(f))

        else:
            DATA = spark.read\
            .format( "com.mongodb.spark.sql.DefaultSource")\
            .option('spark.mongodb.input.sampleSize', sampleSize)\
            .option("database", "qq")\
            .option("spark.mongodb.input.collection", "col_person")\
            .option("badRecordsPath", "/tmp/badRecordsPath")\
            .load().schema



            with open("PersonSchema.json", "w") as f:
                json.dump(DATA.jsonValue(), f)

            with open("PersonSchema.json") as f:
                PersonSchema = T.StructType.fromJson(json.load(f))
            dbutils.widgets.dropdown("SCHEMA_STATUS", "LOADED",
                                 ["LOADED", "NOT LOADED"])
        return PersonSchema

                
st = SparkSet()
SCHEMA = st.SCHEMA


import os
class Path:
    def __init__(self, path):
      self.path = path
      self.mount_path_full = "/dbfs/mnt/" +  str(path)
      self.mount_path = os.path.join("/mnt/", str(path))
      self.dbfs_path = os.path.join("/dbfs/", str(path))
      self.dbfs_dot_path = os.path.join("/dbfs:", str(path))


 
def getInterByIdSingle(self, id=None):
    col = st.findByMongo(COLLECTION='col_interaction')
    col_filter = [{'$match': {'_id': '%s' % id}}]
    result = col.aggregate(col_filter)
    result = list(result)
    #UTILIZAR ALGUMA FORMA DE PERSISTÊNCIA EM MEMÓRIA PARA ACUMULAR OS RESULTADOS
    self.INTER_LIST.append(result)
    size = len(self.INTER_LIST)

    print(size)

def getInterByIdFull(self):

    #THREAD POOL EXECUTOR
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        #ITERABLE
        for item in INTER_ID_LIST:
            #FUNCTION
            futures.append(executor.submit(getInterByIdSingle(), id=item))


##########################################
#PYMONGO
##########################################

from pymongo import * 

dbName = 'qq'
colName = 'col_interaction' 
client = MongoClient(uri_string) 
database = client.get_database(dbName) 
col = database.get_collection(colName)


def colCount():
    colCount =  col.find({},{"_id"}).count()
    return colCount

def colFind(limit=""):
    if limit == "":
        colList = [ x for x in col.find()]
    else:
        colList = [ x for x in col.find().limit(limit)]
    return(colList)

def colRawFind(limit=""):
    if limit == "":
        colList = [ x for x in col.find_raw_batches()]
    else:
        colList = [ b.decode_all(x) for x in col.find_raw_batches().limit(limit)]
    return(colList)

def colFindID():
    idList = [str(x).split("'",4)[3] for x in col.find({},{"_id":1}).limit(5)]
    return(idList)


##########################################
#WIDGETS
##########################################

class Wd:
    def __init__(self, mode=None, name=None,default_value=None, value_list=None):
        self.name = name
        self.default_value = default_value
        self.value_list = value_list
        self.mode = mode
    
    def setWd(self):
        if self.mode == "text":
            dbutils.widgets.text(self.name,
                                 self.default_value)

        elif self.mode == "dropdown":
            dbutils.widgets.dropdown(self.name,
                                 self.default_value,
                                 self.value_list)
            
        elif self.mode == "multiselect":
            dbutils.widgets.multiselect(self.name,
                                 self.default_value,
                                 self.value_list)
    def getWd(self):
        wd = dbutils.widgets.get(self.name)
        return wd
    
    def rmWd(self):
        dbutils.widgets.remove(self.name)
        
      
    def rsWd(self):
        dbutils.widgets.removeAll()
