'''
Author: Rocky Hoo
Date: 2021-07-02 08:28:37
LastEditTime: 2021-07-13 16:42:24
LastEditors: Please set LastEditors
Description: 使用TF-IDF构造特征
FilePath: /tmall_predict/Feature_Engine/TF-IDF.py
'''
#%%
# import findspark
# findspark.init()
from pyspark.sql.session import SparkSession
from memory_reduce import reduce_mem_usage
from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.ml.feature import CountVectorizer, HashingTF, IDF, IDFModel, Tokenizer
import pandas as pd
from pyspark.sql import SQLContext
import gc
#%%
spark = SparkSession.builder \
    .master('local[*]') \
    .config("spark.driver.memory", "8g") \
    .appName('my-cool-app') \
    .getOrCreate()
all_data_test=reduce_mem_usage(pd.read_csv("./data/all_data_test.csv"))
# 内存不足 只取其中了2000条
spark_dff = spark.createDataFrame(all_data_test[:2000])
tokenizer = Tokenizer(inputCol="seller_path",outputCol="sellers")
# spark_dff = sqlContext.createDataFrame(all_data_test)
del all_data_test
all_data_test = tokenizer.transform(spark_dff)
del spark_dff
gc.collect()
#%%
# 提取100个特征
hashingTF=HashingTF(inputCol="sellers",outputCol="rawFeatures",numFeatures=100)
featurizedData=hashingTF.transform(all_data_test)
#%%
featurizedData.select("sellers","rawFeatures").show(truncate=False)
# %%
idf=IDF(inputCol="rawFeatures",outputCol="features")
idfModel=idf.fit(featurizedData)
rescaledData=idfModel.transform(featurizedData)
# %%
df_tfidf=pd.DataFrame(rescaledData.toPandas()["features"])
#%%
df_tfidf=df_tfidf["features"].apply(pd.Series,index=[i for i in range(100)])
# %%
df_tfidf.columns=['tf_idf_'+str(i) for i in df_tfidf.columns]
# %%
all_data_test=pd.concat([all_data_test.toPandas(),df_tfidf],axis=1)
# %%
all_data_test.head(5)
#%%
all_data_test.to_csv("./data/all_data_test_idftf.csv")
# %%
