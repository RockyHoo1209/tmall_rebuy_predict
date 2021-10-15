'''
Author: Rocky Hoo
Date: 2021-07-04 05:17:25
LastEditTime: 2021-07-07 14:30:11
LastEditors: Please set LastEditors
Description: In User Settings Edit  
FilePath: /tmall_predict/Feature_Engine/Spark/features_embeeding.py
'''
#%%
import findspark
findspark.init()

from pyspark.ml.feature import  Word2Vec
from memory_reduce import reduce_mem_usage
import pandas as pd
from pyspark.sql.session import SparkSession
import numpy as np
#%%
spark = SparkSession.builder \
    .master('local[*]') \
    .config("spark.driver.memory", "8g") \
    .appName('my-cool-app') \
    .getOrCreate()
all_data_test=reduce_mem_usage(pd.read_csv("/usr/work/ML_Learning/tmall_predict/data/all_data_test_idftf.csv"))
all_data_test["seller_path2"]=all_data_test["seller_path"].apply(lambda x:x.split(" "))
spark_dff = spark.createDataFrame(all_data_test)

word2Vec = Word2Vec(inputCol="seller_path2",windowSize=5,minCount=5,numPartitions=4,outputCol="w2v_features")
model=word2Vec.fit(spark_dff)
wv=model.transform(spark_dff).toPandas()
#%%
# 不词典转成{word:vec}
vocab=model.getVectors().rdd.collectAsMap()
# %%
# 对每个用户生成一个对应卖家特征
def mean_w2v_(x,size=100):
    try:
        i=0
        for word in x.split(" "):
            if word in vocab.keys():
                i+=1
                if i==1:
                    vec=np.zeros(size)
                vec+=vocab[word].toArray()
        return vec/i
    except Exception as e:
        print(repr(e))
# %%
def get_mean_w2v(df_data,columns,size):
    data_array=[]
    for _,row in df_data.iterrows():
        w2v=mean_w2v_(row[columns],size)
        data_array.append(w2v)
    return pd.DataFrame(data_array)
# %%
df_embedding=get_mean_w2v(all_data_test,"seller_path",100)
# %%
df_embedding.columns=["embedding_"+str(i) for i in df_embedding.columns]
# %%
df_data_test=pd.concat([all_data_test,df_embedding],axis=1)
#%%
df_data_test = df_data_test.loc[:, ~df_data_test.columns.str.contains('^Unnamed')]
# %%
df_data_test.to_csv("/usr/work/ML_Learning/tmall_predict/data/all_data_test_w2v.csv")
# %%
