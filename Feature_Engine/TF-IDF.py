'''
Description: 使用TF-IDF方法提取主要商家
Author: Rocky Hoo
Date: 2021-07-13 16:49:04
LastEditTime: 2021-07-13 17:03:19
LastEditors: Please set LastEditors
CopyRight: 
Copyright (c) 2021 XiaoPeng Studio
'''
from memory_reduce import reduce_mem_usage
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,ENGLISH_STOP_WORDS
from scipy import sparse
import pandas as pd

all_data_test=reduce_mem_usage(pd.read_csv("./data/all_data_test.csv"))

tfidfVec=TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,ngram_range=(1,1),max_features=100)

columns_list=['seller_path']
for i,col in enumerate(columns_list):
    tfidfVec.fit(all_data_test[col])
    data_=tfidfVec.transform(all_data_test[col])
    if i==0:
        data_cat=data_
    else:
        data_cat=sparse.hstack((data_cat,data_))
df_tfidf=pd.DataFrame(data_cat.toarray())
df_tfidf.columns=["tfidf_"+str(i) for i in df_tfidf.columns]
all_data_test=pd.concat([all_data_test,df_tfidf],axis=1)
all_data_test.to_csv("./data/all_data_test_idftf.csv")
    
