'''
Author: Rocky Hoo
Date: 2021-07-03 16:13:57
LastEditTime: 2021-07-13 17:35:13
LastEditors: Please set LastEditors
Description: 嵌入特征
FilePath: /tmall_predict/Feature_Engine/features_embbeding.py
'''
#%%
from memory_reduce import reduce_mem_usage
import gensim
import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd
from pandas.core.algorithms import mode
#%%
all_data_test=reduce_mem_usage(pd.read_csv("./data/all_data_test_idftf.csv"))

model=gensim.models.Word2Vec(
    all_data_test["seller_path"].apply(lambda x:x.split(" ")),
    # max_vocab_size=100,
    window=5,
    min_count=5,
    workers=4
)

# 为每个客户构造一个商家特征
def mean_w2v_(x,model,size=100):
    try:
        i=0
        for word in x.split(' '):
            if word in model.wv.key_to_index:
                i+=1
                if i==1:
                    vec=np.zeros(size)
                # print("wv_word:",model.wv[word])
                vec+=model.wv[word]
        # print("vec/i:",vec/i)
        return vec/i
    except Exception as e:
        print(repr(e))
        return np.zeros(size)    

def get_mean_w2v(df_data,column,model,size):
    data_array=[]
    for index,row in df_data.iterrows():
        w2v=mean_w2v_(row[column],model,size)
        # print("w2v:",w2v)
        data_array.append(w2v)
    return pd.DataFrame(data_array)
#%%
df_embeeding=get_mean_w2v(all_data_test,"seller_path",model,100)
#%%
df_embeeding.columns=['embeeding_'+str(i) for i in df_embeeding.columns]
# %%
all_data_test=pd.concat([all_data_test,df_embeeding],axis=1)
# %%
all_data_test.to_csv("./data/all_data_test_w2v.csv")
# %%
