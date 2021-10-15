'''
Description: 
Author: Rocky Hoo
Date: 2021-06-21 13:49:56
LastEditTime: 2021-06-22 13:33:27
LastEditors: Please set LastEditors
CopyRight: 
Copyright (c) 2021 XiaoPeng Studio
'''
#%%
import findspark
findspark.init() 
from itertools import groupby
from matplotlib import patches
import numpy as np
# import pandas as pd
import databricks.koalas as ks
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pyspark.sql import SparkSession
import warnings
# import os
# os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-8-openjdk-amd64'
from seaborn.categorical import countplot
warnings.filterwarnings("ignore")
#%%
test_data=ks.read_csv("file:///usr/work/ML_Learning/tmall_predict/data/data_format1/data_format1/test_format1.csv")
train_data=ks.read_csv("file:///usr/work/ML_Learning/tmall_predict/data/data_format1/data_format1/train_format1.csv")
user_info=ks.read_csv("file:///usr/work/ML_Learning/tmall_predict/data/data_format1/data_format1/user_info_format1.csv")
user_log=ks.read_csv("file:///usr/work/ML_Learning/tmall_predict/data/data_format1/data_format1/user_log_format1.csv")

''' 查看数据基本信息 '''
# %%
train_data.head(10)
train_data.info()
# %%
test_data.head(10)
test_data.info()
#%%
user_info.head(10)
user_info.info()
#%%
user_log.head(10)
print(user_log.info())
print(user_log.isna().sum())
# %%
''' 计算年龄缺失值信息(和nan表示未知) '''
useful_age=user_info["age_range"].count()-user_info[user_info["age_range"]==0]["age_range"].count()
# 计算缺失值的比率
print("user_info年龄未知占比:%f"%((user_info.shape[0]-useful_age)/user_info.shape[0]))
# %%
''' 计算性别缺失信息(2和Null表示未知) '''
useful_gender=user_info["gender"].count()-user_info[user_info["gender"]==2]["gender"].count()
print("user_info性别未知占比:%f"%((user_info.shape[0]-useful_gender)/user_info.shape[0]))
# %%
user_info.describe()
# %%
''' 查看正负样本的分布 '''
ks.options.plotting.backend = 'matplotlib'
label_gp=train_data.groupby("label")["user_id"].count()
print("正负样本的数量:\n",label_gp)
_,axe=plt.subplots(1,2,figsize=(12,6))
# exolode调整的是突出一块的位置
train_data.label.value_counts().plot(kind="pie",autopct="%1.1f%%",shadow=True,explode=[0,0.1],ax=axe[0])
sns.countplot("label",data=train_data.to_pandas(),ax=axe[1],)
# %%
''' 选取top5店进行复购分析 '''
print("选取top5店铺\n店铺\t购买次数")
# value_counts:将店铺按购买次数合并
print(train_data.merchant_id.value_counts().head(5))
# 获取top5店铺列表并转为list;
top5_list=train_data.merchant_id.value_counts().head(5).axes[0].tolist()
train_merchant_id=train_data.copy()
# 不能直接使用train_merchant_id.merchant_id in top5_list
train_merchant_id["top5"]=train_merchant_id.merchant_id.map(lambda x:1 if x in top5_list else 0)
train_merchant_id=train_merchant_id[train_merchant_id.top5==1]
plt.figure(figsize=(8,6))
plt.title("Merchant VS label")
# hue指定做显示的列名
sax=sns.countplot("merchant_id",hue="label",data=train_merchant_id.to_pandas())
# %%
''' 查看店铺的复购分布是否满足正态分布 '''
# train_data.groupby(["merchant_id"])["label"].mean()为计算复购率
merchant_repeat_buy=[rate for rate in train_data.to_pandas().groupby(["merchant_id"])["label"].mean() if rate<=1 and rate>0]
plt.figure(figsize=(8,6))
ax=plt.subplot(1,2,1)
sns.distplot(merchant_repeat_buy,fit=stats.norm)
ax=plt.subplot(1,2,2)
res=stats.probplot(merchant_repeat_buy,plot=plt)
# %%
''' 查看用户的复购分布(是否满足正态分布) '''
user_repeat_buy=[rate for rate in train_data.to_pandas().groupby(["user_id"])["label"].mean() if rate>0 and rate<=1]
plt.figure(figsize=(8,6))
ax=plt.subplot(1,2,1)
sns.distplot(user_repeat_buy,fit=stats.norm)
ax=plt.subplot(1,2,2)
res=stats.probplot(user_repeat_buy,plot=plt)
# %%
''' 分析用户性别和复购的关系 '''
train_data_user_info=train_data.merge(user_info,on=["user_id"],how="left")
plt.figure(figsize=(8,6))
plt.title("Gender VS Label")
ax=sns.countplot("gender",hue='label',data=train_data_user_info.to_pandas())
for p in ax.patches:
    height=p.get_height()
    print(p,height)
# %%
''' 查看用户性别复购的分布 '''
gender_repeat_buy=[rate for rate in train_data_user_info.to_pandas().groupby(["gender"])["label"].mean() if rate>0 and rate<=1]
plt.figure(figsize=(8,6))
ax=plt.subplot(1,2,1)
sns.distplot(gender_repeat_buy,fit=stats.norm)
ax=plt.subplot(1,2,2)
res=stats.probplot(gender_repeat_buy,plot=plt)
# %%
''' 对用户年龄的分析 '''
plt.figure(figsize=(8,8))
plt.title("Age Vs Label")
ax= sns.countplot("age_range",hue="label",data=train_data_user_info.to_pandas())

# %%
''' 查看用户年龄复购的分布 '''
age_range_repeat_buy=[rate for rate in train_data_user_info.to_pandas().groupby(["age_range"])["label"].mean() if rate>0 and rate<=1]
plt.figure(figsize=(8,4))
ax=plt.subplot(1,2,1)
sns.distplot(age_range_repeat_buy,fit=stats.norm)
ax=plt.subplot(1,2,2)
res=stats.probplot(age_range_repeat_buy,plot=plt)
# %%
