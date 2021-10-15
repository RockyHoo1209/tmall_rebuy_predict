'''
Description: 特征工程文件
Author: Rocky Hoo
Date: 2021-06-24 12:48:25
LastEditTime: 2021-07-13 15:46:27
LastEditors: Please set LastEditors
CopyRight: 
Copyright (c) 2021 XiaoPeng Studio
'''
# %%
import sys   
from memory_reduce import reduce_mem_usage
from os import error
import pandas as pd
import numpy as np
import warnings
from typing import Counter
import copy
from seaborn.categorical import countplot
import gc

# %%
test_data = reduce_mem_usage(pd.read_csv(
    "./data/data_format1/data_format1/test_format1.csv"))
train_data = reduce_mem_usage(pd.read_csv(
    "./data/data_format1/data_format1/train_format1.csv"))
user_info = reduce_mem_usage(pd.read_csv(
    "./data/data_format1/data_format1/user_info_format1.csv"))
user_log = reduce_mem_usage(pd.read_csv(
    "./data/data_format1/data_format1/user_log_format1.csv"))
# %%
all_data = train_data.append(test_data)
all_data = all_data.merge(user_info, on=["user_id"], how="left")
# %%
del train_data, test_data, user_info
gc.collect()
# %%
# 时间序列?
user_log = user_log.sort_values(["user_id", "time_stamp"])
# %%
user_log.head()
# %%
def list_join_func(x): return " ".join([str(i) for i in x])


agg_dict = {
    "item_id": list_join_func,
    "cat_id": list_join_func,
    "seller_id": list_join_func,
    "brand_id": list_join_func,
    "time_stamp": list_join_func,
    "action_type": list_join_func
}

rename_dict = {
    "item_id": "item_path",
    "cat_id": "cat_path",
    "seller_id": "seller_path",
    "brand_id": "brand_path",
    "time_stamp": "time_stamp_path",
    "action_type": "action_type_path"
}
# 根据用户分组聚合进行行为分析
# agg是指对每个分组的元素进行聚合操作


def merge_list(df_ID, join_columns, df_data, agg_dict, rename_dict):
    df_data = df_data.groupby(join_columns).agg(
        agg_dict).reset_index().rename(columns=rename_dict)
    df_ID = df_ID.merge(df_data, on=join_columns, how="left")
    return df_ID


all_data = merge_list(all_data, "user_id", user_log, agg_dict, rename_dict)
# %%
del user_log
gc.collect()
# %%
all_data.head(10)
# %%
# 定义统计函数


def cnt_(x):
    try:
        return len(x.split(" "))
    except:
        return -1


def nunique_(x):
    try:
        return len(set(x.split(" ")))
    except:
        return -1


def max_(x):
    try:
        return np.max([float(i) for i in x.split(" ")])
    except:
        return -1


def min_(x):
    try:
        return np.min([float(i) for i in x.split(" ")])
    except:
        return -1


def std_(x):
    try:
        return np.std([float(i) for i in x.split(" ")])
    except:
        return -1

# 取出频率出现第n的特征项


def most_n(x, n):
    try:
        return Counter(x.split(" ")).most_common(n)[n-1][0]
    except:
        return -1

# 取出频率出现第n的特征对应的出现次数


def most_n_cnt(x, n):
    try:
        return Counter(x.split(" ")).most_common(n)[n-1][1]
    except Exception as e:
        print(repr(e))
        return -1
# %%
# 调用定义的统计函数
def user_cnt(df_data, single_col, name):
    df_data[name] = df_data[single_col].apply(cnt_)
    return df_data


def user_nunique(df_data, single_col, name):
    df_data[name] = df_data[single_col].apply(nunique_)
    return df_data


def user_max(df_data, single_col, name):
    df_data[name] = df_data[single_col].apply(max_)
    return df_data


def user_min(df_data, single_col, name):
    df_data[name] = df_data[single_col].apply(min_)
    return df_data


def user_std(df_data, single_col, name):
    df_data[name] = df_data[single_col].apply(std_)
    return df_data


def user_most_n(df_data, single_col, name, n=1):
    def func(x): return most_n(x, n)
    df_data[name] = df_data[single_col].apply(func)
    return df_data


def user_most_n_cnt(df_data, single_col, name, n=1):
    def func(x): return most_n_cnt(x, n)
    df_data[name] = df_data[single_col].apply(func)
    return df_data


# %%
all_data_test = user_cnt(all_data, "seller_path", "user_cnt")
all_data_test = user_nunique(all_data_test, "seller_path", "seller_nunique")
# 不同种类个数
all_data_test = user_nunique(all_data_test, "cat_path", "cat_nunique")
# 不同品牌个数
all_data_test = user_nunique(all_data_test, "brand_path", "brand_nunique")
# 不同商品个数
all_data_test = user_nunique(all_data_test, "item_path", "item_nunique")
# 活跃天数(?)
all_data_test = user_nunique(
    all_data_test, "time_stamp_path", "time_stamp_nunique")
# 不同用户行为种数
all_data_test = user_nunique(
    all_data_test, "action_type_path", "action_type_nunique")
# 最晚时间
all_data_test = user_max(all_data_test, "time_stamp_path", "time_stamp_max")
# 最早时间
all_data_test = user_min(all_data_test, "time_stamp_path", "time_stamp_min")
# 活跃天数方差
all_data_test = user_std(all_data_test, "time_stamp_path", "time_stamp_std")
# 最早和最晚相差天数
all_data_test["time_stamp_range"] = all_data_test["time_stamp_max"] - \
    all_data_test["time_stamp_min"]
# 用户最喜欢的店铺
all_data_test = user_most_n(all_data_test, "seller_path", "seller_most_1")
# 用户最喜欢的种类
all_data_test = user_most_n(all_data_test, "cat_path", "cat_most_1")
# 用户最喜欢的品牌
all_data_test = user_most_n(all_data_test, "brand_path", "brand_most_1")
# 用户最喜欢的商品
all_data_test = user_most_n(all_data_test, "item_path", "item_most_1")
# 最常见行为
all_data_test = user_most_n(
    all_data_test, "action_type_path", "action_type_most_1")
# 用户最喜欢的店铺 行为次数
all_data_test = user_most_n_cnt(
    all_data_test, "seller_path", "seller_most_cnt_1")
# 用户最喜欢的种类 行为次数
all_data_test = user_most_n_cnt(all_data_test, "cat_path", "cat_most_cnt_1")
# 用户最喜欢的品牌 行为次数
all_data_test = user_most_n_cnt(
    all_data_test, "brand_path", "brand_most_cnt_1")
# 用户最喜欢的商品 行为次数
all_data_test = user_most_n_cnt(all_data_test, "item_path", "item_most_cnt_1")
# 用户最常见行为次数
all_data_test = user_most_n_cnt(
    all_data, "action_type_path", "action_type_most_cnt_1")
# %%
all_data_test["action_type_path"].head(1)
# %%
# 统计某一事件对应行为进行统计，返回进行了相应行为的特征的个数
def col_cnt_(df_data, columns_list, action_type=None):
    try:
        data_dict = {"action_type_path":[]}
        cols_list = copy.deepcopy(columns_list)
        if action_type is not None:
            cols_list.append("action_type_path")
        for col in cols_list:
            data_dict[col] = df_data[col].split(" ")

        # 行为与商家一一对应，详见user  _info
        action_path_len = len(data_dict["action_type_path"])
        ret_data = []
        for action_idx in range(action_path_len):
            data_txt = ""
            for col in columns_list:
                if data_dict["action_type_path"][action_idx] == action_type:
                    data_txt += "_"+data_dict[col][action_idx]
            ret_data.append(data_txt)
        # 返回总的长度和去重后的长度
        return len(ret_data)
    except Exception as e:
        # print(e)
        print(repr(e))
        return -1

def col_nuique_(df_data, columns_list, action_type=None):
    try:
        data_dict = {"action_type_path":[]}
        cols_list = copy.deepcopy(columns_list)
        if action_type is not None:
            cols_list.append("action_type_path")
        for col in cols_list:
            data_dict[col] = df_data[col].split(" ")

        # 行为与商家一一对应，详见user_info
        action_path_len = len(data_dict["action_type_path"])
        ret_data = []
        for action_idx in range(action_path_len):
            data_txt = ""
            for col in columns_list:
                if data_dict["action_type_path"][action_idx] == action_type:
                    data_txt += "_"+data_dict[col][action_idx]
            ret_data.append(data_txt)
        # 返回总的长度和去重后的长度
        return len(set(ret_data))
    except Exception as e:
        print(repr(e))
        return -1


def user_col_cnt(df_data,columns_list,action_type,name):
    df_data[name]=df_data.apply(lambda x:col_cnt_(x,columns_list,action_type),axis=1) 
    return df_data

def user_col_unique(df_data,columns_list,action_type,name):
    df_data[name]=df_data.apply(lambda x:col_nuique_(x,columns_list,action_type),axis=1) 
    return df_data

#%%
click_event="0"
add_car="1"
buy="2"
add_fav="3"
all_data_test=user_col_cnt(all_data_test,["seller_path"],click_event,"user_cnt_0")
all_data_test=user_col_cnt(all_data_test,["seller_path"],add_car,"user_cnt_1")
# 特征组合
all_data_test=user_col_cnt(all_data_test,["seller_path","item_path"],buy,"user_cnt_1") 
all_data_test=user_col_unique(all_data_test,["seller_path","item_path"],add_fav,"user_cnt_1")
# %%
all_data_test.columns
# %%
all_data_test.to_csv("./data/all_data_test2.csv")
# %%
print("ok!")