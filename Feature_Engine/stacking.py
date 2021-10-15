'''
Author: Rocky Hoo
Date: 2021-07-06 05:16:12
LastEditTime: 2021-07-15 00:00:53
LastEditors: Please set LastEditors
Description: stack分类模型 制作stacking特征
FilePath: /tmall_predict/Feature_Engine/Spark/stacking.py
'''

# (train_x,train_y)训练集的数据和标签
# test_x测试集数据
# folds:k折的折数
# kf k折训练模型
# clf_name:分类器的名字
#
# %%
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from xgboost import XGBClassifier
import xgboost
from xgboost.training import train
import lightgbm
from memory_reduce import reduce_mem_usage
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
# %%
'''
@description:对模型进行stacking产生meta数据
@param {*} clf:分类器
@param {*} clf_name:分类器名字
@param {*} train_x
@param {*} train_y
@param {*} test_x
@param {*} kf
@param {*} folds:k-fold对应的fold
@param {*} label_split
@return {*}返回stacking后产生的meta数据
'''


def stacking_clf(clf, clf_name, train_x, train_y, test_x, kf, folds=5, label_split=None):
    # 新建train、test 不覆盖原本数据
    train = np.zeros((train_x.shape[0], 1))
    test = np.zeros((test_x.shape[0], 1))
    test_pre = np.empty((folds, test_x.shape[0], 1))
    cv_scores = []
    # 只用对train_x进行划分，train_y(需要预测的标签)的index与train_x一一对应
    for i, (train_idx, test_idx) in enumerate(kf.split(train_x, label_split)):
        part_train_x = train_x[train_idx]
        part_train_y = train_y[train_idx]
        part_test_x = train_x[test_idx]
        part_test_y = train_y[test_idx]

        if clf_name in ["rf", "ada", "gb", "et", "lr", "knn", "gnb"]:
            train_df = part_test_x["target"]
            clf.fit(part_train_x, part_test_y)
            pre = clf.predict_proba(part_test_x)
            # why reshape?
            train[test_idx] = pre[:, 0].reshape(-1, 1)
            test_pre[i, :] = clf.predict_proba(test_x)[:, 0].reshape(-1, 1)

            cv_scores.append(log_loss(part_test_y, pre[:, 0].reshape(-1, 1)))

        elif clf_name == "xgb":
            # missing?
            train_matrix = clf.DMatrix(
                part_train_x, label=part_train_y, missing=-1)
            test_matrix = clf.DMatrix(
                part_test_x, label=part_test_y, missing=-1)
            z = clf.DMatrix(test_x, label=None, missing=-1)
            # 需要了解每个参数的意义
            params = {
                "booster": "gbtree",
                # what's this?目标是多分类?
                "objective": "multi:softprob",
                "eval_matrix": "mlogloss",
                # what's this?
                "gamma": 1,
                "min_child_weight": 1.5,
                "max_depth": 5,
                # ?
                "lambda": 10,
                #
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "colsample_bylevel": 0.7,
                "eta": 0.03,
                "tree_method": "exact",
                "seed": 2017,
                "num_class": 2
            }

            num_round = 10000
            early_stopping_rounds = 100
            # ?
            watchlist = [(train_matrix, "train"), (test_matrix, "eval")]
            # why if?
            if test_matrix:
                model = clf.train(params, train_matrix, num_boost_round=num_round,
                                  evals=watchlist, early_stopping_rounds=early_stopping_rounds)
                #  why not pred_proba?
                pre = model.predict(
                    test_matrix, ntree_limit=model.best_ntree_limit)
                train[test_idx] = pre[:, 0].reshape(-1, 1)
                test_pre[i, :] = model.predict(z, ntree_limit=model.best_ntree_limit)[
                    :, 0].reshape(-1, 1)
                cv_scores.append(
                    log_loss(part_test_y, pre[:, 0].reshape(-1, 1)))
        elif clf_name == "lgb":
            train_matrix = clf.Dataset(part_train_x, label=part_train_y)
            test_matrix = clf.Dataset(part_test_x, label=part_test_y)
            params = {
                "booster": "gbdt",
                # what's this?目标是多分类?
                "objective": "multiclass",
                "eval_matrix": "multi_logloss",
                # what's this?
                # "gamma":1,
                "min_child_weight": 1.5,
                "num_leaves": 2**5,
                # "max_depth":5,
                # ?
                "lambda_l2": 10,
                #
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "colsample_bylevel": 0.7,
                "learning_rate": 0.3,
                "tree_method": "exact",
                "seed": 2017,
                "num_class": 2,
                "silent": True
            }
            num_round = 10000
            early_stopping_rounds = 100
            if test_matrix:
                model = clf.train(params, train_matrix, num_round, valid_sets=test_matrix,
                                  early_stopping_rounds=early_stopping_rounds)
                # why not matrix
                pre = model.predict(
                    part_test_x, num_iteration=model.best_iteration)
                train[test_idx] = pre[:, 0].reshape(-1, 1)
                # test_pre[i, :] = model.predict(test_x, num_iteration=model.best_iteration)[
                #     :, 0].reshape(-1, 1)
                pre_temp = model.predict(test_x, num_iteration=model.best_iteration)
                test_pre[i, :]=pre_temp[:,0].reshape(-1,1)
                cv_scores.append(
                    log_loss(part_test_y, pre[:, 0].reshape(-1, 1)))
        else:
            raise IOError("Please add new clf.")
        print("%s now score is" % clf_name, cv_scores)
    test[:] = test_pre.mean(axis=0)
    print("%s_score_list is" % clf_name, cv_scores)
    print("%s_score_mean is" % clf_name, np.mean(cv_scores))
    return train.reshape(-1, 1), test.reshape(-1, 1)


'''
@description: 随机森林基分类器
@param {*} train_x
@param {*} train_y
@param {*} test_x
@param {*} kf
@param {*} label_split
@return {*}
'''


def rf_clf(train_x, train_y, test_x, kf, label_split=None):
    # grid_search?
    random_forest = RandomForestClassifier(
        n_estimators=1200, max_depth=20, n_jobs=-1, random_state=2017, max_features="auto", verbose=1)
    rf_train, rf_test = stacking_clf(
        random_forest, "rf", train_x, train_y, test_x, kf)
    return rf_train, rf_test


'''
@description: 
@param {*} train_x
@param {*} train_y
@param {*} test_x
@param {*} kf
@param {*} label_split
@return {*}
'''
def ada_clf(train_x, train_y, test_x, kf, label_split=None):
    # grid_search?
    adaboost = AdaBoostClassifier(
        # 重复定义参数了?
        n_estimators=50, random_state=2017, learning_rate=0.01)
    ada_train, ada_test = stacking_clf(
        adaboost, "ada", train_x, train_y, test_x, kf)
    return ada_train, ada_test


def gb_clf(train_x, train_y, test_x, kf, label_split=None):
    # grid_search?
    gbdt = GradientBoostingClassifier(
        # 重复定义参数了?
        n_estimators=100, random_state=2017, learning_rate=0.04, subsample=0.8, max_depth=5, verbose=1)  # verbose确定打印出来的详细信息
    gbdt_train, gbdt_test = stacking_clf(
        gbdt, "gb", train_x, train_y, test_x, kf)
    return gbdt_train, gbdt_test


def et_clf(train_x, train_y, test_x, kf, label_split=None):
    # grid_search?
    extratree = ExtraTreesClassifier(
        # 重复定义参数了?
        n_estimators=1200, random_state=2017, max_depth=35, verbose=1, max_features="auto", n_jobs=-1)  # verbose确定打印出来的详细信息
    extratree_train, extratree_test = stacking_clf(
        extratree, "et", train_x, train_y, test_x, kf)
    return extratree_train, extratree_test


def xgb_clf(train_x, train_y, test_x, kf, label_split=None):
    # grid_search?
    xgb_train, xgb_test = stacking_clf(
        xgboost, "xgb", train_x, train_y, test_x, kf)
    return xgb_train, xgb_test


def lgb_clf(train_x, train_y, test_x, kf, label_split=None):
    # grid_search?
    lgb_train, lgb_test = stacking_clf(
        lightgbm, "lgb", train_x, train_y, test_x, kf)
    return lgb_train, lgb_test


def gnb_clf(train_x, train_y, test_x, kf, label_split=None):
    # grid_search?
    gnb = GaussianNB()
    gnb_train, gnb_test = stacking_clf(
        gnb, "gnb", train_x, train_y, test_x, kf)
    return gnb_train, gnb_test


def lr_clf(train_x, train_y, test_x, kf, label_split=None):
    # grid_search?
    lr = LogisticRegression(n_jobs=-1, random_state=2017, C=0.1, max_iter=200)
    lr_train, lr_test = stacking_clf(
        lr, "lr", train_x, train_y, test_x, kf)
    return lr_train, lr_test


def knn_clf(train_x, train_y, test_x, kf, label_split=None):
    # grid_search?
    knn = KNeighborsClassifier(n_neighbors=200, n_jobs=-1)
    knn_train, knn_test = stacking_clf(
        knn, "knn", train_x, train_y, test_x, kf)
    return knn_train, knn_test


'''
@description:处理函数中inf和nan值 
@param {*} data
@return {*}
'''


def get_matrix(data):
    where_are_nan = np.isnan(data)
    where_are_inf = np.isinf(data)
    data[where_are_inf] = 0
    data[where_are_nan] = 0
    return data


# %%
all_data_test = reduce_mem_usage(pd.read_csv(
    "../data/all_data_test_w2v.csv"))
#%%
# all_data_test = all_data_test.drop("sellers", axis=1)
without_columns = ["label", "prob", "seller_path", "cat_path",
                   "brand_path", "action_type_path", "item_path", "time_stamp_path"]
feature_columns = [
    col for col in all_data_test.columns if col not in without_columns]
train_x = all_data_test[~all_data_test["label"].isna()][feature_columns].values
train_y = all_data_test[~all_data_test["label"].isna()]["label"].values
test_x = all_data_test[all_data_test["label"].isna()][feature_columns].values
# %%
# 对数组结构的数据初始化为numpy对应数据类型需要type_
train_x = np.float_(get_matrix(np.float_(train_x)))
train_y = np.int_(train_y)
test_x = np.float_(test_x)
#%%
all_data_test["label"].head(5)
# %%
seed = 1
folds = 5
kf = KFold(n_splits=folds, shuffle=True, random_state=0)
# %%
# 选择lgb\xgb作为基模型
clf_list = [lgb_clf, xgb_clf]
clf_list_col = ["lgb", "xgb"]
# %%
# 获取stacking特征
clf_list = clf_list
column_list = []
train_data_list = []
test_data_list = []
test_data_list = []
for clf in clf_list:
    train_data, test_data = clf(train_x, train_y, test_x, kf, label_split=None)
    train_data_list.append(train_data)
    test_data_list.append(test_data)

train_stacking = np.concatenate(train_data_list, axis=1)
test_stacking = np.concatenate(test_data_list, axis=1)

# %%
all_data_test.head(1)
# %%
# 原始特征和stacking特征合并
train=pd.DataFrame(np.concatenate([train_x,train_stacking],axis=1))
test=pd.DataFrame(np.concatenate([test_x,test_stacking],axis=1))
#%%
# 特征重命名
df_train_all=pd.DataFrame(train)
df_train_all.columns=feature_columns+clf_list_col
df_test_all=pd.DataFrame(test)
df_test_all.columns=feature_columns+clf_list_col
#%%
# 获取数据ID及特征标签
df_train_all["user_id"]=all_data_test[~all_data_test["label"].isna()]["user_id"]
df_test_all["user_id"]=all_data_test[all_data_test["label"].isna()]["user_id"]
df_train_all["label"]=all_data_test[~all_data_test["label"].isna()]["label"]
#%%
df_train_all.to_csv("./data/train_all_embedding.csv",header=True,index=False)
df_test_all.to_csv("./data/test_all_embedding.csv",header=True,index=False)
#%%
import pandas as pd 
temp=pd.read_csv("../data/test_all_embedding.csv")
# %%
temp.head(10)
# %%
