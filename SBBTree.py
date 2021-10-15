#%%
# 封装Stacking\Bootstrap\Bagging合成的一个模型
from numpy.lib.function_base import cov
from numpy.lib.mixins import _numeric_methods
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn import feature_selection
from sklearn.metrics import f1_score, scorer
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold,cross_val_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
#%%
class SBBTree():
    def __init__(self,params,stacking_num,bagging_num,bagging_test_size,num_boost_round,early_stopping_rounds):
        self.params=params
        self.stacking_num=stacking_num
        self.bagging_num=bagging_num
        self.bagging_test_size=bagging_test_size
        # 设定训练的lgb迭代次数
        self.num_boost_round=num_boost_round
        self.early_stopping_rounds=early_stopping_rounds
        self.model=lgb
        self.stacking_models=[]
        self.bagging_models=[]


    def fit(self,X,Y):
        if self.stacking_num>1:
            # 第一列存放原始的训练集数据,第二列存放元模型提取特征后的元数据
            layer_train=np.zeros((X.shape[0],2))
            self.SK=StratifiedKFold(n_splits=self.stacking_num,shuffle=True,random_state=1)
            for k,(train_index,test_index) in enumerate(self.SK.split(X,Y)):
                X_train=X[train_index]
                Y_train=Y[train_index]
                X_test=X[test_index]
                Y_test=Y[test_index]

                # lgb_train=lgb.Dataset(X_train,Y_train)
                # lgb_eval=lgb.Dataset(X_test,Y_test,reference=lgb_train)
                
                # gbm=lgb.train(self.params,lgb_train,num_boost_round=self.num_boost_round,valid_sets=lgb_eval,early_stopping_rounds=self.early_stopping_rounds)
                gbm=MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(50, 50), random_state=1).fit(X_train,Y_train)
                self.stacking_models.append(gbm)

                pred_y=gbm.predict_proba(X_test)
                # 生成元数据作为下一层训练的训练集
                layer_train[test_index,1]=pred_y[:,1]
            
            # 将原始数据和元数据放到一起(reshape转换成一列)
            X=np.hstack((X,layer_train[:,1].reshape(-1,1)))
        else:
            pass
        # 通过boostrap采样数据进行bagging
        for bn in range(self.bagging_num):
            X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=self.bagging_test_size,random_state=bn)

            lgb_train=lgb.Dataset(X_train,Y_train)
            lgb_eval=lgb.Dataset(X_test,Y_test,reference=lgb_train)
            gbm=lgb.train(self.params,lgb_train,num_boost_round=self.num_boost_round,valid_sets=lgb_eval,early_stopping_rounds=self.early_stopping_rounds)

            self.bagging_models.append(gbm)

    # 对于预测数据先做stacking的均值处理，然后bagging模型作为第二层的模型预测
    def predict(self,X_pred):
        if self.stacking_num>1:
            test_pred=np.zeros((X_pred.shape[0],self.stacking_num))
            for sn,gbm in enumerate(self.stacking_models):
                pred=gbm.predict(X_pred)
                test_pred[:,sn]=pred
            X_pred=np.hstack((X_pred,test_pred.mean(axis=1).reshape(-1,1)))
        else:
            pass
        for bn,gbm in enumerate(self.bagging_models):
            pred=gbm.predict(X_pred,num_iteration=gbm.best_iteration)
            if bn==0:
                pred_out=pred
            else:
                pred_out+=pred
        return pred_out/self.bagging_num
# %%
train_target=pd.read_csv("./data/train_all_embedding.csv")
features_cols=[col for col in train_target.columns if col not in ["label","user_id"]]
train=train_target[features_cols].values
target=train_target["label"].values
test=pd.read_csv("./data/test_all_embedding.csv")[features_cols].values
#%%
# %%
params={
    "task":"train",
    "boosting_type":"gbdt",
    "objective":"binary",
    "metric":"auc",
    "num_leaves":9,
    "learning_rate":0.03,
    "feature_fraction_seed":2,
    "feature_fraction":0.9,
    "bagging_fraction":0.8,
    "bagging_freq":5,
    "min_data":20,
    "min_hessian":1,
    "verbose":-1,
    "silent":0
}
X_train,X_test,Y_train,Y_test=train_test_split(train,target,test_size=0.4,random_state=0)
lgb_train=lgb.Dataset(X_train,Y_train)
lgb_eval=lgb.Dataset(X_test,Y_test,reference=lgb_train)
gbm=lgb.train(params,lgb_train,num_boost_round=10000,valid_sets=lgb_eval,early_stopping_rounds=100)
#%%
# 特征选择
# 特征选择前后的分数对比
def feature_selection(train,train_sel,target):
    clf=RandomForestClassifier(n_estimators=100,max_depth=2,random_state=0,n_jobs=-1)
    scores=cross_val_score(clf,train,target,cv=5)
    scores_sel=cross_val_score(clf,train_sel,target,cv=5)
    print("No Select Accuracy:%.2f(+/-%.2f)"%(scores.mean(),scores.std()))
    print("Selected Accuracy:%.2f(+/-%.2f)"%(scores_sel.mean(),scores_sel.std()))
# #%%
# # 删除较小的方差
# sel=VarianceThreshold(threshold=(.8*(1-.8)))
# sel=sel.fit(train)
# train_sel=sel.transform(train)
# test_sel=sel.transform(test)
# print("before select",train.shape)
# print("after select",train_sel.shape)
# feature_selection(train,train_sel,target)
#%%
# 基于lgb的特征选择
def lgb_transform(train,test,model,topK):
    train_df=pd.DataFrame(train)
    train_df.columns=range(train.shape[1])

    test_df=pd.DataFrame(test)    
    test_df.columns=range(train.shape[1])

    features_import=pd.DataFrame()
    features_import["importance"]=model.feature_importance()
    features_import["col"]=range(train.shape[1])

    features_import=features_import.sort_values(["importance"],ascending=False).head(topK)
    sel_col=list(features_import.col)
    train_sel=train_df[sel_col]
    test_sel=test_df[sel_col]
    return train_sel,test_sel
#%%
train_sel,test_sel=lgb_transform(train,test,gbm,20)
#%%
#%%
feature_selection(train,train_sel,target)
#%%
model=SBBTree(params=params,stacking_num=2,bagging_num=1,bagging_test_size=0.33,num_boost_round=10000,early_stopping_rounds=200)
model.fit(train_sel.values,target)
# %%
print("ok")
# %%
pred=model.predict(test_sel.values)
# %%
pred
# %%
test_df=pd.read_csv("./data/data_format1/data_format1/test_format1.csv")
# %%
test_df.head()
#%%
data = {'user_id':test_df["user_id"],
       'merchant_id':test_df["merchant_id"],"prob":pred}
# %%
from pandas import Series,DataFrame
df = DataFrame(data)
# %%
df.head(5)
# %%
df.to_csv("./data/result.csv")
# %%
print("ok")
# %%
