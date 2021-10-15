'''
Author: Rocky Hoo
Date: 2021-07-06 05:16:12
LastEditTime: 2021-07-06 09:30:22
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
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
def stacking_clf(clf,clf_name,train_x,train_y,test_x,kf,folds,label_split=None):
    # 新建train、test 不覆盖原本数据
    train=np.zeros((train_x.shape[0],1))
    test=np.zeros((test_x.shape[0],1))
    test_pre=np.empty((folds,test_x.shape[0],1))
    cv_scores=[]
    # 只用对train_x进行划分，train_y(需要预测的标签)的index与train_x一一对应
    for i,(train_idx,test_idx) in enumerate(kf.split(train_x,label_split)):
        part_train_x=train_x[train_idx]
        part_train_y=train_y[train_idx]
        part_test_x=train_x[test_idx]
        part_test_y=train_y[test_idx]
        
        if clf_name in ["rf","ada","gb","et","lr","knn","gnb"]:
            train_df=part_test_x["target"]
            clf.fit(part_train_x,part_test_y)
            pre=clf.predict_proba(part_test_x)
            # why reshape?
            train[test_idx]=pre[:,0].reshape(-1,1)
            test_pre[i,:]=clf.predict_proba(test_x)[:,0].reshape(-1,1)

            cv_scores.append(log_loss(part_test_y,pre[:,0].reshape(-1,1)))
        
# all_data_test=pd.
