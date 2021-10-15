# 存在问题:模型预测值全部为0.
#%%
import itertools
from math import isnan
from matplotlib import colors
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.classification import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection  import train_test_split,cross_val_score,ShuffleSplit,KFold,StratifiedKFold
import warnings
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor
warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"
#%%
train_target=pd.read_csv("./data/train_all_embedding.csv")
features_cols=[col for col in train_target.columns if col not in ["label","user_id"]]
train=train_target[features_cols].values
target=train_target["label"].values
test=pd.read_csv("./data/test_all_embedding.csv")[features_cols].values
# %%
stdScaler=StandardScaler()
X=stdScaler.fit_transform(train)
#%%
# %%
X_train,X_test,Y_train,Y_test=train_test_split(X,target,random_state=0)
#%%
clf=LogisticRegression(random_state=0,solver="lbfgs",multi_class="multinomial").fit(X_train,Y_train)
clf.score(X_test,Y_test)
#%%
# MLP模型
mlp = MLPRegressor(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 5), random_state=1).fit(X_train,Y_train)
mlp.score(X_test,Y_test)
#%%
# 精确度计算
scores=cross_val_score(clf,train,target,cv=5)
print("Accuracy:%0.2f(+/-%0.2f)"%(scores.mean(),scores.std()))
# %%
# f1-score计算
scores=cross_val_score(clf,train,target,cv=5,scoring="f1_macro")
#%%
print("f1-score:%0.2f (+/-%0.2f)"%(scores.mean(),scores.std()))
#%%
#设置交叉验证方式
cv=ShuffleSplit(n_splits=5,test_size=0.3,random_state=0)
cross_val_score(clf,train,target,cv=cv)

# %%
# kfold数据进行切分
kf=KFold(n_splits=5)
for k,(train_index,test_index) in enumerate(kf.split(train)):
    X_train,X_test,Y_train,Y_test=train[train_index],train[test_index],target[train_index],target[test_index]
    clf=clf.fit(X_train,Y_train)
    print(k,clf.score(X_test,Y_test))
#%%
def plot_confusion_matrix(cm,classes,normalize=False,title="Confusion matrix",cmap=plt.cm.Blues):
    if normalize:
        # np.newaxis numpy库中等价于none
        cm=cm.astype("float")/cm.sum(axis=1)[:,np.newaxis]
        print("Nomalize confusion matrix")
    print(cm)
    plt.imshow(cm,interpolation="nearest",cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=45)
    plt.yticks(tick_marks,classes)
    fmt=".2f" if normalize else 'd'
    thresh=cm.max()/2
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,
            i,
            format(cm[i,j],fmt),
            horizontalalignment="center",
            color="white" if cm[i,j]>thresh else "black"
        )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

#%%
# 分层抽样进行kfold
# repeat:重复购买,no-repeat:没有重复购买
class_names=["no-repeat","repeat"]
skf=StratifiedKFold(n_splits=5)
for k,(train_index,test_index) in enumerate(skf.split(train,target)):
    X_train,X_test,Y_train,Y_test=train[train_index],train[test_index],target[train_index],target[test_index]
    clf=clf.fit(X_train,Y_train)
    print(k,clf.score(X_test,Y_test))
    Y_pred=clf.predict(X_test)
    cnf_matrix=confusion_matrix(Y_test,Y_pred)
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cnf_matrix,classes=class_names)
    plt.show()
    print(classification_report(Y_test,Y_pred,target_names=class_names))

#%%
test[np.isnan(test)]=0

test[np.isinf(test)]=0
pred=clf.predict(test)
#%%