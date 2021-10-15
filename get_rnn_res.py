#%%
from Feature_Engine.memory_reduce import reduce_mem_usage
import torch
import numpy as np
from torch.autograd import Variable
import pandas as pd
#%%
test_data=reduce_mem_usage(pd.read_csv("./data/test_all_embedding.csv"))
#%%
features_cols=[]
for i in range(100):
    features_cols.append("embeeding_"+str(i))
features_cols.append("lgb")
features_cols.append("xgb")
features_cols.append("label")
#%%
from rnn import lstm_reg
model = torch.load('./myModel.pkl')
#%%
group_df=test_data.groupby(["time_stamp_nunique"])
groups_df_list=list(test_data.groupby("time_stamp_nunique").groups)
time_test_X=[]
features_cols=[col for col in features_cols if col not in ["label","user_id"]]
for group in groups_df_list:
    df=reduce_mem_usage(group_df.get_group(group))
    test_X=df[features_cols].values
    time_test_X.extend(test_X)

time_test_X=np.array(time_test_X).reshape(-1,1,102)
test_var_x=Variable(torch.Tensor(time_test_X))
pred_y=model(test_var_x)
print(pred_y)
#%%
#%%
with open("./pred_y.txt","w+") as f:
    f.write(str(pred_y))
# %%
torch.set_printoptions(threshold=np.inf)
# %%
res_list=[]
for res_ in pred_y:
    res_list.append(res_[1].item())
#%%
import random
random.shuffle(res_list)
#%%
# %%
test_df=pd.read_csv("./data/data_format1/data_format1/test_format1.csv")
data = {'user_id':test_df["user_id"],
       'merchant_id':test_df["merchant_id"],"prob":res_list}
# %%
from pandas import Series,DataFrame
df = DataFrame(data)
# %%
df.to_csv("./data/result_rnn.csv")
# %%
