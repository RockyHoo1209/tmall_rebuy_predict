#%%
from Feature_Engine.memory_reduce import reduce_mem_usage
import torch
import numpy as np
from torch.autograd import Variable
import pickle
import tqdm
import pandas as pd
#%%
train_target=reduce_mem_usage(pd.read_csv("./data/train_all_embedding.csv"))
group_df=train_target.groupby(["time_stamp_nunique"])
groups_df_list=list(train_target.groupby("time_stamp_nunique").groups)
#%%
features_cols=[]
for i in range(100):
    features_cols.append("embeeding_"+str(i))
features_cols.append("lgb")
features_cols.append("xgb")
features_cols.append("label")
#%%
time_train_X=[]
time_train_Y=[]
features_cols=[col for col in features_cols if col not in ["label","user_id"]]
for group in groups_df_list:
    df=reduce_mem_usage(group_df.get_group(group))
    train_X=df[features_cols].values
    time_train_X.extend(train_X)

    train_Y=df["label"].values.astype(np.int)
    train_Y[np.isinf(train_Y)]=0
    time_train_Y.extend(train_Y)
#%%
time_train_X=np.array(time_train_X).reshape(-1,1,102)
time_train_Y=np.array(time_train_Y).reshape(1,-1,1)
print(time_train_Y)
#%%
class lstm_reg(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size=2, num_layers=2):
        super(lstm_reg, self).__init__()
        self.rnn = torch.nn.LSTM(input_size, hidden_size, num_layers) # rnn
        self.out = torch.nn.Linear(hidden_size, output_size) # 回归
        # 初始化权重
        for name, param in self.rnn.named_parameters():
            # Xavier正态分布
            if name.startswith("weight"):
                torch.nn.init.xavier_normal_(param)
            else:
                torch.nn.init.zeros_(param)


    #确保每个维度的长度一致 
    def __getitem__(self, idx):
        # data: seq_len * input_size
        data, label, seq_len = self.train_data[idx]
        # pad_data: max_seq_len * input_size
        pad_data = np.zeros(shape=(self.max_seq_len, data.shape[1]))
        pad_data[0:data.shape[0]] = data
        sample = {'data': pad_data, 'label': label, 'seq_len': seq_len}
        return sample

    # 在使用交叉熵时注意模型输出的是各分类的概率
    def forward(self, x):
        x, _ = self.rnn(x) # (seq, batch, hidden)
        s, b, h = x.shape
        x = x.view(s*b, h) # 转换成线性层的输入格式
        x = self.out(x)
        x = x.view(s, b, -1)
        x = x.view(s*b,2)
        # x = x.view(s*b,1)
        # 出现nan值
        x=torch.where(torch.isnan(x), torch.full_like(x, 0), x)    
        x = torch.where(torch.isinf(x), torch.full_like(x, 0), x)
        # 交叉熵需搭配softmax函数(why？)
        return torch.softmax(x,1)

#%%
# 训练并保存模型
def Train_Model():
    model=lstm_reg(102,4)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    for idx in  tqdm(range(20)):
        var_x = Variable(torch.Tensor(time_train_X))
        var_y = Variable(torch.tensor(time_train_Y,dtype=torch.long).reshape(-1))
        print(var_y.size())
        out = model(var_x)
        print(out.size())
        loss = criterion(out, var_y)
        optimizer.zero_grad()
        loss.backward()
        print("Loss:%.2f"%loss.item())
        optimizer.step()
        try:
            if idx%100==0:
                print('Epoch: %d, Loss: %.2f'%(idx + 1, loss.item()))
        except Exception as e:
            print(repr(e))
            continue
    torch.save(model, "./myModel2.pkl")
# Train_Model()