import re
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader,Dataset
from sklearn.model_selection import train_test_split
import random
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer
import pandas as pd
import numpy as np
class BertClassificationModel(nn.Module):
    def __init__(self):
        super(BertClassificationModel, self).__init__()
        #使用bert
        model_name = r'D:\hugging_face_models\bert_base_chinese'
        # 读取分词器
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
        # 读取预训练模型
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=model_name)
        self.fc = nn.Linear(768, 2)#二分类

    def forward(self, batch_sentences):  # [batch_size,1]
        sentences_tokenizer = self.tokenizer(batch_sentences,
                                             truncation=True,
                                             padding=True,
                                             max_length=32,
                                             add_special_tokens=True)
        input_ids = torch.tensor(sentences_tokenizer['input_ids'])  # 变量
        attention_mask = torch.tensor(sentences_tokenizer['attention_mask'])  # 变量
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)  # 模型

        last_hidden_state = bert_out[0]  # [batch_size, sequence_length, hidden_size] # 变量
        bert_cls_hidden_state = last_hidden_state[:, 0, :]  # 变量
        fc_out = self.fc(bert_cls_hidden_state)  # 模型
        return fc_out



#读入数据
data = pd.read_csv('train.news.csv')
# 储存新闻标题
title_df = data['Title']
# 储存真假标记
label_df = data['label']
# 转化成numpy的数组
title_array = np.array(title_df)
label_array = np.array(label_df)
X_train,X_test,y_train,y_test = train_test_split(title_array,label_array,test_size=0.3)
data_train=list(zip(X_train,y_train))#训练集
data_test=list(zip(X_test,y_test))#验证集
random.shuffle(data_train)
print(data_train)
class dataLoader():
    def __init__(self,data):
         self.data_train=data

    def __getitem__(self, idx):
        labels = []
        sentences = []
        labels.append(self.data_train[idx][1])
        sentences.append(self.tokenize(self.data_train[idx][0]))

        return sentences, labels
    def tokenize(self, text):
        fileters = ['!', '"', '#', '$', '%', '&', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>','《','》'
                    '\?', '@', '\[', '\\', '\]', '^', '_', '`', '{', '|', '}', '~', '\t', '\n', '\x97', '\x96', '”', '“', '\'']
        # sub方法是替换
        text = re.sub("<.*?>", " ", text, flags=re.S)  # 去掉<...>中间的内容，主要是文本内容中存在<br/>等内容
        text.replace(" ","")
        return text  # 返回文本
    def __len__(self):
        return len(self.data_train)


data_train_trans=dataLoader(data=data_train)
batchsize = 50 # 每次放50个数据参加训练
train_loader = torch.utils.data.DataLoader(data_train_trans, batch_size=batchsize,
                                           shuffle=False)
epoch_num=5
#定义模型
model = BertClassificationModel()
optimizer = optim.AdamW(model.parameters(), lr=2e-5)  # 首先定义优化器，这里用的AdamW，lr是学习率，因为bert用的就是这个

# 这里是定义损失函数，交叉熵损失函数比较常用解决分类问题
criterion = nn.CrossEntropyLoss()
print("模型数据已经加载完成,现在开始模型训练。")
for epoch in range(epoch_num):
    for i, (data, labels) in enumerate(train_loader, 0):
        output = model(data[0])
        optimizer.zero_grad()  # 梯度清0
        loss = criterion(output, labels[0])  # 计算误差
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        # 打印一下每一次数据扔进去学习的进展
        print('batch:%d loss:%.5f' % (i, loss.item()))

    # 打印一下每个epoch的深度学习的进展i
    print('epoch:%d loss:%.5f' % (epoch, loss.item()))
    #每次训练之前打乱训练集
    random.shuffle(data_train)
    train_loader = torch.utils.data.DataLoader(data_train_trans, batch_size=batchsize,shuffle=False)
    if epoch==3:#后两轮降低学习率
        optimizer = optim.AdamW(model.parameters(), lr=1e-5)



# 保存训练后的模型
model.bert.save_pretrained("my_finetuned_model")
model.tokenizer.save_pretrained("my_finetuned_model")

yanzheng=dataLoader(data_test)
yanzheng_loader = torch.utils.data.DataLoader(yanzheng, batch_size=batchsize,
                                           shuffle=False)
#验证
model.eval()#启动测评模式
outcome0=[]
for j, (data, labels) in enumerate(yanzheng_loader, 0):
    output = model(data[0])
    out = output.argmax(dim=1)
    for i in list(out):
        outcome0.append(int(i))

auc_score = roc_auc_score(y_test,outcome0)
print(auc_score)
data_test = pd.read_csv('test.feature.csv')
title_df_test = data_test['Title']
title_array_test = np.array(title_df_test)
extend=[0]*len(title_array_test)
data_test=list(zip(title_array_test,extend))
data_test_trans=dataLoader(data=data_test)
outcome=[]

test_loader = torch.utils.data.DataLoader(data_test_trans, batch_size=batchsize,
                                           shuffle=False)


for j, (data, labels) in enumerate(test_loader, 0):
    output = model(data[0])
    out = output.argmax(dim=1)
    for i in list(out):
        outcome.append(int(i))

print(outcome)
t_df=pd.DataFrame(outcome)
t_df.index=t_df.index+1
t_df.to_csv('outcome.csv')




