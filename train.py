
import torch
from torch.utils.data import DataLoader,Dataset
import os
import re
from random import sample, random
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer


data_base_path="aclImdb_v1/aclImdb"
model_path = r"./aclImdb_v1/aclImdb/mode"


# 1. 准备dataset，这里写了一个数据读取的类，并把数据按照不同的需要进行了分类；
class ImdbDataset(Dataset):
    def __init__(self, mode, testNumber=10000, validNumber=5000):


        super(ImdbDataset, self).__init__()

        # 读取所有的训练文件夹名称
        text_path_test = [os.path.join(data_base_path, i) for i in ["test/neg", "test/pos"]]
        text_path=[]
        text_path.extend([os.path.join(data_base_path, i) for i in ["train/neg", "train/pos"]])

        if mode == "train":
            self.total_file_path_list = []
            for i in text_path:
                self.total_file_path_list.extend([os.path.join(i, j) for j in os.listdir(i)])
            self.total_file_path_list = sample(self.total_file_path_list, testNumber)
        if mode == "test":
            self.total_file_path_list = []
            # 获取测试数据集，默认10000个数据
            for i in text_path_test:
                self.total_file_path_list.extend([os.path.join(i, j) for j in os.listdir(i)])
            self.total_file_path_list = sample(self.total_file_path_list, testNumber)

        if mode == "valid":
            self.total_file_path_list = []
            # 获取验证数据集，默认5000个数据集
            for i in text_path:
                self.total_file_path_list.extend([os.path.join(i, j) for j in os.listdir(i)])
            self.total_file_path_list = sample(self.total_file_path_list, validNumber)

    def tokenize(self, text):

        # 具体要过滤掉哪些字符要看你的文本质量如何

        # 这里定义了一个过滤器，主要是去掉一些没用的无意义字符，标点符号，html字符啥的
        fileters = ['!', '"', '#', '$', '%', '&', '\(', '\)', '\*', '\+', ',', '-', '\.', '/', ':', ';', '<', '=', '>',
                    '\?', '@'
            , '\[', '\\', '\]', '^', '_', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”', '“', ]
        # sub方法是替换
        text = re.sub("<.*?>", " ", text, flags=re.S)  # 去掉<...>中间的内容，主要是文本内容中存在<br/>等内容
        text = re.sub("|".join(fileters), " ", text, flags=re.S)  # 替换掉特殊字符，'|'是把所有要匹配的特殊字符连在一起
        return text  # 返回文本

    def __getitem__(self, idx):
        cur_path = self.total_file_path_list[idx]
        cur_filename = os.path.basename(cur_path)
        labels = []
        sentences = []
        if int(cur_filename.split("_")[-1].split(".")[0]) <= 5:
            label = 0
        else:
            label = 1
        labels.append(label)
        text = self.tokenize(open(cur_path, encoding='UTF-8').read().strip())  # 处理文本中的奇怪符号
        sentences.append(text)
        # 可见我们这里返回了一个list，这个list的第一个值是标签0或者1，第二个值是这句话；
        return sentences, labels

    def __len__(self):
        return len(self.total_file_path_list)


# 2. 这里开始利用huggingface搭建网络模型
# 这个类继承再nn.module,后续再详细介绍这个模块
#
class BertClassificationModel(nn.Module):
    def __init__(self, hidden_size=768):
        super(BertClassificationModel, self).__init__()
        #使用bert
        model_name = r'bert_base_uncased'
        # 读取分词器
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
        # 读取预训练模型
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=model_name)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, batch_sentences):  # [batch_size,1]
        sentences_tokenizer = self.tokenizer(batch_sentences,
                                             truncation=True,
                                             padding=True,
                                             max_length=200,
                                             add_special_tokens=True)
        input_ids = torch.tensor(sentences_tokenizer['input_ids'])  # 变量
        attention_mask = torch.tensor(sentences_tokenizer['attention_mask'])  # 变量
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)  # 模型

        last_hidden_state = bert_out[0]  # [batch_size, sequence_length, hidden_size] # 变量
        bert_cls_hidden_state = last_hidden_state[:, 0, :]  # 变量
        fc_out = self.fc(bert_cls_hidden_state)  # 模型
        return fc_out


# 3. 程序入口，模型也搞完啦，我们可以开始训练，并验证模型的可用性
def main():
    trainNumber = 10000  # 用10000个数据训练
    validNumber = 5000  # 5000个数据参与验证
    batchsize = 80  # 每次放80个数据参加训练

    trainDatas = ImdbDataset(mode="train", testNumber=trainNumber)  # 加载训练集,全量加载，考虑到我的破机器，先加载个100试试吧
    validDatas = ImdbDataset(mode="test", validNumber=validNumber)  # 加载训练集

    train_loader = torch.utils.data.DataLoader(trainDatas, batch_size=batchsize,
                                               shuffle=False)  # 遍历train_dataloader 每次返回batch_size条数据

    val_loader = torch.utils.data.DataLoader(validDatas, batch_size=batchsize, shuffle=False)

    # 这里搭建训练循环，输出训练结果
    epoch_num = 1
    print('training...(约1 hour(CPU))')

    # 初始化模型
    model = BertClassificationModel()
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)  # 首先定义优化器，这里用的AdamW，lr是学习率，因为bert用的就是这个

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
    #保存训练后的模型
    model.bert.save_pretrained("my_finetuned_model")
    model.tokenizer.save_pretrained("my_finetuned_model")

    #验证过程
    print("test...")
    num = 0
    model.eval()  # 不启用 BatchNormalization 和 Dropout，保证BN和dropout不发生变化,主要是在测试场景下使用；
    for j, (data, labels) in enumerate(val_loader, 0):
        output = model(data[0])
        out = output.argmax(dim=1)
        num += (out == labels[0]).sum().item()

    print('Accuracy:', num / validNumber)


if __name__ == '__main__':
    main()