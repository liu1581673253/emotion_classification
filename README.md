# emotion_classification

代码在master那个分支里，叫train.py，GitHub用的不是很熟练，所以传入的东西很混乱
代码在跑通一边后又胡乱改过，不确定是不是能跑

模型使用bert-base-uncased，优化器使用AdamW，使用交叉熵损失函数，学习率设置1e-5，batchsize = 80 ,max_length=200,epoch_num = 1

（其实这个项目做的比较粗糙，因为学到bert模型后，遂用之做python大作业，python大作业和这个项目很像，是中文的虚假新闻检测，也是自然语言处理+二分类。
  那个大作业做的精细一点，且鉴于和这个项目很像，故一并传上。其在测试集上的结果为auc=0.8221）




