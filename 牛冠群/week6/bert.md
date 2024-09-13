1.embedding层
bert中的embedding有三种，分别为word embedding、position embedding、sentence embedding。
在bert-base-chinese这个模型中，词汇数量为21128，embedding维度为768，每条数据长度L为512。
word embedding参数量：21128*768
position embedding参数量：512*768
sentence embedding参数量：2*768
embedding层中的参数为
21128*768+512*768+2*768+768+768

2.self-attention层
self-attention 一共有12层，每层中有两部分组成，分别为multihead-Attention 和Layer Norm层
multihead-Attention 中有Q、K、V三个转化矩阵和一个拼接矩阵，Q、K、V的shape为：768*12*64 +768 第一个768为embedding维度，12为head数量，64为子head的维度，最后加的768为模型中的bias。经过Q、K、V变化后的数据需要concat起来，额外需要一个768*768+768的拼接矩阵。
Layer Norm参数量：768+768
self-attention一层中的参数为：
(768*12*64 +768)*3+768*768+768 +768+768
一共12层
((768*12*64 +768)*3+768*768+768 +768+768)*12

3.feedforward层
feedforward 一共有12层，每层中有两部分组成，分别为feedforward和Layer Norm层
feedforward一层中的参数为：
(768*768*4 +768*4)+(768*4*768+768) + 768+768
一共12层
((768*768*4 +768*4)+(768*4*768+768) + 768+768)*12
