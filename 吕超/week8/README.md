# 1. 需求:
    使用cosine_triplet_loss()函数进行文本匹配
# 2. 修改点
2.1) 修改loader.py中 --> random_train_sample(self):
采出3个样本,2个相似,1个不相似, 顺序为a/p/n;
不需要输出1和-1, 顺序就能说明相似性
2.2) 修改model.py中 forward(),传入3个句子
