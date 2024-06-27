Here is the Tutorial for week2 --- to create a multi-classes classification task based on cross entropy loss

"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个10维向量，所有值在0到1之间
如果前五维的平均值大于后五维的平均值，并且至少有三个值大于0.5，则为类别1
如果前五维的平均值小于后五维的平均值，并且至少有三个值小于0.5，则为类别2
其他情况为类别0
"""
