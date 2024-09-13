# 李超凡的课后作业

## 2024.09.03 week12
    1. 阅读大模型代码梳理各个模型的模型结构 √

## 2024.08.29 week11
    1. 在基于bert的seq2seq任务代码上实现SFT √
    2. 基于新闻标题数据，修改数据集构造函数， 文本拼接、生成掩码、label构造 √

## 2024.08.23 week10
    1. 将文本生成任务的LSTM层替换为Bert √
    2. 为Bert层添加上三角Attention Mask √

## 2024.08.16 week9
    1. 修改ner任务文件，替换LSTM层为Bert,训练模型验证效果 √
    2. 编写predict文件，调用模型执行NER任务 √

## 2024.08.08 week8
    1. 修改model文件loss函数新增传入三个句子时使用triplet √
    2. 修改loader文件适配triplet损失函数√
    3. 修改main文件使用triplet损失函数训练模型√

## 2024.08.01 week7
    1. NLP语言模型任务pipeline √
    2. 文本分类任务 √
    3. 使用不同模型和参数组合训练模型，比较模型准确率 √
    注：由于电脑性能对训练速度的限制，目前仅完成"fast_text", "lstm", "gru", "rnn", "cnn", "gated_cnn", "stack_gated_cnn", "rcnn", "bert", "bert_lstm" 10个模型的不同参数组合验证，后续会继续运行并更新结果。

## 2024.07.25 week6
    1. 计算bert模型可训练参数量 √

## 2024.07.18 week5
    1. 使用gensim训练一个词向量模型 √
    2. 使用kmeans聚类实现对新闻标题聚类 √
    3. 类内点到聚类中心的平均距离对类排序，输出平均距离最小的前k个类标题 √


## 2024.07.08 week4
    1. total_segmentation_combination.py:作业，递归实现全切分并获取所有正确分词组合 √
    2. sbornn.py:手写课上RNN分词 √
    3. new_words_detected.py:基于凝聚度和左右熵的新词发现 √
    4.  tfidf_search_engine.py:基于TF_IDF的搜索引擎实现 √


## 2024.07.04 week3
    1. 运行课堂老师作业案例 √
    2. 参考课上案例，收集新闻标题分类数据集，使用simple rnn实现新闻标题分类 √


## 2024.06.29 week2
    1. 运行课堂老师作业案例 √
    2. 参考课上案例编写复杂逻辑，实现深度学习全流程 √
    3. 课堂主要内容思维导图整理 √


## 2024.06.26 week1
    1.准备开发环境 √
    2.配置github，并提交个人文件夹 √
