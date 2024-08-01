# -*- coding: utf-8 -*-
# name answer_week4
# date 2024/7/13 18:59

import jieba

# 词典。每个词后 存放的是其词频，这是示例
Dict = {"经常":0.1,
        "经":0.05,
        "有":0.1,
        "常":0.001,
        "有意见":0.1,
        "歧":0.001,
        "意见":0.2,
        "见":0.05,
        "意":0.05,
        "见分歧":0.05,
        "分":0.1}


# 根据上方词典，对于输入文本，构造一个存储有所有切分方式的信息字典
# 学术叫法为有向无环图，DAG（Directed Acyclic Graph）
# 直接使用jieba分词
def calc_dag(sentence):
    DAG = {} #DAG空字典，用来存储DAG有向五环图
    N = len(sentence)
    for k in range(N):
        tmplist = []
        i = k
        frag = sentence[k]

        while i < N:
            if frag in Dict:
                 tmplist.append(i)
            i +=1

            frag = sentence[k:i+1]

        if not tmplist:
            tmplist.append(k)

        DAG[k] = tmplist

    return DAG

sentence = "经常有意见分歧"
print(calc_dag(sentence=sentence))

"""
结果为 {0: [0, 1], 1: [1], 2: [2, 4], 3: [3, 4], 4: [4, 6], 5: [5], 6: [6]}
0：[0,1] 代表句子中的第0个字 可以单独组成词，或与第1个字一起成词
2:[2,4] 代表句子中有2个字，可以单独组成词 或第2 到4 个字一起成词
# 以此类推
# 这个字典中实际上九存储量所有可能切分方式的信息

"""


# 将DAG中信息码还原出来。用文本方式展示所有切分

class DAGDecode:
    # 通过2个队列来实现
    def __init__(self,sentence):
        self.setence = sentence
        self.DAG = calc_dag(sentence=sentence) #使用上方函数
        self.length = len(sentence)
        self.unfinish_path = [[]] # 保存待解码 序列的队列
        self.finish_path = [] ## 保存解码完成的队列

    """
    对每一个序列，检查是否需要继续解码
    不需要继续解码的放入解码完成的队列
    需要解码的将生成 新的队列，放入待解码队列
    path  如 ["经常","有","意见"]
    """

    def decode_next(self,path):
        path_length = len(''.join(path))
        print(path_length,'path_length')
        if path_length == self.length: # 已完成解码
            self.finish_path.append(path)
            return

        candidates = self.DAG[path_length]

        new_paths = []
        for candidate in candidates:
            new_paths.append(path + [self.setence[path_length:candidate+1]])
            print(path + [self.setence[path_length:candidate+1]])
        self.unfinish_path += new_paths # 放入带解码队列

        return

    # 递归调用序列解码过程
    def decode(self):
        while self.unfinish_path != []:
            path = self.unfinish_path.pop() # 从待解队列中取出一个序列
            self.decode_next(path)


entence = "经常有意见分歧"
dd = DAGDecode(sentence)
dd.decode()
print(dd.finish_path)