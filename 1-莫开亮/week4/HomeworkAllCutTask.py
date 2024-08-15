# -*- coding: utf-8 -*-
# 词典，每个词后方存储的是其词频，仅为示例，也可自行添加
Dict = {
    "经常": 0.1,
    "经": 0.05,
    "有": 0.1,
    "常": 0.001,
    "有意见": 0.1,
    "歧": 0.001,
    "意见": 0.2,
    "分歧": 0.2,
    "见": 0.05,
    "意": 0.05,
    "见分歧": 0.05,
    "分": 0.1
}


# 实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, dictor) -> list:
    if len(sentence) < 1:
        return [[]]
    else:
        result = []
    for word in dictor:
        if sentence.startswith(word):
            remaining = all_cut(sentence[len(word):], dictor)
            for perm in remaining:
                result.append([word] + perm)
    return result


# 根据上方词典，对于输入文本，构造一个存储有所有切分方式的信息字典
# 学术叫法为有向无环图，DAG（Directed Acyclic Graph），不理解也不用纠结，只当是个专属名词就好
# 这段代码直接来自于jieba分词 get_DAG()
def calc_dag(sentence, dic):
    DAG = {}
    n = len(sentence)
    for i in range(n):
        tempList = []
        word = sentence[i]
        j = i
        while j < n:
            if word in dic:
                tempList.append(j)
            j += 1
            word = sentence[i:j + 1]
        if not tempList:
            tempList.append(i)
        DAG[i] = tempList
    return DAG


# 构造DAGDecode类，用于存储切分信息
class DAGDecode:
    """

    """

    def __init__(self, sentence, dic):
        self.sentence = sentence
        self.DAG = calc_dag(sentence, dic)
        self.length = len(sentence)
        self.unfinish_path = [[]]
        self.finish_path = []

    # 递归调用，直到切分结束
    def decode_next(self, path):
        path_length = len("".join(path))
        if path_length == self.length:
            self.finish_path.append(path)
            return
        new_paths = []
        for candidate in self.DAG[path_length]:
            new_paths.append(path + [self.sentence[path_length:candidate + 1]])
        self.unfinish_path += new_paths
        return

    # 解码
    def decode(self):
        while self.unfinish_path:
            self.decode_next(self.unfinish_path.pop())


# 实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut2(sentence, dic):
    dagDecoder = DAGDecode(sentence, dic)
    dagDecoder.decode()
    return dagDecoder.finish_path


def main():
    sentence = "经常有意见分歧"
    print(all_cut(sentence, Dict))


if __name__ == "__main__":
    main()
