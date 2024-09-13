import jieba

#词典，每个词后方存储的是其词频，仅为示例，也可自行添加
Dict = {"我":0.1,
        "我爱":0.05,
        "北京":0.1,
        "天":0.001,
        "天安门":0.1,
        "爱":0.001,
        "我爱北京":0.2,
        "北":0.2}


#根据上方词典，对于输入文本，构造一个存储有所有切分方式的信息字典
def calc_dag(sentence):
        DAG = {}    #DAG空字典，用来存储DAG有向无环图
        N = len(sentence)
        for k in range(N):
            tmplist = []
            i = k
            frag = sentence[k]
            while i < N:
                if frag in Dict:
                    tmplist.append(i)
                i += 1
                frag = sentence[k:i + 1]
            if not tmplist:
                tmplist.append(k)
            DAG[k] = tmplist
        return DAG

sentence = "我爱北京天安门"
print(calc_dag(sentence))


#将DAG中的信息解码（还原）出来，用文本展示出所有切分方式
class DAGDecode:
    #通过两个队列来实现
    def __init__(self, sentence):
        self.sentence = sentence
        self.DAG = calc_dag(sentence)  #使用了上方的函数
        self.length = len(sentence)
        self.unfinish_path = [[]]   #保存待解码序列的队列
        self.finish_path = []  #保存解码完成的序列的队列

    #对于每一个序列，检查是否需要继续解码
    #不需要继续解码的，放入解码完成队列
    #需要继续解码的，将生成的新队列，放入待解码队列
    #path形如:["经常", "有", "意见"]
    def decode_next(self, path):
        path_length = len("".join(path))
        if path_length == self.length:  #已完成解码
            self.finish_path.append(path)
            return
        candidates = self.DAG[path_length]
        new_paths = []
        for candidate in candidates:
            new_paths.append(path + [self.sentence[path_length:candidate+1]])
        self.unfinish_path += new_paths  #放入待解码对列
        return

    #递归调用序列解码过程
    def decode(self):
        while self.unfinish_path != []:
            path = self.unfinish_path.pop() #从待解码队列中取出一个序列
            self.decode_next(path)     #使用该序列进行解码


sentence = "我爱北京天安门"
dd = DAGDecode(sentence)
dd.decode()
print(dd.finish_path)