#week3作业
import string
import queue
#词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
Dict = {"经常":0.1,
        "经":0.05,
        "有":0.1,
        "常":0.001,
        "有意见":0.1,
        "歧":0.001,
        "意见":0.2,
        "分歧":0.2,
        "见":0.05,
        "意":0.05,
        "见分歧":0.05,
        "分":0.1}

#待切分文本
sentence = "经常有意见分歧"

#实现全切分函数，输出根据字典能够切分出的所有的切分方式

class TrieNode():
    def __init__(self):
        self.mp = {}
        self.word = False
    
    def insert(self, word:string)->None:
        cur = self
        for c in word:
            if c not in cur.mp.keys():
                cur.mp[c] = TrieNode()
            cur = cur.mp[c]
        cur.word = True

    def search(self, word:string)->list:
        add_on = []
        cur = self
        # i = 0
        for i, c in enumerate(word):
            if c not in cur.mp.keys():
                break
            cur = cur.mp[c]
            if cur.word == True:
                add_on.append(i)
        # if cur.word == True:
        #     add_on.append(i)
        return add_on


def all_cut(sentence, Dict):
    #TODO

    root = TrieNode()
    print(Dict.keys())
    for w in Dict.keys():
        # print(w)
        root.insert(w)

    q = queue.Queue()
    q.put((sentence,[]))

    while not q.empty():
        rest, cur = q.get()
        for i in root.search(rest):
            if i >= len(rest):
                target.append(cur)
            else:
                cur.append(rest[:i])
                back = rest[i+1:]
                q.put((back,cur))   

    return target

#目标输出;顺序不重要
target = [
    ['经常', '有意见', '分歧'],
    ['经常', '有意见', '分', '歧'],
    ['经常', '有', '意见', '分歧'],
    ['经常', '有', '意见', '分', '歧'],
    ['经常', '有', '意', '见分歧'],
    ['经常', '有', '意', '见', '分歧'],
    ['经常', '有', '意', '见', '分', '歧'],
    ['经', '常', '有意见', '分歧'],
    ['经', '常', '有意见', '分', '歧'],
    ['经', '常', '有', '意见', '分歧'],
    ['经', '常', '有', '意见', '分', '歧'],
    ['经', '常', '有', '意', '见分歧'],
    ['经', '常', '有', '意', '见', '分歧'],
    ['经', '常', '有', '意', '见', '分', '歧']
]



temp = all_cut(sentence, Dict)
print("99999999999999999999")
for t in temp:
    print(t)
