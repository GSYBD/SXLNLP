#conding:utf8
#week3作业

#词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
from xmlrpc.client import boolean


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
def all_cut(sentence, Dict):
    #TODO
    target = []
    words = split_word(sentence,Dict,10)
    print(words)
    words_back = split_word_back(sentence,Dict,10)
    print(words_back)
    print(">>>>>")
    rec2(words,Dict,target)
    rec2(words_back,Dict,target)
    target.append(words)
    target.append(words_back)
    print("<<<")
    # for aa in target:
    #     print(aa)
    
    setk = {}
    for aa in target:
        setk["/".join(aa)]=1
    target = list(setk.keys())
    for aa in target:
        print(aa.split("/"))
    
    return target

def rec2(words,dict,target):
    for idx,aa in enumerate(words):
        split_words = []
        t_wds = split_word(aa,dict,len(aa)-1) #-1 为了下切了更细
        if len(t_wds)==1: #没切
            #print("没切",aa)
            continue
        else:
            #print("切了",aa)
            for a in range(idx):
                split_words.append(words[a])
            split_words += t_wds
            split_words += words[(idx+1):]
            target.append(split_words)
            rec2(split_words,dict,target)


#正向最大
def split_word(inptext, dict,win_len):
    words = []
    if len(inptext)==1 :
        words.append(inptext)
        return words
    init_win_len = win_len
    win_len = min(len(inptext), win_len)
    word = inptext[:win_len];
    while len(word)>0:
        if len(word)==1 or word in dict:
            words.append(word)
            inptext = inptext[win_len:]
            win_len = min(len(inptext), init_win_len)
            word = inptext[:win_len]
        # elif word in dict:
        #     words.append(word)
        #     inptext = inptext[win_len:]
        #     win_len = min(len(inptext), init_win_len)
        #     word = inptext[:win_len]
        else:
            word = word[0:win_len-1]
            win_len = min(len(word), win_len)
    return words

#反向最大
def split_word_back(inptext, dict,win_len):
    words = []
    init_win_len = win_len
    win_len = min(len(inptext), win_len)
    word = inptext[-win_len:];
    while len(word)>0:
        if len(word)==1 or word in dict:
            words.append(word)
            inptext = inptext[:-win_len]
            win_len = min(len(inptext), init_win_len)
            word = inptext[-win_len:]
        else:
            word = word[-(win_len-1):]
            win_len = min(len(word), win_len)
    return words[::-1]


if __name__ == "__main__":
    #"经常有意见分歧"
    #print(sentence[:1])
    # print(sentence[-6:])
    #print(split_word("我",Dict,0))
    all_cut(sentence,Dict)


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

