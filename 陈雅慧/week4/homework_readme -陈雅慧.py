#week3作业
#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):#需保证词表里的每个单字也在词表里
    max_key_length = 0
    # 遍历字典键，更新最大键长度
    for key in Dict.keys():
        if len(key) > max_key_length:
            max_key_length = len(key)
    target=[]
    path=[]
    length=len(sentence)
    def dfs(i):
        if i==length:
            target.append(path.copy())
        for num in range(max_key_length,-1,-1):
            if sentence[i:i+num] in Dict.keys() and (i+num)<=length:#超出length的范围就不切了
                path.append(sentence[i:i+num])
                dfs(i+num)
                path.pop()#恢复现场
    dfs(0)
    return target,len(target)

if __name__ == '__main__':
    sentence = "经常有意见分歧"
    # 词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
    Dict = {"经常": 0.1,
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
            "分": 0.1}
    print(all_cut(sentence,Dict))

#目标输出;顺序不重要
# target = [
#     ['经常', '有意见', '分歧'],
#     ['经常', '有意见', '分', '歧'],
#     ['经常', '有', '意见', '分歧'],
#     ['经常', '有', '意见', '分', '歧'],
#     ['经常', '有', '意', '见分歧'],
#     ['经常', '有', '意', '见', '分歧'],
#     ['经常', '有', '意', '见', '分', '歧'],
#     ['经', '常', '有意见', '分歧'],
#     ['经', '常', '有意见', '分', '歧'],
#     ['经', '常', '有', '意见', '分歧'],
#     ['经', '常', '有', '意见', '分', '歧'],
#     ['经', '常', '有', '意', '见分歧'],
#     ['经', '常', '有', '意', '见', '分歧'],
#     ['经', '常', '有', '意', '见', '分', '歧']
# ]
#
