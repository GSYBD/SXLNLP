#week3作业
import copy
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
def all_cut(sentence, Dict):
    target = []
    path = []
    def dfs(i, j):
        if i == len(sentence):
            target.append(path.copy())
            return
        for k in range(j+1, len(sentence)+1):
            if sentence[i:k] in Dict.keys():
                path.append(sentence[i:k])
                dfs(k, k)
                path.pop()
    dfs(0, 0)
    return target

result = all_cut(sentence, Dict)
rerank_result = {}
for index, j in enumerate(sorted(result,reverse=True)):
    score_tmp = 0
    for i in j:
        score_tmp += Dict[i]
    rerank_result[score_tmp] = j
    # print(f'序号：{index}')
    print(j, f"——————统计得分： {score_tmp:.3f}")

# 输出得分前三项的分词
print('\n', [k[1] for k in sorted(rerank_result.items(), key=lambda x:x[0], reverse=True)[:3]])

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

