#week3作业
import time

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

#基于深度优先算法-实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut_dfs(sentence, Dict):
    start_time = time.time()

    def dfs(start):
        if start == len(sentence):
            return [[]]
        results = []
        for end in range(start + 1, len(sentence) + 1):
            word = sentence[start:end]
            if word in Dict:
                for sub_str in dfs(end) :
                    results.append([word] + sub_str)
        return results
    target = dfs(0)
    print("深度优先算法耗时：", time.time() - start_time)
    return target

#基于动态规划算法-实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut_lp(sentence, Dict):
    n = len(sentence)
    dp = [[] for _ in range(n + 1)]
    dp[0] = [[]]
    start_time = time.time()

    for i in range(n):
        if dp[i]:
            for j in range(i + 1, n + 1):
                word = sentence[i:j]
                if word not in Dict: continue
                # if word in Dict:
                for prev_segmentation in dp[i]:
                    dp[j].append(prev_segmentation + [word])
    print("动态规划耗时：", time.time() - start_time)
    return dp[n]

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

def main(method,sentence,Dict) :
    return method(sentence,Dict)

target_dfs = main(all_cut_dfs, sentence, Dict)
print(target_dfs)
target_lp = main(all_cut_lp, sentence, Dict)
print(target_lp)
