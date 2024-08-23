import numpy as np

'''
leetcode题解：
https://leetcode.cn/problems/edit-distance/solutions/188223/bian-ji-ju-chi-by-leetcode-solution/
1. 通过题目整理出实际的操作。（每次类比爬楼梯，爬楼梯可以看作1维数组吗？可以，数组长度看作楼梯长度，每个值是当前阶梯的状态）
    a.对w1 添加一个字符
    b.对w2 添加一个字符
    c. 替换一个字符
2. 从最后一个出发，整理出递推公式（状态转移）
    如果当前最后字符相同
    min(arr[i-1][j],arr[i][j-1],arr[i-1][j-1] - 1) + 1
    如果不同
    min(arr[i-1][j],arr[i][j-1],arr[i-1][j-1]) + 1
3. 以数组的形式表达。

ps:数组下标可以理解为字符当前位置。从而对应当前选择的操作。比如arr[i-1][j] 到 arr[i][j] 可以看做操作B
'''
def minDistance(word1: str, word2: str) -> int:
    i = len(word1) + 1
    j = len(word2) + 1

    arr = np.zeros((j,i),dtype=int)
    for j_idx in range(j):
        for i_idx in range(i):
            if(i_idx == 0 or j_idx == 0):
                arr[j_idx][i_idx] = i_idx + j_idx
            else:
                if(word1[i_idx -1] == word2[j_idx - 1]):
                    arr[j_idx][i_idx] = min(arr[j_idx-1][i_idx],arr[j_idx][i_idx-1],arr[j_idx-1][i_idx-1] - 1) + 1
                else:
                    arr[j_idx][i_idx] = min(arr[j_idx-1][i_idx],arr[j_idx][i_idx-1],arr[j_idx-1][i_idx-1]) + 1
    return arr[j_idx][i_idx]

dist =minDistance('horse','ros')
print(dist)