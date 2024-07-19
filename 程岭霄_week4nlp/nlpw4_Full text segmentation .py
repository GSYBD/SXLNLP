import re
import time
import json


def get_dag(sentence, Dict):

    DAG = dict()  # 初始化DAG字典

    # 遍历文本的每个字符
    for i in range(len(sentence)):
        temp_list = list()  # 初始化临时列表，用于存储每个字的后缀词的下标
        k = i  # 从当前字符开始
        # 构建从当前字符开始的后缀词
        while k < len(sentence) and sentence[i:k + 1] in Dict:  # 判断词是否存在词典中
            temp_list.append(k)  # 将后缀词的结束下标加入临时列表
            k += 1  # 移动到下一个字符
        if not temp_list:  # 如果临时列表为空，则后缀词不存在
            temp_list.append(i)  # 将当前字符的下标加入临时列表
        DAG[i] = temp_list  # 将临时列表加入DAG字典
    return DAG

def all_cut(sentence, Dict):
    if not sentence:  # 如果文本为空，返回空列表
        return []
    dag = get_dag(sentence, Dict)  # 生成DAG
    result = []  # 初始化结果列表

    def traverse(index, path):
        if index == len(sentence):  # 如果遍历到文本末尾
            result.append(path[:])  # 将当前路径加入结果列表
            return
        n = dag[index]
        for end in n:  # 遍历当前字符的所有后缀词的结束下标
            traverse(end + 1, path + [sentence[index:end + 1]])  # 递归遍历下一个字符

    traverse(0, [])  # 从第一个字符开始遍历
    return result  # 返回所有可能的切分结果


if __name__ == '__main__':
    # 词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
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

    # 待切分文本
    sentence = "经常有意见分歧"
    result = all_cut(sentence, Dict)  # 获取所有可能的切分结果
    for seg in result:  # 打印每一种切分结果
        print(seg)
