import re
import time
import json
#提供的字表
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
# 将字表整理成前缀字典
def prefix_word_dict(Dict):
    prefix_dict = {}
    for name in Dict.keys():
        word = name
        for i in range(1, len(word)):
            if word[:i] not in prefix_dict:
                prefix_dict[word[:i]] = 0
        prefix_dict[word] =1
    return prefix_dict

#正向切割
def cut_method_forward(string, prefix_dict):
    if string == "":
        return [] #如果字典是空值，返回空列表
    words = []  # 准备用于放入切好的词
    start_index, end_index = 0, 1  #记录窗口的起始位置
    window = string[start_index:end_index] #从第一个字开始
    find_word = window  # 将第一个字先当做默认词
    while start_index < len(string):
        #窗口没有在词典里出现或者窗口尾端没有超过字符串长度执行循环
        if window not in prefix_dict or end_index > len(string):
            words.append(find_word)  #记录找到的词
            start_index += len(find_word)  #更新起点的位置
            print(start_index)
            end_index = start_index + 1
            window = string[start_index:end_index]  #从新的位置开始一个字一个字向后找
            find_word = window
        #窗口是一个词
        elif prefix_dict[window] == 1:
            find_word = window  #查找到了一个词，还要在看有没有比他更长的词
            end_index += 1
            window = string[start_index:end_index]
        #窗口是一个前缀
        elif prefix_dict[window] == 0:
            end_index += 1
            window = string[start_index:end_index]
    #最后找到的window如果不在词典里，把单独的字加入切词结果
    if prefix_dict.get(window) != 1:
        words += list(window)
    else:
        words.append(window)
    return words
#反向切割
def cut_method_backwards(string, prefix_dict):
    if string == "":
        return [] #如果字典是空值，返回空列表
    words = []  # 准备用于放入切好的词
    end_index,start_index = (len(string)-1), len(string)  #记录窗口的起始位置
    window = string[end_index:start_index] #从第一个字开始
    find_word = window  # 将第一个字先当做默认词
    while start_index > 0:
        #窗口没有在词典里出现或者窗口尾端没有超过字符串长度执行循环
        if window not in prefix_dict or end_index <0:
            words.insert(0,find_word)  #记录找到的词
            start_index -= len(find_word)#更新起点的位置
            print(start_index)
            end_index = start_index -1
            window = string[end_index:start_index]  #从新的位置开始一个字一个字向后找
            find_word = window
        #窗口是一个词
        elif prefix_dict[window] == 1:
            find_word = window  #查找到了一个词，还要在看有没有比他更长的词
            end_index -= 1
            window = string[end_index:start_index]
        #窗口是一个前缀
        elif prefix_dict[window] == 0:
            end_index -= 1
            window = string[end_index:start_index]
    #最后找到的window如果不在词典里，把单独的字加入切词结果
    if prefix_dict.get(window) != 1:
        words += list(window)
    else:
        words.insert(0,window)
    return words

prefix_dict= prefix_word_dict(Dict)
string="经常有意见分歧"
words1=cut_method_forward(string, prefix_dict)
words2=cut_method_backwards(string, prefix_dict)
print(prefix_dict)
print(words1)
print(words2)