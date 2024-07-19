
"""
week4作业: 实现全切分函数，输出根据字典能够切分出的所有的切分方式
"""

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
"""
method1: 暴力法
设置一个窗口, 窗口长度为1-max(词典中关键词长度),针对每一个窗口长度
1. 如果能在词典中找到对应的词，对应的词加入结果列表中；待切分句子为该句子除去已切分好的词，然后对句子剩下的部分开始上述过程[窗口长度为1-max(词典中关键词长度)]，直到剩下要切分的句子长度为0
2. 如果在词典中找不到对应的词，那么窗口长度可以换其他的继续寻找，直到能找到字典中对应的词或者窗口长度已达到最大仍然找不到退出
"""
# def all_cut(sentence, Dict):
#     target = []
#     max_word_len = max([len(key) for key in Dict])
#     for i in range(0, min(max_word_len, len(sentence))):
#         result = []
#         first_word = sentence[0:i+1]
#         result.append(first_word)
#         leftSentence = sentence[i+1:]
#         words = mini_cut(leftSentence, Dict)
#         for word in words:
#             result.append(word)
#         target.append(result)
#
#     return target

# def mini_cut(leftSentence, Dict):
#     words = []
#     max_word_len = max([len(key) for key in Dict])
#     for i in range(0, max_word_len):
#         if i >= len(leftSentence):
#             continue
#         if leftSentence[0: i+1] in Dict.keys():
#             print("找到了词表里的关键词:[{}]".format(leftSentence[0:i+1]))
#             words.append(leftSentence[0:i+1])
#             leftSentence = leftSentence[i+1:]
#             print("当前已经分好的词列表为:{}, 剩下要分的词句为:{}".format(words, leftSentence))
#
#         if len(leftSentence) == 0:
#             print("剩余待切分词长度为0，当前这种切分方式可以结束了，切分的词列表为:{}".format(words))
#             return words
#
#     return words

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

def all_cut(sentence, Dict, index, path):
    if index == len(sentence):
        return [path]
    target = []
    for i in range (0, len(sentence)):
        word = sentence[index: i+1]
        if word in Dict:
            # path = path.append(word)
            subTarget = all_cut(sentence, Dict, i+1, path+[word])
            target.extend(subTarget)
    return target


if __name__ == '__main__':
    target = all_cut(sentence, Dict, index=0, path=[])
    print("全切分方式产生的所有分词列表为:{}".format(target))



# isnotEnd = True
    # max_word_len = max([len(key) for key in Dict])
    # max_epch_num = 100
    # current_num = 0
    # while isnotEnd:
    #     current_num += 1
    #     print("当前开始第{}种切分方式".format(current_num))
    #     leftSentence = sentence
    #     every_cut_segmentation_word_list = []
    #     for i in range(0, min(max_word_len, len(leftSentence))):
    #         left_index = 0
    #         right_index = i+1
    #         print("当前尝试从{}个词来分词".format(i+1))
    #         if sentence[left_index:right_index] in Dict.keys():
    #             print("找到了词表里的关键词:[{}]".format(sentence[left_index:right_index]))
    #             every_cut_segmentation_word_list.append(sentence[left_index:right_index])
    #             leftSentence = leftSentence[right_index:]
    #             print("当前已经分好的词列表为:{}, 剩下要分的词句为:{}".format(every_cut_segmentation_word_list, leftSentence))
    #         else:
    #             print("当前切分的词:[{}]不在词表中，开始下一轮循环，多增加一个词".format(sentence[left_index:right_index]))
    #
    #
    #         if len(leftSentence) == 0:
    #             print("剩余待切分词长度为0，当前这种切分方式可以结束了，切分的词列表为:{}".format(every_cut_segmentation_word_list))
    #             target.append(every_cut_segmentation_word_list)
    #             break
    #     # 如何判断所有的切分方式都已经遍历过了
    #     if current_num >= max_epch_num:
    #         isnotEnd = False