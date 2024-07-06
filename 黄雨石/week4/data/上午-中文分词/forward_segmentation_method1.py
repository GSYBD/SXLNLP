#分词方法：最大正向切分的第一种实现方式

import re
import time

#加载词典
def load_word_dict(path):
    max_word_length = 0  # 初始化最大单词长度为0
    word_dict = {}  # 初始化一个空的字典，用于存储词典中的单词
    with open(path, encoding="utf8") as f:  # 打开指定路径的词典文件，使用utf8编码
        for line in f:  # 逐行读取文件内容
            word = line.split()[0]  # 按空格分割每行内容，获取第一个单词
            word_dict[word] = 0  # 将单词加入字典，值为0
            max_word_length = max(max_word_length, len(word))  # 更新最大单词长度
    return word_dict, max_word_length  # 返回词典和最大单词长度


#先确定最大词长度
#从长向短查找是否有匹配的词
#找到后移动窗口
def cut_method1(string, word_dict, max_len):
    words = []
    while string != '':
        lens = min(max_len, len(string))
        word = string[:lens]
        while word not in word_dict:
            if len(word) == 1:
                break
            word = word[:len(word) - 1]
        words.append(word)
        string = string[len(word):]
    return words

#cut_method是切割函数
#output_path是输出路径
def main(cut_method, input_path, output_path):
    word_dict, max_word_length = load_word_dict("dict.txt")
    print("词字典,",word_dict)# 分出来的一个个单词，字典
    print("词长度",max_word_length)

    writer = open(output_path, "w", encoding="utf8")
    start_time = time.time()
    with open(input_path, encoding="utf8") as f:
        for line in f:
            words = cut_method(line.strip(), word_dict, max_word_length)
            writer.write(" / ".join(words) + "\n")
    writer.close()
    print("耗时：", time.time() - start_time)
    return


string = "测试字符串"
word_dict, max_len = load_word_dict("dict.txt")
# print(cut_method1(string, word_dict, max_len))

main(cut_method1, "corpus.txt", "cut_method1_output.txt")
