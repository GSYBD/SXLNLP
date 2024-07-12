# 分词方法：基于词典的全切分，并进一步根据词频计算最终最合理的结果

import re
import time
import json


def load_word_freq_dict(path):
    """
    从文件中加载词频词典和前缀词典
    :param path: 文件路径
    :return: 词频词典和前缀词典
    """
    word_freq_dict = {}
    prefix_dict = {}
    with open(path, encoding="utf8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            word, freq = parts[0], int(parts[1])
            word_freq_dict[word] = freq
            for i in range(1, len(word)):
                if word[:i] not in prefix_dict:  # 不能用前缀覆盖词
                    prefix_dict[word[:i]] = 0  # 前缀
            prefix_dict[word] = 1  # 词
    return word_freq_dict, prefix_dict


def full_cut(string, word_freq_dict, prefix_dict):
    """
    对输入字符串进行全切分，并计算每种切分方式的得分
    :param string: 输入字符串
    :param word_freq_dict: 词频词典
    :param prefix_dict: 前缀词典
    :return: 所有切分结果及其得分，得分最高的切分结果，最高得分
    """
    result = []
    n = len(string)
    dp = [[] for _ in range(n + 1)]
    dp[0] = [([], 0)]  # (切分结果，得分)

    for i in range(n):
        if dp[i]:
            for j in range(i + 1, n + 1):
                word = string[i:j]
                if word in prefix_dict:
                    word_score = word_freq_dict.get(word, 1)  # 如果词不在词频词典中，默认得分为1
                    for split_list, score in dp[i]:
                        dp[j].append((split_list + [word], score + word_score))
                else:
                    word_score = 1
                    for split_list, score in dp[i]:
                        dp[j].append((split_list + [word], score + word_score))

    for split_list, score in dp[n]:
        result.append((split_list, score))

    if not result:
        raise ValueError("No segmentation result found. Please check your input string and dictionaries.")

    # 找到得分最高的切分结果
    best_cut, best_score = max(result, key=lambda x: x[1])
    return result, best_cut, best_score


def main(input_path, output_path):
    """
    主函数，加载输入文件，对每一行进行全切分并计算得分，输出结果到文件
    :param input_path: 输入文件路径
    :param output_path: 输出文件路径
    """
    word_freq_dict, prefix_dict = load_word_freq_dict("dict.txt")
    writer = open(output_path, "w", encoding="utf8")
    start_time = time.time()
    with open(input_path, encoding="utf8") as f:
        for line in f:
            results, best_cut, best_score = full_cut(line.strip(), word_freq_dict, prefix_dict)
            writer.write("所有切分结果及其得分:\n")
            for split_list, score in results:
                writer.write(f"{' / '.join(split_list)} : {score}\n")
            writer.write("\n最高得分的切分结果:\n")
            writer.write(f"{' / '.join(best_cut)} : {best_score}\n")
            writer.write("\n" + "="*50 + "\n")
    writer.close()
    print("耗时：", time.time() - start_time)
    return


string = "王羲之草书《平安帖》共有九行"
word_freq_dict, prefix_dict = load_word_freq_dict("dict.txt")
result, best_cut, best_score = full_cut(string, word_freq_dict, prefix_dict)
print("全切分结果共有%d种：" % (len(result)))
for split_list, score in result:
    print(split_list, score)
print("全切分的最佳结果：")
print(best_cut, best_score)
# 如果要运行整个流程，取消以下注释
# if __name__ == "__main__":
#     main("corpus.txt", "cut_method_output.txt")
