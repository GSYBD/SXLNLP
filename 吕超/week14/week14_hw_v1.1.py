# -*- encoding: utf-8 -*-
"""
week14_hw_v1.1.py
Created on 2024/9/28 11:10
@author: Allan Lyu
@version: v1.1 初始版
@Description: 实现bpe构建词表，并完成文本编码 及文本解码
"""
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config():
    """加载配置信息"""
    config = {}
    config['folder_path'] = os.getenv('FOLDER_PATH', "../RAG/dota2英雄介绍-byRAG/Heroes/")
    config['initial_vocab_size'] = os.getenv('INITIAL_VOCAB_SIZE', 256)
    config['construct_vocab_size'] = os.getenv('CONSTRUCT_VOCAB_SIZE', 500)
    return config


def load_data(folder_path):
    """读取文件夹中的所有txt文件，并将内容合并为一个字符串"""
    totol_text = ""
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            with open(os.path.join(folder_path, file_name), "r", encoding="utf-8") as file:
                intro = file.read()
                totol_text += intro
    return totol_text


def get_stats(ids):
    """统计id对出现的次数"""
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    """合并id对"""
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


def encode(text):
    """文本编码"""
    # given a string, return list of integers (the tokens)
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break  # nothing else can be merged
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens


def decode(ids):
    """文本解码"""
    # given ids (list of integers), return Python string
    vocab = {idx: bytes([idx]) for idx in range(config['initial_vocab_size'])}
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]

    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text


if __name__ == '__main__':
    # 1.加载配置信息
    config = load_config()

    # 2.加载数据
    # folder_path = "../RAG/dota2英雄介绍-byRAG/Heroes/"
    merge_text = load_data(config['folder_path'])
    # logger.info(f"合并后的前1000个字符：{totol_text[:1000]}")
    logger.info(f"训练文本长度为：{len(merge_text)}")

    # 3.构建词表
    vocab_size = config[
        "construct_vocab_size"]  # the desired final vocabulary size  超参数：预期的最终词表大小，根据实际情况自己设置，大的词表会需要大的embedding层
    num_merges = vocab_size - config['initial_vocab_size']

    tokens = merge_text.encode("utf-8")  # raw bytes
    tokens = list(map(int, tokens))  # convert to a list of integers in range 0..255 for convenience
    ids = list(tokens)  # copy so we don't destroy the original list
    merges = {}  # (int, int) -> int
    for i in range(num_merges):
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)
        idx = config['initial_vocab_size'] + i
        # logger.info(f"merging {pair} into a new token {idx}")
        ids = merge(ids, pair, idx)
        merges[pair] = idx
    # logger.info(f"扩充后序列：{ids}")
    logger.info(f"扩充后词表大小：{len(ids)}")

    # 4.保存词表
    with open("my_vocab.txt", "w", encoding="utf-8") as file:
        file.write(str(merges))
        logger.info(f"词表已保存到my_vocab.txt")
        logger.info(f"新增词表大小为：{len(merges)}")

    # 5.文本编码测试 及解码测试
    valtext = 'NLP的任务之一是文本编码，即将文本转换为计算机可以处理的数字序列。'
    encode_result = encode(valtext)
    decode_result = decode(encode(valtext))
    logger.info(f"原始测试文本：{valtext}")
    logger.info(f"编码后的序列：{encode_result}")
    logger.info(f"解码后的文本：{decode_result}")
    logger.info(f"原始文本与解码后的文本是否一致：{valtext == decode_result}")
