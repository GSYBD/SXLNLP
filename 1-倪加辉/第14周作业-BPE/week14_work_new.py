"""
使用BEP构建自己的词表
"""
import os
import random

import jieba


class BpeDemo:
    def __init__(self, folder_path="Heroes"):
        self.vocab_size = 300
        self.load_hero_data(folder_path)



    def load_hero_data(self, folder_path):
        self.hero_data = {}
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".txt"):
                with open(os.path.join(folder_path, file_name), "r", encoding="utf-8") as file:
                    intro = file.read()
                    hero = file_name.split(".")[0]
                    self.hero_data[hero] = intro
        # corpus = {}
        # self.index_to_name = {}
        # index = 0
        # for hero, intro in self.hero_data.items():
        #     corpus[hero] = jieba.lcut(intro)
        #     self.index_to_name[index] = hero
        #     index += 1
        self.build_vocab(self.hero_data)
        return
    # 把文本转化为utf-8的编码
    def build_vocab(self, hero_data):
        self.all_tokens =[]
        # 记录二元组的对应下标
        self.merges = {} # (int, int) -> int
        for hero, intro in hero_data.items():
            text = intro
            if text:
                text = text.lower()
            tokens = text.encode("utf-8")  # raw bytes
            # 转化为utf-8编码
            tokens = list(map(int, tokens))
            # 合并为all_tokens
            self.all_tokens.extend(tokens)


        # print(sorted(((v, k) for k, v in self.counts.items()), reverse=True))
        # print(len(self.counts))
        # top_pair = max(self.counts, key=self.counts.get)
        # print(top_pair)
        # merge
        for i in range(self.vocab_size-256):
            # 统计词频
            counts = self.get_stats(self.all_tokens)
            # 找出最大的 先merge
            top_pair = max(counts, key=counts.get)
            idx = 256 + i
            print(f"merging {top_pair} into a new token {idx}")
            self.all_tokens = self.merge(self.all_tokens, top_pair, idx)
            # 记录变动
            self.merges[top_pair] = idx
        # 构建词表 utf-8 字节编码
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        ids = []
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
            ids.append(idx)



    # 解码
    def decode(self,ids):
        # given ids (list of integers), return Python string
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text

    # 编码
    def encode(self,text):
        # given a string, return list of integers (the tokens)
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # nothing else can be merged
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens

    # 统计二元组词频
    def get_stats(self,tokens):
        counts = {}
        for pair in zip(tokens, tokens[1:]):  # Pythonic way to iterate consecutive elements
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    # 把二元组用新的 idx代替 X = ab
    def merge(self,tokens, pair, idx):
        # in the list of ints (ids), replace all consecutive occurences of pair with the new token idx
        newids = []
        i = 0
        while i < len(tokens):
            # if we are not at the very last position AND the pair matches, replace it
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(tokens[i])
                i += 1
        return newids


if __name__ == '__main__':
    bpe_demo = BpeDemo()

    print(bpe_demo.decode(bpe_demo.encode("英雄名：")))

