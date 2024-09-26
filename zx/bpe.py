import numpy as np
import os
import jieba

def load_hero_data(folder_path):
    hero_data = ""
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            with open(os.path.join(folder_path, file_name), "r", encoding="utf-8") as file:
                intro = file.read()
                hero_data += intro
    return hero_data


def get_tokens(text):
    tokens = text.encode("utf-8")
    tokens = list(map(int, tokens))
    print('---')
    # print(text)
    print("length:", len(text))
    print('---')
    # print(tokens)
    print("length:", len(tokens))
    return tokens
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
  newids = []
  i = 0
  while i < len(ids):
    if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
      newids.append(idx)
      i += 2
    else:
      newids.append(ids[i])
      i += 1
  return newids

# the desired final vocabulary size  超参数：预期的最终词表大小，根据实际情况自己设置，大的词表会需要大的embedding层
def merages_vocal(tokens,vocab_size=300):
    num_merges = vocab_size - 256
    ids = list(tokens)  # copy so we don't destroy the original list

    merges = {}  # (int, int) -> int
    for i in range(num_merges):
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)
        count = stats[pair]
        idx = 256 + i
        # print(f"merging {pair} into a new token {idx},count: {count}")
        ids = merge(ids, pair, idx)
        merges[pair] = idx
    return merges

#建立词表
def build_vocab(merges):
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]
    return vocab


def decode(ids,vocab):
  # given ids (list of integers), return Python string
  tokens = b"".join(vocab[idx] for idx in ids)
  text = tokens.decode("utf-8", errors="replace")
  return text

def encode(text):
  # given a string, return list of integers (the tokens)
  tokens = list(text.encode("utf-8"))
  while len(tokens) >= 2:
    stats = get_stats(tokens)
    pair = min(stats, key=lambda p: merges.get(p, float("inf")))
    if pair not in merges:
      break # nothing else can be merged
    idx = merges[pair]
    tokens = merge(tokens, pair, idx)
  return tokens

if __name__ == "__main__":
   folder_path  = r"../RAG/dota2英雄介绍-byRAG/Heroes"
   vocab_size = 300
   tokens = load_hero_data(folder_path)
   # print(tokens)
   tokens = get_tokens(tokens)
   merges = merages_vocal(tokens,vocab_size)
   # print(merges)
   vocab = build_vocab(merges)
   print(vocab)
   print(decode(
       [65, 32, 80, 114, 111, 103, 114, 97, 109, 109, 260, 263, 153, 258, 73, 110, 116, 114, 111, 100, 117, 99, 116,
        105, 111, 110, 32, 116, 111, 32, 85, 110, 105, 271, 101, ], vocab))
   print(encode("A Programm，仙�Introduction to Uni�e"))