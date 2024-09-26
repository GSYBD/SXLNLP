import glob
import os
import json

#按照bpe的思想，我们统计每个2元组出现次数
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

def getTokens(dirpath):
    content = ""
    txt_files = glob.glob(os.path.join(dirpath, "*.txt"))
    for filepath in txt_files:
        with open(filepath, "r", encoding="utf-8") as file:
            content += file.read()
    tokens = content.encode("utf-8")  # raw bytes
    tokens = list(map(int, tokens))
    return tokens

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

def decode(ids):
  # given ids (list of integers), return Python string
  tokens = b"".join(vocab[idx] for idx in ids)
  text = tokens.decode("utf-8", errors="replace")
  return text


if __name__ == "__main__":
    dirpath = r"E:\Learn\nlp\week14 大语言模型相关第四讲\RAG\dota2英雄介绍-byRAG\Heroes"
    vocab_size = 1000
    num_merges = vocab_size - 256

    tokens = getTokens(dirpath)
    ids = list(tokens)

    merges = {} # (int, int) -> int
    for i in range(num_merges):
      stats = get_stats(ids)
      pair = max(stats, key=stats.get)
      idx = 256 + i
      # print(f"merging {pair} into a new token {idx}")
      ids = merge(ids, pair, idx)
      merges[pair] = idx


    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]


    print(decode([281, 144, 273, 168, 883, 363, 390, 572, 370, 927, 167, 773, 282, 862, 259, 805, 273, 168, 262]))
    #
    print(encode("伐木机对敌人造成伤害并摧毁他周围的树木。"))
