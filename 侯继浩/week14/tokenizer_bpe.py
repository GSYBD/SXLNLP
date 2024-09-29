import os
import glob

def merge_txt_files(directory, output_file):
    # 获取目录下所有的txt文件
    txt_files = glob.glob(os.path.join(directory, '*.txt'))

    # 打开输出文件
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # 遍历每个txt文件
        for txt_file in txt_files:
            # 打开并读取每个txt文件
            with open(txt_file, 'r', encoding='utf-8') as infile:
                # 将文件内容写入输出文件
                outfile.write(infile.read())
                # 添加一个换行符，以便分隔不同的文件内容
                outfile.write('\n')

directory = "./dota2英雄介绍-byRAG/Heroes" # 替换为你的文件夹路径
output_file = './all.txt'  # 合并后的输出文件名
# merge_txt_files(directory, output_file)

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    # in the list of ints (ids), replace all consecutive occurences of pair with the new token idx
    newids = []
    i = 0
    while i < len(ids):
        # if we are not at the very last position AND the pair matches, replace it
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

with open(output_file, 'r', encoding='utf-8') as f:
    text = f.read()
    tokens = text.encode('utf-8')

vocab_size = 512  # the desired final vocabulary size  超参数：预期的最终词表大小，根据实际情况自己设置，大的词表会需要大的embedding层
num_merges = vocab_size - 256
ids = list(map(int, tokens)) # copy so we don't destroy the original list

merges = {} # (int, int) -> int
for i in range(num_merges):
  stats = get_stats(ids)
  pair = max(stats, key=stats.get)
  idx = 256 + i
  print(f"merging {pair} into a new token {idx}")
  ids = merge(ids, pair, idx)
  merges[pair] = idx

vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]

def decode(ids):
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
            break  # nothing else can be merged
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens

str = '暗夜魔王遮住太阳'
print(encode(str))
print(decode(encode(str)))




