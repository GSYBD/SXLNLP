import os

path = "D:/ai/week14 大语言模型相关第四讲(0922)/RAG/dota2英雄介绍-byRAG/Heroes"


def merge_file_one():
    i = 0
    for child_path in os.listdir(path):
        corpus_path = os.path.join(path, child_path)
        text = open(corpus_path, encoding="utf8").read()
        if i == 0:
            # print(text)
            # i += 1
            writer = open(os.path.join(path, "merge.txt"), "a", encoding="utf8")
            writer.write(text)
            writer.close()


def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
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

'''
使用文本训练得到词典
'''
def train_vocab(text):
    vocab_size = 276  # the desired final vocabulary size  超参数：预期的最终词表大小，根据实际情况自己设置，大的词表会需要大的embedding层
    num_merges = vocab_size - 256
    # tokens = list(map(int, tokens))
    ids = list(text.encode("UTF-8"))  # copy so we don't destroy the original list

    merges = {}
    for i in range(num_merges):
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)
        idx = 256 + i
        print(f"merging {pair} into a new token {idx}")
        ids = merge(ids, pair, idx)
        merges[pair] = idx
    return merges


'''
使用merges字典进行压缩
'''
def encode(merges):
    text = "灵体游魂可以控制，拥有回音重踏，回归和自然秩序三个技能。"
    # text = open(os.path.join(path, "merge.txt"), encoding="utf8").read()
    tokens = list(text.encode("UTF-8"))
    print('原本,', tokens)
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break  # nothing else can be merged
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    print("bpe之后:,", tokens)
    return tokens

'''
先根据merges，构建词表，在使用join 替换，这里可以一次性替换，感觉很巧妙
'''
def decode(ids,merges):
  # given ids (list of integers), return Python string
  vocab = {idx: bytes([idx]) for idx in range(256)}
  for (p0, p1), idx in merges.items():
      vocab[idx] = vocab[p0] + vocab[p1]

  tokens = b"".join(vocab[idx] for idx in ids)
  text = tokens.decode("utf-8", errors="replace")
  return text

if __name__ == '__main__':
    # merge_file_one()
    text = open(os.path.join(path, "merge.txt"), encoding="utf8").read()
    merges = train_vocab(text)
    print(merges)
    encode_tokens=encode(merges)
    print(">>>>>decode")
    decode_text = decode(encode_tokens,merges)
    print(decode_text)
