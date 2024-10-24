import os


#使用BPE构建词表

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

def encode(text,merges):
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


def decode(ids,vocab):
  # given ids (list of integers), return Python string
  tokens = b"".join(vocab[idx] for idx in ids)
  text = tokens.decode("utf-8", errors="replace")
  return text


if __name__ == "__main__":
    combined_text = ""
    folder_path="Heroes"
    for filename in os.listdir(folder_path):
        # 检查文件是否是文本文件（这里我们假设文本文件有以下几种常见的后缀）
        if filename.lower().endswith(('.txt', '.text')):
            # 拼接完整的文件路径
            file_path = os.path.join(folder_path, filename)
            # 打开文件并读取内容
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                # 将读取的内容添加到combined_text字符串中
                combined_text += content
    text1=combined_text.replace(" ","")#去除空格
    text=text1.replace("\n","")#去除换行符
    tokens = text.encode("utf-8")  # raw bytes
    vocab_size=500
    num_merges = vocab_size - 256
    ids = list(tokens)  # copy so we don't destroy the original list
    merges = {}  # (int, int) -> int
    for i in range(num_merges):
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)
        idx = 256 + i
        print(f"merging {pair} into a new token {idx}")
        ids = merge(ids, pair, idx)#将形成的词替换原来的词表中的单字
        merges[pair] = idx

    print(merges)
    raw_text="随着时间一年年过去，原型机失败作的残骸越堆越多，他甚至开始怀疑依靠机械飞行是否真的可能。在退休后第十年的第一天，一个阳光明媚、南风微拂的午后。"
    encode_text=encode(raw_text,merges)
    print(f"编码：{encode_text}")
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]
    decode_text=decode(encode_text,vocab)
    print(f"解码：{decode_text}")
