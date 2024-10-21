import os

output_file = 'week14\\merge_h.txt'
with open(output_file, 'r', encoding='utf-8') as file:  
    content = file.read()  # 从文件对象中读取内容

text = content
tokens = text.encode("utf-8") # raw bytes
tokens = list(map(int, tokens)) # convert to a list of integers in range 0..255 for convenience
print('---')
print("length:", len(text))
print('---')
print("length:", len(tokens))


def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]): # Pythonic way to iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
  # in the list of ints (ids), replace all consecutive occurences of pair with the new token idx
  newids = []
  i = 0
  while i < len(ids):
    # if we are not at the very last position AND the pair matches, replace it
    if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
      newids.append(idx)
      i += 2
    else:
      newids.append(ids[i])
      i += 1
  return newids




vocab_size = 512 # the desired final vocabulary size  超参数：预期的最终词表大小，根据实际情况自己设置，大的词表会需要大的embedding层
num_merges = vocab_size - 256
ids = list(tokens) # copy so we don't destroy the original list

merges = {} # (int, int) -> int
for i in range(num_merges):
  stats = get_stats(ids)
  pair = max(stats, key=stats.get)
  idx = 256 + i
  print(f"merging {pair} into a new token {idx}")
  ids = merge(ids, pair, idx)
  merges[pair] = idx
print("tokens length:", len(tokens))
print("ids length:", len(ids))
print(f"compression ratio: {len(tokens) / len(ids):.2f}X")


vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]
print(vocab)

decoded_dict = {}
for key, value in vocab.items():
    try:
        # 解码字节串为字符串，假设编码为 UTF-8
        decoded_value = value.decode('utf-8')
        decoded_dict[key] = decoded_value
    except UnicodeDecodeError:
        print(f"无法解码键 {key} 的值 {value}")
# 打印解码后的字典内容
for key, value in decoded_dict.items():
    print(f"{key}: {value}")
    
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
      break # nothing else can be merged
    idx = merges[pair]
    tokens = merge(tokens, pair, idx)
    print("----------")
    print(pair)
    print(idx)
    print(tokens)
    print("----------")
  return tokens    

  
print(encode("背景故事：也许有人会问：“这个世界是如何形成的？”在所有现存世界中，为什么这个世界的属性如此奇特，如此多样化，其中的生物，文化和传说更是数不胜数呢？"))
print(decode([443, 140, 376, 175, 277, 133, 478, 270, 489, 425, 184, 320, 299, 352, 
              368, 174, 270, 414, 156, 400, 323, 257, 150, 502, 140, 334, 488, 130, 
              279, 149, 357, 162, 317, 259, 256, 159, 414, 157, 292, 485, 511, 407, 
              152, 257, 150, 502, 140, 346, 260, 373, 263, 128, 293, 136, 400, 323, 
              257, 150, 502, 140, 259, 229, 177, 158, 345, 167, 488, 130, 369, 164, 
              337, 135, 374, 185, 260, 488, 130, 369, 164, 276, 154, 328, 183, 427, 
              150, 260, 437, 346, 259, 463, 490, 260, 291, 135, 427, 150, 344, 281, 
              160, 382, 180, 230, 155, 180, 334, 277, 176, 347, 443, 156, 277, 176, 
              229, 145, 162, 256, 159]))


