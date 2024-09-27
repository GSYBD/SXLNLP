import os

def combine_txt_files(input_folder, output_file):
    combined_content = ''
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(input_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                combined_content += f.read()
    with open(output_file, 'w', encoding='utf-8') as output:
        output.write(combined_content)
    return combined_content

input_floder = 'E:\AI\课程资料\第十四周 大语言模型RAG\week14 大语言模型相关第四讲\RAG\dota2英雄介绍-byRAG\Heroes'
output_file = 'E:\AI\课程资料\第十四周 大语言模型RAG\combined_output.txt'

def get_tokens(text):
    tokens = text.encode("utf-8")
    return tokens

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):  # Pythonic way to iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts

tokens = get_tokens(combine_txt_files(input_floder, output_file))
stats = get_stats(tokens)
# print(sorted(((v,k) for k,v in stats.items()), reverse=True))

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

vocab_size = 500 # the desired final vocabulary size  超参数：预期的最终词表大小，根据实际情况自己设置，大的词表会需要大的embedding层
num_merges = vocab_size - 256
ids = list(tokens) # copy so we don't destroy the original list
merges = {} # (int, int) -> int
for i in range(num_merges):
  stats = get_stats(ids)
  pair = max(stats, key=stats.get)
  idx = 256 + i
  # print(f"merging {pair} into a new token {idx}")
  ids = merge(ids, pair, idx)
  merges[pair] = idx

# print("tokens length:", len(tokens))
# print("ids length:", len(ids))
# print(f"compression ratio: {len(tokens) / len(ids):.2f}X")

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
      break # nothing else can be merged
    idx = merges[pair]
    tokens = merge(tokens, pair, idx)
  return tokens

print(encode("在基迪岛上的浓酸密林中"))
print(decode([292, 469, 186, 266, 170, 229, 178, 155, 409, 259, 465, 147, 233, 133, 184, 310, 134, 364, 151, 346]))