from loader import DataGenerator
path = r'F:\1_Python\04_NLP\NLP\每周作业\week14_BPE\data_hero'
generator = DataGenerator(path)
content = generator.load()
content =list(content)
vocab_size = len(set(content))



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



# ---

# ---
vocab_size = vocab_size # the desired final vocabulary size  超参数：预期的最终词表大小，根据实际情况自己设置，大的词表会需要大的embedding层
num_merges = vocab_size - 2000
ids = content # copy so we don't destroy the original list

merges = {} # (int, int) -> int
for i in range(num_merges):
    stats = get_stats(ids)
    pair = max(stats, key=stats.get)
    idx = 256 + i
    print(f"merging {pair} into a new token {idx}")
    ids = merge(ids, pair, idx)
    merges[pair] = idx
  

print(merges)
