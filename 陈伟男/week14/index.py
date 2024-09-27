
def merge(ids, key, id):
  new_ids = []
  i = 0
  while i < len(ids):
    if i < len(ids) - 1 and ids[i] == key[0] and ids[i+1] == key[1]:
      new_ids.append(id)
      i += 2
    else:
      new_ids.append(ids[i])
      i += 1
  return new_ids

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def decode(vocab, ids):
  tokens = b"".join(vocab[idx] for idx in ids)
  text = tokens.decode("utf-8", errors="replace")
  return text

def encode(text, merges):
  tokens = list(text.encode("utf-8"))
  while len(tokens) >= 2:
    stats = get_stats(tokens)
    print(stats)
    pair = min(stats, key=lambda p: merges.get(p, float("inf")))
    if pair not in merges:
      break 
    idx = merges[pair]
    tokens = merge(tokens, pair, idx)
  return tokens

def min():
  vocab_size = 286
  merges = {}
  ids = []
  with open('./corpus.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
      tokens = line.encode('utf-8')
      tokens = list(map(int, tokens))
      ids = ids + tokens
    print(ids)
  for i in range(vocab_size - 256):
    counts = get_stats(ids)
    max_count = max(counts, key=counts.get)
    merges[max_count] = i + 256
    ids = merge(ids, max_count, i + 256)

  vocab = {idx: bytes([idx]) for idx in range(256)}
  for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]
  
  print(decode(vocab, encode("穿梭在街头熙熙攘攘的人群中，李慕的表情有些恍惚。", merges)))
    

if __name__ == '__main__':
  min()
