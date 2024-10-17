# @Version  : 1.0
# @Author   : acyang
# @File     : homework.py
# @Time     : 2024/10/12 17:19
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

def get_vocab(text):
    tokens = text.encode("utf-8")  # raw bytes
    tokens = list(map(int, tokens))
    print(len(tokens))
    vocab_size = 276
    merges = {}
    num_merges = vocab_size - 256
    for i in range(num_merges):
        stats = get_stats(tokens)
        pair = max(stats, key=stats.get)
        idx = 256 + i
        print(f"merging {pair} into a new token {idx}")
        tokens = merge(tokens, pair, idx)
        merges[pair] = idx
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]
    return vocab, merges

def decode(ids, vocab):
  # given ids (list of integers), return Python string
  tokens = b"".join(vocab[idx] for idx in ids)
  text = tokens.decode("utf-8", errors="replace")
  return text

def encode(text, merges):
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


if __name__ == '__main__':
    path = r"../RAG/dota2英雄介绍-byRAG/Heroes/上古巨神.txt"
    with open(path, encoding='utf-8') as f:
        text = f.read()
        print(text)
        vocab, merges = get_vocab(text)
        text1 = decode(encode(text, merges),vocab)
        print(text == text1)
