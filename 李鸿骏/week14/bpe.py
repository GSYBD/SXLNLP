text = "hello world"
tokens = text.encode("utf-8") # raw bytes
tokens = list(map(int, tokens)) # convert to a list of integers in range 0..255 for convenience
print('---')
print(text)
print("length:", len(text))
print('---')
print(tokens)
print("length:", len(tokens))

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]): # Pythonic way to iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts

stats = get_stats(tokens)
print(sorted(((v,k) for k,v in stats.items()), reverse=True))

top_pair = max(stats, key=stats.get)
print(top_pair)

def merge(ids, pair, idx):
    # in the list of ints (ids), replace all consecutive occurences of pair with the new token idx
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

def encode_text(path, vocab_size):
    # 加载语料
    with open(path, "r", encoding="utf-8") as file:
        text = file.read()
    tokens = text.encode("utf-8")  # raw bytes
    num_merges = vocab_size - 256
    ids = list(tokens)  # copy so we don't destroy the original list
    merges = {}  # (int, int) -> int
    for i in range(num_merges):
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)
        idx = 256 + i
        print(f"merging {pair} into a new token {idx}")
        ids = merge(ids, pair, idx)
        merges[pair] = idx
    return ids, merges

def decode_text(ids, merges):
    # decode the ids back to text
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text

ids, merges =  encode_text("./data/heros.txt", 276)
print(decode_text(ids, merges))