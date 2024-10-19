import os

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

def decode(ids, vocab):
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text

def encode(text, merges):
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens


if __name__ == "__main__":
    dir_path = r"D:\JetBrainsProject\PyCharm\NLPLearn\week14\RAG\dota2英雄介绍-byRAG\Heroes"

    corpus = ""
    for path in os.listdir(dir_path):
        path = os.path.join(dir_path, path)
        with open(path, encoding="utf8") as f:
            text = f.read()
            corpus += text + '\n'
    # 建立词表
    vocab_size = 666
    num_merges = vocab_size - 256
    tokens = text.encode("utf-8")  # raw bytes
    tokens = list(map(int, tokens))
    ids = list(tokens)

    merges = {}
    for i in range(num_merges):
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)
        idx = 256 + i
        ids = merge(ids, pair, idx)
        merges[pair] = idx

    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]
        try:
            print(idx, vocab[idx].decode("utf8"))
        except UnicodeDecodeError:
            continue
    string = "亡灵影魔"
    encode_ids = encode(string, merges)
    print("编码结果：", encode_ids)
    decode_string = decode(encode_ids, vocab)
    print("解码结果：", decode_string)