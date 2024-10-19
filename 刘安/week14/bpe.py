vocab_size = 2760  # the desired final vocabulary size  超参数：预期的最终词表大小，根据实际情况自己设置，大的词表会需要大的embedding层
num_merges = vocab_size - 256
merges = {}


def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):  # Pythonic way to iterate consecutive elements
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


def decode(ids, vocab):
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


def main(text):
    ## 按照utf-8编码，将输入字符串转化为编码
    tokens = text.encode("utf-8")  # copy so we don't destroy the original list
    tokens = list(map(int, tokens))
    ids = list(tokens)
    for i in range(num_merges):
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)
        idx = 256 + i
        print(f"merging {pair} into a new token {idx}")
        ids = merge(ids, pair, idx)
        merges[pair] = idx
    # print(encode(tokens))
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]
    print(vocab.__sizeof__())
    return vocab
    # print(merges)
    # print(vocab)


if __name__ == "__main__":
    with open('testText.txt', 'r', encoding='UTF-8') as file:
        content = file.read()
        vocab = main(content)
    ids = [1227, 256, 87, 371, 116, 433, 448, 559, 312, 115, 111, 333, 811, 325, 376, 363, 101, 257, 644, 269, 46]
    print(decode(ids, vocab))
    # print(encode("Die Weltproletarier sollten sich vereinigen."))
