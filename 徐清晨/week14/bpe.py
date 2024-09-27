from itertools import count

from sympy.physics.units import coulombs

text = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."
tokens = text.encode("utf-8") # raw bytes
tokens = list(map(int, tokens)) # convert to a list of integers in range 0..255 for convenience
# print('---')
# print(tokens)

# 统计每个词组出现的次数 (239, 188)
# {(239, 188): 1, (188, 181): 1, (181, 239): 1, (239, 189): 6, (189, 142): 1,
# (142, 239): 1, (189, 137): 1, (137, 239): 1, (189, 131): 1, (131, 239): 1,
# (189, 143): 1, (143, 239): 1, (189, 132): 1, (132, 239): 1, (189, 133): 1,
# (133, 33): 1, (33, 32): 2, (32, 240): 3, (240, 159): 15, (159, 133): 7, (133, 164): 1,
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        # print(pair)
        # exit()
        counts[pair] = counts.get(pair, 0) + 1
    return counts

# stats = get_stats(tokens)
# # print(stats)
# top_pair = max(stats, key=stats.get)
# print(top_pair)

def merge(ids,pair,idx):
    """
    在ids中找到 pair，替换为idx
    也就是说，替换合并的元祖，变成一个新的数来替代

    :param ids:
    :param pair:
    :param idx:
    :return:
    """
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

print(merge([5, 6, 6, 7, 9, 1], (6, 7), 99))

# 最终词表大小
vocab_size = 276
num_merges = vocab_size - 256

ids = list(tokens)
print(ids)

merges = {}

"""
语料token化，找到最高的元祖，替换为数字
继续找最高的元祖，替换
"""
for i in range(num_merges):
    stats = get_stats(ids)
    pair = max(stats, key=stats.get)
    idx = 256+i
    print(f"merging {pair} into a new token {idx}")
    ids = merge(ids,pair,idx)
    merges[pair] = idx

print("tokens length:", len(tokens))
print("ids length:", len(ids))
print(f"compression ratio: {len(tokens) / len(ids):.2f}X")



# 解码
vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]

def decode(ids):
  # given ids (list of integers), return Python string
  tokens = b"".join(vocab[idx] for idx in ids)
  text = tokens.decode("utf-8", errors="replace")
  return text

# print(decode([65, 32, 80, 114, 111, 103, 114, 97, 109, 109, 260, 263, 153, 258, 73, 110, 116, 114, 111, 100, 117, 99, 116, 105, 111, 110, 32, 116, 111, 32, 85, 110, 105, 271, 101,]))

# 编码
print(merges)
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

# print(encode("A Programmer’s Introduction to Unicode"))

print(decode(encode("hello world")))


valtext = "Many common characters, including numerals, punctuation, and other symbols, are unified within the standard and are not treated as specific to any given writing system. Unicode encodes thousands of emoji, with the continued development thereof conducted by the Consortium as a part of the standard.[4] Moreover, the widespread adoption of Unicode was in large part responsible for the initial popularization of emoji outside of Japan. Unicode is ultimately capable of encoding more than 1.1 million characters."
valtext2 = decode(encode(valtext))
print(valtext2 == valtext)