# -*- coding: utf-8 -*-
"""
author: Chris Hu
date: 2024/9/26
desc:
sample
"""


# Load the source text which need to be tokenized
def load_text(filepath):
    text = ""
    with open(filepath, "r", encoding="utf-8") as file:
        intro = file.read()
        text += intro
    return text


# Convert the text to a list of tokens
def get_tokens(text):
    tokens = text.encode("utf-8")
    tokens = list(map(int, tokens))
    return tokens


# Calculate the frequency of each pair of tokens
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


# Merge the pair of tokens into a new token
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


def merge_tokens(tokens, vocab_size=288):
    """
    :param tokens: text pair
    :param vocab_size: the desired final vocabulary size
    :return: final merged tokens
    """
    num_merges = vocab_size - 256
    ids = list(tokens)  # copy so we don't destroy the original list

    merges = {}  # (int, int) -> int
    for i in range(num_merges):
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)
        idx = 256 + i
        # print(f"merging {pair} into a new token {idx}")
        ids = merge(ids, pair, idx)
        merges[pair] = idx
    return merges


# Build the final vocabulary
def build_vocab(merges):
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]
    return vocab


# Decode the tokens back to the original text
def decode(ids, vocab):
    # given ids (list of integers), return Python string
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text


# Encode the text into tokens
def encode(text, merges):
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


if __name__ == "__main__":
    # Test build encode vocabulary
    vocab_size = 288
    tokens = load_text('./Heroes.txt')
    tokens = get_tokens(tokens)
    merges = merge_tokens(tokens, vocab_size)
    vocab = build_vocab(merges)
    print(vocab)

    # Test encode and decode for given text
    encode_tokens = encode("A Programm，仙�Introduction to Uni�e", merges)
    print(encode_tokens)
    print(decode(encode_tokens, vocab))
