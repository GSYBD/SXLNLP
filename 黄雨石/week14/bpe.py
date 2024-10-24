from collections import defaultdict
import re

def get_stats(vocab):
    """计算字符对的频率"""
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    """根据最频繁的字符对更新词汇表"""
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

def bpe(vocab, num_merges=10):
    """执行BPE算法"""
    merges = []
    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        merges.append(best)
        vocab = merge_vocab(best, vocab)
        print(f"Merge {i+1}: {best}")
    return vocab, merges

# 示例词汇表
vocab = {
    'l o w </w>': 5,
    'l o w e r </w>': 2,
    'n e w e s t </w>': 6,
    'w i d e r </w>': 3
}

# 应用BPE
new_vocab, merges = bpe(vocab, num_merges=10)

# 构建子词单元到索引的映射
subword_to_index = {'<unk>': 0}  # 添加未知词标记
index = 1
for word in new_vocab.keys():
    subwords = word.split()
    for subword in subwords:
        if subword not in subword_to_index:
            subword_to_index[subword] = index
            index += 1

# 打印子词单元到索引的映射
print("Subword to Index Mapping:", subword_to_index)

# 编码函数
def encode(text, merges, subword_to_index):
    def split_word(word):
        word = ' '.join(list(word)) + ' </w>'
        while True:
            changed = False
            for pair in merges:
                bigram = ' '.join(pair)
                if bigram in word:
                    word = word.replace(bigram, ''.join(pair))
                    changed = True
            if not changed:
                break
        return word.split()

    encoded_text = []
    for word in text.split():
        subwords = split_word(word)
        encoded_subwords = [subword_to_index.get(subword, 0) for subword in subwords]
        encoded_text.append(encoded_subwords)
    return encoded_text

# 解码函数
def decode(encoded_text, index_to_subword):
    decoded_text = []
    for word in encoded_text:
        subwords = [index_to_subword[idx] for idx in word]
        decoded_word = ''.join(subwords).replace(' </w>', '')
        decoded_text.append(decoded_word)
    return ' '.join(decoded_text)

# 示例文本
text = "where there is a will,there is a way"

# 编码
encoded_text = encode(text, merges, subword_to_index)
print("Encoded Text (as indices):", encoded_text)

# 构建索引到子词单元的映射
index_to_subword = {v: k for k, v in subword_to_index.items()}

# 解码
decoded_text = decode(encoded_text, index_to_subword)
print("Decoded Text:", decoded_text)