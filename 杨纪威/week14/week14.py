class BPE:
    def __init__(self, content_paths):
        self.content_paths = content_paths
        self.utf8_count = 256
        self.vocab_size = 356
        self.loop_num = self.vocab_size - self.utf8_count
        self.vocab = {idx: bytes([idx]) for idx in range(self.utf8_count)}
        self.merges = {}
        self.build_vocab()
    def get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    def merge(self, ids, pair, idx):
        newids = []
        index = 0
        while index < len(ids):
            if index < len(ids) and ids[index] == pair[0] and ids[index + 1] == pair[1]:
                newids.append(idx)
                index += 2
            else:
                newids.append(ids[index])
                index += 1
        return newids
    def build_vocab(self):
        ids = []
        for path in content_paths:
            tokens = open(path, encoding='utf8').read().encode('utf8')
            tokens = list(map(int, tokens))
            ids += tokens
        self.merges = {}
        # 构建字典
        for i in range(self.loop_num):
            stats = self.get_stats(ids)
            top_pair = max(stats, key=stats.get)
            idx = self.utf8_count + i
            ids = self.merge(ids, top_pair, idx)
            self.merges[top_pair] = idx
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens
    def decode(self, ids):
        # given ids (list of integers), return Python string
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text
if __name__ == '__main__':
    content_paths = ["../RAG/dota2英雄介绍-byRAG/Heroes/暗影恶魔.txt", "../RAG/dota2英雄介绍-byRAG/Heroes/大地之灵.txt"]
    bpe = BPE(content_paths)
    text = "他的碎片将触碰到的一切事物感染，而这种感染则逐渐滋生壮大"
    res = bpe.encode(text)
    print(bpe.decode(res))