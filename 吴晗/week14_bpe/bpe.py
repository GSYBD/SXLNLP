import os
import collections



class BPETokenizer():
    # 预留20个unused等token
    def __init__(self, vocab_size=276, file_path=r'E:\xiangmu\SXLNLP\吴晗\week14_bpe\Heroes'):
        self.vocab_size = vocab_size
        self.file_path = file_path
        self.num_merges = self.vocab_size - 256
        self.vocab = self.build_tokenizer()
    
    # 预处理源文本
    def process_file(self):
        self.all_text = ''
        for file in os.listdir(self.file_path):
            path = os.path.join(self.file_path, file)
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
                self.all_text += text + '\n'
        all_tokens = list(map(int, self.all_text.encode('utf8')))
        return all_tokens

    #按照bpe的思想，我们统计每个2元组出现次数
    def get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]): # Pythonic way to iterate consecutive elements
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(self, ids, pair, idx):
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

    def build_tokenizer(self):
        self.all_tokens = self.process_file()
        merges = {}
        ids = list(self.all_tokens)
        for i in range(self.num_merges):
            stats = self.get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            print(f"merging {pair} into a new token {idx}")
            ids = self.merge(ids, pair, idx)
            merges[pair] = idx
        
        print("tokens length:", len(self.all_tokens))
        print("ids length:", len(ids))
        print(f"compression ratio: {len(self.all_tokens) / len(ids):.2f}X")
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        
        return vocab


    def decode(self, ids):
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text


    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            pair = min(stats, key=lambda p: self.vocab.get(p, float("inf")))
            if pair not in self.vocab:
                break # nothing else can be merged
            idx = self.vocab[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens

if __name__ == '__main__':
    tokenizer = BPETokenizer(276, r'E:\xiangmu\SXLNLP\吴晗\week14_bpe\Heroes')
    text = r'技能1：腐朽, 技能描述：不朽尸王偷取区域内所有敌方英雄的力量，造成基础伤害并将敌人的力量据为己有。对非英雄单位造成%creep_damage_multiplier%倍伤害'
    ids = tokenizer.encode(text)
    print(ids)
    text = tokenizer.decode(ids)
    print(text)
