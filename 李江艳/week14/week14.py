from collections import Counter  

class SimpleBPE:  
    def __init__(self, vocab_size):  
        self.vocab_size = vocab_size  
        self.bpe_rules = {}  

    def fit(self, corpus):  
        """构建 BPE 词表，找到最常用的字符对"""  
        # 初始化词汇，添加 </w> 表示单词结束  
        words = [' '.join(list(word)) + ' </w>' for word in corpus]  
        current_vocab = words  

        while len(self.bpe_rules) < self.vocab_size:  
            pairs = Counter()  
            # 统计字符对出现的频率  
            for word in current_vocab:  
                symbols = word.split()  
                for i in range(len(symbols) - 1):  
                    pairs[(symbols[i], symbols[i + 1])] += 1  
            
            if not pairs:  
                break  
            
            # 找到频率最高的字符对  
            most_frequent_pair = pairs.most_common(1)[0][0]  

            # 合并字符对  
            new_word = ''.join(most_frequent_pair)  
            current_vocab = [word.replace(' '.join(most_frequent_pair), new_word) for word in current_vocab]  

            # 记录这个合并规则  
            self.bpe_rules[most_frequent_pair] = new_word  

        return self.bpe_rules  

    def encode(self, text):  
        """使用 BPE 规则编码文本"""  
        words = text.split()  
        encoded_words = []  
        for word in words:  
            for pair, new_word in self.bpe_rules.items():  
                word = word.replace(''.join(pair), new_word)  
            encoded_words.append(word)  
        return ' '.join(encoded_words)  

# 示例使用  
corpus = ["low", "lower", "newer", "wider"]  
bpe = SimpleBPE(vocab_size=10)  # 设定词表大小  
bpe.fit(corpus)  

print("BPE 规则:")  
for pair, new_word in bpe.bpe_rules.items():  
    print(f"{pair} -> {new_word}")  

# 编码文本  
text = "newer wider"  
encoded_text = bpe.encode(text)  
print(f"编码后的文本: {encoded_text}") 
