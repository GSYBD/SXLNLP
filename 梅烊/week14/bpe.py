

import os
from collections import defaultdict


class bpe:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.vocab = defaultdict()
        self.count = 256

    def __build_corpus(self,path):
        content  = []
        for filename in os.listdir(path):
            if os.path.isfile(os.path.join(path, filename)):
                with open(os.path.join(path, filename), "r", encoding="utf-8") as f:
                        file_content = f.read()
                        content+= list(file_content)
        return content

    def __corpus_encode(self,corpus, encoding="utf-8"):
        #  corpus中的字符转utf-8编码
        tokens = "".join(corpus).encode(encoding) # raw bytes
        tokens = list(map(int, tokens))
        return tokens

    # 获取最大词频的二元组
    def __bpe_encode(self,ids):
        bpe_codes = defaultdict()   
        for decimal_tuple in zip(ids, ids[1:]):        
            bpe_codes[decimal_tuple] = bpe_codes.get(decimal_tuple, 0) + 1
        bpe_codes = sorted(((v,k) for k,v in bpe_codes.items()), reverse=True)
        # 返回的格式是 词频，二元组
        return bpe_codes


    # 合并最大词频的二元组， 并将合并后的二元组加入到vocab中
    def __merge(self,tokens, top_pair):
        mergeFlag = False
        new_tokens = []
        for i in range(len(tokens)-1)  :
            if  i < len(tokens)-1 :
                if tokens[i] == top_pair[0] and tokens[i+1] == top_pair[1]:
                    new_tokens.append(self.count)
                    # self.vocab[self.count] = self.vocab[top_pair[0]] , self.vocab[top_pair[1]]
                    # self.count += 1
                    mergeFlag = True
                    tokens.pop(i+1)
                else:
                    new_tokens.append(tokens[i])
        return new_tokens,mergeFlag

    def generate_vocab(self,path):
        # 对语料库进行utf-8编码
        corpus = self.__build_corpus(path)
        ids =self.__corpus_encode(corpus, encoding="utf-8")

        # 构建词表
        while self.count < self.vocab_size and len(ids) > 2:
            bpe_codes = self.__bpe_encode(ids)
            top_pair = bpe_codes[0][1]
            # 如果词频为1，则没有可以合并的词了，则退出循环
            if bpe_codes[0][0] == 1:
                break
            ids_flag = self.__merge(ids, top_pair)
            ids = ids_flag[0]
            mergeFlag = ids_flag[1]
            # 如果没有发生合并，则退出循环
            if not mergeFlag:
                break 
            self.vocab[self.count] = (top_pair[0] ,top_pair[1])
            self.count += 1
        return self.vocab


    def encode(self,sentence):
        # 对sentence进行utf-8编码
        tokens =self.__corpus_encode(sentence) 

        # 使用vocab对tokens进行bpe编码
        # 对vocab中的k,v进行对调
        vocab_t = {v: k for k, v in self.vocab.items()}
        # 对vocab中的每个二元组进行查找，找到则替换为对应的数字
        # 然后删除原来的二元组
        # 然后继续查找，直到找不到为止
        # 然后返回tokens    
        for i in range(len(tokens)-2):
            if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) in vocab_t:
                tokens[i] = vocab_t[(tokens[i], tokens[i+1])]
                tokens.pop(i+1)
        return tokens

    # 对tokens进行bpe解码
    def decode(self,tokens):
        # 对vocab中的每个二元组进行查找，找到则替换为对应的数字
        # 然后删除原来的二元组
        # 然后继续查找，直到找不到为止
        # 然后返回tokens
        while max(tokens) >= 256:
            for i in range(len(tokens)-1):
                if tokens[i] in self.vocab.keys():
                    new_tokens = tokens[:i]
                    new_tokens += list(self.vocab[tokens[i]])
                    new_tokens += tokens[i+1:]
                    tokens = new_tokens
                    
        # 将tokens按utf-8编码转成字符串
        tokens = bytes(tokens).decode("utf-8", errors="strict")        
        return tokens
    

     
if __name__ == "__main__":
    bpe = bpe(3000)
    # corpus = bpe.generate_vocab(r"..\RAG\dota2英雄介绍-byRAG\Heroes")
    corpus = bpe.generate_vocab(r".\test")
    ids = bpe.encode("我是一个人我wo")
    print(ids)
    print(bpe.decode(ids))