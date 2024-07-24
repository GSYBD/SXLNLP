from transformers import BertModel

# bertModel = BertModel.from_pretrained(r"F:\课程\八斗精品班\第六周 预训练模型\bert-base-chinese", return_dict=False)
# total = sum(p.numel() for p in bertModel.parameters())
# print("total param:", total)  # 102267648

class Bert():
    def __init__(self):
        self.vocab_size = 21128
        self.hidden_size = 768
        self.max_length = 512
        self.num_hidden_layers = 12
        self.num_attention_heads = 12
        self.pooler_fc_size = 768
        self.pooler_num_fc_layers = 3

    # 计算总共的参数量，分为embedding层，hidden层，pooler层
    def cal_all(self):
        p_embedding = self.cal_embedding()
        p_hidden = self.cal_hidden()
        p_pooler = self.cal_pooler()

        return p_embedding + p_hidden + p_pooler
    
    # 计算embedding层参数量，包括word/position/sentence/layer_norm四部分
    def cal_embedding(self):
        word_embedding = self.vocab_size * self.hidden_size
        position_embedding = self.max_length * self.hidden_size
        sentence_embedding = 2 * self.hidden_size
        layer_norm = self.cal_layer_norm()

        return word_embedding + position_embedding + sentence_embedding + layer_norm
    
    # 计算layer_norm层参数量，包含alpha, beta两个可训练参数
    def cal_layer_norm(self):
        alpha = 1 * self.hidden_size
        beta = 1 * self.hidden_size

        return alpha + beta
    
    # 计算隐藏层参数量，包含self_attention, feed_forward两部分，最后还要*隐藏层数num_hidden_layers
    def cal_hidden(self):
        self_attention = self.cal_self_attention()
        feed_forwward = self.cal_feed_forward()

        return (self_attention + feed_forwward) * self.num_hidden_layers
    
    # 计算self_attention层参数量，包含multi_head/layer_norm两部分
    def cal_self_attention(self):
        multi_head = self.cal_multi_head()
        layer_norm = self.cal_layer_norm()

        return multi_head + layer_norm
    
    # 计算multi_head层参数量，包含query/key/value/linear/layer_norm四部分
    def cal_multi_head(self):
        single_head = self.hidden_size / self.num_attention_heads
        bias = self.hidden_size # 每个qkv矩阵最后还有个bias偏置
        qkv = self.hidden_size * single_head * self.num_attention_heads + bias
        linear = self.hidden_size * self.hidden_size + bias # qkv变化后的结果要concat，经过线性层变换

        return qkv * 3 + linear

    # 计算feed_forward层参数量，包含fc/layer_norm两部分
    def cal_feed_forward(self):
        fc = self.cal_fc()
        layer_norm = self.cal_layer_norm()

        return fc + layer_norm
    
    # 两层fc = gelu(w1x + b1)w2 + b2
    def cal_fc(self):
        bias = self.hidden_size
        first = self.hidden_size * self.hidden_size * 4 + bias * 4
        second = self.hidden_size * self.hidden_size * 4 + bias

        return first + second
    
    # 计算pooler层参数量，获取[cls]标签的向量，wx+b
    def cal_pooler(self):
        return self.pooler_fc_size * self.hidden_size + self.pooler_fc_size

bert = Bert()
print(bert.cal_all())