class bert_parameters:
    # 定义bert模型的基本参数
    vocab_size = 30522  # 词汇表大小
    hidden_size = 768  # 隐藏层大小
    max_position_embeddings = 512  # 最大位置嵌入数量
    type_vocab_size = 2  # 类型词汇表大小（例如：句子A和句子B）
    num_attention_heads = 12  # 注意力头数量
    intermediate_size = 3072  # 中间层大小
    num_hidden_layers = 12  # 隐藏层数量

    def __init__(self):
        # 初始化类，计算各层参数量并输出
        self.total_embeddings = self.embedding_parameters()  # 计算并获取嵌入层参数量
        self.total_attention = self.self_attention_parameters()  # 计算并获取注意力层参数量
        self.total_feed_forward = self.feed_forward_parameters()  # 计算并获取feed forward层参数量
        self.total_bert_parameters = self.calculate_total_parameters()  # 计算并获取总参数量

    def embedding_parameters(self):
        # 计算Embedding层参数
        token_embeddings = bert_parameters.vocab_size * bert_parameters.hidden_size  # 计算token嵌入参数量
        segment_embeddings = bert_parameters.type_vocab_size * bert_parameters.hidden_size  # 计算segment嵌入参数量
        position_embeddings = bert_parameters.max_position_embeddings * bert_parameters.hidden_size  # 计算position嵌入参数量
        total_embeddings = token_embeddings + segment_embeddings + position_embeddings  # 计算总嵌入参数量
        print(f"Embedding层参数量: {total_embeddings}")  # 输出嵌入层参数量
        return total_embeddings  # 返回嵌入层参数量

    def self_attention_parameters(self):
        # 计算Self-Attention层参数
        QKV_weight = 3 * (bert_parameters.hidden_size * bert_parameters.hidden_size)  # 计算QKV权重参数量
        QKV_bias = 3 * bert_parameters.hidden_size  # 计算QKV偏置参数量
        attention_output_weight = bert_parameters.hidden_size * bert_parameters.hidden_size  # 计算注意力输出权重参数量
        attention_output_bias = bert_parameters.hidden_size  # 计算注意力输出偏置参数量
        attention_layer_norm = 2 * bert_parameters.hidden_size  # 计算注意力层归一化参数量
        total_attention = QKV_weight + QKV_bias + attention_output_weight + attention_output_bias + attention_layer_norm  # 计算总注意力层参数量
        print(f"Self-Attention层参数量: {total_attention}")  # 输出注意力层参数量
        return total_attention  # 返回注意力层参数量

    def feed_forward_parameters(self):
        # 计算Feed Forward层参数
        feed_forward_weight_1 = bert_parameters.hidden_size * bert_parameters.intermediate_size  # 计算feed forward第一层权重参数量
        feed_forward_bias_1 = bert_parameters.intermediate_size  # 计算feed forward第一层偏置参数量
        feed_forward_weight_2 = bert_parameters.intermediate_size * bert_parameters.hidden_size  # 计算feed forward第二层权重参数量
        feed_forward_bias_2 = bert_parameters.hidden_size  # 计算feed forward第二层偏置参数量
        feed_forward_layer_norm = 2 * bert_parameters.hidden_size  # 计算feed forward层归一化参数量
        total_feed_forward = feed_forward_weight_1 + feed_forward_bias_1 + feed_forward_weight_2 + feed_forward_bias_2 + feed_forward_layer_norm  # 计算总feed forward层参数量
        print(f"Feed Forward层参数量: {total_feed_forward}")  # 输出feed forward层参数量
        return total_feed_forward  # 返回feed forward层参数量

    def calculate_total_parameters(self):
        # 计算总参数量
        total_bert_parameters = self.total_embeddings + bert_parameters.num_hidden_layers * (
                self.total_attention + self.total_feed_forward)  # 计算总参数量
        print(f"总参数量: {total_bert_parameters}")  # 输出总参数量
        return total_bert_parameters  # 返回总参数量


if __name__ == "__main__":
    bert_params = bert_parameters()  # 创建bert_parameters类的实例
