from transformers import BertModel

'''

第6周作业：计算bert中的可训练参数数量

'''

bert = BertModel.from_pretrained(r"E:\八斗学院AI课程\NLP\八斗精品班\第六周 预训练模型\bert-base-chinese\bert-base-chinese", return_dict=False)
state_dict = bert.state_dict()

#embedding层可训练参数数量
n_eb_te_w = bert.config.vocab_size * bert.config.hidden_size #Token embedding weight number
n_eb_se_w = bert.config.type_vocab_size * bert.config.hidden_size #Segment embedding weight number
n_eb_pe_w = bert.config.max_position_embeddings * bert.config.hidden_size #Segment embedding weight number
n_eb_LN_w = bert.config.hidden_size #embedding LayerNorm weight number
n_eb_LN_b = bert.config.hidden_size #embedding LayerNorm bias number
n_eb = n_eb_te_w + n_eb_se_w + n_eb_pe_w + n_eb_LN_w + n_eb_LN_b #embedding层可训练参数数量总和
#1层self-attention层参数数量
n_sa_Q_w = bert.config.hidden_size ** 2 #self-attention query weight number
n_sa_Q_b = bert.config.hidden_size #self-attention query bias number
n_sa_K_w = bert.config.hidden_size ** 2 #self-attention key weight number
n_sa_K_b = bert.config.hidden_size #self-attention key bias number
n_sa_V_w = bert.config.hidden_size ** 2 #self-attention value weight number
n_sa_V_b = bert.config.hidden_size #self-attention value bias number
n_sa_ot_ds_w = bert.config.hidden_size ** 2 #self-attention output dense weight number
n_sa_ot_ds_b = bert.config.hidden_size #self-attention output dense weight number
n_sa_ot_LN_w = bert.config.hidden_size #self-attention output LayerNorm weight number
n_sa_ot_LN_b = bert.config.hidden_size #self-attention output LayerNorm bias number
n_sa = n_sa_Q_w + n_sa_Q_b + n_sa_K_w + n_sa_K_b + n_sa_V_w + n_sa_V_b + \
       n_sa_ot_ds_w + n_sa_ot_ds_b + n_sa_ot_LN_w + n_sa_ot_LN_b #self-attention层可训练参数数量总和
#1层feed forward层参数数量
n_ff_im_ds_w = bert.config.intermediate_size * bert.config.hidden_size #feed forward intermediate dense weight number
n_ff_im_ds_b = bert.config.intermediate_size #feed forward intermediate dense bias number
n_ff_ot_ds_w = bert.config.hidden_size * bert.config.intermediate_size #feed forward output dense weight number
n_ff_ot_ds_b = bert.config.hidden_size #feed forward output dense bias number
n_ff_ot_LN_w = bert.config.hidden_size #feed forward output LayerNorm weight number
n_ff_ot_LN_b = bert.config.hidden_size #feed forward output LayerNorm bias number
n_ff = n_ff_im_ds_w + n_ff_im_ds_b + n_ff_ot_ds_w + n_ff_ot_ds_b + \
       n_ff_ot_LN_w + n_ff_ot_LN_b #feed forward层可训练参数数量总和
#pooler层参数数量
n_pl_ds_w = bert.config.hidden_size ** 2 #pooler dense weight number
n_pl_ds_b = bert.config.hidden_size #pooler dense bias number
n_pl = n_pl_ds_w + n_pl_ds_b #pooler层可训练参数数量总和
#bert中的可训练参数总数量
num_calc = n_eb + bert.config.num_hidden_layers * (n_sa + n_ff) + n_pl
print("计算结果是：", num_calc)

#直接输出答案
num_ans = 0
for k, v in state_dict.items():
    # print(k, v.shape)
    # print(v.numpy().size)
    num_ans += v.numpy().size
print("正确答案是：", num_ans)