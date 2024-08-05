from transformers import BertModel

model = BertModel.from_pretrained('bert-base-chinese', return_dict=False)

# 模型实际参数个数为24301056 = 768 × 31642
print('模型实际参数个数为%d' % sum(p.numel() for p in model.parameters()))
# diy计算参数个数为122899968 = 768 × 31642
print('diy计算参数个数为%d' % (768 * (21128 + 512 + 18 + 13 * 768)))

for name, param in model.named_parameters():
    print(name, param.shape)
