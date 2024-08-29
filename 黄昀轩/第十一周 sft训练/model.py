import json
import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

"""
问模型训练
"""


class SFTModel(nn.Module):
	def __init__(self, vocab_size):
		super(SFTModel, self).__init__()
		self.bert = BertModel.from_pretrained('../bert-base-chinese', return_dict=False)
		self.classify = nn.Linear(768, vocab_size)
		self.loss = nn.functional.cross_entropy
	
	def forward(self, x, mask=None, y=None):
		if y is not None:
			# 训练时，构建一个下三角的mask矩阵，让上下文之间没有交互
			x, _ = self.bert(x, attention_mask=mask)
			y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
			return self.loss(y_pred, y)
		else:
			x, _ = self.bert(x, attention_mask=mask)
			y_pred = self.classify(x)
			return y_pred
			
			


# 加载文件
def loader(data_path, window_size):
	sentences = []
	
	with (open(data_path, encoding='utf8') as f):
		for line in f:
			# print(type(line))#如果不loads的话 载入的就是字符串
			item = json.loads(line)  # loads后就是字典格式
			s1 = item['title'].strip()
			s2 = item['content'][:(window_size - len(s1))].strip()
			assert ((len(s1) + len(s2)) == window_size)
			sentences.append((s1, s2))  # 传入合并后的句子
		return sentences


# 建立样本
def build_sample(sentence, tokenizer, window_size):
	s1, s2 = sentence
	len_s1 = len(tokenizer.tokenize(s1))
	input_ids = tokenizer.encode(s1, s2, add_special_tokens=False, padding='max_length', max_length=window_size)
	y = [-100] * len_s1 + tokenizer.encode(s2, add_special_tokens=False, padding=False)
	y = y + [0] * (window_size - len(y))
	
	mask = torch.tril(torch.ones(window_size, window_size),diagonal=(len_s1-1))
	# print(mask)
	assert len(input_ids) == len(y)
	return input_ids, mask, y


# 建立数据集
def build_dataSet(sub_sentences, tokenizer, window_size):
	dataset_x = []
	data_mask = []
	dataset_y = []
	for sentence in sub_sentences:
		input_ids, mask, y = build_sample(sentence, tokenizer, window_size)
		dataset_x.append(input_ids)
		data_mask.append(mask)
		dataset_y.append(y)
		
		x = torch.LongTensor(dataset_x)
		mask = torch.stack(data_mask,dim=0) #用stack命令把mask 拼起来
		y = torch.LongTensor(dataset_y)
	return x, mask, y
def predict_sentence(sentence,model, tokenizer, window_size):
	model.eval()
	with torch.no_grad():
		x = tokenizer.encode(sentence, add_special_tokens=False, padding='max_length', max_length=window_size)
		x = torch.LongTensor(x)
		if torch.cuda.is_available():
			x = x.cuda()
		y = model(x)[0][-1]
		index =torch.argmax(y)
		pred_char = ''.join(tokenizer.decode(index))
		return pred_char

def main():
	"config"
	data_path = 'sample_data.json'
	window_size = 50
	total_train = 104  # 暂时手动获取
	epoch_nums = 10
	batch_size = 10
	vocab_size = 21128
	learning_rate = 1e-3
	tokenizer = BertTokenizer.from_pretrained('../bert-base-chinese', return_dict=False)
	
	model = SFTModel(vocab_size)
	if torch.cuda.is_available():
		model.cuda()
	optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
	sentences = loader(data_path, window_size)
	# 输入多个句子
	for epoch in range(epoch_nums):
		print(f'======第{epoch+1}轮训练开始=====')
		for batch in range(total_train//batch_size):
			x, mask, y = build_dataSet(sentences[batch*batch_size:(batch+1)*batch_size], tokenizer, window_size)
			print(x,y)
			if torch.cuda.is_available():
				x = x.cuda()
				mask.cuda() #这里还有问题
				y = y.cuda()
			# 这里准备替换
			# 训练
			watch_loss = []
			model.train()
			optim.zero_grad()
			loss = model(x, y)  # 计算loss
			print(loss)
			loss.backward()
			optim.step()
			watch_loss.append(loss.item())
			print(watch_loss)
		print("第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
		print(predict_sentence("江西进贤“差生”走廊考试？ 校方称教师缺爱心", model, tokenizer, window_size))
		print(predict_sentence("中宣部推动学雷锋 进教材开微博", model, tokenizer, window_size))
	


if __name__ == '__main__':
	main()
