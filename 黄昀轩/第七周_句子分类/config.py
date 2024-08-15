# coding: utf-8
"""
配置信息参数
"""
with open('chars.txt', 'r', encoding='utf8') as f:
	vocab_len = len(list(f))

	

config = {
	'model_path': 'out_put',
	'class_num': 2,
	'train_data_path': 'data/train_50.csv',
	'valid_data_path': 'data/test_50.csv',
	'model_type': 'bert_lstm',
	'vocab_path': 'chars.txt',
	'vocab_size': vocab_len,
	'max_length': 465,
	'hidden_size': vocab_len+1,
	'kernel_size': 3,
	'num_layers': 2,
	'epoch': 20,
	'batch_size': 128,
	'pooling_style': 'max',
	'optimizer': 'adam',
	'learning_rate': 1e-3,
	'pretrained_model_path':'bert-base-chinese',
	'seed': 987
	
}

# if __name__ =="__main__":
# 	main()
