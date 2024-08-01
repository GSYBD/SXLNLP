import torch

from loader import load_data

'''
测试
'''


class Evaluator():
	def __init__(self, config, model, logger):
		self.config = config
		self.logger = logger
		self.model = model
		self.valid_data = load_data(config['valid_data_path'], config)
		self.stats_dict = {'correct': 0, 'wrong': 0}  # 用于储存测试结果
	
	def eval(self, epoch):
		self.logger.info("--------%d epoch：" % epoch)
		self.model.eval()
		self.stats_dict = {"correct": 0, "wrong": 0}  # 清空上一轮结果
		for idx, batch_data in enumerate(self.valid_data):
			batch_data = [d.cuda() for d in batch_data]
			input_ids, labels = batch_data
			with torch.no_grad():
				pred_res = self.model(input_ids)
			self.write_stats(labels, pred_res)
		acc = self.show_stats()
		return acc
	
	
	def write_stats(self, labels, pred_res):
		# print('labels_type:', type(labels), labels)
		# print('pred_res_type:', type(pred_res), pred_res)
		assert len(labels) == len(pred_res)
		for true_label, pred_label in zip(labels, pred_res):
			pred_label = torch.argmax(pred_label)
			# print('true_labek_type:',type(true_label),true_label)
			# print('pred_labek_type:',type(pred_label),pred_label)
			# true_label = true_label.cpu().numpy()
			# pred_label = pred_label.cpu().numpy()
			if int(pred_label) == int(true_label):
				self.stats_dict['correct'] += 1
			else:
				self.stats_dict['wrong'] += 1
		return
		
	def show_stats(self):
		correct =self.stats_dict['correct']
		wrong =self.stats_dict['wrong']
		self.logger.info("Total_test_nums：%d" % (correct + wrong))
		self.logger.info("correct_nums：%d，wrong_nums：%d" % (correct, wrong))
		self.logger.info("ACC：%f" % (correct / (correct + wrong)))
		self.logger.info("--------------------")
		return correct/(correct+wrong)
	

