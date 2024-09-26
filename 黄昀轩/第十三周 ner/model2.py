import torch
import torch.nn as nn
from torch.optim import Adam,SGD
# from torchcrf import CRF
from transformers import BertModel
# 创建包裹器
#继承object类，这个是不含方法的空类，封装性：通过将配置数据封装在类中，可以隐藏实现细节，只暴露必要的接口。
# 这样，使用配置数据的代码就不需要了解配置数据的具体存储方式，只需要调用类提供的方法即可。
#可维护性：将配置数据封装在类中，可以更容易地管理和更新配置。如果将来需要修改配置数据的存储方式或格式，只需要修改类内部的实现，而不需要修改使用配置数据的代码。
class ConfigWrapper(object):
	def __init__(self,config):
		self.config = config
	def to_dict(self):
		return self.config
# 创建训练模型
class TorchModel(nn.Module):
	def __init__(self,config):
		super(TorchModel,self).__init__()
		self.config = ConfigWrapper(config)
		class_num = config['class_num']
		max_length = config['max_length']
		
		self.bert = BertModel.from_pretrained("../../bert-base-chinese",return_dict=False)
		self.classify = nn.Linear(self.bert.config.hidden_size,class_num)
		self.loss = nn.CrossEntropyLoss(ignore_index=-1)
	def forward(self,x,target=None):
		x,_ =self.bert(x)
		predict = self.classify(x)
		
		if target is not None:
			#view 控制组后一维的维度，其余-1自动
			return self.loss(predict.view(-1,predict.shape[-1]),target.view(-1))
			
		else:
			return y_pred
		
def choose_optimizer(config,model):
	optimizer = config["optimizer"]
	learing_rate = config["learning_rate"]
	if optimizer =='adam':
		return Adam(model.parameters(),lr = learing_rate)
	elif optimizer =="sgd":
		return SGD(model.parameters(),lr =learing_rate)
		
	
if __name__ == "__main__":
    from config import Config
    model = TorchModel(Config)
