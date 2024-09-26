import os

class DataGenerator():
	def __init__(self,data_path):
		self.data_path = data_path
		self.load()


	def load(self):
		filename_path=[]
		for filename in os.listdir(self.data_path):
			if filename.endswith('.txt'):
				file_path = os.path.join(self.data_path,filename)
				filename_path.append(file_path)
		content = ''
		for i in filename_path:
			with open(i, 'r', encoding='utf8') as f:
				context = f.read()
				content+=(context)
		return content
