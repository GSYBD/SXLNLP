# 项目实现
## 1. 描述
基于欧式距离/余弦距离筛选Kmeans的优质分类
## 2. 具体实现
1. 简单修改词向量训练模型，并保存
   <img width="1283" alt="image" src="https://github.com/user-attachments/assets/8e1f7580-fd88-4b6d-bb5e-516992774db3">
2. 基于训练好的模型，实现kmeans聚类  
3. 重点实现部分：基于马氏距离筛选kmeans聚类后的优质类别筛选
## 3. 备注
1. intraClassDistance_kmeans.py文件中calc_intra_class_distance是重点实现部分
2. 计算之后如何给出筛选条件也是一个值得思考的问题，这里暂且简化
