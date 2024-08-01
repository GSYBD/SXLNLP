# 注意1：自定义py文件的名称一定不要和模块名重名
import csv
from sklearn.model_selection import train_test_split
import json
# import jsonlines

# 1.读取csv文件
def read_data(data_path=r'文本分类练习.csv'):
    with open(data_path,'r',encoding='utf-8') as f1:
        # f1；iterable
        # csv.reader(f1) ：返回iterator
        reader = csv.reader(f1)   # 类似于r = f1.read()
        datalist = []
        for row in reader:
            datalist.append(row)
            x_list = []
            y_list = []
        for one_line in datalist[1:]:
            y_list.append(int(one_line[0]))  # 取出label(每行第一位都是label)
            one_list = "".join(one_line[1:]) # 取出评论
            x_list.append(one_list)
    return x_list, y_list

x_list, y_list = read_data()

# def split_data(x_list, y_list, ratio=0.30):  # 70%训练集，30%测试集:
'''
按照指定的比例，划分样本数据集
ratio: 测试数据的比率
'''
X_train, X_test, y_train, y_test = train_test_split(x_list, y_list, test_size=0.30, random_state=50)

"""训练集"""
for x,y in zip(X_train,y_train):
    review = {}
    review['tag']=y
    review['review']=x
    with open(r'sub_train.json','a',encoding='utf-8') as f1:
        f1.write(json.dumps(review,ensure_ascii=False)+"\n")

"""测试集"""
for x,y in zip(X_test,y_test):
    review = {}
    review['tag']=y
    review['review']=x
    with open(r'sub_test.json','a',encoding='utf-8') as f1:
        f1.write(json.dumps(review,ensure_ascii=False)+"\n")

    # return


# if __name__ == '__main__':
#     """获取大文件的数据"""
#     x_list, y_list = read_data()
#     """划分为训练集和测试集及label文件"""
#     split_data(x_list, y_list)
#
#  main()