create_datas 的设计思路
最终目标是
通过传不同的路径 让其读取不一样的 train_data  和test_data

所以在最终的调用函数中要用到  type 是 train 还是 test 来区分

需要用到两个魔法函数   __len__ 来限制  训练集的数据    __gititem__   通过索引来拿到数据
最终都是为了DataLoader函数对数据进行自动的拆分

一个vocob 函数  让字转换成对应的index
一个生成对应的测试集数据和训练集数据


model

就是先做一个emdding 的