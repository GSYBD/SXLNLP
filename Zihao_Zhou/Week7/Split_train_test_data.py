import pandas as pd
import random
dataset = pd.read_csv("文本分类练习.csv")
test_file = open("test.txt", mode="w", encoding="utf-8")
test_file.write("label" + "\t" + "text" + "\n")
test_samples = []
train_file = open("train.txt", mode="w", encoding="utf-8")
train_file.write("label" + "\t" + "text" + "\n")
train_samples = []
for i in range(len(dataset)):
    train_test_flag = random.randint(1, 5)
    if train_test_flag == 1:
        # test_file.write(str(dataset.values[i, 0]) + "\t" + dataset.values[i, 1] + "\n")
        test_samples.append(str(dataset.values[i, 0]) + "\t" + dataset.values[i, 1] + "\n")
    else:
        # train_file.write(str(dataset.values[i, 0]) + "\t" + dataset.values[i, 1] + "\n")
        train_samples.append(str(dataset.values[i, 0]) + "\t" + dataset.values[i, 1] + "\n")

random.shuffle(train_samples)
random.shuffle(train_samples)
random.shuffle(train_samples)

for sample in test_samples:
    test_file.write(sample)
for sample in train_samples:
    train_file.write(sample)
test_file.close()
train_file.close()