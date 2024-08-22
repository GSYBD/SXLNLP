from gensim.models import Word2Vec
import jieba
import numpy as np
from sklearn.cluster import KMeans
import math
from collections import defaultdict


def cosine_distance(vec1, vec2):
    vec1 = vec1 / np.sqrt(np.sum(np.square(vec1)))  #A/|A|
    vec2 = vec2 / np.sqrt(np.sum(np.square(vec2)))  #B/|B|
    return np.sum(vec1 * vec2)

model = Word2Vec.load('model.w2v')
with open('titles.txt', encoding='utf8') as f:
    sentences = set()
    for line in f:
        sentence = line.strip()
        words_string = " ".join(jieba.lcut(sentence))
        sentences.add(words_string)

vectors = []
for sentence in sentences:
    ls = sentence.split()
    vector = np.zeros(model.vector_size)
    for word in ls:
        try:
            vector += model.wv[word]
        except:
            vector= vector
    vectors.append(vector/len(ls))
vectors = np.array(vectors)

##kmeans
clusters_num = int(math.sqrt(len(sentences)))
print("cluster_num:", clusters_num)

kmeans = KMeans(clusters_num)
kmeans.fit(vectors)

center = kmeans.cluster_centers_

dic = defaultdict(list)
dic_dis = defaultdict(list)
for sentence, label in zip(sentences, kmeans.labels_):
    dic[label].append(sentence)
for index, label in enumerate(kmeans.labels_):
    mse = math.pow(np.mean(vectors[index]-center[label]), 2)
    dic_dis[label].append(mse)
for label, mse_ls in dic_dis.items():
    dic_dis[label] = np.mean(mse_ls)

dic_sorted = sorted(dic_dis.items(), key=lambda x:x[1], reverse=True)

print("#######################")
for label, dis in dic_sorted:
    print("cluster:", label)
    print(dic[label][:3])
    print()

#print
# for label, sentences in dic.items():
#     print("cluster:", label)
#     for i in range(min(10, len(sentences))):
#         print(sentences[i].replace(" ",""))
#     print("####################")