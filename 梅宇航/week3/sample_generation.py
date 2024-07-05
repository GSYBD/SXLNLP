from faker import Faker
import random

# 构建词汇表
def build_vocab():
    words = ["安静", "独处", "思考", "内向", "社交", "热闹", "活动"]  # I人和E人词汇表
    vocab = {"pad": 0}
    for index, word in enumerate(words):
        vocab[word] = index + 1            # 每个词对应一个序号
    vocab['unk'] = len(vocab)
    return vocab

# 定义I人关键词
i_keywords = ["安静", "独处", "思考", "内向"]

# 使用faker生成句子
def generate_sentence_with_keyword(keyword):
    fake = Faker("zh_CN")
    sentence = fake.sentence(nb_words=10)
    words = list(sentence)
    index = random.randint(0, len(words) - 1)
    words.insert(index, keyword)
    return ''.join(words)

# 随机生成一个样本
def build_sample(vocab):
    if random.random() < 0.5:
        keyword = random.choice(i_keywords)
        sentence = generate_sentence_with_keyword(keyword)
        y = 1
    else:
        words = list(vocab.keys())
        words.remove('pad')
        words.remove('unk')
        words = [word for word in words if word not in i_keywords]
        keyword = random.choice(words)
        sentence = generate_sentence_with_keyword(keyword)
        y = 0
    return sentence, y

# 生成并打印样本
def generate_samples(num_samples, vocab):
    with open("samples.txt", "w", encoding="utf8") as f:
        for _ in range(num_samples):
            sentence, label = build_sample(vocab)
            print(f"样本: {sentence}, 真实类别: {'I人' if label == 1 else 'E人'}\n")
            f.write(f"样本: {sentence}, 真实类别: {'I人' if label == 1 else 'E人'}\n")

if __name__ == "__main__":
    vocab = build_vocab()
    generate_samples(3000, vocab)
    print("样本生成完毕，保存在 samples.txt 文件中")
