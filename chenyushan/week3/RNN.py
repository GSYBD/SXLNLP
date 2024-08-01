from torchsummary import summary
import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

class TorchRNNModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchRNNModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)  #embedding��
        self.rnn_layer = nn.RNN(vector_dim, vector_dim, bias=True, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(vector_dim)
        self.rnn_layer2 = nn.RNN(vector_dim, 5, bias=True, batch_first=True)
        self.layer_norm2 = nn.LayerNorm(5)
        self.pool = nn.AvgPool1d(sentence_length)   #�ػ���
        # self.classify = nn.Linear(vector_dim, 5)     #���Բ�
        self.activation = nn.Softmax(dim=1)
        self.criterion = nn.NLLLoss()

    #��������ʵ��ǩ������lossֵ������ʵ��ǩ������Ԥ��ֵ
    def forward(self, x, y=None):
        x = self.embedding(x)                      #(batch_size, sen_len) -> (batch_size, sen_len, vector_dim)

        z, _ = self.rnn_layer(x)                      #(batch_size, sen_len, vector_dim) -> (batch_size, sen_len, vector_dim)
        z = self.layer_norm1(z)
        z, _ = self.rnn_layer2(z)
        z = self.layer_norm2(z)
        x = z.transpose(1, 2)                      #(batch_size, sen_len, vector_dim) -> (batch_size, vector_dim, sen_len)
        x = self.pool(x)                           #(batch_size, vector_dim, sen_len)->(batch_size, vector_dim, 1)
        x = x.squeeze()                            #(batch_size, vector_dim, 1) -> (batch_size, vector_dim)

        # x = self.classify(x)                       #(batch_size, vector_dim) -> (batch_size, 1) 3*5 5*1 -> 3*1
        y_pred = self.activation(x)                #(batch_size, 1) -> (batch_size, 1)

        if y is not None:
            y = y.flatten()
            return self.criterion(torch.log(y_pred), y)
        else:
            return y_pred                 #���Ԥ����

def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz��"  #�ַ���
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1   #ÿ���ֶ�Ӧһ�����
    vocab['unk'] = len(vocab) #26
    return vocab

#�������һ������
#����������ѡȡsentence_length����
#��֮Ϊ������
def build_sample(vocab, sentence_length):
    #������ֱ�ѡȡsentence_length���֣������ظ�
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length - 1)]
    if "��" in x:
        x.append("a")
    else:
        x.append("��")
    random.shuffle(x)
    number = x.index("��")
    if 0 <= number <= 2:
        y = 0
    elif 3 <= number <= 5:
        y = 1
    elif 6 <= number <= 8:
        y = 2
    elif 9 <= number <= 11:
        y = 3
    elif 12 <= number <= 14:
        y = 4
    x = [vocab.get(word, vocab['unk']) for word in x]   #����ת������ţ�Ϊ����embedding
    return x, y

#�������ݼ�
#������Ҫ��������������Ҫ�������ɶ���
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append([y])
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)   #����200�����ڲ��Ե�����

    with torch.no_grad():
        y_pred = model(x)  # ģ��Ԥ��
        predicted_classes = torch.argmax(y_pred, dim=1)  # ��ȡ��߸��ʵ��������
        # print(predicted_classes)
        y_true = y.flatten()
        # print('x:', x)
        # print(y_true, y_pred)
        # print('y_pred:', predicted_classes)
        correct_predictions = (predicted_classes == y_true).sum().item()
    # print("Ԥ�����%s, ��ʵ���%s" % (predicted_classes.numpy(), y_true))
    print("��ȷ�ʣ�%f" % (correct_predictions / y_pred.shape[0]))
    return correct_predictions / y_pred.shape[0]


#����ģ��
def build_model(vocab, char_dim, sentence_length):
    model = TorchRNNModel(char_dim, sentence_length, vocab)
    return model

def main():
    #���ò���
    epoch_num = 100        #ѵ������
    batch_size = 30       #ÿ��ѵ����������
    train_sample = 500    #ÿ��ѵ���ܹ�ѵ������������
    char_dim = 20         #ÿ���ֵ�ά��
    sentence_length = 15   #�����ı�����
    learning_rate = 0.001 #ѧϰ��
    # �����ֱ�
    vocab = build_vocab()
    # ����ģ��
    model = build_model(vocab, char_dim, sentence_length)
    # ѡ���Ż���
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # ѵ������
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length) #����һ��ѵ������
            optim.zero_grad()    #�ݶȹ���
            loss = model(x, y)   #����loss
            loss.backward()      #�����ݶ�
            optim.step()         #����Ȩ��
            watch_loss.append(loss.item())
        print("=========\n��%d��ƽ��loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)   #���Ա���ģ�ͽ��
        log.append([acc, np.mean(watch_loss)])
    #��ͼ
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  #��acc����
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  #��loss����
    plt.legend()
    plt.show()
    #����ģ��
    torch.save(model.state_dict(), "model_rnn.pth")
    # ����ʱ�
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

#ʹ��ѵ���õ�ģ����Ԥ��
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # ÿ���ֵ�ά��
    sentence_length = 15  # �����ı�����
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #�����ַ���
    model = build_model(vocab, char_dim, sentence_length)     #����ģ��
    model.load_state_dict(torch.load(model_path))             #����ѵ���õ�Ȩ��
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  #���������л�
    model.eval()   #����ģʽ
    with torch.no_grad():  #�������ݶ�
        result = model.forward(torch.LongTensor(x))  #ģ��Ԥ��
    predicted_classes = torch.argmax(result, dim=1)
    for i in range(len(predicted_classes)):
        print("�����ı���ţ�%s" % i)
        print("Ԥ������ǣ�%s" % predicted_classes[i].item())



if __name__ == "__main__":
    # main()
    # label�� 1   0   4  3
    test_strings = ["fnv��efnvfefnvfe", "��zsdfwzsdfwzsdf", "rqwderqwderqw��e", "nakwwnakw��naffa"]
    predict("model_rnn.pth", "vocab.json", test_strings)