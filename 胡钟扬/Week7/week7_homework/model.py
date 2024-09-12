import torch
import torch.nn as nn


from transformers import BertModel, BertTokenizer




class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        
        hidden_size = config["hidden_size"]
        class_num = config['class_num']
        vocab_size = config['vocab_size']+1
        model_type = config['model_type']
        num_layers = config['num_layers']
        
        self.use_bert = False
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)

        self.encoder = None
        if model_type == "fast_text":
            self.encoder = lambda x: x # 从embedding出来以后，直接去分类器
        elif model_type == "lstm":
            self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "gru":
            self.encoder = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "rnn":
            self.encoder = nn.RNN(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "cnn":
            self.encoder = CNN(config)
        elif model_type == "gated_cnn":
            self.encoder = GatedCNN(config)
        elif model_type == "stack_gated_cnn":
            self.encoder = StackGatedCNN(config)
        elif model_type == "rcnn":
            self.encoder = RCNN(config)
        elif model_type == "bert":
            self.use_bert = True
            self.encoder = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
            hidden_size = self.encoder.config.hidden_size
        elif model_type == "bert_lstm":
            self.use_bert = True
            self.encoder = BertLSTM(config)
            hidden_size = self.encoder.bert.config.hidden_size
        elif model_type == "bert_cnn":
            self.use_bert = True
            self.encoder = BertCNN(config)
            hidden_size = self.encoder.bert.config.hidden_size
        elif model_type == "bert_mid_layer":
            self.use_bert = True
            self.encoder = BertMidLayer(config)
            hidden_size = self.encoder.bert.config.hidden_size
        
        self.classifier = nn.Linear(hidden_size, class_num)
        self.pooling_type = config['pooling_style']
        
        self.loss = nn.functional.cross_entropy
        
    def forward(self,x, target=None):
        if self.use_bert:
            x = self.encoder(x)  # sequence_output, pooling_output
        else:
            # bert以外的其他模型是没有自带embedding的
            x = self.embedding(x)
            x = self.encoder(x)
        
        if isinstance(x, tuple): # 如果是形如rnn的返回值
            x = x[0]       # 我们只取其中的序列 [batch_size, max_len, hidden_size]
        
        if self.pooling_type == 'max':
            self.pooling = nn.MaxPool1d(x.shape[1])
        else:
            self.pooling = nn.AvgPool1d(x.shape[1])
            
        # 池化得到句子向量
        # 池化的输入：(batch_size, hidden_size, max_len) 句子长度必须在最后一维
        x = self.pooling(x.transpose(1,2)).squeeze() # [batch_size, hidden_size]
        
        
        y_pred = self.classifier(x)
        
        if target is not None:
            return self.loss(y_pred, target.squeeze())
        else:
            return y_pred
        




class CNN(nn.Module):
    def __init__(self,config):
        super(CNN, self).__init__()
        self.hidden_size = config['hidden_size']
        self.kernel_size = config['kernel_size']
        self.pad = int((self.kernel_size-1)/2)    
        
        self.cnn = nn.Conv1d(self.hidden_size, self.hidden_size, self.kernel_size, bias=False, padding=self.pad)

    def forward(self, x): #x : (batch_size, max_len, embeding_size)
        pred_y = self.cnn(x.transpose(1,2)).transpose(1,2) # [batch_size, hidden_size]
        return pred_y


class GatedCNN(nn.Module):
    def __init__(self, config):
        super(GatedCNN, self).__init__()
        self.cnn = CNN(config)
        self.gate = CNN(config)
    
    def forward(self, x):
        a = self.cnn(x)
        b = self.gate(x)
        b = torch.sigmoid(b)
        return torch.mul(a, b) # 对位相乘



class StackGatedCNN(nn.Module):
    def __init__(self, config):
        super(StackGatedCNN, self).__init__()
        self.num_layers = config['num_layers']
        self.hidden_size = config['hidden_size']
        self.gcnn_layers = nn.ModuleList(
            GatedCNN(config)
            for i in range(self.num_layers)
        )
        
        self.ff_linear_layer1 = nn.ModuleList(
            nn.Linear(self.hidden_size, self.hidden_size)
            for i in range(self.num_layers)
        )
        
        self.ff_linear_layer2 = nn.ModuleList(
            nn.Linear(self.hidden_size, self.hidden_size)
            for i in range(self.num_layers)
        )
        
        self.bn_after_gcnn = nn.ModuleList(
            nn.LayerNorm(self.hidden_size)
            for i in range(self.num_layers)
        )
        
        
        self.bn_after_ff = nn.ModuleList(
            nn.LayerNorm(self.hidden_size)
            for i in range(self.num_layers)
        )
        
    def forward(self, x):
        # 模仿bert 12层结构
        for i in range(self.num_layers):
            gcnn_x = self.gcnn_layers[i](x)
            
            gcnn_x = gcnn_x + x # 残差
            
            gcnn_x = self.bn_after_gcnn[i](x)
            
            ff1_x = self.ff_linear_layer1[i](gcnn_x)
            ff1_x = torch.relu(ff1_x)
            ff2_x = self.ff_linear_layer2[i](ff1_x)
            
            ff2_x = ff2_x + gcnn_x
            
            x = self.bn_after_ff[i](ff2_x)
        return x #  [batch_size, seq_len, hidden_size]

            



class RCNN(nn.Module):
    def __init__(self, config):
        super( RCNN, self).__init__()
        self.cnn = GatedCNN(config)
        self.rnn = nn.LSTM(config["hidden_size"], config["hidden_size"], batch_first=True)
    
    def forward(self, x):
        x,_ = self.rnn(x)
        x = self.cnn(x)
        return x



class BertLSTM(nn.Module):
    def __init__(self, config):
        super(BertLSTM, self).__init__()
        # self.hidden_size = config['hidden_size']
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
        # 这里必须要用bert自己的hidden_size, 否则跑不起来
        self.lstm = nn.LSTM(input_size=self.bert.config.hidden_size, hidden_size=self.bert.config.hidden_size)
        
        
    def forward(self, x):
        x, _ = self.bert(x)  # [batch_size, seq_len, hidden_size]
        x,(_,_)= self.lstm(x) # 
        
        return x
 
    
    
class BertCNN(nn.Module):
    def __init__(self,config):
        # self.hidden_size = config['hidden_size']
        super(BertCNN, self).__init__()
        self.kernel_size = config['kernel_size']
        self.pad = int((self.kernel_size-1)/2)
        self.bert = BertModel.from_pretrained(config['pretrain_model_path'],return_dict =False)
        self.cnn = nn.Conv1d(in_channels=self.bert.config.hidden_size, out_channels = self.bert.config.hidden_size,\
                            kernel_size= self.kernel_size, padding=self.pad, bias=False)
    def forward(self, x):
        x, _ = self.bert(x)
        x = self.cnn(x.transpose(1,2)).transpose(1,2)
        return x
    
    
    
class BertMidLayer(nn.Module):
    def __init__(self, config):
        super(BertMidLayer, self).__init__()
        self.bert = BertModel.from_pretrained(config['pretrain_model_path'], return_dict = False)
        self.bert.config.output_hidden_states = True
    
    def forward(self, x):
        # 调用self.bert(x)返回一个元组：(sequence_output, pooler_output, hidden_states列表)
        hidden_states:list = self.bert(x)[2] # (13, batch, len, hidden)
        
        # 12 层中的每一层transformer的输出都是 batch_size x L x H

        layer_states = torch.add(hidden_states[-1], hidden_states[-2])
        return layer_states
    
def choose_optimizer(config, model):
    optimizer_type = config["optimizer"]
    
    if optimizer_type == 'adam':
        return torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    if optimizer_type == 'sgd':
        return torch.optim.Adam(model.parameters(), lr =config['learning_rate'])

    raise Exception("Unknown optimizer type: ", optimizer_type)



if __name__ == "__main__":
    pass