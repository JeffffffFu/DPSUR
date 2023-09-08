import torch.nn.functional as F
import torch.nn as nn


class RNN_Relu(nn.Module):
    def __init__(self,max_words=20000,emb_size=100):
        super(RNN_Relu,self).__init__()
        self.max_words = max_words
        self.emb_size = emb_size
        self.Embedding = nn.Embedding(self.max_words,self.emb_size)
        self.fc0 = nn.Linear(self.emb_size, 32)
        self.RNN = nn.LSTM(32,32,num_layers=1,batch_first=True, bidirectional= True)
        self.fc1 = nn.Linear(64,16)
        self.fc2 = nn.Linear(16, 2)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.Embedding(x)
        x = self.fc0(x)
        x = self.relu(x)
        x,_ = self.RNN(x)
        x = F.avg_pool2d(x,(x.shape[1],1)).squeeze()
        x = self.fc1(x)
        out = self.relu(x)
        pred = self.fc2(out)
        return pred

class RNN_Tanh(nn.Module):
    def __init__(self,max_words=20000,emb_size=100):
        super(RNN_Tanh,self).__init__()
        self.max_words = max_words
        self.emb_size = emb_size
        self.Embedding = nn.Embedding(self.max_words,self.emb_size)
        self.fc0 = nn.Linear(self.emb_size, 32)
        self.RNN = nn.LSTM(32,32,num_layers=1,batch_first=True, bidirectional= True)
        self.fc1 = nn.Linear(64,16)
        self.fc2 = nn.Linear(16, 2)
        self.tanh=nn.Tanh()


    def forward(self,x):
        x = self.Embedding(x)
        x = self.fc0(x)
        x = self.tanh(x)
        x,_ = self.RNN(x)
        x = F.avg_pool2d(x,(x.shape[1],1)).squeeze()
        x = self.fc1(x)
        out = self.tanh(x)
        pred = self.fc2(out)
       # return nn.functional.softmax(pred)

        return pred