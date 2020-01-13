import torch.nn as nn

class My_Model(nn.Module):

    def __init__(self):
        super(My_Model, self).__init__()
        self.linear1 = nn.Linear(28*28, 200)
        self.relu1 = nn.ReLU(inplace=True)

        self.linear2 = nn.Linear(200, 200)
        self.relu2 = nn.ReLU(inplace=True)

        self.linear3 = nn.Linear(200, 10)

    def forward(self, x):
        '''前向传播'''
        out = x.view(x.size(0), -1)
        out = self.relu1(self.linear1(out))
        out = self.relu2(self.linear2(out))
        out = self.linear3(out)
        return out