import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timeit


inSize=2
outSize=1


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1=nn.Linear(inSize,3)
        self.fc2=nn.Linear(3,9)
        self.fc3=nn.Linear(9,9)
        self.fc4=nn.Linear(9,3)
        self.fc5=nn.Linear(3,outSize)
        
        #self.fc1.weight.data.fill_(1)
        #self.fc1.bias.data.fill_(1)
        #self.fc2.weight.data.fill_(1)
        #self.fc2.bias.data.fill_(1)
        #self.fc3.weight.data.fill_(1)
        #self.fc3.bias.data.fill_(1)
 


        
    def forward(self,x):

        res = x[0][0]
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        x = self.fc5(x)
        x = x +res
    
        return x
       

net = Net()

xdata=[[1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0],[0.0,0.2,0,4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0],[0.24,0.36,0.48,0.5,0.62,0.74,0.86,0.98,1.0,1,12],[0.46,0.62,0.78,0.94,1.1,1.26,1.42,1.58,1.74,1.9]]
delta = [0.1,0,2,0.12,0.16]

iter = 0

optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

criterion = nn.MSELoss()

for j in range(len(delta)):

    for i in range(len(xdata[j-1])-1):
        input = torch.tensor([[xdata[j-1][i-1],delta[j-1]]]).float()
        target = torch.tensor([[xdata[j-1][i]]]).float()

        for x in range (1000):
            net.zero_grad()
            output=net(input)
            loss = criterion(output,target)
            loss.backward()
            optimizer.step()

            iter = iter +1

print("KLAR")        
input = torch.tensor([[1.2,0.1]])
print(net(input))
input = torch.tensor([[1.5,0.1]])
print(net(input))
input = torch.tensor([[0.2,0.2]])
print(net(input))
input = torch.tensor([[0.86,0.12]])
print(net(input))
input = torch.tensor([[1.42,0.16]])
print(net(input))