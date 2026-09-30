import torch 
import torch.nn as nn

class Train_model(nn.Module):
    def __init__(self,input_feature):
        super().__init__()
        self.input_size=input_feature
        self.layer1=nn.Linear(self.input_size,64)
        self.relu1=nn.ReLU()
        self.layer2=nn.Linear(64,32)
        self.relu2=nn.ReLU()
        self.layer3=nn.Linear(32,16)
        self.relu3=nn.ReLU()
        self.layer4=nn.Linear(16,1)
        self.sigmoid=nn.Sigmoid()
        

    def forward(self,x_train):
        out=self.layer1(x_train)
        out=self.relu1(out)
        out=self.layer2(out)
        out=self.relu2(out)
        out=self.layer3(out)
        out=self.relu3(out)
        out=self.layer4(out)
        out=self.sigmoid(out)

        return out
def CatchTrainData(x_train_loader,x_train):
    loss=nn.BCELoss()
    model=Train_model(x_train.shape[1])
    for epoch in range(10):
        totalloss=0
        optim=torch.optimizer.Adam(model.parameters(),lr=0.01)
        for x_train_batch,y_train_batch in x_train_loader:
            output=model(x_train_batch)
            y_train_batch=y_train_batch.view(-1,1)
            loss_value=loss(output,y_train_batch)
            optim.zero_grad()
            loss_value.backward()
            optim.step()
            totalloss+=loss_value.item()
        average=totalloss/len(x_train_loader)
        print("Epoch:",epoch+1,
              "Average loss",average)










    

