import pandas as pd
from sklearn.model_selection import train_test_split
import torch

class Dataset():
    def __init__(self,data_path):
        self.data=pd.read_csv(data_path)

    def split_data(self):
        x=self.data.iloc[:,:-1]
        y=self.data.iloc[:,-1]
        x_train,y_train,x_test,y_test=train_test_split(x,
                                                       y,
                                                       test_size=0.2,
                                                       random_state=42)

        x_val,x_test,y_val,y_test=train_test_split(x_test,
                                                   y_test,
                                                   test_size=0.5,
                                                   random_state=42)

        x_train=torch.tensor(x_train.to_numpy(),dtype=torch.float32)
        y_train=torch.tensor(y_train.to_numpy(),dtype=torch.float32)
        x_val=torch.tensor(x_val.to_numpy(),dtype=torch.float32)
        y_val=torch.tensor(y_val.to_numpy(),dtype=torch.float32)
        x_test=torch.tensor(x_test.to_numpy(),dtype=torch.float32)
        y_test=torch.tensor(y_test.to_numpy(),dtype=torch.float32)


        return x_train,y_train,x_val,y_val,x_test,y_test
    


