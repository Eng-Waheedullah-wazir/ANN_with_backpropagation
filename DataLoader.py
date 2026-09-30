from torch.utils.data import Dataset,DataLoader

class CustomDatasetTrain(Dataset):
    def __init__(self,x_data,y_data):
        self.x_data=x_data
        self.y_data=y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


class CustomDatasetVal(Dataset):
    def __init__(self,x_data_val,y_data_val):
        self.x_data_val=x_data_val
        self.y_data_val=y_data_val

    def __len__(self):
        return len(self.x_data_val)

    def __getitem__(self,id):
        return self.x_data_val[id],self.y_data_val[id]


class CustomDatasetTest(Dataset):
    def __init__(self,x_data_test,y_data_test):
        self.x_data_test=x_data_test
        self.y_data_test=y_data_test

    def __len__(self):
        return len(self.x_data_test)

    def __getitem__(self,id):
        return self.x_data_test[id],self.y_data_test[id]


def create_data(x_train,y_train,x_val,y_val,x_test,y_test):
    train_data=CustomDatasetTrain(x_train,y_train)
    val_data=CustomDatasetVal(x_val,y_val)
    test_data=CustomDatasetTest(x_test,y_test)

    train_loader=DataLoader(
        train_data,
        shuffle=True,
        batch_size=32,
        drop_last=True
    )

    val_loader=DataLoader(
        val_data,
        shuffle=False,
        batch_size=32,
        drop_last=True

    )

    test_loader=DataLoader(
            test_data,
            shuffle=False,
            batch_size=32,
            drop_last=True
        )
    

    return train_loader,val_loader,test_loader


    