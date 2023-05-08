import math

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, TensorDataset, ConcatDataset

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms."""
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)

        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]
        if self.transform:
            x = self.transform(x.numpy().astype(np.uint8))
        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


def dividing_validation_set(train_data,validation_num):

    if train_data.data.ndim==4:  #CIFRA-10
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    else:
        train_data.data = torch.unsqueeze(train_data.data, 3)
        transform = torchvision.transforms.ToTensor()

    x_data = torch.Tensor(train_data.data)
    y_data = torch.Tensor(train_data.targets)

    ind1=[]
    ind2=[]
    for i in range(len(x_data)-validation_num):
        ind1.append(i)
    for i in range(len(x_data)-validation_num,len(x_data)):
        ind2.append(i)

    x_data_info1 = x_data[ind1].to(torch.float)
    y_data_info1 = y_data[ind1].to(torch.long)
    train_tensor_dataset = CustomTensorDataset((x_data_info1,y_data_info1), transform)


    x_data_info2 = x_data[ind2].to(torch.float)
    y_data_info2 = y_data[ind2].to(torch.long)
    valid_tensor_dataset = CustomTensorDataset((x_data_info2,y_data_info2), transform)

    y_data_info2=y_data_info2.numpy().tolist()
    print("··········label distribution···········")
    for i in range(10):
        print("label:{}".format(i)+"| count:{}".format(y_data_info2.count(i)))
    print()


    return train_tensor_dataset,valid_tensor_dataset

def dividing_validation_set_for_IMDB(train_data,validation_num):

    x_data = torch.Tensor(train_data.tensors[0])
    y_data = torch.Tensor(train_data.tensors[1])

    ind1=[]
    ind2=[]
    print("len(x_data):",len(x_data))
    for i in range(validation_num,len(x_data)):
        ind1.append(i)
    for i in range(validation_num):
        ind2.append(i)

    x_data_info1 = x_data[ind1]
    y_data_info1 = y_data[ind1]
    train_tensor_dataset = CustomTensorDataset((x_data_info1,y_data_info1))


    x_data_info2 = x_data[ind2]
    y_data_info2 = y_data[ind2]
    valid_tensor_dataset = CustomTensorDataset((x_data_info2,y_data_info2))

    y_data_info2=y_data_info2.numpy().tolist()
    print("··········label distribution···········")
    for i in range(1):
        print("label:{}".format(i)+"| count:{}".format(y_data_info2.count(i)))
    print()


    return train_tensor_dataset,valid_tensor_dataset
