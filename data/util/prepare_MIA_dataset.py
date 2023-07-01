# import torchvision.models as models # vgg19 # no channel
# from .preact_resnet import PreActResNet18  # no channel
from typing import Any, Callable, List, Optional, Union, Tuple
from functools import partial
import torchvision.transforms as transforms
import PIL.Image as Image
import torch.nn as nn
import os
import torch
import random
import numpy as np
# import pandas
import torchvision
from tqdm import tqdm

from data.util.get_data import get_data
from model.get_model import get_MIA_model


def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
seed_torch()

class CNN_MIA(nn.Module):
    def __init__(self, input_channel=3, num_classes=10):
        super(CNN_MIA, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=3),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128*6*6, 512),
            nn.Tanh(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def prepare_MIA_dataset(dataset_name, algorithm,device):
    # num_classes, dataset, target_model, shadow_model = get_model_dataset(
    #     dataset, root=root, model_name=model_name)

    train_data,test_data,dataset=get_data(dataset_name, augment=False)

    target_model, shadow_model,num_classes=get_MIA_model(algorithm,dataset_name,device)


    length = len(dataset)
    # each_length = length//4
    each_length=length//6
    # target_train, target_test, shadow_train, shadow_test, _ = torch.utils.data.random_split(
    #     dataset, [each_length, each_length, each_length, each_length, len(dataset)-(each_length*4)])
    target_train=torch.utils.data.Subset(dataset,[i for i in range(0,each_length*2)])
    target_test=torch.utils.data.Subset(dataset,[i for i in range(each_length*2,each_length*3)])
    shadow_train=torch.utils.data.Subset(dataset,[i for i in range(each_length*3,each_length*5)])
    shadow_test=torch.utils.data.Subset(dataset,[i for i in range(each_length*5,each_length*6)])

    return num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model


def get_model_dataset(dataset_name, root, model_name):
    if dataset_name.lower() == "fmnist":
        num_classes = 10
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_set = torchvision.datasets.FashionMNIST(
            root=root, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.FashionMNIST(
            root=root, train=False, download=True, transform=transform)

        dataset = train_set + test_set
        input_channel = 1


    elif dataset_name.lower() == "cifar10":
        num_classes = 10
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train_set = torchvision.datasets.CIFAR10(
            root=root, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR10(
            root=root, train=False, download=True, transform=transform)

        dataset = train_set + test_set
        input_channel = 3

    if isinstance(num_classes, int):
        if model_name == 'cnn':
            target_model = CNN_MIA(input_channel=input_channel, num_classes=num_classes)
            # target_model = CNN(num_classes=num_classes)
            shadow_model = CNN_MIA(input_channel=input_channel, num_classes=num_classes)
            # shadow_model = CNN(num_classes=num_classes)
    else:
        if model_name == 'cnn':
            target_model = CNN_MIA(input_channel=input_channel, num_classes=num_classes[0])
            # target_model = CNN(num_classes=num_classes[0])
            shadow_model = CNN_MIA(input_channel=input_channel, num_classes=num_classes[0])
            # shadow_model = CNN(num_classes=num_classes[0])

    return num_classes, dataset, target_model, shadow_model

if __name__=="__main__":
    dataset='fmnist'
    attr=1
    root="./data"
    model_name='cnn'
    num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model=prepare_dataset(dataset, attr, root, model_name)

    print(len(target_train))
    print(len(target_test))
    print(len(shadow_train))
    print(len(shadow_test))
