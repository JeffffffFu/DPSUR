import itertools

from torch.utils.data import TensorDataset, DataLoader

import os
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import seaborn as sns

def text1():

    directory = os.getcwd() + f"/data/MNIST_DPGEN//"  # 项目所在的目录
    keyword = 'label'  # 要删除的文件扩展名

    for root, dirs, files in os.walk(directory):
        for filename in files:
            # 判断文件名是否包含指定的关键字
            if keyword not in filename:
                file_path = os.path.join(root, filename)
                # 询问用户是否要删除文件
                # answer = input("Do you want to delete file {}? [y/n]".format(file_path))
                # if answer.lower() == 'y':
                    # 删除文件
                os.remove(file_path)

def text2():

    data_trainsforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([.5, .5, .5], [.5, .5, .5])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([.5, .5, .5], [.5, .5, .5])
        ])
    }
    root = os.getcwd() + f"/data/MNIST_DPGEN/"  # 项目所在的目录

    # ImageFolder 通用的加载器
    dataset = torchvision.datasets.ImageFolder(root, transform=data_trainsforms['train'])
    # 构建可迭代的数据装载器
    inputs = DataLoader(dataset=dataset, batch_size=128, shuffle=True, num_workers=1)
    for data, label in inputs:
        plt.imshow(data, cmap='gray')
        print(label)

def text3():
    # 加载.pt文件
    root_images = os.getcwd() + f"/data/MNIST_DPGEN/gen_images.pt"  # 项目所在的目录
    gen_images = torch.load(root_images,map_location=torch.device('cpu'))

    root_labels = os.getcwd() + f"/data/MNIST_DPGEN/gen_labels.pt"  # 项目所在的目录
    gen_labels = torch.load(root_labels,map_location=torch.device('cpu'))

    dataset = TensorDataset(gen_images, gen_labels)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 输出模型参数
    # 迭代数据集
    for batch_data, batch_labels in dataloader:
        print(batch_labels[0])
        img = np.transpose(batch_data[0], (1, 2, 0))

        plt.imshow(img, cmap='gray')
        plt.show()

#打印我们存储的样本序号和出现的频率
def text4():

    index=torch.load("C:\python flie\SA-DPSGD-master/result\csv\SA_DPSGD_loss_add_dp_scatter/fmnist/3.0/0.7\-0.001/2.15_5.01_2048_5000_256_0.59_index.pt")
    print(len(index))
    flat_list = list(itertools.chain.from_iterable(index))
    # 将PyTorch张量转换为int
    my_list = [int(x) for x in flat_list]

    counts = Counter(my_list)
    # for i in range(55000):
    #     print(counts[i])

    #获取元素和出现次数的列表
    elements = counts.keys()
    frequencies = counts.values()
    # #绘制条形图
    plt.bar(elements, frequencies)
    # # 添加标题和标签
    plt.title("Sampling Frequencies of FMNIST")
    plt.xlabel("Record")
    plt.ylabel("Frequencies")

    # 显示图像
    plt.show()




if __name__ == '__main__':
    test()
