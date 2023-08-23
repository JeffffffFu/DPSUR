import torch
import torch.distributions as dist
import numpy as np
def NoisyMax(list,sigma,C,n):

    # 找最小值
    min_loss = min(list)
    # 找到最小值所在位置的索引
    min_index = list.index(min_loss)
    print(f'min_loss:{min_loss},min_index:{min_index}')

    #注意，因为损失函数是非负的，当多一个样本时，对应的损失函数累计之和肯定大于等于之前的，所以是单调的，这里可以用单边的指数函数替代。
    laplace_noise = torch.tensor(np.random.exponential(C*sigma, size=len(list)), dtype=torch.float32, device='cpu')


    # 生成与 A 矩阵形状相同的拉普拉斯噪声
    noised_list =  torch.tensor(list) + laplace_noise/n

    # 找最小值
    min_value, min_index = torch.min(noised_list, dim=0)

    print(f'min_value:{min_value},min_index:{min_index}')

    return min_index

