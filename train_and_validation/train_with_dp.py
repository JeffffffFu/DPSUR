import torch
from torch.utils.data import TensorDataset
import torch.nn.functional as F

from privacy_analysis.RDP.compute_rdp import compute_rdp
from train_and_validation.validation import validation, validation_per_sample
import numpy as np

from utils.NoisyMax import NoisyMax


def train_with_dp(model, train_loader, optimizer,device):
    model.train()
    train_loss = 0.0
    train_acc=0.
    for id,(data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_accum_grad()
        for iid,(X_microbatch, y_microbatch) in enumerate(TensorDataset(data, target)):

            optimizer.zero_microbatch_grad()
            output = model(torch.unsqueeze(X_microbatch, 0))

            if len(output.shape)==2:
                output=torch.squeeze(output,0)
            loss = F.cross_entropy(output, y_microbatch)  #改为负数似然损失函数了，后面记得要改回来

            loss.backward()
            optimizer.microbatch_step()
        optimizer.step_dp()


    return train_loss, train_acc



def train_with_dp_agd(model, train_loader, optimizer,C_t,sigma_t,C_v,sigma_v,device,batch_size,num):
    model.train()
    train_loss = 0.0
    train_acc=0.
    noise_multiplier = sigma_t
    RDP=0.
    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64))+ [128, 256, 512]

    for id,(data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_accum_grad()
        for iid,(X_microbatch, y_microbatch) in enumerate(TensorDataset(data, target)):

            optimizer.zero_microbatch_grad()
            output = model(torch.unsqueeze(X_microbatch, 0))

            if len(output.shape)==2:
                output=torch.squeeze(output,0)
            loss = F.cross_entropy(output, y_microbatch)

            loss.backward()
            optimizer.microbatch_step()


        # 获取原参数和裁剪的梯度值
        model_parameters_clipped = model.parameters()
        gradients_clipped = [param.grad.clone() for param in model_parameters_clipped]

        optimizer.step_dp_agd()   #只是进行了梯度加噪，没有进行梯度下降

        # 获取原参数和裁剪加噪平均后的梯度值
        model_parameters = model.parameters()
        gradients = [param.grad.clone() for param in model_parameters]

        model_parameters_dict = model.state_dict()

        learning_rate = np.linspace(0, 5, 5 + 1)  # 学习率从0-5.0分成5份
        min_index=0
        while min_index==0:
            loss = []

            for i,lr in enumerate(learning_rate):
                # 更新参数
                with torch.no_grad():

                    for param, gradient in zip(model_parameters_dict.values(), gradients):
                        param -= lr * gradient

                model.load_state_dict(model_parameters_dict)
                test_loss = validation_per_sample(model, train_loader, device,C_v)
                loss.append(test_loss)

            #找最小值jinx
            min_index=NoisyMax(loss,sigma_v,C_v,len(target))


            if min_index>0:
                #拿到使得这次loss最小的梯度值
                lr=learning_rate[min_index]
                with torch.no_grad():
                    for param, gradient in zip(model_parameters_dict.values(), gradients):
                        param -= lr * gradient
                model.load_state_dict(model_parameters_dict)
                RDP = RDP + compute_rdp(batch_size / num, sigma_v, 1,orders)  # 这个是进行loss计算，选择最佳学习率的RDP

            else:
                #如果是0最佳的，那么需要进行隐私预算加大，即多分配隐私预算，然后sigma变小
                # 对g进行重加噪，用小的sigma进行重加噪，然后隐私资源消耗
                noise_multiplier=sigma_t*0.99
                print("noise_multiplier:",noise_multiplier)
                with torch.no_grad():
                    for gradient1, gradient2 in zip(gradients_clipped, gradients):

                         gradient1+=C_t * noise_multiplier * torch.randn_like(gradient1)
                         gradient1/=len(target)
                         gradient2=(noise_multiplier*gradient1+sigma_t*gradient2)/(noise_multiplier+sigma_t)

                RDP = RDP + compute_rdp(batch_size / num, sigma_t, 1,orders)  # 这个是进行loss计算，选择最佳学习率的RDP


    return train_loss, train_acc,noise_multiplier,RDP
