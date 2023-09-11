import torch
from torch.utils.data import TensorDataset
import torch.nn.functional as F

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
            loss = F.cross_entropy(output, y_microbatch)

            loss.backward()
            optimizer.microbatch_step()
        optimizer.step_dp()


    return train_loss, train_acc



def train_with_dp_agd(model, train_loader, optimizer,device):
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
            loss = F.cross_entropy(output, y_microbatch)

            loss.backward()
            optimizer.microbatch_step()

        # 这里的核心在于裁剪
        # 保存模型参数
        # 把step_dp里的step拿掉
        # 循环学习率，每次执行step后进行梯度下降
        # 每次学习率得到的新模型求Loss然后保存
        optimizer.step_dp()


    return train_loss, train_acc
