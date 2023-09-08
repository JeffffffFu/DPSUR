

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset



def validation(model, test_loader,device):
    model.eval()
    num_examples = 0
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            test_loss += F.cross_entropy(output, target, reduction='sum')

            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            num_examples += len(data)
    test_loss /= num_examples
    test_acc = 100. * correct / num_examples

    return test_loss, test_acc



def validation_per_sample(model, test_loader,device,C):
    model.eval()
    num_examples = 0
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data,target=data.to(device),target.to(device)
            for x,y in TensorDataset(data,target):
                output = model(torch.unsqueeze(x,0))
                if len(output.shape)==2:
                    output=torch.squeeze(output,0)
                loss=F.cross_entropy(output, y, reduction='sum')
                loss=min(loss,C)   #逐样本loss裁剪
                test_loss += loss
            num_examples += len(data)
    test_loss /= num_examples

    # print(f'Test set: Average loss: {test_loss:.4f}, '
    #       f'Accuracy: {correct}/{num_examples} ({test_acc:.2f}%)')

    return test_loss
