import torch.nn.functional as F


def train(model, train_loader, optimizer):
    model.train()
    train_loss = 0.0
    train_acc= 0.0
    i=0
<<<<<<< HEAD

=======
>>>>>>> fdc65a3 (DPSUR)
    for id, (data, target) in enumerate(train_loader):

        output = model(data)
        loss = F.cross_entropy(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



    return train_loss,train_acc
