import torch

from data.util.sampling import get_data_loaders_possion
from privacy_analysis.RDP.compute_dp_sgd import apply_dp_sgd_analysis
from train_and_validation.train import train
from train_and_validation.train_with_dp import  train_with_dp
from train_and_validation.validation import validation


def DPSGD(train_data, test_data, model,optimizer, batch_size, epsilon_budget, delta,sigma,device):

    minibatch_loader, microbatch_loader = get_data_loaders_possion(minibatch_size=batch_size,microbatch_size=1,iterations=1)

    test_dl = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False)

    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64))+ [128, 256, 512]
    iter = 1
    epsilon = 0.
    best_test_acc=0.
    epsilon_list=[]
    test_loss_list=[]

    while epsilon < epsilon_budget:

        epsilon, best_alpha = apply_dp_sgd_analysis(batch_size / len(train_data), sigma, iter, orders, delta) #comupte privacy cost
        train_dl = minibatch_loader(train_data)  # possion sampling
        for id, (data, target) in enumerate(train_dl):
            optimizer.minibatch_size = len(data)

        train_loss, train_accuracy = train_with_dp(model, train_dl, optimizer,device)

        test_loss, test_accuracy = validation(model, test_dl,device)

        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            best_iter = iter
        epsilon_list.append(torch.tensor(epsilon))
        test_loss_list.append(test_loss)
        print(f'iters:{iter},'f'epsilon:{epsilon:.4f} |'f' Test set: Average loss: {test_loss:.4f},'f' Accuracy:({test_accuracy:.2f}%)')
        iter += 1

    print("------finished ------")
    return test_accuracy,iter,best_test_acc,best_iter,model,[epsilon_list,test_loss_list]
