import torch

from data.util.sampling import get_data_loaders_possion
from privacy_analysis.RDP.compute_dp_sgd import apply_dp_sgd_analysis
from privacy_analysis.RDP.compute_rdp import compute_rdp
from privacy_analysis.RDP.rdp_convert_dp import compute_eps
from train_and_validation.train import train
from train_and_validation.train_with_dp import train_with_dp, train_with_dp_agd
from train_and_validation.validation import validation


def DPAGD(train_data, test_data, model,optimizer, batch_size, epsilon_budget, delta,sigma,C_for_loss,sigma_for_loss,device):

    minibatch_loader, microbatch_loader = get_data_loaders_possion(minibatch_size=batch_size,microbatch_size=1,iterations=1)

    test_dl = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False)

    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64))+ [128, 256, 512]
    iter = 1
    epsilon = 0.
    best_test_acc=0.
    RDP=0.
    epsilon_list=[]
    test_loss_list=[]
    while epsilon < epsilon_budget:

        # epsilon, best_alpha = apply_dp_sgd_analysis(batch_size / len(train_data), sigma, iter, orders, delta) #comupte privacy cost
        RDP1 =  compute_rdp(batch_size / len(train_data), sigma, iter, orders)  #这个是train的RDP


        train_dl = minibatch_loader(train_data)  # possion sampling
        for id, (data, target) in enumerate(train_dl):
            optimizer.minibatch_size = len(data)

        train_loss, train_accuracy,noise_multiper,RDP2 = train_with_dp_agd(model, train_dl, optimizer,optimizer.l2_norm_clip,sigma,C_for_loss,sigma_for_loss,device,batch_size,len(train_data))
        RDP+=RDP2
        sigma=noise_multiper
        optimizer.noise_multiplier=noise_multiper
        test_loss, test_accuracy = validation(model, test_dl,device)
        epsilon,best_alpha=compute_eps(orders, RDP1+RDP, delta)

        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            best_iter = iter

        epsilon_list.append(torch.tensor(epsilon))
        test_loss_list.append(test_loss)
        print(f'iters:{iter},'f'epsilon:{epsilon:.4f} |'f' Test set: Average loss: {test_loss:.4f},'f' Accuracy:({test_accuracy:.2f}%)')
        iter += 1

    print("------finished ------")
    return test_accuracy,iter,best_test_acc,best_iter,model,[epsilon_list,test_loss_list]
