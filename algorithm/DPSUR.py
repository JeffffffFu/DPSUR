from data.util.get_data import get_scatter_transform, get_scattered_dataset, get_scattered_loader
from model.CNN import  CIFAR10_CNN_Tanh, MNIST_CNN_Tanh
from privacy_analysis.RDP.compute_dp_sgd import apply_dp_sgd_analysis
from privacy_analysis.RDP.compute_rdp import compute_rdp
from privacy_analysis.RDP.get_MaxSigma_or_MaxSteps import get_max_steps, get_min_sigma
from privacy_analysis.RDP.rdp_convert_dp import compute_eps
from privacy_analysis.dp_utils import scatter_normalization
from utils.dp_optimizer import DPSGD_Optimizer, DPAdam_Optimizer
import torch

from train_and_validation.train_with_dp import   train_with_dp
from train_and_validation.validation import validation
import copy
import numpy as np

from data.util.sampling import  get_data_loaders_possion

from data.util.dividing_validation_data import dividing_validation_set, dividing_validation_set_for_IMDB
import os
def DPSUR(dataset_name,train_dataset, test_data, model, batch_size, lr, momentum, epsilon_budget,delta, C_t, sigma,use_scattering,input_norm,bn_noise_multiplier,num_groups,size_valid,bs_valid,C_v,beta,device):

    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64))+ [128, 256, 512]
    if dataset_name=='IMDB':
        train_data, valid_data = dividing_validation_set_for_IMDB(train_dataset, size_valid)
    else:
        train_data, valid_data = dividing_validation_set(train_dataset, size_valid)
    test_dl = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, pin_memory=True)

    if dataset_name != 'IMDB':

        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

        valid_loader = torch.utils.data.DataLoader(
            valid_data, batch_size=bs_valid, shuffle=True, num_workers=1, pin_memory=True)

        if use_scattering:
            scattering, K, _ = get_scatter_transform(dataset_name)
            scattering.to(device)
        else:
            scattering = None
            K = 3 if len(train_dataset.data.shape) == 4 else 1

        rdp_norm=0.
        if input_norm == "BN":
            save_dir = f"bn_stats/{dataset_name}"
            os.makedirs(save_dir, exist_ok=True)
            bn_stats, rdp_norm = scatter_normalization(train_loader,
                                                       scattering,
                                                       K,
                                                       device,
                                                       len(train_data),
                                                       len(train_data),
                                                       noise_multiplier=bn_noise_multiplier,
                                                       orders=orders,
                                                       save_dir=save_dir)
            model = CNNS[dataset_name](K, input_norm="BN", bn_stats=bn_stats, size=None)
        else:
            model = CNNS[dataset_name](K, input_norm=input_norm, num_groups=num_groups, size=None)

        model.to(device)
        train_data_scattered = get_scattered_dataset(train_loader, scattering, device, len(train_data))
        valid_data_scattered = get_scattered_dataset(valid_loader, scattering, device, len(valid_data))
        test_dl = get_scattered_loader(test_dl, scattering, device)

        optimizer = DPSGD_Optimizer(
            l2_norm_clip=C_t,
            noise_multiplier=sigma,
            minibatch_size=batch_size,
            microbatch_size=1,
            params=model.parameters(),
            lr=lr,
            momentum=momentum
        )
    else:
        optimizer = DPAdam_Optimizer(
            l2_norm_clip=C_t,
            noise_multiplier=sigma,
            minibatch_size=batch_size,
            microbatch_size=1,
            params=model.parameters(),
            lr=lr)

    minibatch_loader_for_train, microbatch_loader = get_data_loaders_possion(minibatch_size=batch_size, microbatch_size=1, iterations=1)
    minibatch_loader_for_valid, microbatch_loader = get_data_loaders_possion(minibatch_size=bs_valid, microbatch_size=1, iterations=1)

    last_valid_loss = 100000.0
    last_accept_test_acc=0.
    last_model = model
    t = 1
    iter=1
    epsilon=0.

    epsilon_budget_for_train=epsilon_budget/(len(train_data)/len(train_dataset))
    print("privacy budget of training set:",epsilon_budget_for_train)

    epsilon_budget_for_valid_in_all_updates=epsilon_budget/(len(valid_data)/len(train_dataset))
    print("epsilon_budget_for_valid_in_all_updates:",epsilon_budget_for_valid_in_all_updates)

    max_number_of_updates = get_max_steps(epsilon_budget_for_train, delta, batch_size / len(train_data), sigma, orders)
    print("max_number_of_updates:", max_number_of_updates)

    epsilon_budget_for_train_in_one_iter,_=   apply_dp_sgd_analysis(batch_size / len(train_data), sigma, 1, orders,delta)
    print("epsilon_budget_for_train_in_one_iter:",epsilon_budget_for_train_in_one_iter)

    sigma_for_valid = get_min_sigma(epsilon_budget_for_valid_in_all_updates, len(train_data) / len(valid_data) * epsilon_budget_for_train_in_one_iter,delta, bs_valid / len(valid_data), max_number_of_updates,orders)

    while epsilon<epsilon_budget_for_train:

        if dataset_name=='IMDB':
            epsilon, best_alpha = apply_dp_sgd_analysis(batch_size / len(train_data), sigma, t, orders, delta)
            train_dl = minibatch_loader_for_train(train_data)
            valid_dl = minibatch_loader_for_valid(valid_data)

        else:
            if input_norm == "BN":
                rdp = compute_rdp(batch_size / len(train_data), sigma, t, orders)
                epsilon, best_alpha = compute_eps(orders, rdp + rdp_norm, delta)

            else:
                epsilon, best_alpha = apply_dp_sgd_analysis(batch_size / len(train_data), sigma, t, orders, delta)

            train_dl = minibatch_loader_for_train(train_data_scattered)
            valid_dl = minibatch_loader_for_valid(valid_data_scattered)


        train_loss, train_accuracy = train_with_dp(model, train_dl, optimizer,device)

        valid_loss, valid_accuracy = validation(model, valid_dl,device)

        test_loss, test_accuracy = validation(model, test_dl,device)


        deltaE=valid_loss - last_valid_loss
        deltaE=torch.tensor(deltaE).cpu()
        print("Delta E:",deltaE)

        deltaE= np.clip(deltaE,-C_v,C_v)
        deltaE_after_dp =2*C_v*sigma_for_valid*np.random.normal(0,1)+deltaE

        print("Delta E after dp:",deltaE_after_dp)

        if deltaE_after_dp < beta*C_v:
            last_valid_loss = valid_loss
            last_model = copy.deepcopy(model)
            t = t + 1
            print("accept updates，the number of updates t：", format(t))
            last_accept_test_acc=test_accuracy

        else:
            print("reject updates")
            model.load_state_dict(last_model.state_dict(), strict=True)

        print(f'iters:{iter},'f'epsilon:{epsilon:.4f} |'f' Test set: Average loss: {test_loss:.4f},'f' Accuracy:({test_accuracy:.2f}%)')

        iter+=1


    print("------ finished ------")
    return last_accept_test_acc

CNNS = {
    "CIFAR-10": CIFAR10_CNN_Tanh,
    "FMNIST": MNIST_CNN_Tanh,
    "MNIST": MNIST_CNN_Tanh,
}
