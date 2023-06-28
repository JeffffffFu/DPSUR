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


def DPSUR(dataset_name,train_dataset, test_data, model, batch_size, lr, momentum, epsilon_budget,delta, C_t, sigma_t,use_scattering,input_norm,bn_noise_multiplier,num_groups,bs_valid,C_v,beta,sigma_v,device):

    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64))+ [128, 256, 512]

    test_dl = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, pin_memory=True)

    if dataset_name != 'IMDB':

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)


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
                                                       len(train_dataset),
                                                       len(train_dataset),
                                                       noise_multiplier=bn_noise_multiplier,
                                                       orders=orders,
                                                       save_dir=save_dir)
            model = CNNS[dataset_name](K, input_norm="BN", bn_stats=bn_stats, size=None)


        else:
            model = CNNS[dataset_name](K, input_norm=input_norm, num_groups=num_groups, size=None)

        model.to(device)
        train_data_scattered = get_scattered_dataset(train_loader, scattering, device, len(train_dataset))
        test_dl = get_scattered_loader(test_dl, scattering, device)

        optimizer = DPSGD_Optimizer(
            l2_norm_clip=C_t,
            noise_multiplier=sigma_t,
            minibatch_size=batch_size,
            microbatch_size=1,
            params=model.parameters(),
            lr=lr,
            momentum=momentum
        )
    else:
        optimizer = DPAdam_Optimizer(
            l2_norm_clip=C_t,
            noise_multiplier=sigma_t,
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
    best_iter=1
    best_test_acc=0.
    epsilon=0.

    while epsilon<epsilon_budget:

        if dataset_name=='IMDB':
            rdp_train = compute_rdp(batch_size / len(train_dataset), sigma_t, t, orders)
            rdp_valid = compute_rdp(bs_valid / len(train_dataset), sigma_v, t, orders)
            epsilon, best_alpha = compute_eps(orders, rdp_train + rdp_valid, delta)

            train_dl = minibatch_loader_for_train(train_dataset)
            valid_dl = minibatch_loader_for_valid(train_dataset)
            for id, (data, target) in enumerate(train_dl):
                optimizer.minibatch_size = len(data)

        else:
            if input_norm == "BN":
                rdp_train = compute_rdp(batch_size / len(train_dataset), sigma_t, t, orders)
                rdp_valid = compute_rdp(bs_valid / len(train_dataset), sigma_v, t, orders)

                epsilon, best_alpha = compute_eps(orders, rdp_train + rdp_valid+ rdp_norm, delta)

            else:
                rdp_train = compute_rdp(batch_size / len(train_dataset), sigma_t, t, orders)
                rdp_valid = compute_rdp(bs_valid / len(train_dataset), sigma_v, t, orders)
                epsilon, best_alpha = compute_eps(orders, rdp_train+rdp_valid, delta)

            train_dl = minibatch_loader_for_train(train_data_scattered)
            valid_dl = minibatch_loader_for_valid(train_data_scattered)
            for id, (data, target) in enumerate(train_dl):
                optimizer.minibatch_size = len(data)

        train_loss, train_accuracy = train_with_dp(model, train_dl, optimizer,device)

        valid_loss, valid_accuracy = validation(model, valid_dl,device)

        test_loss, test_accuracy = validation(model, test_dl,device)


        deltaE=valid_loss - last_valid_loss
        deltaE=torch.tensor(deltaE).cpu()
        print("Delta E:",deltaE)

        deltaE= np.clip(deltaE,-C_v,C_v)
        deltaE_after_dp =2*C_v*sigma_v*np.random.normal(0,1)+deltaE

        print("Delta E after dp:",deltaE_after_dp)

        if deltaE_after_dp < beta*C_v:
            last_valid_loss = valid_loss
            last_model = copy.deepcopy(model)
            t = t + 1
            print("accept updates，the number of updates t：", format(t))
            last_accept_test_acc=test_accuracy

            if last_accept_test_acc > best_test_acc:
                best_test_acc = last_accept_test_acc
                best_iter = t

        else:
            print("reject updates")
            model.load_state_dict(last_model.state_dict(), strict=True)

        print(f'iters:{iter},'f'epsilon:{epsilon:.4f} |'f' Test set: Average loss: {test_loss:.4f},'f' Accuracy:({test_accuracy:.2f}%)')

        iter+=1


    print("------ finished ------")
    return last_accept_test_acc,t,best_test_acc,best_iter,last_model

CNNS = {
    "CIFAR-10": CIFAR10_CNN_Tanh,
    "FMNIST": MNIST_CNN_Tanh,
    "MNIST": MNIST_CNN_Tanh,
}
