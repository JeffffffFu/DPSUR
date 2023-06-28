from data.util.get_data import get_scatter_transform, get_scattered_dataset, get_scattered_loader
from model.CNN import CIFAR10_CNN_Tanh, MNIST_CNN_Tanh
from privacy_analysis.RDP.compute_dp_sgd import apply_dp_sgd_analysis
from privacy_analysis.RDP.compute_rdp import compute_rdp
from privacy_analysis.RDP.get_MaxSigma_or_MaxSteps import get_noise_multiplier
from privacy_analysis.RDP.rdp_convert_dp import compute_eps
from privacy_analysis.dp_utils import scatter_normalization
from train_and_validation.train import train
from train_and_validation.train_with_dp import train_with_dp
from utils.dp_optimizer import  DPSGD_Optimizer
import torch

from train_and_validation.validation import validation


from data.util.sampling import get_data_loaders_possion
import os

def DPSGD_HF(dataset_name,train_data, test_data, model, batch_size, lr, momentum, epsilon_budget,delta, C_t, sigma,use_scattering,input_norm,bn_noise_multiplier,num_groups,device):

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, pin_memory=True)


    if use_scattering:
        scattering, K, _ = get_scatter_transform(dataset_name)
        scattering.to(device)
    else:
        scattering = None
        K = 3 if len(train_data.data.shape) == 4 else 1

    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64))+ [128, 256, 512]
    rdp_norm=0.
    if input_norm == "BN":
        # compute noisy data statistics or load from disk if pre-computed
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
    test_loader = get_scattered_loader(test_loader, scattering, device)

    minibatch_loader, microbatch_loader = get_data_loaders_possion(minibatch_size=batch_size, microbatch_size=1, iterations=1)


    optimizer = DPSGD_Optimizer(
        l2_norm_clip=C_t,
        noise_multiplier=sigma,
        minibatch_size=batch_size,
        microbatch_size=1,
        params=model.parameters(),
        lr=lr,
        momentum=momentum,
    )

    iter = 1
    epsilon=0.
    while epsilon<epsilon_budget:
        if input_norm == "BN":
            rdp = compute_rdp(batch_size / len(train_data), sigma, iter, orders)
            epsilon, best_alpha = compute_eps(orders, rdp + rdp_norm, delta)

        else:
            epsilon, best_alpha = apply_dp_sgd_analysis(batch_size / len(train_data), sigma, iter, orders, delta)

        train_dl = minibatch_loader(train_data_scattered)
        for id, (data, target) in enumerate(train_dl):
            optimizer.minibatch_size = len(data)

        train_loss, train_accuracy = train_with_dp(model, train_dl, optimizer,device)

        test_loss, test_accuracy = validation(model, test_loader,device)

        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            best_iter = iter

        print(f'iters:{iter},'f'epsilon:{epsilon:.4f} |'f' Test set: Average loss: {test_loss:.4f},'f' Accuracy:({test_accuracy:.2f}%)')
        iter+=1


    print("------ finished ------")
    return test_accuracy,iter,best_test_acc,best_iter,model

CNNS = {
    "CIFAR-10": CIFAR10_CNN_Tanh,
    "FMNIST": MNIST_CNN_Tanh,
    "MNIST": MNIST_CNN_Tanh,
}
