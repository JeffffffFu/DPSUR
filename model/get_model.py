from model.CNN import CIFAR10_CNN_Tanh, CIFAR10_CNN_Relu, MNIST_CNN_Relu, MNIST_CNN_Tanh
from model.RNN import RNN_Tanh, RNN_Relu


def get_model(algorithm,dataset_name,device):

    if algorithm == 'DPSGD':
        if dataset_name == 'MNIST' or dataset_name == 'FMNIST':
                model = MNIST_CNN_Relu(1)
        elif dataset_name == 'CIFAR-10':
                model = CIFAR10_CNN_Relu(3)
        elif dataset_name == 'IMDB':
                model = RNN_Relu()
    else:
        if dataset_name == 'MNIST' or dataset_name == 'FMNIST':
                model = MNIST_CNN_Tanh(1)
        elif dataset_name == 'CIFAR-10':
                model = CIFAR10_CNN_Tanh(3)
        elif dataset_name == 'IMDB':
                model = RNN_Tanh()

    model.to(device=device)

    return model