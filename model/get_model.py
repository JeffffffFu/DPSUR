from model.CNN import CIFAR10_CNN_Tanh, CIFAR10_CNN_Relu, MNIST_CNN_Relu, MNIST_CNN_Tanh, CNN333
from model.RNN import RNN_Tanh, RNN_Relu
import transformers

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
        elif dataset_name =='CIFAR-10-Transformers':
                 #https://github.com/lxuechen/private-transformers
                model_name_or_path = 'google/vit-base-patch16-224'
                config = transformers.AutoConfig.from_pretrained(model_name_or_path)
                config.num_labels = 10

                model = transformers.ViTForImageClassification.from_pretrained(
                    model_name_or_path,
                    config=config,
                    ignore_mismatched_sizes=True  # Default pre-trained model has 1k classes; we only have 10.
                )

                model.requires_grad_(False)
                model.classifier.requires_grad_(True)

        elif dataset_name =='CIFAR-100-Transformers':
                model_name_or_path = 'google/vit-base-patch16-224'
                config = transformers.AutoConfig.from_pretrained(model_name_or_path)
                config.num_labels = 100

                model = transformers.ViTForImageClassification.from_pretrained(
                    model_name_or_path,
                    config=config,
                    ignore_mismatched_sizes=True  # Default pre-trained model has 1k classes; we only have 10.
                )

                model.requires_grad_(False)
                model.classifier.requires_grad_(True)


    model.to(device=device)

    return model