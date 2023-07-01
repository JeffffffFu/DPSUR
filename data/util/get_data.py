import torch
from torchvision import datasets, transforms
from kymatio.torch import Scattering2D
import os
import pickle
import numpy as np
import logging
from torch.utils.data import TensorDataset
from keras.datasets import imdb
from keras.utils import pad_sequences

SHAPES = {
    "CIFAR-10": (32, 32, 3),
    "cifar10_500K": (32, 32, 3),
    "FMNIST": (28, 28, 1),
    "MNIST": (28, 28, 1)
}

def get_scatter_transform(dataset):
    shape = SHAPES[dataset]
    scattering = Scattering2D(J=2, shape=shape[:2])
    K = 81 * shape[2]
    (h, w) = shape[:2]
    return scattering, K, (h//4, w//4)

def get_data(name, augment=False, **kwargs):
    if name == "CIFAR-10":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        if augment:
            train_transforms = [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize,
            ]
        else:
            train_transforms = [
                transforms.ToTensor(),
                normalize,

            ]

        train_set = datasets.CIFAR10(root="./data", train=True,
                                     transform=transforms.Compose(train_transforms),
                                     download=True)

        test_set = datasets.CIFAR10(root="./data", train=False,
                                    transform=transforms.Compose(
                                        [transforms.ToTensor(), normalize]
                                    ))
        dataset=train_set+test_set

    elif name == "CIFAR-10-Transformers":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        if augment:
            train_transforms = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]
        else:

            train_transforms = [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ]

        train_set = datasets.CIFAR10(root="./data", train=True,
                                     transform=transforms.Compose(train_transforms),
                                     download=True)

        test_set = datasets.CIFAR10(root="./data", train=False,
                                    transform=transforms.Compose(
                                        train_transforms
                                    ))
        dataset=train_set+test_set

    elif name == "CIFAR-100-Transformers":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        if augment:
            train_transforms = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]
        else:

            train_transforms = [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ]

        train_set = datasets.CIFAR100(root="./data", train=True,
                                     transform=transforms.Compose(train_transforms),
                                     download=True)

        test_set = datasets.CIFAR100(root="./data", train=False,
                                    transform=transforms.Compose(
                                        train_transforms
                                    ))
        dataset=train_set+test_set

    elif name == "SVHN":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        if augment:
            train_transforms = [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize,
                ]
        else:
            train_transforms = [
                transforms.ToTensor(),
                normalize,
            ]

        train_set = datasets.SVHN(root="./data", train=True,
                                     transform=transforms.Compose(train_transforms),
                                     download=True)

        test_set = datasets.SVHN(root="./data", train=False,
                                    transform=transforms.Compose(
                                        [transforms.ToTensor(), normalize]
                                    ))
        dataset=train_set+test_set

    elif name == "FMNIST":
        train_set = datasets.FashionMNIST(root='./data', train=True,
                                          transform=transforms.ToTensor(),
                                          download=True)

        test_set = datasets.FashionMNIST(root='./data', train=False,
                                         transform=transforms.ToTensor(),
                                         download=True)
        dataset=train_set+test_set

    elif name == "MNIST":
        train_set = datasets.MNIST(root='./data', train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)

        test_set = datasets.MNIST(root='./data', train=False,
                                  transform=transforms.ToTensor(),
                                  download=True)
        dataset=train_set+test_set

    elif name == "EMNIST":
        train_set = datasets.EMNIST(root='./data', split='letters',train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)

        test_set = datasets.EMNIST(root='./data', split='letters',train=False,
                                  transform=transforms.ToTensor(),
                                  download=True)

        dataset=train_set+test_set


    elif name == "cifar10_500K":

        # extended version of CIFAR-10 with pseudo-labelled tinyimages

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        if augment:
            train_transforms = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]
        else:
            train_transforms = [
                transforms.ToTensor(),
                normalize,
            ]

        train_set = SemiSupervisedDataset(kwargs['aux_data_filename'],
                                          root="../data",
                                          train=True,
                                          download=True,
                                          transform=transforms.Compose(train_transforms))
        test_set = None
        dataset=train_set+test_set

    elif name == 'IMDB':
        MAX_WORDS = 20000
        MAX_LEN = 80
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_WORDS)
        x_train = pad_sequences(x_train, maxlen=MAX_LEN, padding='post', truncating='post')
        x_test = pad_sequences(x_test, maxlen=MAX_LEN, padding='post', truncating='post')

        train_set = TensorDataset(torch.LongTensor(x_train), torch.LongTensor(y_train))
        test_set = TensorDataset(torch.LongTensor(x_test), torch.LongTensor(y_test))
        dataset=train_set+test_set

    else:
        raise ValueError(f"unknown dataset {name}")

    return train_set, test_set,dataset

class SemiSupervisedDataset(torch.utils.data.Dataset):
    def __init__(self,
                 aux_data_filename=None,
                 train=False,
                 **kwargs):
        """A dataset with auxiliary pseudo-labeled data"""

        self.dataset = datasets.CIFAR10(train=train, **kwargs)
        self.train = train

        # shuffle cifar-10
        p = np.random.permutation(len(self.data))
        self.data = self.data[p]
        self.targets = list(np.asarray(self.targets)[p])

        if self.train:
            self.sup_indices = list(range(len(self.targets)))
            self.unsup_indices = []

            aux_path = os.path.join(kwargs['root'], aux_data_filename)
            print("Loading data from %s" % aux_path)
            with open(aux_path, 'rb') as f:
                aux = pickle.load(f)
            aux_data = aux['data']
            aux_targets = aux['extrapolated_targets']
            orig_len = len(self.data)

            # shuffle additional data
            p = np.random.permutation(len(aux_data))
            aux_data = aux_data[p]
            aux_targets = aux_targets[p]

            self.data = np.concatenate((self.data, aux_data), axis=0)
            self.targets.extend(aux_targets)

            # note that we use unsup indices to track the labeled datapoints
            # whose labels are "fake"
            self.unsup_indices.extend(
                range(orig_len, orig_len+len(aux_data)))

            logger = logging.getLogger()
            logger.info("Training set")
            logger.info("Number of training samples: %d", len(self.targets))
            logger.info("Number of supervised samples: %d",
                        len(self.sup_indices))
            logger.info("Number of unsup samples: %d", len(self.unsup_indices))
            logger.info("Label (and pseudo-label) histogram: %s",
                        tuple(
                            zip(*np.unique(self.targets, return_counts=True))))
            logger.info("Shape of training data: %s", np.shape(self.data))

        # Test set
        else:
            self.sup_indices = list(range(len(self.targets)))
            self.unsup_indices = []

            logger = logging.getLogger()
            logger.info("Test set")
            logger.info("Number of samples: %d", len(self.targets))
            logger.info("Label histogram: %s",
                        tuple(
                            zip(*np.unique(self.targets, return_counts=True))))
            logger.info("Shape of data: %s", np.shape(self.data))

    @property
    def data(self):
        return self.dataset.data

    @data.setter
    def data(self, value):
        self.dataset.data = value

    @property
    def targets(self):
        return self.dataset.targets

    @targets.setter
    def targets(self, value):
        self.dataset.targets = value

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        self.dataset.labels = self.targets  # because torchvision is annoying
        return self.dataset[item]

    def __repr__(self):
        fmt_str = 'Semisupervised Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Training: {}\n'.format(self.train)
        fmt_str += '    Root Location: {}\n'.format(self.dataset.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.dataset.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.dataset.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

class PoissonSampler(torch.utils.data.Sampler):
    def __init__(self, num_examples, batch_size):
        self.inds = np.arange(num_examples)
        self.batch_size = batch_size
        self.num_batches = int(np.ceil(num_examples / batch_size))
        self.sample_rate = self.batch_size / (1.0 * num_examples)
        super().__init__(None)

    def __iter__(self):
        # select each data point independently with probability `sample_rate`
        for i in range(self.num_batches):
            batch_idxs = np.random.binomial(n=1, p=self.sample_rate, size=len(self.inds))
            batch = self.inds[batch_idxs.astype(np.bool)]
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches

def get_scattered_dataset(loader, scattering, device, data_size):
    # pre-compute a scattering transform (if there is one) and return
    # a TensorDataset

    scatters = []
    targets = []

    num = 0
    for (data, target) in loader:
        data, target = data.to(device), target.to(device)
        if scattering is not None:
            data = scattering(data)
        scatters.append(data)
        targets.append(target)

        num += len(data)
        if num > data_size:
            break

    scatters = torch.cat(scatters, axis=0)
    targets = torch.cat(targets, axis=0)

    scatters = scatters[:data_size]
    targets = targets[:data_size]

    data = torch.utils.data.TensorDataset(scatters, targets)
    return data

def get_scattered_loader(loader, scattering, device, drop_last=False, sample_batches=False):
    # pre-compute a scattering transform (if there is one) and return
    # a DataLoader

    scatters = []
    targets = []

    for (data, target) in loader:
        data, target = data.to(device), target.to(device)
        if scattering is not None:
            data = scattering(data)
        scatters.append(data)
        targets.append(target)

    scatters = torch.cat(scatters, axis=0)
    targets = torch.cat(targets, axis=0)

    data = torch.utils.data.TensorDataset(scatters, targets)

    if sample_batches:
        sampler = PoissonSampler(len(scatters), loader.batch_size)
        return torch.utils.data.DataLoader(data, batch_sampler=sampler,
                                           num_workers=0, pin_memory=False)
    else:
        shuffle = isinstance(loader.sampler, torch.utils.data.RandomSampler)
        return torch.utils.data.DataLoader(data,
                                           batch_size=loader.batch_size,
                                           shuffle=shuffle,
                                           num_workers=0,
                                           pin_memory=False,
                                           drop_last=drop_last)