# DPSUR


*DPSUR: Differentially Private Stochastic Gradient Descent based on Selective Updates and Release*</br>



## Installation

You can install all requirements with:
```bash
pip install -r requirements.txt
```

[//]: # (The code was tested with `python 3.7`, `torch 1.6` and `cpu`.)


## Results

This table presents the main results from our paper. For each dataset, we target the privacy budget `epsilon={1, 2, 3, 4}` and fixed `delta=1e-5`.
For all experiments, we report the average test acc of `5` independent trials.


| Dataset       | epsilon=1 | epsilon=2 | epsilon=3 | epsilon=4 |
|---------------|-----------|--------|-----------|-----------|
| MNIST         | 98.17%    | 98.74% | 98.95%    | 99.05% 
| Fashion-MNIST | 88.01%    | 89.16% | 89.90%    | 90.38%     
| CIFAR-10      | 64.41%    | 69.40% | 70.83%    | 71.45% 
| IMDB          | 66.50%    | 71.02% | 71.16%    | 73.84% 

## Parameters
During the DPSGD phase, for the three image datasets, we adopted the best parameters recommended 
in [Differentially Private Learning Needs Better Features (Or Much More Data)](http://arxiv.org/abs/2011.11660). Specifically, we fine-tuned the noise multiplier 
`sigma` for the various values of privacy budget `epsilon`, 
following the approach outlined in [Differentially Private Learning Needs Better Features (Or Much More Data)](http://arxiv.org/abs/2011.11660) and [DPIS: An Enhanced Mechanism for Differentially Private SGD with Importance Sampling
](https://arxiv.org/abs/2210.09634).
### MNIST

```bash
python main.py --algorithm DPSUR --dataset_name MNIST  --sigma 2.0 --lr 2.0 --batch_size 1024 --size_valid 5000 --C_v=0.001 --bs_valid=256 --beta=-1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --eps=1.0
python main.py --algorithm DPSUR --dataset_name MNIST  --sigma 1.5 --lr 2.0 --batch_size 1024 --size_valid 5000 --C_v=0.001 --bs_valid=256 --beta=-1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --eps=2.0
python main.py --algorithm DPSUR --dataset_name MNIST  --sigma 1.35 --lr 2.0 --batch_size 1024 --size_valid 5000 --C_v=0.001 --bs_valid=256 --beta=-1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --eps=3.0
python main.py --algorithm DPSUR --dataset_name MNIST  --sigma 1.35 --lr 2.0 --batch_size 1024 --size_valid 5000 --C_v=0.001 --bs_valid=256 --beta=-1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --eps=4.0
```

### FMNIST

```bash
python main.py --algorithm DPSUR --dataset_name FMNIST  --sigma 4.0 --lr 4.0 --batch_size 2048 --size_valid 5000 --C_v=0.001 --bs_valid=256 --beta=-1 --input_norm=GroupNorm --num_groups=8 --use_scattering --eps=1.0
python main.py --algorithm DPSUR --dataset_name FMNIST  --sigma 2.15 --lr 4.0 --batch_size 2048 --size_valid 5000 --C_v=0.001 --bs_valid=256 --beta=-1 --input_norm=GroupNorm --num_groups=8 --use_scattering --eps=2.0
python main.py --algorithm DPSUR --dataset_name FMNIST  --sigma 2.15 --lr 4.0 --batch_size 2048 --size_valid 5000 --C_v=0.001 --bs_valid=256 --beta=-1 --input_norm=GroupNorm --num_groups=8 --use_scattering --eps=3.0
python main.py --algorithm DPSUR --dataset_name FMNIST  --sigma 2.15 --lr 4.0 --batch_size 2048 --size_valid 5000 --C_v=0.001 --bs_valid=256 --beta=-1 --input_norm=GroupNorm --num_groups=8 --use_scattering --eps=4.0
```

### CIFAR10

```bash
python main.py --algorithm DPSUR --dataset_name CIFAR-10  --sigma 10.0 --lr 4.0 --batch_size 8192 --size_valid 5000 --C_v=0.001 --bs_valid=256 --beta=-1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --eps=1.0
python main.py --algorithm DPSUR --dataset_name CIFAR-10 --sigma 9.0 --lr 4.0 --batch_size 8192 --size_valid 5000 --C_v=0.001 --bs_valid=256 --beta=-1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --eps=2.0
python main.py --algorithm DPSUR --dataset_name CIFAR-10 --sigma 5.67 --lr 4.0 --batch_size 8192 --size_valid 5000 --C_v=0.001 --bs_valid=256 --beta=-1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --eps=3.0
python main.py --algorithm DPSUR --dataset_name CIFAR-10 --sigma 5.67 --lr 4.0 --batch_size 8192 --size_valid 5000 --C_v=0.001 --bs_valid=256 --beta=-1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --eps=4.0
```

### IMDB

To reproduce the results for linear ScatterNet models, run
```bash
python main.py --algorithm DPSUR --dataset_name IMDB  --sigma 2.0 --lr 0.02 --batch_size 1024 --size_valid 3000 --C_v=0.001 --bs_valid=256 --beta=-1 --eps=1.0
python main.py --algorithm DPSUR --dataset_name IMDB  --sigma 1.8 --lr 0.02 --batch_size 1024 --size_valid 3000 --C_v=0.001 --bs_valid=256 --beta=-1 --eps=2.0
python main.py --algorithm DPSUR --dataset_name IMDB  --sigma 1.35 --lr 0.02 --batch_size 1024 --size_valid 3000 --C_v=0.001 --bs_valid=256 --beta=-1 --eps=3.0
python main.py --algorithm DPSUR --dataset_name IMDB  --sigma 1.23 --lr 0.02 --batch_size 1024 --size_valid 3000 --C_v=0.001 --bs_valid=256 --beta=-1 --eps=4.0
```



Here few additional parameters are used in the DPSGD with ScatterNet, which is deived from [Differentially Private Learning Needs Better Features (Or Much More Data)](http://arxiv.org/abs/2011.11660) and 
[Code](https://github.com/ftramer/Handcrafted-DP). 
* The `input_norm` parameter determines how the ScatterNet features are normalized. 
We support Group Normalization (`input_norm=GN`) 
and (frozen) Batch Normalization (`input_norm=BN`).
* When using Group Normalization, the `num_groups` parameter specifies the number
of groups into which to split the features for normalization.
* When using Batch Normalization, we first privately compute the mean and variance
of the features across the entire training set. This requires adding noise to 
these statistics. The `bn_noise_multiplier` specifies the scale of the noise. 

When using Batch Normalization, we *compose* the privacy losses of the 
normalization step and of the DP-SGD algorithm.
Specifically, we first compute the Rényi-DP budget for the normalization step, 
and then compute the `noise_multiplier` of the DP-SGD algorithm so that the total
privacy budget is used after a fixed number of epochs.
