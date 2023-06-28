##MNIST
python main.py --algorithm DPSUR --dataset_name MNIST  --sigma_t 2.0 --lr 2.0 --batch_size 1024  --C_v=0.001 --sigma_v=1.3 --bs_valid=256 --beta=-1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --eps=1.0
python main.py --algorithm DPSUR --dataset_name MNIST  --sigma_t 1.5 --lr 2.0 --batch_size 1024  --C_v=0.001 --sigma_v=1.0 --bs_valid=256 --beta=-1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --eps=2.0
python main.py --algorithm DPSUR --dataset_name MNIST  --sigma_t 1.35 --lr 2.0 --batch_size 1024 --C_v=0.001 --sigma_v=0.9 --bs_valid=256 --beta=-1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --eps=3.0
python main.py --algorithm DPSUR --dataset_name MNIST  --sigma_t 1.35 --lr 2.0 --batch_size 1024 --C_v=0.001 --sigma_v=0.8 --bs_valid=256 --beta=-1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --eps=4.0
#
##FMNIST
python main.py --algorithm DPSUR --dataset_name FMNIST  --sigma 4.0 --lr 4.0  --batch_size 2048 --C_v=0.001 --sigma_v=1.3 --bs_valid=256 --beta=-1 --input_norm=GroupNorm --num_groups=27 --use_scattering --eps=1.0
python main.py --algorithm DPSUR --dataset_name FMNIST  --sigma 2.15 --lr 4.0 --batch_size 2048 --C_v=0.001 --sigma_v=1.3 --bs_valid=256 --beta=-1 --input_norm=GroupNorm --num_groups=27 --use_scattering --eps=2.0
python main.py --algorithm DPSUR --dataset_name FMNIST  --sigma 2.15 --lr 4.0 --batch_size 2048 --C_v=0.001 --sigma_v=0.8 --bs_valid=256 --beta=-1 --input_norm=GroupNorm --num_groups=27 --use_scattering --eps=3.0
python main.py --algorithm DPSUR --dataset_name FMNIST  --sigma 2.15 --lr 4.0 --batch_size 2048 --C_v=0.001 --sigma_v=0.8 --bs_valid=256 --beta=-1 --input_norm=GroupNorm --num_groups=27 --use_scattering --eps=4.0

#CIFAR-10
python main.py --algorithm DPSUR --dataset_name CIFAR-10 --sigma 10.0 --lr 4.0 --batch_size 8192 --C_v=0.001 --sigma_v=1.3 --bs_valid=256 --beta=-1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --eps=1.0
python main.py --algorithm DPSUR --dataset_name CIFAR-10 --sigma 9.0  --lr 4.0 --batch_size 8192 --C_v=0.001 --sigma_v=1.3 --bs_valid=256 --beta=-1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --eps=2.0
python main.py --algorithm DPSUR --dataset_name CIFAR-10 --sigma 5.67 --lr 4.0 --batch_size 8192 --C_v=0.001 --sigma_v=1.1 --bs_valid=256 --beta=-1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --eps=3.0
python main.py --algorithm DPSUR --dataset_name CIFAR-10 --sigma 5.67 --lr 4.0 --batch_size 8192 --C_v=0.001 --sigma_v=1.1 --bs_valid=256 --beta=-1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --eps=4.0

#IMDB
python main.py --algorithm DPSUR --dataset_name IMDB  --sigma 2.0 --lr 0.02  --batch_size 1024 --C_v=0.001 --sigma_v=1.3 --bs_valid=256 --beta=-1 --eps=1.0
python main.py --algorithm DPSUR --dataset_name IMDB  --sigma 1.8 --lr 0.02  --batch_size 1024 --C_v=0.001 --sigma_v=1.2 --bs_valid=256 --beta=-1 --eps=2.0
python main.py --algorithm DPSUR --dataset_name IMDB  --sigma 1.35 --lr 0.02 --batch_size 1024 --C_v=0.001 --sigma_v=1.0 --bs_valid=256 --beta=-1 --eps=3.0
python main.py --algorithm DPSUR --dataset_name IMDB  --sigma 1.23 --lr 0.02 --batch_size 1024 --C_v=0.001 --sigma_v=0.9 --bs_valid=256 --beta=-1 --eps=4.0


#python main.py --algorithm DPSUR2 --dataset_name IMDB  --sigma 2.0 --lr 0.02 --batch_size 1024 --sigma_for_valid 1.2 --C_v=0.001 --bs_valid=128 --beta=-1 --eps=1.0
#python main.py --algorithm DPSUR2 --dataset_name IMDB  --sigma 1.8 --lr 0.02 --batch_size 1024 --sigma_for_valid 1.1 --C_v=0.001 --bs_valid=128 --beta=-1 --eps=2.0
#python main.py --algorithm DPSUR2 --dataset_name IMDB  --sigma 1.35 --lr 0.02 --batch_size 1024 --sigma_for_valid 1.0 --C_v=0.001 --bs_valid=128 --beta=-1 --eps=3.0
#python main.py --algorithm DPSUR2 --dataset_name IMDB  --sigma 1.23 --lr 0.02 --batch_size 1024 --sigma_for_valid 0.9 --C_v=0.001 --bs_valid=128 --beta=-1 --eps=4.0



python main.py --algorithm DPSUR3 --dataset_name CIFAR-10  --sigma 9.0  --lr 4.0 --batch_size 8192 --sigma_for_valid 1.0 --C_v=0.001 --bs_valid=128 --beta=-1 --input_norm=BN --num_groups=8 --bn_noise_multiplier=8 --use_scattering --eps=2.0
python main.py --algorithm DPSGD-TS --dataset_name CIFAR-10  --sigma 9.0 --lr 4.0 --batch_size 8192 --sigma_for_valid 1.0 --C_v=0.001 --bs_valid=128 --beta=-1 --input_norm=BN --num_groups=8 --bn_noise_multiplier=8 --use_scattering --eps=2.0

#python main.py --algorithm DPSGD-HF --dataset_name CIFAR-10  --sigma 5.67 --lr 4.0 --batch_size 8192 --sigma_for_valid 1.0 --C_v=0.001 --bs_valid=128 --beta=-1 --input_norm=BN --num_groups=8 --bn_noise_multiplier=8 --use_scattering --eps=1.0

#DPSUR3是服务大模型的，目前还是DPSUR2才是最终的算法代码
#python main.py --algorithm DPSUR3 --dataset_name CIFAR-10-Transformers  --sigma 1.25 --lr 2.0 --batch_size 1024 --sigma_for_valid 1.2 --C_v=0.001 --bs_valid=128 --beta=-1 --eps=3.0

#DPSGD-TS目前是暂时有DP的，但是没有选择更新，普通DP
#python main.py --algorithm DPSGD-TS --dataset_name CIFAR-10-Transformers  --sigma 1.25 --lr 2.0 --batch_size 1024 --sigma_for_valid 1.2 --C_v=0.001 --bs_valid=128 --beta=-1 --eps=3.0

#python main.py --algorithm DPSGD-TS --dataset_name CIFAR-10  --sigma 11.0 --lr 4.0 --batch_size 8192 --sigma_for_valid 1.3 --C_v=0.001 --bs_valid=128 --beta=-1 --eps=1.0