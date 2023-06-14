##MNIST
#python main.py --algorithm DPSUR --dataset_name MNIST  --sigma 2.0 --lr 2.0 --batch_size 1024 --size_valid 5000 --C_v=0.001 --bs_valid=256 --beta=-1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --eps=1.0
#python main.py --algorithm DPSUR --dataset_name MNIST  --sigma 1.5 --lr 2.0 --batch_size 1024 --size_valid 5000 --C_v=0.001 --bs_valid=256 --beta=-1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --eps=2.0
#python main.py --algorithm DPSUR --dataset_name MNIST  --sigma 1.35 --lr 2.0 --batch_size 1024 --size_valid 5000 --C_v=0.001 --bs_valid=256 --beta=-1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --eps=3.0
#python main.py --algorithm DPSUR --dataset_name MNIST  --sigma 1.35 --lr 2.0 --batch_size 1024 --size_valid 5000 --C_v=0.001 --bs_valid=256 --beta=-1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --eps=4.0
#
##FMNIST
#python main.py --algorithm DPSUR --dataset_name FMNIST  --sigma 4.0 --lr 4.0 --batch_size 2048 --size_valid 5000 --C_v=0.001 --bs_valid=256 --beta=-1 --input_norm=GroupNorm --num_groups=27 --use_scattering --eps=1.0
#python main.py --algorithm DPSUR --dataset_name FMNIST  --sigma 2.15 --lr 4.0 --batch_size 2048 --size_valid 5000 --C_v=0.001 --bs_valid=256 --beta=-1 --input_norm=GroupNorm --num_groups=27 --use_scattering --eps=2.0
#python main.py --algorithm DPSUR --dataset_name FMNIST  --sigma 2.15 --lr 4.0 --batch_size 2048 --size_valid 5000 --C_v=0.001 --bs_valid=256 --beta=-1 --input_norm=GroupNorm --num_groups=27 --use_scattering --eps=3.0
#python main.py --algorithm DPSUR --dataset_name FMNIST  --sigma 2.15 --lr 4.0 --batch_size 2048 --size_valid 5000 --C_v=0.001 --bs_valid=256 --beta=-1 --input_norm=GroupNorm --num_groups=27 --use_scattering --eps=4.0

#CIFAR-10
#python main.py --algorithm DPSUR --dataset_name CIFAR-10  --sigma 10.0 --lr 4.0 --batch_size 8192 --size_valid 5000 --C_v=0.001 --bs_valid=256 --beta=-1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --eps=1.0
#python main.py --algorithm DPSUR --dataset_name CIFAR-10 --sigma 9.0 --lr 4.0 --batch_size 8192 --size_valid 5000 --C_v=0.001 --bs_valid=256 --beta=-1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --eps=2.0
#python main.py --algorithm DPSUR --dataset_name CIFAR-10 --sigma 5.67 --lr 4.0 --batch_size 8192 --size_valid 5000 --C_v=0.001 --bs_valid=256 --beta=-1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --eps=3.0
#python main.py --algorithm DPSUR --dataset_name CIFAR-10 --sigma 5.67 --lr 4.0 --batch_size 8192 --size_valid 5000 --C_v=0.001 --bs_valid=256 --beta=-1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --eps=4.0

#IMDB
#python main.py --algorithm DPSUR --dataset_name IMDB  --sigma 2.0 --lr 0.02 --batch_size 1024 --size_valid 3000 --C_v=0.001 --bs_valid=256 --beta=-1 --eps=1.0
#python main.py --algorithm DPSUR --dataset_name IMDB  --sigma 1.8 --lr 0.02 --batch_size 1024 --size_valid 3000 --C_v=0.001 --bs_valid=256 --beta=-1 --eps=2.0
#python main.py --algorithm DPSUR --dataset_name IMDB  --sigma 1.35 --lr 0.02 --batch_size 1024 --size_valid 3000 --C_v=0.001 --bs_valid=256 --beta=-1 --eps=3.0
#python main.py --algorithm DPSUR --dataset_name IMDB  --sigma 1.23 --lr 0.02 --batch_size 1024 --size_valid 3000 --C_v=0.001 --bs_valid=256 --beta=-1 --eps=4.0


python main.py --algorithm DPSUR2 --dataset_name IMDB  --sigma 2.0 --lr 0.02 --batch_size 1024 --sigma_for_valid 1.2 --C_v=0.001 --bs_valid=128 --beta=-1 --eps=1.0
python main.py --algorithm DPSUR2 --dataset_name IMDB  --sigma 1.8 --lr 0.02 --batch_size 1024 --sigma_for_valid 1.1 --C_v=0.001 --bs_valid=128 --beta=-1 --eps=2.0
python main.py --algorithm DPSUR2 --dataset_name IMDB  --sigma 1.35 --lr 0.02 --batch_size 1024 --sigma_for_valid 1.0 --C_v=0.001 --bs_valid=128 --beta=-1 --eps=3.0
python main.py --algorithm DPSUR2 --dataset_name IMDB  --sigma 1.23 --lr 0.02 --batch_size 1024 --sigma_for_valid 0.9 --C_v=0.001 --bs_valid=128 --beta=-1 --eps=4.0