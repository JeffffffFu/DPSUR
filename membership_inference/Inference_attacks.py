import argparse
import time
import random

from membership_inference.meminf import *
from membership_inference.prepare_MIA_dataset import prepare_MIA_dataset
from membership_inference.test_meminf import test_meminf


def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

seed_torch()


    

def str_to_bool(string):
    if isinstance(string, bool):
        return string
    if string.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif string.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def Member_inference_attacks(trained_model_path,target_model,dataset):
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda',choices=['cpu', 'cuda'])

    parser.add_argument('-g', '--gpu', type=str, default="0")
    parser.add_argument('-a', '--attributes', type=str, default="race",
                        help="For attrinf, two attributes should be in format x_y e.g. race_gender")
    parser.add_argument('-dn', '--dataset_name', type=str, default="cifar10") #utkface, stl10, fmnist, cifar10
    parser.add_argument('-mod', '--model_name', type=str, default='cnn')  # cnn, vgg19, preactresnet18
    parser.add_argument('-ts', '--train_shadow', action='store_true',default=True)

    args = parser.parse_args()


    device=args.device
    
    attr = args.attributes
    if "_" in attr:
        attr = attr.split("_")

    train_shadow = args.train_shadow
    dataset_name = args.dataset
    model_name = args.model_name
    root = "./data/" + dataset_name


    train_results=[]
    test_results=[]

    # for model_name in models_name:
    #     for dataset_name in datasets_name:
    print("<************ inference_attacks ************ model: " + model_name + " ************ dataset: "+dataset_name)

    # prepare dataset
    num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model = prepare_MIA_dataset(
                dataset_name, attr, root, model_name)
    print("num_classes: ", num_classes)


    # # # -------------- membership inference --------------
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> membership inference >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    meminf_res_train3, meminf_res_test3, meminf_res_train0, meminf_res_test0=test_meminf(device, num_classes, target_train, target_test, shadow_train, shadow_test, dataset_name,
                    target_model, shadow_model, model_name, train_shadow)
    # meminf -- WhiteBox Shadow
    # ***F1, AUC, Acc***
    train_results.append(meminf_res_train3)
    test_results.append(meminf_res_test3)
    # meminf -- BlackBox Shadow
    # ***F1, AUC, Acc***
    train_results.append(meminf_res_train0)
    test_results.append(meminf_res_test0)



    print("train results: ",train_results)
    print("test results: ", test_results)




if __name__ == "__main__":
    start_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
    Member_inference_attacks()
    end_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
    print("start time: ", start_time)
    print("end time: ", end_time)

