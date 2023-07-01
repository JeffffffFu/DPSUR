from membership_inference.meminf import *
from membership_inference.define_models import WhiteBoxAttackModel, ShadowAttackModel


#ML-DOCTOR: Holistic Risk Assessment of Inference Attacks Against Machine Learning Models
def test_meminf(device, num_classes, target_train, target_test, shadow_train, shadow_test,
                target_model, shadow_model,target_model_path,shadow_model_path,attack_path):
    # shadow_path = "./data/shadow_model/shadow_" + dataset_name + "_" + model_name + ".pth"
    # target_path = "./data/target_model/model.pth"
    # attack_path = "./data/inference_attacks/model"

    print(target_model_path)
    batch_size = 128

    #train shadow model
    shadow_trainloader = torch.utils.data.DataLoader(shadow_train, batch_size=batch_size, shuffle=True,
                                                     num_workers=2)
    shadow_testloader = torch.utils.data.DataLoader(shadow_test, batch_size=batch_size, shuffle=True, num_workers=2)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(shadow_model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
    acc_train, acc_test, overfitting = train_shadow_model(shadow_model_path, device, shadow_model, shadow_trainloader,
                                                          shadow_testloader,optimizer,loss)

    print("shadow_model Train Acc: " + str(acc_train) + " Test Acc: " + str(acc_test) + " overfitting rate: " + str(overfitting))

    # buliding attack dataset------- for both mode3 and mode0
    attack_trainloader, attack_testloader = get_attack_dataset_with_shadow(
        target_train, target_test, shadow_train, shadow_test, batch_size)
    # attack_trainloader, attack_testloader = get_attack_dataset_without_shadow(target_train, target_test, batch_size)

    # --------------------------------------- Model 3 -- WhiteBox Shadow ---------------------------------------
    # for white box
    gradient_size = get_gradient_size(target_model)
    total = gradient_size[0][0] // 2 * gradient_size[0][1] // 2

    # choosing attack model
    attack_model3 = WhiteBoxAttackModel(num_classes, total)  # Model 2 and 3 whitebox

    print("=========================== attack_mode3_result: ===========================")
    meminf_res_train3, meminf_res_test3 = attack_mode3(target_model_path, shadow_model_path, attack_path, device,
                                                       attack_trainloader, attack_testloader, target_model,
                                                       shadow_model, attack_model3, 1, num_classes)

    # --------------------------------------- Model 0 -- BlackBox  Shadow ---------------------------------------
    # choosing attack model
    attack_model0 = ShadowAttackModel(num_classes)  # Model 0 BlackBox Shadow

    print("====================== attack_mode0_result: ======================")
    meminf_res_train0, meminf_res_test0 = attack_mode0(target_model_path, shadow_model_path, attack_path, device,
                                                       attack_trainloader, attack_testloader, target_model,
                                                       shadow_model, attack_model0, 1, num_classes)

    #F1 Auc Acc
    print("White-Box/Shadow membership inference result: ========")
    print(" ***[F1, AUC, Acc]***")
    print("train: ", meminf_res_train3)
    print("test: ", meminf_res_test3)

    print( "Black-Box/Shadow membership inference result: ========")
    print(" ***[F1, AUC, Acc]***")
    print("train: ", meminf_res_train0)
    print("test: ", meminf_res_test0)

    return meminf_res_train3, meminf_res_test3, meminf_res_train0, meminf_res_test0
    # WhiteBox Shadow, 	BlackBox Shadow