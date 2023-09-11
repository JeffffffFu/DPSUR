import itertools

import torch
from matplotlib import pyplot as plt
from scipy.integrate import quad

from privacy_analysis.RDP.compute_dp_sgd import apply_dp_sgd_analysis
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
orders = (list(range(2, 64)) + [128, 256, 512])  # 默认的lamda
import math
from scipy.special import erf
import matplotlib.ticker as ticker

def plot_eps():
    eps1_list = []
    eps2_list = []
    eps3_list = []

    iter_list = []
    for i in range(1000):
        eps1, opt_order = apply_dp_sgd_analysis(512 / 60000, 1.23, i, orders, 10 ** (-5))
        eps2, opt_order = apply_dp_sgd_analysis(3000 / 60000, 2.23, i, orders, 10 ** (-5))
        eps3, opt_order = apply_dp_sgd_analysis(3000 / 60000, 3.23, i, orders, 10 ** (-5))
        eps1_list.append(eps1)
        eps2_list.append(eps2)
        eps3_list.append(eps3)
        iter_list.append(i)

    plt.plot(iter_list, eps1_list)
    plt.plot(iter_list, eps2_list)
    plt.plot(iter_list, eps3_list)


    plt.title("eps of diffenent parameter with delta of le-5")
    plt.xlabel('iters')
    plt.ylabel('eps')
    #plt.xlim(3669,5125)
    # plt.ylim(90,98.5)
    #plt.ylim(83,86.5)


    #plt.title('gradient l2_norm in Mnist dataset')
    plt.legend(['eps of DPSGD with sigma=1.23','eps of compute loss with simga=2.23','eps of compute loss with simga=3.23'], loc='best')

    plt.show()

#绘制样本被采样的频数图
def compute_sample_frequencies():

    index=torch.load("C:\python flie\SA-DPSGD-master/result\csv\SA_DPSGD_loss_add_dp_scatter/fmnist/3.0/0.7\-0.001/2.15_5.01_2048_5000_256_0.59_index.pt")
    print(len(index))
    flat_list = list(itertools.chain.from_iterable(index))
    # 将PyTorch张量转换为int
    my_list = [int(x) for x in flat_list]

    counts = Counter(my_list)
    # for i in range(55000):
    #     print(counts[i])

    #获取元素和出现次数的列表
    elements = counts.keys()
    frequencies = counts.values()
    # #绘制条形图
    plt.bar(elements, frequencies)
    # # 添加标题和标签
    plt.title("Sampling Frequencies of FMNIST")
    plt.xlabel("Record")
    plt.ylabel("Frequencies")

    # 显示图像
    plt.show()

#绘制对裁剪后的损失值之差进行高斯加噪的图，不同P下的概率
def plot_Gaussian_probability():
    # 均值和标准差
    mu1, mu2 = -0.1, 0.1
    noise_scale=1.1
    sigma = 2*mu2*noise_scale

    # 生成x轴坐标点
    x = np.linspace(-1, 1, 500)

    # 计算概率密度函数
    y1 = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((x - mu1) ** 2) / (2 * sigma ** 2))
    y2 = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((x - mu2) ** 2) / (2 * sigma ** 2))

    p=mu1*2
    # 绘制图形
    plt.plot(x, y1, label="$\mu=-C$",color='orange')
    plt.plot(x, y2, label="$\mu=C$",color='blue')
    plt.axvline(x=p, color='r', linestyle='-')

    # 填充区域
    mask = x <= p
    plt.fill_between(x[mask], 0, y1[mask], color='orange', alpha=0.7)
    plt.fill_between(x[mask], 0, y2[mask], color='blue', alpha=0.7)

    # 计算填充区域的面积
    area1 = quad(lambda a: (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((a - mu1) ** 2) / (2 * sigma ** 2)), -np.inf, p)[0]
    area2 = quad(lambda a: (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((a - mu2) ** 2) / (2 * sigma ** 2)), -np.inf, p)[0]

    print("mu1的面积：",area1)
    print("mu2的面积：",area2)


    # 显示填充区域的面积
    # plt.text(-1, 0.2, f"Area below x=-0.1 for Gaussian with mu=1: {area1:.3f}", fontsize=6)
    # plt.text(-1, 0.15, f"Area below x=-0.1 for Gaussian with mu=-1: {area2:.3f}", fontsize=6)

    plt.xlabel("Random Variables")
    plt.ylabel("Probability")

    plt.legend()
    plt.grid(True)
    plt.show()

#生成散点图，然后进行裁剪
def discrete_clipping():
    # 生成随机数
    np.random.seed(42)
    x = []
    y = []
    for i in range(50):
        x_i = np.random.uniform(-1, 1)
        y_i = np.random.uniform(-1, 1)
        if abs(x_i) > 0.1:
            x.append(x_i)
            y.append(y_i)

    # 绘制散点图
    plt.scatter(x, y)

    plt.axvline(x=-0.1, color='red')
    plt.axvline(x=0.1, color='red')

    # 添加标签和标题
    plt.xlabel('$\Delta E$')
    # 去掉纵坐标刻度
    plt.yticks([])
    #plt.title('C=[-0.1,0.1]')

    # 显示图形
    plt.show()

#绘制CDF对于均值为C或-C的时候
def plot_erf():
    C = 1
    sigma = 1
    a_list=[2,1.5,1,0.5,0,-1,-1.5,-2,-2.5,-3]
    a_list = [i * 0.1 - 2 for i in range(41)]
    list1=[]
    list2=[]
    for a in a_list:
        p1 = 1 / 2
        p2 = (a - 1) / (2 * math.sqrt(2 * math.pi) * sigma) * (math.exp(-(a - 1) ** 2 / (2 * sigma ** 2)) + 1) / math.sqrt(
            2 * math.pi)
        p3 = (a + 1) / (2 * math.sqrt(2 * math.pi) * sigma) * (math.exp(-(a + 1) ** 2 / (2 * sigma ** 2)) + 1) / math.sqrt(
            2 * math.pi)
        p_total1 = p1 + p2
        p_total2 = p1 + p3
        list1.append(p_total1)
        list2.append(p_total2)
    # 绘制图形
    plt.plot(a_list, list1, label="$\Delta E>0$")
    plt.plot(a_list, list2, label="$\Delta E<0$")
    plt.legend()
    plt.ylabel('probability')
    plt.xlabel('\u03B1')
    plt.show()

def lr_impact():
    lr_list=[3.0,4.0,5.0,6.0,7.0]
    acc_list=[69.15,71.40,71.34,71.28,71.32]

    plt.plot(lr_list, acc_list,marker="o")

    #plt.xlim([1, 3])
    plt.ylim(65, 75)  #设置横纵坐标范围


    plt.xlabel('learning late'+ r' $\eta$',fontsize=16)
    plt.ylabel('test accuarry (%)',fontsize=16)
    #plt.title('gradient l2_norm in Mnist dataset')
    plt.legend(['SU-DPSGD'], prop={'size':18}, loc='best')

    plt.show()

def k_impact():
    lr_list=[0.5,0.6,0.7,0.8,0.9]
    acc_list=[69.82,69.98,70.71,69.18,68.42]

    plt.plot(lr_list, acc_list,marker="o")

    #plt.xlim([1, 3])
    plt.ylim(62, 75)  #设置横纵坐标范围


    plt.xlabel('start valid'+ r' $k$',fontsize=16)
    plt.ylabel('test accuarry',fontsize=16)
    #plt.title('gradient l2_norm in Mnist dataset')
    plt.legend(['SU-DPSGD'], prop={'size':18}, loc='best')

    plt.show()

def alpha_impact():
    lr_list=[-1.5,-1,-0.5,0,0.5,1]
    acc_list=[71.01,71.40,70.98,70.88,70.23,69.72]

    plt.plot(lr_list, acc_list,marker="o")

    #plt.xlim([1, 3])
    plt.ylim(65, 75)  #设置横纵坐标范围


    plt.xlabel('threshold parameter'+ r' $\alpha$',fontsize=16)
    plt.ylabel('test accuarry (%)',fontsize=16)
    #plt.title('gradient l2_norm in Mnist dataset')
    plt.legend(['SU-DPSGD'], prop={'size':18}, loc='best')

    plt.show()

def number_of_validation_impact():
    lr_list=[2000,3000,4000,5000,6000,7000]
    acc_list=[69.81,70.79,70.85,71.40,70.91,70.52]

    plt.plot(lr_list, acc_list,marker="o")

    #plt.xlim([1, 3])
    plt.ylim(65, 75)  #设置横纵坐标范围


    plt.xlabel('number of validation set',fontsize=16)
    plt.ylabel('test accuarry (%)',fontsize=16)
    #plt.title('gradient l2_norm in Mnist dataset')
    plt.legend(['SU-DPSGD'], prop={'size':18}, loc='best')

    plt.show()

def C_impact():
    lr_list = [10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    acc_list = [70.16, 70.51, 70.88, 71.23, 71.40, 71.39, 71.40]

    # 设置横坐标标签和位置
    xticks = np.arange(len(lr_list))
    plt.xticks(xticks, lr_list)

    # 创建折线图
    plt.plot(xticks, acc_list, marker="o")
    plt.ylim(65, 75)

    # 设置坐标轴标签和标题
    plt.xlabel('clipping bound $C_v$',fontsize=16)
    plt.ylabel('test accuarry (%)',fontsize=16)
    plt.legend(['SU-DPSGD'], prop={'size':18}, loc='best')

    # 显示图像
    plt.show()

def validation_batchsize_impact():
    lr_list = [32, 64, 128, 256, 512, 1024, 2048]
    acc_list = [70.5, 70.78, 70.37, 70.9, 70.96, 70.75, 70.41]

    # 设置横坐标标签和位置
    xticks = np.arange(len(lr_list))
    plt.xticks(xticks, lr_list)

    # 创建折线图
    plt.plot(xticks, acc_list, marker="o")
    plt.ylim(65, 75)

    # 设置坐标轴标签和标题
    plt.xlabel('batch size of validation set',fontsize=16)
    plt.ylabel('test accuarry',fontsize=16)
    plt.legend(['SU-DPSGD'], prop={'size':18}, loc='best')

    # 显示图像
    plt.show()

if __name__=="__main__":
    #plot_acc()
    #plot_erf()
    number_of_validation_impact()
    # C_impact()
    # alpha_impact()
    # lr_impact()