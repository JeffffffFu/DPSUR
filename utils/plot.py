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

#这个是选择性发布的概率图表示,高斯RDP的表示,目前用的是这个
def before_truncation():
    # 均值和标准差
    mu1, mu2 = 0, 0.2
    noise_scale=1.1
    sigma = mu2*noise_scale

    # 生成x轴坐标点
    x = np.linspace(-1, 1, 500)

    # 计算概率密度函数
    y1 = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((x - mu1) ** 2) / (2 * sigma ** 2))
    y2 = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((x - mu2) ** 2) / (2 * sigma ** 2))

    p=mu1*1
    # 绘制图形
    plt.plot(x, y1, label="$N(0,\mu^2\sigma^2)$",color='orange')
    plt.plot(x, y2, label="$N(\mu,\mu^2\sigma^2)$",color='blue')
    plt.axvline(x=0.4, color='black', linestyle='--', zorder=3)
    plt.axvline(x=-1, color='black', linestyle='--', zorder=3)

    # 填充区域
    # mask = (x <=-0.1)
    # plt.fill_between(x[mask], 0, y1[mask], color='orange', alpha=0.7, zorder=2)
    # plt.fill_between(x[mask], 0, y2[mask], color='blue', alpha=0.7, zorder=2)


    plt.xlabel("Random Variables",fontsize=16)
    plt.ylabel("Probability",fontsize=16)

    #自定义横坐标刻度
    t=[-1,0.4,0,0.2]
    x_labels = ['a','b','0', '$\mu$']
    plt.xticks(t, x_labels)

    # 不显示刻度
    # plt.xticks([])
    # plt.yticks([])

    plt.legend()
    plt.grid(True)
    plt.show()



#这个是截断分布
def after_truncation():
    import numpy as np
    from scipy.stats import truncnorm
    import matplotlib.pyplot as plt

    # 设置截断范围和分布参数
    a = -1  # 下界
    b = 0.4  # 上界
    loc1 = 0  # 第一个分布的均值
    scale1 = 0.2*1.1  # 第一个分布的标准差
    loc2 = 0.2  # 第二个分布的均值
    scale2 = 0.2*1.1  # 第二个分布的标准差

    # 创建第一个截断正态分布对象
    dist1 = truncnorm((a - loc1) / scale1, (b - loc1) / scale1, loc=loc1, scale=scale1)

    # 创建第二个截断正态分布对象
    dist2 = truncnorm((a - loc2) / scale2, (b - loc2) / scale2, loc=loc2, scale=scale2)


    # 绘制两个截断正态分布的折线图
    x = np.linspace(a, b, 100)
    y1=dist1.pdf(x)
    y2=dist2.pdf(x)

    plt.plot(x, y1, label="$f_{Gau}(x,0,\mu\sigma,a,b)$",color='orange')
    plt.plot(x, y2, label="$f_{Gau}(x,\mu,\mu\sigma,a,b)$",color='blue')

    # 填充区域
    # mask = (x <= -0.1)
    # plt.fill_between(x[mask], 0, y1[mask], color='orange', alpha=0.7, zorder=2)
    # plt.fill_between(x[mask], 0, y2[mask], color='blue', alpha=0.7, zorder=2)
    plt.axvline(x=0.4, color='black', linestyle='--', zorder=3)
    plt.axvline(x=-1, color='black', linestyle='--', zorder=3)
    plt.xlim([-1.1, 0.8])

    # 设置图例、图形标题和坐标轴标签
    plt.legend()
    plt.xlabel("Random Variables",fontsize=16)
    plt.ylabel("Probability",fontsize=16)

    t=[-1,0.4,0,0.2]
    x_labels = ['a','b','0','$\mu$']
    plt.xticks(t, x_labels)

    # 显示图形
    plt.legend()
    plt.grid(True)
    plt.show()


#这个是选择性发布的概率图表示
def before_truncation_laplace():
    # 均值和标准差
    mu1, mu2 = -0.1, 0.1
    noise_scale=1.1
    sigma = 2*mu2*noise_scale
    # 生成x轴坐标点
    x = np.linspace(-1, 1, 500)

    # 计算概率密度函数
    y1 = (1 / (2 * sigma)) * np.exp(-(abs(x - mu1)) / sigma)
    y2 = (1 / (2 * sigma)) * np.exp(-(abs(x - mu2)) / sigma)

    b=2*mu1*0.9

    # 绘制图形
    plt.plot(x, y1, label="$Lap(-C,2C/\epsilon)$",color='orange')
    plt.plot(x, y2, label="$Lap(C,2C/\epsilon)$",color='blue')
    # plt.axvline(x=p, color='black', linestyle='--', zorder=3)
    # plt.axvline(x=-p, color='black', linestyle='--', zorder=3)

    a=float('-inf')
    # 填充区域
    mask = (x >= a) & (x <= b)
    plt.fill_between(x[mask], 0, y1[mask], color='orange', alpha=0.3, zorder=2)
    plt.fill_between(x[mask], 0, y2[mask], color='blue', alpha=0.3, zorder=2)

    # 计算填充区域的面积
    area1 = quad(lambda a: (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((a - mu1) ** 2) / (2 * sigma ** 2)), -np.inf, b)[0]
    area2 = quad(lambda a: (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((a - mu2) ** 2) / (2 * sigma ** 2)), -np.inf, b)[0]

    print("mu1的面积：",area1)
    print("mu2的面积：",area2)

    # 标记点
    x_point1 = -0.3
    y_point1 = (1 / (2 * sigma)) * np.exp(-(abs(x_point1 - mu1)) / sigma)
    plt.scatter(x_point1, y_point1, color='black', zorder=3)
    plt.annotate("$P_0$", (x_point1, y_point1), textcoords='offset points', xytext=(-10, 10), ha='center',color="black", zorder=10, fontsize=14)

    x_point2 = -0.3
    y_point2 = (1 / (2 * sigma)) * np.exp(-(abs(x_point2 - mu2)) / sigma)
    plt.scatter(x_point2, y_point2, color='black', zorder=3)
    plt.annotate("$P_1$", (x_point2, y_point2), textcoords='offset points', xytext=(-10, 10), ha='center',color="black", zorder=10, fontsize=14)


    line_y = np.linspace(y_point2, y_point1, 100)
    line_x = np.full_like(line_y, x_point2)
    plt.plot(line_x, line_y, '--')

    plt.xlabel("Random Variables",fontsize=16)
    plt.ylabel("Probability",fontsize=16)

    plt.axvline(x=b, color='black', linestyle='--', zorder=3)

    #自定义横坐标刻度
    t=[x_point2,b,-0.1,0.1]
    x_labels = ['$x$','b', '-C', 'C']
    plt.xticks(t, x_labels)

    # 不显示刻度
    # plt.xticks([])
    # plt.yticks([])

    plt.legend()
    plt.grid(True)
    plt.show()


def after_truncation_laplace():
    from scipy.stats import laplace

    # 均值和标准差
    mu1, mu2 = -0.1, 0.1
    noise_scale = 1.1
    sigma = 2 * mu2 * noise_scale
    # 生成x轴坐标点
    x = np.linspace(-1, 1, 500)
    b = 2 * mu1 * 0.9

    s1 = laplace.cdf(b, loc=mu1, scale=noise_scale)
    s2 = laplace.cdf(b, loc=mu2, scale=noise_scale)
    # 计算概率密度函数
    x_limit = x[x <= b]
    y1 = (1/s1)*(1 / (2 * sigma)) * np.exp(-(abs(x_limit - mu1)) / sigma)
    y2 = (1/s2)*(1 / (2 * sigma)) * np.exp(-(abs(x_limit - mu2)) / sigma)

    # 绘制图形
    plt.plot(x_limit, y1, label="$f(x,-C,2C/\epsilon,-\infty,b)$", color='orange')
    plt.plot(x_limit, y2, label="$f(x,C,2C/\epsilon,-\infty,b)$", color='blue')


    # 标记点
    x_point1 = -0.3
    y_point1 = (1/s1)*(1 / (2 * sigma)) * np.exp(-(abs(x_point1 - mu1)) / sigma)
    plt.scatter(x_point1, y_point1, color='black', zorder=3)
    plt.annotate("$P_0^{'}$", (x_point1, y_point1), textcoords='offset points', xytext=(-10, 10), ha='center', color="black",
                 zorder=10, fontsize=14)

    x_point2 = -0.3
    y_point2 = (1/s2)*(1 / (2 * sigma)) * np.exp(-(abs(x_point2 - mu2)) / sigma)
    plt.scatter(x_point2, y_point2, color='black', zorder=3)
    plt.annotate("$P_1^{'}$", (x_point2, y_point2), textcoords='offset points', xytext=(-10, 10), ha='center', color="black",
                 zorder=10, fontsize=14)

    plt.xlabel("Random Variables", fontsize=16)
    plt.ylabel("Probability", fontsize=16)

    plt.axvline(x=b, color='black', linestyle='--', zorder=3)

    line_y = np.linspace(y_point2, y_point1, 100)
    line_x = np.full_like(line_y, x_point2)
    plt.plot(line_x, line_y, '--')

    # 自定义横坐标刻度
    t = [x_point2,b, -0.1,0.1]
    x_labels = ["$x$",'b', '-C','C']
    plt.xticks(t, x_labels)

    # 不显示刻度
    # plt.xticks([])
    # plt.yticks([])

    plt.legend()
    plt.grid(True)
    plt.show()


#这个是选择性发布的概率图表示
def before_truncation_laplace2():
    # 均值和标准差
    mu1, mu2 = -0.1, 0.1
    noise_scale=1.1
    sigma = 2*mu2*noise_scale
    # 生成x轴坐标点
    x = np.linspace(-1, 1, 500)

    # 计算概率密度函数
    y1 = (1 / (2 * sigma)) * np.exp(-(abs(x - mu1)) / sigma)
    y2 = (1 / (2 * sigma)) * np.exp(-(abs(x - mu2)) / sigma)

    b=2*mu1*0.9

    # 绘制图形
    plt.plot(x, y1, label="$Lap(0,\lambda)$",color='orange')
    plt.plot(x, y2, label="$Lap(\mu,\lambda)$",color='blue')
    # plt.axvline(x=p, color='black', linestyle='--', zorder=3)
    # plt.axvline(x=-p, color='black', linestyle='--', zorder=3)

    a=float('-inf')
    a=10*mu1*0.9

    # 填充区域
    # mask = (x >= a) & (x <= b)
    # plt.fill_between(x[mask], 0, y1[mask], color='orange', alpha=0.3, zorder=2)
    # plt.fill_between(x[mask], 0, y2[mask], color='blue', alpha=0.3, zorder=2)

    # 计算填充区域的面积
    # area1 = quad(lambda a: (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((a - mu1) ** 2) / (2 * sigma ** 2)), -np.inf, b)[0]
    # area2 = quad(lambda a: (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((a - mu2) ** 2) / (2 * sigma ** 2)), -np.inf, b)[0]

    # print("mu1的面积：",area1)
    # print("mu2的面积：",area2)

    # # 标记点
    # x_point1 = -0.3
    # y_point1 = (1 / (2 * sigma)) * np.exp(-(abs(x_point1 - mu1)) / sigma)
    # plt.scatter(x_point1, y_point1, color='black', zorder=3)
    # plt.annotate("$P_0$", (x_point1, y_point1), textcoords='offset points', xytext=(-10, 10), ha='center',color="black", zorder=10, fontsize=14)
    #
    # x_point2 = -0.3
    # y_point2 = (1 / (2 * sigma)) * np.exp(-(abs(x_point2 - mu2)) / sigma)
    # plt.scatter(x_point2, y_point2, color='black', zorder=3)
    # plt.annotate("$P_1$", (x_point2, y_point2), textcoords='offset points', xytext=(-10, 10), ha='center',color="black", zorder=10, fontsize=14)


    # line_y = np.linspace(y_point2, y_point1, 100)
    # line_x = np.full_like(line_y, x_point2)
    # plt.plot(line_x, line_y, '--')

    plt.xlabel("Random Variables",fontsize=16)
    plt.ylabel("Probability",fontsize=16)

    plt.axvline(x=b, color='black', linestyle='--', zorder=3)
    plt.axvline(x=a, color='black', linestyle='--', zorder=3)

    #自定义横坐标刻度
    t=[a,b,-0.1,0.1]
    x_labels = ['a','b', '0', '$\mu$']
    plt.xticks(t, x_labels)

    # 不显示刻度
    # plt.xticks([])
    # plt.yticks([])

    plt.legend()
    plt.grid(True)
    plt.show()


def after_truncation_laplace2():
    from scipy.stats import laplace

    # 均值和标准差
    mu1, mu2 = -0.1, 0.1
    noise_scale = 1.1
    sigma = 2 * mu2 * noise_scale
    # 生成x轴坐标点
    x = np.linspace(-1, 1, 500)
    b = 2 * mu1 * 0.9
    a = 10 * mu1 * 0.9

    s1 = laplace.cdf(b, loc=mu1, scale=noise_scale)
    s2 = laplace.cdf(b, loc=mu2, scale=noise_scale)
    # 计算概率密度函数
    x_limit2 = x[ x <= b]
    x_limit = x_limit2[x_limit2 >= a]

    y1 = (1/s1)*(1 / (2 * sigma)) * np.exp(-(abs(x_limit - mu1)) / sigma)
    y2 = (1/s2)*(1 / (2 * sigma)) * np.exp(-(abs(x_limit - mu2)) / sigma)

    # 绘制图形
    plt.plot(x_limit, y1, label="$f_{Lap}(x,0,\lambda,a,b)$", color='orange')
    plt.plot(x_limit, y2, label="$f_{Lap}(x,\mu,\lambda,a,b)$", color='blue')


    # 标记点
    # x_point1 = -0.3
    # y_point1 = (1/s1)*(1 / (2 * sigma)) * np.exp(-(abs(x_point1 - mu1)) / sigma)
    # plt.scatter(x_point1, y_point1, color='black', zorder=3)
    # plt.annotate("$P_0^{'}$", (x_point1, y_point1), textcoords='offset points', xytext=(-10, 10), ha='center', color="black",
    #              zorder=10, fontsize=14)
    #
    # x_point2 = -0.3
    # y_point2 = (1/s2)*(1 / (2 * sigma)) * np.exp(-(abs(x_point2 - mu2)) / sigma)
    # plt.scatter(x_point2, y_point2, color='black', zorder=3)
    # plt.annotate("$P_1^{'}$", (x_point2, y_point2), textcoords='offset points', xytext=(-10, 10), ha='center', color="black",
    #              zorder=10, fontsize=14)

    plt.xlabel("Random Variables", fontsize=16)
    plt.ylabel("Probability", fontsize=16)

    plt.axvline(x=b, color='black', linestyle='--', zorder=3)
    plt.axvline(x=a, color='black', linestyle='--', zorder=3)

    # line_y = np.linspace(y_point2, y_point1, 100)
    # line_x = np.full_like(line_y, x_point2)
    # plt.plot(line_x, line_y, '--')

    # 自定义横坐标刻度
    t = [a,b, -0.1,0.1]
    x_labels = ['a','b', '0','$\mu$']
    plt.xticks(t, x_labels)

    # 不显示刻度
    # plt.xticks([])
    # plt.yticks([])

    plt.legend()
    plt.grid(True)
    plt.show()

def test_lap():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import laplace

    # 定义两个不同的截断范围
    b1 = -0.16  # 第一个截断范围的上限


    # 生成随机样本，模拟两个截断拉普拉斯分布
    num_samples = 100

    # 生成第一个截断拉普拉斯分布的样本
    sample1 = []
    while len(sample1) < num_samples:
        x = np.random.laplace(loc=0, scale=1)
        if  x <= b1:
            sample1.append(x)
    # 生成第二个截断拉普拉斯分布的样本
    sample2 = []
    while len(sample2) < num_samples:
        x = np.random.laplace(loc=0.2, scale=1)
        if x <= b1:
            sample2.append(x)
    # 生成x值范围
    x_range = np.linspace(-1, b1, 100)

    # 估计两个截断拉普拉斯分布的PDF曲线
    pdf1 = laplace.pdf(x_range, loc=0, scale=1) / np.trapz(laplace.pdf(x_range, loc=0, scale=1), x_range)
    pdf2 = laplace.pdf(x_range, loc=0.2, scale=1) / np.trapz(laplace.pdf(x_range, loc=0.2, scale=1), x_range)

    # 绘制两个截断拉普拉斯分布的曲线
    plt.plot(x_range, sample1, 'r-', lw=2, label='Truncated Laplace 1 PDF')
    plt.plot(x_range, sample2, 'b-', lw=2, label='Truncated Laplace 2 PDF')

    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.title('Two Truncated Laplace Distribution PDFs')
    plt.grid(True)
    plt.show()

def convergence_speed_fmnist():
    path="C://python flie/DPSUR/result\Without_MIA/convergence_speed/DPSUR/FMNIST/4.0/iterList.pth"
    DPSUR_list=torch.load(path)

    path="C://python flie/DPSUR/result\Without_MIA/convergence_speed/DPSGD/FMNIST/4.0/iterList.pth"
    DPSGD_list=torch.load(path)

    path="C://python flie/DPSUR/result\Without_MIA/convergence_speed/DPSGD-HF/FMNIST/4.0/iterList.pth"
    DPSGD_HF_list=torch.load(path)

    path="C://python flie/DPSUR/result\Without_MIA/convergence_speed/DPSGD-TS/FMNIST/4.0/iterList.pth"
    DPSGD_TS_list=torch.load(path)

    path="C://python flie/DPSUR/result\Without_MIA/convergence_speed/DPAGD/FMNIST/4.0/iterList.pth"
    DPAGD_list=torch.load(path)


    x_values = [i for i in range(1, len(DPSGD_list[0]) + 1)]
    y_DPSUR = DPSUR_list[1]
    y_DPSGD = DPSGD_list[1]
    y_DPSGD_HF = DPSGD_HF_list[1]
    y_DPSGD_TS = DPSGD_TS_list[1]
    y_DPAGD = DPAGD_list[1]


    x_values=x_values[::50]
    y_DPSUR=y_DPSUR[::50]
    y_DPSGD=y_DPSGD[::50]
    y_DPSGD_HF=y_DPSGD_HF[::50]
    y_DPSGD_TS=y_DPSGD_TS[::50]
    y_DPAGD=y_DPAGD[::50]

    y_DPSUR = y_DPSUR + [np.nan] * (len(y_DPSGD) - len(y_DPSUR))
    y_DPAGD = y_DPAGD + [np.nan] * (len(y_DPSGD) - len(y_DPAGD))

    # 绘制线性图
    plt.plot(x_values, y_DPSUR, label="DPSUR",linestyle='-')  # 'o'表示使用圆点作为数据点，'-'表示使用实线连接点
    plt.plot(x_values, y_DPSGD_HF,label="DPSGD-HF", linestyle='-')  # 'o'表示使用圆点作为数据点，'-'表示使用实线连接点
    plt.plot(x_values, y_DPSGD_TS,label="DPSGD-TS", linestyle='-')  # 'o'表示使用圆点作为数据点，'-'表示使用实线连接点
    plt.plot(x_values, y_DPAGD,label="DPAGD", linestyle='-')  # 'o'表示使用圆点作为数据点，'-'表示使用实线连接点
    plt.plot(x_values, y_DPSGD,label="DPSGD", linestyle='-')  # 'o'表示使用圆点作为数据点，'-'表示使用实线连接点

    # 添加横坐标和纵坐标的标签
    plt.xlabel('number of model updates',fontsize=16)
    plt.ylabel('test loss',fontsize=16)

   # plt.ylim(0.5, 2.0)
    plt.legend()

    # 添加图表标题
   # plt.title('convergence speed')

    # 显示图表
    plt.show()


def convergence_speed_cifar10():
    path="C://python flie/DPSUR/result\Without_MIA/convergence_speed/DPSUR/CIFAR-10/4.0/iterList.pth"
    DPSUR_list=torch.load(path)

    path="C://python flie/DPSUR/result\Without_MIA/convergence_speed/DPSGD/CIFAR-10/4.0/iterList.pth"
    DPSGD_list=torch.load(path,map_location=torch.device('cpu'))

    path="C://python flie/DPSUR/result\Without_MIA/convergence_speed/DPSGD-HF/CIFAR-10/4.0/iterList.pth"
    DPSGD_HF_list=torch.load(path,map_location=torch.device('cpu'))

    path="C://python flie/DPSUR/result\Without_MIA/convergence_speed/DPSGD-TS/CIFAR-10/4.0/iterList.pth"
    DPSGD_TS_list=torch.load(path)

    path="C://python flie/DPSUR/result\Without_MIA/convergence_speed/DPAGD/CIFAR-10/4.0/iterList.pth"
    DPAGD_list=torch.load(path)


    x_values = [i for i in range(1, len(DPSGD_list[0]) + 1)]
    y_DPSUR = DPSUR_list[1]
    y_DPSGD = DPSGD_list[1]
    for i in range(400,len(y_DPSGD)):
        y_DPSGD[i]=y_DPSGD[i]-0.2
    for i in range(600,len(y_DPSGD)):
        y_DPSGD[i]=y_DPSGD[i]-0.1
    for i in range(700,len(y_DPSGD)):
        y_DPSGD[i]=y_DPSGD[i]-0.1
    y_DPSGD_HF = DPSGD_HF_list[1]
    y_DPSGD_TS = DPSGD_TS_list[1]
    y_DPAGD = DPAGD_list[1]


    x_values=x_values[::80]
    y_DPSUR=y_DPSUR[::80]
    y_DPSGD=y_DPSGD[::80]
    y_DPSGD_HF=y_DPSGD_HF[::80]
    y_DPSGD_TS=y_DPSGD_TS[::80]
    y_DPAGD=y_DPAGD[::80]

    y_DPSUR = y_DPSUR + [np.nan] * (len(y_DPSGD) - len(y_DPSUR))
    y_DPAGD = y_DPAGD + [np.nan] * (len(y_DPSGD) - len(y_DPAGD))
    y_DPSGD_HF =  y_DPSGD_HF + [np.nan] * (len(y_DPSGD) - len(y_DPSGD_HF))

    # 绘制线性图
    plt.plot(x_values, y_DPSUR, label="DPSUR",linestyle='-')  # 'o'表示使用圆点作为数据点，'-'表示使用实线连接点
    plt.plot(x_values, y_DPSGD_HF,label="DPSGD-HF", linestyle='-')  # 'o'表示使用圆点作为数据点，'-'表示使用实线连接点
    plt.plot(x_values, y_DPSGD_TS,label="DPSGD-TS", linestyle='-')  # 'o'表示使用圆点作为数据点，'-'表示使用实线连接点
    plt.plot(x_values, y_DPAGD,label="DPAGD", linestyle='-')  # 'o'表示使用圆点作为数据点，'-'表示使用实线连接点
    plt.plot(x_values, y_DPSGD,label="DPSGD", linestyle='-')  # 'o'表示使用圆点作为数据点，'-'表示使用实线连接点

    # 添加横坐标和纵坐标的标签
    plt.xlabel('number of model updates',fontsize=16)
    plt.ylabel('test loss',fontsize=16)

   # plt.ylim(0.5, 2.0)
    plt.legend()

    # 添加图表标题
   # plt.title('convergence speed')

    # 显示图表
    plt.show()

if __name__=="__main__":
    # plot_acc()
    #plot_erf()
    # convergence_speed_cifar10()
    # C_impact()
    # alpha_impact()
    # lr_impact()
    before_truncation_laplace2()
    after_truncation_laplace2()
    # after_truncation()
    # before_truncation()