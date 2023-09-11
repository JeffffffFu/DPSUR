import torch
import matplotlib.pyplot as plt
import numpy as np

def convergence_speed():
    path="F://PycharmFile/DPSUR/result/Without_MIA/DPSUR/FMNIST/4.0/iterList.pth"
    DPSUR_list=torch.load(path)

    path="F://PycharmFile/DPSUR/result/Without_MIA/DPSGD/FMNIST/4.0/iterList.pth"
    DPSGD_list=torch.load(path)

    path="F://PycharmFile/DPSUR/result/Without_MIA/DPSGD-HF/FMNIST/4.0/iterList.pth"
    DPSGD_HF_list=torch.load(path)

    path="F://PycharmFile/DPSUR/result/Without_MIA/DPSGD-TS/FMNIST/4.0/iterList.pth"
    DPSGD_TS_list=torch.load(path)

    path="F://PycharmFile/DPSUR/result/Without_MIA/DPAGD/FMNIST/4.0/iterList.pth"
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

if __name__ == '__main__':
    convergence_speed()