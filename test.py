import torch
def test():
    file_path_edge='F://PycharmFile/DPSUR/result/Without_MIA/DPSGD/MNIST/0.1/iterList.pth'
    list = torch.load(file_path_edge)
    print(list)

if __name__ == '__main__':
    test()