import os
import sys
import torch.utils.data
import scipy.io as sio
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from HSI_torch import CommonDataset

def normalize(data):
    """
    normalize the HSI data
    """
    data = data.astype(np.float)
    for i in range(len(data)):
        data[i, :, :] -= data[i, :, :].min()
        data[i, :, :] /= data[i, :, :].max()
    return data

def random_unison(a, b, rstate=None):
    assert len(a) == len(b)
    p = np.random.RandomState(seed=rstate).permutation(len(a))
    return a[p], b[p]


def split_data(pixels, labels,  train_list=None, percent=0.1, splitdset="custom", rand_state=77):
    if splitdset == "sklearn":
        return train_test_split(pixels, labels, test_size=(1 - percent), stratify=labels, random_state=rand_state)
    elif splitdset == "custom":
        label, b = np.unique(labels, return_counts=True)  # 去掉重复元素，并统计其出现的次数
        trnum_list = [int(np.ceil(i * percent)) for i in b]  # 取每类训练集个数 | 向上取整
        trnumber = sum(trnum_list)
        # print(trnumber)
        tenumber = labels.shape[0] - trnumber
        # print(tenumber)
        train_X = np.zeros((trnumber, pixels.shape[1], pixels.shape[2], pixels.shape[3]))
        train_Y = np.zeros(trnumber)
        test_X = np.zeros((tenumber, pixels.shape[1], pixels.shape[2], pixels.shape[3]))
        test_Y = np.zeros(tenumber)
        trcont = 0
        tecont = 0
        for cl in np.unique(labels):
            pixels_cl = pixels[labels == cl]
            labels_cl = labels[labels == cl]
            pixels_cl, labels_cl = random_unison(pixels_cl, labels_cl, rstate=rand_state)
            for cont, (a, b) in enumerate(zip(pixels_cl, labels_cl)):
                if cont < trnum_list[int(cl)]:
                    train_X[trcont, :, :, :] = a
                    train_Y[trcont] = b
                    trcont += 1
                else:
                    test_X[tecont, :, :, :] = a
                    test_Y[tecont] = b
                    tecont += 1
        train_X, train_Y = random_unison(train_X, train_Y, rstate=rand_state)
        return train_X, test_X, train_Y, test_Y
    elif splitdset == "self_spilt":
        label, b = np.unique(labels, return_counts=True)  # 去掉重复元素，并统计其出现的次数
        class_number = label.shape[0]  # 数据集种类数
        trnumber = sum(train_list)
        # print(trnumber)
        tenumber = labels.shape[0] - trnumber
        # print(tenumber)
        # pixels_number = b  # 每类像素个数[ , , , , , , ...]
        train_X = np.zeros((trnumber, pixels.shape[1], pixels.shape[2], pixels.shape[3]))
        train_Y = np.zeros(trnumber)
        test_X = np.zeros((tenumber, pixels.shape[1], pixels.shape[2], pixels.shape[3]))
        test_Y = np.zeros(tenumber)
        # m = 0
        # for i in range(0, class_number):
        #     temp = np.where(labels == i)  # 元组,位置索引
        #     temp1 = random.sample(range(b[i]), train_num[i])
        #     # print(temp1)
        #     # print(temp)
        #     # print(temp[0][30])
        #     for j in range(train_num[i]):
        #         if m < train_num[i]:
        #             train_X[m, :, :, :] = pixels[temp[0][temp1[j]], :, :, :]
        #             label_x[m] = labels[temp[0][temp1[j]]]
        #         m += 1
        trcont = 0
        tecont = 0

        for lb in label:
            pixels_lb = pixels[labels == lb]
            labels_lb = labels[labels == lb]
            pixels_lb, labels_lb = random_unison(pixels_lb, labels_lb, rstate=rand_state)
            for cont, (a, b) in enumerate(zip(pixels_lb, labels_lb)):
                if cont < train_list[int(lb)]:
                    train_X[trcont, :, :, :] = a
                    train_Y[trcont] = b
                    trcont += 1
                else:
                    test_X[tecont, :, :, :] = a
                    test_Y[tecont] = b
                    tecont += 1
        train_X, train_Y = random_unison(train_X, train_Y, rstate=rand_state)
        return train_X, test_X, train_Y, test_Y


def loaddata(names, datapath):
    """
    数据集加载
    :param names: 数据集名称 | IN PU SA KSC
    :param datapath: 数据集存放目录
    :return:data，labels
    """
    data_path = os.path.join(datapath)  # 数据集地址
    if names == 'IN':
        data = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
    elif names == 'PU':
        data = sio.loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path, 'PaviaU_gt.mat'))['paviaU_gt']
    elif names == 'SA':
        data = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
    elif names == 'KSC':
        data = sio.loadmat(os.path.join(data_path, 'KSC.mat'))['KSC']
        labels = sio.loadmat(os.path.join(data_path, 'KSC_gt.mat'))['KSC_gt']
    else:
        print("NO DATASET")
        sys.exit()

    return data, labels

def pad_zero(data, window_size):
    margin = int((window_size - 1) / 2)
    return np.pad(data, ((margin, margin), (margin, margin), (0, 0)))

def CreatimageCube(tot_x, tot_y, Windows_Size, removeZeroLabels=True):  # tot train or test
    """
    :param tot_x:train\test\val
    :param tot_y:train\test\val label
    :param Windows_Size:patch_size
    :param removeZeroLabels: Remove 0 tag, Recommended. if not, it will not
           be usable because of low performance and memory overflow
    :return: patches_x[index, row, col, bands], patches_y[index], kinds, Bands
    """
    margin = int((Windows_Size - 1) / 2)
    Bands = tot_x.shape[2]
    kinds = np.unique(tot_y).shape[0] - 1  # 得到测试或者训练集中的种类数
    paddeddata = pad_zero(tot_x, Windows_Size)
    labelnum = np.sum(tot_y > 0)
    if removeZeroLabels:
        patches_x = np.zeros((labelnum, Windows_Size, Windows_Size, Bands))
        patches_y = np.zeros(labelnum)
        patches_index = 0
        for row in range(margin, paddeddata.shape[0] - margin):
            for col in range(margin, paddeddata.shape[1] - margin):
                if tot_y[row - margin, col - margin] != 0:
                    patch = paddeddata[row - margin:row + margin + 1, col - margin:col + margin + 1, :]
                    patches_x[patches_index, :, :, :] = patch
                    patches_y[patches_index] = tot_y[row - margin, col - margin] - 1
                    patches_index += 1
        del paddeddata
        return patches_x, patches_y, kinds, Bands
    else:
        patches_x = np.zeros((tot_x.shape[0] * tot_x.shape[1], Windows_Size, Windows_Size, Bands))
        patches_y = np.zeros(tot_x.shape[0] * tot_x.shape[1])

        patches_index = 0
        for row in range(margin, paddeddata.shape[0] - margin):
            for col in range(margin, paddeddata.shape[0] - margin):
                patch = paddeddata[row - margin:row + margin + 1, col - margin:col + margin + 1, :]
                patches_x[patches_index, :, :, :] = patch
                patches_y[patches_index] = tot_y[row - margin, col - margin] - 1
                patches_index += 1
        del paddeddata
        return patches_x, patches_y, kinds, Bands


def load_dataset(path, name, batch_size, window_size, test_batch):
    """
    加载数据集
    :param batch_size: batch_size
    :param path: 数据集根目录
    :param name: 数据名 IN,PU,KSC,SA
    :param test_batch: test_size
    :param window_size: path_size
    :return: train_loader, test_loader, val_loader, kinds, bands
    """
    x, y = loaddata(name, path)
    x = normalize(x)  # 归一化
    train_x, train_y, kinds, bands = CreatimageCube(x, y, window_size, removeZeroLabels=True)
    train_x, test_x, train_y, test_y = split_data(train_x, train_y, percent=0.2, splitdset="custom", rand_state=77)
    val_x, test_x, val_y, test_y = split_data(test_x, test_y, percent=0.125, splitdset="custom", rand_state=77)
    train_hyper = CommonDataset((np.transpose(train_x, (0, 3, 1, 2)).astype("float32"), train_y))
    test_hyper = CommonDataset((np.transpose(test_x, (0, 3, 1, 2)).astype("float32"), test_y))
    val_hyper = CommonDataset((np.transpose(val_x, (0, 3, 1, 2)).astype("float32"), val_y))
    train_loader = torch.utils.data.DataLoader(train_hyper, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_hyper, batch_size=test_batch, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_hyper, batch_size=test_batch, shuffle=False)
    del test_hyper, val_hyper, train_x, test_x, train_y, test_y, val_x, val_y
    return train_loader, test_loader, val_loader, kinds, bands, train_hyper.data.shape[1:]

# if __name__ == '__main__':
#     path = r'F:\Residual Spectral–Spatial Attention Network for\Dataset'
#     x, y = loaddata('IP', path)
#     # train_num = [14, 419, 265, 69, 149, 219, 9, 144, 5, 288, 733, 175, 65, 376, 119, 31]
#     # val_num = [4, 133, 99, 21, 52, 73, 3, 48, 1, 93, 242, 56, 24, 123, 41, 12]
#     x = normalize(x)  # 归一化
#     train_x, train_y, kind, Band = CreatimageCube(x, y, 8, 17, removeZeroLabels=True)
#
#     train_x, test_x, train_y, test_y = split_data(train_x, train_y, percent=0.2, splitdset="custom", rand_state=77)
#     val_x, test_x, val_y, test_y = split_data(test_x, test_y, percent=0.125, splitdset="custom", rand_state=77)



