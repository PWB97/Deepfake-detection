import torch
import torchvision
from torch.nn import functional as F
import pandas
import argparse

from PIL import Image
from dataloader import Dataset
import config
import os

import numpy as np
import cv2


def list_file(path, label):
    list = []
    for file in os.listdir(path):
        list.append([path + '/' + file, label])

    return list


def dataset_size(path):
    for file in os.listdir(path + '/0/'):
        img = cv2.imread(path + '/0/' + file)
        print(np.array(img).shape)


def frame_range(src_dir):
    Celeb_real = list_file(src_dir + '/Celeb-real', 1)
    Celeb_synthesis = list_file(src_dir + '/Celeb-synthesis', 0)
    YouTube_real = list_file(src_dir + '/YouTube-real', 1)

    frame_m = []

    for [file, _] in Celeb_real:
        video = cv2.VideoCapture(os.path.join(src_dir, '/Celeb-real', file))
        frame_num = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_m.append(frame_num)
        print(file)
        print(frame_num)
        print('---------------')
    for [file, _] in Celeb_synthesis:
        video = cv2.VideoCapture(os.path.join(src_dir, '/Celeb-synthesis', file))
        frame_num = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_m.append(frame_num)
        print(file)
        print(frame_num)
        print('---------------')
    for [file, _] in YouTube_real:
        video = cv2.VideoCapture(os.path.join(src_dir, '/YouTube-real', file))
        frame_num = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_m.append(frame_num)
        print(file)
        print(frame_num)
        print('---------------')
    frame_m = np.array(frame_m)
    print('====================')
    print(np.max(np.array(frame_m)))
    print(np.min(np.array(frame_m)))
    print(frame_m.shape)
    print(np.median(frame_m))
    print(np.mean(frame_m))


def AUC_loss(outputs, labels, device, gamma, p=2):
    predictions = torch.sigmoid(outputs)
    pos_predict = predictions[torch.where(labels == 0)]
    neg_predict = predictions[torch.where(labels == 1)]
    pos_size = pos_predict.shape[0]
    neg_size = neg_predict.shape[0]
    if pos_size == 0 or neg_size == 0:
        return 0
    else:
        pos_neg_diff = -(torch.matmul(pos_predict, torch.ones([1, neg_size], device=device)) -
                         torch.matmul(torch.ones([pos_size, 1], device=device),
                                      torch.reshape(neg_predict, [-1, neg_size]))
                         - gamma)
    pos_neg_diff = torch.reshape(pos_neg_diff, [-1, 1])
    pos_neg_diff = torch.where(torch.gt(pos_neg_diff, 0), pos_neg_diff, torch.zeros([pos_size * neg_size, 1],
                                                                                    device=device))
    pos_neg_diff = torch.pow(pos_neg_diff, p)

    loss_approx_auc = torch.mean(pos_neg_diff)
    return loss_approx_auc


# def AUC_loss(y_, y, device, gamma, p=2):
#     X = y_[torch.where(y == 0)]
#     Y = y_[torch.where(y == 1)]
#     loss = torch.zeros(1, requires_grad=True, device=device)
#     if X.shape[0] == 0:
#         Y = torch.max(Y, 1)[0]
#         for j in Y:
#             if -j < gamma:
#                 loss = (-(- j - gamma)) ** p + loss
#     if Y.shape[0] == 0:
#         X = torch.max(X, 1)[0]
#         for i in X:
#             if i < gamma:
#                 loss = (-(i - gamma)) ** p + loss
#     if X.shape[0] != 0 and Y.shape[0] != 0:
#         X = torch.max(X, 1)[0]
#         Y = torch.max(Y, 1)[0]
#         for i in X:
#             for j in Y:
#                 if i - j < gamma:
#                     loss = (-(i - j - gamma)) ** p + loss
#     return loss


def merge_labels_to_ckpt(ck_path: str, train_file: str):
    """Merge labels to a checkpoint file.

    Args:
        ck_path(str): path to checkpoint file
        train_file(str): path to train set index file, eg. train.csv

    Return:
        This function will create a {ck_path}_patched.pth file.
    """
    # load model
    print('Loading checkpoint')
    ckpt = torch.load(ck_path)

    # load train files
    print('Loading dataset')
    raw_data = pandas.read_csv(train_file)
    train_set = Dataset(raw_data.to_numpy())

    # patch file name
    print('Patching')
    patch_path = ck_path.replace('.pth', '') + '_patched.pth'

    ck_dict = {'label_map': train_set.labels}
    names = ['epoch', 'model_state_dict', 'optimizer_state_dict']
    for name in names:
        ck_dict[name] = ckpt[name]

    torch.save(ck_dict, patch_path)
    print('Patched checkpoint has been saved to {}'.format(patch_path))


def tensor2im(input_image, imtype=np.uint8):
    """"将tensor的数据类型转成numpy类型，并反归一化.

    Parameters:
        input_image (tensor) --  输入的图像tensor数组
        imtype (type)        --  转换后的numpy的数据类型
    """
    mean = [0.485, 0.456, 0.406]  # 自己设置的
    std = [0.229, 0.224, 0.225]  # 自己设置的
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        for i in range(len(mean)):
            image_numpy[i] = image_numpy[i] * std[i] + mean[i]
        image_numpy = image_numpy * 255
        image_numpy = np.transpose(image_numpy, (1, 2, 0))  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def save_img(im, path):
    """im可是没经过任何处理的tensor类型的数据,将数据存储到path中

    Parameters:
        im (tensor) --  输入的图像tensor数组
        path (str)  --  图像保存的路径
        size (int)  --  一行有size张图,最好是2的倍数
    """
    # im_grid = torchvision.utils.make_grid(im, size) #将batchsize的图合成一张图
    im_numpy = tensor2im(im)  # 转成numpy类型并反归一化
    im_array = Image.fromarray(im_numpy)
    im_array.save(path)


def new_path(path):
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except Exception:
            os.makedirs(path)


def parse_args():
    parser = argparse.ArgumentParser(usage='python3 tools.py -i path/to/train.csv -r path/to/checkpoint')
    parser.add_argument('-i', '--data_path', help='path to your dataset index file')
    parser.add_argument('-r', '--restore_from', help='path to the checkpoint', default=None)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    # merge_labels_to_ckpt(args.restore_from, args.data_path)
    # return_raw_data('/Users/pu/Desktop/Celeb-DF-v2')
    frame_range('/Users/pu/Desktop/Celeb-DF-v2')
