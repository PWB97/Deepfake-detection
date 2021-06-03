import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas
import os
import argparse

from models.model import cRNN, get_resnet_3d, CNN, Baseline
from dataloader import FrameDataset, Dataset
import config
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, roc_auc_score


def test(model: nn.Sequential, test_loader: torch.utils.data.DataLoader, model_type, device):
    model.eval()

    print('Size of Test Set: ', len(test_loader.dataset))

    # 准备在测试集上验证模型性能
    test_loss = 0
    y_gd = []
    frame_y_gd = []
    y_pred = []
    frame_y_pred = []

    with torch.no_grad():
        if config.net_params.get('our'):
            for X, y in tqdm(test_loader, desc='Validating plus frame level'):
                X, y = X.to(device), y.to(device)
                frame_y = y.view(-1, 1)
                frame_y = frame_y.repeat(1, 300)
                frame_y = frame_y.flatten()
                y_, cnn_y = model(X)

                y_ = y_.argmax(dim=1)
                frame_y_ = cnn_y.argmax(dim=1)

                y_gd += y.cpu().numpy().tolist()
                y_pred += y_.cpu().numpy().tolist()
                frame_y_gd += frame_y.cpu().numpy().tolist()
                frame_y_pred += frame_y_.cpu().numpy().tolist()

                test_video_acc = accuracy_score(y_gd, y_pred)
                test_video_auc = roc_auc_score(y_gd, y_pred)
                test_frame_acc = accuracy_score(frame_y_gd, frame_y_pred)
                test_frame_auc = roc_auc_score(frame_y_gd, frame_y_pred)
                print('Test video avg loss: %0.4f, acc: %0.2f, auc: %0.2f\n' % (
                test_loss, test_video_acc, test_video_auc))
                print('Test frame avg loss: %0.4f, acc: %0.2f, auc: %0.2f\n' % (
                test_loss, test_frame_acc, test_frame_auc))


        else:
            for X, y in tqdm(test_loader, desc='Validating plus frame level'):
                X, y = X.to(device), y.to(device)
                cnn_y = model(X)
                frame_y_ = cnn_y.argmax(dim=1)
                frame_y_gd += y.cpu().numpy().tolist()
                frame_y_pred += frame_y_.cpu().numpy().tolist()
            test_frame_acc = accuracy_score(frame_y_gd, frame_y_pred)
            test_frame_auc = roc_auc_score(frame_y_gd, frame_y_pred)
            print('Test frame avg loss: %0.4f, acc: %0.2f, auc: %0.2f\n' % (test_loss, test_frame_acc, test_frame_auc))

    return frame_y_gd, frame_y_pred


def parse_args():
    parser = argparse.ArgumentParser(usage='python3 main.py -i path/to/data -r path/to/checkpoint')
    parser.add_argument('-i', '--data_path', help='path to your datasets', default='/data2/guesthome/wenbop/ffdf_c40')
    # parser.add_argument('-i', '--data_path', help='path to your datasets', default='/Users/pu/Desktop/dataset_dlib')
    parser.add_argument('-r', '--restore_from', help='path to the checkpoint',
                        default='/data2/guesthome/wenbop/modules/ff/bi-model_type-baseline_gru_ep-19.pth')
    # parser.add_argument('-g', '--gpu', help='visible gpu ids', default='4,5,7')
    parser.add_argument('-g', '--gpu', help='visible gpu ids', default='0,1,2,3')
    args = parser.parse_args()
    return args


def draw_auc():
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    args = parse_args()
    data_path = args.data_path
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    raw_data = pandas.read_csv(os.path.join(data_path, '%s.csv' % 'test'))
    dataloader = DataLoader(Dataset(raw_data.to_numpy()), **config.dataset_params)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = Baseline()
    device_count = torch.cuda.device_count()
    if device_count > 1:
        print('使用{}个GPU训练'.format(device_count))
        model = nn.DataParallel(model)
    model.to(device)
    ckpt = {}
    # 从断点继续训练
    if args.restore_from is not None:
        ckpt = torch.load(args.restore_from)
        # model.load_state_dict(ckpt['net'])
        model.load_state_dict(ckpt['model_state_dict'])
        print('Model is loaded from %s' % (args.restore_from))

    y_test, y_score = test(model, dataloader, 'baseline', device)

    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.figure()
        lw = 2
        plt.plot(fpr[0], tpr[0], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()


draw_auc()
