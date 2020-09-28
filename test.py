import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import pandas
import json
import os
import argparse
from tensorboardX import SummaryWriter

import numpy as np
from model import cRNN, get_resnet_3d, CNN, Baseline
from models import model_selection
from classifier import SPPNet, ResNet
from dataloader import FrameDataset, Dataset
import config
import scipy.misc
import tools
import matplotlib.pyplot as plt


def validation(model: nn.Sequential, test_loader: torch.utils.data.DataLoader, epoch, model_type, loss_type, writer,
               device):
    model.eval()

    print('Size of Test Set: ', len(test_loader.dataset))

    # 准备在测试集上验证模型性能
    test_loss = 0
    y_gd = []
    frame_y_gd = []
    y_pred = []
    frame_y_pred = []

    # 不需要反向传播，关闭求导
    with torch.no_grad():
        if config.net_params.get('bi_branch'):
            for X, y in tqdm(test_loader, desc='Validating plus frame level'):
                if model_type == 'end2end':
                    X = X.transpose(1, 2)
                # scipy.misc.toimage(X, cmin=0.0, cmax=1.0).save("./test.png")  # 保存图像
                # print('输出了！')
                X, y = X.to(device), y.to(device)
                frame_y = y.view(-1, 1)
                frame_y = frame_y.repeat(1, 300)
                frame_y = frame_y.flatten()
                y_, cnn_y = model(X)
                y_ = torch.sigmoid(y_)
                cnn_y = torch.sigmoid(cnn_y)

                # y_ = y_.argmax(dim=1)
                # frame_y_ = cnn_y.argmax(dim=1)
                frame_y_ = cnn_y

                y_gd += y.cpu().numpy().tolist()
                y_pred += y_.cpu().numpy().tolist()
                frame_y_gd += frame_y.cpu().numpy().tolist()
                frame_y_pred += frame_y_.cpu().numpy().tolist()

            y_pred_pro = y_pred
            frame_y_pred_pro = frame_y_pred
            y_pred = torch.tensor(y_pred)
            frame_y_pred = torch.tensor(frame_y_pred)
            y_pred = [0 if i < 0.5 else 1 for i in y_pred]
            frame_y_pred = [0 if i < 0.5 else 1 for i in frame_y_pred]
            test_video_acc = accuracy_score(y_gd, y_pred)
            test_video_auc = roc_auc_score(y_gd, y_pred_pro)
            test_frame_acc = accuracy_score(frame_y_gd, frame_y_pred)
            test_frame_auc = roc_auc_score(frame_y_gd, frame_y_pred_pro)

            # f_fpr, f_tpr, _ = roc_curve(np.array(frame_y_gd), np.array(frame_y_pred))
            # v_fpr, v_tpr, _ = roc_curve(np.array(y_gd), np.array(y_pred))
            # f_roc_auc = auc(f_fpr, f_tpr)
            # v_roc_auc = auc(v_fpr, v_tpr)
            #
            # plt.figure()
            # lw = 2
            # plt.plot(f_fpr, f_tpr, color='darkorange',
            #          lw=lw, label='ROC curve (area = %0.2f)' % f_roc_auc)
            # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            # plt.xlim([0.0, 1.0])
            # plt.ylim([0.0, 1.05])
            # plt.xlabel('False Positive Rate')
            # plt.ylabel('True Positive Rate')
            # plt.legend(loc="lower right")
            # plt.savefig('df_frame.png')
            #
            # plt.figure()
            # lw = 2
            # plt.plot(v_fpr, v_tpr, color='darkorange',
            #          lw=lw, label='ROC curve (area = %0.2f)' % v_roc_auc)
            # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            # plt.xlim([0.0, 1.0])
            # plt.ylim([0.0, 1.05])
            # plt.xlabel('False Positive Rate')
            # plt.ylabel('True Positive Rate')
            # plt.legend(loc="lower right")
            # plt.savefig('df_video.png')

        elif config.model_type == 'xception' or config.model_type == 'fwa' or config.model_type == 'res50' or config.model_type == 'res101' or config.model_type == 'res152':
            for X, y in tqdm(test_loader, desc='Validating plus frame level'):
                X, y = X.to(device), y.to(device)
                # X = X.reshape(-1, 3, 64, 64)
                # frame_y = y.view(-1, 1)
                # frame_y = frame_y.repeat(1, 300)
                # frame_y = frame_y.flatten()
                cnn_y = model(X)
                frame_y_ = cnn_y.argmax(dim=1)
                frame_y_gd += y.cpu().numpy().tolist()
                frame_y_pred += frame_y_.cpu().numpy().tolist()
            test_frame_acc = accuracy_score(frame_y_gd, frame_y_pred)
            test_frame_auc = roc_auc_score(frame_y_gd, frame_y_pred)
        else:
            for X, y in tqdm(test_loader, desc='Validating'):
                if model_type == 4:
                    X = X.transpose(1, 2)
                X, y = X.to(device), y.to(device)
                y_ = model(X)
                # # 计算loss
                # if loss_type == 'CE':
                #     loss = F.cross_entropy(y_, y)
                # else:
                #     loss = tools.AUC_loss(y_, y, device, config.gamma)
                # test_loss += loss.item()

                # 收集prediction和ground truth
                y_ = y_.argmax(dim=1)
                y_gd += y.cpu().numpy().tolist()
                y_pred += y_.cpu().numpy().tolist()

            # # 计算loss
            # test_loss /= len(test_loader)
            # 计算正确率
            test_video_acc = accuracy_score(y_gd, y_pred)
            test_video_auc = roc_auc_score(y_gd, y_pred)

    # writer.add_scalar(tag='test_loss', scalar_value=test_loss, global_step=epoch)
    # if not config.model_type == 'xception' and not config.model_type == 'fwa' and not config.model_type == 'res50' and not config.model_type == 'res101' and not config.model_type == 'res152':
    #     writer.add_scalar(tag='test_video_acc', scalar_value=test_video_acc, global_step=epoch)
    #     writer.add_scalar(tag='test_video_auc', scalar_value=test_video_auc, global_step=epoch)
    # if config.net_params.get('bi_branch') or config.model_type == 'xception' or config.model_type == 'fwa' or config.model_type == 'res50' or config.model_type == 'res101' or config.model_type == 'res152':
    #     writer.add_scalar(tag='test_frame_acc', scalar_value=test_frame_acc, global_step=epoch)
    #     writer.add_scalar(tag='test_frame_auc', scalar_value=test_frame_auc, global_step=epoch)
    if not config.model_type == 'xception' and not config.model_type == 'fwa' and not config.model_type == 'res50' and not config.model_type == 'res101' and not config.model_type == 'res152':
        print('[Epoch %3d]Test video avg loss: %0.4f, acc: %0.2f, auc: %0.2f\n' % (
        epoch, test_loss, test_video_acc, test_video_auc))
    if config.net_params.get(
            'bi_branch') or config.model_type == 'xception' or config.model_type == 'fwa' or config.model_type == 'res50' or config.model_type == 'res101' or config.model_type == 'res152':
        print('[Epoch %3d]Test frame avg loss: %0.4f, acc: %0.2f, auc: %0.2f\n' % (
        epoch, test_loss, test_frame_acc, test_frame_auc))

    if not config.model_type == 'xception' and not config.model_type == 'fwa' and not config.model_type == 'res50' and not config.model_type == 'res101' and not config.model_type == 'res152':
        return test_loss, test_video_acc, test_video_auc
    else:
        return test_loss, test_frame_acc, test_frame_auc


def parse_args():
    parser = argparse.ArgumentParser(usage='python3 train.py -i path/to/data -r path/to/checkpoint')
    parser.add_argument('-i', '--data_path', help='path to your datasets',
                        default='/data2/guesthome/wenbop/dataset_dlib')
    # parser.add_argument('-i', '--data_path', help='path to your datasets', default='/Users/pu/Desktop/dataset_dlib')
    parser.add_argument('-r', '--restore_from', help='path to the checkpoint', default='/data2/guesthome/wenbop/modules/auc/bi-model_type-baseline_gru_auc_0.150000_ep-15.pth')
    parser.add_argument('-g', '--gpu', help='visible gpu ids', default='0,1,2,3')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    data_path = args.data_path
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    raw_data = pandas.read_csv(os.path.join(data_path, '%s.csv' % 'test'))
    dataloader = DataLoader(Dataset(raw_data.to_numpy()), **config.dataset_params)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    # model = SPPNet(backbone=50)
    # model = ResNet(layers=50)
    model = Baseline(**config.net_params)
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
    validation(model=model, test_loader=dataloader, epoch=0, model_type=None, writer=None, device=device,
               loss_type=None)
