"""
Copyright (c) 2019, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen
-----------------------------------------------------
Script for testing Capsule-Forensics-v2 on FaceForensics++ database (Real, DeepFakes, Face2Face, FaceSwap)
"""

import sys
sys.setrecursionlimit(15000)
import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from torch.autograd import Variable
import torch.utils.data
import torchvision.datasets as dset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import model_big
import pandas

from utils.dataloader import FrameDataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default ='databases/faceforensicspp', help='path to dataset')
parser.add_argument('--test_set', default ='test', help='test set')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--imageSize', type=int, default=300, help='the height / width of the input image to network')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--outf', default='checkpoints/binary_faceforensicspp', help='folder to output model checkpoints')
parser.add_argument('--random', action='store_true', default=False, help='enable randomness for routing matrix')
parser.add_argument('--id', type=int, default=21, help='checkpoint ID')

opt = parser.parse_args()
print(opt)

if __name__ == '__main__':

    # text_writer = open(os.path.join(opt.outf, 'test.txt'), 'w')

    transform_fwd = transforms.Compose([
        transforms.Resize((opt.imageSize, opt.imageSize)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])


    # dataset_test = dset.ImageFolder(root=os.path.join(opt.dataset, opt.test_set), transform=transform_fwd)
    # assert dataset_test
    # dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers))
    # dataloaders = {}
    # for name in ['train', 'test']:
    #     raw_data = pandas.read_csv(os.path.join(opt.dataset, '%s.csv' % name))
    #     dataloaders[name] = DataLoader(FrameDataset(raw_data.to_numpy()), **config.dataset_params)
    raw_data = pandas.read_csv(os.path.join(opt.dataset, 'test.csv'))
    dataloader_test = DataLoader(FrameDataset(raw_data.to_numpy()),
                                           batch_size=opt.batchSize,
                                           shuffle=True,
                                           num_workers=4,
                                           pin_memory=False)
    vgg_ext = model_big.VggExtractor()
    capnet = model_big.CapsuleNet(2, opt.gpu_id)

    capnet.load_state_dict(torch.load(os.path.join(opt.outf)))
    capnet.eval()

    if opt.gpu_id >= 0:
        vgg_ext.cuda(opt.gpu_id)
        capnet.cuda(opt.gpu_id)


    ##################################################################################

    tol_label = np.array([], dtype=np.float)
    tol_pred = np.array([], dtype=np.float)
    tol_pred_prob = np.array([], dtype=np.float)

    count = 0
    loss_test = 0

    for img_data, labels_data in tqdm(dataloader_test):

        labels_data[labels_data > 1] = 1
        img_label = labels_data.numpy().astype(np.float)

        if opt.gpu_id >= 0:
            img_data = img_data.cuda(opt.gpu_id)
            labels_data = labels_data.cuda(opt.gpu_id)

        input_v = Variable(img_data)

        x = vgg_ext(input_v)
        classes, class_ = capnet(x, random=opt.random)

        output_dis = class_.data.cpu()
        output_pred = np.zeros((output_dis.shape[0]), dtype=np.float)

        for i in range(output_dis.shape[0]):
            if output_dis[i,1] >= output_dis[i,0]:
                output_pred[i] = 1.0
            else:
                output_pred[i] = 0.0

        tol_label = np.concatenate((tol_label, img_label))
        tol_pred = np.concatenate((tol_pred, output_pred))
        
        pred_prob = torch.softmax(output_dis, dim=1)
        tol_pred_prob = np.concatenate((tol_pred_prob, pred_prob[:, 1].data.numpy()))

        count += 1

    acc_test = metrics.accuracy_score(tol_label, tol_pred)
    auc_test = metrics.roc_auc_score(tol_label, tol_pred_prob)
    f1_test = metrics.f1_score(tol_label, tol_pred)
    recall_test = metrics.recall_score(tol_label, tol_pred)
    precision = metrics.precision_score(tol_label, tol_pred)
    loss_test /= count

    fpr, tpr, thresholds = roc_curve(tol_label, tol_pred_prob, pos_label=1)
    np.save('./m/cap/f_fpr.npy', fpr)
    np.save('./m/cap/f_tpr.npy', tpr)
    # eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    #
    # fnr = 1 - tpr
    # hter = (fpr + fnr)/2
    # print('[Epoch %d] Train loss: %.4f   acc: %.2f | Test loss: %.4f  acc: %.2f auc: %.2f'
    #       % (opt.id, acc_test * 100, loss_test, acc_test * 100, auc_test * 100))
    print('[Epoch %d] Test acc: %.2f   AUC: %.2f    f1: %.2f    recall:%.2f     precision:%.2f'
          % (opt.id, acc_test*100, auc_test*100, f1_test, recall_test, precision))
    # text_writer.write('%d,%.2f,%.2f\n'% (opt.id, acc_test*100, eer*100))
    #
    # text_writer.flush()
    # text_writer.close()
