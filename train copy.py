import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
import pandas
import json
import os
import argparse
from tensorboardX import SummaryWriter

from model import cRNN, get_resnet_3d, CNN, Baseline
from dataloader import Dataset
import config
import tools


def train_on_epochs(train_loader:DataLoader, test_loader:DataLoader, restore_from:str=None):
    # 配置训练时环境
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # 实例化计算图模型
    if config.model_type == 'baseline':
        model = Baseline(**config.net_params)
    elif config.model_type == 'cRNN':
        model = cRNN(**config.net_params)
    elif config.model_type == 'end2end':
        model = get_resnet_3d(**config.resnet_3d_params)
    else:
        model = CNN()
    model.to(device)

    # 多GPU训练
    device_count = torch.cuda.device_count()
    if device_count > 1:
        print('使用{}个GPU训练'.format(device_count))
        model = nn.DataParallel(model)

    ckpt = {}
    # 从断点继续训练
    if restore_from is not None:
        ckpt = torch.load(restore_from)
        model.load_state_dict(ckpt['model_state_dict'])
        print('Model is loaded from %s' % (restore_from))

    # 提取网络参数，准备进行训练
    model_params = model.parameters()

    # 设定优化器
    optimizer = torch.optim.Adam(model_params, lr=config.learning_rate)

    if restore_from is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    # 训练时数据
    info = {
        'train_losses': [],
        'train_scores': [],
        'test_losses': [],
        'test_scores': [],
        'test_auc': []
    }

    start_ep = ckpt['epoch'] + 1 if 'epoch' in ckpt else 0

    save_path = './checkpoints'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # 开始训练
    if config.net_params.get('use_gru') and config.loss_type == 'CE':
        writer = SummaryWriter(logdir='./log_model_type_%s_gru' % config.model_type)
    elif config.net_params.get('use_gru') and config.loss_type == 'AUC':
        writer = SummaryWriter(logdir='./log_model_type_%s_gru_auc_%d' %(config.model_type, config.gamma))
    elif not config.net_params.get('use_gru') and config.loss_type == 'CE':
        writer = SummaryWriter(logdir='./log_model_type_%s' % config.model_type)
    elif not config.net_params.get('use_gru') and config.loss_type == 'AUC':
        writer = SummaryWriter(logdir='./log_model_type_%s_auc_%d' %(config.model_type, config.gamma))

    for ep in range(start_ep, config.epoches):
        train_losses, train_scores = train(model, train_loader, optimizer, ep, config.model_type, config.loss_type, writer, device)
        test_loss, test_score, test_auc = validation(model, test_loader, ep, config.model_type, config.loss_type, writer, device)

        # 保存信息
        info['train_losses'].append(train_losses)
        info['train_scores'].append(train_scores)
        info['test_losses'].append(test_loss)
        info['test_scores'].append(test_score)
        info['test_auc'].append(test_auc)

        # 保存模型
        if config.net_params.get('use_gru') and config.loss_type == 'CE':
            ckpt_path = os.path.join(save_path, 'model_type-%s_gru_ep-%d.pth' %(config.model_type, ep))
        elif config.net_params.get('use_gru') and config.loss_type == 'AUC':
            ckpt_path = os.path.join(save_path, 'model_type-%s_gru_auc_%d_ep-%d.pth' % (config.model_type, config.gamma, ep))
        elif not config.net_params.get('use_gru') and config.loss_type == 'CE':
            ckpt_path = os.path.join(save_path, 'model_type-%s_ep-%d.pth' % (config.model_type, ep))
        elif not config.net_params.get('use_gru') and config.loss_type == 'AUC':
            ckpt_path = os.path.join(save_path, 'model_type-%s_auc_%d_ep-%d.pth' % (config.model_type, config.gamma, ep))
        if (ep + 1) % config.save_interval == 0:
            torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'label_map': train_loader.dataset.labels
            }, ckpt_path)
            print('Model of Epoch %3d has been saved to: %s' % (ep, ckpt_path))
    if config.net_params.get('use_gru') and config.loss_type == 'CE':
        with open('./%s_gru_train_info.json' %config.model_type, 'w') as f:
            json.dump(info, f)
    elif config.net_params.get('use_gru') and config.loss_type == 'AUC':
        with open('./%s_gru_auc_%d_train_info.json' %(config.model_type, config.gamma), 'w') as f:
            json.dump(info, f)
    elif not config.net_params.get('use_gru') and config.loss_type == 'CE':
        with open('./%s_train_info.json' %config.model_type, 'w') as f:
            json.dump(info, f)
    elif not config.net_params.get('use_gru') and config.loss_type == 'AUC':
        with open('./%s_auc_%d_train_info.json' %(config.model_type, config.gamma), 'w') as f:
            json.dump(info, f)

    print('训练结束')

def load_data_list(file_path):
    return pandas.read_csv(file_path).to_numpy()

def train(model:nn.Sequential, dataloader:torch.utils.data.DataLoader, optimizer:torch.optim.Optimizer, epoch, model_type, loss_type, writer, device):
    model.train()

    train_losses = []
    train_scores = []

    print('Size of Training Set: ', len(dataloader.dataset))

    for i, (X, y) in enumerate(dataloader):
        if model_type == 4:
            X = X.transpose(1,2)
        X = X.to(device)
        y = y.to(device)

        # 初始化优化器参数
        optimizer.zero_grad()
        # 执行前向传播
        y_ = model(X)

        if loss_type == 'CE':
            loss = F.cross_entropy(y_, y)
        else:
            loss = tools.AUC_loss(y_, y, device, config.gamma)
        # 计算loss

        # 反向传播梯度
        loss.backward()
        optimizer.step()

        y_ = y_.argmax(dim=1)
        acc = accuracy_score(y_.cpu().numpy(), y.cpu().numpy())

        # 保存loss等信息
        train_losses.append(loss.item())
        train_scores.append(acc)

        writer.add_scalar(tag='train_loss', scalar_value=loss.item(), global_step=epoch*len(dataloader) + i)
        writer.add_scalar(tag='train_score', scalar_value=acc, global_step=epoch*len(dataloader) + i)

        if (i + 1) % config.log_interval == 0:
            print('[Epoch %3d]Training %3d of %3d: acc = %.2f, loss = %.2f' % (epoch, i + 1, len(dataloader), acc, loss.item()))

    return train_losses, train_scores

def validation(model:nn.Sequential, test_loader:torch.utils.data.DataLoader, epoch, model_type, loss_type, writer, device):
    model.eval()

    print('Size of Test Set: ', len(test_loader.dataset))

    # 准备在测试集上验证模型性能
    test_loss = 0
    y_gd = []
    y_pred = []

    # 不需要反向传播，关闭求导
    with torch.no_grad():
        for X, y in tqdm(test_loader, desc='Validating'):
            # 对测试集中的数据进行预测
            if model_type == 4:
                X = X.transpose(1, 2)
            X, y = X.to(device), y.to(device)
            y_ = model(X)

            # 计算loss
            if loss_type == 'CE':
                loss = F.cross_entropy(y_, y)
            else:
                loss = tools.AUC_loss(y_, y, device, config.gamma)
            test_loss += loss.item()

            # 收集prediction和ground truth
            y_ = y_.argmax(dim=1)
            y_gd += y.cpu().numpy().tolist()
            y_pred += y_.cpu().numpy().tolist()

    # 计算loss
    test_loss /= len(test_loader)
    # 计算正确率
    test_acc = accuracy_score(y_gd, y_pred)
    test_auc = roc_auc_score(y_gd, y_pred)

    writer.add_scalar(tag='test_loss', scalar_value=test_loss, global_step=epoch)
    writer.add_scalar(tag='test_acc', scalar_value=test_acc, global_step=epoch)
    writer.add_scalar(tag='test_auc', scalar_value=test_auc, global_step=epoch)

    print('[Epoch %3d]Test avg loss: %0.4f, acc: %0.2f, auc: %0.2f\n' % (epoch, test_loss, test_acc, test_auc))

    return test_loss, test_acc, test_auc

def parse_args():
    parser = argparse.ArgumentParser(usage='python3 train.py -i path/to/data -r path/to/checkpoint')
    parser.add_argument('-i', '--data_path', help='path to your datasets', default='/data2/guesthome/wenbop/dataset_dlib')
    # parser.add_argument('-i', '--data_path', help='path to your datasets', default='/Users/pu/Desktop/dataset_dlib')
    parser.add_argument('-r', '--restore_from', help='path to the checkpoint', default=None)
    parser.add_argument('-g', '--gpu', help='visible gpu ids', default='4,5,7')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    data_path = args.data_path
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 准备数据加载器
    dataloaders = {}
    for name in ['train', 'test']:
        raw_data = pandas.read_csv(os.path.join(data_path, '%s.csv' % name))
        dataloaders[name] = DataLoader(Dataset(raw_data.to_numpy()), **config.dataset_params)
    train_on_epochs(dataloaders['train'], dataloaders['test'], args.restore_from)