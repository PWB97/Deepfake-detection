import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
import pandas
import json
import os
import argparse
from tensorboardX import SummaryWriter

from model import cRNN, get_resnet_3d, CNN, Baseline
from models import model_selection
from classifier import SPPNet, ResNet
from dataloader import Dataset, FrameDataset
import config
import tools


def train_on_epochs(train_loader: DataLoader, test_loader: DataLoader, restore_from: str = None, size=300):
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
    elif config.model_type == 'xception':
        model, *_ = model_selection(modelname='xception', num_out_classes=2)
    elif config.model_type == 'fwa':
        model = SPPNet(backbone=50)
    elif config.model_type == 'res50':
        model = ResNet(layers=50)
    elif config.model_type == 'res101':
        model = ResNet(layers=101)
    elif config.model_type == 'res152':
        model = ResNet(layers=152)
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
        ckpt = torch.load(restore_from, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        print('Model is loaded from %s' % restore_from)

    # 提取网络参数，准备进行训练
    model_params = model.parameters()
    # 设定优化器
    # if device_count > 1:
    #     optimizer = torch.optim.Adam([
    #         dict(params=model.module.rnn.parameters()),
    #         dict(params=model.module.fc_cnn.parameters()),
    #         dict(params=model.module.global_pool.parameters()),
    #         dict(params=model.module.fc_rnn.parameters()),
    #         dict(params=model.module.cnn.parameters(), lr=config.learning_rate / 10)
    #     ], lr=config.learning_rate, weight_decay=0.0001)
    # else:
    #     optimizer = torch.optim.Adam([
    #         dict(params=model.rnn.parameters()),
    #         dict(params=model.fc_cnn.parameters()),
    #         dict(params=model.global_pool.parameters()),
    #         dict(params=model.fc_rnn.parameters()),
    #         dict(params=model.cnn.parameters(), lr=config.learning_rate / 10)
    #     ], lr=config.learning_rate, weight_decay=0.0001)
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
        writer = SummaryWriter(logdir='./log_model_type_%s_gru_auc_%d' % (config.model_type, config.gamma))
    elif not config.net_params.get('use_gru') and config.loss_type == 'CE':
        writer = SummaryWriter(logdir='./log_model_type_%s' % config.model_type)
    elif not config.net_params.get('use_gru') and config.loss_type == 'AUC':
        writer = SummaryWriter(logdir='./log_model_type_%s_auc_%d' % (config.model_type, config.gamma))

    for ep in range(start_ep, config.epoches):
        train_losses, train_scores = train(model, train_loader, optimizer, ep, config.model_type, config.loss_type,
                                           writer, device, size=size)
        test_loss, test_score, test_auc = validation(model, test_loader, ep, config.model_type, config.loss_type,
                                                     writer, device, size=size)

        # 保存信息
        info['train_losses'].append(train_losses)
        info['train_scores'].append(train_scores)
        info['test_losses'].append(test_loss)
        info['test_scores'].append(test_score)
        info['test_auc'].append(test_auc)

        # 保存模型
        if config.net_params.get('bi_branch'):
            if config.net_params.get('use_gru') and config.loss_type == 'CE':
                ckpt_path = os.path.join(save_path, 'bi-model_type-%s_gru_ep-%d.pth' % (config.model_type, ep))
            elif config.net_params.get('use_gru') and config.loss_type == 'AUC':
                ckpt_path = os.path.join(save_path, 'bi-model_type-%s_gru_auc_%f_ep-%d.pth' % (
                    config.model_type, config.gamma, ep))
            elif not config.net_params.get('use_gru') and config.loss_type == 'CE':
                ckpt_path = os.path.join(save_path, 'bi-model_type-%s_ep-%d.pth' % (config.model_type, ep))
            elif not config.net_params.get('use_gru') and config.loss_type == 'AUC':
                ckpt_path = os.path.join(save_path,
                                         'bi-model_type-%s_auc_%f_ep-%d.pth' % (config.model_type, config.gamma, ep))
        else:
            if config.net_params.get('use_gru') and config.loss_type == 'CE':
                ckpt_path = os.path.join(save_path, 'nb-model_type-%s_gru_ep-%d.pth' % (config.model_type, ep))
            elif config.net_params.get('use_gru') and config.loss_type == 'AUC':
                ckpt_path = os.path.join(save_path,
                                         'nb-model_type-%s_gru_auc_%f_ep-%d.pth' % (
                                             config.model_type, config.gamma, ep))
            elif not config.net_params.get('use_gru') and config.loss_type == 'CE':
                ckpt_path = os.path.join(save_path, 'nb-model_type-%s_ep-%d.pth' % (config.model_type, ep))
            elif not config.net_params.get('use_gru') and config.loss_type == 'AUC':
                ckpt_path = os.path.join(save_path,
                                         'nb-model_type-%s_auc_%f_ep-%d.pth' % (config.model_type, config.gamma, ep))
        if (ep + 1) % config.save_interval == 0:
            torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'label_map': train_loader.dataset.labels
            }, ckpt_path)
            print('Model of Epoch %3d has been saved to: %s' % (ep, ckpt_path))
    if config.net_params.get('use_gru') and config.loss_type == 'CE':
        with open('./%s_gru_train_info.json' % config.model_type, 'w') as f:
            json.dump(info, f)
    elif config.net_params.get('use_gru') and config.loss_type == 'AUC':
        with open('./%s_gru_auc_%d_train_info.json' % (config.model_type, config.gamma), 'w') as f:
            json.dump(info, f)
    elif not config.net_params.get('use_gru') and config.loss_type == 'CE':
        with open('./%s_train_info.json' % config.model_type, 'w') as f:
            json.dump(info, f)
    elif not config.net_params.get('use_gru') and config.loss_type == 'AUC':
        with open('./%s_auc_%d_train_info.json' % (config.model_type, config.gamma), 'w') as f:
            json.dump(info, f)

    print('over!')


def load_data_list(file_path):
    return pandas.read_csv(file_path).to_numpy()


def train(model: nn.Sequential, dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, epoch,
          model_type, loss_type, writer, device, size=300):
    model.train()

    train_losses = []
    train_scores = []

    print('Size of Training Set: ', len(dataloader.dataset))

    for i, (X, y) in enumerate(dataloader):
        if model_type == 'end2end':
            X = X.transpose(1, 2)
        X = X.to(device)
        y = y.to(device)

        # 初始化优化器参数
        optimizer.zero_grad()
        # 执行前向传播
        if config.net_params.get('bi_branch'):
            frame_y = y.view(-1, 1)
            frame_y = frame_y.repeat(1, size)
            frame_y = frame_y.flatten()
            print(X.shape)
            y_, cnn_y = model(X)
            if loss_type == 'CE':
                video_loss_ce = F.binary_cross_entropy_with_logits(y_, y.reshape(-1, 1).float())
                frame_loss_ce = F.binary_cross_entropy_with_logits(cnn_y, frame_y.reshape(-1, 1).float())
                loss = video_loss_ce + frame_loss_ce
            elif loss_type == 'AUC':
                video_loss_ce = F.binary_cross_entropy_with_logits(y_, y.reshape(-1, 1).float())
                frame_loss_ce = F.binary_cross_entropy_with_logits(cnn_y, frame_y.reshape(-1, 1).float())
                frame_loss_auc = tools.AUC_loss(cnn_y, frame_y, device, config.gamma)
                video_loss_auc = tools.AUC_loss(y_, y, device, config.gamma)
                if frame_loss_auc == 0 and video_loss_auc == 0:
                    loss = frame_loss_ce + video_loss_ce
                else:
                    loss = 0.6 * (frame_loss_ce + video_loss_ce) + 0.4 * (video_loss_auc + frame_loss_auc)
                # loss = video_loss_auc + frame_loss_auc
        elif config.model_type == 'xception' or config.model_type == 'fwa' or config.model_type == 'res50' or config.model_type == 'res101' or config.model_type == 'res152':
            y_ = model(X)
            loss = F.cross_entropy(y_, y)
            if loss_type == 'AUC':
                loss += tools.AUC_loss(y_, y, device, config.gamma)
        else:
            y_ = model(X)
            if loss_type == 'CE':
                loss = F.cross_entropy(y_, y)
            else:
                loss = tools.AUC_loss(y_, y, device, config.gamma) + F.cross_entropy(y_, y)
        # 计算loss

        # 反向传播梯度
        loss.backward()
        optimizer.step()

        # y_ = y_.argmax(dim=1)
        y_ = torch.sigmoid(y_)
        y_ = [0 if i < 0.5 else 1 for i in y_]
        acc = accuracy_score(y.cpu().numpy(), y_)

        # 保存loss等信息
        train_losses.append(loss.item())
        train_scores.append(acc)

        writer.add_scalar(tag='train_loss', scalar_value=loss.item(), global_step=epoch * len(dataloader) + i)
        writer.add_scalar(tag='train_score', scalar_value=acc, global_step=epoch * len(dataloader) + i)

        if (i + 1) % config.log_interval == 0:
            print('[Epoch %3d]Training %3d of %3d: acc = %.2f, loss = %.2f' % (
                epoch, i + 1, len(dataloader), acc, loss.item()))

    return train_losses, train_scores


def validation(model: nn.Sequential, test_loader: torch.utils.data.DataLoader, epoch, model_type, loss_type, writer,
               device, size=300):
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
                X, y = X.to(device), y.to(device)
                frame_y = y.view(-1, 1)
                frame_y = frame_y.repeat(1, size)
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
        elif config.model_type == 'xception' or config.model_type == 'fwa' or config.model_type == 'res50' or config.model_type == 'res101' or config.model_type == 'res152':
            for X, y in tqdm(test_loader, desc='Validating plus frame level'):
                X, y = X.to(device), y.to(device)
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
    if not config.model_type == 'xception' and not config.model_type == 'fwa' and not config.model_type == 'res50' and \
            not config.model_type == 'res101' and not config.model_type == 'res152':
        writer.add_scalar(tag='test_video_acc', scalar_value=test_video_acc, global_step=epoch)
        writer.add_scalar(tag='test_video_auc', scalar_value=test_video_auc, global_step=epoch)
    if config.net_params.get(
            'bi_branch') or config.model_type == 'xception' or config.model_type == 'fwa' \
            or config.model_type == 'res50' or config.model_type == 'res101' or config.model_type == 'res152':
        writer.add_scalar(tag='test_frame_acc', scalar_value=test_frame_acc, global_step=epoch)
        writer.add_scalar(tag='test_frame_auc', scalar_value=test_frame_auc, global_step=epoch)
    if not config.model_type == 'xception' and not config.model_type == 'fwa' and not config.model_type == 'res50' \
            and not config.model_type == 'res101' and not config.model_type == 'res152':
        print('[Epoch %3d]Test video avg loss: %0.4f, acc: %0.2f, auc: %0.2f\n' % (
            epoch, test_loss, test_video_acc, test_video_auc))
    if config.net_params.get(
            'bi_branch') or config.model_type == 'xception' or config.model_type == 'fwa' \
            or config.model_type == 'res50' or config.model_type == 'res101' or config.model_type == 'res152':
        print('[Epoch %3d]Test frame avg loss: %0.4f, acc: %0.2f, auc: %0.2f\n' % (
            epoch, test_loss, test_frame_acc, test_frame_auc))

    if not config.model_type == 'xception' and not config.model_type == 'fwa' and not config.model_type == 'res50' \
            and not config.model_type == 'res101' and not config.model_type == 'res152':
        return test_loss, test_video_acc, test_video_auc
    else:
        return test_loss, test_frame_acc, test_frame_auc


def parse_args():
    parser = argparse.ArgumentParser(usage='python3 train.py -i path/to/data -r path/to/checkpoint')
    parser.add_argument('-i', '--data_path', help='path to your datasets',
                        default='/home/asus/celeb_dataset')
    # parser.add_argument('-i', '--data_path', help='path to your datasets', default='/Users/pu/Desktop/dataset_dlib')
    parser.add_argument('-r', '--restore_from', help='path to the checkpoint', default=None)
    parser.add_argument('-g', '--gpu', help='visible gpu ids', default='0,1,2,3')
    parser.add_argument('-s', '--size', help='size of video frames', default=300)
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
        if config.model_type == 'xception' or config.model_type == 'fwa' or config.model_type == 'res50' or \
                config.model_type == 'res101' or config.model_type == 'res152':
            dataloaders[name] = DataLoader(FrameDataset(raw_data.to_numpy()), **config.dataset_params)
        else:
            if name == 'test':
                dataloaders[name] = DataLoader(Dataset(data_list=raw_data.to_numpy(), aug=False, add_channel=False,
                                                       frame_num=args.size),
                                               **config.dataset_params)
            else:
                dataloaders[name] = DataLoader(Dataset(data_list=raw_data.to_numpy(), aug=False, add_channel=False,
                                                       frame_num=args.size),
                                               **config.dataset_params)
    train_on_epochs(dataloaders['train'], dataloaders['test'], args.restore_from, size=args.size)
