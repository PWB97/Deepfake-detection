from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from tensorboardX import SummaryWriter

from utils.dataloader import Dataset, FrameDataset
from utils.tools import *
from utils.focalloss import *


def train_on_epochs(train_loader: DataLoader, test_loader: DataLoader, restore_from: str = None, size=300):
    # 配置训练时环境
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # 实例化计算图模型
    model, *_ = model_selection(modelname='xception', num_out_classes=config.net_params.get('num_classes'))
    model.to(device)

    # 多GPU训练
    device_count = torch.cuda.device_count()
    if device_count > 1:
        print('使用{}个GPU训练'.format(device_count))
        model = nn.DataParallel(model)

    ckpt = {}
    # 从断点继续训练
    if restore_from is not None:
        if config.model_type == 'fwa':
            ckpt = torch.load(restore_from)
            model.load_state_dict(ckpt['net'])
        else:
            ckpt = torch.load(restore_from, map_location='cpu')
            model.load_state_dict(ckpt['model_state_dict'])
        print('Model is loaded from %s' % restore_from)

    # 提取网络参数，准备进行训练
    model_params = model.parameters()
    optimizer = torch.optim.Adam(model_params, lr=config.learning_rate)
    # optimizer = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    if restore_from is not None and config.model_type != 'fwa':
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    # 训练时数据
    info = {
        'train_losses': [],
        'train_scores': [],
        'test_losses': [],
        'test_scores': [],
        'test_auc': []
    }

    start_ep = ckpt['epoch'] + 1 if 'epoch' in ckpt and config.model_type != 'fwa' else 0

    save_path = './checkpoints/' + config.model_type + str(config.net_params.get('use_gru')) + \
                str(config.net_params.get('bi_branch'))
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # 开始训练
    writer = SummaryWriter(logdir='./log_model_type_%s_auc_%d' % (config.model_type, config.gamma))

    for ep in range(start_ep, config.epoches):
        train_losses, train_scores = train(model, train_loader, optimizer, ep,
                                           writer, device, o_s=config.net_params.get('num_classes'))
        test_loss, test_score, test_auc = validation(model, test_loader, ep,
                                                     writer, device, o_s=config.net_params.get('num_classes'))

        # 保存信息
        info['train_losses'].append(train_losses)
        info['train_scores'].append(train_scores)
        info['test_losses'].append(test_loss)
        info['test_scores'].append(test_score)
        info['test_auc'].append(test_auc)

        # 保存模型
        ckpt_path = os.path.join(save_path,
                                 'xception-%s_auc_%f_ep-%d.pth' % (config.model_type, config.gamma, ep))
        if (ep + 1) % config.save_interval == 0:
            torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'label_map': train_loader.dataset.labels
            }, ckpt_path)
            print('Model of Epoch %3d has been saved to: %s' % (ep, ckpt_path))
        with open('./%s_gru_train_info.json' % config.model_type, 'w') as f:
            json.dump(info, f)
    print('over!')


def load_data_list(file_path):
    return pandas.read_csv(file_path).to_numpy()


def train(model: nn.Sequential, dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, epoch,
          writer, device, o_s=2):
    model.train()

    train_losses = []
    train_scores = []

    print('Size of Training Set: ', len(dataloader.dataset))

    for i, (X, y) in enumerate(dataloader):

        X = X.to(device)
        y = y.to(device)

        # 初始化优化器参数
        optimizer.zero_grad()

        y_ = model(X)
        if o_s == 2:
            loss = F.cross_entropy(y_, y)
        else:
            loss = F.binary_cross_entropy_with_logits(y_, y.reshape(-1, 1).float())
            loss += tools.AUC_loss(y_, y, device, config.gamma)

        # 计算loss
        # 反向传播梯度
        loss.backward()
        optimizer.step()

        if o_s == 2:
            y_ = y_.argmax(dim=1)
            acc = accuracy_score(y_.cpu().numpy(), y.cpu().numpy())
        else:
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


def validation(model: nn.Sequential, test_loader: torch.utils.data.DataLoader, epoch, writer, device, o_s=2):
    model.eval()

    print('Size of Test Set: ', len(test_loader.dataset))

    # 准备在测试集上验证模型性能
    test_loss = 0
    frame_y_gd = []
    frame_y_pred = []

    # 不需要反向传播，关闭求导
    with torch.no_grad():
        if config.net_params.get('bi_branch'):

            for X, y in tqdm(test_loader, desc='Validating plus frame level'):
                X, y = X.to(device), y.to(device)
                cnn_y = model(X)
                if os == 2:
                    frame_y_ = cnn_y.argmax(dim=1)
                else:
                    frame_y_ = torch.sigmoid(cnn_y)

                frame_y_gd += y.cpu().numpy().tolist()
                frame_y_pred += frame_y_.cpu().numpy().tolist()
            if os == 2:
                test_frame_auc = roc_auc_score(frame_y_gd, frame_y_pred)
                frame_y_pred = [0 if i < 0.5 else 1 for i in frame_y_pred]
            else:
                test_frame_auc = roc_auc_score(frame_y_gd, frame_y_pred)
            test_frame_acc = accuracy_score(frame_y_gd, frame_y_pred)
            test_frame_recall = recall_score(frame_y_gd, frame_y_pred)
            test_frame_f1 = f1_score(frame_y_gd, frame_y_pred)
            test_frame_precision = precision_score(frame_y_gd, frame_y_pred)

    # writer.add_scalar(tag='test_loss', scalar_value=test_loss, global_step=epoch)
    writer.add_scalar(tag='test_frame_acc', scalar_value=test_frame_acc, global_step=epoch)
    writer.add_scalar(tag='test_frame_auc', scalar_value=test_frame_auc, global_step=epoch)
    writer.add_scalar(tag='test_frame_recall', scalar_value=test_frame_recall, global_step=epoch)
    writer.add_scalar(tag='test_frame_f1', scalar_value=test_frame_f1, global_step=epoch)
    writer.add_scalar(tag='test_frame_precision', scalar_value=test_frame_precision, global_step=epoch)
    print('[Epoch %3d]Test frame avg loss: %0.4f, acc: %0.2f, auc: %0.2f\n' % (
        epoch, test_loss, test_frame_acc, test_frame_auc))

    return test_loss, test_frame_acc, test_frame_auc


def parse_args():
    parser = argparse.ArgumentParser(usage='python3 train.py -i path/to/data -r path/to/checkpoint')
    # parser.add_argument('-i', '--data_path', help='path to your datasets',
    #                     default='/home/asus/celeb_20')
    parser.add_argument('-i', '--data_path', help='path to your datasets', default='/Users/pu/Desktop/dataset_dlib')
    parser.add_argument('-r', '--restore_from', help='path to the checkpoint', default=None)
    parser.add_argument('-g', '--gpu', help='visible gpu ids', default='0')
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
        dataloaders[name] = DataLoader(FrameDataset(raw_data.to_numpy()), **config.dataset_params)
    train_on_epochs(dataloaders['train'], dataloaders['test'], args.restore_from, size=args.size)
