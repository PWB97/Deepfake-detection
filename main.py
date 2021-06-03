from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from tensorboardX import SummaryWriter

from models.model import cRNN, get_resnet_3d, CNN, Baseline
from fwa.classifier import SPPNet, ResNet
from utils.dataloader import FrameDataset
from utils.tools import *
from utils.focalloss import *
from utils.aucloss import AUCLoss
from self_attention_cv import ResNet50ViT, ViT

models = ['ours', 'cRNN', 'end2end', 'xception', 'fwa', 'resvit', 'vit', 'res50', 'res101', 'res152']


def train_on_epochs(train_loader: DataLoader, test_loader: DataLoader, opt):
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    model_type = models[opt.model_type]
    if model_type == 'baseline':
        model = Baseline(use_gru=opt.use_gru, bi_branch=opt.net_type == 2)
    elif model_type == 'cRNN':
        model = cRNN()
    elif model_type == 'end2end':
        model = get_resnet_3d()
    elif model_type == 'xception':
        model, *_ = model_selection(modelname='xception', num_out_classes=2)
    elif model_type == 'fwa':
        model = SPPNet(backbone=50)
    elif model_type == 'resvit':
        model = ResNet50ViT(img_dim=opt.img_size, pretrained_resnet=True, blocks=6,
                            num_classes=opt.num_classes, dim_linear_block=256, dim=256)
    elif model_type == 'vit':
        model = ViT(img_dim=opt.img_size, in_channels=3, patch_dim=16,
                    num_classes=opt.num_classes, dim=512)
    elif model_type == 'res50':
        model = ResNet(layers=50)
    elif model_type == 'res101':
        model = ResNet(layers=101)
    elif model_type == 'res152':
        model = ResNet(layers=152)
    else:
        model = CNN()

    model.to(device)

    device_count = torch.cuda.device_count()
    if device_count > 1:
        print('Using {} GPUs'.format(device_count))
        model = nn.DataParallel(model)

    ckpt = {}
    restore_from = opt.restore_from
    if restore_from is not None:
        if model_type == 'fwa':
            ckpt = torch.load(restore_from)
            model.load_state_dict(ckpt['net'])
        else:
            ckpt = torch.load(restore_from, map_location='cpu')
            model.load_state_dict(ckpt['model_state_dict'])
        print('Model is loaded from %s' % restore_from)

    model_params = model.parameters()
    optimizer = torch.optim.Adam(model_params, lr=opt.learning_rate)

    if restore_from is not None and model_type != 'fwa':
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    info = {
        'train_losses': [],
        'train_scores': [],
        'test_losses': [],
        'test_scores': [],
        'test_auc': []
    }

    start_ep = ckpt['epoch'] + 1 if 'epoch' in ckpt and model_type != 'fwa' else 0

    save_path = './checkpoints/' + model_type + str(opt.use_gru) + str(opt.net_type)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    writer = SummaryWriter(logdir='./log-model_type:%s-gru:%s-loss:%s-gamma:%s'
                                  % (model_type, str(opt.use_gru), str(opt.loss_type), str(opt.gamma)))

    for ep in range(start_ep, opt.epoch):

        if opt.mode:
            train_losses, train_scores = train(model, train_loader, optimizer, writer, device, ep, opt)
            info['train_losses'].append(train_losses)
            info['train_scores'].append(train_scores)

        test_loss, test_score, test_auc = validation(model, test_loader, writer, device, ep, opt)

        info['test_losses'].append(test_loss)
        info['test_scores'].append(test_score)
        info['test_auc'].append(test_auc)

        ckpt_path = os.path.join(save_path, 'model_type:%s-gru:%s-loss:%s-gamma:%s.pth'
                                 % (model_type, str(opt.use_gru), str(opt.loss_type), str(opt.gamma)))

        if (ep + 1) % opt.save_interval == 0:
            torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'label_map': train_loader.dataset.labels
            }, ckpt_path)
            print('Model of Epoch %3d has been saved to: %s' % (ep, ckpt_path))

    with open('./train_info-model_type:%s-gru:%s-loss:%s-gamma:%s.json'
              % (model_type, str(opt.use_gru), str(opt.loss_type), str(opt.gamma)), 'w') as f:
        json.dump(info, f)

    print('over!')


def load_data_list(file_path):
    return pandas.read_csv(file_path).to_numpy()


def train(model: nn.Sequential, dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer, writer, device, epoch, opt):
    model.train()

    train_losses = []
    train_scores = []

    model_type = models[opt.model_type]

    print('Size of Training Set: ', len(dataloader.dataset))
    if opt.num_classes == 1:
        if opt.loss_type == 2:
            criterion = BCEFocalLoss(gamma=opt.gamma, alpha=opt.alpha)
        if opt.loss_type == 1:
            criterion = AUCLoss(device=device, gamma=opt.gamma, alpha=opt.alpha)
        else:
            criterion = F.binary_cross_entropy_with_logits
    else:
        criterion = F.cross_entropy

    for i, (X, y) in enumerate(dataloader):
        if model_type == 'end2end':
            X = X.transpose(1, 2)
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        if opt.net_type == 2:
            frame_y = y.view(-1, 1)
            frame_y = frame_y.repeat(1, opt.frame_size)
            frame_y = frame_y.flatten()
            y_, cnn_y = model(X)
            if opt.num_classes == 1:
                y = y.reshape(-1, 1).float()
                frame_y = frame_y.reshape(-1, 1).float()
            video_loss = criterion(y_, y)
            frame_loss = criterion(cnn_y, frame_y)
            loss = opt.beta * video_loss + (1 - opt.beta) * frame_loss

        else:
            y_ = model(X)
            if opt.num_classes == 1:
                y = y.reshape(-1, 1).float()
            loss = criterion(y_, y.reshape(-1, 1).float())

        loss.backward()

        optimizer.step()

        if opt.num_classes == 2:
            y_ = y_.argmax(dim=1)
            acc = accuracy_score(y_.cpu().numpy(), y.cpu().numpy())
        else:
            y_ = torch.sigmoid(y_)
            y_ = [0 if i < 0.5 else 1 for i in y_]
            acc = accuracy_score(y.cpu().numpy(), y_)

        train_losses.append(loss.item())
        train_scores.append(acc)

        writer.add_scalar(tag='train_loss', scalar_value=loss.item(), global_step=epoch * len(dataloader) + i)
        writer.add_scalar(tag='train_score', scalar_value=acc, global_step=epoch * len(dataloader) + i)

        if (i + 1) % opt.log_interval == 0:
            print('[Epoch %3d]Training video level %3d of %3d: acc = %.2f, loss = %.2f' % (
                epoch, i + 1, len(dataloader), acc, loss.item()))

    return train_losses, train_scores


def validation(model: nn.Sequential, test_loader: torch.utils.data.DataLoader, writer, device, epoch, opt):
    model.eval()

    print('Size of Test Set: ', len(test_loader.dataset))

    test_loss = 0
    y_gd = []
    frame_y_gd = []
    y_pred = []
    frame_y_pred = []

    # 不需要反向传播，关闭求导
    with torch.no_grad():
        if opt.net_type == 2:
            for X, y in tqdm(test_loader, desc='Validating plus frame level'):
                # if model_type == 'end2end':
                #     X = X.transpose(1, 2)
                X, y = X.to(device), y.to(device)
                frame_y = y.view(-1, 1)
                frame_y = frame_y.repeat(1, opt.frame_size)
                frame_y = frame_y.flatten()
                y_, cnn_y = model(X)
                if opt.num_classes == 2:
                    y_ = y_.argmax(dim=1)
                    frame_y_ = cnn_y.argmax(dim=1)
                else:
                    y_ = torch.sigmoid(y_)
                    frame_y_ = torch.sigmoid(cnn_y)
                y_gd += y.cpu().numpy().tolist()
                y_pred += y_.cpu().numpy().tolist()
                frame_y_gd += frame_y.cpu().numpy().tolist()
                frame_y_pred += frame_y_.cpu().numpy().tolist()

            if opt.num_classes == 1:
                y_pred_pro = y_pred
                frame_y_pred_pro = frame_y_pred
                y_pred = torch.tensor(y_pred)
                frame_y_pred = torch.tensor(frame_y_pred)
                y_pred = [0 if i < 0.5 else 1 for i in y_pred]
                frame_y_pred = [0 if i < 0.5 else 1 for i in frame_y_pred]
                test_video_auc = roc_auc_score(y_gd, y_pred_pro)
                test_frame_auc = roc_auc_score(frame_y_gd, frame_y_pred_pro)
            else:
                test_video_auc = roc_auc_score(y_gd, y_pred)
                test_frame_auc = roc_auc_score(frame_y_gd, frame_y_pred)

            test_video_acc = accuracy_score(y_gd, y_pred)
            test_video_recall = recall_score(y_gd, y_pred)
            test_video_f1 = f1_score(y_gd, y_pred)
            test_video_precision = precision_score(y_gd, y_pred)
            test_frame_acc = accuracy_score(frame_y_gd, frame_y_pred)
            test_frame_recall = recall_score(frame_y_gd, frame_y_pred)
            test_frame_f1 = f1_score(frame_y_gd, frame_y_pred)
            test_frame_precision = precision_score(frame_y_gd, frame_y_pred)

        elif opt.net_type == 1:
            for X, y in tqdm(test_loader, desc='Validating plus frame level'):
                X, y = X.to(device), y.to(device)
                cnn_y = model(X)
                if opt.num_classes == 2:
                    frame_y_ = cnn_y.argmax(dim=1)
                else:
                    frame_y_ = torch.sigmoid(cnn_y)
                frame_y_gd += y.cpu().numpy().tolist()
                frame_y_pred += frame_y_.cpu().numpy().tolist()

            if opt.num_classes == 1:
                frame_y_pred_pro = frame_y_pred
                frame_y_pred = torch.tensor(frame_y_pred)
                frame_y_pred = [0 if i < 0.5 else 1 for i in frame_y_pred]
                test_frame_auc = roc_auc_score(frame_y_gd, frame_y_pred_pro)
            else:
                test_frame_auc = roc_auc_score(frame_y_gd, frame_y_pred)

            test_frame_acc = accuracy_score(frame_y_gd, frame_y_pred)
            test_frame_recall = recall_score(frame_y_gd, frame_y_pred)
            test_frame_f1 = f1_score(frame_y_gd, frame_y_pred)
            test_frame_precision = precision_score(frame_y_gd, frame_y_pred)

        else:
            for X, y in tqdm(test_loader, desc='Validating'):
                # if model_type == 4:
                #     X = X.transpose(1, 2)
                X, y = X.to(device), y.to(device)
                y_ = model(X)

                y_ = y_.argmax(dim=1)
                y_gd += y.cpu().numpy().tolist()
                y_pred += y_.cpu().numpy().tolist()

            test_video_acc = accuracy_score(y_gd, y_pred)
            test_video_auc = roc_auc_score(y_gd, y_pred)
            test_video_recall = recall_score(y_gd, y_pred)
            test_video_f1 = f1_score(y_gd, y_pred)
            test_video_precision = precision_score(y_gd, y_pred)
            print('[Epoch %3d]Test video acc: %0.2f, auc: %0.2f, f1:%0.2f, pre:%0.2f, recall:%0.2f\n' % (
                epoch, test_video_acc, test_video_auc, test_video_f1, test_video_precision, test_video_recall))

        if opt.net_type != 1:
            writer.add_scalar(tag='test_video_acc', scalar_value=test_video_acc, global_step=epoch)
            writer.add_scalar(tag='test_video_auc', scalar_value=test_video_auc, global_step=epoch)
            writer.add_scalar(tag='test_video_recall', scalar_value=test_video_recall, global_step=epoch)
            writer.add_scalar(tag='test_video_f1', scalar_value=test_video_f1, global_step=epoch)
            writer.add_scalar(tag='test_video_precision', scalar_value=test_video_precision, global_step=epoch)
            print('[Epoch %3d]Test video avg loss: %0.4f, acc: %0.2f, auc: %0.2f\n' % (
                epoch, test_loss, test_video_acc, test_video_auc))
        elif opt.net_type != 0:
            writer.add_scalar(tag='test_frame_acc', scalar_value=test_frame_acc, global_step=epoch)
            writer.add_scalar(tag='test_frame_auc', scalar_value=test_frame_auc, global_step=epoch)
            writer.add_scalar(tag='test_frame_recall', scalar_value=test_frame_recall, global_step=epoch)
            writer.add_scalar(tag='test_frame_f1', scalar_value=test_frame_f1, global_step=epoch)
            writer.add_scalar(tag='test_frame_precision', scalar_value=test_frame_precision, global_step=epoch)
            print('[Epoch %3d]Test frame avg loss: %0.4f, acc: %0.2f, auc: %0.2f\n' % (
                epoch, test_loss, test_frame_acc, test_frame_auc))

        if opt.net_type != 1:
            return test_loss, test_video_acc, test_video_auc
        elif opt.net_type != 0:
            return test_loss, test_frame_acc, test_frame_auc


def parse_args():
    parser = argparse.ArgumentParser(usage='python3 main.py to train or test different models with different loss')
    parser.add_argument('-i', '--data_path', help='path to your datasets', default='～')
    parser.add_argument('-r', '--restore_from', help='path to the checkpoint', default=None)
    parser.add_argument('-g', '--gpu', help='visible gpu ids', default='0')
    parser.add_argument('-fs', '--frame_size', help='size of video frames', default=300)
    parser.add_argument('-is', '--img_size', help='size of input image', default=64)
    parser.add_argument('-nt', '--net_type', help='type of net, 0 for video level, 1 for frame level, 2 for dual level',
                        default=0)
    parser.add_argument('-e', '--epoch', help='batch size', default=20)
    parser.add_argument('-b', '--batch_size', help='batch size', default=16)
    parser.add_argument('-l', '--learning_rate', help='learning rate', default=1e-4)
    parser.add_argument('-nc', '--num_classes', help='number of classes', default=1)
    parser.add_argument('-lt', '--loss_type', help='loss type, 0 for CE, 1 for AUC, 2 for focal loss', default=1)
    parser.add_argument('-mt', '--model_type', help='model type, specific model names check the code, 0 for ours',
                        default=0)
    parser.add_argument('--use_gru', help='number of parameter gamma', default=True)
    parser.add_argument('--gamma', help='number of parameter gamma', default=0.15)
    parser.add_argument('--alpha', help='number of parameter alpha', default=0.5)
    parser.add_argument('--beta', help='number of parameter beta', default=0.6)
    parser.add_argument('--log_interval', help='log interval', default=2)
    parser.add_argument('--save_interval', help='save interval', default=1)
    parser.add_argument('--mode', help='train or test mode, True for train mode', default=True)
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_args()
    data_path = opt.data_path
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    dataloaders = {}
    for name in ['train', 'test']:
        raw_data = pandas.read_csv(os.path.join(data_path, '%s.csv' % name))
        if opt.net_type == 1 and opt.mode:
            dataloaders[name] = DataLoader(FrameDataset(raw_data.to_numpy()),
                                           batch_size=opt.batch_size,
                                           shuffle=True,
                                           num_workers=4,
                                           pin_memory=False)
        elif opt.net_type != 1:
            dataloaders[name] = DataLoader(Dataset(data_list=raw_data.to_numpy(), frame_num=opt.size),
                                           batch_size=opt.batch_size,
                                           shuffle=True,
                                           num_workers=4,
                                           pin_memory=False)

    train_on_epochs(dataloaders['train'], dataloaders['test'], opt)
