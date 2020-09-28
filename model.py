import torch
from torch import nn
from torch.nn import functional as F

from torchvision import models as Models

from convlstm import ConvLSTM
from convGRU import ConvGRU
import resnet


class Baseline(nn.Module):

    def __init__(self, use_gru=False, bi_branch=False, rnn_hidden_layers=3, rnn_hidden_nodes=256,
                 num_classes=1, bidirectional=False):

        super(Baseline, self).__init__()

        self.rnn_hidden_layers = rnn_hidden_layers
        self.rnn_hidden_nodes = rnn_hidden_nodes
        self.num_classes = num_classes
        self.bi_branch = bi_branch

        pretrained_cnn = Models.resnet50(pretrained=True)
        cnn_layers = list(pretrained_cnn.children())[:-1]

        self.cnn = nn.Sequential(*cnn_layers)

        rnn_params = {
            'input_size': pretrained_cnn.fc.in_features,
            'hidden_size': self.rnn_hidden_nodes,
            'num_layers': self.rnn_hidden_layers,
            'batch_first': True,
            'bidirectional': bidirectional
        }

        if bidirectional:
            fc_in = 2*rnn_hidden_nodes
        else:
            fc_in = rnn_hidden_nodes

        self.rnn = (nn.GRU if use_gru else nn.LSTM)(**rnn_params)

        self.fc_cnn = nn.Linear(fc_in, num_classes)

        self.global_pool = nn.AdaptiveAvgPool2d(16)

        self.fc_rnn = nn.Linear(256, self.num_classes)

    def forward(self, x_3d):

        cnn_embedding_out = []
        cnn_pred = []

        for t in range(x_3d.size(1)):
            x = self.cnn(x_3d[:, t, :, :, :])
            x = torch.flatten(x, start_dim=1)
            cnn_embedding_out.append(x)

        cnn_embedding_out = torch.stack(cnn_embedding_out, dim=0).transpose(0, 1)

        self.rnn.flatten_parameters()
        rnn_out, _ = self.rnn(cnn_embedding_out, None)

        if self.bi_branch:
            for t in range(rnn_out.size(1)):
                x = rnn_out[:, t, :]
                x = self.fc_cnn(x)
                cnn_pred.append(x)
            cnn_pred = torch.stack(cnn_pred, dim=0).transpose(0, 1)

        x = self.global_pool(rnn_out)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_rnn(x)

        if self.bi_branch:
            return x, cnn_pred.reshape(-1, 1)
        else:
            return x


class CNN(nn.Module):
    def __init__(self, bi_branch=False, num_classes=2):
        super(CNN, self).__init__()

        self.num_classes = num_classes

        # 使用resnet预训练模型来提取特征，去掉最后一层分类器
        pretrained_cnn = Models.resnet50(pretrained=True)
        cnn_layers = list(pretrained_cnn.children())[:-1]

        # 把resnet的最后一层fc层去掉，用来提取特征
        self.cnn = nn.Sequential(*cnn_layers)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.cnn_out = nn.Sequential(
            nn.Linear(2048, 2)
        )

        # self.max_pool = nn.AdaptiveAvgPool2d(16)

        # self.fc_out = nn.Sequential(
        #     nn.Linear(256, self.num_classes)
        # )

    def forward(self, x_3d):
        """
        输入的是T帧图像，shape = (batch_size, t, h, w, 3)
        """
        cnn_embedding_out = []
        for t in range(x_3d.size(1)):
            # 使用cnn提取特征
            x = self.cnn(x_3d[:, t, :, :, :])
            x = torch.flatten(x, start_dim=1)
            x = self.cnn_out(x)
            cnn_embedding_out.append(x)
        cnn_embedding_out = torch.stack(cnn_embedding_out, dim=0).transpose(0, 1)

        x = self.global_pool(cnn_embedding_out)
        # x = self.max_pool(cnn_embedding_out)
        x = torch.flatten(x, start_dim=1)
        # x = self.fc_out(x)

        return x


class cRNN(nn.Module):
    def __init__(self, use_gru=False, bi_branch=False, num_classes=2):
        super(cRNN, self).__init__()

        self.num_classes = num_classes
        self.use_gru = use_gru

        # 使用resnet预训练模型来提取特征，去掉最后一层分类器
        pretrained_cnn = Models.resnet50(pretrained=True)
        cnn_layers = list(pretrained_cnn.children())[:-2]

        # 把resnet的最后一层fc层去掉，用来提取特征
        self.cnn = nn.Sequential(*cnn_layers)

        cRNN_params = {
            'input_dim': 2048,
            'hidden_dim': [256, 256, 512],
            'kernel_size': (1, 1),
            'num_layers': 3,
            'batch_first': True
        } if not use_gru else {
            'input_size': (2, 2),
            'input_dim': 2048,
            'hidden_dim': [256, 256, 512],
            'kernel_size': (1, 1),
            'num_layers': 3,
            'batch_first': True
        }

        self.cRNN = (ConvGRU if use_gru else ConvLSTM)(**cRNN_params)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Linear(512, self.num_classes)
        )

    def forward(self, x_3d):
        cnn_embedding_out = []
        for t in range(x_3d.size(1)):
            # 使用cnn提取特征
            x = self.cnn(x_3d[:, t, :, :, :])
            cnn_embedding_out.append(x)

        x = torch.stack(cnn_embedding_out, dim=0).transpose(0, 1)

        _, outputs = self.cRNN(x)
        x = outputs[0][0] if self.use_gru else outputs[0][1]

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def get_resnet_3d(num_classes=2, model_depth=10, shortcut_type='B', sample_size=112, sample_duration=16):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = resnet.resnet10(
            num_classes=num_classes,
            shortcut_type=shortcut_type,
            sample_size=sample_size,
            sample_duration=sample_duration)
    elif model_depth == 18:
        model = resnet.resnet18(
            num_classes=num_classes,
            shortcut_type=shortcut_type,
            sample_size=sample_size,
            sample_duration=sample_duration)
    elif model_depth == 34:
        model = resnet.resnet34(
            num_classes=num_classes,
            shortcut_type=shortcut_type,
            sample_size=sample_size,
            sample_duration=sample_duration)
    elif model_depth == 50:
        model = resnet.resnet50(
            num_classes=num_classes,
            shortcut_type=shortcut_type,
            sample_size=sample_size,
            sample_duration=sample_duration)
    elif model_depth == 101:
        model = resnet.resnet101(
            num_classes=num_classes,
            shortcut_type=shortcut_type,
            sample_size=sample_size,
            sample_duration=sample_duration)
    elif model_depth == 152:
        model = resnet.resnet152(
            num_classes=num_classes,
            shortcut_type=shortcut_type,
            sample_size=sample_size,
            sample_duration=sample_duration)
    else:
        model = resnet.resnet200(
            num_classes=num_classes,
            shortcut_type=shortcut_type,
            sample_size=sample_size,
            sample_duration=sample_duration)

    return model
