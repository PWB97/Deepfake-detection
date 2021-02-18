img_w = 64
img_h = img_w

dataset_params = {
    'batch_size': 2,
    'shuffle': True,
    'num_workers': 4,
    'pin_memory': False
}

net_params = {
    'use_gru': True,
    'bi_branch': True,
    'num_classes': 1,
    'dct': False,
    'inputgate': False
}

resnet_3d_params = {
    'num_classes': 2,
    'model_depth': 50,
    'shortcut_type': 'B',
    'sample_size': img_h,
    'sample_duration': 30
}

models = {
    1: 'baseline',
    2: 'cRNN',
    3: 'end2end',
    4: 'xception',
    5: 'fwa',
    6: 'cnn',
    7: 'res50',
    8: 'res101',
    9: 'res152'
}

losses = {
    0: 'CE',
    1: 'AUC',
    2: 'focal'
}


gamma = 0.15

model_type = models.get(1)
loss_type = losses.get(1)
learning_rate = 1e-4
epoches = 20
log_interval = 2 # 打印间隔，默认每2个batch_size打印一次
save_interval = 1 # 模型保存间隔，默认每个epoch保存一次
