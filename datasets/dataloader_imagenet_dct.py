import os
import time
import torch
from datasets.dataset_imagenet_dct import ImageFolderDCT
import datasets.cvtransforms as transforms
from datasets import train_y_mean, train_y_std, train_cb_mean, train_cb_std, \
    train_cr_mean, train_cr_std
from datasets import train_y_mean_upscaled, train_y_std_upscaled, train_cb_mean_upscaled, train_cb_std_upscaled, \
    train_cr_mean_upscaled, train_cr_std_upscaled
from datasets import train_dct_subset_mean, train_dct_subset_std
from datasets import train_upscaled_static_mean, train_upscaled_static_std
from datasets import train_upscaled_static_dct_direct_mean, train_upscaled_static_dct_direct_std
from datasets import train_upscaled_static_dct_direct_mean_interp, train_upscaled_static_dct_direct_std_interp

def trainloader_upscaled_dct_direct(args, model='mobilenet'):
    if model == 'mobilenet':
        input_size = 112
    elif model == 'resnet':
        input_size = 56
    else:
        raise NotImplementedError

    traindir = os.path.join(args.data, 'train')
    transform = transforms.Compose([
        transforms.UpsampleCbCr(),
        transforms.SubsetDCT2(channels=args.subset, pattern=args.pattern),
        transforms.RandomResizedCropDCT(size=input_size),
        transforms.Aggregate2(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensorDCT2(),
        transforms.NormalizeDCT(
            train_upscaled_static_dct_direct_mean_interp,
            train_upscaled_static_dct_direct_std_interp,
            channels=args.subset,
            pattern=args.pattern
        )
    ])

    train_dataset = ImageFolderDCT(traindir, transform, backend='dct')

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    train_loader_len = len(train_loader)

    return train_loader, train_sampler, train_loader_len


def valloader_upscaled_dct_direct(args, model='mobilenet'):
    if model == 'mobilenet':
        input_size1 = 128
        input_size2 = 112
    elif model == 'resnet':
        input_size1 = 64
        input_size2 = 56
    else:
        raise NotImplementedError

    valdir = os.path.join(args.data, 'val')
    transform = transforms.Compose([
        transforms.UpsampleCbCr(),
        transforms.SubsetDCT2(channels=args.subset, pattern=args.pattern),
        transforms.Aggregate2(),
        transforms.Resize(input_size1),
        transforms.CenterCrop(input_size2),
        transforms.ToTensorDCT2(),
        transforms.NormalizeDCT(
            train_upscaled_static_dct_direct_mean_interp,
            train_upscaled_static_dct_direct_std_interp,
            channels=args.subset,
            pattern=args.pattern
        )
    ])
    val_loader = torch.utils.data.DataLoader(
        ImageFolderDCT(valdir, transform, backend='dct'),
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    return val_loader

# def trainloader_upscaled_dct_direct(args, model='mobilenet'):
#     if model == 'mobilenet':
#         input_size = 896
#     elif model == 'resnet':
#         input_size = 448
#     else:
#         raise NotImplementedError
#
#     traindir = os.path.join(args.data, 'train')
#     transform = transforms.Compose([
#         transforms.DCTFlatten2D(mux=0b011),
#         transforms.UpsampleCbCrDCT(),
#         transforms.RandomResizedCropDCT(size=input_size),
#         transforms.SubsetDCT2(channels=args.subset, pattern=args.pattern),
#         transforms.Aggregate2(),
#         transforms.CenterCrop(input_size//8),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensorDCT2(),
#         transforms.NormalizeDCT(
#             train_upscaled_static_dct_direct_mean,
#             train_upscaled_static_dct_direct_std,
#             channels=args.subset,
#             pattern=args.pattern
#         )
#     ])
#
#     train_dataset = ImageFolderDCT(traindir, transform, backend='dct')
#
#     if args.distributed:
#         train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
#     else:
#         train_sampler = None
#
#     train_loader = torch.utils.data.DataLoader(
#         train_dataset,
#         batch_size=args.train_batch, shuffle=(train_sampler is None),
#         num_workers=args.workers, pin_memory=True, sampler=train_sampler)
#
#     train_loader_len = len(train_loader)
#
#     return train_loader, train_sampler, train_loader_len
#
#
# def valloader_upscaled_dct_direct(args, model='mobilenet'):
#     if model == 'mobilenet':
#         input_size1 = 1024
#         input_size2 = 896
#     elif model == 'resnet':
#         input_size1 = 512
#         input_size2 = 448
#     else:
#         raise NotImplementedError
#
#     valdir = os.path.join(args.data, 'val')
#     transform = transforms.Compose([
#         transforms.DCTFlatten2D(mux=0b011),
#         transforms.UpsampleCbCrDCT(),
#         transforms.UpsampleDCT(T=input_size1, debug=False),
#         transforms.SubsetDCT2(channels=args.subset, pattern=args.pattern),
#         transforms.Aggregate2(),
#         transforms.CenterCrop(input_size2//8),
#         transforms.ToTensorDCT2(),
#         transforms.NormalizeDCT(
#             train_upscaled_static_dct_direct_mean,
#             train_upscaled_static_dct_direct_std,
#             channels=args.subset,
#             pattern=args.pattern
#         )
#     ])
#     val_loader = torch.utils.data.DataLoader(
#         ImageFolderDCT(valdir, transform, backend='dct'),
#         batch_size=args.test_batch, shuffle=False,
#         num_workers=args.workers, pin_memory=True
#     )
#
#     return val_loader


# Upscaling in the spatial domain
def trainloader_upscaled_static(args, model='mobilenet'):
    traindir = os.path.join(args.data, 'train')

    if model == 'mobilenet':
        input_size = 896
    elif model == 'resnet':
        input_size = 448
    else:
        raise NotImplementedError

    transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.Upscale(upscale_factor=2),
        transforms.TransformUpscaledDCT(),
        transforms.ToTensorDCT(),
        transforms.SubsetDCT(channels=args.subset, pattern=args.pattern),
        transforms.Aggregate(),
        transforms.NormalizeDCT(
            train_upscaled_static_mean,
            train_upscaled_static_std,
            channels=args.subset,
            pattern=args.pattern
        )
    ])

    train_dataset = ImageFolderDCT(traindir, transform)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    train_loader_len = len(train_loader)

    return train_loader, train_sampler, train_loader_len

def valloader_upscaled_static(args, model='mobilenet'):
    valdir = os.path.join(args.data, 'val')

    if model == 'mobilenet':
        input_size1 = 1024
        input_size2 = 896
    elif model == 'resnet':
        input_size1 = 512
        input_size2 = 448
    else:
        raise NotImplementedError

    transform = transforms.Compose([
            transforms.Resize(input_size1),
            transforms.CenterCrop(input_size2),
            transforms.Upscale(upscale_factor=2),
            transforms.TransformUpscaledDCT(),
            transforms.ToTensorDCT(),
            transforms.SubsetDCT(channels=args.subset, pattern=args.pattern),
            transforms.Aggregate(),
            transforms.NormalizeDCT(
                train_upscaled_static_mean,
                train_upscaled_static_std,
                channels=args.subset,
                pattern=args.pattern
            )
        ])

    val_loader = torch.utils.data.DataLoader(
        ImageFolderDCT(valdir, transform),
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    return val_loader

def trainloader_dct_resized(args):
    traindir = os.path.join(args.data, 'train')
    train_dataset = ImageFolderDCT(traindir, transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.TransformDCT(),  # 28x28x192
        transforms.DCTFlatten2D(),
        transforms.UpsampleDCT(upscale_ratio_h=4, upscale_ratio_w=4, debug=False),
        transforms.ToTensorDCT(),
        transforms.SubsetDCT(channels=args.subset),
        transforms.Aggregate(),
        transforms.NormalizeDCT(
            train_dct_subset_mean,
            train_dct_subset_std,
            channels=args.subset
        )
    ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    train_loader_len = len(train_loader)

    return train_loader, train_sampler, train_loader_len

def valloader_dct_resized(args):
    valdir = os.path.join(args.data, 'val')

    val_loader = torch.utils.data.DataLoader(
        ImageFolderDCT(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.TransformDCT(),  # 28x28x192
            transforms.DCTFlatten2D(),
            transforms.UpsampleDCT(upscale_ratio_h=4, upscale_ratio_w=4, debug=False),
            transforms.ToTensorDCT(),
            transforms.SubsetDCT(channels=args.subset),
            transforms.Aggregate(),
            transforms.NormalizeDCT(
                train_dct_subset_mean,
                train_dct_subset_std,
                channels=args.subset
            )
        ])),
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    return val_loader

def trainloader_dct_upscaled(args):
    traindir = os.path.join(args.data, 'train')
    train_dataset = ImageFolderDCT(traindir, transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Upscale(),
        transforms.TransformDCT(),
        transforms.ToTensorDCT(),
        transforms.NormalizeDCT(
            train_y_mean_upscaled, train_y_std_upscaled,
            train_cb_mean_upscaled, train_cb_std_upscaled,
            train_cr_mean_upscaled, train_cr_std_upscaled),
    ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    train_loader_len = len(train_loader)

    return train_loader, train_sampler, train_loader_len

def valloader_dct_upscaled(args):
    valdir = os.path.join(args.data, 'val')

    val_loader = torch.utils.data.DataLoader(
        ImageFolderDCT(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Upscale(),
            transforms.TransformDCT(),
            transforms.ToTensorDCT(),
            transforms.NormalizeDCT(
                train_y_mean_upscaled, train_y_std_upscaled,
                train_cb_mean_upscaled, train_cb_std_upscaled,
                train_cr_mean_upscaled, train_cr_std_upscaled),
        ])),
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    return val_loader

def trainloader_dct_upscaled_subset(args):
    traindir = os.path.join(args.data, 'train')
    train_dataset = ImageFolderDCT(traindir, transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Upscale(),
        transforms.TransformDCT(),
        transforms.ToTensorDCT(),
        transforms.SubsetDCT(args.subset),
        transforms.NormalizeDCT(
            train_y_mean_upscaled, train_y_std_upscaled,
            train_cb_mean_upscaled, train_cb_std_upscaled,
            train_cr_mean_upscaled, train_cr_std_upscaled),
    ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    train_loader_len = len(train_loader)

    return train_loader, train_sampler, train_loader_len

def valloader_dct_upscaled_subset(args):
    valdir = os.path.join(args.data, 'val')

    val_loader = torch.utils.data.DataLoader(
        ImageFolderDCT(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Upscale(),
            transforms.TransformDCT(),
            transforms.ToTensorDCT(),
            transforms.SubsetDCT(args.subset),
            transforms.NormalizeDCT(
                train_y_mean_upscaled, train_y_std_upscaled,
                train_cb_mean_upscaled, train_cb_std_upscaled,
                train_cr_mean_upscaled, train_cr_std_upscaled),
        ])),
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    return val_loader

def trainloader_dct(args):
    traindir = os.path.join(args.data, 'train')
    train_dataset = ImageFolderDCT(traindir, transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.TransformDCT(),
        transforms.ToTensorDCT(),
        transforms.NormalizeDCT(
            train_y_mean, train_y_std,
            train_cb_mean, train_cb_std,
            train_cr_mean, train_cr_std),
    ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    train_loader_len = len(train_loader)

    return train_loader, train_sampler, train_loader_len

def valloader_dct(args):
    valdir = os.path.join(args.data, 'val')

    val_loader = torch.utils.data.DataLoader(
        ImageFolderDCT(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.TransformDCT(),
            transforms.ToTensorDCT(),
            transforms.NormalizeDCT(
                train_y_mean, train_y_std,
                train_cb_mean, train_cb_std,
                train_cr_mean, train_cr_std),
        ])),
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    return val_loader


def trainloader_dct_subset(args):
    traindir = os.path.join(args.data, 'train')
    train_dataset = ImageFolderDCT(traindir, transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.TransformDCT(),
        transforms.ToTensorDCT(),
        transforms.SubsetDCT(args.subset_channels),
        transforms.NormalizeDCT(
            train_y_mean, train_y_std,
            train_cb_mean, train_cb_std,
            train_cr_mean, train_cr_std),
    ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    train_loader_len = len(train_loader)

    return train_loader, train_sampler, train_loader_len

def valloader_dct_subset(args):
    valdir = os.path.join(args.data, 'val')

    val_loader = torch.utils.data.DataLoader(
        ImageFolderDCT(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.TransformDCT(),
            transforms.ToTensorDCT(),
            transforms.SubsetDCT(args.subset_channels),
            transforms.NormalizeDCT(
                train_y_mean, train_y_std,
                train_cb_mean, train_cb_std,
                train_cr_mean, train_cr_std),
        ])),
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    return val_loader

if __name__ == '__main__':
    import numpy as np
    from turbojpeg import TurboJPEG
    from jpeg2dct.numpy import load, loads

    jpeg_encoder = TurboJPEG('/usr/lib/libturbojpeg.so')


    # transform1 =transforms.Compose([
    #     transforms.RandomResizedCrop(224),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.TransformDCT(),  # 28x28x192
    #     transforms.DCTFlatten2D(),
    #     transforms.UpsampleDCT(upscale_ratio_h=4, upscale_ratio_w=4, debug=True),
    #     transforms.ToTensorDCT(),
    # ])

    # transform2 = transforms.Compose([
    #     transforms.DCTFlatten2D(),
    #     transforms.UpsampleDCT(upscale_ratio_h=4, upscale_ratio_w=4, debug=True),
    #     transforms.ToTensorDCT(),
    #     transforms.NormalizeDCT(
    #         train_y_mean_resized, train_y_std_resized,
    #         train_cb_mean_resized, train_cb_std_resized,
    #         train_cr_mean_resized, train_cr_std_resized),
    # ])

    # transform3 =transforms.Compose([
    #     transforms.RandomResizedCrop(224),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ResizedTransformDCT(),
    #     transforms.ToTensorDCT(),
    #     transforms.SubsetDCT(32),
    # ])

    # transform4 = transforms.Compose([
    #     transforms.RandomResizedCrop(896),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.Upscale(upscale_factor=2),
    #     transforms.TransformUpscaledDCT(),
    #     transforms.ToTensorDCT(),
    #     transforms.SubsetDCT(channels=24),
    #     transforms.Aggregate(),
    #     transforms.NormalizeDCT(
    #         train_upscaled_static_mean,
    #         train_upscaled_static_std,
    #         channels=24
    #     )
    #     ])

    # transform5 = transforms.Compose([
    #     transforms.DCTFlatten2D(),
    #     transforms.UpsampleDCT(size_threshold=112 * 8, T=112 * 8, debug=False),
    #     transforms.SubsetDCT2(channels=32),
    #     transforms.Aggregate2(),
    #     transforms.RandomResizedCropDCT(112),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensorDCT2(),
    #     transforms.NormalizeDCT(
    #         train_upscaled_static_mean,
    #         train_upscaled_static_std,
    #         channels=32
    #     )
    # ])

    transform6 = transforms.Compose([
        transforms.DCTFlatten2D(mux=0b011),
        transforms.UpsampleCbCr(),
        transforms.UpsampleDCT(T=512, debug=False),
        transforms.SubsetDCT2(channels=64),
        transforms.Aggregate2(),
        transforms.CenterCrop(448 // 8),
        transforms.ToTensorDCT2(),
        transforms.NormalizeDCT(
            train_upscaled_static_dct_direct_mean,
            train_upscaled_static_dct_direct_std,
            channels=64
        )
    ])

    transform7 = transforms.Compose([
        transforms.UpsampleCbCr(),
        transforms.SubsetDCT2(channels=64),
        transforms.RandomResizedCropDCT(size=448),
        transforms.Aggregate2(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensorDCT2(),
        transforms.NormalizeDCT(
            train_upscaled_static_dct_direct_mean_interp,
            train_upscaled_static_dct_direct_std_interp,
            channels=64,
        )
    ])
    # train_dataset = ImageFolderDCT('/ILSVRC2012/train', transform1, backend='opencv')
    # train_dataset = ImageFolderDCT('/ILSVRC2012/train', transform2
    # , backend='dct')
    train_dataset = ImageFolderDCT('/ILSVRC2012/train', transform7, backend='dct')

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=16, shuffle=(train_sampler is None),
        num_workers=1, pin_memory=True, sampler=train_sampler)

    for i, data in enumerate(train_loader):
        print(data)


