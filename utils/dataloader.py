import torch
from torch.utils import data
from torchvision import transforms

from tqdm import tqdm
import dlib
import cv2
import numpy as np
from PIL import Image

import config
import os
import random
import datasets.cvtransforms as cvtransforms
from datasets.dataset_imagenet_dct import opencv_loader
from datasets import train_upscaled_static_mean, train_upscaled_static_std


class AddSaltPepperNoise(object):

    def __init__(self, density=0.0):
        self.density = density

    def __call__(self, img):
        img = np.array(img)  # 图片转numpy
        img = np.transpose(img, (1, 2, 0))
        h, w, c = img.shape
        Nd = self.density
        Sd = 1 - Nd
        mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[Nd / 2.0, Nd / 2.0, Sd])  # 生成一个通道的mask
        mask = np.repeat(mask, c, axis=2)  # 在通道的维度复制，生成彩色的mask
        img[mask == 0] = 0  # 椒
        img[mask == 1] = 255  # 盐
        img = np.transpose(img, (2, 0, 1))
        img = Image.fromarray(np.uint8(img)).convert('RGB')  # numpy转图片
        return img


class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):
        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, img):
        img = np.array(img)
        img = np.transpose(img, (1, 2, 0))
        h, w, c = img.shape
        N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
        N = np.repeat(N, c, axis=2)
        img = N + img
        img[img > 255] = 255  # 避免有值超过255而反转
        img = Image.fromarray(np.uint8(img)).convert('RGB')
        return img


class FrameDataset(data.Dataset):
    def __init__(self, data_list=[], skip_frame=1, frame_num=300):

        # 用来将类别转换为one-hot数据
        self.labels = []
        # 用来缓存图片数据，直接加载到内存中
        self.images = []

        self.video_list = []
        # 是否直接加载至内存中，可以加快训练速
        self.use_mem = False

        self.skip_frame = skip_frame
        self.frame_num = frame_num
        self.data_list = self._build_data_list(data_list)

    def __len__(self):
        # return len(self.data_list) // self.time_step
        return len(self.data_list)

    def __getitem__(self, index):
        # 每次读取time_step帧图片
        # index = index * self.time_step
        # img = []
        # video_name = self.video_list[index]
        # for it in self.data_list:
        #     if it[1] == video_name:
        #         imgs.append(it)
        img = self.data_list[index]

        # 图片读取来源，如果设置了内存加速，则从内存中读取
        if self.use_mem:
            X = self.images[img[3]]
        else:
            X = self._read_img_and_transform(img[2])

        # 转换成tensor
        # X = torch.stack(X, dim=0)

        # 为这些图片指定类别标签
        y = torch.tensor(self._label_category(img[0]))
        return X, y

    def transform(self, img):
        return transforms.Compose([
            transforms.Resize((config.img_w, config.img_h)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])(img)

    def _read_img_and_transform(self, img: str):
        return self.transform(Image.open(img).convert('RGB'))

    def _build_data_list(self, data_list=[]):
        '''
        构建数据集
        '''
        if len(data_list) == 0:
            return []

        data_group = {}
        for x in tqdm(data_list, desc='Building dataset'):
            # 将视频分别按照classname和videoname分组
            [classname, videoname] = x[0:2]
            self.video_list.append(videoname)
            if classname not in data_group:
                data_group[classname] = {}
            if videoname not in data_group[classname]:
                data_group[classname][videoname] = []

            # 将图片数据加载到内存
            if self.use_mem:
                self.images.append(self._read_img_and_transform(x[2]))
            # path = x[2].replace('/home/puwenbo', '/Users/pu/Desktop')
            # if os.path.exists(path):
            if os.path.exists(x[2]):
                data_group[classname][videoname].append(list(x) + [len(self.images) - 1])

        self.video_list = list(set(self.video_list))
        # 处理类别变量
        self.labels = list(data_group.keys())

        ret_list = []
        n = 0

        # 填充数据
        for classname in data_group:
            video_group = data_group[classname]
            for videoname in video_group:
                video_len = len(video_group[videoname])
                if video_len < self.frame_num:
                    video_group[videoname] += [video_group[videoname][-1]] * (self.frame_num - video_len)
                    ret_list += video_group[videoname]
                else:
                    ret_list += video_group[videoname][:self.frame_num]
                n += len(video_group[videoname])

        random.shuffle(ret_list)
        return ret_list

    def _label_one_hot(self, label):
        '''
        将标签转换成one-hot形式
        '''
        if label not in self.labels:
            raise RuntimeError('不存在的label！')
        one_hot = [0] * len(self.labels)
        one_hot[self.labels.index(label)] = 1
        return one_hot

    def _label_category(self, label):
        '''
        将标签转换成整型
        '''
        if label not in self.labels:
            raise RuntimeError('不存在的label！')
        c_label = self.labels.index(label)
        return c_label


class Dataset(data.Dataset):
    def __init__(self, data_list=[], skip_frame=1, frame_num=300,
                 aug=False, add_channel=False, dct=False, salt=False):

        # 用来将类别转换为one-hot数据
        self.labels = []
        # 用来缓存图片数据，直接加载到内存中
        self.images = []

        self.video_list = []
        # 是否直接加载至内存中，可以加快训练速
        self.use_mem = False

        self.skip_frame = skip_frame
        self.frame_num = frame_num
        self.data_list = self._build_data_list(data_list)
        self.detector = dlib.get_frontal_face_detector()
        self.aug = aug
        self.add_channel = add_channel
        self.dct = dct
        self.salt = salt
        self.time_step = 30

    def __len__(self):
        # return len(self.data_list) // self.time_step
        return len(self.video_list)

    def __getitem__(self, index):
        # 每次读取time_step帧图片
        # index = index * self.time_step
        imgs = []
        video_name = self.video_list[index]
        for it in self.data_list:
            if it[1] == video_name:
                imgs.append(it)
        # imgs = self.data_list[index:index + self.time_step]

        # 图片读取来源，如果设置了内存加速，则从内存中读取
        if self.use_mem:
            X = [self.images[x[3]] for x in imgs]
        else:
            X = [self._read_img_and_transform(x[2]) for x in imgs]

        # 转换成tensor
        X = torch.stack(X, dim=0)

        # 为这些图片指定类别标签
        y = torch.tensor(self._label_category(imgs[0][0]))
        return X, y

    def get_landmarks(self, image):
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        rects = self.detector(img_gray, 0)
        if len(rects) != 0:
            for i in range(len(rects)):
                landmarks = np.matrix([[p.x, p.y] for p in self.predictor(image, rects[i]).parts()])
            return landmarks
        else:
            return False

    def img_add_channel(self, img, mask_b):
        img = img.numpy()
        img_new = np.concatenate((img, mask_b))
        img_new = torch.from_numpy(img_new)
        return torch.tensor(img_new, dtype=torch.float32)

    def get_face_xray(self, img):
        landmarks = self.get_landmarks(img)
        if type(landmarks) is not bool:
            mask = self.get_image_hull_mask(np.shape(img), landmarks).astype(np.uint8)
            g_mask = cv2.GaussianBlur(mask.astype(np.uint8) * 255, (5, 5), sigmaX=0) / 255
            mask_b = 4 * g_mask * (1 - g_mask)
            mask_b = cv2.resize(mask_b, (config.img_w, config.img_w))
            mask_b = mask_b[np.newaxis, :, :]
        else:
            mask_b = np.zeros((1, config.img_w, config.img_w))
        return mask_b

    def img_aug(self, img):
        landmarks = self.get_landmarks(img)
        g_img = cv2.GaussianBlur(img, (5, 5), sigmaX=0)
        if type(landmarks) is not bool:
            mask = self.get_image_hull_mask(np.shape(img), landmarks).astype(np.uint8)
            g_mask = cv2.GaussianBlur(mask.astype(np.uint8) * 255, (5, 5), sigmaX=0) / 255
            mask_b = 4 * g_mask * (1 - g_mask)
            img = np.where(mask_b[:, :, np.newaxis] > 0, g_img, img)
        else:
            img = g_img
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img

    def img_aug_full(self, img):
        landmarks = self.get_landmarks(img)
        g_img = cv2.GaussianBlur(img, (5, 5), sigmaX=0)
        if type(landmarks) is not bool:
            mask = self.get_image_hull_mask(np.shape(img), landmarks).astype(np.uint8)
            img = np.where(mask > 0, g_img, img)
        else:
            img = g_img
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img

    def cvt_transform(self, img):
        return cvtransforms.Compose([
            cvtransforms.RandomResizedCrop(config.img_w),
            # cvtransforms.RandomHorizontalFlip(),
            cvtransforms.Upscale(upscale_factor=2),
            cvtransforms.TransformUpscaledDCT(),
            cvtransforms.ToTensorDCT(),
            cvtransforms.SubsetDCT(channels=192),
            cvtransforms.Aggregate(),
            cvtransforms.NormalizeDCT(
                train_upscaled_static_mean,
                train_upscaled_static_std,
                channels=192
            )
        ])(img)

    def transform(self, img):
        if not self.salt:
            return transforms.Compose([
                transforms.Resize((config.img_w, config.img_h)),
                # AddSaltPepperNoise(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])(img)
        else:
            return transforms.Compose([
                transforms.Resize((config.img_w, config.img_h)),
                AddSaltPepperNoise(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])(img)

    def get_image_hull_mask(self, image_shape, image_landmarks, ie_polys=None):
        # get the mask of the image
        if image_landmarks.shape[0] != 68:
            raise Exception(
                'get_image_hull_mask works only with 68 landmarks')
        int_lmrks = np.array(image_landmarks, dtype=np.int)

        # hull_mask = np.zeros(image_shape[0:2]+(1,), dtype=np.float32)
        hull_mask = np.full(image_shape[0:2] + (1,), 0, dtype=np.float32)

        cv2.fillConvexPoly(hull_mask, cv2.convexHull(
            np.concatenate((int_lmrks[0:9],
                            int_lmrks[17:18]))), (1,))

        cv2.fillConvexPoly(hull_mask, cv2.convexHull(
            np.concatenate((int_lmrks[8:17],
                            int_lmrks[26:27]))), (1,))

        cv2.fillConvexPoly(hull_mask, cv2.convexHull(
            np.concatenate((int_lmrks[17:20],
                            int_lmrks[8:9]))), (1,))

        cv2.fillConvexPoly(hull_mask, cv2.convexHull(
            np.concatenate((int_lmrks[24:27],
                            int_lmrks[8:9]))), (1,))

        cv2.fillConvexPoly(hull_mask, cv2.convexHull(
            np.concatenate((int_lmrks[19:25],
                            int_lmrks[8:9],
                            ))), (1,))

        cv2.fillConvexPoly(hull_mask, cv2.convexHull(
            np.concatenate((int_lmrks[17:22],
                            int_lmrks[27:28],
                            int_lmrks[31:36],
                            int_lmrks[8:9]
                            ))), (1,))

        cv2.fillConvexPoly(hull_mask, cv2.convexHull(
            np.concatenate((int_lmrks[22:27],
                            int_lmrks[27:28],
                            int_lmrks[31:36],
                            int_lmrks[8:9]
                            ))), (1,))

        # nose
        cv2.fillConvexPoly(
            hull_mask, cv2.convexHull(int_lmrks[27:36]), (1,))

        if ie_polys is not None:
            ie_polys.overlay_mask(hull_mask)
        return hull_mask

    def _read_img_and_transform(self, img: str):
        if not self.dct:
            img = Image.open(img).convert('RGB')
            if self.aug:
                img = self.img_aug(np.array(img))
                return self.transform(img)
            if self.add_channel:
                xray = self.get_face_xray(np.array(img))
                img = self.transform(img)
                img = self.img_add_channel(img, xray)
                return img
            else:
                return self.transform(img)
        else:
            img = opencv_loader(img)
            img, _, _ = self.cvt_transform(img)
            return img

    def _build_data_list(self, data_list=[]):
        """
        构建数据集
        """
        if len(data_list) == 0:
            return []

        data_group = {}
        for x in tqdm(data_list, desc='Building dataset'):
            # 将视频分别按照classname和videoname分组
            [classname, videoname] = x[0:2]
            self.video_list.append(videoname)
            if classname not in data_group:
                data_group[classname] = {}
            if videoname not in data_group[classname]:
                data_group[classname][videoname] = []

            # 将图片数据加载到内存
            if self.use_mem:
                self.images.append(self._read_img_and_transform(x[2]))
            if os.path.exists(x[2]):
                data_group[classname][videoname].append(list(x) + [len(self.images) - 1])

        self.video_list = list(set(self.video_list))
        # 处理类别变量
        self.labels = list(data_group.keys())

        ret_list = []
        n = 0

        # 填充数据
        for classname in data_group:
            video_group = data_group[classname]
            for videoname in video_group:
                video_len = len(video_group[videoname])
                if video_len < self.frame_num:
                    video_group[videoname] += [video_group[videoname][-1]] * (self.frame_num - video_len)
                    ret_list += video_group[videoname]
                else:
                    ret_list += video_group[videoname][0:self.frame_num:self.time_step]
                n += len(video_group[videoname])

        return ret_list

    def _label_one_hot(self, label):
        """
        将标签转换成one-hot形式
        """
        if label not in self.labels:
            raise RuntimeError('不存在的label！')
        one_hot = [0] * len(self.labels)
        one_hot[self.labels.index(label)] = 1
        return one_hot

    def _label_category(self, label):
        """
        将标签转换成整型
        """
        if label not in self.labels:
            raise RuntimeError('不存在的label！')
        c_label = self.labels.index(label)
        return c_label
