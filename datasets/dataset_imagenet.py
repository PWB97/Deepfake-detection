from .vision import VisionDataset
from jpeg2dct.numpy import load, loads
from torchvision.datasets.imagenet import ImageFolder
from PIL import Image

import os
import os.path
import sys
import numpy as np
import torch
from turbojpeg import TurboJPEG

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(dir, class_to_idx, extensions=None, is_valid_file=None):
    images = []
    dir = os.path.expanduser(dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class DatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid_file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader, extensions=None, transform=None, target_transform=None, is_valid_file=None):
        super(DatasetFolder, self).__init__(root)
        self.transform = transform
        self.target_transform = target_transform
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                                                                 "Supported extensions are: " + ",".join(extensions)))

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)


class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid_file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples

class DatasetFolderDCT(VisionDataset):
    def __init__(self, root, loader, extensions=None, transform=None, target_transform=None, is_valid_file=None, subset=0):
        super(DatasetFolderDCT, self).__init__(root)
        self.transform = transform
        self.target_transform = target_transform
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                                                                 "Supported extensions are: " + ",".join(extensions)))

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        # self.jpeg = TurboJPEG('/home/kai.x/work/local/lib/libturbojpeg.so')
        self.jpeg = TurboJPEG('/usr/lib/libturbojpeg.so')
        self.subset = list(map(int, subset.split(','))) if subset else []

    def _find_classes(self, dir):
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)

        # with open(path, 'rb') as src:
        #     buffer = src.read()
        # dct_y_bak, dct_cb_bak, dct_cr_bak = loads(buffer)

        if self.transform is not None:
            sample = self.transform(sample)

        # sample_resize = sample.resize((224*2, 224*2), resample=0)
        # PIL to numpy
        sample = np.asarray(sample)
        # RGB to BGR
        sample = sample[:, :, ::-1]
        # JPEG Encode
        sample = np.ascontiguousarray(sample, dtype="uint8")
        sample = self.jpeg.encode(sample, quality=100, jpeg_subsample=2)
        dct_y, dct_cb, dct_cr = loads(sample)   # 28

        # sample_resize = np.asarray(sample_resize)
        # sample_resize = sample_resize[:, :, ::-1]
        # sample_resize = np.ascontiguousarray(sample_resize, dtype="uint8")
        # sample_resize = self.jpeg.encode(sample_resize, quality=100)
        # _, dct_cb_resize, dct_cr_resize = loads(sample_resize)   # 28
        # dct_cb_resize, dct_cr_resize = torch.from_numpy(dct_cb_resize).permute(2, 0, 1).float(), \
        #                  torch.from_numpy(dct_cr_resize).permute(2, 0, 1).float()

        # dct_y_unnormalized, dct_cb_unnormalized, dct_cr_unnormalized = loads(sample, normalized=False)   # 28
        # dct_y_normalized, dct_cb_normalized, dct_cr_normalized = loads(sample, normalized=True)   # 28
        # total_y = (dct_y-dct_y_bak).sum()
        # total_cb = (dct_cb-dct_cb_bak).sum()
        # total_cr = (dct_cr-dct_cr_bak).sum()
        # print('{}, {}, {}'.format(total_y, total_cb, total_cr))
        dct_y, dct_cb, dct_cr = torch.from_numpy(dct_y).permute(2, 0, 1).float(), \
                                torch.from_numpy(dct_cb).permute(2, 0, 1).float(), \
                                torch.from_numpy(dct_cr).permute(2, 0, 1).float()

        # transform = transforms.Resize(28, interpolation=2)
        # dct_cb_resize2 = [transform(Image.fromarray(dct_c.numpy())) for dct_c in dct_cb]

        if self.subset:
            dct_y, dct_cb, dct_cr = dct_y[self.subset[0]:self.subset[1]], dct_cb[self.subset[0]:self.subset[1]], \
                                    dct_cr[self.subset[0]:self.subset[1]]

        if self.target_transform is not None:
            dct_y = self.target_transform[0](dct_y)
            dct_cb = self.target_transform[1](dct_cb)
            dct_cr = self.target_transform[2](dct_cr)

        return dct_y, dct_cb, dct_cr, target

    def __len__(self):
        return len(self.samples)

class ImageFolderDCT(DatasetFolderDCT):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None, subset=None):
        super(ImageFolderDCT, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file, subset=subset)
        self.imgs = self.samples

class TinyDatasetFolder(VisionDataset):
    def __init__(self, root, loader, extensions=None, transform=None, target_transform=None, is_valid_file=None, quality=None):
        super(TinyDatasetFolder, self).__init__(root)
        self.transform = transform
        self.target_transform = target_transform
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                                                                 "Supported extensions are: " + ",".join(extensions)))
        self.jpeg = TurboJPEG('/home/kai.x/work/local/lib/libturbojpeg.so')

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.quality = quality

    def _find_classes(self, dir):
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)

        # RGB -> BGR
        img = np.asarray(sample)
        img = img[:, :, ::-1]
        # Convert to uint8, this is critical
        img = np.ascontiguousarray(img, dtype="uint8")

        encoded_img = self.jpeg.encode(img, quality=self.quality)
        decoded_img = self.jpeg.decode(encoded_img) # BGR

        # BGR -> RGB
        sample = decoded_img[:, :, ::-1]
        sample = Image.fromarray(sample)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)


class TinyImageFolder(TinyDatasetFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None, quality=None):
        super(TinyImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file, quality=quality)
        self.imgs = self.samples

if __name__ == '__main__':
    dataset = 'imagenet'

    import torch
    import torchvision.transforms as transforms
    from utils import val_y_mean, val_y_std, val_cb_mean, val_cb_std, val_cr_mean, val_cr_std
    from utils import train_y_mean, train_y_std, train_cb_mean, train_cb_std, train_cr_mean, train_cr_std

    if dataset == 'imagenet':
        val_normalize = []
        val_normalize_y = transforms.Normalize(mean=val_y_mean,
                                               std=val_y_std)
        val_normalize_cb = transforms.Normalize(mean=val_cb_mean,
                                                std=val_cb_std)
        val_normalize_cr = transforms.Normalize(mean=val_cr_mean,
                                                std=val_cr_std)
        val_normalize.append(val_normalize_y)
        val_normalize.append(val_normalize_cb)
        val_normalize.append(val_normalize_cr)
        val_loader = torch.utils.data.DataLoader(
            ImageFolderDCT('/mnt/ssd/kai.x/dataset/ILSVRC2012/val', transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
            ]), val_normalize),
            batch_size=64, shuffle=False,
            num_workers=1, pin_memory=True)

        train_normalize = []
        train_normalize_y = transforms.Normalize(mean=train_y_mean,
                                                 std=train_y_std)
        train_normalize_cb = transforms.Normalize(mean=train_cb_mean,
                                                  std=train_cb_std)
        train_normalize_cr = transforms.Normalize(mean=train_cr_mean,
                                                  std=train_cr_std)
        train_normalize.append(train_normalize_y)
        train_normalize.append(train_normalize_cb)
        train_normalize.append(train_normalize_cr)

        train_dataset = ImageFolderDCT('/mnt/ssd/kai.x/dataset/ILSVRC2012/train', transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ]), train_normalize)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=64, shuffle=True,
            num_workers=1, pin_memory=True)

    elif dataset == 'tiny-imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        val_loader = torch.utils.data.DataLoader(
            ImageFolderDCT('/mnt/ssd/kai.x/tiny-imagenet-200/val', transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                # normalize,
            ])),
            batch_size=1, shuffle=False,
            num_workers=1, pin_memory=True)

    from torchvision.utils import save_image
    dct_y_mean_total, dct_y_std_total = [], []
    # for batch_idx, (dct_y, dct_cb, dct_cr, targets) in enumerate(val_loader):
    for batch_idx, (dct_y, dct_cb, dct_cr, targets) in enumerate(val_loader):
        coef = dct_y.numpy()
        dct_y_mean, dct_y_std = [], []

        for c in coef:
            c = c.reshape((64, -1))
            dct_y_mean.append([np.mean(x) for x in c])
            dct_y_std.append([np.std(x) for x in c])

        dct_y_mean_np = np.asarray(dct_y_mean).mean(axis=0)
        dct_y_std_np = np.asarray(dct_y_std).mean(axis=0)
        dct_y_mean_total.append(dct_y_mean_np)
        dct_y_std_total.append(dct_y_std_np)
        print('The mean of dct_y is: {}'.format(dct_y_mean_np))
        print('The std of dct_y is: {}'.format(dct_y_std_np))

    print('The mean of dct_y is: {}'.format(np.asarray(dct_y_mean_total).mean(axis=0)))
    print('The std of dct_y is: {}'.format(np.asarray(dct_y_std_total).mean(axis=0)))


