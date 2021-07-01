import torch
import pandas
from PIL import Image

from utils.dataloader import Dataset
from make_train_test import *
from meso.meso import *


def load_datas(src_path, files=[]):
    datas = []
    for file in files:
        img = Image.open(os.path.join(src_path, file)).convert('RGB')
        img.save("./images/" + file)
        img = img.resize((64, 64), Image.ANTIALIAS)
        data = np.array(img)
        data = np.transpose(data, (2, 0, 1))
        datas.append(data)
    return np.array(datas)


def video_frame_face_extractor(path, output):
    import dlib
    face_detector = dlib.cnn_face_detection_model_v1('./mmod_human_face_detector.dat')
    video_fd = cv2.VideoCapture(path)
    if not video_fd.isOpened():
        print('Skpped: {}'.format(path))

    frame_index = 0
    success, frame = video_fd.read()
    while success:
        frame_path = os.path.join(output + '/frame/%s_%d.jpg' % (path.split('/')[-1], frame_index))
        cv2.imwrite(frame_path, frame)
        img_path = os.path.join(output + '/face/%s_%d.jpg' % (path.split('/')[-1], frame_index))
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 1)
        if len(faces):
            # For now only take biggest face
            face = faces[0].rect
            x, y, size = get_boundingbox(face, width, height)
            # generate cropped image
            cropped_face = frame[y:y + size, x:x + size]
            cv2.imwrite(img_path, cropped_face)

        frame_index += 1
        success, frame = video_fd.read()

    video_fd.release()


def list_file(path, label):
    list = []
    for file in os.listdir(path):
        list.append([path + '/' + file, label])

    return list


def dataset_size(path):
    for file in os.listdir(path + '/0/'):
        img = cv2.imread(path + '/0/' + file)
        print(np.array(img).shape)


def frame_range(src_dir):
    Celeb_real = list_file(src_dir + '/Celeb-real', 1)
    Celeb_synthesis = list_file(src_dir + '/Celeb-synthesis', 0)
    YouTube_real = list_file(src_dir + '/YouTube-real', 1)

    frame_m = []

    for [file, _] in Celeb_real:
        video = cv2.VideoCapture(os.path.join(src_dir, '/Celeb-real', file))
        frame_num = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_m.append(frame_num)
        print(file)
        print(frame_num)
        print('---------------')
    for [file, _] in Celeb_synthesis:
        video = cv2.VideoCapture(os.path.join(src_dir, '/Celeb-synthesis', file))
        frame_num = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_m.append(frame_num)
        print(file)
        print(frame_num)
        print('---------------')
    for [file, _] in YouTube_real:
        video = cv2.VideoCapture(os.path.join(src_dir, '/YouTube-real', file))
        frame_num = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_m.append(frame_num)
        print(file)
        print(frame_num)
        print('---------------')
    frame_m = np.array(frame_m)
    print('====================')
    print(np.max(np.array(frame_m)))
    print(np.min(np.array(frame_m)))
    print(frame_m.shape)
    print(np.median(frame_m))
    print(np.mean(frame_m))


def build_AUC_loss(outputs, labels, gamma, power_p):
    posi_idx = tf.where(tf.equal(labels, 1.0))
    neg_idx = tf.where(tf.equal(labels, -1.0))
    prdictions = tf.nn.softmax(outputs)
    posi_predict = tf.gather(prdictions, posi_idx)
    posi_size = tf.shape(posi_predict)[0]
    neg_predict = tf.gather(prdictions, neg_idx)
    neg_size = tf.shape(posi_predict)[0]
    posi_neg_diff = tf.reshape(
        -(tf.matmul(posi_predict, tf.ones([1, neg_size])) -
          tf.matmul(tf.ones([posi_size, 1]), tf.reshape(neg_predict, [-1, neg_size])) - gamma),
        [-1, 1])
    posi_neg_diff = tf.where(tf.greater(posi_neg_diff, 0), posi_neg_diff, tf.zeros([posi_size * neg_size, 1]))
    posi_neg_diff = tf.pow(posi_neg_diff, power_p)
    loss_approx_auc = tf.reduce_mean(posi_neg_diff)
    return loss_approx_auc


def auc_loss(y_pred, y_true, gamma, p=2):
    pos = tf.boolean_mask(y_pred, tf.cast(y_true, tf.bool))
    neg = tf.boolean_mask(y_pred, ~tf.cast(y_true, tf.bool))
    pos = tf.expand_dims(pos, 0)
    neg = tf.expand_dims(neg, 1)
    difference = tf.zeros_like(pos * neg) + pos - neg - gamma
    masked = tf.boolean_mask(difference, difference < 0.0)
    return tf.reduce_sum(tf.pow(-masked, p))


def AUC_loss(y_pred, y_true, device, gamma, p=2):
    pred = torch.sigmoid(y_pred)
    pos = pred[torch.where(y_true == 0)]
    neg = pred[torch.where(y_true == 1)]
    pos = torch.unsqueeze(pos, 0)
    neg = torch.unsqueeze(neg, 1)
    diff = torch.zeros_like(pos * neg, device=device) + pos - neg - gamma
    masked = diff[torch.where(diff < 0.0)]
    return torch.mean(torch.pow(-masked, p))


# def AUC_loss(outputs, labels, device, gamma, p=2):
#     predictions = torch.sigmoid(outputs)
#     pos_predict = predictions[torch.where(labels == 0)]
#     neg_predict = predictions[torch.where(labels == 1)]
#     pos_size = pos_predict.shape[0]
#     neg_size = neg_predict.shape[0]
#     # if pos_size == 0 or neg_size == 0:
#     #     return 0
#     # else:
#     if pos_size != 0 and neg_size != 0:
#         pos_neg_diff = -(torch.matmul(pos_predict, torch.ones([1, neg_size], device=device)) -
#                          torch.matmul(torch.ones([pos_size, 1], device=device),
#                                       torch.reshape(neg_predict, [-1, neg_size]))
#                          - gamma)
#         pos_neg_diff = torch.reshape(pos_neg_diff, [-1, 1])
#         pos_neg_diff = torch.where(torch.gt(pos_neg_diff, 0), pos_neg_diff, torch.zeros([pos_size * neg_size, 1],
#                                                                                         device=device))
#     elif neg_size == 0:
#         pos_neg_diff = -(pos_predict - gamma)
#         pos_neg_diff = torch.where(torch.gt(pos_neg_diff, 0), pos_neg_diff, torch.zeros([pos_size, 1], device=device))
#     else:
#         pos_neg_diff = -(-neg_predict - gamma)
#         pos_neg_diff = torch.where(torch.gt(pos_neg_diff, 0), pos_neg_diff, torch.zeros([neg_size, 1], device=device))
#
#     pos_neg_diff = torch.pow(pos_neg_diff, p)
#
#     loss_approx_auc = torch.mean(pos_neg_diff)
#     return loss_approx_auc


# def AUC_loss(y_, y, device, gamma, p=2):
#     X = y_[torch.where(y == 0)]
#     Y = y_[torch.where(y == 1)]
#     loss = torch.zeros(1, requires_grad=True, device=device)
#     if X.shape[0] == 0:
#         Y = torch.max(Y, 1)[0]
#         for j in Y:
#             if -j < gamma:
#                 loss = (-(- j - gamma)) ** p + loss
#     if Y.shape[0] == 0:
#         X = torch.max(X, 1)[0]
#         for i in X:
#             if i < gamma:
#                 loss = (-(i - gamma)) ** p + loss
#     if X.shape[0] != 0 and Y.shape[0] != 0:
#         X = torch.max(X, 1)[0]
#         Y = torch.max(Y, 1)[0]
#         for i in X:
#             for j in Y:
#                 if i - j < gamma:
#                     loss = (-(i - j - gamma)) ** p + loss
#     return loss


def merge_labels_to_ckpt(ck_path: str, train_file: str):
    """Merge labels to a checkpoint file.

    Args:
        ck_path(str): path to checkpoint file
        train_file(str): path to train set index file, eg. train.csv

    Return:
        This function will create a {ck_path}_patched.pth file.
    """
    # load model
    print('Loading checkpoint')
    ckpt = torch.load(ck_path)

    # load train files
    print('Loading dataset')
    raw_data = pandas.read_csv(train_file)
    train_set = Dataset(raw_data.to_numpy())

    # patch file name
    print('Patching')
    patch_path = ck_path.replace('.pth', '') + '_patched.pth'

    ck_dict = {'label_map': train_set.labels}
    names = ['epoch', 'model_state_dict', 'optimizer_state_dict']
    for name in names:
        ck_dict[name] = ckpt[name]

    torch.save(ck_dict, patch_path)
    print('Patched checkpoint has been saved to {}'.format(patch_path))


def tensor2im(input_image, imtype=np.uint8):
    """"将tensor的数据类型转成numpy类型，并反归一化.

    Parameters:
        input_image (tensor) --  输入的图像tensor数组
        imtype (type)        --  转换后的numpy的数据类型
    """
    mean = [0.485, 0.456, 0.406]  # 自己设置的
    std = [0.229, 0.224, 0.225]  # 自己设置的
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        for i in range(len(mean)):
            image_numpy[i] = image_numpy[i] * std[i] + mean[i]
        image_numpy = image_numpy * 255
        image_numpy = np.transpose(image_numpy, (1, 2, 0))  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def save_img(im, path):
    """im可是没经过任何处理的tensor类型的数据,将数据存储到path中

    Parameters:
        im (tensor) --  输入的图像tensor数组
        path (str)  --  图像保存的路径
        size (int)  --  一行有size张图,最好是2的倍数
    """
    # im_grid = torchvision.utils.make_grid(im, size) #将batchsize的图合成一张图
    im_numpy = tensor2im(im)  # 转成numpy类型并反归一化
    im_array = Image.fromarray(im_numpy)
    im_array.save(path)


def new_path(path):
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except Exception:
            os.makedirs(path)


def read_npy(src):
    arr = np.load(src)
    for i in arr:
        print(i)


def parse_args():
    parser = argparse.ArgumentParser(usage='python3 tools.py -i path/to/train.csv -r path/to/checkpoint')
    parser.add_argument('-i', '--data_path', help='path to your dataset index file')
    parser.add_argument('-r', '--restore_from', help='path to the checkpoint', default=None)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    # xcp = '/home/asus/Code/checkpoint/ff/xcept/nb-model_type-xception_ep-19.pth'
    # meso = ''
    # msi = '/home/asus/Code/checkpoint/ff/msin/weights.h5'
    # cap = '/home/asus/Code/checkpoint/ff/cap/capsule_18.pt'
    # model_pred('/home/asus/ffdf/test/1', 'xception', xcp)
    # model_pred('/home/asus/ffdf_40/test/1', 'cap', cap)
    # model_pred('/home/asus/ffdf/test/1', 'msi', msi)
    # model_pred('/home/asus/ffdf_40/test/0', 'msi', msi)
    read_npy('/Users/pu/Downloads/images/c23/0/tcap.txt.npy')
    read_npy('/Users/pu/Downloads/images/c23/0/tmsi.txt.npy')
    read_npy('/Users/pu/Downloads/images/c23/0/txcep.txt.npy')
    print('=================')
    read_npy('/Users/pu/Downloads/images/c23/1/tcap.txt.npy')
    read_npy('/Users/pu/Downloads/images/c23/1/tmsi.txt.npy')
    read_npy('/Users/pu/Downloads/images/c23/1/txcep.txt.npy')
    print('=================')
    read_npy('/Users/pu/Downloads/images/c40/0/tcap.txt.npy')
    read_npy('/Users/pu/Downloads/images/c40/0/tmsi.txt.npy')
    read_npy('/Users/pu/Downloads/images/c40/0/txcep.txt.npy')
    print('=================')
    read_npy('/Users/pu/Downloads/images/c40/1/tcap.txt.npy')
    read_npy('/Users/pu/Downloads/images/c40/1/tmsi.txt.npy')
    read_npy('/Users/pu/Downloads/images/c40/1/txcep.txt.npy')
