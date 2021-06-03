import os
import cv2
from tqdm import tqdm
import argparse
import dlib
import json
import random
from facenet_pytorch.models.mtcnn import MTCNN
import numpy as np

from utils import tools

default_types = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']


def outputDir(output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for folder in ['train', 'test']:
        folder_path = os.path.join(output_dir, folder)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
            print('Folder {} is created'.format(folder_path))


def generate_dfdc_train_only(src_dir, output_dir, fr_ratio, tfake_num, fake_num):
    outputDir(output_dir)

    real = []
    fake = []

    for sub_dir in os.listdir(src_dir):
        with open(src_dir + '/' + sub_dir + '/metadata.json', 'r') as f:
            data = json.load(f)
            for name in data:
                if data[name]['label'] == 'FAKE':
                    fake.append([src_dir + '/' + sub_dir + '/' + name, 0])
                else:
                    real.append([src_dir + '/' + sub_dir + '/' + name, 1])

    train_fake = random.sample(fake, fake_num)
    train_real = random.sample(real, fake_num * fr_ratio)
    train = train_real + train_fake
    test_fake = random.sample([e for e in fake if e not in train_fake], tfake_num)
    test_real = random.sample([e for e in real if e not in train_real], tfake_num * fr_ratio)
    test = test_fake + test_real
    del fake, train_fake, test_fake, real, train_real, test_real

    train_set = return_dataset_o(train, src_dir, output_dir)
    with open(output_dir + '/train.csv', 'w') as f:
        f.write('\n'.join([','.join(line) for line in train_set]))

    test_set = return_dataset_o(test, src_dir, output_dir, type='test')
    with open(output_dir + '/test.csv', 'w') as f:
        f.write('\n'.join([','.join(line) for line in test_set]))


def generate_dfdc(src_dir, output_dir):
    outputDir(output_dir)

    trainList = []
    testList = []

    for dir in os.listdir(src_dir):
        if dir == 'test_videos':
            for name in os.listdir(src_dir + '/' + dir):
                testList.append([src_dir + '/' + dir + '/' + name, 0])
        else:
            with open(src_dir + '/' + dir + '/metadata.json', 'r') as f:
                data = json.load(f)
                for name in data:
                    if data[name]['label'] == 'Fake':
                        trainList.append([src_dir + '/' + dir + '/' + name, 0])
                    else:
                        trainList.append([src_dir + '/' + dir + '/' + name, 1])

    train_set = return_dataset(trainList, src_dir, output_dir)
    with open(output_dir + '/train.csv', 'w') as f:
        f.write('\n'.join([','.join(line) for line in train_set]))

    test_set = return_dataset(testList, src_dir, output_dir, type='test')
    with open(output_dir + '/test.csv', 'w') as f:
        f.write('\n'.join([','.join(line) for line in test_set]))


def open_json_read(json_name, src_dir, type, c, control=0):
    List = []

    fake = []
    real = []

    with open(json_name, 'r') as f:
        data = json.load(f)
        for it in data:
            if control == 1:
                List.append(
                    [src_dir + '/manipulated_sequences/' + type + '/' + c + '/videos/' + it[0] + '_' + it[1] + '.mp4',
                     0])
                fake.append([src_dir + '/manipulated_sequences/' + type + '/' + c + '/videos/' + it[0] + '_' + it[1] +
                             '.mp4', 0])
            else:
                List.append(
                    [src_dir + '/manipulated_sequences/' + type + '/' + c + '/videos/' + it[0] + '_' + it[1] + '.mp4',
                     0])
                fake.append([src_dir + '/manipulated_sequences/' + type + '/' + c + '/videos/' + it[0] + '_' + it[1] +
                             '.mp4', 0])
                List.append([src_dir + '/original_sequences' + '/youtube/' + c + '/videos/' + it[1] + '.mp4', 1])
                real.append([src_dir + '/original_sequences' + '/youtube/' + c + '/videos/' + it[1] + '.mp4', 1])

    print('total' + json_name + ': %d' % len(List))
    if control == 0:
        print('fake len:', len(fake))
    print('real len:', len(real))

    return List


def return_ff_dataset(src_dir, type, c, control):
    trainList = open_json_read('./train.json', src_dir, type, c, control)
    testList = open_json_read('./test.json', src_dir, type, c, control)
    valList = open_json_read('./val.json', src_dir, type, c, control)

    return trainList, testList, valList


def generate_ff(src_dir, output_dir, type='all', c='c23'):
    outputDir(output_dir)

    trainList = []
    testList = []
    valList = []

    control = 1

    if type == 'all':
        for type in default_types:
            if type == 'NeuralTextures':
                control = 0
            trainL, testL, valL = return_ff_dataset(src_dir, type, c, control)
            trainList += trainL
            testList += testL
            valList += valL
    else:
        trainList, testList, valList = return_ff_dataset(src_dir, type, c, 0)

    print(len(trainList))
    print(len(testList))
    random.shuffle(trainList)
    random.shuffle(testList)

    train_set = return_dataset(trainList, src_dir, output_dir)
    with open(output_dir + '/train.csv', 'w') as f:
        f.write('\n'.join([','.join(line) for line in train_set]))
    test_set = return_dataset(testList, src_dir, output_dir, type='test')
    with open(output_dir + '/test.csv', 'w') as f:
        f.write('\n'.join([','.join(line) for line in test_set]))
    # val_set = return_dataset(valList, src_dir, output_dir, type='val')
    # with open(output_dir + '/val.csv', 'w') as f:
    #     f.write('\n'.join([','.join(line) for line in val_set]))


def generate_celeb(src_dir, output_dir, split=10):
    outputDir(output_dir)

    # 划分测试集和训练集
    dataList = []
    testList = []

    Celeb_real = tools.list_file(src_dir + '/Celeb-real', 1)
    Celeb_synthesis = tools.list_file(src_dir + '/Celeb-synthesis', 0)
    YouTube_real = tools.list_file(src_dir + '/YouTube-real', 1)
    total_real = Celeb_real + YouTube_real

    print('real:%d' % len(total_real))
    print('fake: %d' % len(Celeb_synthesis))

    dataList = total_real + Celeb_synthesis

    j = i = 0
    with open(src_dir + '/List_of_testing_videos.txt') as testFile:
        for line in testFile:
            words = line.split(' ')
            testList.append([src_dir + '/' + words[1].rstrip(), int(words[0])])
            if int(words[0]) == 1:
                i += 1
            else:
                j += 1

    print('test real %d' % i)
    print('test fake %d' % j)

    trainList = [i for i in dataList if i not in testList]

    if split != 0:
        trainList = return_split(trainList, split)
        testList = return_split(testList, split)

    train_set = return_dataset(trainList, src_dir, output_dir)
    test_set = return_dataset(testList, src_dir, output_dir, type='test')

    datas = [train_set, test_set]
    names = ['train', 'test']
    for i in range(2):
        with open(output_dir + '/' + names[i] + '.csv', 'w') as f:
            f.write('\n'.join([','.join(line) for line in datas[i]]))


def return_split(data_set: [], split=10):
    print('splitting!')
    fake = []
    real = []

    for it in data_set:
        if it[1] == 0:
            fake.append(it)
        else:
            real.append(it)

    fake_len = len(real) // split
    random.shuffle(fake)
    fake = fake[0:fake_len]

    print('split fake:', len(fake))
    print('split real:', len(real))
    return fake + real


def _bbox_in_img(img, bbox):
    """
    check whether the bbox is inner an image.
    :param img: (3-d np.ndarray), image
    :param bbox: (list) [x, y, width, height]
    :return: (bool), whether bbox in image size.
    """
    if not isinstance(img, np.ndarray):
        raise ValueError("input image should be ndarray!")
    if len(img.shape) != 3:
        raise ValueError("input image should be (w,h,c)!")
    h = img.shape[0]
    w = img.shape[1]
    x_in = 0 <= bbox[0] <= w
    y_in = 0 <= bbox[1] <= h
    x1_in = 0 <= bbox[0] + bbox[2] <= w
    y1_in = 0 <= bbox[1] + bbox[3] <= h
    return x_in and y_in and x1_in and y1_in


def _enlarged_bbox(bbox, expand):
    """
    enlarge a bbox by given expand param.
    :param bbox: [x, y, width, height]
    :param expand: (tuple) (h,w), expanded pixels in height and width. if (int), same value in both side.
    :return: enlarged bbox
    """
    if isinstance(expand, int):
        expand = (expand, expand)
    s_0, s_1 = bbox[1], bbox[0]
    e_0, e_1 = bbox[1] + bbox[3], bbox[0] + bbox[2]
    x = s_1 - expand[1]
    y = s_0 - expand[0]
    x1 = e_1 + expand[1]
    y1 = e_0 + expand[0]
    width = x1 - x
    height = y1 - y
    return x, y, width, height


def _box_mode_cvt(bbox):
    """
    convert box from FCOS([xyxy], float) output to [x, y, width, height](int).
    :param bbox: (dict), an output from FCOS([x, y, x1, y1], float).
    :return: (list[int]), a box with [x, y, width, height] format.
    """
    if bbox is None:
        raise ValueError("There is no box in the dict!")
    # FCOS box format is [x, y, x1, y1]
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cvt_box = [int(bbox[0]), int(bbox[1]), max(int(w), 0), max(int(h), 0)]
    return cvt_box


def crop_bbox(img, bbox):
    """
    crop an image by giving exact bbox.
    :param img:
    :param bbox: [x, y, width, height]
    :return: cropped image
    """
    if not _bbox_in_img(img, bbox):
        raise ValueError("bbox is out of image size!img size: {0}, bbox size: {1}".format(img.shape, bbox))
    s_0 = bbox[1]
    s_1 = bbox[0]
    e_0 = bbox[1] + bbox[3]
    e_1 = bbox[0] + bbox[2]
    cropped_img = img[s_0:e_0, s_1:e_1, :]
    return cropped_img


def face_boxes_post_process(img, box, expand_ratio):
    """
    enlarge and crop the face patch from image
    :param img: ndarray, 1 frame from video
    :param box: output of MTCNN
    :param expand_ratio: default: 1.3
    :return:
    """
    box = [max(b, 0) for b in box]
    box_xywh = _box_mode_cvt(box)
    expand_w = int((box_xywh[2] * (expand_ratio - 1)) / 2)
    expand_h = int((box_xywh[3] * (expand_ratio - 1)) / 2)
    enlarged_box = _enlarged_bbox(box_xywh, (expand_h, expand_w))
    try:
        res = crop_bbox(img, enlarged_box)
    except ValueError:
        try:
            res = crop_bbox(img, box_xywh)
        except ValueError:
            return img
    return res


def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


def detect_face(frame, face_detector, type='dlib'):
    if type == 'dlib':
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 1)
        if len(faces):
            # For now only take biggest face
            face = faces[0].rect
            x, y, size = get_boundingbox(face, width, height)
            # generate cropped image
            cropped_face = frame[y:y + size, x:x + size]
            return cropped_face
        else:
            return None
    else:
        boxes, _ = face_detector.detect(frame)
        if boxes is not None:
            best_box = boxes[0, :]
            best_face = face_boxes_post_process(frame, best_box, expand_ratio=1.33)
            return best_face
        else:
            return None


def return_dataset_o(list, src_dir, output_dir, type='train'):
    face_detector = dlib.cnn_face_detection_model_v1('./mmod_human_face_detector.dat')
    dataset = []
    for videoName, className in tqdm(list):
        class_dir = os.path.join(output_dir, type, str(className))
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)

        video_path = os.path.join(src_dir, videoName)
        video_fd = cv2.VideoCapture(video_path)
        if not video_fd.isOpened():
            print('Skpped: {}'.format(video_path))
            continue

        frame_index = 0
        success, frame = video_fd.read()
        while success:
            img_path = os.path.join(output_dir, type, str(className),
                                    '%s_%d.jpg' % (
                                    videoName.split('/')[-4] + '_' + videoName.split('/')[-1], frame_index))
            height, width = frame.shape[:2]
            height = int(height / 2)
            width = int(width / 2)
            frame = cv2.resize(frame, (width, height))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector(gray, 1)
            if len(faces):
                # For now only take biggest face
                face = faces[0].rect
                x, y, size = get_boundingbox(face, width, height)
                # generate cropped image
                cropped_face = frame[y:y + size, x:x + size]
                cv2.imwrite(img_path, cropped_face)
            # info = [str(className), videoName.split('/')[-1], img_path.replace('/Users/pu/Desktop', '/home/puwenbo')]
            info = [str(className), videoName.split('/')[-4] + '_' + videoName.split('/')[-1], img_path]
            # 将视频帧信息保存起来
            dataset.append(info)
            frame_index += 1
            success, frame = video_fd.read()

        video_fd.release()

    return dataset


def return_dataset(list, src_dir, output_dir, type='train', fd='dlib'):
    if fd == 'dlib':
        face_detector = dlib.cnn_face_detection_model_v1('./mmod_human_face_detector.dat')
    else:
        face_detector = MTCNN(margin=0, keep_all=False, select_largest=False, thresholds=[0.6, 0.7, 0.7],
                              min_face_size=60, factor=0.8, device='cuda').eval()
    dataset = []
    for videoName, className in tqdm(list):
        class_dir = os.path.join(output_dir, type, str(className))
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)
        video_path = os.path.join(src_dir, videoName)
        video_fd = cv2.VideoCapture(video_path)
        if not video_fd.isOpened():
            print('Skpped: {}'.format(video_path))
            continue

        frame_index = 0
        success, frame = video_fd.read()
        while success:
            img_path = os.path.join(output_dir, type, str(className), '%s_%d.png'
                                    % (videoName.split('/')[-4] + '_' + videoName.split('/')[-1], frame_index))
            cropped_face = detect_face(frame, face_detector, fd)
            if cropped_face is not None:
                cv2.imwrite(img_path, cropped_face)
                info = [str(className), videoName.split('/')[-4] + '_' + videoName.split('/')[-1], img_path]
                # 将视频帧信息保存起来
                dataset.append(info)
            frame_index += 1
            success, frame = video_fd.read()
        print(frame_index)
        video_fd.release()

    return dataset


def parse_args():
    parser = argparse.ArgumentParser(usage='make_train_test.py used for make datasets, including FF++, DFDC, Celeb-DF')
    parser.add_argument('-i', '--src_dir', help='path to datasets', default='')
    parser.add_argument('-o', '--output_dir', help='path to output', default='')
    parser.add_argument('-t', '--type', help='used for FF++', default='all')
    parser.add_argument('-g', '--gpu', default='7')
    parser.add_argument('-s', '--split', help='the split of pos and neg samples', type=int, default=60)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.split != 0:
        output = args.output_dir + str(args.split)
    else:
        output = args.output_dir

    # generate_celeb(args.src_dir, output, args.split)
    # generate_ff(args.src_dir, args.output_dir, args.type)
    generate_dfdc_train_only(args.src_dir, output, args.split, 3, 5)
