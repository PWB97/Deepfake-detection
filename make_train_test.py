import os
import cv2
from tqdm import tqdm
import argparse
import dlib
import json
import random
from facenet_pytorch.models.mtcnn import MTCNN
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
    # datas = [train_set, test_set]
    # names = ['train', 'test']
    # for i in range(2):
    #     with open(output_dir + '/' + names[i] + '.csv', 'w') as f:
    #         f.write('\n'.join([','.join(line) for line in datas[i]]))


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
                    [src_dir + '/manipulated_sequences/' + type + '/' + c + '/videos/' + it[0] + '_' + it[1] + '.mp4', 0])
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


def generate_fftest(src_dir, output_dir, type='all', c='c23'):
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

    random.shuffle(trainList)
    random.shuffle(testList)
    random.shuffle(valList)

    train_set = return_dataset(trainList, src_dir, output_dir)
    with open(output_dir + '/train.csv', 'w') as f:
        f.write('\n'.join([','.join(line) for line in train_set]))
    test_set = return_dataset(testList, src_dir, output_dir, type='test')
    with open(output_dir + '/test.csv', 'w') as f:
        f.write('\n'.join([','.join(line) for line in test_set]))
    val_set = return_dataset(valList, src_dir, output_dir, type='val')
    with open(output_dir + '/val.csv', 'w') as f:
        f.write('\n'.join([','.join(line) for line in val_set]))


def generate_data(src_dir, output_dir, split=10):
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
        cropped_face = face_detector(frame)
        return cropped_face


def return_dataset(list, src_dir, output_dir, type='train', fd='MTCNN'):
    if fd == 'dlib':
        face_detector = dlib.cnn_face_detection_model_v1('./mmod_human_face_detector.dat')
    else:
        face_detector = MTCNN(device='cuda')
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
            img_path = os.path.join(output_dir, type, str(className),'%s_%d.jpg'
                                    % (videoName.split('/')[-4] + '_' + videoName.split('/')[-1], frame_index))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cropped_face = detect_face(gray, face_detector, fd)
            if cropped_face is not None:
                cv2.imwrite(img_path, cropped_face)
            # info = [str(className), videoName.split('/')[-1], img_path.replace('/Users/pu/Desktop', '/home/puwenbo')]
            info = [str(className), videoName.split('/')[-4] + '_' + videoName.split('/')[-1], img_path]
            # 将视频帧信息保存起来
            dataset.append(info)
            frame_index += 1
            success, frame = video_fd.read()

        video_fd.release()

    return dataset


def parse_args():
    parser = argparse.ArgumentParser(usage='')
    parser.add_argument('-i', '--src_dir', help='path to datasets', default='/home/asus/Datasets/Deepfake/faceforencis++')
    parser.add_argument('-o', '--output_dir', help='path to output', default='/home/asus/test/')
    parser.add_argument('-t', '--type', default='all')
    parser.add_argument('-g', '--gpu', default='0')
    parser.add_argument('-s', '--split', type=int, default=10)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # generate_data(args.src_dir, args.output_dir + str(args.split), args.split)
    generate_fftest(args.src_dir, args.output_dir, args.type)
    # generate_dfdc(**vars(args))
