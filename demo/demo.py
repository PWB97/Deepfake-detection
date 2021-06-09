import torch
from torch import nn
from torchvision import transforms
import numpy as np
import cv2
from facenet_pytorch.models.mtcnn import MTCNN
from sklearn.metrics import accuracy_score
from models.model import Baseline


def load_model(restore_from, device):
    model = Baseline(use_gru=True, bi_branch=True)

    model.to(device)

    device_count = torch.cuda.device_count()
    # if device_count > 1:
    #     print('Using {} GPUs'.format(device_count))
    model = nn.DataParallel(model)

    if restore_from is not None:
        ckpt = torch.load(restore_from, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        print('Model is loaded from %s' % restore_from)

    model.eval()

    return model

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

def detect_face(frame, face_detector):
        boxes, _ = face_detector.detect(frame)
        if boxes is not None:
            best_box = boxes[0, :]
            best_face = face_boxes_post_process(frame, best_box, expand_ratio=1.33)
            return best_face
        else:
            return None


def load_data(path, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    face_detector = MTCNN(margin=0, keep_all=False, select_largest=False, thresholds=[0.6, 0.7, 0.7],
                          min_face_size=60, factor=0.8, device=device).eval()
    video_fd = cv2.VideoCapture(path)
    if not video_fd.isOpened():
        print('problem of reading video')
        return

    frame_index = 0
    faces = []
    success, frame = video_fd.read()
    while success:
        cropped_face = detect_face(frame, face_detector)
        cropped_face = cv2.resize(cropped_face, (64, 64))
        if cropped_face is not None:
            cropped_face = transform(cropped_face)
            faces.append(cropped_face)
        frame_index += 1
        success, frame = video_fd.read()
    video_fd.release()
    print('video frame length:', frame_index)
    faces = torch.stack(faces, dim=0)
    faces = torch.unsqueeze(faces, 0)
    y = torch.ones(frame_index).type(torch.IntTensor)
    return faces, y


def main(args):
    frame_y_gd = []
    y_pred = []
    frame_y_pred = []
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = load_model(args.restore_from, device)
    data, y = load_data(args.path, device)
    X = data.to(device)
    y_, cnn_y = model(X)
    y_ = torch.sigmoid(y_)
    frame_y_ = torch.sigmoid(cnn_y)
    frame_y_gd += y.detach().numpy().tolist()
    frame_y_pred += frame_y_.detach().numpy().tolist()
    frame_y_pred = torch.tensor(frame_y_pred)
    frame_y_pred = [0 if i < 0.5 else 1 for i in frame_y_pred]
    test_frame_acc = accuracy_score(frame_y_gd, frame_y_pred)
    print('video is fake:', (y_ >= 0.5).item())
    print('frame level acc:', test_frame_acc)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_from', type=str, default='./bi-model_type-baseline_gru_auc_0.150000_ep-10.pth')
    parser.add_argument('--path', type=str, default='./video/id0_id1_0002.mp4')
    args = parser.parse_args()
    main(args)
