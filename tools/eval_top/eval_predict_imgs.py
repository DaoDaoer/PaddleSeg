import os
import argparse

import numpy as np
from PIL import Image
import time
import paddle
import paddle.nn.functional as F

from paddleseg.utils import metrics, TimeAverager, calculate_eta, logger, progbar
from paddleseg.core import infer
from paddleseg.transforms import Resize

np.set_printoptions(suppress=True)


def parse_args():
    parser = argparse.ArgumentParser(description='Model prediction')

    parser.add_argument(
        "--val_path",
        dest="val_path",
        help="",
        default='data/mini_supervisely/val.txt',
        type=str)
    parser.add_argument(
        '--label_dir',
        dest='label_dir',
        help='',
        type=str,
        default='data/mini_supervisely')
    parser.add_argument(
        '--pred_dir', dest='pred_dir', type=str, default='./predict')
    return parser.parse_args()


args = parse_args()
num_classes = 2
ignore_index = 255
transforms = [Resize((192, 192))]

intersect_area_all = 0
pred_area_all = 0
label_area_all = 0
with open(args.val_path, 'r') as f:
    lines = f.readlines()
    num_images = len(lines)
    for line in lines:
        img_path, label = line.split()
        label_path = os.path.join(args.label_dir, label)
        label = np.asarray(Image.open(label_path))

        label = label.astype('int64')

        ori_shape = label.shape[-2:]
        img_path = img_path.rsplit('.')[0] + '.png'
        pred_path = os.path.join(args.pred_dir, img_path)
        pred = np.asarray(Image.open(pred_path))
        pred = paddle.to_tensor(pred, dtype='int32')
        pred = pred.unsqueeze((0, 1))
        pred = infer.reverse_transform(pred, ori_shape, transforms)

        label = paddle.to_tensor(label)
        label = label.unsqueeze((0, 1))
        intersect_area, pred_area, label_area = metrics.calculate_area(
            pred, label, num_classes, ignore_index=ignore_index)

        intersect_area_all = intersect_area_all + intersect_area
        pred_area_all = pred_area_all + pred_area
        label_area_all = label_area_all + label_area

    class_iou, miou = metrics.mean_iou(intersect_area_all, pred_area_all,
                                       label_area_all)
    class_acc, acc = metrics.accuracy(intersect_area_all, pred_area_all)
    kappa = metrics.kappa(intersect_area_all, pred_area_all, label_area_all)

    logger.info(
        "[EVAL] #Images: {} mIoU: {:.4f} Acc: {:.4f} Kappa: {:.4f} ".format(
            num_images, miou, acc, kappa))
    logger.info("[EVAL] Class IoU: \n" + str(np.round(class_iou, 4)))
    logger.info("[EVAL] Class Acc: \n" + str(np.round(class_acc, 4)))
