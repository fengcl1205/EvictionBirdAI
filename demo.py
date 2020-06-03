#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from lib.config import config as cfg
from lib.utils.nms_wrapper import nms
from lib.utils.test import im_detect
from lib.nets.vgg16 import vgg16
from lib.utils.timer import Timer
import time
from socket import *


CLASSES = ('__background__',  # always index 0
                          'crow', 'magpie', 'pigeon', 'swallow', 'sparrow', 'airplane',  'person')
# CLASSES = ('__background__',  # always index 0
#                          'airplane', 'bird', 'person')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',), 'res101': ('vgg16_faster_rcnn_iter_99990.ckpt',)}
DATASETS = {'pascal_voc': ('voc_2007_trainvWrote snapshot to: D:\pyworkspace\Faster-RCNN-TensorFlow-Python3-master\default\voc_2007_trainval\default_bird\vgg16_faster_rcnn_iter_1400.ckptal',), 'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}


# picture detection drawing
def vis_detections(im, class_name, dets, inds, CONF_THRESH):
    """Draw detected bounding boxes."""
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')
    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  CONF_THRESH),
                 fontsize=14)

    plt.axis('off')
    plt.tight_layout()
    plt.draw()


# picture detection
def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""
    # Load the demo image
    im_file = os.path.join(cfg.FLAGS2["data_dir"], 'demo', image_name)
    im = cv2.imread(im_file)
    # im = frame
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    # print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))
    # Visualize detections for each class
    CONF_THRESH = 0.4
    NMS_THRESH = 0.1
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        if len(inds) == 0:
            continue
        vis_detections(im, cls, dets, inds, CONF_THRESH)  # 图片检测构图


# 内容、时间、摄像头编号
def socket_client_target_detection(detectContent, detectTime, cameraNumber, images, targetNum):
    # create socket
    tcp_client_socket = socket(AF_INET, SOCK_STREAM)
    # target info
    server_ip = input("请输入服务器ip:")
    server_port = int(input("请输入服务器port:"))
    # connet servier
    tcp_client_socket.connect((server_ip, server_port))
    # send info
    send_data = {'detectContent':detectContent, 'detectTime': detectTime, 'cameraNumber': cameraNumber}
    tcp_client_socket.send(bytes(str(send_data), encoding='gbk'))
    # Return data
    recvData = tcp_client_socket.recv(1024)
    # close
    tcp_client_socket.close()


# video detection drawing
def vis_detections_video(im, class_name, dets, start_time, time_takes, inds, CONF_THRESH):
    """Draw detected bounding boxes."""
    if len(inds) == 0:
        cv2.imshow("video capture", im)
    else:
        for i in inds:
            bbox = dets[i, :4]  # coordinate
            score = dets[i, -1]  # degree of confidence
            x1 = bbox[0]
            y1 = bbox[1]
            x2 = bbox[2]
            y2 = bbox[3]
            end_time = time.time()
            current_time = time.ctime()  # current time
            fps = round(1/(end_time - start_time), 2)
            if class_name == 'sparrow' or class_name == 'crow' or class_name == 'magpie' or class_name == 'pigeon' or class_name == 'swallow':
                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
            elif class_name == 'airplane':
                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
            elif class_name == 'person':
                cv2.rectangle(im, (x1, y1), (x2, y2), (255, 0, 0), 2)

            cv2.putText(im, str(round(score, 2)), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)  # score
            cv2.putText(im, str(class_name), (int(x1+70), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # type
            cv2.putText(im, str(current_time), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # drawing time
            cv2.putText(im, "fps:"+str(fps), (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # frame frequency
            cv2.putText(im, "takes time :"+str(round(time_takes*1000, 1))+"ms", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # detection time
    cv2.imshow("video capture", im)


# 视频检测
def demo_video(sess, net, frame, cameraNumber):
    """Detect object classes in an image using pre-computed object proposals."""
    im = frame
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    # print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))
    # Visualize detections for each class
    CONF_THRESH = 0.4  # threshold
    NMS_THRESH = 0.1
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        vis_detections_video(im, cls, dets, timer.start_time, timer.total_time, inds, CONF_THRESH)
        # socket_client_target_detection(cls, timer.start_time, cameraNumber, images, targetNum)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    # tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default', NETS[demonet][0])
    demonet = 'vgg16'
    tfmodel = os.getcwd()+ '/default/voc_2007_trainval/default_bird/vgg16_faster_rcnn_iter_200000.ckpt'
    if not os.path.isfile(tfmodel + '.meta'):
        print(tfmodel)
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16(batch_size=1)
    else:
        raise NotImplementedError

    n_classes = len(CLASSES)
    # create the structure of the net having a certain shape (which depends on the number of classes) 
    net.create_architecture(sess, "TEST", n_classes,
                            tag='default', anchor_scales=[8, 16, 32, 64])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))
    # 视频
    # cap = cv2.VideoCapture(r'D:\\pyworkspace\EvictionBirdAI\\data\\video\\06.mp4')
    # while(True):
    #     ret, frame = cap.read()
    #     demo_video(sess, net, frame, 1)
    #     key = cv2.waitKey(1)
    #     if key == ord('q') or key == ord('Q') or key == 27:  # ESC:27  key: quit program
    #         break
    #     # if ret == False:
    #     #     break
    #
    # cap.release()
    # cv2.destroyAllWindows()

    # 图片
    im_names = ['1.jpg','2.jpg','3.jpg','4.jpg','5.jpg','6.jpg','7.jpg','8.jpg','9.jpg','10.jpg','11.jpg','12.jpg','13.jpg','14.jpg','15.jpg']
    # im_names = ['0007.jpg']
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(im_name))
        demo(sess, net, im_name)
    plt.show()

























