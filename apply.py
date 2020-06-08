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

import os
import cv2
import numpy as np
import tensorflow as tf
from lib.utils.nms_wrapper import nms
from lib.utils.test import im_detect
from lib.nets.vgg16 import vgg16
from lib.utils.timer import Timer
import time
from socket import *
import argparse


CLASSES = ('__background__',  # always index 0
                          'crow', 'magpie', 'pigeon', 'swallow', 'sparrow', 'airplane',  'person')


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='disperse bird apply ')
    parser.add_argument('--url', dest='camera_url', help='please input camera url')
    parser.add_argument('--project_address', dest='project_address', help='please input EvictionBirdAI project address')
    args = parser.parse_args()
    return args


def socket_client_target_detection(detectCls, detectNum, detectImg, detectTime, cameraNumber, disperse_sign):
    # create socket
    tcp_client_socket = socket(AF_INET, SOCK_STREAM)
    # target info
    server_ip = ''
    server_port = int('')
    # connet servier
    tcp_client_socket.connect((server_ip, server_port))
    # send info
    send_data = {'detectContent': detectCls, 'detectTime': detectTime, 'cameraNumber': cameraNumber}
    tcp_client_socket.send(bytes(str(send_data), encoding='gbk'))
    # Return data
    recvData = tcp_client_socket.recv(1024)
    # close
    tcp_client_socket.close()


# video detection drawing
def vis_detections_video(im, class_name, dets, start_time, time_takes, inds, CONF_THRESH):
    """Draw detected bounding boxes."""
    for i in inds:
        bbox = dets[i, :4]  # coordinate
        score = dets[i, -1]  # degree of confidence
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]
        end_time = time.time()
        # current_time = time.ctime()  # current time
        fps = round(1/(end_time - start_time), 2)
        if class_name == 'sparrow' or class_name == 'crow' or class_name == 'magpie' or class_name == 'pigeon' or class_name == 'swallow':
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
        elif class_name == 'airplane':
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
        elif class_name == 'person':
            cv2.rectangle(im, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(im, str(round(score, 2)), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)  # score
        cv2.putText(im, str(class_name), (int(x1+70), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # type
        # cv2.putText(im, str(current_time), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # drawing time
        cv2.putText(im, "fps:"+str(fps), (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # frame frequency
        cv2.putText(im, "takes time :"+str(round(time_takes*1000, 1))+"ms", (30, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)  # detection time
    return im


def demo_video(sess, net, frame, camera_url):
    """Detect object classes in an image using pre-computed object proposals."""
    im = frame
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    # Visualize detections for each class
    CONF_THRESH = 0.6  # threshold
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
        if cls == 'crow' or cls == 'magpie' or cls == 'pigeon' or cls == 'swallow' \
                or cls == 'sparrow' and len(inds) != 0:
            if time.time() - timer_trigger.start_time > residence_time:
                images = vis_detections_video(im, cls, dets, timer.start_time, timer.total_time, inds, CONF_THRESH)
                socket_client_target_detection(cls, len(inds), images, time.ctime(), camera_url, True)
                timer_trigger.tic()  # 修改起始时间
            else:
                images = vis_detections_video(im, cls, dets, timer.start_time, timer.total_time, inds, CONF_THRESH)
                socket_client_target_detection(cls, len(inds), images, time.ctime(), camera_url, False)
        elif cls == 'airplane' and len(inds) != 0:
            pass
        elif cls == 'person' and len(inds) != 0:
            pass
        else:
            pass


if __name__ == '__main__':
    args = parse_args()
    camera_url = args.camera_url
    project_address = args.project_address

    demonet = 'vgg16'
    tfmodel = project_address + '/default/voc_2007_trainval/default_bird/vgg16_faster_rcnn_iter_200000.ckpt'
    if not os.path.isfile(tfmodel + '.meta'):
        print(tfmodel)
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))
    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)
    # load networkzhe
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
    residence_time = 2
    print('Loaded network {:s}'.format(tfmodel))
    timer_trigger = Timer()
    timer_trigger.tic()
    # url = 'rtsp://admin:123456@192.168.1.13:554'
    cap = cv2.VideoCapture(camera_url)
    # cap = cv2.VideoCapture(r'D:\\pyworkspace\EvictionBirdAI\\data\\video\\05.mp4')
    while (cap.isOpened()):
    # while(True):
        ret, frame = cap.read()
        demo_video(sess, net, frame, camera_url)
        key = cv2.waitKey(1)
        if key == ord('q') or key == ord('Q') or key == 27:  # ESC:27  key: quit program
            break
        # if ret == False:
        #     break

    cap.release()
    cv2.destroyAllWindows()


























