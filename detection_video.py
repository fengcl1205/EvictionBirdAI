#!/usr/bin/env python

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
import threading
from business.utils import yaml_helper
import multiprocessing
from business.utils import path_helper as ph


# CLASSES = ('__background__',  # always index 0
#                           'crow', 'magpie', 'pigeon', 'swallow', 'sparrow', 'airplane',  'person')
# CLASSES = ('__background__',  # always index 0
#                          'airplane', 'bird', 'person')

project_address = ph.get_local_project_path(os.path.dirname(os.path.abspath(__file__)), 0)
business_path_config = yaml_helper.get_data_from_yaml(project_address + '/business/config/business_config.yaml')
detect_categories_config = yaml_helper.get_data_from_yaml(project_address + '/business/config/detect_cls.yaml')
# 触发报警的最大"连续"识别次数
max_residence_frame = business_path_config['max_lazy_frequency']
# 网络摄像头
camera_url_list = business_path_config['camera_url']
ftp_images_retain_time = business_path_config['ftp_images_retain_time']
local_business_logs_path = business_path_config['local_business_logs_path']
local_business_logs_retain_time = business_path_config['local_business_logs_retain_time']
detect_threshold = business_path_config['detect_threshold']
nms_threshold = business_path_config['nms_threshold']
local_cap_video_path = business_path_config['local_cap_video_path']
all_detect_categories = detect_categories_config['all_detect_categories']
target_detect_categories = detect_categories_config['target_detect_categories']
CLASSES = tuple(all_detect_categories)
TARGET_CLASSES = tuple(target_detect_categories)

# CLASSES = ('__background__',
#                  'aeroplane', 'bicycle', 'boat',
#                  'bottle', 'bus', 'car', 'cat', 'chair',
#                  'cow', 'diningtable', 'dog', 'horse',
#                  'motorbike', 'person', 'pottedplant',
#                  'sheep', 'sofa', 'train', 'tvmonitor', 'crow', 'magpie', 'pigeon', 'swallow', 'sparrow')

# 接收摄影机串流影像，采用多线程的方式，降低缓冲区栈图帧的问题。
class IpCamCapture:
    def __init__(self, url):
        self.url = url
        self.Frame = list()
        self.status = False
        self.isstop = False
        self.capture = cv2.VideoCapture(self.url)

    def start(self):
        print('ipcam started!')
        # 把程序放进子线程，daemon=True 表示该线程会随着主线程关闭而关闭。
        threading.Thread(target=self.query_frame, daemon=True, args=()).start()

    # 停止无限循环的开关
    def stop(self):
        self.isstop = True
        print('ipcam stopped!')

    # 当有需要影像时，再回传最新的影像。
    def get_frame(self):
        return self.Frame

    def query_frame(self):
        while not self.isstop:
            self.status, self.Frame = self.capture.read()
            # 摄像头传来数据由于转义等未知原因无法读取时，重新调用摄像头
            if not self.status:
                print('视频流发现问题，矫正中...')
                self.capture = cv2.VideoCapture(self.url)
                self.status, self.Frame = self.capture.read()
        self.capture.release()


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

'''
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
    CONF_THRESH = 0.7
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
            return
        vis_detections(im, cls, dets, inds, CONF_THRESH)  # 图片检测构图
'''

# 内容、时间、摄像头编号
def socket_client_target_detection(detect_cls, detect_amount, detect_img, detect_time, camera_number, disperse_sign):
    # create socket
    tcp_client_socket = socket(AF_INET, SOCK_STREAM)
    # target info
    server_ip = input("请输入服务器ip:")
    server_port = int(input("请输入服务器port:"))
    # connet servier
    tcp_client_socket.connect((server_ip, server_port))
    # send info
    send_data = {'disperse_sign': disperse_sign, 'detectContent': detect_cls, 'detect_amount': detect_amount,
                 'detect_img': detect_img, 'detectTime': detect_time, 'cameraNumber': camera_number}
    tcp_client_socket.send(bytes(str(send_data), encoding='gbk'))
    # Return data
    recvData = tcp_client_socket.recv(1024)
    # close
    tcp_client_socket.close()


# video detection drawing
def vis_detections_video(im, class_name, dets, start_time, time_takes, inds, CONF_THRESH):
    """Draw detected bounding boxes."""
    if len(inds) == 0:
        # cv2.imshow("video capture", im)
        return im
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
            if class_name in TARGET_CLASSES:
                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
            else:
                cv2.rectangle(im, (x1, y1), (x2, y2), (255, 0, 0), 2)

            cv2.putText(im, str(round(score, 2)), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)  # score
            cv2.putText(im, str(class_name), (int(x1+70), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # type
            cv2.putText(im, str(current_time), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # drawing time
            cv2.putText(im, "fps:"+str(fps), (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # frame frequency
            cv2.putText(im, "takes time :"+str(round(time_takes*1000, 1))+"ms", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # detection time
        return im


# 视频检测
def demo_video(sess, net, frame1, frame2, frame3, frame4, width, height, camera_url, max_residence_frame):
    """Detect object classes in an image using pre-computed object proposals."""

    timer = Timer()
    timer.tic()
    scores1, boxes1 = im_detect(sess, net, frame1)
    scores2, boxes2 = im_detect(sess, net, frame2)
    scores3, boxes3 = im_detect(sess, net, frame3)
    scores4, boxes4 = im_detect(sess, net, frame4)

    timer.toc()
    # print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))
    # Visualize detections for each class
    CONF_THRESH = detect_threshold  # threshold
    NMS_THRESH = nms_threshold
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes1 = boxes1[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores1 = scores1[:, cls_ind]
        dets1 = np.hstack((cls_boxes1,
                          cls_scores1[:, np.newaxis])).astype(np.float32)
        keep1 = nms(dets1, NMS_THRESH)
        dets1 = dets1[keep1, :]
        inds1 = np.where(dets1[:, -1] >= CONF_THRESH)[0]
        im1 = vis_detections_video(frame1, cls, dets1, timer.start_time, timer.total_time, inds1, CONF_THRESH)

        cls_boxes2 = boxes2[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores2 = scores2[:, cls_ind]
        dets2 = np.hstack((cls_boxes2,
                           cls_scores2[:, np.newaxis])).astype(np.float32)
        keep2 = nms(dets2, NMS_THRESH)
        dets2 = dets2[keep2, :]
        inds2 = np.where(dets2[:, -1] >= CONF_THRESH)[0]
        im2 = vis_detections_video(frame2, cls, dets2, timer.start_time, timer.total_time, inds2, CONF_THRESH)

        cls_boxes3 = boxes3[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores3 = scores3[:, cls_ind]
        dets3 = np.hstack((cls_boxes3,
                           cls_scores3[:, np.newaxis])).astype(np.float32)
        keep3 = nms(dets3, NMS_THRESH)
        dets3 = dets3[keep3, :]
        inds3 = np.where(dets3[:, -1] >= CONF_THRESH)[0]
        im3 = vis_detections_video(frame3, cls, dets3, timer.start_time, timer.total_time, inds3, CONF_THRESH)

        cls_boxes4 = boxes4[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores4 = scores4[:, cls_ind]
        dets4 = np.hstack((cls_boxes4,
                           cls_scores4[:, np.newaxis])).astype(np.float32)
        keep4 = nms(dets4, NMS_THRESH)
        dets4 = dets4[keep4, :]
        inds4 = np.where(dets4[:, -1] >= CONF_THRESH)[0]
        im4 = vis_detections_video(frame4, cls, dets4, timer.start_time, timer.total_time, inds4, CONF_THRESH)

        frameLeftUp = cv2.resize(im1, (int(width // 4), int(height // 4)), interpolation=cv2.INTER_CUBIC)
        frameRightUp = cv2.resize(im2, (int(width // 4), int(height // 4)), interpolation=cv2.INTER_CUBIC)
        frameLeftDown = cv2.resize(im3, (int(width // 4), int(height // 4)), interpolation=cv2.INTER_CUBIC)
        frameRightDown = cv2.resize(im4, (int(width // 4), int(height // 4)), interpolation=cv2.INTER_CUBIC)
        ss1 = np.hstack((frameLeftUp, frameRightUp))
        ss2 = np.hstack((frameLeftDown, frameRightDown))
        ss = np.vstack((ss1, ss2))
        cv2.setWindowProperty('video', cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)
        cv2.imshow('video', ss)
        # socket_client_target_detection(cls, timer.start_time, cameraNumber, images, targetNum)


# 系统参数解析
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='disperse bird apply ')
    parser.add_argument('--gpu_num', dest='gpu_num', help='please input gpu number')
    args = parser.parse_args()
    return args


def picture_deal():
    demonet = 'vgg16'
    tfmodel = project_address + '/default/voc_2007_trainval/default_bird/vgg16_faster_rcnn_iter_300000.ckpt'
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
    # 图片
    im_names = ['1.jpg','2.jpg', '3.jpg','4.jpg','5.jpg','6.jpg','7.jpg','8.jpg','9.jpg','10.jpg','11.jpg','12.jpg','13.jpg','14.jpg','15.jpg']
    # im_names = ['0007.jpg']
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(im_name))
        # demo(sess, net, im_name)
    plt.show()


def video_deal(video_name):
    demonet = 'vgg16'
    tfmodel = project_address + '/default/voc_2007_trainval/default_bird/vgg16_faster_rcnn_iter_300000.ckpt'
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
    cap = cv2.VideoCapture(project_address + '/data/video/' + video_name)
    # while (cap.isOpened()):
    print('Monitoring')
    while (True):
        ret, frame = cap.read()
        if not (ret):
            print('略过')
            cap = cv2.VideoCapture(project_address + '/data/video/' + video_name)
            ret, frame = cap.read()
            # demo_video(sess, net, frame, 1, '')
            key = cv2.waitKey(1)
            if key == ord('q') or key == ord('Q') or key == 27:  # ESC:27  key: quit program
                break
            continue
        # demo_video(sess, net, frame, 1)
        key = cv2.waitKey(1)
        if key == ord('q') or key == ord('Q') or key == 27:  # ESC:27  key: quit program
            break


def cam(queue, camera_url):
    demonet = 'vgg16'
    tfmodel = project_address + '/default/voc_2007_trainval/default_bird/vgg16_faster_rcnn_iter_300000.ckpt'
    if not os.path.isfile(tfmodel + '.meta'):
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
    # print('Loaded network {:s}'.format(tfmodel))
    timer_trigger = Timer()
    timer_trigger.tic()

    ipcam1 = IpCamCapture(camera_url[0])
    ipcam2 = IpCamCapture(camera_url[1])
    ipcam3 = IpCamCapture(camera_url[2])
    ipcam4 = IpCamCapture(camera_url[3])
    ipcam1.start()
    ipcam2.start()
    ipcam3.start()
    ipcam4.start()
    # width, height = ipcam1.get_frame_width_higth
    # ipcam = IpCamCapture(camera_url)
    # ipcam.start()
    # 暂停1秒，确保影像已经填充
    time.sleep(1)
    print(str(time.time()) + ' Monitoring ...')
    width = 1920
    height = 1080
    while True:
        frame1 = ipcam1.get_frame()
        frame2 = ipcam2.get_frame()
        frame3 = ipcam3.get_frame()
        frame4 = ipcam4.get_frame()
        demo_video(sess, net, frame1, frame2, frame3, frame4, width, height, camera_url, max_residence_frame)
        key = cv2.waitKey(1)
        if key == ord('q') or key == ord('Q') or key == 27:  # ESC:27  key: quit program
            ipcam1.stop()
            ipcam2.stop()
            ipcam3.stop()
            ipcam4.stop()
            break
    input_p.close()
    print(str(time.time()) + ' 识别系统已关闭')
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parse_args()
    project_address = ph.get_local_project_path(os.path.dirname(os.path.abspath(__file__)), 0)
    business_path_config = yaml_helper.get_data_from_yaml(project_address + '/business/config/business_config.yaml')
    # 触发报警的最大"连续"识别次数
    max_residence_frame = business_path_config['max_lazy_frequency']
    output_p, input_p = multiprocessing.Pipe()

    data_sources = 'net_video'

    if data_sources == 'net_video':
        # 网络摄像头
        camera_url_list = business_path_config['camera_url']
        multiprocessing.set_start_method(method='spawn')  # init
        queue = multiprocessing.Queue(maxsize=10)
        processes = list()
        # for camera_url in camera_url_list:
        #     processes.append(multiprocessing.Process(target=cam, args=(queue, camera_url, )))
        processes.append(multiprocessing.Process(target=cam, args=(queue, camera_url_list)))
        # for index in range(len(camera_url_list)):
        #     processes.append(multiprocessing.Process(target=cam, args=(queue, camera_url_list[index])))
        for process in processes:
            process.daemon = True
            process.start()
        for process in processes:
            process.join()

    elif data_sources == 'local_video':
        video_deal('02.mp4')
    elif data_sources == 'local_pic':
        picture_deal()





















