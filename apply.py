#!/usr/bin/env python

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
import datetime
from socket import *
import argparse
import threading
from business.utils import yaml_helper
import multiprocessing as mp
from ftplib import FTP, error_perm


CLASSES = ('__background__',  'crow', 'magpie', 'pigeon', 'swallow', 'sparrow', 'airplane',  'person')
# 当前触发报警的"连续"识别次数
residence_frame = 0


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


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='disperse bird apply ')
    parser.add_argument('--gpu_num', dest='gpu_num', help='please input gpu number')
    args = parser.parse_args()
    return args


# 判断ftp路径是否存在
def judge_ftp_path(ftp, path):
    try:
        ftp.cwd(path)
    except error_perm:
        try:
            ftp.mkd(path)
        except error_perm:
            meg = 'Change directory failed!: %s' % path
            print(meg)
    return


# 将检测到目前的图像存入ftp
def detection_persistence(detect_time, images):
    ftp = FTP()
    try:
        ip = business_path_config['ftp_ip_port'][0]
        port = business_path_config['ftp_ip_port'][1]
        user = business_path_config['ftp_ip_port'][2]
        passwd = business_path_config['ftp_ip_port'][3]
        ftp_path = business_path_config['ftp_ip_port'][4]
        ftp.set_debuglevel(2)
        ftp.connect(ip, port)
        ftp.login(user, passwd)
        bufsize = 4096  # 设置的缓冲区大小
        ftp.cwd(ftp_path + '/capture_picture')
        path = str(detect_time).split()[0]
        judge_ftp_path(ftp, path)
        ftp.storbinary('STOR %s' % os.path.basename(detect_time), images, bufsize)  # 上传文件
        ftp.set_debuglevel(0)
    except Exception as e:
        print('INFO:', e)
    finally:
        ftp.quit()


def socket_client_target_detection(detect_cls, detect_num, detect_img, detect_time, camera_number, disperse_sign):
    # create socket
    tcp_client_socket = socket(AF_INET, SOCK_STREAM)
    # target info
    server_ip = business_path_config['application_system_ip_port'][0]
    server_port = int(business_path_config['application_system_ip_port'][1])
    # connet servier
    tcp_client_socket.connect((server_ip, server_port))
    # send info
    send_data = {'detectContent': detect_cls, 'detectTime': detect_time, 'cameraNumber': camera_number}
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
        '''
        cv2.putText(im, str(round(score, 2)), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)  # score
        cv2.putText(im, str(class_name), (int(x1+70), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # type
        cv2.putText(im, str(current_time), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # drawing time
        cv2.putText(im, "fps:"+str(fps), (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # frame frequency
        cv2.putText(im, "takes time :"+str(round(time_takes*1000, 1))+"ms", (30, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)  # detection time
        '''
    return im


def demo_video(sess, net, frame, camera_url, max_residence_frame):
    """Detect object classes in an image usi， ng pre-computed object proposals."""
    global residence_frame
    im = frame
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    # 标识是否发送报警通讯
    warning_flag = False
    # Visualize detections for each class
    CONF_THRESH = 0.78  # threshold
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
            residence_frame += 1
            if residence_frame == max_residence_frame:
                images = vis_detections_video(im, cls, dets, timer.start_time, timer.total_time, inds, CONF_THRESH)
                detect_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                socket_client_target_detection(cls, len(inds), images, detect_time, camera_url, True)
                # 多线程写入磁盘
                t_persistence = threading.Thread(target=detection_persistence, daemon=True, args=(detect_time, images)).start()
                t_persistence.join()  # 设置主线程等待子线程结束
                timer.tic()  # 修改起始时间
                residence_frame = 0
                warning_flag = True
        else:
            residence_frame = 0
        if not warning_flag:
            images = vis_detections_video(im, cls, dets, timer.start_time, timer.total_time, inds, CONF_THRESH)
            socket_client_target_detection(cls, len(inds), images, time.ctime(), camera_url, False)


def cam(queue, camera_url):
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
    print('Loaded network {:s}'.format(tfmodel))
    timer_trigger = Timer()
    timer_trigger.tic()
    ipcam = IpCamCapture(camera_url)
    ipcam.start()
    # 暂停1秒，确保影像已经填充队列
    time.sleep(1)
    print(str(time.time()) + ' Monitoring ...')
    while True:
        frame = ipcam.get_frame()
        demo_video(sess, net, frame, camera_url, max_residence_frame)
        key = cv2.waitKey(1)
        if key == ord('q') or key == ord('Q') or key == 27:  # ESC:27  key: quit program
            ipcam.stop()
            break
    print(str(time.time()) + ' 识别系统已关闭')
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parse_args()
    # gpu_num = args.gpu_num
    project_address = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + '/EvictionBirdAI'
    business_path_config = yaml_helper.get_data_from_yaml(project_address + '/business/config/business_config.yaml')
    # 触发报警的最大"连续"识别次数
    max_residence_frame = business_path_config['max_lazy_frequency']
    # 网络摄像头
    camera_url_list = business_path_config['camera_url']
    mp.set_start_method(method='spawn')  # init
    queue = mp.Queue(maxsize=10)
    processes = list()
    for camera_url in camera_url_list:
        processes.append(mp.Process(target=cam, args=(queue, camera_url)))
    for process in processes:
        process.daemon = True
        process.start()
    for process in processes:
        process.join()
