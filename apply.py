#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
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
from apscheduler.schedulers.background import BackgroundScheduler
from business.utils import path_helper as ph
from business.utils import log_helper
import sys


project_address = ph.get_local_project_path(os.path.dirname(os.path.abspath(__file__)), 0)
business_path_config = yaml_helper.get_data_from_yaml(project_address + '/business/config/business_config.yaml')
detect_categories_config = yaml_helper.get_data_from_yaml(project_address + '/business/config/detect_cls.yaml')
eliminate_misjudgment = yaml_helper.get_data_from_yaml(project_address + '/business/config/eliminate_misjudgment.yaml')
# 触发报警的最大"连续"识别次数
max_residence_frame = business_path_config['max_lazy_frequency']
# 网络摄像头
camera_url_list = business_path_config['camera_url']
# 网络摄像头名称
camera_name_list = business_path_config['camera_name']
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
# CLASSES = ('__background__',  'crow', 'magpie', 'pigeon', 'swallow', 'sparrow', 'airplane',  'person')
# 使得图片矩阵显示全面
np.set_printoptions(threshold=np.inf)


# 接收摄影机串流影像，采用多线程的方式，降低缓冲区栈图帧的问题。
class IpCamCapture:
    def __init__(self, url):
        self.url = url
        self.Frame = list()
        self.status = False
        self.isstop = False
        self.capture = cv2.VideoCapture(self.url)

    def start_t(self, camera_url):
        log_helper.log_out('info', local_business_logs_path,
                           camera_url + ' 摄像头打卡')
        # 把程序放进子线程，daemon=True 表示该线程会随着主线程关闭而关闭。
        threading.Thread(target=self.query_frame, daemon=True, args=()).start()

    # 停止无限循环的开关
    def stop(self, camera_url):
        self.isstop = True
        log_helper.log_out('info', local_business_logs_path,
                           camera_url + ' 摄像头关闭')

    # 当有需要影像时，再回传最新的影像。
    def get_frame(self):
        return self.Frame

    def query_frame(self):
        try:
            while not self.isstop:
                self.status, self.Frame = self.capture.read()
                # 摄像头传来数据由于转义等未知原因无法读取时，重新调用摄像头
                if not self.status:
                    log_helper.log_out('info', local_business_logs_path,
                                       '视频流发现问题，矫正中...')
                    self.capture = cv2.VideoCapture(self.url)
                    self.status, self.Frame = self.capture.read()
            self.capture.release()
        except BaseException as e:
            log_helper.log_out('error', local_business_logs_path,
                           'File: ' + e.__traceback__.tb_frame.f_globals['__file__']
                           + ', lineon: ' + str(e.__traceback__.tb_lineno) + ', error info: '
                           + str(e))
            raise


class socket_c:
    def __init__(self):
        self.tcp_client_socket = socket(AF_INET, SOCK_STREAM)
        self.server_ip = business_path_config['application_system_ip_port'][0]
        self.server_port = int(business_path_config['application_system_ip_port'][1])
        self.send_data = dict()

    def socket_conn(self):
        try:
            # connet servier
            self.tcp_client_socket.connect((self.server_ip, self.server_port))
        except BaseException as e:
            log_helper.log_out('error', local_business_logs_path,
                       'File: ' + e.__traceback__.tb_frame.f_globals['__file__']
                       + ', lineon: ' + str(e.__traceback__.tb_lineno) + ', error info: '
                       + str(e))

    def socket_cend(self, target_info, detect_img, detect_time, camera_number, disperse_sign):
        try:
            if disperse_sign == '1' and len(target_info) != 0:
                self.send_data = {"disperseSign": disperse_sign, "detectTargetInfo": target_info,
                                  "detectTime": detect_time, "cameraNumber": camera_number}
                self.tcp_client_socket.send(str(self.send_data).encode())
        except BaseException as e:
            log_helper.log_out('error', local_business_logs_path,
                       'File: ' + e.__traceback__.tb_frame.f_globals['__file__']
                       + ', lineon: ' + str(e.__traceback__.tb_lineno) + ', error info: '
                       + str(e))

    def socket_recv(self):
        recvData = self.tcp_client_socket.recv(1024)
        return recvData.decode()

    def socket_clse(self):
        self.tcp_client_socket.close()


# 检测到目标的图片并且持久化
def detection_persistence(detect_time, images, camera_url):
    '''
    qn_ftp = QnFtp()
    try:
        qn_ftp.upload(detect_time, images)
    except BaseException as e:
         log_helper.log_out('error', local_business_logs_path, 'File: ' + e.__traceback__.tb_frame.f_globals['__file__']
                           + ', lineon: ' + str(e.__traceback__.tb_lineno) + ', error info: '
                           + str(e))
         raise
    finally:
        qn_ftp.ftp_quit()
    '''
    try:
        path = local_cap_video_path + '/' + camera_url + '/' + detect_time.split()[0]
        if not os.path.exists(path):
            os.makedirs(path)
        cv2.imwrite(path + '/' + detect_time + '.jpg', images)
    except BaseException as e:
        log_helper.log_out('error', local_business_logs_path, 'File: ' + e.__traceback__.tb_frame.f_globals['__file__']
                           + ', lineon: ' + str(e.__traceback__.tb_lineno) + ', error info: '
                           + str(e))
        raise


# 检测到目标的视频并且持久化
def detection_video_persistence(detect_time, images):
    '''
    qn_ftp = QnFtp()
    try:
        qn_ftp.upload(detect_time, images)
    except BaseException as e:
         log_helper.log_out('error', local_business_logs_path, 'File: ' + e.__traceback__.tb_frame.f_globals['__file__']
                           + ', lineon: ' + str(e.__traceback__.tb_lineno) + ', error info: '
                           + str(e))
         raise
    finally:
        qn_ftp.ftp_quit()
    '''
    try:
        path = local_cap_video_path + '/' + detect_time.split()[0]
        if not os.path.exists(path):
            os.makedirs(path)
        cv2.imwrite(path + '/' + detect_time + '.jpg', images)
    except BaseException as e:
        log_helper.log_out('error', local_business_logs_path, 'File: ' + e.__traceback__.tb_frame.f_globals['__file__']
                           + ', lineon: ' + str(e.__traceback__.tb_lineno) + ', error info: '
                           + str(e))
        raise


# 清空本地指定日期范围外的日志文件
def clear_local_business_logs(current_time_y, current_time_m, current_time_d):
    try:
        files = os.listdir(local_business_logs_path)
        for file in files:  # 例2020-06-12
            file_path = os.path.join(local_business_logs_path, file)
            if os.path.isdir(file_path):  # 日志是按月为单位目录存储的
                if (current_time_y - int(file.split('-')[0])) * 12 + (current_time_m - int(file.split('-')[1]))\
                        > ftp_images_retain_time:
                    os.remove(os.path.join(local_business_logs_path, file))
    except BaseException as e:
        log_helper.log_out('error', local_business_logs_path,
                       'File: ' + e.__traceback__.tb_frame.f_globals['__file__']
                       + ', lineon: ' + str(e.__traceback__.tb_lineno) + ', error info: '
                       + str(e))
        raise


# 清空本地指定日期范围外的捕获图像
def clear_local_capture_images(current_time_y, current_time_m, current_time_d):
    try:
        files = os.listdir(local_cap_video_path)
        for file in files:  # 例2020-06-12
            file_path = os.path.join(local_cap_video_path, file)
            if os.path.isdir(file_path):  # 日志是按月为单位目录存储的
                if (datetime.date(int(current_time_y), int(current_time_m), int(current_time_d)) -
                    datetime.date(int(file.split('-')[0]), int(file.split('-')[1]), int(file.split('-')[2]))).days \
                        > ftp_images_retain_time:
                    shutil.rmtree(os.path.join(local_cap_video_path, file))
    except BaseException as e:
        log_helper.log_out('error', local_business_logs_path,
                           'File: ' + e.__traceback__.tb_frame.f_globals['__file__']
                           + ', lineon: ' + str(e.__traceback__.tb_lineno) + ', error info: '
                           + str(e))
        raise


# 系统参数解析
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='disperse bird apply ')
    parser.add_argument('--gpu_num', dest='gpu_num', help='please input gpu number')
    args = parser.parse_args()
    return args


# video detection drawing
def vis_detections_video(im, class_name, dets, start_time, time_takes, inds, CONF_THRESH, camera_url):
    """Draw detected bounding boxes."""
    invalid_target_ele = list()
    valid_target_info = list()
    if len(inds) != 0:
        for i in inds:
            bbox = dets[i, :4]  # coordinate
            score = dets[i, -1]  # degree of confidence
            x1 = bbox[0]
            y1 = bbox[1]
            x2 = bbox[2]
            y2 = bbox[3]
            temp = ''
            temp += str(x1)
            temp += '#'
            temp += str(y1)
            temp += '#'
            temp += str(x2)
            temp += '#'
            temp += str(y2)
            # 如果坐标值在误判区则不显示
            if len(camera_url_list[0]) != 0 and camera_url == camera_url_list[0]:
                if temp in eliminate_misjudgment['misjudgment_coordinate_SouthTower']:
                    invalid_target_ele.append(class_name)
                    continue
            elif len(camera_url_list[1]) != 0 and camera_url == camera_url_list[1]:
                if temp in eliminate_misjudgment['misjudgment_coordinate_NorthTower']:
                    invalid_target_ele.append(class_name)
                    continue
            elif len(camera_url_list[2]) != 0 and camera_url == camera_url_list[2]:
                if temp in eliminate_misjudgment['misjudgment_coordinate_DirectionalStation']:
                    invalid_target_ele.append(class_name)
                    continue
            elif len(camera_url_list[3]) != 0 and camera_url == camera_url_list[3]:
                if temp in eliminate_misjudgment['misjudgment_coordinate_NorthSlide']:
                    invalid_target_ele.append(class_name)
                    continue
            # 记录有效坐标信息
            centre_x = round(((x1 - x2) / 2), 2) + x2
            centre_y = round(((y1 - y2) / 2), 2) + y2

            valid_target_info.append(list((centre_x, centre_y)))
            # 打印误判坐标信息
            # log_helper.log_out('info', local_business_logs_path,
            #                    'x1      '+str(x1)+'y1     '+str(y1)+'x2     '+str(x2)+'y2       '+str(y2))

            end_time = time.time()
            # current_time = time.ctime()  # current time
            fps = round(1/(end_time - start_time), 2)
            if class_name in TARGET_CLASSES:
                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
            else:
                cv2.rectangle(im, (x1, y1), (x2, y2), (255, 0, 0), 2)

            cv2.putText(im, str(round(score, 2)), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)  # score
            cv2.putText(im, str(class_name), (int(x1+70), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # type
            # cv2.putText(im, str(current_time), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # drawing time
            # cv2.putText(im, "fps:"+str(fps), (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # frame frequency
            # cv2.putText(im, "takes time :"+str(round(time_takes*1000, 1))+"ms", (30, 90), cv2.FONT_HERSHEY_SIMPLEX,
            #            1, (255, 255, 255), 2)  # detection time

    return im, invalid_target_ele, valid_target_info


# 摄像头视频检测
def demo_video(sess, net, frame, camera_url, lazy_frequency, sc):
    """Detect object classes in an image usi， ng pre-computed object proposals."""
    im = frame
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    # 在一针图像中是否找到目标
    find_target_flag = False
    # 一帧频图像中每一类目标个数
    target_info = dict()
    detect_time = None
    # Visualize detections for each class
    CONF_THRESH = detect_threshold  # threshold
    NMS_THRESH = nms_threshold
    try:
        video_name = ''
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1  # because we skipped background
            cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
            images, invalid_target_ele, valid_target_info = vis_detections_video(
                im, cls, dets, timer.start_time, timer.total_time, inds, CONF_THRESH, camera_url)
            frame = cv2.resize(images, (int(1920 // 2), int(1080 // 2)), interpolation=cv2.INTER_CUBIC)
            cv2.imshow(video_name, frame)
            video_name = camera_name_list[camera_url_list.index(camera_url)]

            # if camera_url == 'rtsp://admin:123456@192.168.1.13:554':
            #     video_name = 'South Tower'
            # elif camera_url == 'rtsp://admin:123456@192.168.1.14:554':
            #     video_name = 'North Tower'
            # elif camera_url == 'rtsp://admin:123456@192.168.1.15:554':
            #     video_name = 'Directional Station'
            # elif camera_url == 'rtsp://admin:123456@192.168.1.16:554':
            #     video_name = 'North Slide'
            if len(inds) != 0 and cls in TARGET_CLASSES and (len(inds) - len(invalid_target_ele)) != 0:
                find_target_flag = True
                # target_info[cls] = len(inds) - len(invalid_target_ele)
                target_info[cls] = valid_target_info
            # 多线程检测目标写入磁盘
            detect_time = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S.%f')
            t_persistence = threading.Thread(target=detection_persistence, daemon=True, args=(
                detect_time, images, camera_url)).start()
            t_persistence.join()  # 设置主线程等待子线程结束
        if find_target_flag:
            lazy_frequency += 1
        else:
            lazy_frequency = 0
        if lazy_frequency == max_residence_frame:
            sc.socket_conn()
            sc.socket_cend(target_info, '', detect_time, camera_url, '1')
            sc.socket_clse()
            lazy_frequency = 0
        return lazy_frequency
    except BaseException as e:
        log_helper.log_out('error', local_business_logs_path, 'File: ' + e.__traceback__.tb_frame.f_globals['__file__']
                           + ', lineon: ' + str(e.__traceback__.tb_lineno) + ', error info: '
                           + str(e))
        raise


# 相机触发函数
def cam(queue, camera_url):
    demonet = 'vgg16'
    tfmodel = project_address + '/default/voc_2007_trainval/default_bird/vgg16_faster_rcnn_iter_300000.ckpt'
    if not os.path.isfile(tfmodel + '.meta'):
        log_helper.log_out('info', local_business_logs_path,
                           tfmodel)
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
    sc = socket_c()
    threading.Thread(target=sc.socket_conn, daemon=True, args=()).start()
    try:
        n_classes = len(CLASSES)
        # create the structure of the net having a certain shape (which depends on the number of classes)
        net.create_architecture(sess, "TEST", n_classes,
                                tag='default', anchor_scales=[8, 16, 32, 64])
        saver = tf.train.Saver()
        saver.restore(sess, tfmodel)
        # print('Loaded network {:s}'.format(tfmodel))
        log_helper.log_out('info', local_business_logs_path,
                           'Loaded network {:s}'.format(tfmodel))
        timer_trigger = Timer()
        timer_trigger.tic()
        ipcam = IpCamCapture(camera_url)
        ipcam.start_t(camera_url)
        # 暂停1秒，确保影像已经填充队列
        time.sleep(1)
        # print(str(time.time()) + ' Monitoring ...')
        log_helper.log_out('info', local_business_logs_path,
                           str(time.time()) + ' Monitoring ...')
        # 发现目标的连续次数
        lazy_frequency = 0
        while True:
            frame = ipcam.get_frame()
            lazy_frequency = demo_video(sess, net, frame, camera_url, lazy_frequency, sc)
            key = cv2.waitKey(1)
            if key == ord('q') or key == ord('Q') or key == 27:  # ESC:27  key: quit program
                ipcam.stop(camera_url)
                break
        print(str(time.time()) + ' 识别系统已关闭')
        log_helper.log_out('info', local_business_logs_path,
                           str(time.time()) + ' 识别系统已关闭')
    except BaseException as e:
        log_helper.log_out('error', local_business_logs_path, 'File: ' + e.__traceback__.tb_frame.f_globals['__file__']
                           + ', lineon: ' + str(e.__traceback__.tb_lineno) + ', error info: '
                           + str(e))
        raise
    finally:
        cv2.destroyAllWindows()
        sess.close()
        sc.socket_clse()


# 清空指定日期前的ftp上的捕获图像和本地的日志
def clear_folds():
    try:
        current_time_y = datetime.datetime.now().strftime('%Y')
        current_time_m = datetime.datetime.now().strftime('%m')
        current_time_d = datetime.datetime.now().strftime('%d')
        scheduler = BackgroundScheduler()
        # 定时清理
        scheduler.add_job(func=clear_local_business_logs, args=(current_time_y, current_time_m, current_time_d),
                          trigger='cron', month='*', day='*', hour='0', minute='0')
        scheduler.add_job(func=clear_local_capture_images, args=(current_time_y, current_time_m, current_time_d),
                          trigger='cron', month='*', day='1', hour='0', minute='0')
        scheduler.start()
    except BaseException as e:
        log_helper.log_out('error', local_business_logs_path, 'File: ' + e.__traceback__.tb_frame.f_globals['__file__']
                           + ', lineon: ' + str(e.__traceback__.tb_lineno) + ', error info: '
                           + str(e))
        raise


if __name__ == '__main__':
    args = parse_args()
    # gpu_num = args.gpu_num
    try:
        # 超过某时间推出程序##################
        detect_time = datetime.datetime.now().strftime('%Y-%m-%d')
        print(detect_time)
        if str(detect_time) > '2020-10-15':
            sys.exit(0)

        # 定期清理ftp上的检测图像和日志文件
        clear_folds()
        mp.set_start_method(method='spawn')  # init
        queue = mp.Queue(maxsize=10)
        processes = list()
        # 摄像头进程
        for camera_url in camera_url_list:
            processes.append(mp.Process(target=cam, args=(queue, camera_url)))
        for process in processes:
            process.daemon = True
            process.start()
        for process in processes:
            process.join()
    except BaseException as e:
        log_helper.log_out('error', local_business_logs_path, 'File: ' + e.__traceback__.tb_frame.f_globals['__file__']
                           + ', lineon: ' + str(e.__traceback__.tb_lineno) + ', error info: '
                           + str(e))
