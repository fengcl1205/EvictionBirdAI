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
# CLASSES = ('__background__',  'crow', 'magpie', 'pigeon', 'swallow', 'sparrow', 'airplane',  'person')
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


# ftp模块
class QnFtp:
    def __init__(self):
        self.ip = business_path_config['ftp_ip_port_user_passwd'][0]
        self.port = business_path_config['ftp_ip_port_user_passwd'][1]
        self.user = business_path_config['ftp_ip_port_user_passwd'][2]
        self.passwd = business_path_config['ftp_ip_port_user_passwd'][3]
        self.ftp_cap_video_path = business_path_config['ftp_cap_video_path']
        self.bufsize = 4096
        self.ftp = FTP()
        self.ftp.set_debuglevel(2)
        # 登录
        try:
            self.ftp.connect(self.ip, self.port)
            self.ftp.login(self.user, self.passwd)
            log_helper.log_out('info', local_business_logs_path, ' ftp登录成功')
        except BaseException as e:
            log_helper.log_out('error', local_business_logs_path,
                               'File: ' + e.__traceback__.tb_frame.f_globals['__file__']
                               + ', lineon: ' + str(e.__traceback__.tb_lineno) + ', error info: '
                               + str(e))
            exit()  # 停止程序
            raise

    # 上传
    def upload(self, detect_time, images):
        try:
            time_ymd = str(detect_time).split()[0]  # 以日为单位存储
            self.judge_ftp_path(time_ymd)
            self.ftp.storbinary('STOR %s' % os.path.basename(detect_time), images, self.bufsize)  # 上传文件
        except BaseException as e:
            log_helper.log_out('error', local_business_logs_path,
                           'File: ' + e.__traceback__.tb_frame.f_globals['__file__']
                           + ', lineon: ' + str(e.__traceback__.tb_lineno) + ', error info: '
                           + str(e))
            # exit()  # 停止程序
            raise

        log_helper.log_out('info', local_business_logs_path,
                           str(detect_time) + ' 上传成功')

    # 判断ftp路径是否存在
    def judge_ftp_path(self, path):
        try:
            self.ftp.cwd(path)
        except error_perm:
            try:
                self.ftp.mkd(path)
            except BaseException as e:
                log_helper.log_out('error', local_business_logs_path,
                           'File: ' + e.__traceback__.tb_frame.f_globals['__file__']
                           + ', lineon: ' + str(e.__traceback__.tb_lineno) + ', error info: '
                           + str(e))
                raise

    # 查询并删除指定天数前的视频捕捉图像
    def clear_cap_img_dir(self, current_time_y, current_time_m, current_time_d):
        try:
            self.ftp.cwd(self.ftp_cap_video_path)
        except:
            self.ftp.mkd(self.ftp_cap_video_path)
        try:
            files = self.ftp.nlst()
            for file in files:  # 例2020-06-12
                # 满足条件则删除目录
                if datetime.datetime(current_time_y, current_time_m, current_time_d) - \
                        datetime.datetime(int(file.split('-')[0]), int(file.split('-')[1]), int(file.split('-')[2])) \
                        > ftp_images_retain_time:
                    self.ftp.delete(file)
        except BaseException as e:
            log_helper.log_out('error', local_business_logs_path,
                               'File: ' + e.__traceback__.tb_frame.f_globals['__file__']
                               + ', lineon: ' + str(e.__traceback__.tb_lineno) + ', error info: '
                               + str(e))
            raise

    # 退出
    def ftp_quit(self):
        self.ftp.set_debuglevel(0)
        self.ftp.quit()


# 检测到目标的图像并且持久化
def detection_persistence(detect_time, images):
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
        files = os.listdir(local_cap_video_path)
        for file in files:  # 例2020-06-12
            file_path = os.path.join(local_cap_video_path, file)
            if os.path.isdir(file_path):  # 日志是按月为单位目录存储的
                if (current_time_y - int(file.split('-')[0])) * 12 + (current_time_m - int(file.split('-')[1]))\
                        > ftp_images_retain_time:
                    shutil.rmtree(os.path.join(local_cap_video_path, file))
    except BaseException as e:
        log_helper.log_out('error', local_cap_video_path,
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


# 发送消息到应用系统
def socket_client_target_detection(detect_cls, detect_num, detect_img, detect_time, camera_number, disperse_sign):
    # create socket
    tcp_client_socket = socket(AF_INET, SOCK_STREAM)
    try:
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
    except BaseException as e:
        log_helper.log_out('error', local_business_logs_path,
                       'File: ' + e.__traceback__.tb_frame.f_globals['__file__']
                       + ', lineon: ' + str(e.__traceback__.tb_lineno) + ', error info: '
                       + str(e))
        raise
    finally:
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
        if class_name in target_detect_categories:
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
        else:
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


# 摄像头视频检测
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
    CONF_THRESH = detect_threshold  # threshold
    NMS_THRESH = nms_threshold
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        if cls in target_detect_categories and len(inds) != 0:
            residence_frame += 1
            if residence_frame == max_residence_frame:
                images = vis_detections_video(im, cls, dets, timer.start_time, timer.total_time, inds, CONF_THRESH)
                detect_time = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S.%f')
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
    while True:
        frame = ipcam.get_frame()
        demo_video(sess, net, frame, camera_url, max_residence_frame)
        key = cv2.waitKey(1)
        if key == ord('q') or key == ord('Q') or key == 27:  # ESC:27  key: quit program
            ipcam.stop(camera_url)
            break
    # print(str(time.time()) + ' 识别系统已关闭')
    log_helper.log_out('info', local_business_logs_path,
                       str(time.time()) + ' 识别系统已关闭')
    cv2.destroyAllWindows()


# 清空指定日期前的ftp上的捕获图像和本地的日志
def clear_folds():
    qn_ftp = QnFtp()
    try:
        current_time_y = datetime.datetime.now().strftime('%Y')
        current_time_m = datetime.datetime.now().strftime('%m')
        current_time_d = datetime.datetime.now().strftime('%d')
        scheduler = BackgroundScheduler()
        scheduler.add_job(func=clear_local_business_logs, args=(current_time_y, current_time_m, current_time_d),
                          trigger='cron', month='*', day='*', hour='0', minute='0')
        scheduler.add_job(func=clear_local_business_logs, args=(current_time_y, current_time_m, current_time_d),
                          trigger='cron', month='*', day='1', hour='0', minute='0')
        scheduler.start()
    except BaseException as e:
        log_helper.log_out('error', local_business_logs_path, 'File: ' + e.__traceback__.tb_frame.f_globals['__file__']
                           + ', lineon: ' + str(e.__traceback__.tb_lineno) + ', error info: '
                           + str(e))
        raise
    finally:
        qn_ftp.ftp_quit()


if __name__ == '__main__':
    args = parse_args()
    # gpu_num = args.gpu_num

    try:
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
