#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
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
from apscheduler.schedulers.background import BackgroundScheduler
from business.utils import path_helper as ph
from business.utils import log_helper
import queue


project_address = ph.get_local_project_path(os.path.dirname(os.path.abspath(__file__)), 0)
business_path_config = yaml_helper.get_data_from_yaml(project_address + '/business/config/business_config.yaml')
detect_categories_config = yaml_helper.get_data_from_yaml(project_address + '/business/config/detect_cls.yaml')
eliminate_misjudgment = yaml_helper.get_data_from_yaml(project_address + '/business/config/eliminate_misjudgment.yaml')
# 触发报警的最大"连续"识别次数
max_lazy_frequency = business_path_config['max_lazy_frequency']
# # 同一摄像头触发驱赶后的不工作时间(秒)
dispersed_rest_time = business_path_config['dispersed_rest_time']
# 网络摄像头
camera_url_list = business_path_config['camera_url']
# 网络摄像头名称
camera_name_list = business_path_config['camera_name']
ftp_images_retain_time = business_path_config['ftp_images_retain_time']
local_business_logs_path = business_path_config['local_business_logs_path']
local_business_logs_retain_time = business_path_config['local_business_logs_retain_time']
print_console_flag = business_path_config['print_console_flag']
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


class socket_c:
    def __init__(self):
        self.tcp_client_socket = socket(AF_INET, SOCK_STREAM)
        self.server_ip = business_path_config['application_system_ip_port'][0]
        self.server_port = int(business_path_config['application_system_ip_port'][1])
        self.send_data = dict()
        self.conn_status = 0

    def socket_conn(self):
        try:
            # connet servier
            self.tcp_client_socket.connect((self.server_ip, self.server_port))
            self.conn_status = 1
        except BaseException as e:
            log_helper.log_out('error', 'File: ' + e.__traceback__.tb_frame.f_globals['__file__']
                       + ', lineon: ' + str(e.__traceback__.tb_lineno) + ', error info: ' + str(e))
            self.tcp_client_socket.close()
            self.conn_status = 0

    def get_conn_status(self):
        return self.conn_status

    def socket_cend(self, target_info, detect_img, detect_time, camera_number, disperse_sign):
        try:
            if disperse_sign == '1' and len(target_info) != 0:
                self.send_data = {"disperseSign": disperse_sign, "detectTargetInfo": target_info,
                                  "detectTime": detect_time, "cameraNumber": camera_number}
                self.tcp_client_socket.send((str(self.send_data) + '##').encode())
        except BaseException as e:
            log_helper.log_out('error', 'File: ' + e.__traceback__.tb_frame.f_globals['__file__']
                       + ', lineon: ' + str(e.__traceback__.tb_lineno) + ', error info: ' + str(e))

    def socket_recv(self):
        recvData = self.tcp_client_socket.recv(1024)
        return recvData.decode()

    def socket_clse(self):
        self.tcp_client_socket.close()


def release_camera(cap):
    cap.release()


def image_put(queue, queue1, camera_url):
    # continuous_interruption_count = 0
    log_helper.log_out('info', str(time.time()) + ' ' + camera_url + ' 加载影像数据')
    try:
        release_camera_tiem = time.time()
        capture = cv2.VideoCapture(camera_url)
        while True:
            status, frame = capture.read()
            if not status:
                # continuous_interruption_count += 1
                log_helper.log_out('info', camera_url + ' 视频流发现问题，矫正中...')
                capture = cv2.VideoCapture(camera_url)
                status, frame = capture.read()
                # if not status:
                #     continuous_interruption_count += 1
            # else:
            #     continuous_interruption_count = 0
            # 连续中断指定帧数则退出
            # if continuous_interruption_count >= 10:
            #     raise IOError(str(camera_url) + ' 摄像头发生异常而中断！')
            if time.time() - release_camera_tiem > 30:
                print(camera_url+' 释放摄像头中。。。')
                release_camera(capture)
                release_camera_tiem = time.time()
                capture = cv2.VideoCapture(camera_url)
                status, frame = capture.read()
            queue.put(frame)
            if queue.qsize() > 1:
                queue.get()
            else:
                time.sleep(0.04)
            if queue1.qsize() != 0:
                return
        print('加载数据模块退出')
        # capture.release()
    except BaseException as e:
        log_helper.log_out('error', 'File: ' + e.__traceback__.tb_frame.f_globals['__file__']
                       + ', lineon: ' + str(e.__traceback__.tb_lineno) + ', error info: '+ str(e))
        raise


# 检测到目标的图片并且持久化
def detection_persistence(detect_time, images, camera_url):
    try:
        path = local_cap_video_path + '/' + camera_url + '/' + detect_time.split()[0]
        if not os.path.exists(path):
            os.makedirs(path)
        cv2.imwrite(path + '/' + detect_time + '.jpg', images)
    except BaseException as e:
        log_helper.log_out('error', 'File: ' + e.__traceback__.tb_frame.f_globals['__file__']
                           + ', lineon: ' + str(e.__traceback__.tb_lineno) + ', error info: ' + str(e))
        raise


# 检测到目标的视频并且持久化
def detection_video_persistence(detect_time, images):
    try:
        path = local_cap_video_path + '/' + detect_time.split()[0]
        if not os.path.exists(path):
            os.makedirs(path)
        cv2.imwrite(path + '/' + detect_time + '.jpg', images)
        log_helper.log_out('info', str(time.time()) + ' ' + detect_time + '.jpg' + ' 目标图片持久化完成')
    except BaseException as e:
        log_helper.log_out('error', 'File: ' + e.__traceback__.tb_frame.f_globals['__file__']
                           + ', lineon: ' + str(e.__traceback__.tb_lineno) + ', error info: ' + str(e))
        raise


# 清空本地指定日期范围外的日志文件
def clear_local_business_logs(current_time_y, current_time_m, current_time_d):
    try:
        data_folder_path = os.listdir(local_business_logs_path)
        for date_folder in data_folder_path:
            if (datetime.date(int(current_time_y), int(current_time_m), int(current_time_d)) -
                datetime.date(int(date_folder.split('-')[0]), int(date_folder.split('-')[1]),
                              int(date_folder.split('-')[2][:5]))).days > ftp_images_retain_time:
                shutil.rmtree(os.path.join(local_business_logs_path, date_folder))
                log_helper.log_out('info', str(time.time()) + ' 日志清理完成')
    except BaseException as e:
        log_helper.log_out('error', 'File: ' + e.__traceback__.tb_frame.f_globals['__file__']
                       + ', lineon: ' + str(e.__traceback__.tb_lineno) + ', error info: ' + str(e))
        raise


# 清空本地指定日期范围外的捕获图像
def clear_local_capture_images(current_time_y, current_time_m, current_time_d):
    try:
        ip_layer = os.listdir(local_cap_video_path)
        for ip_folder in ip_layer:
            ip_folder_path = os.path.join(local_cap_video_path, ip_folder)
            if os.path.isdir(ip_folder_path):
                date_layer = os.listdir(ip_folder_path)
                for date_folder in date_layer:
                    if (datetime.date(int(current_time_y), int(current_time_m), int(current_time_d)) -
                        datetime.date(int(date_folder.split('-')[0]), int(date_folder.split('-')[1]),
                                      int(date_folder.split('-')[2]))).days > ftp_images_retain_time:
                        shutil.rmtree(os.path.join(ip_folder_path, date_folder))
                        log_helper.log_out('info', str(time.time()) + ' 检测目标图像清理完成')
    except BaseException as e:
        log_helper.log_out('error', 'File: ' + e.__traceback__.tb_frame.f_globals['__file__']
                           + ', lineon: ' + str(e.__traceback__.tb_lineno) + ', error info: ' + str(e))
        raise


# 系统参数解析
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='disperse bird apply ')
    parser.add_argument('--gpu_num', dest='gpu_num', help='please input gpu number')
    parser.add_argument('--recognition_accuracy', dest='recognition_accuracy', help='please input recognition_accuracy')
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
            # 增加误判（可以用的，只是暂时不用）
            # 如果坐标值在误判区则不显示
            '''
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
            '''
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


# 季节性强制排除
def seasonal_target_exclusion(detect_time, target):
    month = detect_time.split(' ')[0].split('-')[1]
    if month == '03' or month == '04' or month == '05':  # 春
        return target
    elif month == '06' or month == '07' or month == '08' or month == '09':  # 夏
        if target == 'crow':
            return 'swallow'
        else:
            return target
    elif month == '10' or month == '11':  # 秋
        return target
    elif month == '12' or month == '01' or month == '02':  # 冬
        return target


# 摄像头视频检测
def demo_video(sess, net, frame, camera_url, lazy_frequency, dispersed_time):
    """Detect object classes in an image usi， ng pre-computed object proposals."""
    im = frame
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    images = frame
    # socket 和图片存储路径以ip命名
    camera_url_fold_name = camera_url.split('@')[1].split(':')[0]
    # 在一针图像中是否找到目标
    find_target_flag = False
    # 一帧频图像中每一类目标个数
    target_info = dict()
    detect_time = 0.
    # Visualize detections for each class
    CONF_THRESH = detect_threshold  # threshold
    NMS_THRESH = nms_threshold
    sc = socket_c()
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
            video_name = camera_name_list[camera_url_list.index(camera_url)]
            video_name = video_name.split('##')
            coordinate = video_name[1].split('-')
            cv2.imshow(video_name[0], frame)
            cv2.moveWindow(video_name[0], int(coordinate[0]), int(coordinate[1]))
            detect_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            if len(inds) != 0 and cls in TARGET_CLASSES and (len(inds) - len(invalid_target_ele)) != 0:
                find_target_flag = True
                # target_info[cls] = len(inds) - len(invalid_target_ele)
                tran_cls = seasonal_target_exclusion(detect_time, cls)
                target_info[tran_cls] = valid_target_info

        if find_target_flag:
            lazy_frequency += 1
        else:
            lazy_frequency = 0
        if (lazy_frequency == max_lazy_frequency) and ((time.time() - dispersed_time) > dispersed_rest_time):
            dispersed_time = time.time()
            # 多线程检测目标写入磁盘
            t_persistence = threading.Thread(target=detection_persistence, daemon=True, args=(
                detect_time, images, camera_url_fold_name)).start()
            # t_persistence.join()  # 设置主线程等待子线程结束
            sc.socket_conn()
            if sc.get_conn_status() == 1:
                sc.socket_cend(target_info, '', detect_time, camera_url_fold_name, '1')
            lazy_frequency = 0
        return lazy_frequency, dispersed_time
    except BaseException as e:
        log_helper.log_out('error', 'File: ' + e.__traceback__.tb_frame.f_globals['__file__']
                           + ', lineon: ' + str(e.__traceback__.tb_lineno) + ', error info: ' + str(e))
        raise
    finally:
        sc.socket_clse()


# 相机触发函数
def cam(queue, queue1, camera_url):
    demonet = 'vgg16'
    tfmodel = project_address + '/default/voc_2007_trainval/default_bird/vgg16_faster_rcnn_iter_300000.ckpt'
    if not os.path.isfile(tfmodel + '.meta'):
        log_helper.log_out('info', tfmodel)
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
    try:
        n_classes = len(CLASSES)
        # create the structure of the net having a certain shape (which depends on the number of classes)
        net.create_architecture(sess, "TEST", n_classes,
                                tag='default', anchor_scales=[8, 16, 32, 64])
        saver = tf.train.Saver()
        saver.restore(sess, tfmodel)
        print('Loaded network {:s}'.format(tfmodel))
        log_helper.log_out('info', 'Loaded network {:s}'.format(tfmodel))
        log_helper.log_out('info', str(time.time()) + ' Monitoring ...')
        # 发现目标的连续次数
        lazy_frequency = 0
        # 记录上次触发驱鸟跑时间
        dispersed_time = 0.
        # 休眠一会以保证数据先加入缓冲区
        # time.sleep(2)
        while True:
            frame = queue.get()
            lazy_frequency, dispersed_time = demo_video(sess, net,  frame, camera_url, lazy_frequency, dispersed_time)
            # cv2.imshow('',frame)
            # time.sleep(0.01)
            key = cv2.waitKey(1)
            if key == ord('q') or key == ord('Q') or key == 27:  # ESC:27  key: quit program
                break
        print(str(time.time()) + ' ' + camera_url + ' 识别系统已关闭')
        log_helper.log_out('info', str(time.time()) + ' ' + camera_url + ' 识别系统已关闭')
        return
    except BaseException as e:
        log_helper.log_out('error', camera_url + 'File: ' + e.__traceback__.tb_frame.f_globals['__file__'] +
                           ', lineon: ' + str(e.__traceback__.tb_lineno) + ', error info: ' + str(e))
        raise
    finally:
        cv2.destroyAllWindows()
        sess.close()
        queue1.put(0)


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
                          trigger='cron', month='*', day='*', hour='0', minute='0')
        scheduler.start()
    except BaseException as e:
        log_helper.log_out('error', 'File: ' + e.__traceback__.tb_frame.f_globals['__file__']
                           + ', lineon: ' + str(e.__traceback__.tb_lineno) + ', error info: ' + str(e))
        raise


def fun1():
    camera_url1 = 'rtsp://admin:123456@192.168.1.14:554'
    queue_0 = queue.Queue(maxsize=2)
    queue1_0 = queue.Queue(maxsize=1)
    threading.Thread(target=image_put, args=(queue_0, queue1_0, camera_url1)).start()
    threading.Thread(target=cam, args=(queue_0, queue1_0, camera_url1)).start()


def fun2():
    camera_url2 = 'rtsp://admin:123456@192.168.1.15:554'

    queue_1 = queue.Queue(maxsize=2)
    queue1_1 = queue.Queue(maxsize=1)
    threading.Thread(target=image_put, args=(queue_1, queue1_1, camera_url2)).start()
    threading.Thread(target=cam, args=(queue_1, queue1_1, camera_url2)).start()


def fun3():
    camera_url3 = 'rtsp://admin:123456@192.168.1.16:554'
    queue_2 = queue.Queue(maxsize=2)
    queue1_2 = queue.Queue(maxsize=1)
    threading.Thread(target=image_put, args=(queue_2, queue1_2, camera_url3)).start()
    threading.Thread(target=cam, args=(queue_2, queue1_2, camera_url3)).start()


def fun4():
    camera_url4 = 'rtsp://admin:123456@192.168.1.17:554'
    queue_3 = queue.Queue(maxsize=2)
    queue1_3 = queue.Queue(maxsize=1)
    threading.Thread(target=image_put, args=(queue_3, queue1_3, camera_url4)).start()
    threading.Thread(target=cam, args=(queue_3, queue1_3, camera_url4)).start()


def fun(queue1, queue2, camera_url):
    threading.Thread(target=image_put, args=(queue1, queue2, camera_url)).start()
    threading.Thread(target=cam, args=(queue1, queue2, camera_url)).start()


if __name__ == '__main__':
    args = parse_args()
    # print(args.recognition_accuracy)
    # gpu_num = args.gpu_num
    try:
        detect_time = datetime.datetime.now().strftime('%Y-%m-%d')
        print(detect_time)
        if str(detect_time) > '2020-11-01':
            sys.exit(0)
        # 定期清理ftp上的检测图像和日志文件
        clear_folds()
        mp.set_start_method(method='spawn')  # init
        processes = list()
        # 摄像头进程
        # aa = [0]
        # queue = mp.Queue(maxsize=2)
        # queue1 = mp.Queue(maxsize=1)
        # camera_url = 'rtsp://admin:123456@192.168.1.14:554'
        # for ele in aa:
        #     processes.append(mp.Process(target=image_put, args=(queue, queue1, camera_url,ele)))
        #     processes.append(mp.Process(target=cam, args=(queue, queue1, camera_url)))

        '''
        camera_url1 = 'rtsp://admin:123456@192.168.1.16:554'
        camera_url2 = 'rtsp://admin:123456@192.168.1.17:554'
        queue_0 = queue.Queue(maxsize=2)
        queue1_0 = queue.Queue(maxsize=1)
        threading.Thread(target=image_put, args=(queue_0, queue1_0, camera_url1)).start()
        threading.Thread(target=cam, args=(queue_0, queue1_0, camera_url1)).start()

        queue_1 = queue.Queue(maxsize=2)
        queue1_1 = queue.Queue(maxsize=1)
        threading.Thread(target=image_put, args=(queue_1, queue1_1, camera_url2)).start()
        threading.Thread(target=cam, args=(queue_1, queue1_1, camera_url2)).start()
        '''
        # 多进程
        for camera_url in camera_url_list:
            if camera_url == r'rtsp://admin:123456@192.168.1.14:554':
                processes.append(mp.Process(target=fun1, args=()))
            if camera_url == r'rtsp://admin:123456@192.168.1.15:554':
                processes.append(mp.Process(target=fun2, args=()))
            if camera_url == r'rtsp://admin:123456@192.168.1.16:554':
                processes.append(mp.Process(target=fun3, args=()))
            if camera_url == r'rtsp://admin:123456@192.168.1.17:554':
                processes.append(mp.Process(target=fun4, args=()))
        for process in processes:
            process.daemon = True
            process.start()
        for process in processes:
            process.join()

        '''
        for index in range(len(camera_url_list)):
            locals()['queue_' + str(index)] = queue.Queue(maxsize=2)
            locals()['queue1_' + str(index)] = queue.Queue(maxsize=1)
            if index == 0:
                processes.append(mp.Process(target=fun, args=(queue_0, queue1_0, camera_url_list[index])))
            elif index == 1:
                processes.append(mp.Process(target=fun, args=(queue_1, queue1_1, camera_url_list[index])))
            elif index == 2:
                processes.append(mp.Process(target=fun, args=(queue_2, queue1_2, camera_url_list[index])))
            elif index == 3:
                processes.append(mp.Process(target=fun, args=(queue_3, queue1_3, camera_url_list[index])))
            elif index == 4:
                processes.append(mp.Process(target=fun, args=(queue_4, queue1_4, camera_url_list[index])))
            elif index == 5:
                processes.append(mp.Process(target=fun, args=(queue_5, queue1_5, camera_url_list[index])))

        for process in processes:
            process.daemon = True
            process.start()
        for process in processes:
            process.join()
        '''
    except BaseException as e:
        log_helper.log_out('error', 'File: ' + e.__traceback__.tb_frame.f_globals['__file__']
                           + ', lineon: ' + str(e.__traceback__.tb_lineno) + ', error info: ' + str(e))
