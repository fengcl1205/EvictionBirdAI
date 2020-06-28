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

capture1 = cv2.VideoCapture('rtsp://admin:123456@192.168.1.13:554')
capture2 = cv2.VideoCapture('rtsp://admin:123456@192.168.1.14:554')
capture3 = cv2.VideoCapture('rtsp://admin:123456@192.168.1.15:554')
capture4 = cv2.VideoCapture('rtsp://admin:123456@192.168.1.16:554')
width = (int(capture1.get(cv2.CAP_PROP_FRAME_WIDTH)))
height = (int(capture1.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# 暂停1秒，确保影像已经填充
time.sleep(1)
print(str(time.time()) + ' Monitoring ...')
while True:
    status, Frame = capture1.read()
    status2, Frame2 = capture2.read()
    status3, Frame3 = capture3.read()
    status4, Frame4 = capture4.read()
    # demo_video(sess, net, frame, camera_url, max_residence_frame, input_p)
    frameLeftUp = cv2.resize(Frame, (int(width//4), int(height//4)), interpolation=cv2.INTER_CUBIC)
    frameRightUp = cv2.resize(Frame2, (int(width//4), int(height//4)), interpolation=cv2.INTER_CUBIC)
    frameLeftDown = cv2.resize(Frame3, (int(width // 4), int(height // 4)), interpolation=cv2.INTER_CUBIC)
    frameRightDown = cv2.resize(Frame4, (int(width // 4), int(height // 4)), interpolation=cv2.INTER_CUBIC)
    ss1 = np.hstack((frameLeftUp, frameRightUp))
    ss2 = np.hstack((frameLeftDown, frameRightDown))
    ss = np.vstack((ss1, ss2))
    cv2.imshow('f', ss)
    key = cv2.waitKey(1)
    if key == ord('q') or key == ord('Q') or key == 27:  # ESC:27  key: quit program
        # ipcam.stop()
        break
# input_p.close()
print(str(time.time()) + ' 识别系统已关闭')
Frame.realse()
Frame2.realse()
cv2.destroyAllWindows()