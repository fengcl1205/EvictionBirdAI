# --** coding="UTF-8" **--
#
# author:SueMagic  time:2019-01-01
import sys
import os
import random
from xml.dom.minidom import parse
import xml.dom.minidom
import xml.etree.ElementTree as ET


def getLength(number):
    Length = 0
    while number != 0:
        Length += 1
        number = number // 10    #关键，整数除法去掉最右边的一位
    return Length


# 批量修改文件名（将图片统一命名：000001/000002 ...）
def updata_file_name():
    filepath = r'E:\Data\鸟类标注数据\bird_data_summary\JPEGImages\011.sparrow'
    fileList = os.listdir(filepath)
    # 输出此文件夹中包含的文件名称
    print("修改前：" + str(fileList)[1])
    # 得到进程当前工作目录
    currentpath = os.getcwd()
    # 将当前工作目录修改为待修改文件夹的位置
    os.chdir(filepath)
    # 名称变量
    num = 6684
    plan_num_len = 6
    # 遍历文件夹中所有文件
    for fileName in fileList:
        # 匹配文件名正则表达式
        len = getLength(num)
        str_len = ''
        for i in range(plan_num_len - len):
            str_len += '0'
        os.rename(fileName, str_len + str(num) + '.' + fileName.split('.')[-1])

        # 改变编号，继续下一项
        num = num + 1
    print("***************************************")
    # 改回程序运行前的工作目录
    os.chdir(currentpath)
    # 刷新
    sys.stdin.flush()
    # 输出修改后文件夹中包含的文件名称
    print("修改后：" + str(os.listdir(filepath))[1])


# 批量修改图片后缀
def updata_file_tail_name():
    currentpath = os.getcwd()
    filepath = r'D:\pyworkspace\Faster-RCNN-TensorFlow-Python3-master\data\VOCdevkit2007\VOC2007\JPEGImages\\'
    fileList = os.listdir(filepath)
    for fileName in fileList:
        if fileName.split('.')[-1] != 'jpg':
            new_file_name = fileName.replace(fileName.split('.')[-1], 'jpg')
            os.rename(filepath+fileName, filepath+new_file_name)

    os.chdir(currentpath)
    # 刷新
    sys.stdin.flush()
    # 输出修改后文件夹中包含的文件名称
    print("修改后：" + str(os.listdir(filepath))[1])


# 修改xml文件
def analysis_xml(opr):

    # 将其他类别改成bird
    if opr == 'up_cls_name':
        pathname3 = r'E:\Data\鸟类标注数据\bird_data_summary\JPEGImages\008.seagull - 副本\outputs\\'
        filepath = pathname3
        fileList = os.listdir(filepath)
        for fileName in fileList:
            print('up_cls_name     ' + fileName)
            doc = ET.parse(filepath+fileName)
            root = doc.getroot()
            for child in root.findall('object'):
                child.find('name').text = 'pigeon'
                # print(child.find('name').text)
            print(fileName+'   finish')
            doc.write(filepath+fileName)
    # 将xml文件中图片地址改成当前工程实际地址，文件夹名和文件名
    elif opr == 'up_path_name':
        plan_path = 'D:\pyworkspace\EvictionBirdAI\data\VOCdevkit2007\VOC2007\Annotations\\'
        images_path = 'D:\pyworkspace\EvictionBirdAI\data\VOCdevkit2007\VOC2007\JPEGImages\\'
        target_folder = plan_path.split('\\')[-3]
        fileList = os.listdir(images_path)
        for fileName in fileList:
            print('up_path_name    ' + fileName)
            fileNameHead = fileName[:-4]
            doc = ET.parse(plan_path + fileNameHead + '.xml')
            root = doc.getroot()
            root.find('path').text = images_path + fileNameHead + '.jpg'
            root.find('filename').text = fileNameHead + '.jpg'
            root.find('folder').text = target_folder
            doc.write(plan_path + fileNameHead + '.xml')


# 生成训练用的main文件夹
def create_main_fold():
    import os
    import random
    trainval_percent = 0.8
    train_percent = 0.7
    xmlfilepath = 'D:\pyworkspace\EvictionBirdAI\data\VOCdevkit2007\VOC2007\Annotations'  # 绝对路径
    txtsavepath = 'D:\pyworkspace\EvictionBirdAI\data\VOCdevkit2007\VOC2007\ImageSets\Main/'  # 生成的四个文件的存储路径
    total_xml = os.listdir(xmlfilepath)
    num = len(total_xml)
    list = range(num)
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(list, tv)
    train = random.sample(trainval, tr)
    ftrainval = open(txtsavepath+'trainval.txt', 'w')
    ftest = open(txtsavepath+'test.txt', 'w')
    ftrain = open(txtsavepath+'train.txt', 'w')
    fval = open(txtsavepath+'val.txt', 'w')
    for i in list:
        name = total_xml[i][:-4] + '\n'
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)
    print('over')
    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()


# 对齐同时修改xml 和图片的名字
def update_file_name_two_file():
    filepath_images = r'E:\Data\鸟类标注数据\bird_data_summary\JPEGImages\012.person'
    fileList_images = os.listdir(filepath_images)
    filepath = filepath_images+r'\outputs'
    fileList = os.listdir(filepath)

    print(fileList_images)
    # 输出此文件夹中包含的文件名称
    print("修改前：" + str(fileList)[1])

    # 名称变量
    num = 6835
    plan_num_len = 6
    # 遍历文件夹中所有文件
    for fileName in fileList:
        if os.path.isdir(filepath_images+'\\'+fileName):
            continue
        print(fileName)
        # 匹配文件名正则表达式
        len = getLength(num)
        str_len = ''
        for i in range(plan_num_len - len):
            str_len += '0'
        os.rename(filepath+'\\'+fileName, filepath+'\\'+str_len + str(num) + '.xml')
        if fileName[:-4]+'.jpg' in fileList_images:
            os.rename(filepath_images+'\\'+fileName[:-4]+'.jpg', filepath_images+'\\'+str_len + str(num) + '.jpg')
        else:
            print()

        # 改变编号，继续下一项
        num = num + 1
    print("***************************************")
    # 改回程序运行前的工作目录
    # 刷新
    sys.stdin.flush()
    # 输出修改后文件夹中包含的文件名称
    print("修改后：" + str(os.listdir(filepath))[1])


# createimageSets()
# updata_file_name()
# update_file_name_two_file()
# analysis_xml('up_cls_name')
# create_main_fold()

#流程
create_main_fold()
# analysis_xml('up_cls_name')
analysis_xml('up_path_name')