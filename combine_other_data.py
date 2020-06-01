import os
import shutil
import sys


def getLength(number):
    Length = 0
    while number != 0:
        Length += 1
        number = number // 10    #关键，整数除法去掉最右边的一位
    return Length


def find_images_from_main():
    targetpath = r'C:\Users\l\Desktop\voc2007data\VOCdevkit2007\VOC2007\ImageSets\Main\bird_val.txt'
    f = open(targetpath)
    file_num_deal_list = list()
    file_num_list = f.readlines()
    for file_num in file_num_list:
        sss = file_num.split(' ')[1]
        if file_num.split()[1] == '1':
            file_num_deal_list.append(file_num.split(' ')[0])
    f.close()

    filer_image_folder = r'C:\Users\l\Desktop\voc2007data\filer_images\bird_val\\'
    filepath = r'C:\Users\l\Desktop\voc2007data\VOCdevkit2007\VOC2007\JPEGImages\\'
    fileList = os.listdir(filepath)
    for fileName in fileList:
        if fileName.split('.')[0] in file_num_deal_list:
            shutil.copy(filepath+fileName, filer_image_folder + fileName)

# 根据图片文件寻找xml并复制到指定文件夹
def find_xml_file_from_images_name():
    filepath = r'C:\Users\l\Desktop\新建文件夹\aa'
    fileList = os.listdir(filepath)
    target_images_list = list()
    for fileName in fileList:
        target_images_list.append(fileName.split('.')[0])
    xmlpath = r'C:\Users\l\Desktop\新建文件夹\bb\\'
    xmltargetpath = r'C:\Users\l\Desktop\voc2007data\filter_xml\\'
    fileList1 = os.listdir(xmlpath)
    for xmlname in fileList1:
        if xmlname.split('.')[0] in target_images_list:
            shutil.copy(xmlpath + xmlname, xmltargetpath + xmlname)


# xml 和 images 文件对比看看是不是一一对应了
def xml_images_synchronization():
    images_path = r'C:\Users\l\Desktop\新建文件夹\aa'
    images_list = os.listdir(images_path)
    target_images_list = list()
    for fileName in images_list:
        target_images_list.append(fileName.split('.')[0])
    xmlpath = r'C:\Users\l\Desktop\新建文件夹\bb\\'
    xmlist = os.listdir(xmlpath)
    print('xml 不在 images里的有')
    target_xml_list = list()
    for xmlname in xmlist:
        target_xml_list.append(xmlname.split('.')[0])
        if xmlname.split('.')[0] not in target_images_list:
            print(xmlname)
    print('查找完毕')
    print('image 不在 xml里的有')
    for fileName in images_list:
        if xmlname.split('.')[0] not in target_xml_list:
            print(fileName)
    print('查找完毕')



# 同步修改图片和xml的名字
def updata_file_name():
    imagespath = r'D:\pyworkspace\Faster-RCNN-TensorFlow-Python3-master\data\vocdevkit2007_bird_filter\filer_images/'
    xmlpath = r'D:\pyworkspace\Faster-RCNN-TensorFlow-Python3-master\data\vocdevkit2007_bird_filter\filter_xml/'
    num = 3386
    plan_num_len = 6
    fileList = os.listdir(imagespath)
    for fileName in fileList:
        file_num = fileName.split('.')[0]
        print(fileName)
        len = getLength(num)
        str_len = ''
        for i in range(plan_num_len - len):
            str_len += '0'
        os.rename(imagespath+fileName, imagespath+str_len + str(num) + '.' + fileName.split('.')[-1])
        os.rename(xmlpath+file_num+'.xml', xmlpath+str_len + str(num) + '.xml')
        num = num + 1
    sys.stdin.flush()


# find_xml_file_from_images_name()
# updata_file_name()
xml_images_synchronization()