# coding=utf-8
import os  # 打开文件时需要
from PIL import Image
import re


# 可以检测文件是否是坏文件
Start_path = r'E:\Data\鸟类标注数据\bird_data_summary\JPEGImages\person - 副本\\'  # 你的图片目录
iphone5_width = 333  # 图片最大宽度
iphone5_depth = 500  # 图片最大高度

list = os.listdir(Start_path)
# print list
count = 0
for pic in list:
    path = Start_path + pic
    print(path)
    im = Image.open(path)
    w, h = im.size
    # print w,h
    # iphone 5的分辨率为1136*640，如果图片分辨率超过这个值，进行图片的等比例压缩

    if w > iphone5_width:
        # print(pic)
        # print("图片名称为" + pic + "图片被修改")
        h_new = iphone5_width * h // w
        w_new = iphone5_width
        count = count + 1
        out = im.resize((w_new, h_new), Image.ANTIALIAS)
        new_pic = re.sub(pic[:-4], pic[:-4] + '_new', pic)
        # print new_pic
        new_path = Start_path + new_pic
        # out.save(new_path)

    if h > iphone5_depth:
        # print(pic)
        # print("图片名称为" + pic + "图片被修改")
        w = iphone5_depth * w / h
        h = iphone5_depth
        count = count + 1
        out = im.resize((w_new, h_new), Image.ANTIALIAS)
        new_pic = re.sub(pic[:-4], pic[:-4] + '_new', pic)
        # print new_pic
        new_path = Start_path + new_pic
        # out.save(new_path)

print('END')
count = str(count)
print("共有" + count + "张图片尺寸被修改")
