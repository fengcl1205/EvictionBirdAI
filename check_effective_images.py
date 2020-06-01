
from xml.dom.minidom import parse
import xml.dom.minidom
import os

if os.path.exists(r'E:\Data\鸟类标注数据\bird_data_summary\JPEGImages\aa.txt'):
    os.remove(r'E:\Data\鸟类标注数据\bird_data_summary\JPEGImages\aa.txt')

xmlfile = r'E:\Data\鸟类标注数据\bird_data_summary\JPEGImages\008.seagull\outputs'
list_dir = os.listdir(xmlfile)
for i in list_dir:
    try:
        path = xmlfile + r"/"+i
        DOMTree = parse(path)
        collection = DOMTree.documentElement
        width = int(collection.getElementsByTagName("width")[0].childNodes[0].data)
        height = int(collection.getElementsByTagName("height")[0].childNodes[0].data)
        boxs = collection.getElementsByTagName('bndbox')
        for x in range(len(boxs)): # 检测每一个盒子
            xmin = int(boxs[x].getElementsByTagName("xmin")[0].childNodes[0].data)
            ymin = int(boxs[x].getElementsByTagName("ymin")[0].childNodes[0].data)
            xmax = int(boxs[x].getElementsByTagName("xmax")[0].childNodes[0].data)
            ymax = int(boxs[x].getElementsByTagName("ymax")[0].childNodes[0].data)
            assert width >0 and height >0 # 长宽大于0
            assert xmin >=0 and xmax >=0 and ymax >=0 and ymin >=0  #坐标大于0
            assert xmin < xmax and ymin < ymax # 左坐标小于右坐标
            assert xmax <= width and ymax <= height # 右边坐标不越界
    except:
        print('出现异常')
        with open(r'E:\Data\鸟类标注数据\bird_data_summary\JPEGImages\aa.txt', 'a') as f:
            f.write(i+'----'+ str(x)+'\r')
print('over')

