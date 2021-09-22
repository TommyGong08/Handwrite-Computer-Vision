import math
import xml.etree.cElementTree as et
import os

class_num = {
    'red': 0,
    'blue': 1
}

f = open('data/mydata.txt', 'w')
# 读取xml路径
xml_path = 'data/image_voc'
xml_file_names = os.listdir(xml_path)
for xml_file_name in xml_file_names:
    xml_file_name_path = os.path.join(xml_path, xml_file_name)
    tree = et.parse(xml_file_name_path)
    root = tree.getroot()
    filename = root.find('filename')
    names = root.findall('object/name')
    boxes = root.findall('object/bndbox')
    data = []
    data.append(filename.text)
    for name, box in zip(names, boxes):
        cls = class_num[name.txt]
        x1, y1, x2, y2 = int(box[0].text), int(box[1].text), int(box[2].text), int(box[3].text)
        cx, cy, w, h = math.floor((x2 - x1) / 2), math.floor((y2 - y1) / 2), x2-x1, y2-y1
        data.append(cls)
        data.append(cx)
        data.append(cy)
        data.append(w)
        data.append(h)
    _str = ''
    for i in data:
        _str = _str + ' ' + str(i)
    f.write(_str + '\n')
f.close()
