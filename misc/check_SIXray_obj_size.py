import os
from xml.etree.ElementTree import parse

# 전체 이미지 사이즈 대비 각 클래스의 평균 크기(height x width)를 확인 
# 학습 데이터 : 7150장 
# 전체 이미지 평균 크기 : 403344 
# Gun 평균 크기      : 21822 (5%)   / 32x32기준 약 7x7
# Knife 평균 크기    : 46057 (11%)  / 32x32기준 약 10x10
# Wrench 평균 크기   : 19156 (4.7%) / 32x32기준 약 7x7
# Pliers 평균 크기   : 14698 (3.6%) / 32x32기준 약 6x6
# Scissors 평균 크기 : 6987  (1.7%) / 32x32기준 약 4x4

SIXray_ann_folder = '../SIXray/train/annotation'
xml_files = os.listdir(SIXray_ann_folder)

image_size_list = [] 
classes_size_list = {'Gun':[], 'Knife':[], 'Wrench':[], 'Pliers':[], 'Scissors':[], 'Hammer':[]}

for i, xml in enumerate(xml_files) : 
    tree = parse(os.path.join(SIXray_ann_folder, xml))
    root = tree.getroot() 

    image_width = float(root.find('size').findtext('width'))
    image_height = float(root.find('size').findtext('height'))

    image_size_list.append(image_width*image_height)

    objs = root.findall('object')
    for obj in objs :
        obj_name = obj.findtext('name')
        if obj_name == None : continue 
        
        x1 = float(obj.find('bndbox').findtext('xmin'))
        y1 = float(obj.find('bndbox').findtext('ymin'))
        x2 = float(obj.find('bndbox').findtext('xmax'))
        y2 = float(obj.find('bndbox').findtext('ymax'))

        classes_size_list[obj_name].append((x2-x1) * (y2-y1))

    if i % 200 == 0 : 
        print(f'{i}/{len(xml_files)} done')

print(f'Avg image size : {sum(image_size_list)/len(image_size_list)}')
for k, v in classes_size_list.items():
    if len(v) > 0 : 
        print(f'Avg {k} size : {sum(v)/len(v)}')
    else :
        print(f'{k} has no element')
