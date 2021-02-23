import os 
import json
from xml.etree.ElementTree import parse

eval_annotation_folder = '../../SIXray/eval/annotation'
coco_annotation_folder = '../../SIXray/eval/coco_annotation'

xmls = os.listdir(eval_annotation_folder)

CLASS_NAME = ['BACKGROUND','Gun', 'Knife', 'Wrench', 'Pliers', 'Scissors', 'Hammer']

categories = [
    {'id' : 0, 'name' : 'BACKGROUND'}, 
    {'id' : 1, 'name' : 'Gun'}, 
    {'id' : 2, 'name' : 'Knife'}, 
    {'id' : 3, 'name' : 'Wrench'}, 
    {'id' : 4, 'name' : 'Pliers'}, 
    {'id' : 5, 'name' : 'Scissors'}, 
    {'id' : 6, 'name' : 'Hammer'}, 
]

images = [] 
annotations = [] 
annotation_id = 0
for i, xml in enumerate(xmls): 
    xml_file = os.path.join(eval_annotation_folder, xml)
    tree = parse(xml_file)
    root = tree.getroot()

    image_filename = root.findall("filename")[0].text 
    image_id = int(image_filename[1:].split('.jpg')[0])
    width = root.find('size').findtext('width')
    height = root.find('size').findtext('height')
    path = ''

    images.append({
        'id' : image_id, 
        'dataset_id' : 0, 
        'path': path,
        'width': width,
        'height': height,
        'file_name': image_filename
    })

    objs = root.findall('object')
    for obj in objs : 
        obj_name = obj.findtext('name')
        if obj_name == None : continue 
        category_id = CLASS_NAME.index(obj_name)
        x1 = float(obj.find('bndbox').findtext('xmin'))
        y1 = float(obj.find('bndbox').findtext('ymin'))
        x2 = float(obj.find('bndbox').findtext('xmax'))
        y2 = float(obj.find('bndbox').findtext('ymax'))
        
        x = x1
        y = y1 
        # x = (x1+x2)/2
        # y = (y1+y2)/2 
        w = x2-x1
        h = y2-y1

        bbox = [x, y, w, h]

        annotations.append({
            'id': annotation_id , 
            'image_id': image_id,
            'category_id': category_id,
            'segmentation': [],
            'area': 0,
            'bbox': bbox,
            'iscrowd': False,
            'color': ''
        })
        annotation_id += 1
    
    if i % 200 == 0 : 
        print(f'{i}/{len(xmls)}')

json_file = {
    'images' : images, 
    'categories': categories, 
    'annotations': annotations
}

with open(os.path.join(coco_annotation_folder, 'coco_SIXray.json'), 'w') as f:
    json.dump(json_file, f)
