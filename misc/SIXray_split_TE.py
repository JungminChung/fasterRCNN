import os
import random
import shutil
from xml.etree.ElementTree import parse

sixray_path = '../../SIXray'
image_path = '../../SIXray/origin/image'
annotation_path = '../../SIXray/origin/annotation'

images = os.listdir(image_path)
annotations = os.listdir(annotation_path)

for i, ann in enumerate(annotations) : 
    annotation_file = os.path.join(annotation_path, ann)
    rand = random.random()
    go_to = 'train' if rand > 0.2 else 'eval'
    
    # xml part 
    tree = parse(annotation_file)
    root = tree.getroot()

    image_filename = root.findall("filename")[0].text # filename's num is always one 
    image_file = os.path.join(image_path, image_filename)

    # copy part 
    shutil.copy(annotation_file, 
                os.path.join(sixray_path, go_to, 'annotation', ann))
    shutil.copy(image_file,
                os.path.join(sixray_path, go_to, 'image', image_filename))

    if i % 200 == 0 : 
        print(i, '/', len(annotations))