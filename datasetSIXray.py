import os
from PIL import Image
from xml.etree.ElementTree import parse
from torch.utils.data.dataset import Dataset
from torchvision import transforms

class sixrayDataset(Dataset):
    CLASS_NAME = ['BACKGROUND','Gun', 'Knife', 'Wrench', 'Pliers', 'Scissors', 'Hammer']
    def __init__(self, sixray_folder, mode):
        self.sixray_folder = sixray_folder
        self.mode = mode # 'train' or 'eval'

        self.image_folder = os.path.join(self.sixray_folder, self.mode, 'image')
        self.annotation_folder = os.path.join(self.sixray_folder, self.mode, 'annotation')

        self.transform = transforms.Compose([
                            transforms.ToTensor(),
                        ])

    def __len__(self):
        return len(os.listdir(self.annotation_folder))
    
    def __getitem__(self, index):
        annotation_file_name = os.listdir(self.annotation_folder)[index]
        annotation_file = os.path.join(self.annotation_folder, annotation_file_name)

        tree = parse(annotation_file)
        root = tree.getroot()
        
        # image info
        image_filename = root.findtext("filename")
        image_width = root.find('size').findtext('width')
        image_height = root.find('size').findtext('height')
        
        image_file = os.path.join(self.image_folder, image_filename)
        image = Image.open(image_file)
        image = self.transform(image)

        # object info 
        objs = root.findall('object')
        boxes = []
        for obj in objs :
            obj_name = obj.findtext('name')
            if obj_name == None : continue 
            obj_index = sixrayDataset.CLASS_NAME.index(obj_name)
            
            x1 = float(obj.find('bndbox').findtext('xmin'))
            y1 = float(obj.find('bndbox').findtext('ymin'))
            x2 = float(obj.find('bndbox').findtext('xmax'))
            y2 = float(obj.find('bndbox').findtext('ymax'))
            box = [x1, y1, x2, y2]
            
            boxes.append([obj_index, box])
        
        image_id = int(image_filename[1:].split('.jpg')[0]) # Only use for evaluate 

        return image, boxes, image_id