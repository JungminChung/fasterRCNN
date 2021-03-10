import os
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
import pycocotools.coco as coco
from torchvision import transforms
import numpy as np

class cocoDataset(Dataset):
    def __init__(self, coco_folder, mode):
        super(cocoDataset, self).__init__()
        self.coco_folder = coco_folder
        self.mode = mode # 'train' or 'eval'
        self.split = 'train2017' if self.mode == 'train' else 'val2017'

        self.image_folder = os.path.join(self.coco_folder, 'images', self.split)
        self.annotation_folder = os.path.join(self.coco_folder, 'annotations')
        self.annotation_path = os.path.join(self.annotation_folder, f'instances_{self.split}.json')

        self.transform = transforms.Compose([
                            transforms.ToTensor(),
                        ])

        self.max_objs = 128
        self.class_name = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
        'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush']

        self._valid_ids = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 
        14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
        24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 
        37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 
        58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 
        72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 
        82, 84, 85, 86, 87, 88, 89, 90]

        print('==> initializing coco {} data.'.format(self.split))
        self.coco = coco.COCO(self.annotation_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)

        print('Loaded {} {} samples'.format(self.split, self.num_samples))
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.image_folder, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)
    
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        boxes = [] 
        for i in range(num_objs):
            ann = anns[i]
            box = ann['bbox'] # x1, y1, w, h
            if box[2] <= 0 or box[3] <= 0 : 
                continue 
            box = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
            obj_index = ann['category_id']
            
            boxes.append([obj_index, box])

        return image, boxes, img_id