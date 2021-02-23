### For SIXray dataset ###
import tqdm
import json
from pycocotools import coco
from pycocotools.cocoeval import COCOeval

def evaluate(model, dataloader, args): 
    coco_ = coco.COCO(args.coco_target_path) # TODO : This makes noise on run terminal. have to extract to out side of code file 

    detections = [] 
    progress = tqdm.tqdm(dataloader)
    for images, targets, image_id in progress:
        progress.set_description('EVALUATE PROCESS ')
        images = list(image.to(args.device) for image in images)
        targets = [{k: v.to(args.device) for k, v in t.items()} for t in targets]
        
        model.eval()
        predictions = model(images)[0] # 0 means, it returns only one prediction because of 1 batch size 
        
        for i in range(len(predictions['scores'])):
            if predictions['scores'][i].item() < args.conf_threshold : continue 
            
            img_id         = int(image_id[0])
            category_id    = int(predictions['labels'][i].item())
            score          = float("{:.2f}".format(predictions['scores'][i].item()))
            x1, y1, x2, y2 = predictions['boxes'][i].tolist()
            bbox           = [x1, y1, x2-x1, y2-y1]
            
            detection = {
                "image_id": img_id,
                "category_id": category_id,
                "bbox": bbox,
                "score": score
            }
            detections.append(detection)
    
    json.dump(detections, open(args.coco_predic_path, 'w'))
    coco_dets = coco_.loadRes(args.coco_predic_path)
    coco_eval = COCOeval(coco_, coco_dets, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


def test(model, image_path, args, epoch=None):
    import os
    from PIL import Image, ImageDraw
    from torchvision import transforms
    from xml.etree.ElementTree import parse

    CLASS_NAME = ['BACKGROUND', 'Gun'   , 'Knife' , 'Wrench', 'Pliers', 'Scissors', 'Hammer']
    COLOR      = ['red'       , 'orange', 'yellow', 'green' , 'blue'  , 'navy'    , 'purple']

    print('### TEST using : ', image_path)
    image = Image.open(image_path)
    pil_img = image
    image_draw = ImageDraw.Draw(pil_img)
    image = transforms.Compose([transforms.ToTensor()])(image)
    # image = [image.to(args)]
    image = [image.to(args.device)]

    # Draw Prediction part 
    model.eval()
    predictions = model(image)[0]

    for i in range(len(predictions['scores'])):
        if predictions['scores'][i].item() < 0.20 : continue 
        # if predictions['scores'][i].item() < args.conf_threshold : continue 
        category_id    = int(predictions['labels'][i].item())
        score          = float("{:.2f}".format(predictions['scores'][i].item()))
        x1, y1, x2, y2 = predictions['boxes'][i].tolist()
        
        obj_name = CLASS_NAME[category_id]
        color = COLOR[category_id]

        image_draw.rectangle(xy=[(x1, y1), (x2, y2)], outline=color)
        image_draw.text(xy=(x1, y1-10), text=f'{obj_name}_{score}', fill=color)

    # Draw Label part 
    file_name = image_path.split('/')[-1].split('.')[0]
    xml_file = file_name + '.xml'
    xml_folder = '../SIXray/eval/annotation'
    xml_path = os.path.join(xml_folder, xml_file)

    tree = parse(xml_path)
    root = tree.getroot()
    objs = root.findall('object')
    boxes = []
    for obj in objs :
        obj_name = obj.findtext('name')
        if obj_name == None : continue 
        obj_index = CLASS_NAME.index(obj_name)
        color = 'black'

        x1 = float(obj.find('bndbox').findtext('xmin'))
        y1 = float(obj.find('bndbox').findtext('ymin'))
        x2 = float(obj.find('bndbox').findtext('xmax'))
        y2 = float(obj.find('bndbox').findtext('ymax'))

        image_draw.rectangle(xy=[(x1, y1), (x2, y2)], outline=color)
        image_draw.text(xy=(x1, y1-10), text=obj_name, fill=color)
    
    # Save part 
    if epoch is None : 
        epoch = ''
    img_save_folder = 'test_img'
    save_path = os.path.join(img_save_folder, file_name+f'_{epoch}.jpg')
    pil_img.save(save_path)