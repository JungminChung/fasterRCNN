import torch 

def collate_sixray(batch):
    # images : [batch_size, channel, height, width]
    # annots : (batch_size) [
    #              [[index, [x1, y1, x2, y2]], [index, [x1, y1, x2, y2]], [index, [x1, y1, x2, y2]]],
    #              [[index, [x1, y1, x2, y2]]],
    #              [[index, [x1, y1, x2, y2]], [index, [x1, y1, x2, y2]]] 
    #          ]
    # image_id : (batch_size) [id1, id2, ... ,idx]
    
    images, annots, image_id = zip(*batch)
    
    batch_size = len(images)
    
    images = list(image for image in images)
    max_obj_num = max(list(len(objs_per_imgs) for objs_per_imgs in annots))

    targets = []
    for i in range(batch_size):
        t = {} 
        t['boxes'] = torch.tensor([0, 0, 1, 1], dtype=torch.float32).expand(max_obj_num, 4)
        t['labels'] = torch.zeros(max_obj_num, dtype=torch.int64)

        for j in range(len(annots[i])):
            t['boxes'][j] = torch.tensor(annots[i][j][1]) # i : batch, j : obj, 1 : coordinate 
            t['labels'][j] = torch.tensor(annots[i][j][0]) # i : batch, j : obj, 0 : index    
        
        targets.append(t)

    return images, targets, image_id