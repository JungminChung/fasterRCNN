import os
import json
import tqdm 
import random
import argparse

import torch 
import torchvision
import torch.nn as nn
from torch import optim
import models.resnet as resnet
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.detection.faster_rcnn import FasterRCNN
from models.detection.anchor_utils import AnchorGenerator

from datasetSIXray import sixrayDataset 
from utils.evaluate import evaluate, test
from utils.collators import collate_sixray
from utils.feature_extract import get_resent_features
from utils.folder_utils import check_dir, get_working_dir

from torch.nn.parallel import DataParallel as DP
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data.distributed import DistributedSampler

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=140)
    parser.add_argument('--batch_size', type=int, default=17)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.005)
    parser.add_argument('--eval_epoch', type=int, default=5)
    parser.add_argument('--save_epoch', type=int, default=5)
    parser.add_argument('--device', type=str)
    parser.add_argument('--model', type=str, default='res34')
    # parser.add_argument('--model', type=str, default='res50')

    parser.add_argument('--img_min_size', type=int, default=600)
    parser.add_argument('--img_max_size', type=int, default=1000)

    parser.add_argument('--coco_target_path', type=str, 
                        default='../SIXray/eval/coco_annotation/coco_SIXray.json')
    parser.add_argument('--coco_predic_path', type=str, 
                        default='../SIXray/eval/coco_annotation/coco_predict.json')

    parser.add_argument('--conf_threshold', type=float, default=0.3)
    
    parser.add_argument('--model_save_folder', type=str, default='weights')
    parser.add_argument('--test_img_name', type=str, default='')
    parser.add_argument('--test_img_folder', type=str, default='../SIXray/eval/image')

    return parser.parse_args()

def main():
    args = parse_args()

    working_dir = get_working_dir(args)

    # init writer for tensorboard 
    writer_dir = os.path.join('runs', working_dir)
    check_dir(writer_dir)
    writer = SummaryWriter(writer_dir)
    print(f'Tensorboard info will be saved in \'{writer_dir}\'')

    # save args in run folder 
    with open(os.path.join(writer_dir, 'args.txt'), 'w') as f: 
        json.dump(args.__dict__, f, indent=4)

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(args)

    # Dataset 
    train_dataset = sixrayDataset('../SIXray', mode='train')
    eval_dataset = sixrayDataset('../SIXray', mode='eval')

    # Dataloader 
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, 
                            shuffle=True, num_workers=0, 
                            collate_fn=collate_sixray)

    eval_loader = DataLoader(dataset=eval_dataset, batch_size=1, 
                            shuffle=False, num_workers=0, 
                            collate_fn=collate_sixray)

    # Models 
    if args.model == 'res34': 
        backbone = torchvision.models.resnet34(pretrained=True) # res 34
        out_ch = 512 # resnet18,34 : 512 
    elif args.model == 'res50':
        backbone = torchvision.models.resnet50(pretrained=True) # res 50 
        out_ch = 2048 # resnet50~152 : 2048
    else : 
        assert()

    backbone = get_resent_features(backbone)
    backbone.out_channels = out_ch

    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                        aspect_ratios=((0.5, 1.0, 2.0),))
    
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], 
                                                    output_size=7, 
                                                    sampling_ratio=2)
    model = FasterRCNN(backbone=backbone, 
                    num_classes=7, # 6 class + 1 background 
                    rpn_anchor_generator=anchor_generator,
                    box_roi_pool=roi_pooler,
                    min_size=args.img_min_size, 
                    max_size=args.img_max_size).to(args.device)

    # Optimizer 
    optimizer = optim.SGD(model.parameters(), 
                          lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    schedular = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[90, 120], gamma=0.1)
    
    # train 
    global_step = 0
    for epoch in range(1, args.epochs+1):
        progress = tqdm.tqdm(train_loader)
        for images, targets, _ in progress:
            model.train() 
            
            images = list(image.to(args.device) for image in images)
            targets = [{k: v.to(args.device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            loss_cls = loss_dict['loss_classifier'].item()
            loss_reg = loss_dict['loss_box_reg'].item()
            loss_obj = loss_dict['loss_objectness'].item()
            loss_rpn_reg = loss_dict['loss_rpn_box_reg'].item()

            progress.set_description(f'Train {epoch} / {args.epochs}, ' +
                                    f'Loss : (HC){loss_cls:.3f}, (HR){loss_reg:.3f}, ' +
                                    f'(RO){loss_obj:.3f}, (RR){loss_rpn_reg:.3f} ')
            
        if epoch % args.save_epoch == 0 : 
            save_dir = os.path.join(args.model_save_folder, working_dir)
            check_dir(save_dir)
            torch.save(model.state_dict(), 
                       os.path.join(save_dir, f'{args.model}_{epoch}.ckpt'))
                       
        if epoch % args.eval_epoch == 0 : 
            evaluate(model, eval_loader, args)
            if args.test_img_name == '': 
                image_path = os.path.join(args.test_img_folder, 
                                          random.sample(os.listdir(args.test_img_folder), 1)[0])
            test(model, image_path, args, epoch)
        
        ## Tensor Board 
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step)
        writer.add_scalar('loss_classifier', loss_cls, global_step)
        writer.add_scalar('loss_box_reg', loss_reg, global_step)
        writer.add_scalar('loss_objectness', loss_obj, global_step)
        writer.add_scalar('loss_rpn_box_reg', loss_rpn_reg, global_step)
        global_step += 1

        schedular.step()


if __name__=='__main__':
    main()
