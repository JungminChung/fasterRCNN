import os
import json
import tqdm 
import random
import argparse
import math

import torch 
import torchvision
import torch.nn as nn
from torch import optim
import models.resnet as resnet
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.detection.faster_rcnn import FasterRCNN
from models.detection.anchor_utils import AnchorGenerator
from models.detection.faster_rcnn import fasterrcnn_resnet50_fpn

from datasets.SIXray import sixrayDataset 
from datasets.coco import cocoDataset
from utils.evaluate import evaluate, test
from utils.collators import collate_sixray
from utils.feature_extract import get_resent_features
from utils.folder_utils import check_dir, get_working_dir_name, make_results_folders

from torch.nn.parallel import DataParallel as DP
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data.distributed import DistributedSampler

from models.AAA import AAA 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=140)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.005)
    parser.add_argument('--eval_epoch', type=int, default=5)
    parser.add_argument('--save_epoch', type=int, default=5)
    parser.add_argument('--device', type=str)
    parser.add_argument('--dataset', type=str, default='sixray', help='sixray / coco')
    parser.add_argument('--model', type=str, default='res34')
    parser.add_argument('--results_root', type=str, default='runs')
    parser.add_argument('--anchor_init_size', type=int, default=16)

    parser.add_argument('--img_min_size', type=int, default=600)
    parser.add_argument('--img_max_size', type=int, default=1000)

    parser.add_argument('--coco_target_path', type=str, 
                        default='../SIXray/eval/coco_annotation/coco_SIXray.json')
    parser.add_argument('--coco_predict_name', type=str, default='coco_predict.json')

    parser.add_argument('--conf_threshold', type=float, default=0.3)
    
    parser.add_argument('--model_save_folder', type=str, default='weights')
    parser.add_argument('--test_img_name', type=str, default='')
    parser.add_argument('--test_img_folder', type=str, default='../SIXray/eval/image')

    # attention 
    parser.add_argument('--intra_identity', action='store_true')
    parser.add_argument('--inter_identity', action='store_true')
    
    parser.add_argument('--mix_inter_intra', type=str, default='avg', 
                        help='avg / else')

    # test 
    parser.add_argument('--_draw_atten_img', action='store_true')

    return parser.parse_args()

def main():
    args = parse_args()

    working_dir_name = get_working_dir_name(args.results_root, args)
    working_dir = os.path.join(args.results_root, working_dir_name)
    check_dir(working_dir)
    print(f'Working Dir : {working_dir}')
    make_results_folders(working_dir) # 'weights' / 'test_img'

    # update args 
    args.working_dir = working_dir
    args.weights_dir = os.path.join(args.working_dir, 'weights')
    args.test_img_dir = os.path.join(args.working_dir, 'test_img')

    # init writer for tensorboard 
    writer = SummaryWriter(working_dir)
    print(f'Tensorboard info will be saved in \'{working_dir}\'')

    # save args in run folder 
    with open(os.path.join(working_dir, 'args.txt'), 'w') as f: 
        json.dump(args.__dict__, f, indent=4)

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(args)

    # Dataset 
    if args.dataset == 'sixray':
        train_dataset = sixrayDataset('../SIXray', mode='train')
        eval_dataset = sixrayDataset('../SIXray', mode='eval')
    elif args.dataset ==  'coco' :
        train_dataset = cocoDataset('../coco', mode='train')
        eval_dataset = cocoDataset('../coco', mode='eval')
    else : 
        raise RuntimeError('Invalide dataset type')


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
        backbone = get_resent_features(backbone)
        out_ch = 512 # resnet18,34 : 512 
    elif args.model == 'res50':
        backbone = torchvision.models.resnet50(pretrained=True) # res 50 
        backbone = get_resent_features(backbone)
        out_ch = 2048 # resnet50~152 : 2048
    elif args.model == 'res34AAA':
        backbone = AAA('res34', True, args)
        out_ch = 512 # resnet18,34 : 512 
    else : 
        assert()

    # Anchor size : ((size, size*2, size*4, size*8, size*16), )
    anchor_size = (tuple(int(args.anchor_init_size * math.pow(2, i)) for i in range(5)), )

    backbone.out_channels = out_ch

    anchor_generator = AnchorGenerator(sizes=anchor_size,
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

    # if args.model == 'res50fpn':
    #     model = fasterrcnn_resnet50_fpn(pretrained=True).to(args.device)
    #     model.rpn.anchor_generator.sizes = ((8,), (16,), (32,), (64,), (128,))

    # Optimizer 
    optimizer = optim.SGD(model.parameters(), 
                          lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    schedular = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[90, 120], gamma=0.1)
    
    # train 
    global_step = 0
    accuracies = {}
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

            progress.set_description(f'Train {epoch} / {args.epochs}, lr : {optimizer.param_groups[0]["lr"]} ' +
                                    f'Loss : [TT]{losses:.3f}, [HC]{loss_cls:.3f}, [HR]{loss_reg:.3f}, ' +
                                    f'[RO]{loss_obj:.3f}, [RR]{loss_rpn_reg:.3f} ')
            
        if epoch % args.save_epoch == 0 : 
            torch.save(model.state_dict(), 
                       os.path.join(args.weights_dir, f'{args.model}_{epoch}.ckpt'))
                       
        if epoch % args.eval_epoch == 0 : 
            accuracies = evaluate(model, eval_loader, args, epoch, accs=accuracies, update_acc=True)
            if args.test_img_name == '': 
                image_path = os.path.join(args.test_img_folder, 
                                          random.sample(os.listdir(args.test_img_folder), 1)[0])
                args.test_img_name = image_path
            test(model, image_path, args, epoch)
        
        ## Tensor Board 
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step)
        writer.add_scalar('loss_classifier', loss_cls, global_step)
        writer.add_scalar('loss_box_reg', loss_reg, global_step)
        writer.add_scalar('loss_objectness', loss_obj, global_step)
        writer.add_scalar('loss_rpn_box_reg', loss_rpn_reg, global_step)
        global_step += 1

        schedular.step()

    ## Evaluate rankings 
    accuracies = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)
    print('##### TOP 3 models by iou 0.5 value #####')
    for i in range(3) : 
        print(f'TOP {i+1} : epoch {accuracies[i][0]}, accuracy {accuracies[i][1]}')

if __name__=='__main__':
    main()
