import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from medpy import metric

def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator, RandomGenerator1
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    img_path = args.root_path+"/img"
    label_path = args.root_path+"/label"
    db_train = Synapse_dataset(img_dir=img_path, label_dir=label_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                    [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    val_image_path = "data/Synapse/less002/val/img"
    val_label_path = "data/Synapse/less002/val/label"
    val_dir = "lists/lists_Synapse/less002"
    db_val = Synapse_dataset(img_dir=val_image_path, label_dir=val_label_path, list_dir=val_dir, split="val", 
                             transform=transforms.Compose(
                                    [RandomGenerator1(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    # optimizer = optim.Adam(model.parameters(), lr=base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001, amsgrad=False)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0

    best_loss = float('inf')
    best_dice = 0.0
    best_model_path = os.path.join(snapshot_path, 'best_model.pth')

    def validate(model, valloader, criterion_ce, criterion_dice):
        model.eval()  
        val_loss = 0  
        val_loss_ce = 0  
        val_loss_dice = 0  
        total_dice = 0 

        with torch.no_grad():  
            for batch in valloader:  
                images, labels = batch['image'].cuda(), batch['label'].cuda()  
                outputs = model(images)  
                loss_ce = criterion_ce(outputs, labels.long())  
                loss_dice = criterion_dice(outputs, labels, softmax=True)  
                loss = 0.5 * loss_ce + 0.5 * loss_dice  

                val_loss += loss.item()  
                val_loss_ce += loss_ce.item()  
                val_loss_dice += loss_dice.item()  
                outputs = torch.softmax(outputs, dim=1)
                pred = outputs.argmax(dim=1).cpu().numpy()
                true = labels.cpu().numpy()

                dice_score = metric.binary.dc(pred==1, true==1)
                total_dice += dice_score

        avg_val_loss = val_loss / len(valloader)  
        avg_val_loss_ce = val_loss_ce / len(valloader)  
        avg_val_loss_dice = val_loss_dice / len(valloader)  
        avg_val_dice = total_dice / len(valloader)  

        model.train()  
        return avg_val_loss, avg_val_loss_ce, avg_val_loss_dice, avg_val_dice

    for epoch_num in range(max_epoch):  
        epoch_loss = 0  
        epoch_loss_ce = 0  
        epoch_loss_dice = 0  
        total_dice = 0 

        for i_batch, sampled_batch in enumerate(trainloader):  
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']  
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()  
            outputs = model(image_batch) 
            loss_ce = ce_loss(outputs, label_batch[:].long())  
            loss_dice = dice_loss(outputs, label_batch, softmax=True)  
            loss = 0.5 * loss_ce + 0.5 * loss_dice  

            optimizer.zero_grad()  
            loss.backward()  
            optimizer.step()  

            epoch_loss += loss.item()  
            epoch_loss_ce += loss_ce.item()  
            epoch_loss_dice += loss_dice.item()  

            with torch.no_grad():  
                outputs = torch.softmax(outputs, dim=1)  
                pred = outputs.argmax(dim=1).cpu().numpy()  
                true = label_batch.cpu().numpy()  
                dice_score = metric.binary.dc(pred == 1, true == 1)  # 假设1是目标类别  
                total_dice += dice_score  

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9  
            for param_group in optimizer.param_groups:  
                param_group['lr'] = lr_  

            iter_num = iter_num + 1  
            writer.add_scalar('info/lr', lr_, iter_num)  
            writer.add_scalar('info/total_loss', loss, iter_num)  
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)  

        avg_loss = epoch_loss / len(trainloader)  
        avg_loss_ce = epoch_loss_ce / len(trainloader)  
        avg_loss_dice = epoch_loss_dice / len(trainloader)  
        avg_dice = total_dice / len(trainloader)  

        # 验证  
        avg_val_loss, avg_val_loss_ce, avg_val_loss_dice, avg_val_dice = validate(model, valloader, ce_loss, dice_loss)  

        logging.info(f'Epoch [{epoch_num+1}/{max_epoch}] - '  
                     f'Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '  
                     f'Train Dice: {avg_dice:.4f}, Val Dice: {avg_val_dice:.4f}')   

        writer.add_scalar('epoch/avg_loss', avg_loss, epoch_num)  
        writer.add_scalar('epoch/avg_loss_ce', avg_loss_ce, epoch_num)  
        writer.add_scalar('epoch/avg_loss_dice', avg_loss_dice, epoch_num)  
        writer.add_scalar('epoch/avg_dice', avg_dice, epoch_num)  
        writer.add_scalar('epoch/val_loss', avg_val_loss, epoch_num)  
        writer.add_scalar('epoch/val_dice', avg_val_dice, epoch_num)

        if avg_val_dice > best_dice:  
            best_dice = avg_val_dice  
            best_loss = avg_val_loss  
            torch.save(model.state_dict(), best_model_path)  
            logging.info(f"New best model saved with loss: {best_loss:.4f} and Dice: {best_dice:.4f}")  

    writer.close()  
    logging.info(f"Best model saved with loss: {best_loss:.4f} and Dice: {best_dice:.4f}")  
    return "Training Finished!"
