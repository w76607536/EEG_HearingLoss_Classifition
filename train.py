'''
Description: train
Author: Wu Yubo
Date: 2022-05-17 00:08:58
LastEditTime: 2022-05-18 23:42:10
LastEditors:  
'''
import argparse
import os
import losses
from utils import str2bool, count_params
from tqdm import tqdm
from collections import OrderedDict
from glob import glob
import joblib
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch
import archs
import pandas as pd
import torch.optim as optim
from sklearn.model_selection import train_test_split
from dataset import Dataset
from metrics import classification_accuracy

arch_names = list(archs.__dict__.keys())
loss_names = list(losses.__dict__.keys())
loss_names.append('BCEWithLogitsLoss')

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='Dense_Unet',
                        choices=arch_names,
                        help='model architecture: ' +
                            ' | '.join(arch_names) +
                            ' (default: NestedUNet)')
    parser.add_argument('--dataset', default="BOBOWork",
                        help='dataset name')
    parser.add_argument('--aug', default=False, type=str2bool)
    parser.add_argument('--epochs', default=10000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early-stop', default=30, type=int,
                        metavar='N', help='early stopping (default: 20)')
    parser.add_argument('-b', '--batch-size', default=18, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                            ' | '.join(['Adam', 'SGD']) +
                            ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--input-channels', default=1, type=int,help='input channels')
    parser.add_argument('--num-classes', default=5, type=int, help='num_classes')
    parser.add_argument('--test', default=False, type=bool, help='is_test')
    #parser.add_argument('--image-ext', default='png',help='image file extension')
    #parser.add_argument('--mask-ext', default='png',help='mask file extension')
    parser.add_argument('--weight-decay', default=1e-4, type=float,help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    args = parser.parse_args()

    return args

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(args, train_loader, model, lossFunc, optimizer, epoch, scheduler=None):
    losses = AverageMeter()
    accs = AverageMeter()

    model.train()
    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        input = input.cuda()
        target = target.cuda()

        # compute output
        output = model(input)
        loss = lossFunc(output, target)
        acc = classification_accuracy(output, target)

        losses.update(loss.item(), input.size(0))
        accs.update(acc, input.size(0))

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log = OrderedDict([
        ('loss', losses.avg),
        ('acc', accs.avg),
    ])

    return log


def validate(args, val_loader, model, lossFunc):
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)
            loss = lossFunc(output, target)
            acc = classification_accuracy(output, target)

            losses.update(loss.item(), input.size(0))
            accs.update(acc, input.size(0))

    log = OrderedDict([
        ('loss', losses.avg),
        ('acc', accs.avg),
    ])

    return log

def main():
    args = parse_args()
    #args.dataset = "datasets"

    if args.name is None:
        args.name = '%s_%s_woDS' %(args.dataset, args.arch)

    if not os.path.exists('models/%s' %args.name):
        os.makedirs('models/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' % (arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' % args.name)

    lossFunc = nn.CrossEntropyLoss()

    cudnn.benchmark = True


    img_paths = glob(r'Data/inputs/trainImage/*')

    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = \
        train_test_split(img_paths, img_paths, test_size=0.2, random_state=41)
    print("train_num:%s" % str(len(train_img_paths)))
    print("val_num:%s" % str(len(val_img_paths)))
    print("=> creating model %s" % args.arch)
    model = archs.__dict__[args.arch](args, args.input_channels, args.num_classes)

    model = model.cuda()

    print(count_params(model))

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)

    if args.test:
       # print(args.test)

        train_dataset = Dataset(args, train_img_paths[0:16],  args.aug, args.input_channels)
        val_dataset = Dataset(args, val_img_paths[0:16], False, args.input_channels)
    else:
        train_dataset = Dataset(args, train_img_paths, args.aug, args.input_channels)
        val_dataset = Dataset(args, val_img_paths, False, args.input_channels)

    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False)
    
    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'loss', 'acc', 'val_loss', 'val_acc'
    ])


    best_acc = 0
    trigger = 0

    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' %(epoch, args.epochs))

        # train for one epoch
        train_log = train(args, train_loader, model, lossFunc, optimizer, epoch)
        # evaluate on validation set
        val_log = validate(args, val_loader, model, lossFunc)

        print('loss %.4f - acc %.4f - val_loss %.4f - val_acc %.4f - best_acc %.4f'
            %(train_log['loss'], train_log['acc'], val_log['loss'], val_log['acc'], best_acc))

        tmp = pd.Series([
            epoch,
            args.lr,
            train_log['loss'],
            train_log['acc'],
            val_log['loss'],
            val_log['acc'],
        ], index=['epoch', 'lr', 'loss', 'acc', 'val_loss', 'val_acc'])

        log = log.append(tmp, ignore_index=True)
        log.to_csv('models/%s/log.csv' % args.name, index=False)

        trigger += 1

        if val_log['acc'] > best_acc:
            torch.save(model.state_dict(), 'models/%s/model.pth' %args.name)
            best_acc = val_log['acc']
            print("=> saved best model")
            trigger = 0

        # early stopping
        if not args.early_stop is None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
