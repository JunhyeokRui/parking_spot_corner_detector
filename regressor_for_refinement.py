from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet34, resnet18
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import matplotlib.ticker as plticker

 
# from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig


# model_names = sorted(name for name in models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='two_corner_in_training')
# Datasets
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--train-batch', default=256, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=256, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--n_class', default=2, type=int)
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--train_trials', required=True, type=str)
parser.add_argument('--test_trials', required=True, type=str)
parser.add_argument('--run', required=True, type=str)

args = parser.parse_args()

# Use CUDA
use_cuda = torch.cuda.is_available()

# Random seed
# if args.manualSeed is None:
#     args.manualSeed = random.randint(1, 10000)
# random.seed(args.manualSeed)
# torch.manual_seed(args.manualSeed)
# if use_cuda:
#     torch.cuda.manual_seed_all(args.manualSeed)

writer = SummaryWriter('./runs/{}'.format(args.run))


class cropped_dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, trials_to_select, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_dir = os.path.join('../refinement_data','images')
        self.label_dir = os.path.join('../refinement_data','ratio_labels')
        self.transform = transform
        self.image_list = []
        self.label_list = []
        for this_trial in os.listdir(self.data_dir):
            if 'trial' in this_trial and this_trial[6] in trials_to_select:
                for this_image in os.listdir(os.path.join(self.data_dir, this_trial)):
                    if 'jpeg' in this_image:
                        self.image_list.append(os.path.join(self.data_dir,this_trial,this_image))
                        self.label_list.append(os.path.join(self.label_dir,this_trial,'{}.npy'.format(this_image[:-5])))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        img = Image.open(self.image_list[idx])
        r_x, r_y = np.load(self.label_list[idx])
        if self.transform:
            sample = self.transform(img)

        return sample, r_x, r_y

# class croppeddataset(Dataset):
#     """Face Landmarks dataset."""

#     def __init__(self, dataset_name, split, transform=None):
#         """
#         Args:
#             csv_file (string): Path to the csv file with annotations.
#             root_dir (string): Directory with all the images.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         self.data_dir = os.path.join('./dataset','2_in_cropped')
#         self.split_dir = os.path.join('./position_labels','2_in.txt')
#         self.transform = transform
#         self.img_lists = open(self.split_dir,'r').readlines()
#         self.split = split

#     def __len__(self):
#         return len(self.img_lists)

#     def __getitem__(self, idx):
#         # print(self.img_lists)
#         img_name, left_x,left_y,right_x,right_y = self.img_lists[idx][:-1].split(',')
        
#         img = Image.open(os.path.join(self.data_dir,'new_image_{}.jpeg'.format(img_name)))

#         if self.transform:
#             sample = self.transform(img)

#         return sample, int(left_x)/512,int(left_y)/288,int(right_x)/512,int(right_y)/288

def main():
    global best_acc
    # start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch
    # Data
    # print('==> Preparing dataset %s' % args.dataset)

    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomApply(transforms.ColorJitter(brightness=.5, hue=.3),p=0.5),
        # transforms.ColorJitter(saturation=(0,0.2),hue=(-0.1,0.1)),
        transforms.Resize([50,50]),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomApply(transforms.ColorJitter(brightness=.5, hue=.3),p=0.5),
        transforms.Resize([50,50]),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_trials = args.train_trials.split(',')
    test_trials = args.test_trials.split(',')


    trainset = cropped_dataset(train_trials ,transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
    testset = cropped_dataset(test_trials ,transform_test)
    testloader = data.DataLoader(testset, batch_size=args.train_batch, shuffle=False, num_workers=args.workers)

    print(len(trainset))
    # print(len(testset))

    # Model
    # print("==> creating model '{}'".format(args.arch))
    model = resnet18()
    # model.fc = nn.Linear(512, 2)
    # model = model.cuda()
    
    # loading_classifier_state = torch.load('./checkpoint/{}/300_epoch_model.pth'.format(args.dataset))
    # model.load_state_dict(loading_classifier_state)
    model.fc = nn.Sequential(nn.Linear(512,2),nn.Sigmoid())
    # model.fc = nn.Linear(512,4)

    model = model.cuda()
    # model = model.float()s\
    # model = torch.nn.DataParallel(model).cuda()
    # cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=0)
    # Train and val
    # best_accuracy =0
    for epoch in range(1, args.epochs+1):
        # adjust_learning_rate(optimizer, epoch)

        # print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
        test_loss = test_output_sample_images(testloader, model, criterion, epoch, use_cuda)

        print('train : epoch / loss = {} / {}'.format(epoch, train_loss))
        print('test : epoch / loss = {} / {}'.format(epoch, test_loss))

        if epoch%50 ==0:
            save_checkpoint(model.state_dict(),epoch)

        writer.add_scalar('loss/train', train_loss, epoch)
        writer.add_scalar('loss/test', test_loss, epoch)


def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    end = time.time()

    # bar = Bar('Processing', max=len(trainloader))
    t_loss =0
    total =0
    for batch_idx, (inputs, p_x,p_y) in enumerate(tqdm(trainloader)):
        # measure data loading time
        # data_time.update(time.time() - end)
        # print(left_x)
        # print(inputs)
        if use_cuda:
            inputs, p_x,p_y = inputs.cuda(), p_x.cuda(),p_y.cuda()
        # inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        # compute output
        outputs = model(inputs)
        # print(outputs.shape)
        targets = torch.stack((p_x,p_y),1).float()
        # print(outputs, targets)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        t_loss += loss.item()
        total += 1
    avg_loss = t_loss / total

    return avg_loss 


def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    # batch_time = AverageMeter()
    # data_time = AverageMeter()
    # losses = AverageMeter()
    # top1 = AverageMeter()
    # top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    # bar = Bar('Processing', max=len(testloader))
    t_loss=0
    total=0
    for batch_idx, (inputs, p_x,p_y) in enumerate(testloader):
        # measure data loading time
        # data_time.update(time.time() - end)

        if use_cuda:
            inputs, p_x,p_y = inputs.cuda(), p_x.cuda(),p_y.cuda()
        # inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        with torch.no_grad():
        # compute output
            outputs = model(inputs)
            targets = torch.stack((p_x,p_y),1).float()
            loss = criterion(outputs, targets)

        t_loss += loss.item()
        total += 1
    avg_loss = t_loss / total
    return avg_loss

def test_output_sample_images(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    # batch_time = AverageMeter()
    # data_time = AverageMeter()
    # losses = AverageMeter()
    # top1 = AverageMeter()
    # top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    # bar = Bar('Processing', max=len(testloader))
    t_loss=0
    total=0
    for batch_idx, (inputs, p_x,p_y) in enumerate(testloader):
        # measure data loading time
        # data_time.update(time.time() - end)

        if use_cuda:
            inputs, p_x,p_y = inputs.cuda(), p_x.cuda(),p_y.cuda()
        # inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        with torch.no_grad():
        # compute output
            outputs = model(inputs)
            targets = torch.stack((p_x,p_y),1).float()
            loss = criterion(outputs, targets)

        t_loss += loss.item()
        total += 1
    avg_loss = t_loss / total

    for i in range(10):

        sample_img = inputs[i,:].squeeze()
        sample_img = transforms.ToPILImage()(sample_img)
        

        fig,ax = plt.subplots(1)
        ax.set_aspect('equal')

        # Show the image
        ax.imshow(sample_img)#.transpose(0,2).transpose(0,1))
        circ = Circle((int(outputs[i][0]*50),int(outputs[i][0]*50)),4,color='b')
        ax.add_patch(circ)

        filepath = os.path.join('refinement_network_weights',args.run,'test_at_epoch_{}'.format(epoch))
        os.makedirs(filepath,exist_ok = True)

        plt.savefig(os.path.join(filepath,'sample_test_output_{}.png'.format(i)))
        plt.close()





    return avg_loss

def save_checkpoint(state, epoch, checkpoint='refinement_network_weights', best=None):
    filepath = os.path.join(checkpoint,args.run)
    os.makedirs(filepath,exist_ok = True)
    if best is not None:
        torch.save(state, os.path.join(filepath,'best_model.pth'))
    else:
        torch.save(state, os.path.join(filepath,'{}_epoch_model.pth'.format(epoch)))

if __name__ == '__main__':
    main()