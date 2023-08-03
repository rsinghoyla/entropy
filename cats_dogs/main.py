'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
import datasets
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset # Gives easier dataset managment and creates mini batches
from torchvision.datasets import ImageFolder

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--save_dir',type= str, default = 'catsdogs')
parser.add_argument('--bpp_factor',type=float,default = None)
parser.add_argument('--bpp_factor_bal',type=float,default = None)
parser.add_argument('--rand',type=float,default = 1)
parser.add_argument('--prt',type=float,default = None)
parser.add_argument('--prop', type = float, default = 1)
parser.add_argument('--sort_order', type = str, default = 'rand')
parser.add_argument('--sort_on',type = str, default = 'size')
parser.add_argument('--no_dropout',action='store_true')
parser.add_argument('--no_augment',action='store_true')
parser.add_argument('--finetune',type = str, choices=['none','imagenet','swav'],default = 'none')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'mps'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
from sklearn.model_selection import train_test_split
dataset = ImageFolder("data/PetImages/train/")
train_data, test_data, train_label, test_label = train_test_split(dataset.imgs, dataset.targets, test_size=0.2, random_state=42)
if not args.no_augment:
    train_transform = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225] )]) # train transform
else:
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225] )]) # train transform
test_transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225] )])
 # test transform
# if args.bpp_factor is not None:
#     if args.bpp_factor_bal is None:
#         if args.bpp_factor>0:
#             trainset = datasets.ImageLoader(train_data, train_transform,bpp_thresh_cat=0.0425/args.bpp_factor,bpp_thresh_dog=0.05/args.bpp_factor)
#         if args.bpp_factor==0:
#             trainset = datasets.ImageLoader(train_data, train_transform,bpp_thresh_cat=0,bpp_thresh_dog=args.rand)
#         if args.bpp_factor<0:
#             trainset = datasets.ImageLoader(train_data, train_transform,bpp_thresh_cat=0.065*args.bpp_factor,bpp_thresh_dog=0.08*args.bpp_factor)
#     else:
#         trainset = datasets.ImageLoader(train_data, train_transform,bpp_thresh2_cat=0.0425/args.bpp_factor,bpp_thresh2_dog=0.05/args.bpp_factor, bpp_thresh_cat=0.065*args.bpp_factor_bal, bpp_thresh_dog=0.08*args.bpp_factor_bal)
# elif args.prt is not None:
#     trainset = datasets.ImageLoader(train_data, train_transform, page_rank_thresh = args.prt)
# else:
trainset = datasets.ImageLoader(train_data, train_transform, sort_on = args.sort_on, sort_order = args.sort_order, proportion = args.prop)
testset = datasets.ImageLoader(test_data, test_transform)
print(trainset.__len__(),testset.__len__())
# import scipy.io
#  scipy.io.savemat('x.mat',{'d':trainset.dataset})

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True, num_workers=1)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=64, shuffle=False, num_workers=1)
c = 0
for t in trainloader.dataset.dataset:
    c+=t[1]
print("balance",c)

save_dir = args.save_dir+'_'+str(trainset.__len__())+'_'+str(trainset.stats)
classes = ( 'cat', 'dog', )

# Model
print('==> Building model..')
# net = VGG('VGG19')
#net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
#net = SimpleDLA()
from torchvision import models

if args.finetune =='swav':
    
    net = resnet50(output_dim=0, eval_mode=True)#torch.hub.load('facebookresearch/swav:main', 'resnet50')
    state_dict = torch.load("swav_800ep_pretrain.pth.tar")


    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    for k, v in net.state_dict().items():
        if k not in list(state_dict):
            logger.info('key "{}" could not be found in provided state dict'.format(k))
        elif state_dict[k].shape != v.shape:
            logger.info('key "{}" is of different shape in model and provided state dict'.format(k))
            state_dict[k] = v


    net.load_state_dict(state_dict, strict=False)
    reglog = RegLog(2)
    reglog = reglog.to(device)
else:
    if args.finetune == 'imagenet':
        net = models.resnet18(pretrained=True)
        for param in net.parameters():
            param.requires_grad = False
    elif args.finetune == 'none':
        net = models.resnet18(pretrained=False)
        
    num_ftrs = net.fc.in_features
    if args.no_dropout:
        net.fc = nn.Linear(num_ftrs, 2)
    else:
        print("applying dropout")
        net.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 2)
        )


net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=args.lr,
#                      momentum=0.9, weight_decay=5e-4)
if args.finetune != 'swav':
    optimizer = optim.RMSprop(net.parameters(),lr=args.lr, weight_decay=1e-4)
else:
    optimizer = optim.RMSprop(reglog.parameters(),lr=args.lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch,writer):
    print('\nEpoch: %d' % epoch)
    if args.finetune != 'swav':
        net.train()
    else:
        net.eval()
        reglog.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        if args.finetune != 'swav':
            outputs = net(inputs)
        else:
            with torch.no_grad():
                outputs = net(inputs)
            
            outputs = reglog(outputs)
            
        loss = criterion(outputs, targets)
        loss.backward()
        
        nn.utils.clip_grad_value_(net.parameters(), 0.1)
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        writer.add_scalar('training loss', train_loss,epoch )
        writer.add_scalar("train Accuracy", 100.*correct/total, epoch)
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch,writer):
    global best_acc
    net.eval()
    if args.finetune == 'swav':
        reglog.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            if args.finetune != 'swav':
                outputs = net(inputs)
            else:
                outputs = net(inputs)
                outputs = reglog(outputs)
                
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            writer.add_scalar('test loss', test_loss,epoch )
            writer.add_scalar("test Accuracy", 100.*correct/total, epoch)
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        if not os.path.isdir(os.path.join(save_dir,'checkpoint')):
            os.mkdir(os.path.join(save_dir,'checkpoint'))
        torch.save(state, os.path.join(save_dir,'checkpoint/ckpt.pth'))
        best_acc = acc
    if epoch%20 == 0:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        if not os.path.isdir(os.path.join(save_dir,'checkpoint')):
            os.mkdir(os.path.join(save_dir,'checkpoint'))
        torch.save(state, os.path.join(save_dir,'checkpoint/ckpt_'+str(epoch)+'.pth'))
        

if __name__ == '__main__':

    writer = SummaryWriter(os.path.join('runs',save_dir))
    
    if args.finetune == 'imagenet':
        num_epoch = 10
    elif args.finetune == 'swav':
        num_epoch = 5
    else:
        num_epoch = 200
    for epoch in range(start_epoch, start_epoch+num_epoch):
        train(epoch,writer)
        test(epoch,writer)
        scheduler.step()
    writer.close()
