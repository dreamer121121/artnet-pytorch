import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data  import DataLoader
from configparser import ConfigParser
import argparse

import utils
from utils import log
from video_dataset import VideoFramesDataset
from artnet import ARTNet

from tensorboardX import SummaryWriter

#实例化一个写入器
writer = SummaryWriter(log_dir='./logs/')

log_file = "spatial.log"
log_stream = open("spatial.log", "a")
# Chnage mpl backend
matplotlib.use('Agg')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to config file', required=True)
    args = parser.parse_args()

    config = ConfigParser()
    config.read(args.config)

    #加载数据
    train_loader, validation_loader = load_data(config['Train Data']) #返回数据生成器
    train_losses, val_losses = train(config['Train'], train_loader, validation_loader)

    # save_result(train_losses, val_losses, config['Train Result'])

def load_data(params):
    """Load data for training"""

    log('Loading data...',file=log_stream)
    #数据增广方式
    transform = transforms.Compose([
        transforms.Resize((params.getint('width'), params.getint('height'))),
        transforms.RandomCrop((params.getint('crop'), params.getint('crop'))),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_set = VideoFramesDataset(params['path'], frame_num=16, transform=transform,split='./train_rgb_split1.txt')
    val_set = VideoFramesDataset(params['path'],frame_num=16,transform=transform,split='./val_rgb_split1.txt')

    # Shuffle dataset

    batch_size = params.getint('batch_size')

    #注意此处训练集和验证集都传入的是train_set，通过采样器来进行划分训练集和验证集
    train_loader = DataLoader(train_set, batch_size=batch_size)
    validation_loader = DataLoader(val_set, batch_size=batch_size)
    log('Done loading data',file=log_stream)

    log('****Dataset info****',file=log_stream)
    log(f'Number of classes: {train_set.num_classes}',file=log_stream)
    log(f'Class list: {", ".join(train_set.cls_lst)}',file=log_stream)
    log(f'Numer of training samples: {len(train_set)}',file=log_stream)
    log(f'Numer of validation samples: {len(val_set)}',file=log_stream)
    return train_loader, validation_loader

def train(params, train_loader, validation_loader):
    start_epoch = int(params['start_epoch'])
    artnet = ARTNet(num_classes=params.getint('num_classes'))

    #载入最新的保存点继续训练
    checkpoints = os.listdir("./ckpt/")
    if len(checkpoints) > 0:
        checkpoints.sort()
        torch.load('./ckpt/'+checkpoints[-1])

    # device = 'cuda'
    device = 'cuda'

    # Load pretrained model
    if 'pretrained' in params and params['pretrained'] != 'None':
        artnet.load_state_dict(torch.load(params['pretrained']))

    #  Choose training devices
    if params.getboolean('cuda'):
        devices = ['cuda:' + id for id in params['gpus'].split(',')]
        if len(devices) > 1: #使用一机多卡同时进行训练
            artnet = nn.DataParallel(artnet, device_ids=devices)
    else:
        device = 'cpu'
    artnet = artnet.to(device) #将网络转换到指定的设备上

    #Choose optimizer 选择动量梯度下降作为优化器
    optimizer = optim.SGD(artnet.parameters(), lr=params.getfloat('lr'), momentum=params.getfloat('momentum'))
    #Choose criterion 选择交叉熵损失函数
    criterion = nn.CrossEntropyLoss()

    # Learning rate decay config
    lr_steps = [int(step) for step in params.get('lr_steps').split(',')]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_steps, gamma=0.1)

    training_losses = [] #记录每一个epoch的训练损失
    validating_losses = [] #记录每一个epoch的验证损失
    for epoch in range(start_epoch,params.getint('num_epochs')):
        log('Starting epoch %i:' % (epoch + 1),log_stream)
        log('*********Training*********',file=log_stream)
        artnet.train()
        training_loss = 0
        training_progress = tqdm(enumerate(train_loader))
        correct = 0
        for batch_index, (frames, label) in training_progress:
            print("frames.shape:",frames.size()) #[N,16,3,112,112]
            training_progress.set_description('Batch no. %i: ' % batch_index)
            frames, label = frames.to(device), label.to(device)

            optimizer.zero_grad()
            output = artnet(frames)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

            # Calculating accuracy
            prediction = F.softmax(output, dim=1)
            print("predictions:",prediction)
            prediction = prediction.argmax(dim=1)
            print("predict",prediction)
            prediction, label = prediction.to('cpu'), label.to('cpu')
            correct += prediction.eq(torch.LongTensor(label)).sum()
            print("correct:",correct)

        else:
            avg_loss = training_loss / len(train_loader)
            writer.add_scalar("trainloss",avg_loss,epoch+1)
            accuracy = correct / (len(train_loader) * train_loader.batch_size)
            training_losses.append(avg_loss)
            log(f'Training loss: {avg_loss}',file=log_stream)
            log(f'Training accuracy: {accuracy:0.2f}',file=log_stream)
            writer.add_scalar("train_loss",avg_loss,epoch+1)
            writer.add_scalar("train_accuracy", avg_loss, epoch + 1)

        log('*********Validating*********',file=log_stream)
        artnet.eval()
        validating_loss = 0
        validating_progress = tqdm(enumerate(validation_loader))
        correct = 0
        with torch.no_grad():
            for batch_index, (frames, label) in validating_progress:
                validating_progress.set_description('Batch no. %i: ' % batch_index)
                frames, label = frames.to(device), label.to(device)

                output = artnet(frames)
                loss = criterion(output, label)
                validating_loss += loss.item()

                # Calculating accuracy
                _, prediction = output.max(dim=1) #???
                prediction, label = prediction.to('cpu'), label.to('cpu')
                correct += prediction.eq(torch.LongTensor(label)).sum()
            else:
                avg_loss = validating_loss / len(validation_loader)
                accuracy = correct / (len(train_loader) * validation_loader.batch_size)
                validating_losses.append(avg_loss)
                log(f'Validation loss: {avg_loss}',file=log_stream)
                log(f'Validation accuracy: {accuracy:0.2f}',file=log_stream)
                writer.add_scalar("val_loss", avg_loss, epoch + 1)
                writer.add_scalar("val_accuracy", avg_loss, epoch + 1)
        log('=============================================',file=log_stream)
        log('Epoch %i complete' % (epoch + 1),file=log_stream)

        if (epoch + 1) % params.getint('ckpt') == 0:
            log('Saving checkpoint...' ,file=log_stream)
            torch.save(artnet.state_dict(), os.path.join(params['ckpt_path'], 'artnet_%03i.pth' % (epoch + 1)))

        # Update LR
        scheduler.step()
    log('Training complete, saving final model....',file=log_stream)
    torch.save(artnet.state_dict(), os.path.join(params['ckpt_path'], 'artnet_final.pth'))
    return training_losses, validating_losses



def save_result(train_losses, val_losses, params):
    """Saving result in term of training loss and validation loss"""

    # Save log
    file_path = os.path.join(os.path.join(params['path'], 'result.txt'))
    with open(file_path, 'w') as f:
        for i in range(len(train_losses)):
            f.write('Epoch %i: training loss - %0.4f, validation loss - %0.4f\n' % (i + 1, train_losses[i], val_losses[i]))

    # Save chart
    data = { 'epoch': range(1, len(train_losses) + 1), 'train': train_losses, 'val': val_losses}
    plt.plot('epoch', 'train', data=data, label='Training loss', color='blue' )
    plt.plot('epoch', 'val', data=data, label='Validation loss', color='red' )
    plt.legend()
    plt.savefig(os.path.join(params['path'], 'result.png'))


if __name__ == '__main__':
    main()

