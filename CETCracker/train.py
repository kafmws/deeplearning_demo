import argparse
import os
import time

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
import matplotlib.pyplot as plt

from datasets import CaptchaData
from predict import predict
from models import Models

batch_size = 128
base_lr = 0.001
max_epoch = 200
arch = 'resnet18'
model_name = 'model.pth'
model_dir = './checkpoints/'
restore = False
train_dir = './data/train'
test_dir = './data/test'

if not os.path.exists('./checkpoints'):
    os.mkdir('./checkpoints')


def calculat_acc(output, target):
    output, target = output.view(-1, 26), target.view(-1, 26)
    output = nn.functional.softmax(output, dim=1)
    output = torch.argmax(output, dim=1)
    target = torch.argmax(target, dim=1)
    output, target = output.view(-1, 4), target.view(-1, 4)
    correct_list = []
    for i, j in zip(target, output):
        if torch.equal(i, j):
            correct_list.append(1)
        else:
            correct_list.append(0)
    acc = sum(correct_list) / len(correct_list)
    return acc


def train():
    transforms = Compose([ToTensor()])
    train_dataset = CaptchaData(train_dir, transform=transforms)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0,
                                   shuffle=True, drop_last=True)
    test_data = CaptchaData(test_dir, transform=transforms)
    test_data_loader = DataLoader(test_data, batch_size=batch_size,
                                  num_workers=0, shuffle=False, drop_last=True)

    cnn = Models(arch=arch)
    # print(cnn)

    if torch.cuda.is_available():
        cnn.cuda()
    if restore:
        cnn.load_state_dict(torch.load(model_path))
    #        freezing_layers = list(cnn.named_parameters())[:10]
    #        for param in freezing_layers:
    #            param[1].requires_grad = False
    #            print('freezing layer:', param[0])

    optimizer = torch.optim.Adam(cnn.parameters(), lr=base_lr)
    criterion = nn.MultiLabelSoftMarginLoss()

    loss_history = []
    for epoch in range(1, max_epoch + 1):
        start_time = time.time()

        mean_loss_history = []
        mean_acc_history = []
        cnn.train()
        for img, target in train_data_loader:
            img = Variable(img)
            target = Variable(target)
            if torch.cuda.is_available():
                img = img.cuda()
                target = target.cuda()
            output = cnn(img)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = calculat_acc(output, target)
            mean_acc_history.append(float(acc))
            mean_loss_history.append(float(loss))
        print('train_loss: {:.4}\t|\ttrain_acc: {:.4}'.format(
            torch.mean(torch.Tensor(mean_loss_history)),
            torch.mean(torch.Tensor(mean_acc_history)),
        ))
        loss_history.append(torch.Tensor(mean_loss_history))

        if epoch % 10 == 0:
            predict(model_path=model_path, arch=arch)

        torch.save(cnn.state_dict(), model_path)
        print('epoch: {}\t|\ttime: {:.4f}s'.format(epoch, time.time() - start_time))

    loss_list = list(map(lambda x: float(torch.mean(x)), loss_history))
    plt.clf()
    plt.plot(list(range(1, len(loss_list) + 1)), loss_list, label='train loss', linewidth=2, color='r',
             marker='o', markerfacecolor='r',
             markersize=5)
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.show()
    print("Total paramerters number of {} is {}  ".format(arch, sum(x.numel() for x in cnn.parameters())))
    print("Total paramerters number of {} trainable paramerters in networks is {}  ".format(
        arch, sum(x.numel() for x in cnn.parameters() if x.requires_grad)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train CNN')
    parser.add_argument('--model_dir', default='./checkpoints/', type=str, required=False,
                        help='model save path')
    parser.add_argument('--train_dir', default='./data/train', type=str, required=False, help='train set path')
    parser.add_argument('--test_dir', default='./data/test', type=str, required=False, help='test set path')
    parser.add_argument('--base_lr', default=0.001, type=float, required=False, help='base learning rate')
    parser.add_argument('--restore', default=False, type=bool, required=False, help='continue to train')
    parser.add_argument('--max_epoch', default=80, type=int, required=False, help='max epoch number')
    parser.add_argument('--batch_size', default=128, type=int, required=False, help='batch size')
    parser.add_argument('--arch', default='cnnv2', type=str, required=False, help='NN architecture',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'squeezenet1_0',
                                 'squeezenet1_1',
                                 'cnn', 'cnnv2'])
    args = parser.parse_args()

    batch_size = args.batch_size
    max_epoch = args.max_epoch
    train_dir = args.train_dir
    model_dir = args.model_dir
    test_dir = args.test_dir
    restore = args.restore
    base_lr = args.base_lr
    arch = args.arch

    model_name = 'model_' + arch + '.pth'
    model_path = os.path.join(model_dir, model_name)

    train()
    pass
