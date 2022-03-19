import os
import json
import random
import PIL.Image
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision.models.resnet import resnet50
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Compose, Resize, CenterCrop


class AnimalsDataset(Dataset):

    def __init__(self, metadata_path, transform=None):
        json_data = json.load(open(metadata_path))
        data = []
        id_cnt = {}
        id_label = []
        for item in json_data['annotations']:
            data.append({'id': item['image_id'], 'label': item['category_id']})
            id_cnt[item['image_id']] = id_cnt.get(item['image_id'], 0) + 1
        # discard the image with the same id
        for item in data:
            if id_cnt[item['id']] == 1:
                id_label.append(item)
        label_name = {item['id']: item['name'] for item in json_data['categories']}
        self.data = id_label
        self.path_prefix = '/home/kafm/data/ENA24-detection/'
        self.label_name = label_name
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = os.path.join(self.path_prefix, self.data[index]['id'] + '.jpg')
        img = PIL.Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.data[index]['label']


def train(model, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100.0 * batch_idx / len(train_loader), loss.item()))


# split the dataset into train, val and test
def split_dataset(dataset, train_ratio, val_ratio, test_ratio):
    train_size = int(len(dataset) * train_ratio)
    val_size = int(len(dataset) * val_ratio)
    assert train_ratio + val_ratio + test_ratio == 1.0
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = \
        torch.utils.data.random_split(dataset, [train_size, val_size, test_size],
                                      generator=torch.Generator().manual_seed(0))
    return train_dataset, val_dataset, test_dataset


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # fix random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.


if __name__ == "__main__":
    # fix random seed
    seed_torch()

    animal_labels = []

    transforms = Compose([Resize(446), ToTensor(), ])
    animals_dataset = AnimalsDataset('../dataset/meta/ena24/ena24.json', transform=transforms)
    train_dataset, val_dataset, test_dataset = split_dataset(animals_dataset, 0.8, 0.1, 0.1)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    # val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=4)
    # test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    net = resnet50(pretrained=True)
    train(net,
          train_dataloader,
          optimizer=optim.Adam(net.parameters(), lr=0.001),
          criterion=nn.CrossEntropyLoss(),
          epoch=400)
    # test_csv = os.open('../dataset/meta/ena24/test.csv', os.O_WRONLY | os.O_CREAT)
    # train_csv = os.open('../dataset/meta/ena24/train.csv', os.O_WRONLY | os.O_CREAT)
