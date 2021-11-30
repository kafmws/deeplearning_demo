import os

import PIL.ImageShow
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

alphabet = 'abcdefghijklmnopqrstuvwxyz'


def img_loader(img_path):
    img = Image.open(img_path)
    return img.convert('1')  # '1'


def make_dataset(data_path, alphabet, num_class, num_char):
    img_names = os.listdir(data_path)
    samples = []
    for img_name in img_names:
        img_path = os.path.join(data_path, img_name)
        # target_str = img_name.split('.')[0]
        target_str = img_name[0:4]
        assert len(target_str) == num_char
        target = []
        for char in target_str:
            vec = [0] * num_class
            vec[alphabet.find(char)] = 1
            target += vec
        samples.append((img_path, target))
    return samples


class CaptchaData(Dataset):
    def __init__(self, data_path, num_class=26, num_char=4,
                 transform=None, target_transform=None, alphabet=alphabet):
        super(Dataset, self).__init__()
        self.data_path = data_path
        self.num_class = num_class
        self.num_char = num_char
        self.transform = transform
        self.target_transform = target_transform
        self.alphabet = alphabet
        self.samples = make_dataset(self.data_path, self.alphabet,
                                    self.num_class, self.num_char)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, target = self.samples[index]
        img = img_loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, torch.Tensor(target)


def character_cnt(base='rawdata/backup'):
    char_cnt = {}
    picture_cnt = {}
    for ch in range(ord('a'), + ord('a') + 26):
        char_cnt[chr(ch)] = 0
        picture_cnt[chr(ch)] = 0
    for base, dir, filenames in os.walk(base):
        for filename in filenames:
            for ch in filename[0:4]:
                char_cnt[ch] += 1
            for ch in set(filename[0:4]):
                picture_cnt[ch] += 1
    distribution = pd.DataFrame.from_dict(char_cnt, orient='index', columns=['all_cnt'])\
        .reset_index().rename(columns={'index': 'char'}) \
        .join(pd.DataFrame.from_dict(picture_cnt, orient='index', columns=['pic_cnt']), on='char')
    return distribution


if __name__ == '__main__':
    distribution = character_cnt()
    print(distribution.transpose().to_markdown())
    pass
