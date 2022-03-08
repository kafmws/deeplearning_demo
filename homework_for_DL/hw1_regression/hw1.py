import torch.nn
from torch.utils.data import Dataset, DataLoader


class CovidDataset(Dataset):
    def __init__(self, filepath, is_train):
        data, labels = [], []
        # lines = open(filepath).readlines()
        # column_names = lines[0].split(',')
        # for line in lines[1:]:
        #     line = map(lambda s: float(s), line.split(','))[1:]
        #     if is_train:
        #
        #     data.append()

        self.data = data

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):

        return image, label