from torch.utils.data.dataset import Dataset
from skimage.io import imread
import numpy as np
import pandas as pd
import os

import torch
import torchvision as tv


class fundusDataset(Dataset):
    def __init__(self, csv_path, base_path=r'C:\Users\shiqi\PycharmProjects\norm_512', crop_type=None, crop_size=(512, 512), downsample: int = None):
        super().__init__()
        self.csv_path = csv_path
        dtype_dic = {'Filename': str, 'RandomID': str, 'age': int, 'sex': int, 'T2D': int, 'HT': int}
        df = pd.read_excel(csv_path, engine='openpyxl', converters=dtype_dic)
        self.num_items = len(df)
        self.metrics = {col: df[col] if col == 'Filename' or col == 'RandomID' or col == 'path' else torch.as_tensor(df[col]).float() for col in df.columns}
        self.base_path = base_path

        self.crop_type = crop_type
        self.crop_size = crop_size

        self.downsample = downsample

    def __len__(self):
        return self.num_items

    def __getitem__(self, index):
        item = {col: values[index] for col, values in self.metrics.items()}

        img_path = os.path.join(self.base_path, '{}.tiff'.format(item['Filename']))
        img = imread(img_path, as_gray=False)

        # img = imread(img_path, as_gray=True)
        # img = (img * 255).astype(np.uint8)
        # read gray image

        transform_list = []
        transform_list += [tv.transforms.ToPILImage()]
        if self.crop_type is not None:
            if self.crop_type == 'center':
                transform_list += [tv.transforms.CenterCrop(self.crop_size)]
            elif self.crop_type == 'random':
                transform_list += [tv.transforms.RandomCrop(self.crop_size)]
            else:
                raise ValueError('unknwon crop type: {}'.format(self.crop_type))

        if self.downsample is not None and self.downsample > 1:
            transform_list += [tv.transforms.Resize(tuple(np.array(self.crop_size) // self.downsample))]

        transform_list += [tv.transforms.ToTensor()]

        img = tv.transforms.Compose(transform_list)(img)

        item['image'] = img

        return item
