import os

import numpy as np
import torch.utils.data as data
import json
from PIL import Image
import matplotlib.pyplot as plt


def gen_records(labels, path):
    ret = []
    for lb in labels:
        for img in os.listdir(f'{path}/{lb}'):
            img_path = f'{path}/{lb}/{img}'
            label = labels.index(lb)
            ret.append({'path': img_path, 'label': label})
    return ret


def gen_cookie(records, path):
    for x in records:
        path_target = f"{path}/{x['label']}_{x['path'].split('/')[-1]}"
        import shutil
        shutil.copyfile(x['path'], path_target)


def default_loader(path):
    return Image.open(path).convert('RGB')


class AGG4k(data.Dataset):
    name = 'AGG4k'
    labels = ['amusement', 'contentment', 'excitement', 'anger', 'awe', 'disgust', 'sadness', 'fear']

    def __init__(self, root, scale=224, train=True, transform=None, target_transform=None, loader=default_loader):
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.img_dir = f"{root}/{scale}x/{'train' if train else 'test'}"
        img_list = os.listdir(self.img_dir)
        imgs = []
        for img_id in img_list:
            label = int(img_id.split('_')[0])
            imgs.append((img_id, label))
        self.imgs = imgs

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(f'{self.img_dir}/{fn}')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    project_root = '../..'
    labels = ['amusement', 'contentment', 'excitement', 'anger', 'awe', 'disgust', 'sadness', 'fear']
    target_path = project_root + '/cookie/data/AGG4k/origin'
    if not os.path.exists(target_path):
        os.makedirs(target_path)
        os.makedirs(target_path + '/train')
        os.makedirs(target_path + '/test')
    path = project_root + '/dataset_root/AGG4k'
    records = gen_records(labels, path + '/train')
    gen_cookie(records, target_path + '/train')
    records = gen_records(labels, path + '/test')
    gen_cookie(records, target_path + '/test')
    from utils.rescale import rescale_img

    target_path = target_path.replace('/origin', '')
    rescale_size = 224
    rescale_img(target_path + '/origin', target_path + f'/{rescale_size}x', size=rescale_size)

    # agg = AGG4k(project_root+'/cookie/data/AGG4k', scale=224)
