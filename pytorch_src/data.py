from __future__ import print_function, division
import torch

import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset
import random
from torchvision import transforms
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import copy


class Train_Dataset(Dataset):
    def __init__(self, labeled_fname, unlabeled_fname, config):
        self.labeled_data = {}
        self.unlabeled_data = {}
        self.config = config

        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.tensor = transforms.ToTensor()
        self.transforms = transforms.Compose([
            transforms.RandomAffine(10, translate=(0.1, 0.1), shear=2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter()
        ])

        self.len = 1000

        self.total_labeled = 0
        with open(labeled_fname, 'r') as f:
            for l in f.readlines():
                l = l.split(',')
                label = int(l[1])
                if label not in self.labeled_data:
                    self.labeled_data[label] = []
                self.labeled_data[label].append(l[0])
                self.total_labeled += 1

        self.total_unlabeled = 0
        with open(unlabeled_fname, 'r') as f:
            for l in f.readlines():
                l = l.split(',')
                label = int(l[1])
                if label not in self.unlabeled_data:
                    self.unlabeled_data[label] = []
                self.unlabeled_data[label].append(l[0])
                self.total_unlabeled += 1

        with tqdm(total=self.total_labeled) as pbar:
            for key in list(self.labeled_data.keys()):
                for i, fname in enumerate(self.labeled_data[key]):
                    self.labeled_data[key][i] = self.load(fname)
                    pbar.set_description("Loading labeled images")
                    pbar.update(1)

        if self.config.null_space_tuning or self.config.mixmatch:
            with tqdm(total=self.total_unlabeled) as pbar:
                for key in list(self.unlabeled_data.keys()):
                    for i, fname in enumerate(self.unlabeled_data[key]):
                        self.unlabeled_data[key][i] = self.load(fname)
                        pbar.set_description("Loading unlabeled images")
                        pbar.update(1)

            tmp = {}
            for key in list(self.unlabeled_data.keys()):
                indices = np.arange(0, len(self.unlabeled_data[key]))
                random.shuffle(indices)
                if len(indices)%2 != 0:
                    indices = indices[0:-1]
                    self.total_unlabeled -= 1
                for i in range(0,len(indices),2):
                    random_class = random.randint(0, len(self.labeled_data)-1)
                    if random_class not in tmp:
                        tmp[random_class] = []
                    tmp[random_class].append([self.unlabeled_data[key][indices[i]],
                                              self.unlabeled_data[key][indices[i+1]]])

            self.unlabeled_data = tmp

    def assign_class(self, model, device):
        model.eval()
        tmp = {}
        with torch.no_grad():
            with tqdm(total=self.total_unlabeled) as pbar:
                for c in self.unlabeled_data:
                    for i, equiv in enumerate(self.unlabeled_data[c]):
                        avg_logit = torch.zeros(len(self.unlabeled_data))
                        for img in equiv:
                            img = self.test_preprocess(img).to(device)
                            output = model(img.unsqueeze(0))
                            logit = torch.softmax(output, dim=1)
                            avg_logit += logit.cpu().squeeze()

                            pbar.set_description("Reassigning unlabeled image labels")
                            pbar.update(1)

                        avg_logit /= len(equiv)
                        new_class = int(torch.argmax(avg_logit).item())
                        if new_class not in tmp:
                            tmp[new_class] = []
                        tmp[new_class].append(equiv)
        self.unlabeled_data = tmp


    def __len__(self):
        return self.len

    def load(self, fname):
        img = io.imread(fname)
        img = Image.fromarray(img.astype('uint8'))
        return img

    def test_preprocess(self, img):
        img = np.array(img)
        img = img / 255.0
        img = transform.resize(img, [224, 224, 3], mode='constant', anti_aliasing=True)
        img = self.tensor(img)
        img = self.norm(img.float())

        img = img.type(torch.float32)
        return img

    def preprocess(self, img):
        if self.transforms:
            img = self.transforms(img)
            img = np.array(img)
            img = img / 255.0
            img = transform.resize(img, [224, 224, 3], mode='constant', anti_aliasing=True)
            img = self.tensor(img)
            img = self.norm(img.float())
            img = img.type(torch.float32)

        return img

    def __getitem__(self, i):
        label = random.randint(0, len(self.labeled_data)-1)
        idx = random.randint(0, len(self.labeled_data[label])-1)
        img = self.labeled_data[label][idx]

        self.idx = i
        img = self.preprocess(img.copy())

        if self.config.null_space_tuning or self.config.mixmatch:

            ul_label = random.randint(0, len(self.unlabeled_data)-1)
            ul_equiv_idx = random.randint(0, len(self.unlabeled_data[ul_label])-1)
            ul_imgs = self.unlabeled_data[ul_label][ul_equiv_idx]
            idx = random.randint(0, len(ul_imgs) -1)
            ul1 = self.preprocess(ul_imgs[idx].copy())
            ul2 = self.preprocess(ul_imgs[idx].copy())

            idx2 = random.choice([x for x in range(0, len(ul_imgs)) if x != idx])
            ule1 = self.preprocess(ul_imgs[idx2].copy())
            ule2 = self.preprocess(ul_imgs[idx2].copy())


            return {'image': img, \
                    'target': torch.tensor(label), \
                    'ul_img1': ul1,\
                    'ul_img2': ul2,\
                    'ule_img1': ule1,
                    'ule_img2': ule2}
        else:
            return {'image': img, \
                    'target': torch.tensor(label)}



class Test_Dataset(Dataset):
    def __init__(self, fname, config):
        self.data = {}
        with open(fname, 'r') as f:
            for l in f.readlines():
                l = l.split(',')
                self.data[l[0]] = int(l[1])
        self.keys = list(self.data.keys())

        with tqdm(total=len(self.keys)) as pbar:
            for key in self.keys:
                self.data[key] = (self.data[key], self.load(key))
                pbar.set_description("Loading images")
                pbar.update(1)

        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.tensor = transforms.ToTensor()
        self.config = config

    def __len__(self):
        return len(self.keys)

    def load(self, fname):
        img = io.imread(fname)
        return img

    def preprocess(self, img):
        img = img / 255.0

        img = transform.resize(img, [224, 224, 3], mode='constant', anti_aliasing=True)
        img = self.tensor(img)
        img = self.norm(img.float())

        img = img.type(torch.float32)
        return img

    def __getitem__(self, idx):
        key = self.keys[idx]
        label = self.data[key][0]
        img = self.data[key][1]

        img = self.preprocess(img)

        return {'image': img, \
                'target': torch.tensor(label), \
                'file': key}
