import math

import torch
from torch import nn
import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset, store_dataset
import random
import util.util as util
import torch.nn.functional as F
def pad_tensor(input):

    height_org, width_org = input.shape[2], input.shape[3]
    divide = 16

    if width_org % divide != 0 or height_org % divide != 0:

        width_res = width_org % divide
        height_res = height_org % divide
        if width_res != 0:
            width_div = divide - width_res
            pad_left = int(width_div / 2)
            pad_right = int(width_div - pad_left)
        else:
            pad_left = 0
            pad_right = 0

        if height_res != 0:
            height_div = divide - height_res
            pad_top = int(height_div  / 2)
            pad_bottom = int(height_div  - pad_top)
        else:
            pad_top = 0
            pad_bottom = 0

            padding = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
            input = padding(input).data
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    height, width = input.shape[2], input.shape[3]
    assert width % divide == 0, 'width cant divided by stride'
    assert height % divide == 0, 'height cant divided by stride'

    return input, pad_left, pad_right, pad_top, pad_bottom

def pad_tensor_back(input, pad_left, pad_right, pad_top, pad_bottom):
    height, width = input.shape[2], input.shape[3]
    return input[:,:, pad_top: height - pad_bottom, pad_left: width - pad_right]


class UnalignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.transform = get_transform(opt)
        if opt.phase == 'train':
            self.dir_B = os.path.join(opt.dataroot, opt.phase + 'D')
            self.B_imgs, self.B_paths = store_dataset(self.dir_B)
            self.B_size = len(self.B_paths)
            self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        else:
            self.dir_A = opt.dataroot

        self.A_imgs, self.A_paths = store_dataset(self.dir_A)
        self.A_size = len(self.A_paths)


    def __getitem__(self, index):
        A_img = self.A_imgs[index % self.A_size]
        A_path = self.A_paths[index % self.A_size]
        if self.opt.phase == 'train':
            B_img = self.B_imgs[index % self.B_size]
            B_path = self.B_paths[index % self.B_size]
            B_img = self.transform(B_img)
        A_img = self.transform(A_img)





        if self.opt.resize_or_crop == 'no':
            input_img = A_img

            A_img_gamma = (A_img.clone() + 1) / 2.
            ra, ga, ba = A_img_gamma[0], A_img_gamma[1], A_img_gamma[2]  # 转换到0-1
            ya = (0.299 * ra + 0.587 * ga + 0.114 * ba)
            A_img_gamma = ya * A_img_gamma + (1 - ya) * torch.pow(A_img_gamma, 1 / 2.1)  # 改进的gamma校正

            A_img_gamma = A_img_gamma * 2 - 1

        else:

            if (not self.opt.no_flip) and random.random() < 0.5:
                idx1 = [i for i in range(A_img.size(2) - 1, -1, -1)]
                idx1 = torch.LongTensor(idx1)
                A_img = A_img.index_select(2, idx1)


                if self.opt.phase == 'train':
                    idx2 = [i for i in range(B_img.size(2) - 1, -1, -1)]
                    idx2 = torch.LongTensor(idx2)
                    B_img = B_img.index_select(2, idx2)
            if (not self.opt.no_flip) and random.random() < 0.5:
                idx1 = [i for i in range(A_img.size(1) - 1, -1, -1)]
                idx1 = torch.LongTensor(idx1)
                A_img = A_img.index_select(1, idx1)
                if self.opt.phase == 'train':
                    idx2 = [i for i in range(B_img.size(1) - 1, -1, -1)]
                    idx2 = torch.LongTensor(idx2)
                    B_img = B_img.index_select(1, idx2)
            if self.opt.vary == 1 and (not self.opt.no_flip) and random.random() < 0.5:

                input_img = (A_img.clone() + 1) / 2.
                if self.opt.light == 1:
                    gamma = random.uniform(1.1, 1.8)
                    input_img = torch.pow(input_img, gamma)  # 随机亮度对比度调整
                elif self.opt.light == 2:
                    times = random.randint(200,400)/100.
                    input_img = input_img/times
                else:
                    gamma = random.uniform(0.5, 1.5)  # 随机取浮点数[0.8,1.7)
                    input_img = torch.pow(input_img, gamma)  # 随机亮度对比度调整
                input_img = input_img * 2 - 1

            else:
                input_img = A_img.clone()

            A_img_gamma = (A_img.clone() + 1) / 2.
            ra, ga, ba = A_img_gamma[0], A_img_gamma[1], A_img_gamma[2]  # 转换到0-1
            ya = (0.299 * ra + 0.587 * ga + 0.114 * ba)
            A_img_gamma = torch.clamp(ya * A_img_gamma + (1 - ya) * torch.pow(A_img_gamma, 1 / 2.1), 0, 1)  # 改进的gamma校正
            A_img_gamma = A_img_gamma * 2 - 1


        if self.opt.phase == 'train':
            return {'A': A_img_gamma,'B': B_img, 'A_img': A_img, 'input_img': input_img,
                        'A_paths': A_path, 'B_paths': B_path}
        else:
            return {'A': A_img_gamma,  'A_img': A_img, 'input_img': input_img,
                    'A_paths': A_path, }

    def __len__(self):
        if self.opt.phase == 'train':
            return max(self.A_size, self.B_size)
        else:
            return self.A_size

    def name(self):
        return 'UnalignedDataset'

