import torch
import os
import math
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import cv2
# from torch.utils.serialization import load_lua
from lib.nn import SynchronizedBatchNorm2d as SynBN2d
import random
from torchvision.ops import DeformConv2d
# from BiSeNet.lib.models import model_factory
# from BiSeNet.configs import set_cfg_from_file
import argparse
from pytorch_msssim import MS_SSIM
from models.transformer.Models import Encoder_patch66
from transformers import AutoModel, AutoImageProcessor
# from kornia.color import rgb_to_lab, rgb_to_hsv
from performer_pytorch import SelfAttention as Performer_self
from performer_pytorch import CrossAttention as Performer_cross
# import models.clip.clip as clip
import torchvision.transforms as transforms
# from ultralytics import YOLO
###############################################################################
# Functions
###############################################################################

def pad_tensor(input):
    height_org, width_org = input.shape[2], input.shape[3]
    divide = 32

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
            pad_top = int(height_div / 2)
            pad_bottom = int(height_div - pad_top)
        else:
            pad_top = 0
            pad_bottom = 0

        padding = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
        input = padding(input)
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    height, width = input.data.shape[2], input.data.shape[3]
    assert width % divide == 0, 'width cant divided by stride'
    assert height % divide == 0, 'height cant divided by stride'

    return input, pad_left, pad_right, pad_top, pad_bottom


def pad_tensor_back(input, pad_left, pad_right, pad_top, pad_bottom):
    height, width = input.shape[2], input.shape[3]
    return input[:, :, pad_top: height - pad_bottom, pad_left: width - pad_right]


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'synBN':
        norm_layer = functools.partial(SynBN2d, affine=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm)
    return norm_layer


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, gpu_ids=[], skip=False,
             opt=None):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())
    if which_model_netG == 'sid_unet_resize':
        netG = Unet_resize_conv(opt, skip)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) >= 0:
        netG.cuda(device=gpu_ids[0])
        netG = torch.nn.DataParallel(netG, gpu_ids)
    netG.apply(weights_init)
    return netG


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, gpu_ids=[], patch=False):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())
    if which_model_netD == 'no_norm_4':
        netD = NoNormDiscriminator(input_nc, ndf, n_layers_D, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'fea_no_norm_4':
        netD = fea_NoNormDiscriminator(input_nc, ndf, n_layers_D, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    if use_gpu:
        netD.cuda(device=gpu_ids[0])
        netD = torch.nn.DataParallel(netD, gpu_ids)
    netD.apply(weights_init)
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        # return self.loss(input, target_tensor)
        # non_target_tensor = self.get_target_tensor(input, not target_is_real)
        loss = self.loss(input, target_tensor)
        return loss


class NoNormDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, use_sigmoid=False, gpu_ids=[]):
        super(NoNormDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids

        kw = 4
        padw = int(np.ceil((kw - 1) / 2))
        sequence = [
            nn.Conv2d(input_nc + 1, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        self.pad = nn.ReflectionPad2d(1)
        self.pool = nn.MaxPool3d(kernel_size=3, stride=1)
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        # if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
        #     return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        # else:
        inputl = (input + 1) / 2.
        attenmap = self.pool(self.pad(inputl)) * 2 - 1
        input = torch.concat([input,attenmap],dim=1)
        return self.model(input)


class fea_NoNormDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=4, use_sigmoid=False, gpu_ids=[]):
        super(fea_NoNormDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids

        kw = 4
        padw = int(np.ceil((kw - 1) / 2))
        sequence = [
            nn.Conv2d(input_nc, ndf * 8, kernel_size=kw, stride=2, padding=padw),
            # CSDN_Tem(input_nc, ndf*8),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 8
        nf_mult_prev = 8
        for n in range(3, 1, -1):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** (n - 1), 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw),
                # CSDN_Tem(ndf * nf_mult_prev, ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = 1
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw),
            # CSDN_Tem(ndf * nf_mult_prev, ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        # sequence += [ CSDN_Tem(ndf * nf_mult, 1)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]
            # sequence += [nn.Tanh()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        # if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
        #     return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        # else:
        return self.model(input)
class CSDN_Tem(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CSDN_Tem, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class Focus(nn.Module):
    def __init__(self, c1, c2):
        super(Focus, self).__init__()
        self.pus = nn.PixelUnshuffle(2)
        self.conv = nn.Conv2d(c1 * 4, c2, kernel_size=1)

    def forward(self, x):
        return self.conv(self.pus(x))
class Gradmap(nn.Module):
    def __init__(self):
        super(Gradmap, self).__init__()
        laplace = torch.FloatTensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]).unsqueeze(0).unsqueeze(0).cuda()
        self.weight_lap = nn.Parameter(data=laplace, requires_grad=True)


    def forward(self, input):
        gradmap = F.conv2d(input, self.weight_lap, padding=1)
        return gradmap
class PerformerMemoryBank(nn.Module):
    def __init__(self, feature_dim=128, capacity=100, top_k=5):
        super().__init__()
        self.sim_threshold = 0.95
        self.feature_dim = feature_dim
        self.capacity = capacity
        self.top_k = top_k
        self.training = True
        self.mem_h = 20
        self.mem_w = 20

        # 初始化预分配内存
        self.similar_feats = torch.zeros(capacity, feature_dim, device='cuda')
        self.original_feats = torch.zeros(capacity, feature_dim, 20, 20, device='cuda')  # 假设空间维度为8x8
        self.access_counts = torch.zeros(capacity, dtype=torch.int64, device='cuda')
        self.current_size = 0

        # 注意力模块
        self.cross_attn = Performer_cross(dim=feature_dim, heads=4, dim_head=feature_dim // 4, causal=False)
        self.self_attn = Performer_self(dim=feature_dim, heads=4, dim_head=feature_dim // 4, causal=False)



        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.gamma_proj = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1),
            nn.SELU(),
            nn.Conv2d(feature_dim, 1, kernel_size=1),
            nn.Sigmoid(),
        )
    def set_mode(self, training=True):
        self.training = training

    def _extract_similarity_feature(self, feat):
        if feat.dim() != 4:
            raise ValueError(f"Input must be 4D (B,C,H,W), got {feat.shape}")
        return F.adaptive_avg_pool2d(feat, (1, 1)).squeeze(-1).squeeze(-1)

    def _calc_similarity(self, query_feat):
        if self.current_size == 0:
            return None

        query_norm = F.normalize(query_feat, dim=-1)  # (B, C)
        memory_norm = F.normalize(self.similar_feats[:self.current_size], dim=-1)  # (current_size, C)
        return F.cosine_similarity(query_norm.unsqueeze(1), memory_norm.unsqueeze(0), dim=2)  # (B, current_size)

    def retrieve_similar(self, query_feat):
        sim_matrix = self._calc_similarity(query_feat) # (B, current_size)
        if sim_matrix is None:
            return None, None, torch.zeros(query_feat.size(0), 1, 1, device=query_feat.device)
        self.top_k = min(self.current_size, self.top_k)
        top_scores, top_indices = torch.topk(sim_matrix, k=self.top_k, dim=1) #(B, top_k)
        top_indices = top_indices.clamp(0, self.current_size - 1)

        if self.training and self.access_counts is not None:
            # 向量化更新访问计数
            counts = torch.zeros_like(self.access_counts)
            counts.scatter_add_(0, top_indices.view(-1), torch.ones_like(top_indices.view(-1)))
            self.access_counts += counts

        B, k = top_indices.shape
        selected_feats = self.original_feats[top_indices.view(-1)].view(B, k, self.original_feats.shape[1], self.original_feats.shape[2], self.original_feats.shape[3])
        avg_sim = top_scores.mean(dim=1, keepdim=True).view(-1, 1, 1)

        return selected_feats, top_indices, avg_sim

    def add_feature(self, sim_feat, original_feat):
        if not self.training:
            return

        B = sim_feat.size(0)
        device = sim_feat.device
        new_sim = sim_feat.to(device)
        new_original = original_feat.to(device)

        with torch.no_grad():
            non_dup_new_sim = new_sim
            non_dup_new_original = new_original
            B_non_dup = B

            if self.current_size > 0:
                sim_with_memory = self._calc_similarity(new_sim)
                if sim_with_memory is not None:
                    is_duplicate_new = (sim_with_memory > self.sim_threshold).any(dim=1)
                    non_dup_new_sim = new_sim[~is_duplicate_new]
                    non_dup_new_original = new_original[~is_duplicate_new]
                    B_non_dup = non_dup_new_sim.size(0)
                    if B_non_dup == 0:
                        return


        with torch.no_grad():
            total_needed = self.current_size + B_non_dup
            if total_needed > self.capacity:
                need_replace = total_needed - self.capacity


                start_idx = 0 if self.current_size > 0 else 0
                candidate_idx = torch.arange(start_idx, self.current_size, device=device)
                if len(candidate_idx) == 0:
                    candidate_idx = torch.arange(self.current_size, device=device)
                select_num = min(need_replace, len(candidate_idx))
                _, replace_idx_in_candidate = torch.topk(
                    self.access_counts[candidate_idx], k=select_num, largest=False
                )
                replace_idx = candidate_idx[replace_idx_in_candidate]


                front_idx = torch.arange(select_num, device=device)


                front_sim = self.similar_feats[front_idx].clone()
                front_original = self.original_feats[front_idx].clone()
                front_counts = self.access_counts[front_idx].clone()


                self.similar_feats[replace_idx] = front_sim
                self.original_feats[replace_idx] = front_original
                self.access_counts[replace_idx] = front_counts


                self.similar_feats[front_idx] = non_dup_new_sim[:select_num].clone()
                self.original_feats[front_idx] = non_dup_new_original[:select_num].clone()
                self.access_counts[front_idx] = 0

                self.current_size = self.capacity

            else:
                self.similar_feats[B_non_dup: self.current_size + B_non_dup] = self.similar_feats[
                                                                               :self.current_size].clone()
                self.original_feats[B_non_dup: self.current_size + B_non_dup] = self.original_feats[
                                                                                :self.current_size].clone()
                self.access_counts[B_non_dup: self.current_size + B_non_dup] = self.access_counts[
                                                                               :self.current_size].clone()

                self.similar_feats[:B_non_dup] = non_dup_new_sim.clone()
                self.original_feats[:B_non_dup] = non_dup_new_original.clone()
                self.access_counts[:B_non_dup] = 0

                self.current_size += B_non_dup

    def forward(self, x):
        B, C, H, W = x.shape

        x_flat = x.flatten(2).transpose(1, 2)  # [B, HW, C]
        sim_feat = self._extract_similarity_feature(x)  # [B, C]

        sa_out = self.self_attn(x_flat).transpose(1, 2).view(B, C, H, W)

        memory_feats, _, avg_sim = self.retrieve_similar(sim_feat)  # memory_feats: [B, k, C, mem_h, mem_w] or None

        if memory_feats is not None:

            B, k, C0, mh, mw = memory_feats.shape
            context = memory_feats.permute(0, 1, 3, 4, 2).contiguous().view(B, k * (mh * mw), C)  # [B, k*L_mem, C]

            ca_out = self.cross_attn(x_flat, context=context)  # [B, mem_h*mem_w, C]
            ca_out = ca_out.transpose(1, 2).view(B, C, H, W)

            gamma_map = self.gamma_proj(torch.concat([sa_out, ca_out], dim=1))
            out = (1 - gamma_map) * sa_out + gamma_map * ca_out

        else:

            out = sa_out

        if self.training:

            self.add_feature(sim_feat, x)


        return self.gamma * x + out


class Unet_resize_conv(nn.Module):
    def __init__(self, opt, skip):
        super(Unet_resize_conv, self).__init__()

        self.opt = opt
        self.skip = skip
        p = 1
        self.pad = nn.ReflectionPad2d(1)
        self.pool = nn.MaxPool3d(kernel_size=3, stride=1)

        self.conv0n_1 = nn.Conv2d(3, 4, 3, padding=p)
        self.conv0n_2 = nn.Conv2d(4, 4, 3, padding=p)
        self.conv0n_3 = nn.Conv2d(4, 1, 3, padding=p)
        self.Act0n1 = nn.SELU()
        self.Act0n2 = nn.SELU()
        self.Act0n3 = nn.SELU()

        self.conv0_1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv0_2 = nn.Conv2d(6, 8, 3, padding=1)




        self.conv1_1l = nn.Conv2d(8, 8, 3, padding=p)
        self.Act1_1l = nn.SELU(inplace=True)
        self.bn1_1l = nn.BatchNorm2d(8)

        self.conv1_2l = nn.Conv2d(16, 16, 3, padding=p)
        self.Act1_2l = nn.SELU(inplace=True)
        self.bn1_2l = nn.BatchNorm2d(16)

        self.conv1_3l = nn.Conv2d(8, 8, 3, padding=p)
        self.Act1_3l = nn.SELU(inplace=True)
        self.bn1_3l = nn.BatchNorm2d(8)

        self.conv1_4l = nn.Conv2d(16, 8, 1)

        self.conv2_1l = nn.Conv2d(16, 16, 3, padding=p)
        self.Act2_1l = nn.SELU(inplace=True)
        self.bn2_1l = nn.BatchNorm2d(16)

        self.conv2_2l = nn.Conv2d(32, 32, 3, padding=p)
        self.Act2_2l = nn.SELU(inplace=True)
        self.bn2_2l = nn.BatchNorm2d(32)

        self.conv2_3l = nn.Conv2d(16, 16, 3, padding=p)
        self.Act2_3l = nn.SELU(inplace=True)
        self.bn2_3l = nn.BatchNorm2d(16)

        self.conv2_4l = nn.Conv2d(32, 16, 1)

        self.conv3_1l = nn.Conv2d(32, 32, 3, padding=p)
        self.Act3_1l = nn.SELU(inplace=True)
        self.bn3_1l = nn.BatchNorm2d(32)

        self.conv3_2l = nn.Conv2d(64, 64, 3, padding=p)
        self.Act3_2l = nn.SELU(inplace=True)
        self.bn3_2l = nn.BatchNorm2d(64)

        self.conv3_3l = nn.Conv2d(32, 32, 3, padding=p)
        self.Act3_3l = nn.SELU(inplace=True)
        self.bn3_3l = nn.BatchNorm2d(32)

        self.conv3_4l = nn.Conv2d(64, 32, 1)

        self.conv4_1l = nn.Conv2d(64, 64, 3, padding=p)
        self.Act4_1l = nn.SELU(inplace=True)
        self.bn4_1l = nn.BatchNorm2d(64)

        self.conv4_2l = nn.Conv2d(128, 128, 3, padding=p)
        self.Act4_2l = nn.SELU(inplace=True)
        self.bn4_2l = nn.BatchNorm2d(128)

        self.conv4_3l = nn.Conv2d(64, 64, 3, padding=p)
        self.Act4_3l = nn.SELU(inplace=True)
        self.bn4_3l = nn.BatchNorm2d(64)

        self.conv4_4l = nn.Conv2d(128, 64, 1)




        self.conv1_1 = nn.Conv2d(8, 8, 3, padding=p)
        self.Act1_1 = nn.SELU(inplace=True)
        self.bn1_1 = nn.BatchNorm2d(8)

        self.conv1_2 = nn.Conv2d(16, 16, 3, padding=p)
        self.Act1_2 = nn.SELU(inplace=True)
        self.bn1_2 = nn.BatchNorm2d(16)
        self.max_pool1 = nn.MaxPool2d(2)

        self.focus1 = Focus(8, 8)
        self.conv1_3 = nn.Conv2d(8, 8, 3, padding=p)
        self.Act1_3 = nn.SELU(inplace=True)
        self.bn1_3 = nn.BatchNorm2d(8)

        self.conv1_4 = nn.Conv2d(16, 8, 1)

        self.conv2_1 = nn.Conv2d(16, 16, 3, padding=p)
        self.Act2_1 = nn.SELU(inplace=True)
        self.bn2_1 = nn.BatchNorm2d(16)

        self.conv2_2 = nn.Conv2d(32, 32, 3, padding=p)
        self.Act2_2 = nn.SELU(inplace=True)
        self.bn2_2 = nn.BatchNorm2d(32)
        self.max_pool2 = nn.MaxPool2d(2)

        self.focus2 = Focus(16, 16)
        self.conv2_3 = nn.Conv2d(16, 16, 3, padding=p)
        self.Act2_3 = nn.SELU(inplace=True)
        self.bn2_3 = nn.BatchNorm2d(16)

        self.conv2_4 = nn.Conv2d(32, 16, 1)

        self.conv3_1 = nn.Conv2d(32, 32, 3, padding=p)
        self.Act3_1 = nn.SELU(inplace=True)
        self.bn3_1 = nn.BatchNorm2d(32)

        self.conv3_2 = nn.Conv2d(64, 64, 3, padding=p)
        self.Act3_2 = nn.SELU(inplace=True)
        self.bn3_2 = nn.BatchNorm2d(64)
        self.max_pool3 = nn.MaxPool2d(2)

        self.focus3 = Focus(32, 32)
        self.conv3_3 = nn.Conv2d(32, 32, 3, padding=p)
        self.Act3_3 = nn.SELU(inplace=True)
        self.bn3_3 = nn.BatchNorm2d(32)

        self.conv3_4 = nn.Conv2d(64, 32, 1)

        self.conv4_1 = nn.Conv2d(64, 64, 3, padding=p)
        self.Act4_1 = nn.SELU(inplace=True)
        self.bn4_1 = nn.BatchNorm2d(64)

        self.conv4_2 = nn.Conv2d(128, 128, 3, padding=p)
        self.Act4_2 = nn.SELU(inplace=True)
        self.bn4_2 = nn.BatchNorm2d(128)
        self.max_pool4 = nn.MaxPool2d(2)

        self.focus4 = Focus(64, 64)
        self.conv4_3 = nn.Conv2d(64, 64, 3, padding=p)
        self.Act4_3 = nn.SELU(inplace=True)
        self.bn4_3 = nn.BatchNorm2d(64)

        self.conv4_4 = nn.Conv2d(128, 64, 1)

        self.reflect = nn.ReflectionPad2d(4)


        self.conv5_1 = nn.Conv2d(128, 128, 3, padding=p)
        self.Act5_1 = nn.SELU(inplace=True)
        self.bn5_1 = nn.BatchNorm2d(128)
        self.conv5_2 = nn.Conv2d(256, 256, 3, padding=p) #CSDN_Tem(512,256)
        self.Act5_2 = nn.SELU(inplace=True)
        self.bn5_2 = nn.BatchNorm2d(256)
        self.focus5 = Focus(128, 128)
        self.Act5_3 = nn.SELU(inplace=True)
        self.bn5_3 = nn.BatchNorm2d(128)
        self.conv5_4 = nn.Conv2d(256, 256, 1)

        self.deconv5 = nn.Conv2d(256, 128, 3, padding=p)
        self.conv6_1 = nn.Conv2d(256, 128, 3, padding=p)
        self.Act6_1 = nn.SELU(inplace=True)
        self.bn6_1 = nn.BatchNorm2d(128)


        self.conv6_2 = nn.Conv2d(128, 128, 3, padding=p)
        self.Act6_2 = nn.SELU(inplace=True)
        self.bn6_2 = nn.BatchNorm2d(128)

        self.conv6_3 = nn.Conv2d(128, 128, 3, padding=p)
        self.Act6_3 = nn.SELU(inplace=True)
        self.bn6_3 = nn.BatchNorm2d(128)
        self.conv6_4 = nn.Conv2d(256, 128, 1)

        self.deconv6 = nn.Conv2d(128, 64, 3, padding=p)

        self.conv7_1 = nn.Conv2d(128, 64, 3, padding=p)
        self.Act7_1 = nn.SELU(inplace=True)
        self.bn7_1 = nn.BatchNorm2d(64)


        self.conv7_2 = nn.Conv2d(64, 64, 3, padding=p)
        self.Act7_2 = nn.SELU(inplace=True)
        self.bn7_2 = nn.BatchNorm2d(64)

        self.conv7_3 = nn.Conv2d(64, 64, 3, padding=p)
        self.Act7_3 = nn.SELU(inplace=True)
        self.bn7_3 = nn.BatchNorm2d(64)
        self.conv7_4 = nn.Conv2d(128, 64, 1)

        self.deconv7 = nn.Conv2d(64, 32, 3, padding=p)

        self.conv8_1 = nn.Conv2d(64, 32, 3, padding=p)
        self.Act8_1 = nn.SELU(inplace=True)
        self.bn8_1 = nn.BatchNorm2d(32)


        self.conv8_2 = nn.Conv2d(32, 32, 3, padding=p)
        self.Act8_2 = nn.SELU(inplace=True)
        self.bn8_2 = nn.BatchNorm2d(32)

        self.conv8_3 = nn.Conv2d(32, 32, 3, padding=p)
        self.Act8_3 = nn.SELU(inplace=True)
        self.bn8_3 = nn.BatchNorm2d(32)
        self.conv8_4 = nn.Conv2d(64, 32, 1)

        self.deconv8 = nn.Conv2d(32, 16, 3, padding=p)

        self.conv9_1 = nn.Conv2d(32, 16, 3, padding=p)
        self.Act9_1 = nn.SELU(inplace=True)
        self.bn9_1 = nn.BatchNorm2d(16)


        self.conv9_2 = nn.Conv2d(16, 16, 3, padding=p)
        self.Act9_2 = nn.SELU(inplace=True)
        self.bn9_2 = nn.BatchNorm2d(16)

        self.conv9_3 = nn.Conv2d(16, 16, 3, padding=p)
        self.Act9_3 = nn.SELU(inplace=True)
        self.bn9_3 = nn.BatchNorm2d(16)
        self.conv9_4 = nn.Conv2d(32, 16, 1)

        self.conv10 = nn.Conv2d(16, 3, 3, padding=p)

        self.memory_bank = PerformerMemoryBank()

        if self.opt.phase == "train":
            train_phase = True
        else:
            train_phase = False
            checkpoint = torch.load('models/memory_bank.pth')
            self.memory_bank.similar_feats = checkpoint['similar_feats']
            self.memory_bank.original_feats = checkpoint['original_feats']
            self.memory_bank.access_counts = checkpoint['access_counts']
            self.memory_bank.current_size = checkpoint['current_size']



        self.memory_bank.set_mode(training=train_phase)

    def rgb_to_gray(self, img):
        return 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]

    def _max_scale(self, x):
        x_max = x.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0] + 1e-8
        return x / x_max

    def color_cast_map(self, img):
        mean_rgb = img.mean(dim=[2, 3], keepdim=True)  # 全局均值基准
        deviation = (img - mean_rgb).pow(2).sum(dim=1, keepdim=True).sqrt()
        return self._max_scale(deviation)  # 值越大偏色越严重

    def blur_map(self, img):
        gray = self.rgb_to_gray(img)

        lap_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                                  dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
        edge_strength = F.conv2d(gray, lap_kernel, padding=1).abs()

        return self._max_scale(edge_strength)

    def noise_map(self, img):
        pad = nn.ReflectionPad2d(1)
        avg_pool = nn.AvgPool3d(3, stride=1)
        blurred = avg_pool(pad(img))
        noise_residual = (img - blurred).abs().mean(dim=1, keepdim=True)

        return self._max_scale(noise_residual)

    def get_quality_maps(self, img):
        blur = self.blur_map(img)
        noise = self.noise_map(img)
        return torch.cat([blur, noise], dim=1)  # (B,3,H,W)
    def forward(self, input, reverse = False, vis = False):
        flag = 0
        if self.opt.flag:
            if input.size()[3] > 2200:
                avg = nn.AvgPool2d(2)
                input = avg(input)
                flag = 1


        input, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(input)

        input1 = (input + 1) / 2.
        attenmap = self.pool(self.pad(input1))
        attenmap = attenmap + 0.0001
        colormap = torch.clamp(input1 / attenmap, 0,1)
        maps = self.get_quality_maps(colormap)
        attenmap = torch.abs(1 - attenmap)
        color = self.color_cast_map(colormap)


        if self.opt.use_norm == 1:

            x = self.Act0n2(self.conv0n_2(self.Act0n1(self.conv0n_1(torch.concat([maps, attenmap], dim=1)))))
            mask = self.Act0n3(self.conv0n_3(x))
            mask = torch.concat([mask, color, attenmap], dim=1)

            conv0 = self.conv0_1(input)
            conv0l = self.conv0_2(torch.concat([-input, mask],dim=1))
            x1_1 = self.bn1_1(self.Act1_1(self.conv1_1(conv0)))
            x1_1l = self.bn1_1l(self.Act1_1l(self.conv1_1l(conv0l)))
            x1_2 = self.bn1_3(self.Act1_3(F.interpolate(self.conv1_3(self.focus1(conv0)), scale_factor=2, mode='bilinear')))
            x1_2l = self.bn1_3l(self.Act1_3l(F.interpolate(self.conv1_3l(self.focus1(conv0l)), scale_factor=2, mode='bilinear')))
            x = self.conv1_4(torch.concat([x1_1, x1_1l], dim=1))
            xl = self.conv1_4l(torch.concat([x1_2, x1_2l], dim=1))
            conv13 = self.bn1_2(self.Act1_2(self.conv1_2(torch.concat([x, conv0], dim=1))))
            conv1 = self.max_pool1(conv13)
            conv13l = self.bn1_2l(self.Act1_2l(self.conv1_2l(torch.concat([xl, conv0l], dim=1))))
            conv1l = self.max_pool1(conv13l)


            x2_1 = self.bn2_1(self.Act2_1(self.conv2_1(conv1)))
            x2_1l = self.bn2_1l(self.Act2_1l(self.conv2_1l(conv1l)))
            x2_2 = self.bn2_3(
                self.Act2_3(F.interpolate(self.conv2_3(self.focus2(conv1)), scale_factor=2, mode='bilinear')))
            x2_2l = self.bn2_3l(
                self.Act2_3l(F.interpolate(self.conv2_3l(self.focus2(conv1l)), scale_factor=2, mode='bilinear')))
            x = self.conv2_4(torch.concat([x2_1, x2_1l], dim=1))
            xl = self.conv2_4l(torch.concat([x2_2, x2_2l], dim=1))
            conv23 = self.bn2_2(self.Act2_2(self.conv2_2(torch.concat([x, conv1], dim=1))))
            conv2 = self.max_pool2(conv23)
            conv23l = self.bn2_2l(self.Act2_2l(self.conv2_2l(torch.concat([xl, conv1l], dim=1))))
            conv2l = self.max_pool2(conv23l)


            conv2p = self.reflect(conv2)
            conv2lp = self.reflect(conv2l)
            fft = torch.fft.fft2(conv2p, dim=(-2, -1), norm='ortho')
            mag = torch.abs(fft)
            phase = torch.angle(fft)
            fftl = torch.fft.fft2(conv2lp, dim=(-2, -1), norm='ortho')
            magl = torch.abs(fftl)
            phasel = torch.angle(fftl)
            x3_1 = self.bn3_1(self.Act3_1(self.conv3_1(phase)))
            x3_1l = self.bn3_1l(self.Act3_1l(self.conv3_1l(phasel)))
            x3_2 = self.bn3_3(
                self.Act3_3(F.interpolate(self.conv3_3(self.focus3(mag)), scale_factor=2, mode='bilinear')))
            x3_2l = self.bn3_3l(
                self.Act3_3l(F.interpolate(self.conv3_3l(self.focus3(magl)), scale_factor=2, mode='bilinear')))
            x = self.conv3_4(torch.concat([x3_1, x3_1l], dim=1))
            xl = self.conv3_4l(torch.concat([x3_2, x3_2l], dim=1))
            real = (x3_2) * torch.cos(x)
            imag = (x3_2) * torch.sin(x)
            x = torch.complex(real, imag)
            x = torch.fft.ifft2(x, norm='ortho').real  # 舍弃虚部
            real = xl * torch.cos(x3_1l)
            imag = xl * torch.sin(x3_1l)
            xl = torch.complex(real, imag)
            xl = torch.fft.ifft2(xl, norm='ortho').real  # 舍弃虚部
            x = x[:, :, 4:-4, 4:-4]
            xl = xl[:, :, 4:-4, 4:-4]
            conv33 = self.bn3_2(self.Act3_2(self.conv3_2(torch.concat([x, conv2], dim=1))))
            conv3 = self.max_pool3(conv33)
            conv33l = self.bn3_2l(self.Act3_2l(self.conv3_2l(torch.concat([xl, conv2l], dim=1))))
            conv3l = self.max_pool3(conv33l)

            # conv3l = conv3 + conv3l
            conv3p = self.reflect(conv3)
            conv3lp = self.reflect(conv3l)
            fft = torch.fft.fft2(conv3p, dim=(-2, -1), norm='ortho')
            mag = torch.abs(fft)
            phase = torch.angle(fft)
            fftl = torch.fft.fft2(conv3lp, dim=(-2, -1), norm='ortho')
            magl = torch.abs(fftl)
            phasel = torch.angle(fftl)
            x4_1 = self.bn4_1(self.Act4_1(self.conv4_1(phase)))
            x4_1l = self.bn4_1l(self.Act4_1l(self.conv4_1l(phasel)))
            x4_2 = self.bn4_3(
                self.Act4_3(F.interpolate(self.conv4_3(self.focus4(mag)), scale_factor=2, mode='bilinear')))
            x4_2l = self.bn4_3l(
                self.Act4_3l(F.interpolate(self.conv4_3l(self.focus4(magl)), scale_factor=2, mode='bilinear')))
            x = self.conv4_4(torch.concat([x4_1, x4_1l], dim=1))
            xl = self.conv4_4l(torch.concat([x4_2, x4_2l], dim=1))
            real = (x4_2) * torch.cos(x)
            imag = (x4_2) * torch.sin(x)
            x = torch.complex(real, imag)
            x = torch.fft.ifft2(x, norm='ortho').real  # 舍弃虚部
            real = xl * torch.cos(x4_1l)
            imag = xl * torch.sin(x4_1l)
            xl = torch.complex(real, imag)
            xl = torch.fft.ifft2(xl, norm='ortho').real  # 舍弃虚部
            x = x[:, :, 4:-4, 4:-4]
            xl = xl[:, :, 4:-4, 4:-4]
            conv43 = self.bn4_2(self.Act4_2(self.conv4_2(torch.concat([x, conv3], dim=1))))
            conv4 = self.max_pool4(conv43)
            conv43l = self.bn4_2l(self.Act4_2l(self.conv4_2l(torch.concat([xl, conv3l], dim=1))))
            conv4l = self.max_pool4(conv43l)


            x5_1 = self.bn5_1(self.Act5_1(self.conv5_1(conv4)))
            x5_2 = self.bn5_3(self.Act5_3(self.memory_bank(conv4l)))
            x = self.conv5_4(torch.concat([x5_1, x5_2], dim=1))
            conv53 = self.bn5_2(self.Act5_2(self.conv5_2(x)))


            conv5 = F.interpolate(conv53, scale_factor=2, mode='bilinear')
            deconv5 = self.deconv5(conv5) + conv43 + conv43l
            up6 = torch.cat([deconv5, conv43], 1)
            up61 = torch.cat([deconv5, conv43l], 1)
            x6_1 = self.bn6_1(self.Act6_1(self.conv6_1(up6)))
            x6_12 = self.bn6_1(self.Act6_1(self.conv6_1(up61)))
            x6_21 = self.bn6_3(
                self.Act6_3(F.interpolate(self.conv6_3(self.focus5(deconv5)), scale_factor=2, mode='bilinear')))
            x6_22 = self.bn6_3(
                self.Act6_3(F.interpolate(self.conv6_3(self.focus5(self.focus5(deconv5))), scale_factor=4, mode='bilinear')))
            x = deconv5 + self.conv6_4(torch.concat([x6_1 + x6_12, x6_21 + x6_22], dim=1))
            conv63 = self.bn6_2(self.Act6_2(self.conv6_2(x)))

            conv6 = F.interpolate(conv63, scale_factor=2, mode='bilinear')
            deconv6 = self.deconv6(conv6) + conv33 + conv33l
            up7 = torch.cat([deconv6, conv33], 1)
            up71 = torch.cat([deconv6, conv33l], 1)
            x7_1 = self.bn7_1(self.Act7_1(self.conv7_1(up7)))
            x7_12 = self.bn7_1(self.Act7_1(self.conv7_1(up71)))
            x7_21 = self.bn7_3(
                self.Act7_3(F.interpolate(self.conv7_3(self.focus4(deconv6)), scale_factor=2, mode='bilinear')))
            x7_22 = self.bn7_3(
                self.Act7_3(F.interpolate(self.conv7_3(self.focus4(self.focus4(deconv6))), scale_factor=4, mode='bilinear')))
            x = deconv6 + self.conv7_4(torch.concat([x7_1 + x7_12, x7_21 + x7_22], dim=1))
            conv73 = self.bn7_2(self.Act7_2(self.conv7_2(x)))

            conv7 = F.interpolate(conv73, scale_factor=2, mode='bilinear')
            deconv7 = self.deconv7(conv7) + conv23 + conv23l
            up8 = torch.cat([deconv7, conv23], 1)
            up81 = torch.cat([deconv7, conv23l], 1)
            x8_1 = self.bn8_1(self.Act8_1(self.conv8_1(up8)))
            x8_12 = self.bn8_1(self.Act8_1(self.conv8_1(up81)))
            x8_21 = self.bn8_3(
                self.Act8_3(F.interpolate(self.conv8_3(self.focus3(deconv7)), scale_factor=2, mode='bilinear')))
            x8_22 = self.bn8_3(
                self.Act8_3(F.interpolate(self.conv8_3(self.focus3(self.focus3(deconv7))), scale_factor=4, mode='bilinear')))
            x = deconv7 + self.conv8_4(torch.concat([x8_1 + x8_12, x8_21 + x8_22], dim=1))
            conv83 = self.bn8_2(self.Act8_2(self.conv8_2(x)))

            conv8 = F.interpolate(conv83, scale_factor=2, mode='bilinear')
            deconv8 = self.deconv8(conv8) + conv13 + conv13l
            up9 = torch.cat([deconv8, conv13], 1)
            up91 = torch.cat([deconv8, conv13l], 1)
            x9_1 = self.bn9_1(self.Act9_1(self.conv9_1(up9)))
            x9_12 = self.bn9_1(self.Act9_1(self.conv9_1(up91)))
            x9_21 = self.bn9_3(
                self.Act9_3(F.interpolate(self.conv9_3(self.focus2(deconv8)), scale_factor=2, mode='bilinear')))
            x9_22 = self.bn9_3(
                self.Act9_3(F.interpolate(self.conv9_3(self.focus2(self.focus2(deconv8))), scale_factor=4, mode='bilinear')))
            x = deconv8 + self.conv9_4(torch.concat([x9_1 + x9_12, x9_21 + x9_22], dim=1))
            conv9 = self.Act9_2(self.conv9_2(x))
            latent = self.conv10(conv9)
            output = latent + input

            output = pad_tensor_back(output, pad_left, pad_right, pad_top, pad_bottom)

            if flag:
                output = F.interpolate(output, scale_factor=2, mode='bilinear')


            if self.opt.phase=="train":
                torch.save({
                    'similar_feats': self.memory_bank.similar_feats,
                    'original_feats': self.memory_bank.original_feats,
                    'access_counts': self.memory_bank.access_counts,
                    'current_size': self.memory_bank.current_size
                }, 'models/memory_bank.pth')

                if vis:
                    return output, x5_1, x5_2, mask
                else:
                    return output, x5_1, x5_2
            else:
                if vis:
                    return output, x5_1, x5_2, mask
                else:
                    return output, x5_1, x5_2
class YOLOv8FeatureLoss(nn.Module):
    def __init__(self, model_path="yolov8n-seg.pt", device="cuda"):
        super().__init__()
        self.yolo = YOLO(model_path)
        self.model = self.yolo.model
        self.model.to(device).eval()


        for p in self.model.parameters():
            p.requires_grad = False

        self.device = device
        self.feature_weight = 1
        self.input_size = (320, 320)


        self.neck_feats = []
        def hook_fn(module, inp, out):
            self.neck_feats.append(out)


        self.model.model[1].register_forward_hook(hook_fn)
        self.cosSim = nn.CosineSimilarity(dim=1, eps=1e-6)

    def _preprocess(self, img_tensor):
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)
        if img_tensor.shape[2:] != self.input_size:
            img_tensor = F.interpolate(img_tensor, size=self.input_size,
                                       mode="bilinear", align_corners=False)
        img_tensor = img_tensor * 0.5 + 0.5
        return img_tensor.to(self.device).float()

    def _extract_features(self, img_tensor):
        self.neck_feats.clear()
        _ = self.model(img_tensor)
        feat = self.neck_feats[-1]
        if isinstance(feat, (list, tuple)):
            feat = feat[0]
        return F.adaptive_avg_pool2d(feat, (8, 8))

    def forward(self, original_img, original_img1, enhanced_img, enhanced_img1):
        orig_prep = self._preprocess(original_img)
        orig_prep1 = self._preprocess(original_img1)
        enh_prep  = self._preprocess(enhanced_img)
        enh_prep1 = self._preprocess(enhanced_img1)

        orig_feats = self._extract_features(orig_prep)
        orig_feats1 = self._extract_features(orig_prep1)
        enh_feats  = self._extract_features(enh_prep)
        enh_feats1 = self._extract_features(enh_prep1)
        loss = self.feature_weight * ( torch.mean((enh_feats - orig_feats)**2) + torch.mean((enh_feats1 - orig_feats)**2)
                                       + torch.mean((enh_feats - orig_feats1)**2) + torch.mean((enh_feats1 - orig_feats1)**2))
        return loss
class DinoFeatureLoss(nn.Module):
    def __init__(self, model_name="facebook/dinov2-small", feature_dim=384, mapped_dim=384):
        super().__init__()
        model_name = "models/models/huggingface/hub/models--facebook--dinov2-small/snapshots/ed25f3a31f01632728cabb09d1542f84ab7b0056"
        self.dino = AutoModel.from_pretrained(model_name)
        self.dino.eval()
        for p in self.dino.parameters():
            p.requires_grad = False

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.cosSim = nn.CosineSimilarity(dim=1, eps=1e-6)

    def preprocess(self, x):
        x = (x + 1.0) / 2.0
        return (x - self.mean.cuda()) / self.std.cuda()

    def extract_feat(self, x):
        return self.dino(pixel_values=x).last_hidden_state[:, 0]  # CLS token

    def forward(self, target, target1, pred0, pred1=None, nograd=False):
        """
        pred, target: shape (B, 3, H, W), value range [-1, 1]
        """
        pred0 = self.preprocess(pred0)
        target = self.preprocess(target)
        target1 = self.preprocess(target1)
        pred1 = self.preprocess(pred1)


        feat_pred0 = self.extract_feat(pred0)
        feat_target = self.extract_feat(target)
        feat_target1 = self.extract_feat(target1)
        feat_pred1 = self.extract_feat(pred1)


        if pred1 is None:
            loss = 1 - self.cosSim(feat_pred0, feat_target).mean()
        else:
            if nograd:
                loss = 0.25 * (torch.mean((feat_pred0 - feat_target) ** 2) + torch.mean((feat_pred1 - feat_target) ** 2)
                            + torch.mean((feat_pred0 - feat_target1) ** 2) + torch.mean((feat_pred1 - feat_target1) ** 2))
            else:
                loss = torch.mean(1 - self.cosSim(feat_pred0, feat_target)) + torch.mean(1 - self.cosSim(feat_pred1, feat_target))


        return loss
class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X, opt, vggLoss=False):
        h = F.relu(self.conv1_1(X), inplace=True)
        h = F.relu(self.conv1_2(h), inplace=True)
        # relu1_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h), inplace=True)
        h = F.relu(self.conv2_2(h), inplace=True)
        # relu2_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h), inplace=True)
        h = F.relu(self.conv3_2(h), inplace=True)
        h = F.relu(self.conv3_3(h), inplace=True)
        relu3_3 = h
        if opt.vgg_choose != "no_maxpool":
            h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h), inplace=True)
        relu4_1 = h
        h = F.relu(self.conv4_2(h), inplace=True)
        relu4_2 = h
        conv4_3 = self.conv4_3(h)
        h = F.relu(conv4_3, inplace=True)
        relu4_3 = h

        if opt.vgg_choose != "no_maxpool":
            if opt.vgg_maxpooling:
                h = F.max_pool2d(h, kernel_size=2, stride=2)

        relu5_1 = F.relu(self.conv5_1(h), inplace=True)
        relu5_2 = F.relu(self.conv5_2(relu5_1), inplace=True)
        conv5_3 = self.conv5_3(relu5_2)
        h = F.relu(conv5_3, inplace=True)
        relu5_3 = h
        if opt.vgg_choose == "conv4_3":
            return conv4_3
        elif opt.vgg_choose == "relu4_2":
            return relu4_2
        elif opt.vgg_choose == "relu4_1":
            return relu4_1
        elif opt.vgg_choose == "relu4_3":
            if not vggLoss:
                return relu4_3
            else:
                return relu4_3, relu3_3
        elif opt.vgg_choose == "conv5_3":
            return conv5_3
        elif opt.vgg_choose == "relu5_1":
            return relu5_1
        elif opt.vgg_choose == "relu5_2":
            return relu5_2
        elif opt.vgg_choose == "relu5_3" or "maxpool":
            return relu5_3


def vgg_preprocess(batch, opt):
    tensortype = type(batch.data)
    (r, g, b) = torch.chunk(batch, 3, dim=1)
    batch = torch.cat((b, g, r), dim=1)  # convert RGB to BGR
    batch = (batch + 1) * 255 * 0.5  # [-1, 1] -> [0, 255]
    if opt.vgg_mean:
        mean = tensortype(batch.data.size())
        mean[:, 0, :, :] = 103.939
        mean[:, 1, :, :] = 116.779
        mean[:, 2, :, :] = 123.680
        batch = batch.sub(Variable(mean))  # subtract mean
    return batch


class PerceptualLoss(nn.Module):
    def __init__(self, opt):
        super(PerceptualLoss, self).__init__()
        self.opt = opt
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.maxpool = nn.MaxPool2d(2)
        self.pad = nn.ReflectionPad2d(1)
        self.pool = nn.MaxPool3d(kernel_size=3, stride=1)

    def vgg_fea1(self, vgg, img):
        img_vgg = vgg_preprocess(img, self.opt)
        img_fea = vgg(img_vgg, self.opt)
        return img_fea

    def vgg_fea(self, vgg, img, target):
        img_vgg = vgg_preprocess(img, self.opt)
        target_vgg = vgg_preprocess(target, self.opt)
        img_fea = vgg(img_vgg, self.opt)
        target_fea = vgg(target_vgg, self.opt)
        return img_fea, target_fea

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img, self.opt)
        target_vgg = vgg_preprocess(target, self.opt)
        img_fea = vgg(img_vgg, self.opt)
        target_fea = vgg(target_vgg, self.opt)
        loss = torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)
        return loss

    def compute_vgg_loss1(self, vgg, img,  target):
        img_vgg = vgg_preprocess(img, self.opt)
        target_vgg = vgg_preprocess(target, self.opt)
        img_fea, img_fea_33 = vgg(img_vgg, self.opt, True)
        target_fea, target_fea_33 = vgg(target_vgg, self.opt, True)
        loss1 = 0.5 * (torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2))
        loss2 = 0.5 * (torch.mean((self.instancenorm(img_fea_33) - self.instancenorm(target_fea_33)) ** 2))
        return loss1 + loss2

    def compute_vgg_loss2(self, vgg, img, img1, target, target1):
        img_vgg = vgg_preprocess(img, self.opt)
        img1_vgg = vgg_preprocess(img1, self.opt)
        target_vgg = vgg_preprocess(target, self.opt)
        target1_vgg = vgg_preprocess(target1, self.opt)
        img_fea, img_fea_33 = vgg(img_vgg, self.opt, True)
        img1_fea, img1_fea_33 = vgg(img1_vgg, self.opt, True)
        target_fea, target_fea_33 = vgg(target_vgg, self.opt, True)
        target1_fea, target1_fea_33 = vgg(target1_vgg, self.opt, True)
        loss1 = 0.25 * (torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)
                        + torch.mean((self.instancenorm(img_fea) - self.instancenorm(target1_fea)) ** 2))
        loss2 = 0.5 * (torch.mean((self.instancenorm(img1_fea).detach() - self.instancenorm(img_fea).detach())**2))
        loss3 = 0.25 * (torch.mean((self.instancenorm(img_fea_33) - self.instancenorm(target_fea_33)) ** 2)
                        + torch.mean((self.instancenorm(img_fea_33) - self.instancenorm(target1_fea_33)) ** 2))
        loss4 = 0.5 * (torch.mean((self.instancenorm(img1_fea_33) - self.instancenorm(img_fea_33).detach())**2))
        # loss3 = (torch.mean((self.instancenorm(img1_fea) - self.instancenorm(img_fea))**2))
        return loss1 + loss2 + loss3 + loss4

    def compute_vgg_loss3(self, vgg, img, img1, target, target1):
        img_vgg = vgg_preprocess(img, self.opt)
        img1_vgg = vgg_preprocess(img1, self.opt)
        target_vgg = vgg_preprocess(target, self.opt)
        target1_vgg = vgg_preprocess(target1, self.opt)
        img_fea, img_fea_33 = vgg(img_vgg, self.opt, True)
        img1_fea, img1_fea_33 = vgg(img1_vgg, self.opt, True)
        target_fea, target_fea_33 = vgg(target_vgg, self.opt, True)
        target1_fea, target1_fea_33 = vgg(target1_vgg, self.opt, True)
        loss1 = 0.25 * (torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)
                        + torch.mean((self.instancenorm(img_fea) - self.instancenorm(target1_fea)) ** 2))
        loss2 = 0.25 * (torch.mean((self.instancenorm(img1_fea) - self.instancenorm(target_fea)) ** 2)
                        + torch.mean((self.instancenorm(img1_fea) - self.instancenorm(target1_fea)) ** 2))
        loss3 = 0.25 * (torch.mean((self.instancenorm(img_fea_33) - self.instancenorm(target_fea_33)) ** 2)
                        + torch.mean((self.instancenorm(img_fea_33) - self.instancenorm(target1_fea_33)) ** 2))
        loss4 = 0.25 * (torch.mean((self.instancenorm(img1_fea_33) - self.instancenorm(target_fea_33)) ** 2)
                        + torch.mean((self.instancenorm(img1_fea_33) - self.instancenorm(target1_fea_33)) ** 2))
        # loss3 = (torch.mean((self.instancenorm(img1_fea) - self.instancenorm(img_fea))**2))
        return loss1 + loss2 + loss3 + loss4

def load_vgg16(model_dir, gpu_ids):
    """ Use the model from https://github.com/abhiskk/fast-neural-style/blob/master/neural_style/utils.py """
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    # if not os.path.exists(os.path.join(model_dir, 'vgg16.weight')):
    #     if not os.path.exists(os.path.join(model_dir, 'vgg16.t7')):
    #         os.system('wget https://www.dropbox.com/s/76l3rt4kyi3s8x7/vgg16.t7?dl=1 -O ' + os.path.join(model_dir, 'vgg16.t7'))
    #     vgglua = load_lua(os.path.join(model_dir, 'vgg16.t7'))
    #     vgg = Vgg16()
    #     for (src, dst) in zip(vgglua.parameters()[0], vgg.parameters()):
    #         dst.data[:] = src
    #     torch.save(vgg.state_dict(), os.path.join(model_dir, 'vgg16.weight'))
    vgg = Vgg16()
    # vgg.cuda()
    if len(gpu_ids) > 0:
        vgg.cuda(device=gpu_ids[0])
        vgg.load_state_dict(torch.load(os.path.join(model_dir, 'vgg16.weight')))
        vgg = torch.nn.DataParallel(vgg, gpu_ids)
    else:
        vgg.load_state_dict(torch.load(os.path.join(model_dir, 'vgg16.weight')))
    return vgg

class Colorloss(nn.Module):
    def __init__(self):
        super(Colorloss, self).__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.pool = nn.MaxPool3d(kernel_size=3, stride=1)
        self.pool2d = nn.MaxPool2d(kernel_size=16)
        self.ssim = MS_SSIM(data_range=1.0)
        self.sigmoid = nn.Sigmoid()



    def colorloss(self, input, target):
        target = (target + 1) / 2.
        attenmap0 = self.pool(self.pad(target))
        attenmap0 = attenmap0 + 0.0001
        colormap0 = target / attenmap0

        input = (input + 1) / 2.
        attenmap = self.pool(self.pad(input))
        attenmap = (attenmap + 0.0001)

        loss = torch.mean((input - colormap0 * attenmap) ** 2)

        return loss

    def colorloss1(self, input, target, L):
        target = (target + 1) / 2.
        attenmap_t = self.pool(self.pad(target))
        attenmap_t = attenmap_t + 0.0001
        colormap_t = target / attenmap_t

        input = (input + 1) / 2.
        attenmap = self.pool(self.pad(input))
        attenmap = (attenmap + 0.0001)

        loss1 = torch.mean((input - colormap_t * attenmap) ** 2)
        loss2 = torch.mean((L - attenmap) ** 2)
        return loss1 + loss2

    def colorloss2(self, input, input1, target):
        target = (target + 1) / 2.
        attenmap0 = self.pool(self.pad(target))
        attenmap0 = attenmap0 + 0.0001
        colormap0 = target / attenmap0

        input = (input + 1) / 2.
        attenmap = self.pool(self.pad(input))
        attenmap = (attenmap + 0.0001)

        input1 = (input1 + 1) / 2.
        attenmap1 = self.pool(self.pad(input1))
        attenmap1 = (attenmap1 + 0.0001)

        loss = torch.mean((input1 - input.detach()) ** 2) + torch.mean((input - colormap0 * attenmap) ** 2)

        return loss

    def colorloss3(self, input, input1, target):
        target = (target + 1) / 2.
        attenmap0 = self.pool(self.pad(target))
        attenmap0 = attenmap0 + 0.0001
        colormap0 = torch.clamp(target / attenmap0, 0,1)

        input = (input + 1) / 2.
        attenmap = self.pool(self.pad(input))
        attenmap = (attenmap + 0.0001)

        input1 = (input1 + 1) / 2.
        attenmap1 = self.pool(self.pad(input1))
        attenmap1 = (attenmap1 + 0.0001)

        loss = torch.mean((input1 - colormap0 * attenmap1) ** 2) + torch.mean((input - colormap0 * attenmap) ** 2)

        return loss

















