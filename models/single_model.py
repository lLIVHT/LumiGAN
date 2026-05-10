import copy

import numpy as np
import torch
from torch import nn
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
import random
import torch.nn.functional as F
# from . import networks
from pytorch_msssim import MS_SSIM
import matplotlib.pyplot as plt
from PIL import Image

class SingleModel(BaseModel):
    def name(self):
        return 'SingleGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if self.opt.net == 'network1':
            from . import networks1 as networks
        elif self.opt.net == 'network11':
            from . import networks11 as networks
        elif self.opt.net == 'network2':
            from . import networks2 as networks
        elif self.opt.net == 'network3':
            from . import networks3 as networks
        else:
            from . import networks as networks
        nb = opt.batchSize
        size = opt.fineSize
        self.opt = opt
        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)

        self.input_img = self.Tensor(nb, opt.input_nc, size, size)
        self.A_img = self.Tensor(nb, opt.input_nc, size, size)




        self.loss_G = 0
        self.loss_G_A = 0
        self.loss_D_A = 0
        self.loss_D_fea = 0
        self.loss_D_P = 0
        self.loss_G_g = 0
        self.loss_G_fea = 0

        self.ssim = MS_SSIM(data_range=1.0)
        self.colorloss = networks.Colorloss()
        if opt.vgg > 0 or opt.fea == 1:
            self.vgg_loss = networks.PerceptualLoss(opt)
            self.vgg_loss.cuda()
            self.vgg = networks.load_vgg16("./model", self.gpu_ids)
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False
            self.dino_loss = networks.DinoFeatureLoss().cuda()
            # self.clip_loss = networks.CLIPSemanticLoss().cuda()
            # self.yolo_loss = networks.YOLOv8FeatureLoss().cuda()

        skip = True if opt.skip > 0 else False
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids,
                                        skip=skip, opt=opt)


        if self.isTrain:
            use_sigmoid = True
            self.netD_A = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids, False)

            if self.opt.patchD:
                self.netD_P = networks.define_D(opt.input_nc, opt.ndf,
                                                opt.which_model_netD,
                                                opt.n_layers_patchD, opt.norm, use_sigmoid, self.gpu_ids, True)
            if self.opt.fea == 1:
                self.netD_fea = networks.define_D(512, opt.ndf,
                                                  opt.which_model_netD_fea,
                                                  opt.n_layers_patchD, opt.norm, True, self.gpu_ids, True)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch, self.gpu_ids)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch, self.gpu_ids)
                if self.opt.patchD:
                    self.load_network(self.netD_P, 'D_P', which_epoch, self.gpu_ids)
                if self.opt.fea == 1:
                    self.load_network(self.netD_fea, 'D_fea', which_epoch, self.gpu_ids)

        if self.isTrain:
            self.old_lr = opt.lr
            self.old_lr_D = opt.lr_D
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)


            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG_A.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr_D, betas=(opt.beta1, 0.999))
            if self.opt.patchD:
                self.optimizer_D_P = torch.optim.Adam(self.netD_P.parameters(), lr=opt.lr_D, betas=(opt.beta1, 0.999))
            if self.opt.fea == 1:
                self.optimizer_D_fea = torch.optim.Adam(self.netD_fea.parameters(), lr=opt.lr_D, betas=(opt.beta1, 0.999))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)
        if self.isTrain:
            networks.print_network(self.netD_A)
            if self.opt.patchD:
                networks.print_network(self.netD_P)
            if self.opt.fea == 1:
                networks.print_network(self.netD_fea)
        if opt.isTrain:
            self.netG_A.train()
        else:
            self.netG_A.eval()
        print('-----------------------------------------------')

    def set_input(self, input):
        if self.opt.phase == 'train':
            input_B = input['B']
        input_A = input['A']

        input_img = input['input_img']
        A_img= input['A_img']
        self.A_img.resize_(A_img.size()).copy_(A_img)
        self.input_img.resize_(input_img.size()).copy_(input_img)
        self.input_A.resize_(input_A.size()).copy_(input_A)
        if self.opt.phase == 'train':
            self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths']
        self.input_img0 = self.input_img.clone()

    def predict(self):
        with torch.no_grad():
            # import time
            # inp_tensor = torch.rand(1, 3, 1200, 900).cuda()
            # # get time
            # start_time = time.time()
            # _ = self.netG_A.forward(inp_tensor)
            # used_time = time.time() - start_time
            # ave_time = used_time
            # print('Time (S) = {:.4f}'.format(ave_time))
            # self.Visualize_feature_maps()
            vis = False
            self.real_A = Variable(self.input_A)
            self.real_img = Variable(self.input_img)

            if vis:
               self.fake_B, _, _ , maps = self.netG_A.forward(self.real_img, vis=vis)
               fake_B = util.tensor2im(self.fake_B.data)
               maps = (maps * 2 - 1)
               maps = util.tensor2im(maps.data)

               return OrderedDict([('fake_B', fake_B), ('maps', maps)])
            else:
                self.fake_B, _, _ = self.netG_A.forward(self.real_img)
                fake_B = util.tensor2im(self.fake_B.data)
                return OrderedDict([('fake_B', fake_B)])
    def Visualize_feature_maps(self):

        self.real_A = Variable(self.input_A)
        self.real_img = Variable(self.input_img)
        model = self.netG_A
        # 选择你感兴趣的层（可以是卷积层，激活层等）
        layer_names = ["module.conv0","module.bn2_1","module.bn2_3","module.bn2_2", "module.bn4_1","module.bn4_3","module.bn4_2",
                       "module.bn5_1","module.bn5_3","module.bn5_2","module.bn8_1","module.bn8_3","module.bn8_2"]  # ResNet的层名，可以根据需要选择
        intermediate_outputs = []
        hooks = []

        # 钩子函数：提取中间层的输出
        def hook_fn(module, input, output):
            intermediate_outputs.append(output)

        # 注册钩子以提取特定层的输出
        for name, module in model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(hook_fn)
                hooks.append(hook)


        # 前向传播
        with torch.no_grad():
            self.fake_B,_,_ = self.netG_A.forward(self.real_img)

        # 选择要显示的通道
        def select_channels(layer_output, num_channels=16):
            # 计算每个通道的平均激活值
            # layer_output = layer_output.unsqueeze(0)  # 去掉batch维度，得到 [num_channels]
            channel_mean = layer_output.mean(dim=(1, 2))  # [num_channels]


            # 按照响应强度排序，选择前num_channels个通道
            top_channels = torch.topk(channel_mean, num_channels, dim=0).indices
            return top_channels

        # 可视化特征图
        for i, layer_output in enumerate(intermediate_outputs):
            print(f"Layer {layer_names[i]} output shape: {layer_output.shape}")

            # 获取该层的特征图
            feature_maps = layer_output[0]  # [num_channels, height, width]，即选择batch size为1的输出

            # 选择前16个响应最强的通道
            top_channels = select_channels(feature_maps, num_channels=16)

            # 可视化选定的特征图
            plt.figure(figsize=(15, 15))
            for j, channel in enumerate(top_channels):
                name = layer_names[i].split(".")[-1]
                plt.subplot(4, 4, j + 1)
                plt.imshow(feature_maps[channel].cpu().detach().numpy(), cmap='viridis')
                plt.axis('off')
            plt.suptitle(f"feature maps from layer: {name}",fontsize=40)
            # 使用tight_layout()去除留白
            plt.tight_layout(pad=0.5)  # 减小pad来减少间距
            plt.subplots_adjust(hspace=0.005, wspace=0.05)  # 增加行间距和列间距
            # 保存图片到文件
            output_dir = "output_fea_images"

            os.makedirs(output_dir, exist_ok=True)  # 创建保存图片的文件夹（如果不存在）

            # 保存为 PNG 格式
            save_path = os.path.join(output_dir, f"{layer_names[i]}_feature_maps.png")
            plt.savefig(save_path, bbox_inches='tight')  # 使用 bbox_inches='tight' 去除多余空白
            plt.savefig(save_path)
            print(f"Saved feature maps of layer {layer_names[i]} to {save_path}")

            plt.show()

        # 清理钩子
        for hook in hooks:
            hook.remove()

    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake, fake1=None):
        # Real
        pred_real = netD.forward(real.detach())
        pred_fake = netD.forward(fake.detach())
        if fake1 is not None:
            pred_fake1 = netD.forward(fake1.detach())
            loss_D_fake1 = self.criterionGAN(pred_fake1, False)

        loss_D_real = self.criterionGAN(pred_real, True)
        loss_D_fake = self.criterionGAN(pred_fake, False)

        if fake1 is not None:
            loss_D = loss_D_real + 0.5 * loss_D_fake + 0.5 * loss_D_fake1
        else:
            loss_D = loss_D_real + loss_D_fake
        return loss_D

    def backward_D_A(self):
        if self.opt.input_2:
            c = random.randint(0, 1)
            if c == 0:
                fake_B = self.fake_B
            elif c == 1:
                fake_B = self.fake_B0
        else:
            fake_B = self.fake_B
        if self.opt.realBf:
            self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B, self.real_Bf)
        else:
            self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
        self.loss_D_A.backward()

    def backward_D_fea(self):
        if self.opt.input_2:
            c = random.randint(0, 1)
            if c == 0:
                fake_patch_1 = self.fake_patch_1
            elif c == 1:
                fake_patch_1 = self.fake0_patch_1
        else:
            fake_patch_1 = self.fake_patch_1
        self.loss_D_fea = 0
        for i in range(self.opt.patchD_3):

            fake_B_fea, real_B_fea = self.vgg_loss.vgg_fea(self.vgg, fake_patch_1[i],
                                                                  self.real_patch_1[i])
            if self.opt.realBf:
                real_Bf_fea = self.vgg_loss.vgg_fea1(self.vgg, self.realf_patch_1[i])

                loss_D_fea = self.backward_D_basic(self.netD_fea, real_B_fea, fake_B_fea,real_Bf_fea.detach())
            else:
                loss_D_fea = self.backward_D_basic(self.netD_fea, real_B_fea, fake_B_fea)
            self.loss_D_fea = self.loss_D_fea + loss_D_fea / float(self.opt.patchD_3)
        self.loss_D_fea.backward()

    def backward_D_P(self):
        self.loss_D_P = 0
        if self.opt.input_2:
            c = random.randint(0, 1)
            if c == 0:
                fake_patch_1 = self.fake_patch_1
            elif c == 1:
                fake_patch_1 = self.fake0_patch_1
        else:
            fake_patch_1 = self.fake_patch_1
        for i in range(self.opt.patchD_3):
            if self.opt.realBf:

                loss_D_P = self.backward_D_basic(self.netD_P, self.real_patch_1[i], fake_patch_1[i],self.realf_patch_1[i].detach())
            else:
                loss_D_P = self.backward_D_basic(self.netD_P, self.real_patch_1[i], fake_patch_1[i])
            self.loss_D_P = self.loss_D_P + loss_D_P / float(self.opt.patchD_3)

        self.loss_D_P.backward()


    def pool(self, input):
        pool = nn.MaxPool2d(kernel_size=2)
        return pool(input)

    def Mask_pool(self, input, rate):
        pool_input = F.interpolate(self.pool(input), scale_factor=2, mode='bilinear')

        output = input * (1 - rate) + pool_input * rate

        return output
    def randMask(self, input, rate):

        h = input.size(2)
        w = input.size(3)
        size = 8
        num = int((w * h / (size * size)))
        pos = random.sample(range(num), k=int(num * rate))
        h_num = int(h // size)
        w_num = int(w // size)

        for i in pos:
            x, y = size * int(i // h_num), size * int(i % w_num)
            refer = input[:, :, x, y].clone()
            input[:, :, x: x + size, y: y + size] = refer.unsqueeze(2).unsqueeze(3)
        return input

    def color_deviation(self,input):
        a, b, c = random.uniform(0.1, 1), random.uniform(0.1, 1), random.uniform(0.1, 1)
        output = torch.concat([torch.unsqueeze(a * input[:, 0, :, :], dim=1), torch.unsqueeze(b * input[:, 1, :, :], dim=1),
                               torch.unsqueeze(c * input[:, 2, :, :], dim=1)], dim=1)
        return output
    def addnoise(self, input, min, max):
        min_std, max_std = min / 255, max / 255.
        std = (torch.rand(1) * (max_std - min_std) + min_std).cuda()
        noise = torch.rand_like(input) * std
        output = input + noise
        return output
    def randnaddnoise(self, input, min, max):
        min_std, max_std = min / 255, max / 255.
        std = (torch.rand(1) * (max_std - min_std) + min_std).cuda()
        noise = torch.randn_like(input) * std
        output = input + noise
        return output

    def forward(self, epoch, flag1=0):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.A_img = Variable(self.A_img)
        # self.input_img = self.input_img0
        if epoch >= 100:

            if self.opt.input_2 :
                # with torch.no_grad():
                self.real_img0 = Variable(self.input_img0)
                self.fake_B0 , self.x5_10 ,self.x5_20, self.map0 = self.netG_A.forward(self.real_img0, vis=True)

            self.input_img = (self.input_img + 1) / 2.

            if self.opt.addcolor:
                if random.random() < 0.5:
                    self.input_img = self.color_deviation(self.input_img)

            if self.opt.niter == 150:
                rate = max(1 - (epoch - 100) / 50 * 0.25, 0)
            else:
                rate = max(0.75 - (epoch - 100) / 50 * 0.25, 0)
            if self.opt.pool:

                # rate = random.random()
                if rate:
                    self.input_img = self.Mask_pool(self.input_img, rate)
                    # self.input_img = self.randMask(self.input_img, rate)

            if self.opt.addnoise:
                if random.random() < 0.5:
                    self.input_img = self.randnaddnoise(self.input_img, 0, 5)
                self.input_img = self.addnoise(self.input_img, rate * 50, rate * 50)

            self.input_img = torch.clamp(self.input_img, min=0, max=1)
            self.input_img = self.input_img * 2 - 1

            if self.opt.realBf:

                c = random.randint(0, 2)
                self.real_Bf = self.real_B.clone()
                self.real_Bf = (self.real_Bf + 1) / 2.
                if c ==0 or c ==2:
                    rate = (random.random() + 1)/2. - 0.25
                    self.real_Bf = self.Mask_pool(self.real_Bf, rate)
                elif c ==1 or c == 2:
                    self.real_Bf = self.randnaddnoise(self.real_Bf, 1, 5)
                self.real_Bf = torch.clamp(self.real_Bf, min=0, max=1)
                self.real_Bf = self.real_Bf * 2 - 1

            self.real_img = Variable(self.input_img)
            self.fake_B, self.x5_1, self.x5_2, self.map = self.netG_A.forward(self.real_img, vis=True)
        if epoch < 100:
            self.input_img3 = (self.real_B + 1) / 2.
            self.input_img3 = self.Mask_pool(self.input_img3, 1.0)
            self.input_img3 = self.randnaddnoise(self.input_img3, 0, 5)
            self.input_img3 = torch.clamp(self.input_img3, min=0, max=1)
            self.input_img3 = self.input_img3 * 2 - 1
            self.real_img3 = Variable(self.input_img3)
            self.fake_B3, self.x5_13, self.x5_23, self.map3 = self.netG_A.forward(self.real_img3, vis=True)
        if epoch >= 100:
            if self.opt.patchD or self.opt.patch_vgg:
                self.fake_patch_1 = []
                self.fake0_patch_1 = []
                self.realf_patch_1 = []
                self.real_patch_1 = []
                self.input_patch_1 = []
                self.realA_patch_1 = []
                self.A_img_patch_1 = []
                w = self.real_A.size(3)
                h = self.real_A.size(2)
                for i in range(self.opt.patchD_3):
                    w_offset_1 = random.randint(0, max(0, w - self.opt.patchSize - 1))
                    h_offset_1 = random.randint(0, max(0, h - self.opt.patchSize - 1))

                    self.real_patch_1.append(self.real_B[:, :, h_offset_1:h_offset_1 + self.opt.patchSize,
                                             w_offset_1:w_offset_1 + self.opt.patchSize])
                    if self.opt.realBf:
                        self.realf_patch_1.append(self.real_Bf[:, :, h_offset_1:h_offset_1 + self.opt.patchSize,
                                                 w_offset_1:w_offset_1 + self.opt.patchSize])
                    if self.opt.input_2:
                        self.fake0_patch_1.append(self.fake_B0[:, :, h_offset_1:h_offset_1 + self.opt.patchSize,
                                                 w_offset_1:w_offset_1 + self.opt.patchSize])
                    self.fake_patch_1.append(self.fake_B[:, :, h_offset_1:h_offset_1 + self.opt.patchSize,
                                             w_offset_1:w_offset_1 + self.opt.patchSize])
                    self.realA_patch_1.append(self.real_A[:, :, h_offset_1:h_offset_1 + self.opt.patchSize,
                                              w_offset_1:w_offset_1 + self.opt.patchSize])
                    self.A_img_patch_1.append(self.A_img[:, :, h_offset_1:h_offset_1 + self.opt.patchSize,
                                                     w_offset_1:w_offset_1 + self.opt.patchSize])


    def backward_G(self, epoch):
        self.loss_G_A = 0
        self.loss_mirror = 0
        self.loss_vgg_b = 0
        self.loss_dino = 0

        if epoch < 100:
            l1_loss = torch.mean(torch.abs(self.fake_B3 - self.real_B))
            self.loss_mirror = l1_loss + 1 - self.ssim((self.fake_B3 + 1) / 2.,(self.real_B + 1) / 2.)

        if epoch >= 100:
            if self.opt.colorloss:
                mirror1 = torch.mean((self.x5_10.detach() - self.x5_1)**2)
                mirror2 = torch.mean((self.x5_20.detach() - self.x5_2)**2)
                mirror_reg = 3 * self.colorloss.colorloss3(self.fake_B, self.fake_B0, self.A_img)



                self.loss_mirror = mirror_reg + mirror1 + mirror2
            self.w_DA = 1.5 - self.loss_mirror
            self.w_Dp = 1.5 - self.loss_mirror
            self.w_Dfea = 1.5 - self.loss_mirror
            self.w_vgg = 0.5
            pred_fake = self.netD_A.forward(self.fake_B)
            if self.opt.input_2:
                pred_fake0 = self.netD_A.forward(self.fake_B0)
                self.loss_G_g = self.criterionGAN(pred_fake, True) + self.criterionGAN(pred_fake0, True)
            else:
                self.loss_G_g = self.criterionGAN(pred_fake, True)

            self.loss_G_A = self.loss_G_A + self.w_DA * self.loss_G_g
            self.loss_G_fea = 0
            if self.opt.fea:
                for i in range(self.opt.patchD_3):
                    if self.opt.input_2:
                        self.fake_B_fea, self.fake_B0_fea = self.vgg_loss.vgg_fea(self.vgg, self.fake_patch_1[i],
                                                                                  self.fake0_patch_1[i])
                        pred_fake_fea = self.netD_fea.forward(self.fake_B_fea)
                        pred_fake0_fea = self.netD_fea.forward(self.fake_B0_fea)
                        loss_G_fea = (self.criterionGAN(pred_fake_fea, True)) + (self.criterionGAN(pred_fake0_fea, True))
                    else:
                        self.fake_B_fea = self.vgg_loss.vgg_fea1(self.vgg, self.fake_patch_1[i])
                        pred_fake_fea = self.netD_fea.forward(self.fake_B_fea)
                        loss_G_fea = (self.criterionGAN(pred_fake_fea, True))
                    self.loss_G_fea += loss_G_fea / float(self.opt.patchD_3)
                self.loss_G_A = self.loss_G_A + self.w_Dfea * self.loss_G_fea

            loss_G_A = 0

            if self.opt.patchD and self.opt.patchD_3 > 0:
                for i in range(self.opt.patchD_3):
                    pred_fake_patch = self.netD_P.forward(self.fake_patch_1[i])

                    if self.opt.input_2:
                        pred_fake0_patch = self.netD_P.forward(self.fake0_patch_1[i])
                        loss_G_p = (self.criterionGAN(pred_fake_patch, True)) + (self.criterionGAN(pred_fake0_patch, True))
                    else:
                        loss_G_p = (self.criterionGAN(pred_fake_patch, True))
                    loss_G_A = loss_G_p / float(self.opt.patchD_3)
                    self.loss_G_A = self.loss_G_A + self.w_Dfea * loss_G_A

                self.loss_G_A += self.w_Dp * loss_G_A / float(self.opt.patchD_3)


            if epoch < 0:
                vgg_w = 0
            else:
                vgg_w = self.w_vgg + mirror1 + mirror_reg
                dino_w = self.w_vgg + mirror2 + mirror_reg
            if self.opt.vgg > 0:
                if self.opt.input_2:
                    self.loss_vgg_b = self.vgg_loss.compute_vgg_loss3(self.vgg, self.fake_B, self.fake_B0,
                                                                      self.real_A,
                                                                      self.A_img) * self.opt.vgg if self.opt.vgg > 0 else 0
                    self.loss_dino = self.dino_loss(self.A_img, self.real_A, self.fake_B,
                                                              self.fake_B0, nograd = True) * self.opt.vgg if self.opt.vgg > 0 else 0

                else:
                    self.loss_vgg_b = self.vgg_loss.compute_vgg_loss4(self.vgg, self.fake_B,
                                                                         self.real_A,self.A_img) * self.opt.vgg if self.opt.vgg > 0 else 0

                loss_vgg_patch = 0
                # loss_dino_patch = 0
                for i in range(self.opt.patchD_3):
                    if self.opt.input_2:
                        loss_vgg_patch += self.vgg_loss.compute_vgg_loss3(self.vgg, self.fake_patch_1[i],
                                                                          self.fake0_patch_1[i],
                                                                          self.realA_patch_1[i],
                                                                          self.A_img_patch_1[i]) * self.opt.vgg
                    else:
                        loss_vgg_patch += self.vgg_loss.compute_vgg_loss5(self.vgg, self.fake_patch_1[i],
                                                                                 self.realA_patch_1[i],self.A_img_patch_1[i]) * self.opt.vgg


                self.loss_vgg_b += loss_vgg_patch / float(self.opt.patchD_3)
                self.loss_vgg_b = self.loss_vgg_b * vgg_w
                # self.loss_dino += loss_dino_patch / float(self.opt.patchD_3)
                self.loss_dino = self.loss_dino * dino_w

        if epoch < 100:
            self.loss_G = self.loss_mirror
        else:
            self.loss_G = self.loss_G_A + self.loss_vgg_b + self.loss_dino + self.loss_mirror

        self.loss_G.backward()

    def optimize_parameters(self, epoch, i):

        # forward
        if self.opt.fea == 1:
            self.set_requires_grad([self.netD_A, self.netD_fea], False)
        if self.opt.patchD:
            self.set_requires_grad([self.netD_A, self.netD_P], False)

        self.forward(epoch)
        self.optimizer_G.zero_grad()
        self.backward_G(epoch)
        self.optimizer_G.step()


        if epoch >= 100:
            if self.opt.fea == 1:
                self.set_requires_grad([self.netD_A, self.netD_fea], True)
            if self.opt.patchD:
                self.set_requires_grad([self.netD_A, self.netD_P], True)
            self.optimizer_D_A.zero_grad()
            self.backward_D_A()
            self.optimizer_D_A.step()
            if self.opt.patchD:
                self.optimizer_D_P.zero_grad()
                self.backward_D_P()
                self.optimizer_D_P.step()
            if self.opt.fea :
                self.optimizer_D_fea.zero_grad()
                self.backward_D_fea()
                self.optimizer_D_fea.step()

    def get_current_errors(self, epoch):
        D_A = self.loss_D_A.item()   if epoch>=100 else 0
        D_P = self.loss_D_P.item() if epoch>=100 and self.opt.patchD else 0
        D_fea = self.loss_D_fea.item() if epoch>=100 and self.opt.fea else 0
        G_A = self.loss_G_A.item() if epoch>=100 else 0
        loss_mirror = self.loss_mirror.item() if self.opt.colorloss else 0
        vgg = self.loss_vgg_b.item() / self.opt.vgg if epoch>=100 and self.opt.vgg > 0 else 0
        dino = self.loss_dino.item() if epoch>=100 and self.opt.vgg > 0 else 0

        return OrderedDict(
            [('D_A', D_A), ('G_A', G_A), ("vgg", vgg),("dino", dino), ("D_P", D_P), ("D_fea", D_fea), ("loss_mirror", loss_mirror)])



    def get_current_visuals(self, epoch):
        if epoch < 100:
            fake_B3 = util.tensor2im(self.fake_B3.data)
            input_img3 = util.tensor2im(self.input_img3.data)
        if epoch >= 100:
            real_A = util.tensor2im(self.real_A.data)
            fake_B = util.tensor2im(self.fake_B.data)
            input_img = util.tensor2im(self.input_img.data)
            if self.opt.input_2:
                input_img0 = util.tensor2im(self.input_img0.data)
                fake_B0 = util.tensor2im(self.fake_B0.data)
        if epoch < 100:
            return OrderedDict(
                [('fake_B3', fake_B3), ('input_img3', input_img3)])
        else:
            return OrderedDict(
                [('real_A', real_A), ('fake_B', fake_B), ('fake_B0', fake_B0), ('input_img', input_img), ('input_img0', input_img0)])

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        # self.save_network(self.dino_loss.mapping, 'Dino_map', label, self.gpu_ids)
        if self.opt.patchD:
            self.save_network(self.netD_P, 'D_P', label, self.gpu_ids)
        if self.opt.fea:
            self.save_network(self.netD_fea, 'D_fea', label, self.gpu_ids)

    def update_learning_rate(self):

        if self.opt.new_lr:
            lr = self.old_lr / 2
            lr_D = self.old_lr_D / 2
        else:
            lrd = self.opt.lr / self.opt.niter_decay
            lr = self.old_lr - lrd

            lrd_D = self.opt.lr_D / self.opt.niter_decay
            lr_D = self.old_lr_D - lrd_D
        for param_group in self.optimizer_D_A.param_groups:
            param_group['lr'] = lr_D
        if self.opt.patchD:
            for param_group in self.optimizer_D_P.param_groups:
                param_group['lr'] = lr_D
        if self.opt.fea:
            for param_group in self.optimizer_D_fea.param_groups:
                param_group['lr'] = lr_D

        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
