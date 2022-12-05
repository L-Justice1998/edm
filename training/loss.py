# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
from torch_utils import persistence
from PIL import Image
import numpy as np
import os
from skimage import io
import datetime


def target_to_image(target):
    target = target.permute(0,2,3,1)
    target = target.cpu().detach().numpy()
    print(target[0])
    for i in range(target.shape[0]):
        res = target[i] #得到batch中其中一步的图片
        image = Image.fromarray(np.uint8(res)).convert('RGB')
        #通过时间命名存储结果
        timestamp = datetime.datetime.now().strftime("%M-%S")
        savepath = timestamp + '_r.jpg'
        # image.save('./target_images/'+savepath)

			

#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
#新家的distillationloss
@persistence.persistent_class
class EDMDistillationLoss:
    def __init__(self, teacher_net, device, ratio, num_steps=256 , sigma_min=0.002, sigma_max=80, rho=7,sigma_data = 0.5):
        #外面传入老师模型和ratio（如果原始是256 最终蒸馏成32 则ratio是8）
        #nums_steps是当前步数，即上述的256
        self.teacher_net = teacher_net
        self.ratio = ratio
        self.num_steps = num_steps
        self.sigma_data = sigma_data
        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(sigma_min, self.teacher_net.sigma_min)
        sigma_max = min(sigma_max, self.teacher_net.sigma_max)
        
        # Time step discretization.
        step_indices = torch.arange(num_steps, dtype=torch.float64)
        #t_steps是方差的函数
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        self.t_steps = torch.cat([self.teacher_net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
        self.t_steps = self.t_steps.to(device)

    def __call__(self, net,images, labels=None, augment_pipe=None):
        #这里的net就是student_net
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        # 时间步 随机生成
        i = torch.randint(0, self.num_steps // self.ratio ,(images.shape[0], 1,1,1)) * self.ratio
        i = i.to(images.device)

        # 通过随机生成的时间步，索引采样到噪声的方差
        t_i = self.t_steps[i]
        t_iplus1 = self.t_steps[i+1]
        #因为t_iplus2不一定存在 则如果不存在直接用第i+1时间得到的方差
        t_iplus2 = self.t_steps[(i+2).clamp_(max=self.num_steps)]
        sigma = t_i
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        n = torch.randn_like(y) * sigma
        x_i = y + n 
        #两次edm 确定性采样的公式
        with torch.no_grad():
            d_i =(x_i-self.teacher_net(x_i, t_i, labels, augment_labels=augment_labels))/ t_i
            x_iplus1_bar = x_i + (t_iplus1 - t_i) * d_i
            d_i_bar = (x_iplus1_bar - self.teacher_net(x_iplus1_bar, t_iplus1, labels, 
                        augment_labels=augment_labels))/ t_iplus1
            x_iplus1 = x_i + (t_iplus1 - t_i) * 0.5 * (d_i + d_i_bar)

            d_iplus1 = (x_iplus1 - self.teacher_net(x_iplus1, t_iplus1,labels, 
                        augment_labels=augment_labels)) / t_iplus1
            x_iplus2_bar = x_iplus1 + (t_iplus2 - t_iplus1) * d_iplus1
            d_iplus1_bar = (x_iplus2_bar - self.teacher_net(x_iplus2_bar,t_iplus2,labels, 
                            augment_labels=augment_labels)) / t_iplus2
            x_iplus2 = x_iplus1 + (t_iplus2 - t_iplus1) * 0.5 * (d_iplus1 + d_iplus1_bar)

            target = x_i - (t_i/(t_iplus2-t_i)) * (x_iplus2 - x_i)
            target_to_image(target)
            
        # 最终的loss
        loss = weight * ((net(x_i, t_i ,labels, augment_labels=augment_labels) - target) ** 2)
        return loss
