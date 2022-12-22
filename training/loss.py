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


def target_to_image(target,k):
    assert k.shape[0] == target.shape[0]
    k = k.reshape(-1)
    target = (target * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    for i in range(target.shape[0]):
        res = target[i] #得到batch中其中一步的图片
        image = Image.fromarray(np.uint8(res)).convert('RGB')
        #通过时间命名和时间步存储结果
        timestamp = datetime.datetime.now().strftime("%M-%S")
        savepath = timestamp +  'index' + str(k[i]) + '.jpg'
        image.save('./target_images/'+savepath)

			

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
#新加的distillationloss
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
        # print(self.t_steps[torch.arange(self.num_steps // self.ratio )*self.ratio])
        # exit()

    def __call__(self, net,images, labels=None, augment_pipe=None):
        #这里的net就是student_net
        # 注意，在生成代码中 采取二阶步的时候的判断条件是 i < num_steps -1
        # 即i最多等于num_steps - 2 若是四步的则是最多等于num_steps - 4
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        # 时间步 随机生成 randint左闭右开，会采到[0,num_steps),不会采到num_steps
        # i = torch.randint(0, self.num_steps - 1 ,(images.shape[0], 1,1,1)) 
        # 这里i的取值是0到num_steps-1
        index = torch.randint(0, (self.num_steps // self.ratio) ,(images.shape[0], 1,1,1)) * self.ratio
        # index = torch.randint(0, (self.num_steps - 2) ,(images.shape[0], 1,1,1))
        i = index.to(images.device)

        # 通过随机生成的时间步，索引采样到噪声的方差
        t_i = self.t_steps[i]
        t_iplus1 = self.t_steps[(i+1).clamp_(max = self.num_steps - 1)]
        #因为t_iplus2不一定存在 则如果不存在直接用第i+1时间得到的方差
        t_iplus2 = self.t_steps[(i+2).clamp_(max = self.num_steps - 1)]
        sigma = t_i
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        n = torch.randn_like(y) * sigma
        x_i = y + n 
        # 两次edm 确定性采样的公式
        # 若t_iplus1 和t_iplus2被截断，因为截断后有相减的项是为0的，则从后往前四次NFE最多退化成只有一次NFE的情况。
        with torch.no_grad():
            # 原edm采样公式去除噪声部分的一阶部分
            d_i = (x_i - self.teacher_net(x_i, t_i, labels, augment_labels=augment_labels)) / t_i
            x_iplus1_bar = x_i + (t_iplus1 - t_i) * d_i
            d_i_bar = (x_iplus1_bar - self.teacher_net(x_iplus1_bar, t_iplus1, labels, 
                        augment_labels=augment_labels)) / t_iplus1
            x_iplus1 = x_i + (t_iplus1 - t_i) * 0.5 * (d_i + d_i_bar)

            # 原edm采样公式去除噪声部分的一阶部分 但是是第二次
            d_iplus1 = (x_iplus1 - self.teacher_net(x_iplus1, t_iplus1,labels, 
                        augment_labels=augment_labels)) / t_iplus1
            x_iplus2_bar = x_iplus1 + (t_iplus2 - t_iplus1) * d_iplus1
            d_iplus1_bar = (x_iplus2_bar - self.teacher_net(x_iplus2_bar,t_iplus2,labels, 
                            augment_labels=augment_labels)) / t_iplus2
            x_iplus2 = x_iplus1 + (t_iplus2 - t_iplus1) * 0.5 * (d_iplus1 + d_iplus1_bar)
            # target计算
            target = x_i - (t_i/(t_iplus2-t_i)) * (x_iplus2 - x_i)
            # target_to_image(target,index)
            
        # 最终的loss
        loss = weight * ((net(x_i, t_i ,labels, augment_labels=augment_labels) - target) ** 2)
        return loss
    
@persistence.persistent_class
class EDMDistillationLoss1:
    #这个损失函数是将一阶步和二阶步蒸馏成一步 而不是两个一阶步和二阶步蒸馏成一步
    def __init__(self, teacher_net, device, ratio = 1, num_steps=256 , sigma_min=0.002, sigma_max=80, rho=7,
        sigma_data = 0.5):
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
        # print(self.t_steps[torch.arange(self.num_steps // self.ratio ) * self.ratio])
        # exit

    def __call__(self, net,images, labels=None, augment_pipe=None):
        #这里的net就是student_net
        # 注意，在生成代码中 采取二阶步的时候的判断条件是 i < num_steps -1
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        # 时间步 随机生成 randint左闭右开，会采到[0,num_steps),不会采到num_steps
        # i = torch.randint(0, self.num_steps - 1 ,(images.shape[0], 1,1,1)) 
        # 这里i的取值是0到num_steps-1
        index = torch.randint(0, (self.num_steps  // self.ratio)  ,(images.shape[0], 1,1,1)) * self.ratio 
        # index = torch.randint(0, (self.num_steps - 1)  ,(images.shape[0], 1,1,1))
        i = index.to(images.device)

        # 通过随机生成的时间步，索引采样到噪声的方差
        # t_i.size为[128,1,1,1]
        t_i = self.t_steps[i]
        t_iplus1 = self.t_steps[(i+1).clamp_(max = self.num_steps - 1)]
        sigma = t_i
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        n = torch.randn_like(y) * sigma
        x_i = y + n 
        # 一次edm 确定性采样的公式
        # 若t_iplus1 被截断，因为截断后有相减的项是为0的，则两次NFE最多退化成只有一次NFE的情况。
        with torch.no_grad():
            # 原edm采样公式去除噪声部分的一阶和二阶步
            d_i = (x_i - self.teacher_net(x_i, t_i, labels, augment_labels=augment_labels)) / t_i
            x_iplus1_bar = x_i + (t_iplus1 - t_i) * d_i
            d_i_bar = (x_iplus1_bar - self.teacher_net(x_iplus1_bar, t_iplus1, labels, 
                        augment_labels=augment_labels)) / t_iplus1
            x_iplus1 = x_i + (t_iplus1 - t_i) * 0.5 * (d_i + d_i_bar)
            # target = x_i - (x_iplus1 - x_i) * (t_i / (t_iplus1 - t_i))
            target = x_i - (x_iplus1 - x_i) * (t_i / (t_iplus1 - t_i).clamp_(max = -1e-20))
            # target_to_image(target,index)

            
        # 最终的loss
        loss = weight * ((net(x_i, t_i ,labels, augment_labels=augment_labels) - target) ** 2)
        return loss
    
@persistence.persistent_class
class DDIMDistillationLoss:
    #用DDIM的生成样本做蒸馏
    def __init__(self, teacher_net, device, ratio = 2, num_steps=256,
                sigma_min=None, sigma_max=None, rho=7, 
                solver='Euler', discretization='iddpm', schedule='linear', scaling='none',
                epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
                S_churn=0, S_min=0, S_max=float('inf'), S_noise=1):
        #外面传入老师模型和ratio（如果原始是256 最终蒸馏成32 则ratio是8）
        #nums_steps是当前步数，即上述的256
        # 这个α是步长增长参数 默认是一就是中值
        self.teacher_net = teacher_net
        self.ratio = ratio
        self.num_steps = num_steps
        self.S_noise = S_noise
        self.alpha = alpha
        self.device = device
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.solver = solver
        self.discretization = discretization
        self.schedule = schedule
        self.scaling = scaling
        self.epsilon_s = epsilon_s
        self.C_1 = C_1
        self.C_2 = C_2 
        self.M = M    


    def __call__(self, net,images, labels=None, augment_pipe=None):
        # 这里传进里的是student net
        # Select default noise level range based on the specified time step discretization.
        sigma_min = self.sigma_min
        sigma_max = self.sigma_max
        if sigma_min is None:
            sigma_min = {'iddpm': 0.002}[self.discretization]

        if sigma_max is None:
            sigma_max = {'iddpm': 81}[self.discretization]


        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(sigma_min, self.teacher_net.sigma_min)
        sigma_max = min(sigma_max, self.teacher_net.sigma_max)

        # Define time steps in terms of noise level.
        step_indices = torch.arange(self.num_steps, dtype=torch.float64)
 
        if self.discretization == 'iddpm':
            u = torch.zeros(self.M + 1, dtype=torch.float64)
            alpha_bar = lambda j: (0.5 * np.pi * j / self.M / (self.C_2 + 1)).sin() ** 2
            for j in torch.arange(self.M, 0, -1): # M, ..., 1
                u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=self.C_1) - 1).sqrt()
            u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
            sigma_steps = u_filtered[((len(u_filtered) - 1) / (self.num_steps - 1) * step_indices).round().to(torch.int64)]
      
        # Define noise level schedule.
        assert self.schedule == 'linear'
        sigma = lambda t: t
        sigma_deriv = lambda t: 1
        sigma_inv = lambda sigma: sigma
        
        # Define scaling schedule.
        assert self.scaling == 'none'
        s = lambda t: 1
        s_deriv = lambda t: 0

        # Time step discretization.
        step_indices = torch.arange(self.num_steps, dtype=torch.float64) 
        t_steps = sigma_inv(self.teacher_net.round_sigma(sigma_steps))
        self.t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0
        self.t_steps = self.t_steps.to(self.device)
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        # 时间步 随机生成 randint左闭右开，会采到[0,num_steps),不会采到num_steps
        # i = torch.randint(0, self.num_steps - 1 ,(images.shape[0], 1,1,1)) 
        # 这里i的取值是0到num_steps-1
        index = torch.randint(0, (self.num_steps // self.ratio) ,(images.shape[0], 1,1,1)) * self.ratio
        i = index.to(images.device)

        # 通过随机生成的时间步，索引采样到噪声的方差
        t_i = self.t_steps[i]
        t_iplus1 = self.t_steps[(i+1)]
        t_iplus2 = self.t_steps[(i+2)]
        # σ(t)已经在前面定义 不需要再特殊定义
        # sigma = t_i
        weight = 1 / (sigma(t_i)) ** 2
        n = torch.randn_like(y) * sigma(t_i)
        x_i = y + n 
        with torch.no_grad():
            # two Euler steps
            h_i = t_iplus1 - t_i
            p_i = sigma_deriv(t_i) / sigma(t_i) + s_deriv(t_i) /s(t_i)
            q_i = sigma_deriv(t_i) * s(t_i) / sigma(t_i)
            denoised = self.teacher_net(x_i/s(t_i), sigma(t_i), labels, augment_labels=augment_labels).to(torch.float64)
            d_cur = p_i * x_i - q_i * denoised
            x_iplus1 = x_i + self.alpha * h_i * d_cur

            h_iplus1 = t_iplus2 - t_iplus1
            p_iplus1 = sigma_deriv(t_iplus1) / sigma(t_iplus1) +s_deriv(t_iplus1) / s(t_iplus1)
            q_iplus1 = sigma_deriv(t_iplus1) * s(t_iplus1) / sigma(t_iplus1)
            denoised_plus1 = self.teacher_net(x_iplus1/s(t_iplus1), sigma(t_iplus1), labels, augment_labels=augment_labels).to(torch.float64)
            d_cur_plus1= p_iplus1 * x_iplus1 - q_iplus1 * denoised_plus1
            x_iplus2 = x_iplus1 + self.alpha * h_iplus1 * d_cur_plus1
            # x_iplus1_final = x_i + self.alpha * h_i * 1/2 * (d_cur + d_cur_plus1)

            target = p_i / q_i * x_i - (x_iplus2 - x_i) / q_i / (h_iplus1 + h_i)
            # target = p_i / q_i * x_i - (x_iplus1_final - x_i) / q_i / (t_iplus1 - t_i)
            # target_to_image(target, index)
            # print(target.shape)
            
        # 最终的loss
        loss = weight * ((net(x_i, t_i ,labels, augment_labels=augment_labels) - target) ** 2)
        # print(target.equal((net(x_i, t_i ,labels, augment_labels=augment_labels))))
        # print(weight[0])
        # print(net(x_i, t_i ,labels, augment_labels=augment_labels)[1].equal(target[1]))
        # print(net(x_i, t_i ,labels, augment_labels=augment_labels).shape)
        # print(type((net(x_i, t_i ,labels, augment_labels=augment_labels))))
        return loss
@persistence.persistent_class
class GerneralDistillationLoss:
    #用DDIM的生成样本做蒸馏
    def __init__(self, teacher_net, device, ratio = 2, num_steps=256,
                sigma_min=None, sigma_max=None, rho=7, sigma_data = 0.5,
                solver='Euler', discretization='iddpm', schedule='linear', scaling='none',
                epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
                S_churn=0, S_min=0, S_max=float('inf'), S_noise=1):
        #外面传入老师模型和ratio（如果原始是256 最终蒸馏成32 则ratio是8）
        #nums_steps是当前步数，即上述的256
        # 这个α是步长增长参数 默认是一就是中值
        self.teacher_net = teacher_net
        self.ratio = ratio
        self.num_steps = num_steps
        self.sigma_data = sigma_data
        self.S_noise = S_noise
        self.alpha = alpha
        self.device = device
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.sigma_data = sigma_data
        self.solver = solver
        self.discretization = discretization
        self.schedule = schedule
        self.scaling = scaling
        self.epsilon_s = epsilon_s
        self.C_1 = C_1
        self.C_2 = C_2     
        self.M = M


    def __call__(self, net,images, labels=None, augment_pipe=None):
        # Helper functions for VP & VE noise level schedules.
        vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
        vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
        vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
        ve_sigma = lambda t: t.sqrt()
        ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
        ve_sigma_inv = lambda sigma: sigma ** 2
        # Select default noise level range based on the specified time step discretization.
        sigma_min = self.sigma_min
        sigma_max = self.sigma_max
        if sigma_min is None:
            vp_def = vp_sigma(beta_d=19.1, beta_min=0.1)(t=self.epsilon_s)
            sigma_min = {'vp': vp_def, 've': 0.02, 'iddpm': 0.002, 'edm': 0.002}[self.discretization]

        if sigma_max is None:
            vp_def = vp_sigma(beta_d=19.1, beta_min=0.1)(t=1)
            sigma_max = {'vp': vp_def, 've': 100, 'iddpm': 81, 'edm': 80}[self.discretization]


        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(sigma_min, self.teacher_net.sigma_min)
        sigma_max = min(sigma_max, self.teacher_net.sigma_max)

        # Compute corresponding betas for VP.
        vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / self.epsilon_s - np.log(sigma_max ** 2 + 1)) / (self.epsilon_s - 1)
        vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

        # Define time steps in terms of noise level.
        step_indices = torch.arange(self.num_steps, dtype=torch.float64)
        if self.discretization == 'vp':
            orig_t_steps = 1 + step_indices / (self.num_steps - 1) * (self.epsilon_s - 1)
            sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
        elif self.discretization == 've':
            orig_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (self.num_steps - 1)))
            sigma_steps = ve_sigma(orig_t_steps)
        elif self.discretization == 'iddpm':
            u = torch.zeros(self.M + 1, dtype=torch.float64)
            alpha_bar = lambda j: (0.5 * np.pi * j / self.M / (self.C_2 + 1)).sin() ** 2
            for j in torch.arange(self.M, 0, -1): # M, ..., 1
                u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=self.C_1) - 1).sqrt()
            u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
            sigma_steps = u_filtered[((len(u_filtered) - 1) / (self.num_steps - 1) * step_indices).round().to(torch.int64)]
        else:
            assert self.discretization == 'edm'
            sigma_steps = (sigma_max ** (1 / self.rho) + step_indices / (self.num_steps - 1) * (sigma_min ** (1 / self.rho) - sigma_max ** (1 / self.rho))) ** self.rho

        # Define noise level schedule.
        if self.schedule == 'vp':
            sigma = vp_sigma(vp_beta_d, vp_beta_min)
            sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
            sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
        elif self.schedule == 've':
            sigma = ve_sigma
            sigma_deriv = ve_sigma_deriv
            sigma_inv = ve_sigma_inv
        else:
            assert self.schedule == 'linear'
            sigma = lambda t: t
            sigma_deriv = lambda t: 1
            sigma_inv = lambda sigma: sigma
        
        # Define scaling schedule.
        if self.scaling == 'vp':
            s = lambda t: 1 / (1 + self.sigma(t) ** 2).sqrt()
            s_deriv = lambda t: -self.sigma(t) * self.sigma_deriv(t) * (self.s(t) ** 3)
        else:
            assert self.scaling == 'none'
            s = lambda t: 1
            s_deriv = lambda t: 0


        # Time step discretization.
        step_indices = torch.arange(self.num_steps, dtype=torch.float64) 
        t_steps = sigma_inv(self.teacher_net.round_sigma(sigma_steps))
        self.t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0
        self.t_steps = self.t_steps.to(self.device)
        #这里的net就是student_net
        # 注意，在生成代码中 采取二阶步的时候的判断条件是 i < num_steps -1
        # 即i最多等于num_steps - 2 若是四步的则是最多等于num_steps - 4
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        # 时间步 随机生成 randint左闭右开，会采到[0,num_steps),不会采到num_steps
        # i = torch.randint(0, self.num_steps - 1 ,(images.shape[0], 1,1,1)) 
        # 这里i的取值是0到num_steps-1
        index = torch.randint(0, (self.num_steps // self.ratio) ,(images.shape[0], 1,1,1)) * self.ratio
        i = index.to(images.device)

        # 通过随机生成的时间步，索引采样到噪声的方差
        t_i = self.t_steps[i]
        t_iplus1 = self.t_steps[(i+1).clamp_(max = self.num_steps - 1)]
        t_iplus2 = self.t_steps[(i+2).clamp_(max = self.num_steps - 1)]
        # σ(t)已经在前面定义 不需要再特殊定义
        # sigma = t_i
        weight = 1 / (sigma(t_i)) ** 2
        n = torch.randn_like(y) * sigma(t_i)
        x_i = y + n 
        with torch.no_grad():
            # two Euler steps
            h_i = t_iplus1 - t_i
            p_i = sigma_deriv(t_i) / sigma(t_i) + s_deriv(t_i) /s(t_i)
            q_i = sigma_deriv(t_i) * s(t_i) / sigma(t_i)
            denoised = self.teacher_net(x_i/s(t_i), sigma(t_i), labels, augment_labels=augment_labels).to(torch.float64)
            d_cur = p_i * x_i - q_i * denoised
            x_iplus1 = x_i + self.alpha * h_i * d_cur

            h_iplus1 = t_iplus2 - t_iplus1
            p_iplus1 = sigma_deriv(t_iplus1) / sigma(t_iplus1) +s_deriv(t_iplus1) / s(t_iplus1)
            q_iplus1 = sigma_deriv(t_iplus1) * s(t_iplus1) / sigma(t_iplus1)
            denoised_plus1 = self.teacher_net(x_iplus1/s(t_iplus1), sigma(t_iplus1), labels, augment_labels=augment_labels).to(torch.float64)
            d_cur_plus1= p_iplus1 * x_iplus1 - q_iplus1 * denoised_plus1
            x_iplus2 = x_iplus1 + self.alpha * h_iplus1 * d_cur_plus1

            target = p_i / q_i * x_i - (x_iplus2 - x_i) / q_i / (t_iplus2 - t_i)
            # target_to_image(target, index)
            # print(target.shape)
            
        # 最终的loss
        loss = weight * ((net(x_i, t_i ,labels, augment_labels=augment_labels) - target) ** 2)
        print((net(x_i, t_i ,labels, augment_labels=augment_labels) - target)[0])
        # print(target.equal((net(x_i, t_i ,labels, augment_labels=augment_labels))))
        # print(weight[0])
        # print(net(x_i, t_i ,labels, augment_labels=augment_labels)[1].equal(target[1]))
        exit()
        # print(net(x_i, t_i ,labels, augment_labels=augment_labels).shape)
        # print(type((net(x_i, t_i ,labels, augment_labels=augment_labels))))
        return loss
    
