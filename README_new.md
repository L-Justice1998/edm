# cifar10类别
    飞机， 汽车， 鸟， 猫， 鹿， 狗， 青蛙， 马， 船以及卡车
# 推导过程
    https://www.overleaf.com/read/qrshfwcxtthf
# 运行example.py
    CUDA_VISIBLE_DEVICES=1 python example.py

# 训练代码
precond 表示的是损失函数的选择</p>
num_steps 表示的是从多少步开始进行蒸馏</p>
transfer 表示的是从哪里获得老师模型</p>

    CUDA_VISIBLE_DEVICES=2,3  torchrun --standalone --nproc_per_node=2 train.py --outdir=training-runs \
    --data=datasets/cifar10-32x32.zip --cond=0 --arch=ddpmpp \
    --precond=edm_distillation\
    --batch-gpu=128\
    --duration=20  --num_steps=32\
    --transfer=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl

    
# 数据经过处理 
在可视化target时用下列操作将数据变成图片</p>

    images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()


# 对原始模型测fid 
目前的实验是对于cifar10  uncond的
## 16步生成图像

    torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp-edm-16 --seeds=0-49999 --subdirs \
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl \
        --steps=16

## 测16步生成图像的fid

    torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp-edm-16 \
        --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz

测得结果是1.97306
    
## 32步生成图像

    torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp-edm-32 --seeds=0-49999 --subdirs \
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl \
        --steps=32

## 测32步生成图像的fid

    torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp-edm-32 \
        --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz

测得结果是1.99815
# 考虑不用二阶的方法 而是在每一步里只采用一个NFE
## 16步生成图像
    torchrun --standalone --nproc_per_node=4 generate.py --outdir=fid-tmp-edm-16-1nfe --seeds=0-49999 --subdirs \
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl \
        --steps=16 --sampler=edm_distillation_sampler --ratio=1

## 测16步生成图像的fid

    torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp-edm-16-1nfe \
        --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz

测得结果是4.09465
    
## 32步生成图像
    torchrun --standalone --nproc_per_node=4 generate.py --outdir=fid-tmp-edm-32-1nfe --seeds=0-49999 --subdirs \
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl \
        --steps=32 --sampler=edm_distillation_sampler --ratio=1

## 测32步生成图像的fid

    torchrun --standalone --nproc_per_node=4 fid.py calc --images=fid-tmp-edm-32-1nfe \
        --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz

测得结果是4.09467
## 由于效果不好，考虑是否是因为NFE太少，尝试64步的一阶采样
    torchrun --standalone --nproc_per_node=2 generate.py --outdir=fid-tmp-edm-64-1nfe --seeds=0-49999 --subdirs \
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl \
        --steps=64--sampler=edm_distillation_sampler --ratio=1

## 测64步生成图像的fid

    torchrun --standalone --nproc_per_node=2 fid.py calc --images=fid-tmp-edm-64-1nfe \
        --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz
测得的结果是4.09465
则说明一阶的采样效果不好。


# 从蒸馏好的模型中生成图片 
注意采样时的schedule保持一致 以及num_steps要确定

## 用蒸馏好的模型进行生成
    torchrun --standalone --nproc_per_node=4 generate.py --outdir=fid-tmp-edm-distillation-32-16 --seeds=0-49999 --subdirs \
    --network='./distilltion_model/network-snapshot-020000.pkl' \
    --steps=32
    
## 对生成的图片fid测试
    torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp-edm-distillation-32-16 \
        --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz

测得结果是3.35788
## 由于效果不好 对第一个以2509张图片训练的模型测fid
    torchrun --standalone --nproc_per_node=4 generate.py --outdir=fid-tmp-edm-distillation-32-16-002509 --seeds=0-49999 --subdirs \
    --network='./distilltion_model/network-snapshot-002509.pkl' \
    --steps=32

## 对生成的图片fid测试
    torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp-edm-distillation-32-16-002509  \
        --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz

测得结果是2.93074

## 尝试一下将蒸馏后的模型用二阶采样
    CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 generate.py --outdir=fid-tmp-edm-distillation-32-32-002509-test --seeds=0-49999 --subdirs \
    --network='./distilltion_model/network-snapshot-32-32-002509.pkl' \
    --steps=16  --sampler=edm_sampler 

    torchrun --standalone --nproc_per_node=4 fid.py calc --images=fid-tmp-edm-distillation-32-32-002509-test  \
        --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz
测得结果是3.92025
# 由于效果不好 将一次采样步中的两次NFE蒸馏成一步 不像先前将两次采样步中的四次NFE蒸馏成一步
    CUDA_VISIBLE_DEVICES=2,3  torchrun --standalone --nproc_per_node=2 train.py --outdir=training-runs \
    --data=datasets/cifar10-32x32.zip --cond=0 --arch=ddpmpp \
    --precond=edm_distillation1\
    --batch-gpu=128\
    --duration=20  --num_steps=32 --ratio=1\
    --transfer=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl
## 但此时还是需要32步 和上面相同 用2509张图片训练的模型
    torchrun --standalone --nproc_per_node=4 generate.py --outdir=fid-tmp-edm-distillation-32-32-002509 --seeds=0-49999 --subdirs \
    --network='./distilltion_model/network-snapshot-32-32-002509.pkl' \
    --steps=32  --sampler=edm_distillation_sampler --ratio=1 
## 测fid
    torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp-edm-distillation-32-32-002509 \
        --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz
测得结果是2.92239

## 用20000张图片训练的模型
    torchrun --standalone --nproc_per_node=4 generate.py --outdir=fid-tmp-edm-distillation-32-32-020000 --seeds=0-49999 --subdirs \
    --network='./distilltion_model/network-snapshot-32-32-020000.pkl' \
    --steps=32  --sampler=edm_distillation_sampler --ratio=1
## 测fid
    torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp-edm-distillation-32-32-020000 \
        --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz
测得结果是3.44677


# 基于edm采样器的效果不明显，则先对原来的ddim采样进行蒸馏
## 先对模型进行ddim的32步采样
    torchrun --standalone --nproc_per_node=2 generate.py --outdir=fid-tmp-ddim-32 --seeds=0-49999 --subdirs \
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl \
        --steps=32 --solver=euler --disc=iddpm --schedule=linear --scaling=none
##  fid测试
    torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp-ddim-32 \
        --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz
测得结果是4.63411
## 对模型进行ddim的64步采样
    torchrun --standalone --nproc_per_node=2 generate.py --outdir=fid-tmp-ddim-64-test --seeds=0-49999 --subdirs \
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl \
        --steps=64 --solver=euler --disc=iddpm --schedule=linear --scaling=none
##  fid测试
    torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp-ddim-64 \
        --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz
测得结果是3.30465

## 进行64步到32步的蒸馏 只用五千张图
     CUDA_VISIBLE_DEVICES=2,3  torchrun --standalone --nproc_per_node=2 train.py --outdir=training-runs-new \
    --data=datasets/cifar10-32x32.zip --cond=0 --arch=ddpmpp \
    --precond=DDIM_distillation\
    --batch-gpu=128\
    --duration=5  --num_steps=64 --ratio=2\ 
    --transfer=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl

