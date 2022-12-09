#推导过程
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
py
测得结果是1.99815

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

# 将16步的变成8步 

    CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 train.py --outdir=training-runs \
        --data=datasets/cifar10-32x32.zip --cond=0 --arch=ddpmpp \
        --precond=edm_distillation\
        --batch-gpu=128\
        --duration=20  --num_steps=16\
        --transfer='./distillation_model/network-snapshot-020000.pkl'
