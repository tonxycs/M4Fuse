 1.About M4Fuse:

# Dataset

Get BraTS 2019 and BraTS 2021 dataset [here](https://www.med.upenn.edu/cbica/brats2019/data.html)

![image](https://github.com/tonxycs/M4Fuse/blob/main/Vis/dataset.png)


# Models
 *e.g.Tiny-0.29M and Spe. Seeing ~Network/M4Fuse.py*

model = M4Fuse(
    num_classes=4,
    input_channels=4,
    c_list = [16, 32, 48, 64, 96, 128], 
    modalities=1
).to(config['device'])

# Use
 2.According to Ours ->  bash ./Run_config.sh 

 --running 2019 dataset, you can python T19.py and it is ok
 
 --running 2021 dataset, you need to python T21.py combined with Eval.py (by the best weight /.pth)

 3.Seeing log -> tail -f ./log/train.log

 4.Training method: 1. Train-Valid-Eval 2. 5-Fold CE.

# PIP

mamba_ssm = 2.2.2

torch = 2.0.1+cu118

torchvision = 0.15.2+cu118

timm = 1.0.19

monai = 1.3.0

nibabel = 5.2.1

scipy = 1.10.1

***Notes: For urgent citation, usage, or further assistance, please feel free to contact me [tonxycs@gmail.com] without any hesitation. You are very welcome!***
