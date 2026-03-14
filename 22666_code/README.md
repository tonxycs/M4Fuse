## 1.About M4Fuse:

$
mamba_ssm = 2.2.2
torch = 2.0.1+cu118
torchvision = 0.15.2+cu118
timm = 1.0.19
monai = 1.3.0
nibabel = 5.2.1
scipy = 1.10.1
From A100
Seeing requirement.txt (!pip install -r requirements.txt)
$

# e.g 
@ **Tiny-0.29M and Spe. Seeing M4Fuse.py**

$
model = M4Fuse(
    num_classes=4,
    input_channels=4,
    c_list = [16, 32, 48, 64, 96, 128], 
    modalities=1
).to(config['device'])
$


## 2.According to Ours ->  bash ./22666_code/Run_config.sh

## 3.Seeing log -> tail -f ./22666_code/log/train.log

## 4.Training method: 1. Train-Valid-Eval 2. 5-Fold CE.

## Notes: Due to space limitations, the Metrics (e.g. 95% confidence interval ...) have not been fully included in the Paper and  All .pth will open later (2026)
