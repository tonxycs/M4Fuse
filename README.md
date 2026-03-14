 1.About M4Fuse:

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
# Dataset
* BarTS2019:(HGG/LGG)

BarTS2019/
├── processed/
│   ├── MICCAI_BraTS_2019_Data_Training/
│   │   ├── LGG
│   │   │   ├── BraTS19_TMC_30014_1
│   │   │   │	├── BraTS19_TMC_30014_1_t1.nii.gz
│   │   │   │	├── BraTS19_TMC_30014_1_t1cd.nii.gz
│   │   │   │	├── BraTS19_TMC_30014_1_t2.nii.gz
│   │   │   │	├── BraTS19_TMC_30014_1_flair.nii.gz
│   │   │   │	├── BraTS19_TMC_30014_1_seg.nii.gz
│   │   │   │	├── BraTS19_TMC_30014_1_pkl_ui8f32b0.pkl
│   │   │   ├── ...
│   │   ├── HGG
│   │   │   ├── ...
│   │   ├── cross_validation
│   │	│	├── t1.txt
│   │	│	├── t2.txt
│   │	│	├── ...
│   │	│	├── v1.txt
│   │	│	├── v2.txt
│   │	│	├── ...
│   ├── ...


* BarTS2021: 

BarTS2021/
├── processed/
│   ├── MICCAI_BraTS_2021_Data_Training/
│   │   ├── BraTS21_00000
│   │   │	├── BraTS2021_00000_flair.nii.gz
│   │   │	├── BraTS2021_00000_t1.nii.gz
│   │   │	├── BraTS2021_00000_t2.nii.gz
│   │   │	├── BraTS2021_00000_seg.nii.gz
│   │   │	├── BraTS2021_00000_t1ce.nii.gz
│   │   ├── ...
│   │   ├── BraTS21_16666
│   │   │	├── BraTS2021_01666_flair.nii.gz
│   │   │	├── BraTS2021_01666_t1.nii.gz
│   │   │	├── BraTS2021_01666_t2.nii.gz
│   │   │	├── BraTS2021_01666_seg.nii.gz
│   │   │	├── BraTS2021_01666_t1ce.nii.gz


# Models
@ **Tiny-0.29M and Spe. Seeing M4Fuse.py**


$
model = M4Fuse(
    num_classes=4,
    input_channels=4,
    c_list = [16, 32, 48, 64, 96, 128], 
    modalities=1
).to(config['device'])
$

# Use
 2.According to Ours ->  bash ./Run_config.sh 

 --running 2019 dataset, you can python T19.py and it is ok
 
 --running 2021 dataset, you need to python T21.py combined with Eval.py (by the best weight /.pth)

 3.Seeing log -> tail -f ./log/train.log

 4.Training method: 1. Train-Valid-Eval 2. 5-Fold CE.

 Notes: For urgent citation, usage, or further assistance, please feel free to contact me [tonxycs@gmail.com] without any hesitation. You are very welcome!
