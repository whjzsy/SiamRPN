
# Siamese-RPN

This is a PyTorch implementation of SiameseRPN. This project is mainly based on 

This repository includes testing and tracking codes. 

## Data preparation:
```bash
First get VID dataset and youtube-bb dataset. 

python bin/create_dataset_ytbid.py 

The command above will get a dataset, The dataset in the baiduyundisk. Use this data to create lmdb.
链接:https://pan.baidu.com/s/1QnQEM_jtc3alX8RyZ3i4-g  密码:myq4

python bin/create_lmdb.py
```
## Training
```bash
python bin/train_siamrpn.py 
## Testing
```
## Testing
```bash
python bin/test_siamrpn.py 
```

## Model Download:
```bash
Pretrained model on Imagenet: https://drive.google.com/drive/folders/1HJOvl_irX3KFbtfj88_FVLtukMI1GTCR

Model with 0.626 AUC: https://pan.baidu.com/s/1vSvTqxaFwgmZdS00U3YIzQ  keyword:v91k
```
## Reference
```bash
[1] Li B , Yan J , Wu W , et al. High Performance Visual Tracking with Siamese Region Proposal Network[C]// 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2018.
```
