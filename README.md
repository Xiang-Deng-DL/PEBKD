# Personalized Education: Blind Knowledge Distillation

This repository is a PyTorch implementation of "Personalized Education: Blind Knowledge Distillation". The pretrained teachers are downloaded from [SSKD]. You can also download from [pretrained teachers].

## Requirements

The code was tested on
```
Python 3.6
torch 1.2.0
torchvision 0.4.0
```

## Training
To train small students from scratch by distilling knowledge from teacher networks with PE, first download the pretrained teachers into the "pretrained_teachers" folder, and then simply run the command below:<br>`sh train.sh`


[SSKD]: https://drive.google.com/drive/folders/1vJ0VdeFRd9a50ObbBD8SslBtmqmj8p8r
[pretrained teachers]: https://drive.google.com/drive/folders/1FI0uiVTpmW8djapeE-TPNsQ9owOD23EK

## Hyperparameters on other datasets
Tiny ImageNet: --weight_decay 0.0001 --pro 0.3 --alphas '0.1, 0.5, 1.0' --ps '64, 32' --search_T 4 --kd_T 4 --epochs 100 --lr_decay_epochs '30,60,90' --updata_epoch 30<br>
ImageNet: --weight_decay 0.00005 --pro 0.5 --alphas '0.1, 0.3' --ps '224, 112' --epochs 100 --search_T 2 --kd_T 2 --lr_decay_epochs '30,60,90' --updata_epoch 20 

## Notes
PE-BKD is a general framework. You can design **the prior region** based on the prior knowledge that you have about the target task. 


## Citation
If you find this code helpful, you may consider citing this paper:
```bibtex
@inproceedings{deng2022personalized,
  title={Personalized Education: Blind Knowledge Distillation},
  author={Deng, Xiang and Zheng, Jian and Zhang, Zhongfei},
  booktitle = {Proceedings of the 2022 European Conference on Computer Vision},
  year={2022}
}
```
