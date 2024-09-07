# Network Pruning via Performance Maximization
PyTorch Implementation of [Device-wise Federated Network Pruning](https://openaccess.thecvf.com/content/CVPR2024/papers/Gao_Device-Wise_Federated_Network_Pruning_CVPR_2024_paper.pdf) (CVPR 2024).
# Requirements
pytorch  
torchvision
# Usage
To train a DWNP base model
```
CUDA_VISIBLE_DEVICES=0,1 python co_train.py --gpu_visible 0,1 \
--world_size 10 --local_steps 5 --hyper_interval 5 --p 0.8 --model_name resnet56 \
--d_p 0.5 --method dynamic --start_epoch 20 --epoch 200
```
To prune the model
```
python pruning_resnet.py --method dynamic
```
To finetune the model 
```
CUDA_VISIBLE_DEVICES=0,1 python train_model.py --gpu_visible 0,1 --world_size 10 \
 --train_base False --warmup True --sch cos --lr 0.125
```
# Citation
If you found this repository is helpful, please consider to cite our paper:
```
@inproceedings{gao2024device,
  title={Device-Wise Federated Network Pruning},
  author={Gao, Shangqian and Li, Junyi and Zhang, Zeyu and Zhang, Yanfu and Cai, Weidong and Huang, Heng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12342--12352},
  year={2024}
}
