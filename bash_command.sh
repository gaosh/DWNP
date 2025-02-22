nohup python co_train.py --gpu_visible 4,5 \
--world_size 10 --local_steps 5 --p 0.8 --d_p 0.5 --method dynamic --start_epoch 25 > co_train_10c.txt 2>&1 &


CUDA_VISIBLE_DEVICES=2,3 nohup python train_model.py --gpu_visible 2,3 --world_size 10 \
 --train_base False --warmup True --sch cos --lr 0.125 > res56_ft.txt 2>&1 &

CUDA_VISIBLE_DEVICES=2,3 nohup python train_model.py --gpu_visible 2,3 --world_size 10 \
 --train_base False --lr 4e-3 --sch cos --warmup True --opt AdamW > res56_ft_adam.txt 2>&1 &
