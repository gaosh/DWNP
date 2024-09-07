nohup python co_train.py --gpu_visible 1,2 \
--world_size 10 --local_steps 5 --hyper_interval 5 --p 0.8 --model_name mobnetv2 \
--d_p 0.5 --method dynamic --start_epoch 20 --epoch 200 > co_train_dmbv2.txt 2>&1 &

nohup python co_train.py --gpu_visible 3,4 --world_size 10 \
 --model_name mobnetv2 --local_steps 5 --hyper_interval 5 --method static \
 --start_epoch 20 --epoch 200 --split_test False > co_train_smbv2.txt 2>&1 &

python pruning_mobnet.py --method dynamic
python pruning_mobnet.py --method static


nohup python train_model.py --gpu_visible 3,4 --world_size 10 \
 --train_base False --model_name mobnetv2 --method static \
  --warmup True --sch cos --lr 0.1 --epoch 200 --split_test False > mobnetv2_static_ft.txt 2>&1 &

nohup python train_model.py --gpu_visible 1,2 --world_size 10 \
 --train_base False --model_name mobnetv2 --method dynamic_train \
  --warmup True --sch cos --lr 0.1 --epoch 200 > mobnetv2_ft.txt 2>&1 &
##################################################################
nohup python train_model.py --gpu_visible 3,4 --world_size 10 \
 --train_base False --model_name mobnetv2 --method static --opt AdamW \
  --warmup True --sch cos --lr 0.001 --epoch 200 --split_test False > mobnetv2_static_ft.txt 2>&1 &

nohup python train_model.py --gpu_visible 1,2 --world_size 10 \
 --train_base False --model_name mobnetv2 --method dynamic_train --opt AdamW \
  --warmup True --sch cos --lr 0.001 --epoch 200 > mobnetv2_ft.txt 2>&1 &