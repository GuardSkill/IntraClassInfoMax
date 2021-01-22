CUDA_VISIBLE_DEVICES=1 nohup python3 train.py --net resnet18 --epochs 40 -l 1e-2 --foldername TestP0.1 --alpha 0 --beta 0 --gamma 0.1 > TestP0.1.log &
CUDA_VISIBLE_DEVICES=2 nohup python3 train.py --net resnet18 --epochs 40 -l 1e-2 --alpha 0 --beta 0 --gamma 0.1 --foldername P0.11Dlr0.01  &
CUDA_VISIBLE_DEVICES=2 nohup python3 train_incls.py --net resnet18 --epochs 40 -l 1e-2 --alpha 0.1 --beta 0.1 --gamma 0.1 &
CUDA_VISIBLE_DEVICES=1 nohup python3 train_incls.py --net resnet18 --epochs 40 -l 1e-2 --alpha 0.1 --beta 0.1 --gamma 0.1  --foldername Dlr0.01 &

nohup tensorboard --logdir=./logs/test/art_painting-photo-sketch_to_cartoon/ &