#TARGETED
python attack_SES_ver2.py --n_wb 20 --iterw 50 --apply_softmax --reduce_lr --lr 0.01 --dataset "imagenet1000" --dataset_root "../datasets/imagenet1000" --iterw 50
python attack_SES_ver2.py --n_wb 20 --iterw 50 --apply_softmax --reduce_lr --lr 0.01 --victim densenet121 --dataset "imagenet1000" --dataset_root "../datasets/imagenet1000" --iterw 50
python attack_SES_ver2.py --n_wb 20 --iterw 50 --apply_softmax --reduce_lr --lr 0.01 --victim resnext50_32x4d --dataset "imagenet1000" --dataset_root "../datasets/imagenet1000" --iterw 50



