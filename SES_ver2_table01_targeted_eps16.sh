#TARGETED

python attack_SES_ver2.py --n_wb 20 --iterw 50 --victim densenet121 --eps 16 --apply_softmax --reduce_lr --ghost resnet resnext --random_range 0.22 --dataset "imagenet1000" --dataset_root "../datasets/imagenet1000"
python attack_SES_ver2.py --n_wb 20 --iterw 50 --victim resnext50_32x4d --eps 16 --apply_softmax --reduce_lr --ghost resnet resnext --random_range 0.22 --dataset "imagenet1000" --dataset_root "../datasets/imagenet1000"
python attack_SES_ver2.py --n_wb 20 --iterw 50 --victim vgg19 --eps 16 --apply_softmax --reduce_lr --ghost resnet resnext --random_range 0.22 --dataset "imagenet1000" --dataset_root "../datasets/imagenet1000"







