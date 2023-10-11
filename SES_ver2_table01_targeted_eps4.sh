#TARGETED SES vs BASES

#SES
python attack_SES_ver2.py --n_wb 20 --iterw 50 --victim densenet121 --eps 4 --apply_softmax --reduce_lr --ghost resnet resnext --random_range 0.22 "imagenet1000" --dataset_root "../datasets/imagenet1000"
#BASES
python attack_SES_ver2.py --n_wb 20 --iterw 50 --victim densenet121 --eps 4 --ghost noghost "imagenet1000" --dataset_root "../datasets/imagenet1000"

#SES
python attack_SES_ver2.py --n_wb 20 --iterw 50 --victim vgg19 --eps 4 --apply_softmax --reduce_lr --ghost resnet resnext --random_range 0.22 "imagenet1000" --dataset_root "../datasets/imagenet1000"
#BASES
python attack_SES_ver2.py --n_wb 20 --iterw 50 --victim vgg19 --eps 4 --ghost noghost "imagenet1000" --dataset_root "../datasets/imagenet1000"

#SES
python attack_SES_ver2.py --n_wb 20 --iterw 50 --victim resnext50_32x4d --eps 4 --apply_softmax --reduce_lr --ghost resnet resnext --random_range 0.22 "imagenet1000" --dataset_root "../datasets/imagenet1000"
#BASES
python attack_SES_ver2.py --n_wb 20 --iterw 50 --victim resnext50_32x4d --eps 4 --ghost noghost "imagenet1000" --dataset_root "../datasets/imagenet1000"




