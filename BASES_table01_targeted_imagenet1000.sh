#TARGETED
python attack_SES.py --n_im 1000 --target vgg19 --gpu 1 --dataset "imagenet1000" --imagenet_folder "../datasets/imagenet1000"
python attack_SES.py --n_im 1000 --target densenet121 --gpu 1 --dataset "imagenet1000" --imagenet_folder "../datasets/imagenet1000"
python attack_SES.py --n_im 1000 --target resnext50_32x4d --gpu 1 --dataset "imagenet1000" --imagenet_folder "../datasets/imagenet1000"