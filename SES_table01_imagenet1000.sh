#UNTARGETED
python attack_SES.py --n_im 1000 --target vgg19 --untargeted --dataset "imagenet1000" --imagenet_folder "../datasets/imagenet1000" --iterw 20
python attack_SES.py --n_im 1000 --target densenet121 --untargeted --dataset "imagenet1000" --imagenet_folder "../datasets/imagenet1000" --iterw 20
python attack_SES.py --n_im 1000 --target resnext50_32x4d --untargeted --dataset "imagenet1000" --imagenet_folder "../datasets/imagenet1000" --iterw 20

python attack_SES.py --n_im 1000 --target vgg19 --untargeted --dataset "imagenet1000" --imagenet_folder "../datasets/imagenet1000" --iterw 50
python attack_SES.py --n_im 1000 --target densenet121 --untargeted --dataset "imagenet1000" --imagenet_folder "../datasets/imagenet1000" --iterw 50
python attack_SES.py --n_im 1000 --target resnext50_32x4d --untargeted --dataset "imagenet1000" --imagenet_folder "../datasets/imagenet1000" --iterw 50




