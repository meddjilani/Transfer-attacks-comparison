#Remaining work for table01: add target models

export CUDA_VISIBLE_DEVICES=0

#MI
python MI-FGSM.py --target vgg19 --model resnet18 --n_examples 10000 --batch_size 32 --eps 0.031 --alpha 0.0078 --steps 10 --decay 1.0

#Ghost
python GN-PGD.py --target vgg19 --model resnet18 --n_examples 10000 --batch_size 32 --eps 0.031 --alpha 0.0078 --steps 10

#DI
python DI-FGSM-PGD.py --target vgg19 --model resnet18 --n_examples 10000 --batch_size 32 --eps 0.031 --alpha 0.0078 --steps 10 --resize_rate 0.9 --diversity_prob 0.5

#MI+Ghost
python GN-MI-FGSM.py --target vgg19 --model resnet18 --n_examples 10000 --batch_size 32 --eps 0.031 --alpha 0.0078 --steps 10 --decay 1.0

#DI+Ghost
python GN-DI-FGSM-PGD.py --target vgg19 --model resnet18 --n_examples 10000 --batch_size 32 --eps 0.031 --alpha 0.0078 --steps 10 --resize_rate 0.9 --diversity_prob 0.5

#MI+DI
python DI-FGSM.py --target vgg19 --model resnet18 --n_examples 10000 --batch_size 32 --eps 0.031 --alpha 0.0078 --steps 10 --decay 1.0 --resize_rate 0.9 --diversity_prob 0.5

#MI+DI+Ghost
python GN-DI-FGSM.py --target vgg19 --model resnet18 --n_examples 10000 --batch_size 32 --eps 0.031 --alpha 0.0078 --steps 10 --decay 1.0 --resize_rate 0.9 --diversity_prob 0.5

#Admix
python adm.py --target vgg19 --model resnet18 --n_examples 4 --batch_size 1 --admix-m1 0.5 --admix-m2 0.5

#Ensemble MI
python MI-FGSM-ENS.py --target vgg19 --model resnet18 resnet34 --n_examples 10000 --batch_size 32 --eps 0.031 --alpha 0.0078 --steps 10 --decay 1.0

#Ensemble MI+Ghost
python GN-MI-FGSM-ENS.py --target vgg19 --model resnet18 resnet34 --n_examples 10000 --batch_size 32 --eps 0.031 --alpha 0.0078 --steps 10 --decay 1.0

#MI+SGM
python SGM-MI-FGSM.py --target vgg19 --model resnet18 --n_examples 10000 --batch_size 32 --eps 0.031 --alpha 0.0078 --steps 10 --decay 1.0 --gamma 0.5

#DI+SGM
python SGM-DI-FGSM-PGD.py --target vgg19 --model resnet18 --n_examples 10000 --batch_size 32 --eps 0.031 --alpha 0.0078 --steps 10 --resize_rate 0.9 --diversity_prob 0.5

#MI+DI+SGM
python SGM-DI-FGSM.py --target vgg19 --model resnet18 --n_examples 10000 --batch_size 32 --eps 0.031 --alpha 0.0078 --steps 10 --decay 1.0 --resize_rate 0.9 --diversity_prob 0.5

