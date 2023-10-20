#Remaining work for table01: add target models

export CUDA_VISIBLE_DEVICES=0
N_IMG=10000

for seed in 42 1 10 100 1000; do
  #MI
  python MI-FGSM.py --target vgg19 --model resnet18 --n_im $N_IMG --batch_size 32 --eps 0.031 --alpha 0.0078 --seed $seed --steps 10 --decay 1.0
  python MI-FGSM.py --target vgg19 --model resnet50 --n_im $N_IMG --batch_size 32 --eps 0.031 --alpha 0.0078 --seed $seed --steps 10 --decay 1.0

  #Ghost
  python GN-PGD.py --target vgg19 --model resnet18 --n_im $N_IMG --batch_size 32 --eps 0.031 --alpha 0.0078 --seed $seed --steps 10
  python GN-PGD.py --target vgg19 --model resnet50 --n_im $N_IMG --batch_size 32 --eps 0.031 --alpha 0.0078 --seed $seed --steps 10

  #DI
  python DI-FGSM-PGD.py --target vgg19 --model resnet18 --n_im $N_IMG --batch_size 32 --eps 0.031 --alpha 0.0078 --seed $seed --steps 10 --resize_rate 0.9 --diversity_prob 0.5
  python DI-FGSM-PGD.py --target vgg19 --model resnet50 --n_im $N_IMG --batch_size 32 --eps 0.031 --alpha 0.0078 --seed $seed --steps 10 --resize_rate 0.9 --diversity_prob 0.5


  #MI+Ghost
  python GN-MI-FGSM.py --target vgg19 --model resnet18 --n_im $N_IMG --batch_size 32 --eps 0.031 --alpha 0.0078 --seed $seed --steps 10 --decay 1.0
  python GN-MI-FGSM.py --target vgg19 --model resnet50 --n_im $N_IMG --batch_size 32 --eps 0.031 --alpha 0.0078 --seed $seed --steps 10 --decay 1.0

  #DI+Ghost
  python GN-DI-FGSM-PGD.py --target vgg19 --model resnet18 --n_im $N_IMG --batch_size 32 --eps 0.031 --alpha 0.0078 --seed $seed --steps 10 --resize_rate 0.9 --diversity_prob 0.5
  python GN-DI-FGSM-PGD.py --target vgg19 --model resnet50 --n_im $N_IMG --batch_size 32 --eps 0.031 --alpha 0.0078 --seed $seed --steps 10 --resize_rate 0.9 --diversity_prob 0.5

  #MI+DI
  python DI-FGSM.py --target vgg19 --model resnet18 --n_im $N_IMG --batch_size 32 --eps 0.031 --alpha 0.0078 --seed $seed --steps 10 --decay 1.0 --resize_rate 0.9 --diversity_prob 0.5
  python DI-FGSM.py --target vgg19 --model resnet50 --n_im $N_IMG --batch_size 32 --eps 0.031 --alpha 0.0078 --seed $seed --steps 10 --decay 1.0 --resize_rate 0.9 --diversity_prob 0.5

  #MI+DI+Ghost
  python GN-DI-FGSM.py --target vgg19 --model resnet18 --n_im $N_IMG --batch_size 32 --eps 0.031 --alpha 0.0078 --seed $seed --steps 10 --decay 1.0 --resize_rate 0.9 --diversity_prob 0.5
  python GN-DI-FGSM.py --target vgg19 --model resnet50 --n_im $N_IMG --batch_size 32 --eps 0.031 --alpha 0.0078 --seed $seed --steps 10 --decay 1.0 --resize_rate 0.9 --diversity_prob 0.5

  #Admix
  python adm.py --target vgg19 --model resnet18 --n_im $N_IMG --batch_size 32 --admix-m1 0.5 --admix-m2 0.5
  python adm.py --target vgg19 --model resnet50 --n_im $N_IMG --batch_size 32 --admix-m1 0.5 --admix-m2 0.5

  #Ensemble MI
  python MI-FGSM-ENS.py --target vgg19 --model resnet18 resnet50 --n_im $N_IMG --batch_size 32 --eps 0.031 --alpha 0.0078 --seed $seed --steps 10 --decay 1.0

  #Ensemble MI+Ghost
  python GN-MI-FGSM-ENS.py --target vgg19 --model resnet18 resnet50 --n_im $N_IMG --batch_size 32 --eps 0.031 --alpha 0.0078 --seed $seed --steps 10 --decay 1.0

  #MI+SGM
  python SGM-MI-FGSM.py --target vgg19 --model resnet18 --n_im $N_IMG --batch_size 32 --eps 0.031 --alpha 0.0078 --seed $seed --steps 10 --decay 1.0 --gamma 0.5
  python SGM-MI-FGSM.py --target vgg19 --model resnet50 --n_im $N_IMG --batch_size 32 --eps 0.031 --alpha 0.0078 --seed $seed --steps 10 --decay 1.0 --gamma 0.5

  #DI+SGM
  python SGM-DI-FGSM-PGD.py --target vgg19 --model resnet18 --n_im $N_IMG --batch_size 32 --eps 0.031 --alpha 0.0078 --seed $seed --steps 10 --resize_rate 0.9 --diversity_prob 0.5
  python SGM-DI-FGSM-PGD.py --target vgg19 --model resnet50 --n_im $N_IMG --batch_size 32 --eps 0.031 --alpha 0.0078 --seed $seed --steps 10 --resize_rate 0.9 --diversity_prob 0.5

  #MI+DI+SGM
  python SGM-DI-FGSM.py --target vgg19 --model resnet18 --n_im $N_IMG --batch_size 32 --eps 0.031 --alpha 0.0078 --seed $seed --steps 10 --decay 1.0 --resize_rate 0.9 --diversity_prob 0.5
  python SGM-DI-FGSM.py --target vgg19 --model resnet50 --n_im $N_IMG --batch_size 32 --eps 0.031 --alpha 0.0078 --seed $seed --steps 10 --decay 1.0 --resize_rate 0.9 --diversity_prob 0.5
done
