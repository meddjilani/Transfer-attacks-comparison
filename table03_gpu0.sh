export CUDA_VISIBLE_DEVICES=0

#MI
python MI-FGSM.py --target Standard --model resnet18 --n_examples 10000 --batch_size 32 --eps 0.031 --alpha 0.0078 --steps 10 --decay 1.0 --seed 42

#Ghost
python GN-PGD.py --target Standard --model resnet18 --n_examples 10000 --batch_size 32 --eps 0.031 --alpha 0.0078 --steps 10 --seed 42

#DI
python DI-FGSM-PGD.py --target Standard --model resnet18 --n_examples 10000 --batch_size 32 --eps 0.031 --alpha 0.0078 --steps 10 --resize_rate 0.9 --diversity_prob 0.5 --seed 42

#MI+Ghost
python GN-MI-FGSM.py --target Standard --model resnet18 --n_examples 10000 --batch_size 32 --eps 0.031 --alpha 0.0078 --steps 10 --decay 1.0 --seed 42

#DI+Ghost
python GN-DI-FGSM-PGD.py --target Standard --model resnet18 --n_examples 10000 --batch_size 32 --eps 0.031 --alpha 0.0078 --steps 10 --resize_rate 0.9 --diversity_prob 0.5 --seed 42

#MI+DI
python DI-FGSM.py --target Standard --model resnet18 --n_examples 10000 --batch_size 32 --eps 0.031 --alpha 0.0078 --steps 10 --decay 1.0 --resize_rate 0.9 --diversity_prob 0.5 --seed 42

#MI+DI+Ghost
python GN-DI-FGSM.py --target Standard --model resnet18 --n_examples 10000 --batch_size 32 --eps 0.031 --alpha 0.0078 --steps 10 --decay 1.0 --resize_rate 0.9 --diversity_prob 0.5 --seed 42

#Admix
python adm.py --target Standard --model resnet18 --n_examples 10000 --batch_size 32 --admix-m1 0.5 --admix-m2 0.5 --seed 42

#Ensemble MI
python MI-FGSM-ENS.py --target Standard --model resnet18 resnet34 resnet50 densenet161 densenet169 densenet121 --n_examples 10000 --batch_size 32 --eps 0.031 --alpha 0.0078 --steps 10 --decay 1.0 --seed 42

#Ensemble MI+Ghost
python GN-MI-FGSM-ENS.py --target Standard --model resnet18 resnet34 resnet50 densenet161 densenet169 densenet121 --n_examples 10000 --batch_size 32 --eps 0.031 --alpha 0.0078 --steps 10 --decay 1.0 --seed 42

#MI+SGM
python SGM-MI-FGSM.py --target Standard --model resnet18 --n_examples 10000 --batch_size 32 --eps 0.031 --alpha 0.0078 --steps 10 --decay 1.0 --gamma 0.5 --seed 42

#DI+SGM
python SGM-DI-FGSM-PGD.py --target Standard --model resnet18 --n_examples 10000 --batch_size 32 --eps 0.031 --alpha 0.0078 --steps 10 --resize_rate 0.9 --diversity_prob 0.5 --seed 42

#MI+DI+SGM
python SGM-DI-FGSM.py --target Standard --model resnet18 --n_examples 10000 --batch_size 32 --eps 0.031 --alpha 0.0078 --steps 10 --decay 1.0 --resize_rate 0.9 --diversity_prob 0.5 --seed 42

