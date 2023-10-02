#Remaining work for table01: +SGM and add target models

#MI
python MI-FGSM.py --target resnet50 --model resnet18 --n_examples 4 --batch_size 2 --eps 0.031 --alpha 0.0078 --steps 10 --decay 1.0

#Ghost
python GN-PGD.py --target resnet50 --model resnet18 --n_examples 4 --batch_size 2 --eps 0.031 --alpha 0.0078 --steps 10

#DI
python DI-FGSM-PGD.py --target resnet50 --model resnet18 --n_examples 4 --batch_size 2 --eps 0.031 --alpha 0.0078 --steps 10 --resize_rate 0.9 --diversity_prob 0.5

#MI+Ghost
python GN-MI-FGSM.py --target resnet50 --model resnet18 --n_examples 4 --batch_size 2 --eps 0.031 --alpha 0.0078 --steps 10 --decay 1.0

#DI+Ghost
python GN-DI-FGSM-PGD.py --target resnet50 --model resnet18 --n_examples 4 --batch_size 2 --eps 0.031 --alpha 0.0078 --steps 10 --resize_rate 0.9 --diversity_prob 0.5

#MI+DI
python DI-FGSM.py --target resnet50 --model resnet18 --n_examples 4 --batch_size 2 --eps 0.031 --alpha 0.0078 --steps 10 --decay 1.0 --resize_rate 0.9 --diversity_prob 0.5

#MI+DI+Ghost
python GN-DI-FGSM.py --target resnet50 --model resnet18 --n_examples 4 --batch_size 2 --eps 0.031 --alpha 0.0078 --steps 10 --decay 1.0 --resize_rate 0.9 --diversity_prob 0.5

#Admix
python adm.py --target resnet50 --model resnet18 --n_examples 4 --batch_size 1 --admix-m1 0.5 --admix-m2 0.5

#Ensemble MI
python MI-FGSM-ENS.py --target resnet50 --model resnet18 resnet34 --n_examples 4 --batch_size 2 --eps 0.031 --alpha 0.0078 --steps 10 --decay 1.0

#Ensemble MI+Ghost
python GN-MI-FGSM-ENS.py --target resnet50 --model resnet18 resnet34 --n_examples 4 --batch_size 2 --eps 0.031 --alpha 0.0078 --steps 10 --decay 1.0



#TREMBA
#before running, in config/ directory modify the train json file 
python train_generator.py --config config/train.json --surrogate_names resnet34 resnet50
#before running, in config/ directory modify the attack  json file 
python attack.py --config config/attack.json --device cuda --save_prefix your_save_prefix --model_name densenet161 --generator_name Cifar10_densenet161_densenet169_vgg16train_untarget


# BASES
python attack_BASES.py --untargeted --n_im 2 --loss_name cw --iters 10 --fuse loss --x 3 --iterw 10 --target vgg19 --surrogate_names resnet18 resnet50 #resnet34 densenet161 densenet169 densenet121

# EBAD
python attack_EBAD.py --untargeted --n_im 2 --loss_name cw --iters 10 --fuse loss --x 3 --iterw 10 --target vgg19 --surrogate_names resnet18 resnet50 #resnet34 densenet161 densenet169 densenet121

# SES
python attack_SES.py --untargeted --n_im 2 --loss_name cw --iters 10 --fuse loss --x 3 --iterw 10 --target vgg19 --surrogate_names resnet18 resnet50 #resnet34 densenet161 densenet169 densenet121

# SES+DI
python attack_SES.py --untargeted --n_im 2 --loss_name cw --iters 10 --fuse loss --x 3 --iterw 10 --target vgg19 --surrogate_names resnet18 resnet50 #resnet34 densenet161 densenet169 densenet121 --algo 'di' --resize_rate 0.9 --diversity_prob 0.5

# SES(with weight balancing from EBAD)
python attack_SES_wb.py --untargeted --n_im 2 --loss_name cw --iters 10 --fuse loss --x 3 --iterw 10 --target vgg19 --surrogate_names resnet18 resnet50 #resnet34 densenet161 densenet169 densenet121


