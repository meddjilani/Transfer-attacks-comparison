export CUDA_VISIBLE_DEVICES=1

#SES
python attack_SES.py --untargeted --n_im 10000 --loss_name cw --iters 10 --fuse loss --x 3 --iterw 30 --target Standard --surrogate_names resnet18 resnet50 resnet34 densenet161 densenet169 densenet121 --seed 42

# BASES
python attack_BASES.py --untargeted --n_im 10000 --loss_name cw --iters 10 --fuse loss --x 3 --iterw 30 --target Standard --surrogate_names resnet18 resnet50 resnet34 densenet161 densenet169 densenet121 --seed 42

# EBAD
python attack_EBAD.py --untargeted --n_im 10000 --loss_name cw --iters 10 --fuse loss --x 3 --iterw 30 --target Standard --surrogate_names resnet18 resnet50 resnet34 densenet161 densenet169 densenet121 --seed 42


# SES+DI
python attack_SES.py --untargeted --n_im 10000 --loss_name cw --iters 10 --fuse loss --x 3 --iterw 30 --target Standard --algo 'di' --resize_rate 0.9 --diversity_prob 0.5 --surrogate_names resnet18 resnet50 resnet34 densenet161 densenet169 densenet121 --seed 42

# SES+SGM
python attack_SES.py --untargeted --n_im 10000 --loss_name cw --iters 10 --fuse loss --x 3 --iterw 30 --target Standard --algo 'sgm' --gamma 0.5 --resize_rate 0.9 --diversity_prob 0.5 --surrogate_names resnet18 resnet50 resnet34 densenet161 densenet169 densenet121 --seed 42

# SES(with weight balancing from EBAD)
python attack_SES_wb.py --untargeted --n_im 10000 --loss_name cw --iters 10 --fuse loss --x 3 --iterw 30 --target Standard --surrogate_names resnet18 resnet50 resnet34 densenet161 densenet169 densenet121 --seed 42

#TREMBA
#before running, in config/ directory modify the train json file 
python train_generator.py --config config/train.json --surrogate_names resnet34 resnet50 densenet161 densenet169 densenet121
#before running, in config/ directory modify the attack  json file 
python attack.py --n_im 10000 --query_limit 100 --config config/attack.json --device cuda --save_prefix your_save_prefix --model_name Standard --generator_name Cifar10_resnet34_resnet50_densenet161_densenet169_densenet121train_untarget --seed 42


