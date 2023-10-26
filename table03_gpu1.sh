export CUDA_VISIBLE_DEVICES=2
for seed in 42 1 10 100 1000; do
  for target in "Peng2023Robust" "Wang2023Better WRN-70-16" "Rebuffi2021Fixing 70 16 cutmix extra" "Gowal2021Improving 70 16 ddpm 100m" "Rebuffi2021Fixing 70 16 cutmix ddpm" "Carmon2019Unlabeled" "Gowal2020Uncovering 70 16" "Hendrycks2019Using" "Cui2020Learnable 34 10" "Wong2020Fas"; do
    #SES
    python attack_SES.py --untargeted --n_im 10000 --loss_name cw --iters 10 --fuse loss --x 3 --iterw 30 --target $target --surrogate_names resnet18 resnet50 resnet34 densenet161 densenet169 densenet121 --seed $seed  
    # BASES
    python attack_BASES.py --untargeted --n_im 10000 --loss_name cw --iters 10 --fuse loss --x 3 --iterw 30 --target $target --surrogate_names resnet18 resnet50 resnet34 densenet161 densenet169 densenet121 --seed $seed
    # EBAD
    python attack_EBAD.py --untargeted --n_im 10000 --loss_name cw --iters 10 --fuse loss --x 3 --iterw 30 --target $target --surrogate_names resnet18 resnet50 resnet34 densenet161 densenet169 densenet121 --seed $seed
    # SES+DI
    python attack_SES.py --untargeted --n_im 10000 --loss_name cw --iters 10 --fuse loss --x 3 --iterw 30 --target $target --algo 'di' --resize_rate 0.9 --diversity_prob 0.5 --surrogate_names resnet18 resnet50 resnet34 densenet161 densenet169 densenet121 --seed $seed
    # SES+SGM
    python attack_SES.py --untargeted --n_im 10000 --loss_name cw --iters 10 --fuse loss --x 3 --iterw 30 --target $target --algo 'sgm' --gamma 0.5 --resize_rate 0.9 --diversity_prob 0.5 --surrogate_names resnet18 resnet50 resnet34 densenet161 densenet169 densenet121 --seed $seed
    # SES(with weight balancing from EBAD)
    python attack_SES_wb.py --untargeted --n_im 10000 --loss_name cw --iters 10 --fuse loss --x 3 --iterw 30 --target $target --surrogate_names resnet18 resnet50 resnet34 densenet161 densenet169 densenet121 --seed $seed
    #TREMBA
    #before running, in config/ directory modify the train json file 
    python train_generator_TREMBA.py --config TREMBA/config/train.json --surrogate_names resnet34 resnet50 densenet161 densenet169 densenet121 --n_im_train 60000
    #before running, in config/ directory modify the attack  json file 
    python attack_TREMBA.py --n_im 10000 --query_limit 100 --config TREMBA/config/attack.json --device cuda --target Standard --generator_name Cifar10_resnet34_resnet50_densenet161_densenet169_densenet121train_untarget --seed $seed
  done
done