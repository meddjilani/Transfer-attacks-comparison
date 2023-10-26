export CUDA_VISIBLE_DEVICES=2

for seed in 42 1 10 100 1000; do
  for target in "Peng2023Robust" "Wang2023Better_WRN-70-16" "Rebuffi2021Fixing_70_16_cutmix_extra" "Gowal2021Improving_70_16_ddpm_100m" "Rebuffi2021Fixing_70_16_cutmix_ddpm" "Carmon2019Unlabeled" "Gowal2020Uncovering_70_16" "Hendrycks2019Using" "Cui2020Learnable_34_10" "Wong2020Fast"; do
    #SES
    python attack_SES.py --untargeted --n_im 10000 --loss_name cw --iters 10 --fuse loss --x 3 --iterw 30 --target $target --surrogate_names Cui2023Decoupled_WRN-28-10 Wang2023Better_WRN-28-10 Gowal2020Uncovering_70_16_extra Rebuffi2021Fixing_106_16_cutmix_ddpm Gowal2020Uncovering_28_10_extra Zhang2020Geometry Rade2021Helper_R18_ddpm Wang2020Improving Zhang2019Theoretically Andriushchenko2020Understanding --seed $seed  
    # BASES
    python attack_BASES.py --untargeted --n_im 10000 --loss_name cw --iters 10 --fuse loss --x 3 --iterw 30 --target $target --surrogate_names Cui2023Decoupled_WRN-28-10 Wang2023Better_WRN-28-10 Gowal2020Uncovering_70_16_extra Rebuffi2021Fixing_106_16_cutmix_ddpm Gowal2020Uncovering_28_10_extra Zhang2020Geometry Rade2021Helper_R18_ddpm Wang2020Improving Zhang2019Theoretically Andriushchenko2020Understanding --seed $seed
    # EBAD
    python attack_EBAD.py --untargeted --n_im 10000 --loss_name cw --iters 10 --fuse loss --x 3 --iterw 30 --target $target --surrogate_names Cui2023Decoupled_WRN-28-10 Wang2023Better_WRN-28-10 Gowal2020Uncovering_70_16_extra Rebuffi2021Fixing_106_16_cutmix_ddpm Gowal2020Uncovering_28_10_extra Zhang2020Geometry Rade2021Helper_R18_ddpm Wang2020Improving Zhang2019Theoretically Andriushchenko2020Understanding --seed $seed
    # SES+DI
    python attack_SES.py --untargeted --n_im 10000 --loss_name cw --iters 10 --fuse loss --x 3 --iterw 30 --target $target --algo 'di' --resize_rate 0.9 --diversity_prob 0.5 --surrogate_names Cui2023Decoupled_WRN-28-10 Wang2023Better_WRN-28-10 Gowal2020Uncovering_70_16_extra Rebuffi2021Fixing_106_16_cutmix_ddpm Gowal2020Uncovering_28_10_extra Zhang2020Geometry Rade2021Helper_R18_ddpm Wang2020Improving Zhang2019Theoretically Andriushchenko2020Understanding --seed $seed
    # SES+SGM
    python attack_SES.py --untargeted --n_im 10000 --loss_name cw --iters 10 --fuse loss --x 3 --iterw 30 --target $target --algo 'sgm' --gamma 0.5 --resize_rate 0.9 --diversity_prob 0.5 --surrogate_names Cui2023Decoupled_WRN-28-10 Wang2023Better_WRN-28-10 Gowal2020Uncovering_70_16_extra Rebuffi2021Fixing_106_16_cutmix_ddpm Gowal2020Uncovering_28_10_extra Zhang2020Geometry Rade2021Helper_R18_ddpm Wang2020Improving Zhang2019Theoretically Andriushchenko2020Understanding --seed $seed
    # SES(with weight balancing from EBAD)
    python attack_SES_wb.py --untargeted --n_im 10000 --loss_name cw --iters 10 --fuse loss --x 3 --iterw 30 --target $target --surrogate_names Cui2023Decoupled_WRN-28-10 Wang2023Better_WRN-28-10 Gowal2020Uncovering_70_16_extra Rebuffi2021Fixing_106_16_cutmix_ddpm Gowal2020Uncovering_28_10_extra Zhang2020Geometry Rade2021Helper_R18_ddpm Wang2020Improving Zhang2019Theoretically Andriushchenko2020Understanding --seed $seed
    #TREMBA
    #before running, in config/ directory modify the train json file 
    python train_generator_TREMBA.py --config TREMBA/config/train.json --surrogate_names Cui2023Decoupled_WRN-28-10 Wang2023Better_WRN-28-10 Gowal2020Uncovering_70_16_extra Rebuffi2021Fixing_106_16_cutmix_ddpm Gowal2020Uncovering_28_10_extra Zhang2020Geometry Rade2021Helper_R18_ddpm Wang2020Improving Zhang2019Theoretically Andriushchenko2020Understanding --n_im_train 60000
    #before running, in config/ directory modify the attack  json file 
    python attack_TREMBA.py --n_im 10000 --query_limit 100 --config TREMBA/config/attack.json --device cuda --target Standard --generator_name Cifar10_resnet34_resnet50_densenet161_densenet169_densenet121train_untarget --seed $seed
  done
done