python MI-FGSM.py --model Gowal2021Improving_28_10_ddpm_100m --target Carmon2019Unlabeled --n_examples 10 --eps 0.031 --alpha 0.0078 --steps  10 --decay 1.0

python DI-FGSM.py --model Gowal2021Improving_28_10_ddpm_100m --target Carmon2019Unlabeled --n_examples 10 --eps 0.031 --alpha 0.0078 --steps  10 --decay 1.0 --resize_rate 0.9 --diversity_prob 0.5 --random_start False

python MI-FGSM_ENS.py --model Gowal2021Improving_28_10_ddpm_100m Standard --target Carmon2019Unlabeled --n_examples 10 --eps 0.031 --alpha 0.0078 --steps  10 --decay 1.0

python TI-FGSM.py --model Gowal2021Improving_28_10_ddpm_100m --target Carmon2019Unlabeled --n_examples 10 --eps 0.031 --alpha 0.0078 --steps  10 --decay 1.0 --resize_rate 0.9 --diversity_prob 0.5 --random_start False --kernel_name gaussian --len_kernel 15 --nsig 3

python VMI-FGSM.py --model Gowal2021Improving_28_10_ddpm_100m --target Carmon2019Unlabeled --n_examples 10 --eps 0.031 --alpha 0.0078 --steps  10 --decay 1.0 --N 5 --beta 1.5

python VNI-FGSM.py --model Gowal2021Improving_28_10_ddpm_100m --target Carmon2019Unlabeled --n_examples 10 --eps 0.031 --alpha 0.0078 --steps  10 --decay 1.0 --N 5 --beta 1.5

python adm.py --model Gowal2021Improving_28_10_ddpm_100m --target Carmon2019Unlabeled --n_examples 10 --batch_size 1 --admix-m1 0.5 --admix-m2 0.5

#Move to Skip gradient method
# For SGM download the resnet pretrained model from https://github.com/huyvnphan/PyTorch_CIFAR10.git and put it in cifar10_models/state_dicts/resnet50.pt
python attack_sgm.py  --target Carmon2019Unlabeled --n_examples 10 --num-steps 10 --step-size 2 --gamma 0.5 --momentum 0.0 --print_freq 10


# Move to MetaAttack_ICLR2020/meta_attack/meta_attack_cifar/
python test_all.py --attacked_model Carmon2019Unlabeled --n_examples 3 --maxiter 1000 --max_fintune_iter 63 --finetune_interval 3 --learning_rate 0.01 --update_pixels 125 --simba_update_pixels 125 --total_number 1000 --untargeted True --istransfer False
  

#Move to Bases 
python query_w_bb.py --model_name Carmon2019Unlabeled --surrogate_names Ding2020MMA Standard --bound linf --eps 8 --iters 10 --gpu 0 --fuse loss --loss_name cw --x 3 --lr 0.005 --iterw 50 --n_im 2 --untargeted True

#Move to Bases/comparison/GFCS
python GFCS_main.py --model_name Carmon2019Unlabeled --surrogate_names Ding2020MMA Standard --n_examples 3 --target_label 2 --GFCS --num_step 500 --linf 0.0625 --step_size 0.005 --device cuda