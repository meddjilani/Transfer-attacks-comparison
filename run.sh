python MI-FGSM.py --model Gowal2021Improving_28_10_ddpm_100m --target Carmon2019Unlabeled --n_examples 10 --eps 0.031 --alpha 0.0078 --steps  10 --decay 1.0

python GN-MI-FGSM.py --model Gowal2021Improving_28_10_ddpm_100m --target Carmon2019Unlabeled --n_examples 10 --eps 0.031 --alpha 0.0078 --steps  10 --decay 1.0

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
python query_w_bb.py --model_name Carmon2019Unlabeled --surrogate_names Ding2020MMA Standard --bound linf --eps 8 --iters 10 --gpu 0 --fuse loss --loss_name cw --x 3 --lr 0.005 --iterw 50 --n_im 2 --untargeted

#Move to Bases/comparison/GFCS
python GFCS_main.py --model_name Carmon2019Unlabeled --surrogate_names Ding2020MMA Standard --n_examples 3 --target_label 2 --GFCS --num_step 500 --linf 0.0625 --step_size 0.005 --device cuda

#Move to pytorch-gd-uap
python train.py --model Carmon2019Unlabeled --n_examples 1000 --max_iter 10000 --eps 8 --batch_size 100  --dataset_name cifar10 --patience_interval 5 --id 1 --prior_type no_data

#TREMBA
#before running, in config/ directory modify the train json file 
python train_generator.py --config config/train_untarget.json --surrogate_names Standard Ding2020MMA
#before running, in config/ directory modify the attack  json file 
python attack.py --config config/attack_untarget.json --device cuda --save_prefix your_save_prefix --model_name Carmon2019Unlabeled --generator_name Cifar10_Standard_Ding2020MMA_untarget --dataset cifar10

#DaST
#move to DaST/ and create an a folder that has same name as the value of --save_folder
#train gan and substitute model
python dast_cifar10.py --source_model Gowal2021Improving_28_10_ddpm_100m --model Carmon2019Unlabeled --workers 2 --batch_size 200 --niter 4 --lr 0.0001 --beta1 0.5 --alpha 0.2 --beta 0.1 --G_type 1 --save_folder saved_model
#attack
python evaluation.py --substitute_model Gowal2021Improving_28_10_ddpm_100m --model Carmon2019Unlabeled --n_examples 100 --batch_size 5 --eps 0.03137254901960784 --alpha 0.00784313725490196 --decay 1.0 --steps 10 --path saved_model/netD_epoch_1.pth --adv BIM --mode dast --target_label 0