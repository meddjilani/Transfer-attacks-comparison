export CUDA_VISIBLE_DEVICES=0
N_IMG=1000
for seed in 42 1 10 100 1000; do
  for target in "Peng2023Robust" "Wang2023Better_WRN-70-16" "Rebuffi2021Fixing_70_16_cutmix_extra" "Gowal2021Improving_70_16_ddpm_100m" "Rebuffi2021Fixing_70_16_cutmix_ddpm" "Carmon2019Unlabeled" "Gowal2020Uncovering_70_16" "Hendrycks2019Using" "Cui2020Learnable_34_10" "Wong2020Fast"; do
    #MI
    python MI-FGSM.py --target $target --model Zhang2020Geometry --n_examples $N_IMG --batch_size 32 --eps 0.031 --alpha 0.0078 --steps 10 --decay 1.0 --seed $seed
    #Ghost
    python GN-PGD.py --target $target --model Zhang2020Geometry --n_examples $N_IMG --batch_size 32 --eps 0.031 --alpha 0.0078 --steps 10 --seed $seed
    #DI
    python DI-FGSM-PGD.py --target $target --model Zhang2020Geometry --n_examples $N_IMG --batch_size 32 --eps 0.031 --alpha 0.0078 --steps 10 --resize_rate 0.9 --diversity_prob 0.5 --seed $seed
    #MI+Ghost
    python GN-MI-FGSM.py --target $target --model Zhang2020Geometry --n_examples $N_IMG --batch_size 32 --eps 0.031 --alpha 0.0078 --steps 10 --decay 1.0 --seed $seed
    #DI+Ghost
    python GN-DI-FGSM-PGD.py --target $target --model Zhang2020Geometry --n_examples $N_IMG --batch_size 32 --eps 0.031 --alpha 0.0078 --steps 10 --resize_rate 0.9 --diversity_prob 0.5 --seed $seed
    #MI+DI
    python DI-FGSM.py --target $target --model Zhang2020Geometry --n_examples $N_IMG --batch_size 32 --eps 0.031 --alpha 0.0078 --steps 10 --decay 1.0 --resize_rate 0.9 --diversity_prob 0.5 --seed $seed
    #MI+DI+Ghost
    python GN-DI-FGSM.py --target $target --model Zhang2020Geometry --n_examples $N_IMG --batch_size 32 --eps 0.031 --alpha 0.0078 --steps 10 --decay 1.0 --resize_rate 0.9 --diversity_prob 0.5 --seed $seed
    #Admix
    python adm.py --target $target --model Zhang2020Geometry --n_examples $N_IMG --batch_size 32 --admix-m1 0.5 --admix-m2 0.5 --seed $seed
    #Ensemble MI
    python MI-FGSM-ENS.py --target $target --model Cui2023Decoupled_WRN-28-10 Wang2023Better_WRN-28-10 Gowal2020Uncovering_70_16_extra Rebuffi2021Fixing_106_16_cutmix_ddpm Gowal2020Uncovering_28_10_extra Zhang2020Geometry Rade2021Helper_R18_ddpm Wang2020Improving Zhang2019Theoretically Andriushchenko2020Understanding --n_examples $N_IMG --batch_size 32 --eps 0.031 --alpha 0.0078 --steps 10 --decay 1.0 --seed $seed
    #Ensemble MI+Ghost
    python GN-MI-FGSM-ENS.py --target $target --model Cui2023Decoupled_WRN-28-10 Wang2023Better_WRN-28-10 Gowal2020Uncovering_70_16_extra Rebuffi2021Fixing_106_16_cutmix_ddpm Gowal2020Uncovering_28_10_extra Zhang2020Geometry Rade2021Helper_R18_ddpm Wang2020Improving Zhang2019Theoretically Andriushchenko2020Understanding --n_examples $N_IMG --batch_size 32 --eps 0.031 --alpha 0.0078 --steps 10 --decay 1.0 --seed $seed
    #MI+SGM
    python SGM-MI-FGSM.py --target $target --model Zhang2020Geometry --n_examples $N_IMG --batch_size 32 --eps 0.031 --alpha 0.0078 --steps 10 --decay 1.0 --gamma 0.5 --seed $seed
    #DI+SGM
    python SGM-DI-FGSM-PGD.py --target $target --model Zhang2020Geometry --n_examples $N_IMG --batch_size 32 --eps 0.031 --alpha 0.0078 --steps 10 --resize_rate 0.9 --diversity_prob 0.5 --seed $seed
    #MI+DI+SGM
    python SGM-DI-FGSM.py --target $target --model Zhang2020Geometry --n_examples $N_IMG --batch_size 32 --eps 0.031 --alpha 0.0078 --steps 10 --decay 1.0 --resize_rate 0.9 --diversity_prob 0.5 --seed $seed
  done
done