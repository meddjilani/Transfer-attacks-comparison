#!/bin/bash -l

#SBATCH --mail-user=salah.ghamizi@uni.lu
#SBATCH -J "C10_MIFGSM"
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -C volta
#SBATCH -G 1
#SBATCH --time=16:00:00
#SBATCH -p gpu
#SBATCH --mail-type=end fail

print_error_and_exit() { echo "***ERROR*** $*"; exit 1; }
module purge || print_error_and_exit "No 'module'command"

cd ~/transfer/
python3 -m venv ~/venv/salah
source ~/venv/salah/bin/activate
module load lang/Python/3.8.6-GCCcore-10.2.0
pip install --user  -r requirements.txt

MODEL='Gowal2021Improving_28_10_ddpm_100m'
TARGET='Carmon2019Unlabeled'
N_EXAMPLES=10000
EPS=8/255
ALPHA=2/255
DECAY=1.0
STEPS=100

#CUDA_VISIBLE_DEVICES=0 python MI-FGSM.py --model $MODEL  --target $TARGET --n_examples $N_EXAMPLES --eps $EPS --alpha $ALPHA --steps $STEPS --decay $DECAY

MODELS=('Andriushchenko2020Understanding'  'Carmon2019Unlabeled'  'Sehwag2020Hydra'  'Wang2020Improving'  'Hendrycks2019Using'  'Rice2020Overfitting'  'Zhang2019Theoretically'  'Engstrom2019Robustness'  'Chen2020Adversarial'  'Huang2020Self'  'Pang2020Boosting'  'Wong2020Fast'  'Ding2020MMA'  'Zhang2019You'  'Zhang2020Attacks'  'Wu2020Adversarial_extra'  'Wu2020Adversarial'  'Gowal2020Uncovering_70_16'  'Gowal2020Uncovering_70_16_extra'  'Gowal2020Uncovering_34_20'  'Gowal2020Uncovering_28_10_extra'  'Sehwag2021Proxy'  'Sehwag2021Proxy_R18'  'Sehwag2021Proxy_ResNest152'  'Sitawarin2020Improving'  'Chen2020Efficient'  'Cui2020Learnable_34_20'  'Cui2020Learnable_34_10'  'Zhang2020Geometry'  'Rebuffi2021Fixing_28_10_cutmix_ddpm'  'Rebuffi2021Fixing_106_16_cutmix_ddpm'  'Rebuffi2021Fixing_70_16_cutmix_ddpm'  'Rebuffi2021Fixing_70_16_cutmix_extra'  'Sridhar2021Robust'  'Sridhar2021Robust_34_15'  'Rebuffi2021Fixing_R18_ddpm'  'Rade2021Helper_R18_extra'  'Rade2021Helper_R18_ddpm'  'Rade2021Helper_extra'  'Rade2021Helper_ddpm'  'Huang2021Exploring'  'Huang2021Exploring_ema'  'Addepalli2021Towards_RN18'  'Addepalli2021Towards_WRN34'  'Gowal2021Improving_70_16_ddpm_100m'  'Dai2021Parameterizing'  'Gowal2021Improving_28_10_ddpm_100m'  'Gowal2021Improving_R18_ddpm_100m'  'Chen2021LTD_WRN34_10'  'Chen2021LTD_WRN34_20'  'Standard'  'Kang2021Stable'  'Jia2022LAS-AT_34_10'  'Jia2022LAS-AT_70_16'  'Pang2022Robustness_WRN28_10'  'Pang2022Robustness_WRN70_16'  'Addepalli2022Efficient_RN18'  'Addepalli2022Efficient_WRN_34_10')
for MODEL in ${MODELS[@]}; do
 for TARGET in ${MODELS[@]}; do
   CUDA_VISIBLE_DEVICES=0 python MI-FGSM.py --model $MODEL  --target $TARGET --n_examples $N_EXAMPLES --eps $EPS --alpha $ALPHA --steps $STEPS --decay $DECAY
   done
done
