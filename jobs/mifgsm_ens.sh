#!/bin/bash -l

#SBATCH --mail-user=salah.ghamizi@uni.lu
#SBATCH -J "C10_MIFGSM_ENS"
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -C volta
#SBATCH -G 1
#SBATCH --time=24:00:00
#SBATCH -p gpu
#SBATCH --mail-type=end,fail

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
BATCH_SIZE=256
EPS=0.0313
ALPHA=0.0078
DECAY=1.0
STEPS=100
TARGETS='Standard Andriushchenko2020Understanding Carmon2019Unlabeled Gowal2021Improving_28_10_ddpm_100m Chen2020Adversarial Engstrom2019Robustness Wong2020Fast Ding2020MMA Gowal2021Improving_70_16_ddpm_100m Rebuffi2021Fixing_28_10_cutmix_ddpm Rebuffi2021Fixing_70_16_cutmix_extra'
MODELS1='Standard Andriushchenko2020Understanding Carmon2019Unlabeled Gowal2021Improving_28_10_ddpm_100m Chen2020Adversarial Engstrom2019Robustness Wong2020Fast Ding2020MMA Gowal2021Improving_70_16_ddpm_100m Rebuffi2021Fixing_28_10_cutmix_ddpm Rebuffi2021Fixing_70_16_cutmix_extra'
MODELS2='Standard Andriushchenko2020Understanding Carmon2019Unlabeled Gowal2021Improving_28_10_ddpm_100m Chen2020Adversarial Engstrom2019Robustness Wong2020Fast Ding2020MMA Gowal2021Improving_70_16_ddpm_100m Rebuffi2021Fixing_28_10_cutmix_ddpm Rebuffi2021Fixing_70_16_cutmix_extra'
MODELS='Standard Andriushchenko2020Understanding Carmon2019Unlabeled Gowal2021Improving_28_10_ddpm_100m Engstrom2019Robustness Wong2020Fast Ding2020MMA Gowal2021Improving_70_16_ddpm_100m Rebuffi2021Fixing_28_10_cutmix_ddpm Rebuffi2021Fixing_70_16_cutmix_extra'

for MODEL1 in $MODELS1
do
  for MODEL2 in $MODELS
  do
    if [[ "$MODEL1" == "$MODEL2" ]]; then
    continue
    fi
    for MODEL3 in $MODELS2
    do
      if [[ "$MODEL3" == "$MODEL2" || "$MODEL3" == "$MODEL1" ]]; then
      continue
      fi
       for TARGET in $TARGETS
       do
         CUDA_VISIBLE_DEVICES=0 python MI-FGSM_ENS.py --model $MODEL1 $MODEL2 $MODEL3 --target $TARGET --n_examples $N_EXAMPLES --eps $EPS --alpha $ALPHA --steps $STEPS --decay $DECAY --batch_size $BATCH_SIZE
       done
    done
  done
done