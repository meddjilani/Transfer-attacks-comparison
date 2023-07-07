#!/bin/bash -l

#SBATCH --mail-user=salah.ghamizi@uni.lu
#SBATCH -J "C10_META"
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
BATCH_SIZE=1024
EPS=0.0313
LR=0.01
STEPS=10000

# generating gradients for meta model training
CUDA_VISIBLE_DEVICES=0 python cifar_main.py


# training meta model
cd "./MetaAttack_ICLR2020/meta_training/cifar_meta_training"
CUDA_VISIBLE_DEVICES=0 python cifar_train.py

TARGETS='Standard Andriushchenko2020Understanding Carmon2019Unlabeled Gowal2021Improving_28_10_ddpm_100m Chen2020Adversarial Engstrom2019Robustness Wong2020Fast Ding2020MMA Gowal2021Improving_70_16_ddpm_100m Rebuffi2021Fixing_28_10_cutmix_ddpm Rebuffi2021Fixing_70_16_cutmix_extra'


for TARGET in $TARGETS
do

 CUDA_VISIBLE_DEVICES=0 python meta_attack.py --target $TARGET --n_examples $N_EXAMPLES --eps $EPS --lr $LR --steps $STEPS --batch_size $BATCH_SIZE
done
