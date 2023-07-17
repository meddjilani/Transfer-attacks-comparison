DEVICE=0

SURROGATES="Wang2023Better_WRN-70-16 Wang2023Better_WRN-28-10  Xu2023Exploring_WRN-28-10 Huang2021Exploring   Wu2020Adversarial_extra Carmon2019Unlabeled Sitawarin2020Improving Addepalli2022Efficient_WRN_34_10  Zhang2019You  Ding2020MMA"
TARGET='Standard'
CUDA_VISIBLE_DEVICES=$DEVICE python New_Attack/lgv_query_w_bb_avg_weights.py --surrogate_names $SURROGATES --model_name $TARGET