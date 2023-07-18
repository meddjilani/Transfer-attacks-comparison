DEVICE=1

SURROGATES="Rebuffi2021Fixing_70_16_cutmix_extra Huang2022Revisiting_WRN-A4 Pang2022Robustness_WRN70_16 Huang2021Exploring Carmon2019Unlabeled Jia2022LAS-AT_70_16 Gowal2020Uncovering_70_16 Debenedetti2022Light_XCiT-S12 Andriushchenko2020Understanding Standard"
TARGET='Wang2023Better_WRN-70-16'
CUDA_VISIBLE_DEVICES=$DEVICE python New_Attack/lgv_query_w_bb_avg_weights.py --surrogate_names $SURROGATES --model_name $TARGET --iterw 500