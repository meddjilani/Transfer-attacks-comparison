#****************************** BASES ****************************** 
python attack_BASES.py --untargeted --n_im 100 --loss_name cw --iters 100 --fuse loss --x 3 --iterw 10 --target Standard --surrogate_names Ding2020MMA Zhang2019You Addepalli2022Efficient_WRN_34_10 Sitawarin2020Improving Carmon2019Unlabeled Wu2020Adversarial_extra Huang2021Exploring Xu2023Exploring_WRN-28-10


#****************************** EBAD ****************************** 
python attack_EBAD.py --untargeted --n_im 100 --loss_name cw --iters 100 --fuse loss --x 3 --iterw 10 --target Standard --surrogate_names Ding2020MMA Zhang2019You Addepalli2022Efficient_WRN_34_10 Sitawarin2020Improving Carmon2019Unlabeled Wu2020Adversarial_extra Huang2021Exploring Xu2023Exploring_WRN-28-10


#****************************** SES (ours) ****************************** 
python attack_SES.py --untargeted --n_im 100 --loss_name cw --iters 100 --fuse loss --x 3 --iterw 10 --target Standard --surrogate_names Ding2020MMA Zhang2019You Addepalli2022Efficient_WRN_34_10 Sitawarin2020Improving Carmon2019Unlabeled Wu2020Adversarial_extra Huang2021Exploring Xu2023Exploring_WRN-28-10


#****************************** SES+wb (ours) ****************************** 
python attack_SES_wb.py --untargeted --n_im 100 --loss_name cw --iters 100 --fuse loss --x 3 --iterw 10 --target Standard --surrogate_names Ding2020MMA Zhang2019You Addepalli2022Efficient_WRN_34_10 Sitawarin2020Improving Carmon2019Unlabeled Wu2020Adversarial_extra Huang2021Exploring Xu2023Exploring_WRN-28-10


#****************************** SES (ours) ****************************** 
#UNTARGETED
python attack_SES.py --untargeted --n_im 10 --loss_name cw --iters 10 --fuse loss --x 3 --iterw 10 --target resnet18 --surrogate_names resnet50 resnet34


#TARGETED
python attack_SES.py --n_im 10 --loss_name cw --iters 10 --fuse loss --x 3 --iterw 10 --target resnet18 --surrogate_names resnet50 resnet34


#INTER ARCHITECTURE 
python attack_SES.py --untargeted --n_im 10 --loss_name cw --iters 10 --fuse loss --x 3 --iterw 10 --target resnet18 --surrogate_names  densenet169 densenet121



#Robust models
python attack_SES.py --untargeted --n_im 10 --loss_name cw --iters 10 --fuse loss --x 3 --iterw 10 --target Addepalli2021Towards_RN18 --surrogate_names Sitawarin2020Improving Chen2020Efficient









