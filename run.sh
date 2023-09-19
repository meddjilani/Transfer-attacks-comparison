
#****************************** BASES ****************************** 
python sm_baseline.py --untargeted --n_im 10 --loss_name cw --iters 10 --fuse loss --x 3 --lr 0.5 --iterw 10 --target resnet18 --surrogate_names resnet50 resnet34


#****************************** OURS ****************************** 
#UNTARGETED
python sm_baseline_ghost.py --untargeted --n_im 10 --loss_name cw --iters 10 --fuse loss --x 3 --lr 0.5 --iterw 10 --target resnet18 --surrogate_names resnet50 resnet34


#TARGETED
python sm_baseline_ghost.py --n_im 10 --loss_name cw --iters 10 --fuse loss --x 3 --lr 0.5 --iterw 10 --target resnet18 --surrogate_names resnet50 resnet34


#INTER ARCHITECTURE 
python sm_baseline_ghost.py --untargeted --n_im 10 --loss_name cw --iters 10 --fuse loss --x 3 --lr 0.5 --iterw 10 --target resnet18 --surrogate_names  densenet169 densenet121



#Robust models
python sm_baseline_ghost.py --untargeted --n_im 10 --loss_name cw --iters 10 --fuse loss --x 3 --lr 0.5 --iterw 10 --target Addepalli2021Towards_RN18 --surrogate_names Sitawarin2020Improving Chen2020Efficient









