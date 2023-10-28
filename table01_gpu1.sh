#Remaining work for table01: add target models

export CUDA_VISIBLE_DEVICES=1

for seed in 42 1 10; do
  # SES
  python attack_SES.py --untargeted --n_im 10000 --loss_name cw --iters 10 --fuse loss --x 3 --iterw 10 --target vgg19 --surrogate_names resnet18 resnet50 #resnet34 densenet161 densenet169 densenet121

  #TREMBA

  #before running, in config/ directory modify the train json file
  python train_generator_TREMBA.py --config TREMBA/config/train.json --surrogate_names resnet34 resnet50 --n_im_train 50000
  #before running, in config/ directory modify the attack  json file
  python attack_TREMBA.py --config TREMBA/config/attack.json --device cuda --save_prefix xp1 --target densenet161 --generator_name Cifar10_resnet34_resnet50train_untarget --n_im 10000

  # BASES
  python attack_BASES.py --untargeted --n_im 10000 --loss_name cw --iters 10 --fuse loss --x 3 --iterw 10 --target vgg19 --surrogate_names resnet18 resnet50 #resnet34 densenet161 densenet169 densenet121

  # EBAD
  python attack_EBAD.py --untargeted --n_im 10000 --loss_name cw --iters 10 --fuse loss --x 3 --iterw 10 --target vgg19 --surrogate_names resnet18 resnet50 #resnet34 densenet161 densenet169 densenet121


  # SES+DI
  python attack_SES.py --untargeted --n_im 10000 --loss_name cw --iters 10 --fuse loss --x 3 --iterw 10 --target vgg19 --algo 'di' --resize_rate 0.9 --diversity_prob 0.5 --surrogate_names resnet18 resnet50 #resnet34 densenet161 densenet169 densenet121

  # SES+SGM
  python attack_SES.py --untargeted --n_im 10000 --loss_name cw --iters 10 --fuse loss --x 3 --iterw 10 --target vgg19 --algo 'sgm' --gamma 0.5 --resize_rate 0.9 --diversity_prob 0.5 --surrogate_names resnet18 resnet50 #resnet34 densenet161 densenet169 densenet121


  # SES(with weight balancing from EBAD)
  python attack_SES_wb.py --untargeted --n_im 10000 --loss_name cw --iters 10 --fuse loss --x 3 --iterw 10 --target vgg19 --surrogate_names resnet18 resnet50 #resnet34 densenet161 densenet169 densenet121
done

