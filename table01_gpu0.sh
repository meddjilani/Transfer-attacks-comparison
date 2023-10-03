#Remaining work for table01: add target models

export CUDA_VISIBLE_DEVICES=0
#TREMBA

#before running, in config/ directory modify the train json file
python train_generator_TREMBA.py --config TREMBA/config/train.json --surrogate_names resnet34 resnet50 --n_im_train 50000
#before running, in config/ directory modify the attack  json file
python attack_TREMBA.py --config TREMBA/config/attack.json --device cuda --save_prefix xp1 --target densenet161 --generator_name Cifar10_densenet161_densenet169_vgg16train_untarget