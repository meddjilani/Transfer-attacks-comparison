import torch
import torch.nn as nn
from cifar10_models.resnet import resnet50, resnet18
from robustbench.data import load_cifar10
from robustbench.utils import load_model, clean_accuracy
import torchattacks
import json
import os
import argparse

#add:
    # args parm of di
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Gowal2021Improving_28_10_ddpm_100m')
    parser.add_argument('--target', type=str, default= 'Carmon2019Unlabeled')
    parser.add_argument('--n_examples', type=int, default=10)
    parser.add_argument('--eps', type = float, default=8/255)
    parser.add_argument('--alpha', type=float,default=2/255)
    parser.add_argument('--decay', type=float,default= 1.0)
    parser.add_argument('--steps', type=int,default=10)  
    parser.add_argument('--resize_rate', type=float,default= 0.9)
    parser.add_argument('--diversity_prob', type=float,default= 0.5)
    parser.add_argument('--random_start', type=bool,default= False)
    args = parser.parse_args()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    source_model = load_model(model_name=args.model, dataset='cifar10', threat_model='Linf')
    source_model.to(device)

    target_model = load_model(model_name= args.target, dataset='cifar10', threat_model='Linf')
    target_model.to(device)


    x_test, y_test = load_cifar10(n_examples=args.n_examples)
    x_test, y_test = x_test.to(device), y_test.to(device)

    print('Running DI-FGSM attack')
    attack = torchattacks.DIFGSM(source_model, eps=args.eps, alpha=args.alpha, steps=args.steps, decay=args.decay,
                                 resize_rate=args.resize_rate, diversity_prob=args.diversity_prob, random_start=args.random_start)
    adv_images_DI = attack(x_test, y_test)

    acc = clean_accuracy(target_model, x_test, y_test) 
    rob_acc = clean_accuracy(target_model, adv_images_DI, y_test) 
    print(args.target, 'Clean Acc: %2.2f %%'%(acc*100))
    print(args.target, 'Robust Acc: %2.2f %%'%(rob_acc*100))