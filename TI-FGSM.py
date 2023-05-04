from comet_ml import Experiment
import torch
import torch.nn as nn
from cifar10_models.resnet import *
from cifar10_models.densenet import *
from cifar10_models.vgg import *
from robustbench.data import load_cifar10
from robustbench.utils import load_model, clean_accuracy
import torchattacks
import json
import os
from Normalize import Normalize
import argparse


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
    parser.add_argument('--kernel_name', type=str,default= 'gaussian')
    parser.add_argument('--len_kernel', type=int,default= 15)
    parser.add_argument('--nsig', type=int,default= 3)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    source_model = load_model(model_name=args.model, dataset='cifar10', threat_model='Linf')
    source_model.to(device)

    target_model = load_model(model_name= args.target, dataset='cifar10', threat_model='Linf')
    target_model.to(device)

    x_test, y_test = load_cifar10(n_examples=args.n_examples)
    x_test, y_test = x_test.to(device), y_test.to(device)

    print('Running TI-FGSM attack')
    attack = torchattacks.TIFGSM(source_model, eps=args.eps, alpha=args.alpha, steps=args.steps, decay=args.decay,
                                resize_rate=args.resize_rate, diversity_prob=args.diversity_prob, random_start=args.random_start,
                                kernel_name=args.kernel_name, len_kernel=args.len_kernel, nsig=args.nsig)
    adv_images_TI = attack(x_test, y_test)

    acc = clean_accuracy(target_model, x_test, y_test) 
    rob_acc = clean_accuracy(target_model, adv_images_TI, y_test) 
    print(args.target, 'Clean Acc: %2.2f %%'%(acc*100))
    print(args.target, 'Robust Acc: %2.2f %%'%(rob_acc*100))


    experiment = Experiment(
    api_key="RDxdOE04VJ1EPRZK7oB9a32Gx",
    project_name="Black-box attack comparison cifar10",
    workspace="meddjilani",
    )

    metrics = {'Clean accuracy': acc, 'Robust accuracy': rob_acc}
    experiment.log_metrics(metrics, step=1)

    parameters = {'attack':'TI-FGSM', 'source':args.model, 'target':args.target, 'n_examples':args.n_examples, 'eps':args.eps, 'alpha':args.alpha, 'steps':args.steps, 'decay':args.decay,
                  'resize_rate':args.resize_rate, 'diversity_prob':args.diversity_prob, 'random_start':args.random_start,
                  'kernel_name':args.kernel_name, 'len_kernel':args.len_kernel, 'nsig':args.nsig
    }
    experiment.log_parameters(parameters)