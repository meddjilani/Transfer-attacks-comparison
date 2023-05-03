import tqdm
import torch
import torch.nn as nn
from cifar10_models.resnet import *
from cifar10_models.densenet import *
from cifar10_models.vgg import *
from robustbench.data import load_cifar10
from robustbench.utils import load_model, clean_accuracy
import torchattacks_ens.attacks.mifgsm as taamifgsm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F

import json
import os
from Normalize import Normalize

from admix import Admix_Attacker
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Gowal2021Improving_28_10_ddpm_100m')
    parser.add_argument('--target', type=str, default= 'Carmon2019Unlabeled')
    parser.add_argument('--n_examples', type=int, default=10)
    parser.add_argument("--admix-m1", type=float, help="Admix")
    parser.add_argument("--admix-m2", type=float, help="Admix")
    parser.add_argument("--admix-portion", type=float, help="Admix")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    source_model = load_model(model_name=args.model, dataset='cifar10', threat_model='Linf')
    source_model.to(device)

    target_model = load_model(model_name= args.target, dataset='cifar10', threat_model='Linf')
    target_model.to(device)

    x_test, y_test = load_cifar10(n_examples=args.n_examples)
    x_test, y_test = x_test.to(device), y_test.to(device)

    dataset_test = TensorDataset(x_test, y_test)
    data_loader = DataLoader(dataset_test, batch_size=20)

    #admix
    attack = Admix_Attacker("admix",source_model, F.cross_entropy,args)

    source_model.eval()
    i = 0
    for inputs, labels in data_loader:
        inputs = inputs#.to(device)
        labels = labels#.to(device)

        with torch.no_grad():
            _, preds = torch.max(source_model(inputs), dim=1)

        inputs_adv = attack.perturb(inputs, preds)
        if i==0:
            adv_images_Admix = inputs_adv
        else:
            adv_images_Admix = torch.cat([adv_images_Admix,inputs_adv],dim=0)
        i += 1

    #admix

    acc = clean_accuracy(target_model, x_test, y_test) 
    rob_acc = clean_accuracy(target_model, adv_images_Admix, y_test) 
    print(args.target, 'Clean Acc: %2.2f %%'%(acc*100))
    print(args.target, 'Robust Acc: %2.2f %%'%(rob_acc*100))