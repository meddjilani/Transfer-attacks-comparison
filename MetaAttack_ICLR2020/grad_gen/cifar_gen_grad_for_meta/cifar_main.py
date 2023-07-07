from __future__ import print_function

import sys
sys.path.append("./MetaAttack_ICLR2020/grad_gen/cifar_gen_grad_for_meta/")

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from cifar_models import *
from utils import save_gradient
from robustbench.utils import load_model


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Gradient computation')
parser.add_argument("--model", type = str, default = 'Carmon2019Unlabeled', help = 'the model selected to be used for meta-model')
parser.add_argument("--dataset", choices=["mnist", "cifar10", "imagenet"], default = "cifar10")

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if __name__ == '__main__':

    models_name = args.model.split(" ")
    transform_train = transforms.Compose([
                      transforms.ToTensor(),
                      ])

    transform_test = transforms.Compose([
                      transforms.ToTensor(),
                      ])

    batch_size = 25
    trainset = torchvision.datasets.CIFAR10(root='../datasets', train = True, download = True, transform = transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle = True, num_workers = 2)

    testset = torchvision.datasets.CIFAR10(root='../datasets', train = False, download = True, transform = transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 2)

    for i in range(len(models_name)):
        m_name = models_name[i]
        net = load_model(model_name=m_name, dataset=args.dataset, threat_model='Linf', model_dir="../models")
        net = net.to(device)
        save_gradient(net, device, trainloader, m_name, mode = 'train')
        save_gradient(net, device, testloader, m_name, mode = 'test')
