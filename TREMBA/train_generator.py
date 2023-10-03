import sys
sys.path.append('../')

import argparse
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
from FCN import *
from utils import *
import torchvision.models as models
from robustbench.data import load_cifar10
from robustbench.utils import load_model

from Normalize import Normalize
from cifar10_models.vgg import vgg16_bn,vgg11_bn,vgg19_bn, vgg13_bn
from cifar10_models.resnet import resnet18,resnet34,resnet50
from cifar10_models.inception import inception_v3
from cifar10_models.googlenet import googlenet
from cifar10_models.densenet import densenet161, densenet121, densenet169
from cifar10_models.mobilenetv2 import mobilenet_v2

#import warnings


if __name__ == '__main__':

    #warnings.simplefilter("ignore") 
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_im_train', type=int, default=60000)
    parser.add_argument('--n_im_test', type=int, default=100)
    parser.add_argument('--config', default='config/train.json', help='config file')
    parser.add_argument('--surrogate_names',  nargs='+', default=['Standard','Ding2020MMA'], help='Surrogate models to use.')

    args = parser.parse_args()

    with open(args.config) as config_file:
        state = json.load(config_file)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_dataset, range(args.n_im_train)), batch_size=state['batch_size_train'], shuffle=True, pin_memory=True)
    test_dataset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader( torch.utils.data.Subset(test_dataset, range(args.n_im_test)), batch_size=state['batch_size_test'], shuffle=False, pin_memory=True) 

 
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2471, 0.2435, 0.2616]    
    nets = []
    for surrogate_model in args.surrogate_names:
        robust_model_source = False
        if surrogate_model =='vgg19':
            pretrained_model = vgg19_bn(pretrained=True)
        elif surrogate_model =='vgg16':
            pretrained_model = vgg16_bn(pretrained=True)
        elif surrogate_model =='vgg13':
            pretrained_model = vgg13_bn(pretrained=True)
        elif surrogate_model =='vgg11':
            pretrained_model = vgg11_bn(pretrained=True)
        elif surrogate_model =='resnet18':
            pretrained_model = resnet18(pretrained=True)
        elif surrogate_model =='resnet34':
            pretrained_model = resnet34(pretrained=True)
        elif surrogate_model =='resnet50':
            pretrained_model = resnet50(pretrained=True)
        elif surrogate_model =='densenet121':
            pretrained_model = densenet121(pretrained=True)
        elif surrogate_model =='densenet161':
            pretrained_model = densenet161(pretrained=True)
        elif surrogate_model =='densenet169':
            pretrained_model = densenet169(pretrained=True)
        elif surrogate_model =='googlenet':
            pretrained_model = googlenet(pretrained=True)
        elif surrogate_model =='mobilenet':
            pretrained_model = mobilenet_v2(pretrained=True)
        elif surrogate_model =='inception':
            pretrained_model = inception_v3(pretrained=True)
        else:
            robust_model_source = True

        if robust_model_source:
            pretrained_model = load_model(surrogate_model, dataset='cifar10', threat_model='Linf')
        else:
            pretrained_model = nn.Sequential(
                Normalize(mean, std),
                pretrained_model
            )
            pretrained_model.eval()
    
        pretrained_model.to(device)
        nets.append(pretrained_model)


    model = nn.Sequential(
        Cifar10_Encoder(),
        Cifar10_Decoder()
    )
    model.to(device)


    optimizer_G = torch.optim.SGD(model.parameters(), state['learning_rate_G'], momentum=state['momentum'],
                                weight_decay=0, nesterov=True)
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=state['epochs'] // state['schedule'],
                                                gamma=state['gamma'])

    hingeloss = MarginLoss(margin=state['margin'], target=state['target'])

    if state['target']:
        save_name = "Cifar10_{}{}_target_{}.pytorch".format("_".join(args.surrogate_names), state['save_suffix'], state['target_class'])
    else:
        save_name = "Cifar10_{}{}_untarget.pytorch".format("_".join(args.surrogate_names), state['save_suffix'])

    def train():
        model.train()

        for batch_idx, (data, label) in enumerate(train_loader):
            nat = data.to(device)
            if state['target']:
                label = state['target_class']
            else:
                label = label.to(device)

            losses_g = []
            optimizer_G.zero_grad()
            for net in nets:
                noise = model(nat)
                adv = torch.clamp(noise * state['epsilon'] + nat, 0, 1)
                logits_adv = net(adv)
                loss_g = hingeloss(logits_adv, label)
                losses_g.append("%4.2f" % loss_g.item())
                loss_g.backward()
            optimizer_G.step()

            if (batch_idx + 1) % state['log_interval'] == 0:
                print("batch {}, losses_g {}".format(batch_idx + 1, dict(zip(args.surrogate_names, losses_g))))

    def test():
        model.eval()
        loss_avg = [0.0 for i in range(len(nets))]
        success = [0 for i in range(len(nets))]

        for batch_idx, (data, label) in enumerate(test_loader):
            nat = data.to(device)
            if state['target']:
                label = state['target_class']
            else:
                label = label.to(device)
            noise = model(nat)
            adv = torch.clamp(noise * state['epsilon'] + nat, 0, 1)
            
            for j in range(len(nets)):
                logits = nets[j](adv)
                loss = hingeloss(logits, label)
                loss_avg[j] += loss.item()
                if state['target']:
                    success[j] += int((torch.argmax(logits, dim=1) == label).sum())
                else:
                    success[j] += int((torch.argmax(logits, dim=1) == label).sum())

        state['test_loss'] = [loss_avg[i] / len(test_loader) for i in range(len(loss_avg))]
        state['test_successes'] = [success[i] / len(test_loader.dataset) for i in range(len(success))]
        state['test_success'] = 0.0
        for i in range(len(state['test_successes'])):
            state['test_success'] += state['test_successes'][i]/len(state['test_successes'])


    for epoch in range(state['epochs']):
        scheduler_G.step()
        state['epoch'] = epoch
        train()
        torch.cuda.empty_cache()
        with torch.no_grad():
            test()
        if not os.path.exists("G_weight"):
            os.mkdir("G_weight")
        torch.save(model.state_dict(), os.path.join("G_weight", save_name))
        print("epoch {}, Current success: {}".format(epoch, state['test_success']))