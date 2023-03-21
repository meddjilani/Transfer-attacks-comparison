import argparse
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from Normalize import Normalize
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from FCN import *
from utils import *
import torchvision.models as models
import copy
from cifar10_models.resnet import *
from cifar10_models.densenet import *
from cifar10_models.vgg import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, nargs="+", default=[0,1,2,3])
    parser.add_argument('--config', default='config/train_untarget.json', help='config file')

    args = parser.parse_args()
    
    with open(args.config) as config_file:
        state = json.load(config_file)

    with open('../config_ids_source_targets.json','r') as f:
        config_sources = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#

    nlabels, mean, std = 10, [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010] #
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.CIFAR10(root='./dataset/Cifar10', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./dataset/Cifar10', train=False, download=True, transform=transform_test)
    print("state: ",state)

    #sampler = SubsetRandomSampler(list(range(500)))
    train_loader = DataLoader(train_dataset, batch_size=state['batch_size'],shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=state['batch_size'], shuffle=False, num_workers=2)


    source_models = []
    for source in config_sources['sources']:
        if source == 'Resnet50': #use it
            source_model = resnet50(pretrained=True)
        elif source == 'Resnet34': 
            source_model = resnet34(pretrained=True)
        elif source  == 'Resnet18':
            source_model = resnet18(pretrained=True)
        elif source == 'Densenet169': 
            source_model = densenet169(pretrained=True)
        elif source == 'Densenet161': 
            source_model = densenet161(pretrained=True)
        elif source == 'Densenet121': 
            source_model = densenet121(pretrained=True)
        elif source == 'Vgg19':
            source_model = vgg19_bn(pretrained=True)
        elif source  == 'Resnet50_non_normalize': 
            source_model = resnet50(pretrained=False)
            PATH = 'resnet50_128_100'
            source_model.load_state_dict(torch.load(PATH,map_location=device))
        else :
            raise ValueError('Source model is not recognizable')
        
        source_model = nn.Sequential(
            Normalize(mean, std),
            source_model
        )
        source_models.append(source_model)


    model = nn.Sequential(
        Cifar10_Encoder(),
        Cifar10_Decoder()
    )

    for i in range(len(source_models)):
        #source_models[i] = torch.nn.DataParallel(source_models[i], args.device)
        source_models[i].eval()
        source_models[i].to(device)

    #model = torch.nn.DataParallel(model, args.device)
    model.to(device)

    optimizer_G = torch.optim.SGD(model.parameters(), state['learning_rate_G'], momentum=state['momentum'],
                                nesterov=True)
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=200)
    # scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=state['epochs'] // state['schedule'],
    #                                             gamma=state['gamma'])

    hingeloss = MarginLoss(margin=state['margin'], target=state['target'])

    if state['target']:
        save_name = "Cifar10{}{}_target_{}.pytorch".format("_".join(state['model_name']), state['save_suffix'], state['target_class'])
    else:
        save_name = "Cifar10{}{}_untarget.pytorch".format("_".join(state['model_name']), state['save_suffix'])

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
            for source_model in source_models:
                noise = model(nat)
                adv = torch.clamp(noise * state['epsilon'] + nat, 0, 1)
                logits_adv = source_model(adv)
                loss_g = hingeloss(logits_adv, label)
                losses_g.append("%4.2f" % loss_g.item())
                loss_g.backward()
            optimizer_G.step()
            if (batch_idx + 1) % state['log_interval'] == 0:
                print("batch {}, losses_g {}".format(batch_idx + 1, dict(zip(state['model_name'], losses_g))))

    def test():
        model.eval()
        loss_avg = [0.0 for i in range(len(source_models))]
        success = [0 for i in range(len(source_models))]

        for batch_idx, (data, label) in enumerate(test_loader):
            nat = data.to(device)
            if state['target']:
                label = state['target_class']
            else:
                label = label.to(device)
            noise = model(nat)
            adv = torch.clamp(noise * state['epsilon'] + nat, 0, 1)
            
            for j in range(len(source_models)):
                logits = source_models[j](adv)
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

    best_success = 1
    for epoch in range(state['epochs']):
        state['epoch'] = epoch
        train()
        torch.cuda.empty_cache()
        if epoch % 1 == 0:
            with torch.no_grad():
                test()
            if best_success > state['test_success']:
                best_success = state['test_success']
        torch.save(model.state_dict(), os.path.join("G_weight", save_name))
        print("epoch {}, Current success: {}, Best success: {}".format(epoch, state['test_success'], best_success))
        scheduler_G.step()