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

#import warnings


if __name__ == '__main__':

    #warnings.simplefilter("ignore") 
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/train_untarget.json', help='config file')
    parser.add_argument('--surrogate_names',  nargs='+', default=['Standard','Ding2020MMA'], help='Surrogate models to use.')

    args = parser.parse_args()

    with open(args.config) as config_file:
        state = json.load(config_file)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=state['batch_size'], shuffle=True, pin_memory=True)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=state['batch_size'], shuffle=False, pin_memory=True) 

    nets = []
    for model_name in args.surrogate_names:
        pretrained_model = load_model(model_name, dataset='cifar10', threat_model='Linf')
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

    best_success = 0.0
    for epoch in range(state['epochs']):
        scheduler_G.step()
        state['epoch'] = epoch
        train()
        torch.cuda.empty_cache()
        if epoch % 10 == 0:
            with torch.no_grad():
                test()
            print(state)
        if not os.path.exists("G_weight"):
            os.mkdir("G_weight")
        torch.save(model.state_dict(), os.path.join("G_weight", save_name))
        print("epoch {}, Current success: {}, Best success: {}".format(epoch, state['test_success'], best_success))