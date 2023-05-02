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

#add:
    # args parm of mi


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    #select the first n images correctly classified by all targeted models
    parser.add_argument('--model', type=str, default='Gowal2021Improving_28_10_ddpm_100m')
    parser.add_argument('--target', type=str, default= 'Carmon2019Unlabeled')
    parser.add_argument('--n_examples', type=int, default=10)
    parser.add_argument('--eps', type = float, default=8/255)
    parser.add_argument('--alpha', type=float,default=2/255)
    parser.add_argument('--decay', type=float,default= 1.0)
    parser.add_argument('--steps', type=int,default=10)

    args = parser.parse_args()
    with open('config_ids_source_targets.json','r') as f:
        config = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #mean, std  = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    source_model = load_model(model_name=args.model, dataset='cifar10', threat_model='Linf')
    source_model.to(device)

    # source = config['sources'][0] #select first element as source
    # if source == 'Resnet50': #use it
    #     source_model = resnet50(pretrained=True)
    # elif source  == 'Resnet34':
    #     source_model = resnet34(pretrained=True)
    # elif source == 'Resnet18': 
    #     source_model = resnet18(pretrained=True)
    # elif source == 'Densenet169': 
    #     source_model = densenet169(pretrained=True)
    # elif source == 'Densenet161': 
    #     source_model = densenet161(pretrained=True)
    # elif source == 'Densenet121': 
    #     source_model = densenet121(pretrained=True)
    # elif source == 'Vgg19':
    #     source_model = vgg19_bn(pretrained=True)
    # elif source == '1':
    #     source_model = load_model(model_name='Rebuffi2021Fixing_70_16_cutmix_extra', dataset='cifar10', threat_model='Linf')
    # elif source == '7':
    #     source_model = load_model(model_name='Gowal2021Improving_28_10_ddpm_100m', dataset='cifar10', threat_model='Linf')
    # elif source  == 'Resnet50_non_normalize': 
    #     source_model = resnet50(pretrained=False)
    #     PATH = 'resnet50_128_100'
    #     source_model.load_state_dict(torch.load(PATH,map_location=device))
    # else :
    #     raise ValueError('Source model is not recognizable')
    
    # if not(source.isnumeric()):
    #     print('add normalisation to source')
    #     source_model = nn.Sequential(
    #     Normalize(mean, std),
    #     source_model
    # )
    # else:
    #     print('source from robustbench')
    # source_model.to(device)
    # source_model.eval()

    target_model = load_model(model_name= args.target, dataset='cifar10', threat_model='Linf')
    target_model.to(device)

    # target_models = [] 
    # rank_target_models = []
    # for rank, model_id in config['targets'].items():
    #     print('Loading model ', rank)
    #     target_model = load_model(model_name=model_id, dataset='cifar10', threat_model='Linf')
    #     target_model.to(device)
    #     target_models.append(target_model)
    #     rank_target_models.append(rank)


    x_test, y_test = load_cifar10(n_examples=args.n_examples)
    x_test, y_test = x_test.to(device), y_test.to(device)

    # ids = config['ids']
    # x_test_correct , y_test_correct = x_test[ids,:,:,:] , y_test[ids]

    print('Running MI-FGSM attack')
    eps = 16/255
    attack = torchattacks.MIFGSM(source_model, eps=args.eps, steps=args.steps, decay=args.decay)
    adv_images_MI = attack(x_test, y_test)


    # MI_results = dict()
    acc = clean_accuracy(target_model, x_test, y_test) 
    rob_acc = clean_accuracy(target_model, adv_images_MI, y_test) 
    print(args.target, 'Clean Acc: %2.2f %%'%(acc*100))
    print(args.target, 'Robust Acc: %2.2f %%'%(rob_acc*100))

    # for i,target_model in enumerate(target_models):
    #     acc = clean_accuracy(target_model, adv_images_MI, y_test_correct)

    #     print(str(i)+'Robust Acc: %2.2f %%'%(acc*100))
    #     MI_results[rank_target_models[i]] = acc

    # if not os.path.exists('results'):
    #     os.makedirs('results') 
    
    # file_path = 'results/results_{}_{}_{}.json'.format(source,x_test_correct.size(0),eps)

    # if os.path.exists(file_path):
    #     with open(file_path,'r') as f:
    #         data = json.load(f)
    # else:
    #     data = dict()

    # data_MIFGSM = data.get('MI-FGSM',{})
    # data_MIFGSM.update(MI_results)
    # data['MI-FGSM'] = data_MIFGSM

    # with open(file_path,'w') as f:
    #     json.dump(data, f, indent=4)