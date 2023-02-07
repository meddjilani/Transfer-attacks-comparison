import argparse
import torch
import torch.nn as nn
from robustbench.utils import load_model
from robustbench.data import load_cifar10
from cifar10_models.resnet import *
from cifar10_models.densenet import *
from cifar10_models.vgg import *

import math
import json
from Normalize import Normalize

#add:
    # device for gpu
    #error: resnet50 pth

#future:
    #cross validation

# modified robustbench.utils
def correct_predict_ids(models,
                   x: torch.Tensor,
                   y: torch.Tensor,
                   n_images,
                   log_interval,
                   print_log,
                   batch_size: int = 100,
                   device: torch.device = None):
    if device is None:
        device = x.device
    acc = 0.
    ids = []
    n_batches = math.ceil(x.shape[0] / batch_size)
    with torch.no_grad():
        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) *
                       batch_size].to(device)
            y_curr = y[counter * batch_size:(counter + 1) *
                       batch_size].to(device)
            
            accs_models= torch.empty((len(models),batch_size), dtype=torch.float).to(device) #rmv
            for i,model in enumerate(models):
                output = model(x_curr)
                accs_models[i,:] = (output.max(1)[1] == y_curr).float()
            
            ids_batch = torch.nonzero(accs_models.bool().all(dim=0))
            ids.extend( (ids_batch + counter * batch_size).
                        view(-1).tolist()
            )
            if (counter+1) % log_interval == 0 and print_log:
                print('{} elements selected from batch {}, total of {}'.format(torch.numel(ids_batch),counter+1,len(ids)))

            if len(ids) >= n_images:
                break

    if len(ids) < n_images:
        print('Only ',len(ids),' correctly classified images by all target models')
    
    return ids[:n_images]
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #select the first n images correctly classified by all targeted models
    parser.add_argument('--n_images', type=int, default=100 , help = 'number of test images')
    parser.add_argument('--batch_size', type=int, default=20 , help = 'batch size')
    parser.add_argument('--sources', nargs = '+', default=['Resnet50','Resnet18'], help = 'source model')
    parser.add_argument('--targets', type=int,nargs = '+', default = [1, 3, 61], help = 'ranks of target models from cifar-10 linf, eps=2/288 robustbench leaderboad')
    parser.add_argument('--log_interval', default= 1, help = 'print each n batch')
    parser.add_argument('--print_log', default=True, help = 'print log in terminal')
    parser.add_argument('--output', default='config_ids_source_targets', help = 'file name contains the correctly classified images plus the source and target models')#output dir or filename

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mean, std  = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    source_models = []
    for source in args.sources:
        if source == 'Resnet50': 
            source_model = resnet50(pretrained=True)
        elif source == 'Resnet34': 
            source_model = resnet34(pretrained=True)
        elif source == 'Resnet18':
            source_model = resnet18(pretrained=True)
        elif source == 'Densenet169': 
            source_model = densenet169(pretrained=True)
        elif source == 'Densenet161': 
            source_model = densenet161(pretrained=True)
        elif source == 'Densenet121': 
            source_model = densenet121(pretrained=True)
        elif source == 'Vgg19':
            source_model = vgg19_bn(pretrained=True)
        elif source == 'Resnet50_non_normalize': 
            source_model = resnet50(pretrained=False)
            PATH = 'resnet50_128_100'
            source_model.load_state_dict(torch.load(PATH,map_location=device))
        else :
            raise ValueError('Source model is not recognizable')
        
        source_model = nn.Sequential(
            Normalize(mean, std),
            source_model
        )
        #todevice
        source_model.eval() 
        source_models.append(source_model)


    rank_to_id = {
        1 : 'Rebuffi2021Fixing_70_16_cutmix_extra',
        2 : 'Gowal2021Improving_70_16_ddpm_100m',
        3 : 'Gowal2020Uncovering_70_16_extra',
        4 : 'Rebuffi2021Fixing_106_16_cutmix_ddpm',
        5 : 'Rebuffi2021Fixing_70_16_cutmix_ddpm',
        61 : 'Standard',
    }

    target_models = [] 
    rank_to_id_selected = dict()
    for rank in args.targets:
        if rank in rank_to_id.keys():
            print('Loading model ', rank)
            target_model = load_model(model_name=rank_to_id[rank], dataset='cifar10', threat_model='Linf')
            #todevice
            target_models.append(target_model)
            rank_to_id_selected[rank] = rank_to_id[rank]
        else: 
            raise ValueError('target model with ranking = ',rank,' is not available')

    x_test, y_test = load_cifar10()
    
    ids = correct_predict_ids(target_models+source_models, x_test, y_test, args.n_images, args.log_interval, args.print_log, args.batch_size) # + source_models get correct  predict images from source models

    config = {
        'ids' : ids,
        'sources' : args.sources,
        'targets' : rank_to_id_selected,
    }


    with open(args.output + '.json','w') as f:
        json.dump(config, f, indent=4)