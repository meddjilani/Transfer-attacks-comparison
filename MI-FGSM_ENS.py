import torch
import torch.nn as nn
from cifar10_models.resnet import *
from cifar10_models.densenet import *
from cifar10_models.vgg import *
from robustbench.data import load_cifar10
from robustbench.utils import load_model, clean_accuracy
import torchattacks_ens.attacks.mifgsm_ens as taamifgsm_ens
import json
import os
from Normalize import Normalize

#add:
    # args parm of mi
if __name__ == '__main__':
    
    with open('config_ids_source_targets.json','r') as f:
        config = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mean, std  = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    source_models = []
    for source in config['sources']:
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
        #todevice
        source_model.eval()
        source_models.append(source_model)

    print('sources',config['sources'], len(source_models))
    target_models = [] 
    rank_target_models = []
    for rank, model_id in config['targets'].items():
        print('Loading model ', rank)
        target_model = load_model(model_name=model_id, dataset='cifar10', threat_model='Linf')
        #todevice
        target_models.append(target_model)
        rank_target_models.append(rank)



    x_test, y_test = load_cifar10()

    ids = config['ids']
    x_test_correct , y_test_correct = x_test[ids,:,:,:] , y_test[ids]

    print('Running MI-FGSM ENS attack')
    attack = taamifgsm_ens.MIFGSM_ENS(source_models[0], source_models, eps=8/255, steps=10, decay=1.0)
    adv_images_MI_ENS = attack(x_test_correct, y_test_correct)

    MI_ENS_results = dict()
    for i,target_model in enumerate(target_models):
        acc = clean_accuracy(target_model, adv_images_MI_ENS, y_test_correct)
        print('Robust Acc {}: %2.2f %%'%(acc*100))
        MI_ENS_results[rank_target_models[i]] = acc

    if not os.path.exists('results'):
        os.makedirs('results') 
    
    file_path = 'results/results_{}_{}.json'.format('_'.join(config['sources']),x_test_correct.size(0))

    if os.path.exists(file_path):
        with open(file_path,'r') as f:
            data = json.load(f)
    else:
        data = dict()

    data_MIFGSM_ENS = data.get('MI-FGSM ENS',{})
    data_MIFGSM_ENS.update(MI_ENS_results)
    data['MI-FGSM ENS'] = data_MIFGSM_ENS

    with open(file_path,'w') as f:
        json.dump(data, f, indent=4)