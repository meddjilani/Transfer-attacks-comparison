import torch
import torch.nn as nn
from cifar10_models.resnet import resnet50, resnet18
from robustbench.data import load_cifar10
from robustbench.utils import load_model, clean_accuracy
import torchattacks
import json
import os

#add:
    # args parm of di
if __name__ == '__main__':
    
    with open('config_ids_source_targets.json','r') as f:
        config = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    source = config['sources'][0] #select first element as source
    if source == 'Resnet50': #use it
        source_model = resnet50(pretrained=True)
    elif source  == 'Resnet18':
        source_model = resnet18(pretrained=True)
    elif source  == 'Resnet50_non_normalize': 
        source_model = resnet50(pretrained=False)
        PATH = 'resnet50_128_100'
        source_model.load_state_dict(torch.load(PATH,map_location=device))
    elif source == '1':
        source_model = load_model(model_name='Rebuffi2021Fixing_70_16_cutmix_extra', dataset='cifar10', threat_model='Linf')
    elif source == '7':
        source_model = load_model(model_name='Gowal2021Improving_28_10_ddpm_100m', dataset='cifar10', threat_model='Linf')
    elif source == '23':
        source_model = load_model(model_name='Carmon2019Unlabeled', dataset='cifar10', threat_model='Linf')
    elif source == '61':
        source_model = load_model(model_name='Standard', dataset='cifar10', threat_model='Linf') 
    else :
        raise ValueError('Source model is not recognizable')

    source_model.to(device)
    source_model.eval()


    target_models = [] 
    rank_target_models = []
    for rank, model_id in config['targets'].items():
        print('Loading model ', rank)
        target_model = load_model(model_name=model_id, dataset='cifar10', threat_model='Linf')
        target_model.to(device)
        target_models.append(target_model)
        rank_target_models.append(rank)



    x_test, y_test = load_cifar10()

    ids = config['ids']
    x_test_correct , y_test_correct = x_test[ids,:,:,:] , y_test[ids]

    print('Running DI-FGSM attack')
    eps = 16/255
    attack = torchattacks.DIFGSM(source_model, eps=eps, alpha=2/255, steps=10, decay=0.0, resize_rate=0.9, diversity_prob=0.5, random_start=False)
    adv_images_DI = attack(x_test_correct.to(device), y_test_correct.to(device))

    DI_results = dict()
    for i,target_model in enumerate(target_models):
        acc = clean_accuracy(target_model, adv_images_DI, y_test_correct)
        print('Robust Acc {}: %2.2f %%'%(acc*100))
        DI_results[rank_target_models[i]] = acc

    if not os.path.exists('results'):
        os.makedirs('results') 
    
    file_path = 'results/results_{}_{}_{}.json'.format(source,x_test_correct.size(0),eps)

    if os.path.exists(file_path):
        with open(file_path,'r') as f:
            data = json.load(f)
    else:
        data = dict()

    data_DIFGSM = data.get('DI-FGSM',{})
    data_DIFGSM.update(DI_results)
    data['DI-FGSM'] = data_DIFGSM

    with open(file_path,'w') as f:
        json.dump(data, f, indent=4)