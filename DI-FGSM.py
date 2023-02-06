import torch
import torch.nn as nn
from cifar10_models.resnet import resnet50
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

    if config['source'] == 'Resnet50':
        PATH = 'model_64_100'
        source_model = resnet50(pretrained=False)
        
    else:
        raise ValueError('Source model is not recognizable')

    source_model.load_state_dict(torch.load(PATH,map_location=device))
    source_model.eval()


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

    print('Running DI-FGSM attack')
    attack = torchattacks.DIFGSM(source_model, eps=8/255, alpha=2/255, steps=10, decay=0.0, resize_rate=0.9, diversity_prob=0.5, random_start=False)
    adv_images_DI = attack(x_test_correct, y_test_correct)

    DI_results = dict()
    for i,target_model in enumerate(target_models):
        acc = clean_accuracy(target_model, adv_images_DI, y_test_correct)
        print('Robust Acc {}: %2.2f %%'%(acc*100))
        DI_results[rank_target_models[i]] = acc

    if not os.path.exists('results'):
        os.makedirs('results') 
    
    file_path = 'results/results_{}_{}.json'.format(config['source'],x_test_correct.size(0))

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