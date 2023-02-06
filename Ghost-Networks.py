import torch
import torch.nn as nn
from cifar10_models.ghost_resnet import resnet50
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
        ghost_source_model = resnet50(pretrained=False)
        
    else:
        raise ValueError('Source model is not recognizable')

    ghost_source_model.load_state_dict(torch.load(PATH,map_location=device))
    ghost_source_model.eval()


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

    print('Running Ghost networks attack')
    attack = torchattacks.MIFGSM(ghost_source_model, eps=8/255, steps=10, decay=1.0)
    adv_images_GN = attack(x_test_correct, y_test_correct)

    GN_results = dict()
    for i,target_model in enumerate(target_models):
        acc = clean_accuracy(target_model, adv_images_GN, y_test_correct)
        print('Robust Acc {}: %2.2f %%'%(acc*100))
        GN_results[rank_target_models[i]] = acc

    if not os.path.exists('results'):
        os.makedirs('results') 
    
    file_path = 'results/results_{}_{}.json'.format(config['source'],x_test_correct.size(0))

    if os.path.exists(file_path):
        with open(file_path,'r') as f:
            data = json.load(f)
    else:
        data = dict()

    data_GNMIFGSM = data.get('GN-MIFGSM',{})
    data_GNMIFGSM.update(GN_results)
    data['GN-MIFGSM'] = data_GNMIFGSM

    with open(file_path,'w') as f:
        json.dump(data, f, indent=4)