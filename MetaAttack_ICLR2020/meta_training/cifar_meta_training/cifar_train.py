#python scripts
__author__='Du Jiawei NUS/IHPC'
__email__='dujw@ihpc.a-star.edu'
#Descrption:
import torch, os
import os.path as osp
import numpy as np
from cifar import cifar
from torch.utils.data import DataLoader
import argparse
from meta import Meta

torch.manual_seed(222)
torch.cuda.manual_seed_all(222)
np.random.seed(222)


MODELS='Standard Andriushchenko2020Understanding Carmon2019Unlabeled Gowal2021Improving_28_10_ddpm_100m Chen2020Adversarial Engstrom2019Robustness Wong2020Fast Ding2020MMA Gowal2021Improving_70_16_ddpm_100m Rebuffi2021Fixing_28_10_cutmix_ddpm Rebuffi2021Fixing_70_16_cutmix_extra'.split(' ')
ROBUST_PERF = [0, 43.93, 59.53, 63.38, 51.56, 49.25,43.21, 41.44, 66.10, 60.73, 66.58 ]
MODELS_SORT = sorted(range(len(ROBUST_PERF)), key=ROBUST_PERF.__getitem__)

def get_target_model(model_name, device="cuda"):
    net = load_model(model_name=model_name, dataset=args.dataset, threat_model='Linf', model_dir="../models")
    net = net.to(device)
    return net

def swapaxis(*input):
    result = []
    for each in input:
        each = each.transpose(0,1)
        result.append(each)
    return result

def save_model(model, acc,target):
    model_file_path = './checkpoint/meta'
    if not os.path.exists(model_file_path):
        os.makedirs(model_file_path)

    file_name = str(acc) + 'cifar_' + MODELS[target] + '.pt'
    save_model_path = os.path.join(model_file_path, file_name)
    torch.save(model.state_dict(), save_model_path)

def load_model(model, acc,target):
    model_checkpoint_path = './checkpoint/meta/' + str(acc) + 'cifar_' + MODELS[target] + '.pt'
    assert os.path.exists(model_checkpoint_path)
    model.load_state_dict(torch.load(model_checkpoint_path))
    return model

def main(args):
    #print(args)
    TARGET_MODEL = args.model
    config = [
        ('conv2d', [32, 3, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [32]),
        ('conv2d', [64, 32, 4, 4, 2, 1]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [128, 64, 4, 4, 2, 1]),
        ('relu', [True]),
        ('bn', [128]),
        ('conv2d', [256, 128, 4, 4, 2, 1]),
        ('relu', [True]),
        ('bn', [256]),
        ('convt2d', [256, 128, 4, 4, 2, 1]),
        ('relu', [True]),
        ('bn', [128]),
        ('convt2d', [128, 64, 4, 4, 2, 1]),
        ('relu', [True]),
        ('bn', [64]),
        ('convt2d', [64, 32, 4, 4, 2, 1]),
        ('relu', [True]),
        ('bn', [32]),
        ('convt2d', [32, 3, 3, 3, 1, 1]),
   ]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    maml = Meta(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())

    if args.meta_models =="top":
        models = MODELS[MODELS_SORT[-args.task_num:]]

    elif args.meta_models =="bottom":
        models = MODELS[MODELS_SORT[:args.task_num]]
        
    elif args.meta_models =="medium":
        nb_buckets = len(MODELS_SORT)//args.task_num
        models = MODELS[(nb_buckets//2)*args.task_num:(nb_buckets//2)*(args.task_num+1)]
    else:
        models = MODELS[args.meta_models.split(";")]
    # initiate different datasets 
    minis = []
    for i in range(args.task_num):
        path = osp.join("./zoo_cw_grad_cifar/train/", models[i] + "_cifar.npy")
        mini = cifar(path, 
                    mode='train', 
                    n_way=args.n_way, 
                    k_shot=args.k_spt, 
                    k_query=args.k_qry, 
                    batchsz=100, 
                    resize=args.imgsz)
        db = DataLoader(mini,args.batchsize, shuffle=True, num_workers=0, pin_memory=True)
        minis.append(db)

    path_test = osp.join("./zoo_cw_grad_cifar/test/", MODELS[TARGET_MODEL] + "_cifar.npy")
    mini_test = cifar(path_test, 
                    mode='test', 
                    n_way=1, 
                    k_shot=args.k_spt,
                    k_query=args.k_qry,
                    batchsz=100, 
                    resize=args.imgsz)

    mini_test = DataLoader(mini_test, 10, shuffle=True, num_workers=0, pin_memory=True)
    
    # start training
    step_number = len(minis[0])
    BEST_ACC = 1.5
    target_model = get_target_model(TARGET_MODEL, device)


    for epoch in range(args.epoch//100):
        minis_iter = []
        for i in range(len(minis)):
            minis_iter.append(iter(minis[i]))
        mini_test_iter = iter(mini_test)

        for step in range(step_number):
            batch_data = []
            for each in minis_iter:
                batch_data.append(each.next())

            accs = maml(batch_data, device)
            if (step + 1) % step_number  == 0:
                print('Training acc:', accs)
                if accs[0] < BEST_ACC:
                    BEST_ACC = accs[0]
                    save_model(maml, BEST_ACC, TARGET_MODEL)

            if (epoch  + 1) % 15 == 0 and step ==0:  # evaluation
                accs_all_test = []
                for i in range(3):
                    test_data = mini_test_iter.next()
                    accs = maml.finetunning(test_data,target_model,device)
                    accs_all_test.append(accs)

                accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                print('Test acc:', accs)

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=254600)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=10)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=10)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=32)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--batchsize', type=int, help='batchsize', default=64)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=3)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-2)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=1e-2)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=20)
    argparser.add_argument('--meta_optim_choose', help='reptile or maml', default="reptile")
    argparser.add_argument('--resume', action = 'store_true',help='load model or not', default=False)

    argparser.add_argument("--model", type=str, default='Carmon2019Unlabeled',
                        help='the model selected to be used for meta-model')
    argparser.add_argument("--dataset", choices=["mnist", "cifar10", "imagenet"], default="cifar10")
    argparser.add_argument("--meta_models", choices=["top", "medium", "bottom"], default="medium")

    args = argparser.parse_args()

    main(args)
