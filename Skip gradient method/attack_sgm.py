from comet_ml import Experiment
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from advertorch.attacks import LinfPGDAttack, MomentumIterativeAttack

#import pretrainedmodels
from utils_sgm import register_hook_for_resnet, register_hook_for_densenet
from utils_data import SubsetCifar10, save_images
import json 
import robustbench


import os
from robustbench.data import load_cifar10
from cifar10_models.resnet import resnet50



parser = argparse.ArgumentParser(description='PyTorch ImageNet Attack Evaluation')
parser.add_argument('--target', type=str, default= 'Standard')
parser.add_argument('--n_examples', type=int, default=10)
parser.add_argument('--input_dir', default='./cifar10_32', help='the path of original dataset') #medben
parser.add_argument('--output_dir', default='adv_images', help='the path of the saved dataset')
parser.add_argument('--arch', default='resnet50',
                    help='source model for black-box attack evaluation',)
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='input batch size for adversarial attack')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=8, type=float,
                    help='perturbation')
parser.add_argument('--num-steps', dest='num_steps', default=10, type=int,
                    help='perturb number of steps')
parser.add_argument('--step-size', dest='step_size', default=2, type=float,
                    help='perturb step size')
parser.add_argument('--gamma', default=0.5, type=float)
parser.add_argument('--momentum', default=0.0, type=float)
parser.add_argument('--print_freq', default=10, type=int)


args = parser.parse_args()

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}


# simple Module to normalize an image
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, x):
        return (x - self.mean.type_as(x)[None,:,None,None]) / self.std.type_as(x)[None,:,None,None]


def generate_adversarial_example(model, target_model, data_loader, adversary, img_path, label):
    """
    evaluate model by black-box attack
    """
    model.eval()

    for batch_idx, (inputs, true_class, idx) in enumerate(data_loader):
        inputs, true_class = \
            inputs.to(device), true_class.to(device)

        # attack
        inputs_adv = adversary.perturb(inputs, true_class)
        # save_images(inputs_adv.detach().cpu().numpy(), img_list=img_path,
        #             idx=idx, output_dir=args.output_dir)
        if batch_idx==0:
            adv_images= inputs_adv
        else:
            adv_images = torch.cat([adv_images,inputs_adv],dim=0)
        # assert False
        if batch_idx % args.print_freq == 0:
            print('generating: [{0}/{1}]'.format(batch_idx, len(data_loader)))
    
    rob_acc = robustbench.utils.clean_accuracy(target_model, adv_images, label)
    print(args.target, 'Robust Acc:', rob_acc)
    return rob_acc


def main():

    x_test, y_test = load_cifar10(n_examples=args.n_examples)
    x_test, y_test = x_test.to(device), y_test.to(device)

    class ImageDataset(torch.utils.data.Dataset):
        def __init__(self, images, labels):
            self.images = images
            self.labels = labels

        def __len__(self):
            return len(self.images)

        def __getitem__(self, index):
            return self.images[index], self.labels[index], index

    testset = ImageDataset(x_test, y_test)
    data_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size)

    target_model = robustbench.utils.load_model(model_name=args.target, dataset='cifar10', threat_model='Linf')
    target_model.to(device)
    acc = robustbench.utils.clean_accuracy(target_model,x_test ,y_test)
    print(args.target, 'Clean Acc:', acc)

    # create models
    net = resnet50(pretrained=True)
    model = nn.Sequential(Normalize(mean=[0.4914, 0.4822, 0.4465], std= [0.2471, 0.2435, 0.2616]), net) #medben
    model = model.to(device)
    model.eval()

    # create adversary attack
    epsilon = args.epsilon / 255.0
    if args.step_size < 0:
        step_size = epsilon / args.num_steps
    else:
        step_size = args.step_size / 255.0

    # if args.gamma < 1.0:
    #     print('using our method')
    #     register_hook(model, args.arch, args.gamma, is_conv=args.is_conv)

    # using our method - Skip Gradient Method (SGM)
    if args.gamma < 1.0:
        if args.arch in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
            register_hook_for_resnet(model, arch=args.arch, gamma=args.gamma)
        elif args.arch in ['densenet121', 'densenet169', 'densenet201']:
            register_hook_for_densenet(model, arch=args.arch, gamma=args.gamma)
        else:
            raise ValueError('Current code only supports resnet/densenet. '
                             'You can extend this code to other architectures.')

    if args.momentum > 0.0:
        print('using PGD attack with momentum = {}'.format(args.momentum))
        adversary = MomentumIterativeAttack(predict=model, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                                            eps=epsilon, nb_iter=args.num_steps, eps_iter=step_size,
                                            decay_factor=args.momentum,
                                            clip_min=0.0, clip_max=1.0, targeted=False)
    else:
        print('using linf PGD attack')
        adversary = LinfPGDAttack(predict=model, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                                  eps=epsilon, nb_iter=args.num_steps, eps_iter=step_size,
                                  rand_init=False, clip_min=0.0, clip_max=1.0, targeted=False)

    rob_acc = generate_adversarial_example(model=model, target_model=target_model, data_loader=data_loader,
                                 adversary=adversary, img_path='adv_images', label = y_test)


    experiment = Experiment(
    api_key="RDxdOE04VJ1EPRZK7oB9a32Gx",
    project_name="Black-box attack comparison cifar10",
    workspace="meddjilani",
    )

    metrics = {'Clean accuracy': acc, 'Robust accuracy': rob_acc}
    experiment.log_metrics(metrics, step=1)

    parameters = {'attack':'SGM', 'source':args.arch, 'target':args.target, 'n_examples':args.n_examples, 'epsilon':args.epsilon,
                'num-steps':args.num_steps, 'step-size':args.step_size, 'gamma':args.gamma, 'momentum':args.momentum}
    experiment.log_parameters(parameters)

if __name__ == '__main__':
    main()
