import os
import argparse
from utils import save_gradient
from Skip_gradient_method.utils_sgm import register_hook_for_resnet, register_hook_for_densenet
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import robustbench

parser = argparse.ArgumentParser(description='PyTorch SGM Attack Evaluation')
parser.add_argument('--model', type=str, default='Gowal2021Improving_28_10_ddpm_100m')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='input batch size for gradient computation')
parser.add_argument("--dataset", choices=["mnist", "cifar10", "imagenet"], default="cifar10")
parser.add_argument('--n_examples', type=int, default=0)

arguments = parser.parse_args()


def run(args, device="cuda", path="./MetaAttack_ICLR2020/checkpoints/gradients/"):
    target_model = robustbench.utils.load_model(model_name=args.model, dataset=args.dataset, threat_model='Linf')
    target_model.to(device)

    train_dataset = datasets.CIFAR10(root='./data',
                                     train=True,
                                     transform=transforms.Compose([transforms.ToTensor()]),
                                     download=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              pin_memory=True)

    test_dataset = datasets.CIFAR10(root='./data',
                                    train=False,
                                    transform=transforms.Compose([transforms.ToTensor()]),
                                    download=True)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             pin_memory=True)

    n_batches = args.n_examples // args.batch_size
    save_gradient(target_model, device, test_loader, args.model, path, mode='test', n_batches=n_batches)
    save_gradient(target_model, device, train_loader, args.model, path, mode='train',n_batches=n_batches)

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    run(arguments,device)