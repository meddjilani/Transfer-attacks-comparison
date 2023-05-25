import torch
from meta_attack_cifar.learner import Learner
from models import *
from robustbench.utils import load_model
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
MODELS= ['VGG("VGG13")','VGG("VGG16")','VGG("VGG19")','VGG("VGG11")', 'ResNet18()', 'MobileNetV2()', 'SENet18()', 'GoogLeNet()', 'PreActResNet18()', 'MobileNet()']
def load_attacked_model(args, index, device):

    net = load_model(model_name=args.target, dataset=args.dataset, threat_model='Linf') #print('Loading chec.....)
    return net.to(device)

class Meta(torch.nn.Module):
    def __init__(self):
        super(Meta, self).__init__() 
        self.net = Learner(config, 3, 32)
    def forward(self,x):
        return x

def load_meta_model(args, meta_model_path, device):
    # meta_model = Learner(config, 3, 32)
    meta_model = Meta()
    pretrained_dict = torch.load(meta_model_path)
#       pretrained_dict = {k[4:]: v for k, v in pretrained_dict.items()}
    meta_model.load_state_dict(pretrained_dict)
    meta_model.net.eval()
    return meta_model.net.to(device)