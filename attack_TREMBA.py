import sys
sys.path.append('/')

from comet_ml import Experiment
import argparse
import torchvision.models as models
import torch
import torchvision
import torchvision.transforms as transforms
import os
import json
from robustbench.utils import load_model, clean_accuracy
from utils import *
from TREMBA.FCN import *
from app_config import COMET_APIKEY, COMET_WORKSPACE, COMET_PROJECT
from Normalize import Normalize
from cifar10_models.vgg import vgg16_bn,vgg11_bn,vgg19_bn, vgg13_bn
from cifar10_models.resnet import resnet18,resnet34,resnet50
from cifar10_models.inception import inception_v3
from cifar10_models.googlenet import googlenet
from cifar10_models.densenet import densenet161, densenet121, densenet169
from cifar10_models.mobilenetv2 import mobilenet_v2


def EmbedBA(function, encoder, decoder, image, label, config, latent=None, query_limit = 100):
    device = image.device

    if latent is None:
        latent = encoder(image.unsqueeze(0)).squeeze().view(-1)
    momentum = torch.zeros_like(latent)
    dimension = len(latent)
    noise = torch.empty((dimension, config['sample_size']), device=device)
    origin_image = image.clone()
    last_loss = []
    lr = config['lr']
    for iter in range(config['num_iters']+1):
        perturbation = torch.clamp(decoder(latent.unsqueeze(0)).squeeze(0)*config['epsilon'], -config['epsilon'], config['epsilon'])
        logit, loss = function(torch.clamp(image+perturbation, 0, 1), label)
        if config['target']:
            success = torch.argmax(logit, dim=1) == label
        else:
            success = torch.argmax(logit, dim=1) !=label
        last_loss.append(loss.item())

        if function.current_counts >= query_limit:
            break
        
        if bool(success.item()):
            return True, torch.clamp(image+perturbation, 0, 1)

        nn.init.normal_(noise)
        noise[:, config['sample_size']//2:] = -noise[:, :config['sample_size']//2]
        latents = latent.repeat(config['sample_size'], 1) + noise.transpose(0, 1)*config['sigma']
        perturbations = torch.clamp(decoder(latents)*config['epsilon'], -config['epsilon'], config['epsilon'])
        _, losses = function(torch.clamp(image.expand_as(perturbations) + perturbations, 0, 1), label)

        grad = torch.mean(losses.expand_as(noise) * noise, dim=1)

        if iter % config['log_interval'] == 0 and config['print_log']:
            print("iteration: {} loss: {}, l2_deviation {}".format(iter, float(loss.item()), float(torch.norm(perturbation))))

        momentum = config['momentum'] * momentum + (1-config['momentum'])*grad

        latent = latent - lr * momentum

        last_loss = last_loss[-config['plateau_length']:]
        if (last_loss[-1] > last_loss[0]+config['plateau_overhead'] or last_loss[-1] > last_loss[0] and last_loss[-1]<0.6) and len(last_loss) == config['plateau_length']:
            if lr > config['lr_min']:
                lr = max(lr / config['lr_decay'], config['lr_min'])
            last_loss = []

    return False, origin_image


parser = argparse.ArgumentParser()
parser.add_argument('--n_im', type=int, default=100)
parser.add_argument('--config', default='config/attack.json', help='config file')
parser.add_argument('--device', default='cuda', help='Device for Attack')
parser.add_argument('--save_prefix', default=None, help='override save_prefix in config file')
parser.add_argument('--target', default='Carmon2019Unlabeled')
parser.add_argument('--query_limit', type=int, default=100)
parser.add_argument('--generator_name', default='Cifar10_densenet161_densenet169_vgg16train_untarget')

args = parser.parse_args()
query_limit = args.query_limit

with open(args.config) as config_file:
    state = json.load(config_file)

if args.save_prefix is not None:
    state['save_prefix'] = args.save_prefix
state['model_name'] = args.target

new_state = state.copy()
new_state['batch_size'] = 1
new_state['test_bs'] = 1

experiment = Experiment(
    api_key=COMET_APIKEY,
    project_name=COMET_PROJECT,
    workspace=COMET_WORKSPACE,
)
parameters = {'attack': 'TREMBA attack', **vars(args), **state}
experiment.log_parameters(parameters)


device = torch.device(args.device if torch.cuda.is_available() else "cpu")

weight = torch.load(os.path.join("TREMBA/G_weight", args.generator_name + ".pytorch"), map_location=device)

encoder_weight = {}
decoder_weight = {}
for key, val in weight.items():
    if key.startswith('0.'):
        encoder_weight[key[2:]] = val
    elif key.startswith('1.'):
        decoder_weight[key[2:]] = val

nlabels = 10
transform = transforms.Compose([
    transforms.ToTensor(),
])
test_dataset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(torch.utils.data.Subset(test_dataset, range(args.n_im)), batch_size=new_state['batch_size'], shuffle=False, pin_memory=True)


mean = [0.4914, 0.4822, 0.4465]
std = [0.2471, 0.2435, 0.2616] 
robust_model_target = False
if args.target =='vgg19':
    model = vgg19_bn(pretrained=True)
elif args.target =='vgg16':
    model = vgg16_bn(pretrained=True)
elif args.target =='vgg13':
    model = vgg13_bn(pretrained=True)
elif args.target =='vgg11':
    model = vgg11_bn(pretrained=True)
elif args.target =='inception':
    model = inception_v3(pretrained=True)
elif args.target =='googlenet':
    model = googlenet(pretrained=True)
elif args.target =='mobilenet':
    model = mobilenet_v2(pretrained=True)
elif args.target =='resnet50':
    model = resnet50(pretrained=True)
elif args.target =='resnet34':
    model = resnet34(pretrained=True)
elif args.target =='resnet18':
    model = resnet18(pretrained=True)
elif args.target =='densenet169':
    model = densenet169(pretrained=True)
elif args.target =='densenet161':
    model = densenet161(pretrained=True)
elif args.target =='densenet121':
    model = densenet121(pretrained=True)
else:
    robust_model_target = True

if robust_model_target:
    model = load_model(args.target, dataset='cifar10', threat_model='Linf')
else:
    model = nn.Sequential(
        Normalize(mean, std),
        model
    )
    model.eval() 

model.to(device)

encoder = Cifar10_Encoder()
decoder = Cifar10_Decoder()
encoder.load_state_dict(encoder_weight)
decoder.load_state_dict(decoder_weight)

model.to(device)
encoder.to(device)
encoder.eval()
decoder.to(device)
decoder.eval()


F = Function(model, state['batch_size'], state['margin'], nlabels, state['target'])

count_success = 0
count_total = 0
# if not os.path.exists(state['save_path']):
#     os.mkdir(state['save_path'])

for i, (images, labels) in enumerate(dataloader):
    images = images.to(device)
    labels = int(labels)
    logits = model(images)
    correct = torch.argmax(logits, dim=1) == labels
    if correct:
        torch.cuda.empty_cache()
        if state['target']:
            labels = state['target_class']


        with torch.no_grad():
            success, adv = EmbedBA(F, encoder, decoder, images[0], labels, state, query_limit = query_limit)

        count_success += int(success)
        count_total += int(correct)
        print("image: {} eval_count: {} success: {} average_count: {} success_rate: {}".format(i, F.current_counts, success, F.get_average(), float(count_success) / float(count_total)))
        F.new_counter()

# if state['target']:
#     np.save(os.path.join(state['save_path'], '{}_class_{}.npy'.format(state['save_prefix'], state['target_class'])), np.array(F.counts))
# else:
#     np.save(os.path.join(state['save_path'], '{}.npy'.format(state['save_prefix'])), np.array(F.counts))

success_rate = float(count_success) / float(count_total)
print("success rate {}".format(success_rate))
print("average eval count {}".format(F.get_average()))
metrics = {'Queries': F.get_average(), 'success_rate': success_rate}
experiment.log_metrics(metrics)
