import argparse
import torchvision.models as models
import os
import json
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import robustbench
from utils import *
from FCN import *
from Normalize import Normalize, Permute
import json



def EmbedBA(function, encoder, decoder, image, label, config, latent=None):
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
        if function.current_counts > 50000:
            break
        
        if bool(success.item()):
            return True, torch.clamp(image+perturbation, 0, 1)

        nn.init.normal_(noise)
        noise[:, config['sample_size']//2:] = -noise[:, :config['sample_size']//2]
        latents = latent.repeat(config['sample_size'], 1) + noise.transpose(0, 1)*config['sigma']
        perturbations = torch.clamp(decoder(latents)*config['epsilon'], -config['epsilon'], config['epsilon'])
        _, losses = function(torch.clamp(image.expand_as(perturbations) + perturbations, 0, 1), label)

        grad = torch.mean(losses.expand_as(noise) * noise, dim=1)

        if (iter+1) % config['log_interval'] == 0 and config['print_log']:
            print("iteration: {} loss: {}, l2_deviation {}".format(iter+1, float(loss.item()), float(torch.norm(perturbation))))

        momentum = config['momentum'] * momentum + (1-config['momentum'])*grad

        latent = latent - lr * momentum

        last_loss = last_loss[-config['plateau_length']:]
        if (last_loss[-1] > last_loss[0]+config['plateau_overhead'] or last_loss[-1] > last_loss[0] and last_loss[-1]<0.6) and len(last_loss) == config['plateau_length']:
            if lr > config['lr_min']:
                lr = max(lr / config['lr_decay'], config['lr_min'])
            last_loss = []

    return False, origin_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json', help='config file')
    parser.add_argument('--device', default='cuda:0', help='Device for Attack')
    parser.add_argument('--save_prefix', default=None, help='override save_prefix in config file')
    parser.add_argument('--model_name', default=None)
    args = parser.parse_args()

    with open(args.config) as config_file:
        state = json.load(config_file)

    with open('config_ids_source_targets.json','r') as f:
        config_ids_targets = json.load(f)

    if args.save_prefix is not None:
        state['save_prefix'] = args.save_prefix
    if args.model_name is not None:
        state['model_name'] = args.model_name

    new_state = state.copy()
    new_state['batch_size'] = 1
    new_state['test_bs'] = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weight = torch.load(os.path.join("G_weight", state['generator_name']+".pytorch"), map_location=device)

    encoder_weight = {}
    decoder_weight = {}
    for key, val in weight.items():
        if key.startswith('0.'):
            encoder_weight[key[2:]] = val
        elif key.startswith('1.'):
            decoder_weight[key[2:]] = val

    #\
    nlabels, mean, std = 10, [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = datasets.CIFAR10(root='./dataset/Cifar10', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./dataset/Cifar10', train=False, download=True, transform=transform_test)
    print("state: ",state)
    #ids to select correctly classified images by sources
    ids = config_ids_targets['ids']
    x,y = robustbench.data.load_cifar10()
    x,y = x[ids,:,:,:], y[ids]

    class ImageDataset(torch.utils.data.Dataset):
        def __init__(self, images, labels):
            self.images = images
            self.labels = labels

        def __len__(self):
            return len(self.images)

        def __getitem__(self, index):
            return self.images[index], self.labels[index]

    testset = ImageDataset(x, y)
    dataloader = torch.utils.data.DataLoader(testset, batch_size=1, num_workers=2)

    #\


    if state['model_name'] == 'Resnet34':
        target = models.resnet34(pretrained=True)
    elif state['model_name'] == 'Carmon2019Unlabeled': #
        target_model = robustbench.utils.load_model(model_name='Carmon2019Unlabeled', dataset='cifar10', threat_model='Linf')
    elif state['model_name'] == 'Standard': #
        print('selecting standard')
        target_model = robustbench.utils.load_model(model_name='Standard', dataset='cifar10', threat_model='Linf')


    encoder = Cifar10_Encoder()
    decoder = Cifar10_Decoder()
    encoder.load_state_dict(encoder_weight)
    decoder.load_state_dict(decoder_weight)

    target_model.to(device)
    target_model.eval()
    encoder.to(device)
    encoder.eval()
    decoder.to(device)
    decoder.eval()

    F = Function(target_model, state['batch_size'], state['margin'], nlabels, state['target'])

    count_success = 0
    count_total = 0
    if not os.path.exists(state['save_path']):
        os.mkdir(state['save_path'])

    for i, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = int(labels)
        logits = target_model(images)
        correct = torch.argmax(logits, dim=1) == labels
        print(images.shape,labels) #medben
        if correct:
            torch.cuda.empty_cache()
            if state['target']:
                labels = state['target_class']
                with torch.no_grad():
                    success, adv = EmbedBA(F, encoder, decoder, images[0], labels, state, latents.view(-1))

            else:
                with torch.no_grad():
                    success, adv = EmbedBA(F, encoder, decoder, images[0], labels, state)

            count_success += int(success)
            count_total += int(correct)
            print("image: {} eval_count: {} success: {} average_count: {} success_rate: {}".format(i, F.current_counts, success, F.get_average(), float(count_success) / float(count_total)))
            F.new_counter()

    success_rate = float(count_success) / float(count_total)
    if state['target']:
        np.save(os.path.join(state['save_path'], '{}_class_{}.npy'.format(state['save_prefix'], state['target_class'])), np.array(F.counts))
    else:
        np.save(os.path.join(state['save_path'], '{}.npy'.format(state['save_prefix'])), np.array(F.counts))
    print("success rate {}".format(success_rate))
    print("average eval count {}".format(F.get_average()))
    TREMBA_results = {61: 1-success_rate}


    if not os.path.exists('../results'):
        os.makedirs('../results') 
    
    file_path = 'results/results_{}_{}.json'.format(source,x_test_correct.size(0))

    if os.path.exists(file_path):
        with open(file_path,'r') as f:
            data = json.load(f)
    else:
        data = dict()

    data_TREMBA = data.get('TREMBA',{})
    data_TREMBA.update(TREMBA_results)
    data['TREMBA'] = data_TREMBA

    with open(file_path,'w') as f:
        json.dump(data, f, indent=4)