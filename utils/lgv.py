import os, copy
from robustbench.utils import load_model
from torchattacks import LGV, BIM
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torch

class LGVModel(torch.nn.Module):
    def __init__(self, device, base_model, lgv_models):
        super(LGVModel, self).__init__()
        self.device = device
        self.base_model = base_model
        self.lgv_models = lgv_models
        self.counter = 0

    def forward(self, input):
        self.last_model = self.lgv_models[self.counter%len(self.lgv_models)].to(self.device)
        self.counter+=1


        return self.last_model(input)



def load_model_lgv(model_name, device, dataset='cifar10', threat_model='Linf', base_path="../models", batch_size=8,
                   epochs=10, nb_models_epoch=4, lr=0.05):
    base_model = load_model(model_name, dataset=dataset, threat_model=threat_model, model_dir=base_path).to(device)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = CIFAR10(root='../data', train=True, download=True, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,shuffle=False)  # sampler=torch.utils.data.sampler.SubsetRandomSampler(range(args.n_im))

    atk = LGV(base_model, train_loader, lr=lr, epochs=epochs, nb_models_epoch=nb_models_epoch, wd=1e-4, n_grad=1, attack_class=BIM, eps=4/255, alpha=4/255/10, steps=50, verbose=True)

    models_path = os.path.join(base_path, "lgv",model_name, dataset, threat_model)
    if os.path.isdir(models_path):
        list_models = []
        files = [f for f in os.listdir(models_path) if os.path.isfile(os.path.join(models_path, f))]
        for f in files:
            model = copy.deepcopy(base_model)
            weights = torch.load(os.path.join(models_path, f))
            model.load_state_dict(weights.get('state_dict'))
            list_models.append(model)
        atk.list_models = list_models
    else:
        print("LGV models not found for {}. Collecting them!".format(models_path))
        atk.collect_models()
        atk.save_models(models_path)

    return LGVModel(device=device, base_model=base_model,lgv_models=atk.list_models)