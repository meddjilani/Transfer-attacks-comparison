from comet_ml import Experiment
import argparse
import torch
from gduap import gd_universal_adversarial_perturbation, get_data_loader, get_fooling_rate, get_baseline_fooling_rate, get_model
from analyze import get_tf_uap_fooling_rate
from robustbench.utils import load_model, clean_accuracy
from robustbench.data import load_cifar10
from torch.utils.data import TensorDataset, DataLoader
from app_config import COMET_APIKEY, COMET_WORKSPACE, COMET_PROJECT
import json


def validate_arguments(args):
    models = ['vgg16', 'vgg19', 'googlenet', 'resnet50', 'resnet152']

    if not (args.model in models):
        print('Argument Error: invalid network')
        exit(-1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Carmon2019Unlabeled')
    parser.add_argument('--prior_type', default='no_data',
                        help='Which kind of prior to use')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='The batch size to use for training and testing')
    parser.add_argument('--patience_interval', type=int, default=5,
                        help='The number of iterations to wait to verify convergence')
    parser.add_argument('--dataset_name', choices=["mnist", "cifar10", "imagenet"], default='cifar10',
                        help='The dataset to be used for final evaluation')
    parser.add_argument('--id',
                        help='An identification number (e.g. SLURM Job ID) that will prefix saved files')
    parser.add_argument('--baseline', action='store_true',
                        help='Obtain a fooling rate for a baseline random perturbation')
    parser.add_argument('--tf_uap', default=None,
                        help='Obtain a fooling rate for a input TensorFlow UAP')
    parser.add_argument('--n_examples', type=int, default=100)
    parser.add_argument('--max_iter', type=int, default=10000)
    parser.add_argument('--eps', type=int, default=8)


    args = parser.parse_args()
    #validate_arguments(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = get_model(args.model, device)
    model = load_model(model_name=args.model, dataset='cifar10', threat_model='Linf') #
    print(args.model)
    model.to(device)

    if args.baseline:
        print("Obtaining baseline fooling rate...")
        baseline_fooling_rate = get_baseline_fooling_rate(model, device, disable_tqdm=True)
        print(f"Baseline fooling rate for {args.model}: {baseline_fooling_rate}")
        return

    if args.tf_uap:
        print(f"Obtaining fooling rate for TensorFlow UAP called {args.tf_uap} using {args.model}...")
        tf_fooling_rate = get_tf_uap_fooling_rate(args.tf_uap, model, device)
        print(f"Fooling rate for {args.tf_uap}: {tf_fooling_rate}")
        return

    # create a universal adversarial perturbation
    uap = gd_universal_adversarial_perturbation(model, args.model, args.prior_type, args.batch_size, device, args.dataset_name, args.patience_interval, args.id, args.n_examples, args.max_iter, args.eps, disable_tqdm=True)

    # perform a final evaluation
    config = {}
    with open('config_ids_source_targets.json','r') as f:
        config = json.load(f)
        
    experiment = Experiment(
        api_key=COMET_APIKEY,
        project_name=COMET_PROJECT,
        workspace=COMET_WORKSPACE,
    )
    

    parameters = {'attack': 'DF UAP', **vars(args), **config}
    experiment.log_parameters(parameters)

    x_t, y_t = load_cifar10(n_examples=args.n_examples) if args.dataset_name=="cifar10" else None
    dataset = TensorDataset(torch.FloatTensor(x_t), torch.LongTensor(y_t))
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        pin_memory=True)


    final_fooling_rate = get_fooling_rate(model, uap, loader, device, experiment)
    print(f"Final fooling rate on {args.dataset_name}: {final_fooling_rate}")

if __name__ == '__main__':
    main()
