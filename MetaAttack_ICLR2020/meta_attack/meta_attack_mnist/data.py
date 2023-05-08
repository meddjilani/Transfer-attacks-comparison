import torch
from torchvision import datasets, transforms

class LeNormalize(object):
    '''
        normalize -1 to 1
    '''
    def __call__(self, tensor):
        for t in tensor:
            t.sub_(0.5)
#           t.sub_(0.5).mul_(2.0)
        return tensor

def mnist(args):
    if isinstance(args, dict):
        use_cuda = not args['no_cuda'] and torch.cuda.is_available()
        batch_size = args['batch_size']
        test_batch_size = args['test_batch_size']
    else:
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        batch_size = args.batch_size
        test_batch_size = args.test_batch_size

    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', 
                           train = True, 
                           download = True,
                           transform = transforms.Compose([
                                       transforms.ToTensor(),
#                                      LeNormalize(),
                    #                  transforms.Normalize((0.1307,), (0.3081, ))
                            ])),
                           batch_size = batch_size, 
                           shuffle = True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', 
                           train = False, 
                           download = True,
                           transform = transforms.Compose([
                                       transforms.ToTensor(),
#                                      LeNormalize(),
                    #                  transforms.Normalize((0.1307,), (0.3081, ))
                            ])),
                           batch_size = test_batch_size, 
                           shuffle = False, **kwargs)

    return train_loader, test_loader

def load_data(args):
    if args.dataset == 'mnist':
        train_loader, test_loader = mnist(args)
    else:
        pass
    
    return train_loader, test_loader
    
