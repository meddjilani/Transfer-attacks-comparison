# New attack cifar10

Our new attack on cifar10 transfers adversarial examples from an ensemble of surrogate models to a black-box victim model.  
We utilize the feedback of the victim model to adjust the weights attributed to each model in the ensemble.  
We diversify the surrogate models by applying the property of ghost networks with keeping the same search space dimension.  

Scripts are to generate using cifar10 images.  

Download pretrained models on cifar-10 from : https://github.com/huyvnphan/PyTorch_CIFAR10.git, then move the state_dicts folder to `cifar10_models/`
