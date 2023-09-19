# New attack cifar10

New attacks transfers adversarial examples from an ensemble of surrogate models to a black-box victim model.<br />
We utilize the feedback from the victim model to adjust the weights attributed to each model in the ensemble.<br />
We diversify the surrogate models by applying the property of ghost networks with keeping the same search space dimension.<br />

Scripts are to generate using cifar10 images. <br />

Download pretrained models on cifar-10 from : https://github.com/huyvnphan/PyTorch_CIFAR10.git, then move the state_dicts folder to `cifar10_models`
