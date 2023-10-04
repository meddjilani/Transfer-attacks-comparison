# New attack Imagenet

* Our new attack on Imagenet transfers adversarial examples from an ensemble of surrogate models to a black-box victim model.  
* We utilize the feedback of the victim model to adjust the weights attributed to each model in the ensemble.  
* We diversify the surrogate models by applying the property of ghost networks with keeping the same search space dimension.  

The evaluation of our proposed approach is done using pretrained models from torchvision
