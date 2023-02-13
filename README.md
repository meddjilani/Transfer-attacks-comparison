# Transfer-attacks-comparison
Comparing State of the art black-box attacks methods such as MI, DI, Admix Ghost networks by attacking some robust models trained on cifar10 that are available in the robustbench leaderboad.

## Reproducing the results
- Run the following command to get correctly classified images by all sources and targets models
```bash
python data.py --sources Resnet50 --targets 1 2 3 61 --n_images 100
```
 by doing that, you discard images that were already misclassified by the targets model. The ids of images will be stored in config file



- Run this command to evaluate the attack (MI-FGSM.py, DI-FGSM.py, adm.py)  
The ids of correctly classified test images will be loaded from config file generated by data.py
```bash
python MI-FGSM.py
```

## Results

the following results are obtained by using the default parameters of each attack. no hyper parameters tuning have been made
### Source: Resnet50
### num_images: 100 

MI-FGSM  |  TI-FGSM  |  Admix  |  VMI-FGSM  |  VNI-FGSM  |  GN-MIFGSM  |  DI-FGSM
---------|-----------|---------|------------|------------|-------------|---------
1.0      |  1.0      |  1.0    |  1.0       |  1.0       |  0.99       |  1.0
1.0      |  1.0      |  1.0    |  1.0       |  1.0       |  1.0        |  1.0
1.0      |  1.0      |  1.0    |  1.0       |  1.0       |  1.0        |  1.0
1.0      |  1.0      |  0.99   |  1.0       |  1.0       |  0.99       |  1.0
1.0      |  0.98     |  0.99   |  1.0       |  1.0       |  0.99       |  1.0
0.43     |  0.91     |  0.25   |  0.3       |  0.33      |  0.72       |  0.83


### MI FGSM ensemble
#### Source: Resnet50, Densenet169, Vgg19
#### num_images: 100

MI-FGSM ENS |
----------- |
1.0         |
1.0         |   
1.0         |
1.0         |
1.0         |
0.22        |


#### Source: Resnet50, Densenet169, Vgg19
#### num_images: 100

MI-FGSM ENS
-----------
1.0
1.0
1.0
1.0
1.0
0.24


### Dast
#### Source: vgg
#### num_images: 100

| <sub>Model ID</sub> | <sub> Dast Robust accuracy</sub> |
|:---:|:---:|
| <sub>**Carmon2019Unlabeled**</sub> | <sub>99%</sub> |
| <sub>**Standard**</sub> | <sub>56%</sub> |




### TREMBA 

#### Source: resnet50
#### num_images: 100

| <sub>Model ID</sub> | <sub> Tremba Robust accuracy</sub> |
|:---:|:---:|
| <sub>**Carmon2019Unlabeled**</sub> | <sub>100%</sub> |
| <sub>**Standard**</sub> | <sub>64%</sub> |

