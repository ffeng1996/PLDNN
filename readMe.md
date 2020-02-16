## Requirements
Recommend install in virtual environment
```
$ conda create -n yourenvname python=2.7 anaconda
```

Activate the enviorment
```
conda activate yourenvname
```
Install the required the packages inside the virtual environment
```
sh installation.sh
```
## Run the models

For MLP on MNIST in section **3.1**

```
bash expt1/fc/run.sh
```

For CNN on MNIST experiment in section **3.1**

```
bash expt1/cnn/runMNIST.sh
```
For CNN on CIFAR-10 experiment in section **3.1**

```
bash expt1/cnn/runMNIST.sh
```

For AlexNet on ImageNet experiment in section **3.1**

```
bash expt1/cnn_large/runAlexNet.sh
```

For VGG-16 on ImageNet experiment in section **3.1**

```
bash expt1/cnn/runVGG.sh
```

For ResNet-16 on ImageNet experiment in section **3.1**

```
bash expt1/cnn/runResNet.sh
```

For experiments in section **4.2**

```
bash expt2/fc/run.sh
```

For experiments in section **5.1**

```
bash expt3/fc/run_pathnet.sh
```
For mlp on mnist in section **5.2**

```
bash expt3/fc/run_pnn.sh
```
For cnn on cifar in section **5.2**
```
bash expt3/cnn/run_cifar_pnn.sh
```

## Dataset

- MNIST is inside this folder
- CIFAR dataset need to be prepared using pylearn2 following https://github.com/MatthieuCourbariaux/BinaryConnect

## For CNN in expt3:
- First download CIFAR dataset and put the data in expt3/cnn/cifar10(100) 
- Run the scripts to build up pickle input format (need minor changes on the path):
```
cd pylearn2/pylearn2/datasets
python cifar10.py
python cifar100.py
cd power-law/pylearn2/pylearn2/scripts/datasets
python make_cifar10_gcn_whitened.py
python make_cifar100_gcn_whitened.py
```

## Other notes
- Due to github file size limitation, models in VGG and AlexNet on ImageNet in section 3.2 cannot upload in the repo. You can download on this [Google drive link](https://drive.google.com/drive/folders/1ceJs87P4g5VGdDyA8fIZWaN8-_MRtiCC?usp=sharing). Please place the three files under:
```
expt1/cnn_large
```
- For GPU support during training, please refer to this [tutorial](https://lasagne.readthedocs.io/en/latest/user/installation.html#cuda).
- Please make sure the configuration of environmental variables has been done under root path.  