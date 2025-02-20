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

## Configuration
- CIFAR dataset need to be prepared using pylearn2 following https://github.com/MatthieuCourbariaux/BinaryConnect
- Due to ICML submission file size limitation, models in VGG-16, AlexNet and ResNet-50 on ImageNet in section 3.2 cannot upload in the repo. You can download on: [VGG-16](https://onedrive.live.com/?cid=761731c648d29f43&id=761731C648D29F43%21111&authkey=!AEWctt6d3NFNz3g), [AlexNet](http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel) and [ResNet-50](https://onedrive.live.com/?cid=761731c648d29f43&id=761731C648D29F43%21113&authkey=!ACrUOTES_OAP8f8). Please place the three files under:
```
expt1/cnn_large
```
- For GPU support during training, please refer to this [tutorial](https://lasagne.readthedocs.io/en/latest/user/installation.html#cuda).
- Please make sure the configuration of environmental variables has been done under root path.  
- For CNN in ./expt3 first download CIFAR dataset and put the data in expt3/cnn/cifar10(100) and run the scripts to build up pickle input format (need minor changes on the path):
```
cd pylearn2/pylearn2/datasets
python cifar10.py
python cifar100.py
cd power-law/pylearn2/pylearn2/scripts/datasets
python make_cifar10_gcn_whitened.py
python make_cifar100_gcn_whitened.py
```
## Run the models

For MLP on MNIST in Section **3.1**

```
bash expt1/fc/run.sh
```

For CNN on MNIST experiment in Section **3.1**

```
bash expt1/cnn/runMNIST.sh
```
For CNN on CIFAR-10 experiment in Section **3.1**

```
bash expt1/cnn/runCIFAR.sh
```

For AlexNet on ImageNet experiment in Section **3.1**

```
bash expt1/cnn_large/runAlexNet.sh
```

For VGG-16 on ImageNet experiment in Section **3.1**

```
bash expt1/cnn/runVGG.sh
```

For ResNet-16 on ImageNet experiment in Section **3.1**

```
bash expt1/cnn/runResNet.sh
```
For MLP (for CNN/VGG-16/AlexNet/ResNet-50, please change the corresponding scripts same as Section **3.1**)in section **3.2**, if you choose iterative pruning (Please change the directories of models):

```
python expt1/LTH/main.py --prune_type=lt --arch_type=fc2 --dataset=mnist --prune_percent=5 --prune_iterations=19
bash expt1/fc/run.sh
```
Or you use one-shot pruning (need to modify the prune fraction in bash file):
```
bash expt1/fc/run.sh
```

For experiments in section **3.3 & 4.2** (slightly change on dataset configuration, marked in code)

```
bash expt2/run.sh
```

For experiments in section **5.1 & 5.2**

```
bash expt3/fc/run_pathnet.sh
```

For mlp on mnist in section **5.3**

```
bash expt3/fc/run_pnn.sh
```

For cnn on cifar in section **5.3**
```
bash expt3/cnn/run_cifar_pnn.sh
```


