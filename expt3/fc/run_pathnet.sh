#!/bin/bash
# AoB: using random attachment
# ApB: means using preferential attachment
# ds: original dense-sparse procedure

# connect fraction = 5%
python mnist_prefer_pathnet.py --method="AoB" --permute_size=8  --connect_fraction=0.05  --model="custom_mlp:2,1024,0.0,0.0";
python mnist_prefer_pathnet.py --method="ApB" --permute_size=8 --connect_fraction=0.05  --model="custom_mlp:2,1024,0.0,0.0";
python mnist_prefer_pathnet.py --method="ds" --permute_size=8  --connect_fraction=0.05  --model="custom_mlp:2,1024,0.0,0.0";


# connect fraction = 10, 20, 30
for((j=1;j<=3;j++));
do
	a =`echo "$j*0.1"|bc`;
	echo $a

	python mnist_prefer_pathnet.py --method="AoB" --permute_size=8  --connect_fraction=$a --model="custom_mlp:2,1024,0.0,0.0";
	python mnist_prefer_pathnet.py --method="ApB" --permute_size=8 --connect_fraction=$a --model="custom_mlp:2,1024,0.0,0.0";
	python mnist_prefer_pathnet.py --method="ds" --permute_size=8  --connect_fraction=$a --model="custom_mlp:2,1024,0.0,0.0";

done



