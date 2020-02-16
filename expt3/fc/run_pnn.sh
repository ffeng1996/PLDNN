#!/bin/bash
# AoB: using random attachment
# ApB: means using preferential attachment

# separate net
for((i=1;i<=3;i++));
do
	a=`echo "$i*10"|bc`;
	b=`echo "0"|bc`;
	echo $a $b
	python mnist_prefer_pnn.py --method="AoB" --permute_size=8 --add_nodes=$a --connect_fraction=$b;
	python mnist_prefer_pnn.py --method="ApB" --permute_size=8 --add_nodes=$a --connect_fraction=$b;

done

# dense pnn
for((i=1;i<=3;i++));
do
	a=`echo "$i*10"|bc`;
	b=`echo "1"|bc`;
	echo $a $b
	python mnist_prefer_pnn.py --method="AoB" --permute_size=8 --add_nodes=$a --connect_fraction=$b;
	python mnist_prefer_pnn.py --method="ApB" --permute_size=8 --add_nodes=$a --connect_fraction=$b;

done

# using a fraction of lateral connections
for((i=1;i<=3;i++));
do
	for((j=0;j<=4;j++));
	do

		a=`echo "$i*10"|ibc`;
		b=`echo "2^$j*0.05"|bc`;
		echo $a $b
		python mnist_prefer_pnn.py --method="AoB" --permute_size=8 --add_nodes=$a --connect_fraction=$b;
		python mnist_prefer_pnn.py --method="ApB" --permute_size=8 --add_nodes=$a --connect_fraction=$b;
	done
done
