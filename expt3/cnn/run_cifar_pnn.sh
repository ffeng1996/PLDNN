#!/bin/bash
# separate network
for((i=1;i<=4;i++));
do

	a=`echo "$i*4"|bc`;
	b=`echo "0"|bc`;
	echo $a $b
	python cifar_prefer.py --method="AoB"  --add_nodes=$a --connect_fraction=$b;
	python cifar_prefer.py --method="ApB"  --add_nodes=$a --connect_fraction=$b;

done

# dense progressive network
for((i=1;i<=4;i++));
do
	a=`echo "$i*4"|bc`;
	b=`echo "1"|bc`;
	echo $a $b
	python cifar_prefer.py --method="AoB"  --add_nodes=$a --connect_fraction=$b;
	python cifar_prefer.py --method="ApB"  --add_nodes=$a --connect_fraction=$b;
done

# random attachment (AoB) or preferential attachment (ApB) 
for((i=1;i<=4;i++));
do
	a=`echo "$i*4"|bc`;
	b=`echo "0.1"|bc`;
	echo $a $b
	python cifar_prefer.py --method="AoB"  --add_nodes=$a --connect_fraction=$b;
	python cifar_prefer.py --method="ApB"  --add_nodes=$a --connect_fraction=$b;
done




