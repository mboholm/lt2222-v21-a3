#!/bin/bash
train_file=/home/xsayas@GU.GU.SE/scratch/lt2222-v21-resources/svtrain.lower.txt
test_file=/home/xsayas@GU.GU.SE/scratch/lt2222-v21-resources/svtest.lower.txt
my_ks=( 50 100 150 200 250 300 )
my_rs=( 50 100 150 200 250 300 )

if [ ! -d models ]
then
   mkdir models
fi

if [ ! -d outputs ]
then
   mkdir outputs
fi

echo "--------"
echo "TRAIN"
for k in ${my_ks[@]}
do
   echo "k"$k" (hidden layer)"
   python3 train.py $train_file models/model_k$k --k $k
done

for r in ${my_rs[@]}
do
   echo "r"$r" (epochs)"
   python3 train.py $train_file models/model_r$r --r $r
done

echo "--------"
echo "EVAL"
for k in ${my_ks[@]}
do
   echo "k"$k" (hidden layer)"
   python3 eval.py models/model_k$k $test_file outputs/output_k$k
done

for r in ${my_rs[@]}
do
   echo "r"$r" (epochs)"
   python3 eval.py models/model_r$r $test_file outputs/output_r$r
done
