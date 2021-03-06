#!/bin/bash
set -e

# python prepare.py
cd detector
maxeps=200
f=0
startep=151
#CUDA_VISIBLE_DEVICES=0 python main.py --model res18 -b 12 --resume 064.ckpt --save-dir res18/retrft96$f/ --epochs $maxeps --config config_training$f
CUDA_VISIBLE_DEVICES=0 python main.py --model res18 -b 12 --resume results/res18/retrft96$f/150.ckpt --save-dir res18/retrft96$f/ --epochs $maxeps --config config_training$f --start-epoch $startep
for (( i=$maxeps; i<=$maxeps; i+=1))
do
    echo "process $i epoch"

	if [ $i -lt 10 ]; then
	    CUDA_VISIBLE_DEVICES=0 python main.py --model res18 -b 12 --resume results/res18/retrft96$f/00$i.ckpt --test 1 --save-dir res18/retrft96$f/ --config config_training$f
	elif [ $i -lt 100 ]; then
	    CUDA_VISIBLE_DEVICES=0 python main.py --model res18 -b 12 --resume results/res18/retrft96$f/0$i.ckpt --test 1 --save-dir res18/retrft96$f/ --config config_training$f
	elif [ $i -lt 1000 ]; then
	    CUDA_VISIBLE_DEVICES=0 python main.py --model res18 -b 12 --resume results/res18/retrft96$f/$i.ckpt --test 1 --save-dir res18/retrft96$f/ --config config_training$f
	else
	    echo "Unhandled case"
    fi

    if [ ! -d "results/res18/retrft96$f/val$i/" ]; then
        mkdir results/res18/retrft96$f/val$i/
    fi
    mv results/res18/retrft96$f/bbox/*.npy results/res18/retrft96$f/val$i/
done