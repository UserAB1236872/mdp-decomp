#!/usr/bin/env bash

python main_highway.py --total_episodes 400 --lr 0.01 --train_interval 25 --eval_episodes 5 --no_cuda --train
#echo "Testing ..."
#python main_highway.py --total_episodes 400 --lr 0.0001 --train_interval 25 --eval_episodes 20 --no_cuda --test
#echo "MSX ..."
#python main_highway.py --total_episodes 400 --lr 0.0001 --train_interval 25 --eval_episodes 10 --no_cuda --eval_msx
