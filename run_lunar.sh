#!/usr/bin/env bash

#python main_lunar.py --lr 0.0001 --total_episodes 400  --train_interval 25 --eval_episodes 10 --env DecomposedLunarLander-v2 --no_cuda --train
#echo "Testing ..."
#python main_lunar.py --lr 0.0001  --total_episodes 400 --train_interval 25 --eval_episodes 20 --env DecomposedLunarLander-v2 --no_cuda --test
#echo "MSX ..."
python main_lunar.py --lr 0.0001  --total_episodes 400 --train_interval 25 --eval_episodes 1 --env DecomposedLunarLander-v2 --no_cuda --eval_msx --up
