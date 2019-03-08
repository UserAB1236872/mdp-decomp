#!/usr/bin/env bash

python main_lunar.py --total_episodes 100  --train_interval 25 --eval_episodes 5 --env DecomposedLunarLander-v2 --train --no_cuda --runs 1 --lr 0.001 --update_target_interval 100
#echo "Testing ..."
#python main_lunar.py --lr 0.0001  --total_episodes 400 --train_interval 25 --eval_episodes 20 --env DecomposedLunarLander-v2 --no_cuda --test
#echo "MSX ..."
#python main_lunar.py --lr 0.0001  --total_episodes 400 --train_interval 25 --eval_episodes 5 --env DecomposedLunarLander-v2 --no_cuda --eval_msx
