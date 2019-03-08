#!/usr/bin/env bash

python main_cliffworld.py --total_episodes 500 --train_interval 25 --eval_episodes 10 --env CliffworldDeterministic-v0 --no_cuda --train --use_planner --runs 5 --lr 0.01
python main_cliffworld.py --total_episodes 500 --train_interval 25 --eval_episodes 50 --env CliffworldDeterministic-v0 --no_cuda --test
python main_cliffworld.py --total_episodes 500 --train_interval 25 --eval_episodes 10 --env CliffworldDeterministic-v0 --no_cuda --eval_msx

python main_cliffworld.py --total_episodes 500 --train_interval 25 --eval_episodes 10 --env Cliffworld-v0 --no_cuda --train --use_planner --runs 5 --lr 0.01
python main_cliffworld.py --total_episodes 500 --train_interval 25 --eval_episodes 50 --env Cliffworld-v0 --no_cuda --test
python main_cliffworld.py --total_episodes 500 --train_interval 25 --eval_episodes 10 --env Cliffworld-v0 --no_cuda --eval_msx