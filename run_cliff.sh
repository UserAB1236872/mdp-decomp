#!/usr/bin/env bash

python main_cliffworld.py --total_episodes 500 --train_interval 25 --eval_episodes 20 --env CliffworldDeterministic-v0 --no_cuda --train --use_planner --runs 5
python main_cliffworld.py --total_episodes 500 --train_interval 25 --eval_episodes 100 --env CliffworldDeterministic-v0 --no_cuda --test
python main_cliffworld.py --total_episodes 500 --train_interval 25 --eval_episodes 20 --env CliffworldDeterministic-v0 --no_cuda --eval_msx

python main_cliffworld.py --total_episodes 500 --train_interval 25 --eval_episodes 20 --env Cliffworld-v0 --no_cuda --train --use_planner --runs 5
python main_cliffworld.py --total_episodes 500 --train_interval 25 --eval_episodes 100 --env Cliffworld-v0 --no_cuda --test
python main_cliffworld.py --total_episodes 500 --train_interval 25 --eval_episodes 20 --env Cliffworld-v0 --no_cuda --eval_msx