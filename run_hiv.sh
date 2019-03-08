#!/usr/bin/env bash

#python main_hiv.py --lr 0.0001 --total_episodes 1000  --train_interval 25 --eval_episodes 10 --env HivSimulator-v0 --no_cuda --train
##echo "Testing ..."
python main_hiv.py --lr 0.0001  --total_episodes 1000 --train_interval 25 --eval_episodes 10 --env HivSimulator-v0 --no_cuda --test
echo "MSX ..."
python main_hiv.py --lr 0.0001  --total_episodes 1000 --train_interval 25 --eval_episodes 3 --env HivSimulator-v0 --no_cuda --eval_msx
