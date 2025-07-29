#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/home/lishengjie/study/jiahao/bupt_2507/isfusion

TASK_DESC=$1
PORT=$((8000 + RANDOM %57535))


CONFIG=configs/isfusion/isfusion_0075voxel.py

CUDA_VISIBLE_DEVICES=3 \
python $(dirname "$0")/train.py $CONFIG \
--extra_tag $TASK_DESC




