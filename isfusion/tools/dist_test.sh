#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:/home/lishengjie/study/jiahao/bupt_2507/isfusion

CONFIG=configs/isfusion/isfusion_0075voxel.py      

CHECKPOINT=/home/lishengjie/study/jiahao/bupt_2507/isfusion/work_dirs/isfusion_0075voxel/epoch8/epoch_3.pth

echo "使用的配置文件: $CONFIG"
echo "使用的权重文件: $CHECKPOINT"
echo "传递给 test.py 的额外参数: $@"

CUDA_VISIBLE_DEVICES=3 \
python $(dirname "$0")/test.py $CONFIG $CHECKPOINT $@

# 运行： bash tools/dist_test.sh --eval bbox --out /home/lishengjie/study/jiahao/bupt_2507/isfusion/output/pkls/test1.pkl --show_dir /home/lishengjie/study/jiahao/bupt_2507/isfusion/output/shows/