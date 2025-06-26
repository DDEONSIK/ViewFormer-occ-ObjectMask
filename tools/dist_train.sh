#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}

# 학습 명령어:
# nohup ./tools/torchrun_occ_train.sh ./projects/configs/viewformer/viewformer_r50_704x256_seq_90e.py 2 --work-dir ./work_dirs/ori_multi_train > ori_train_multi.log 2>&1 &
