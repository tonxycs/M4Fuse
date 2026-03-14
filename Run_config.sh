#!/bin/bash

nohup python ./22666_code/T19.py \
    --data_dir "" \
    --output_root "" \
    --batch_size 2 \
    --learning_rate 1e-4 \
    --num_epochs 300 \
    --crop_size 128 128 128 \
    --num_workers 8


# nohup python ./22666_code/T21.py \
#   --batch_size 2 \
#   --num_workers 8 \
#   --epochs 300 \
#   --target_size 64 128 128 \  #128 128 128
#   --lr 1e-4 \
#   --result_dir ./22666_code/Results \
#   > ./22666_code/log/log 2>&1 &

