#!/usr/bin/env bash

python trainval_net.py --dataset coco --net vgg16 \
                       --save_dir models \
                       --cuda