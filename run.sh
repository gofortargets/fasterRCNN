#!/usr/bin/env bash

#python trainval_net.py --_dataset pascal_voc --net vgg16 \
#                       --save_dir models \
#                       --cuda
python trainval_net.py --dataset thilini --net vgg16 \
                       --save_dir models \
                       --cuda \
                       --mGPUs \
                       --bs 10
