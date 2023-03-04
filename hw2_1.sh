#!/bin/bash

python3 hw2_1_inf.py $1
# python -m pytorch_fid $1 hw2_data/face/val --num-workers 0
python3 face_recog.py --image_dir $1
# TODO - run your inference Python3 code