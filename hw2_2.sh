#!/bin/bash
start=$(date +%s)
python3 hw2_2_inf.py $1
end=$(date +%s)
echo "Elapsed Time: $(($end-$start)) seconds"
# TODO - run your inference Python3 code