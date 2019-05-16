#!/bin/sh
# 0.0 can be replaced with any paramger such as 0.4 and 0.7
# 100 means the number of iterations
# i corresponds to the number for running paralel
i=$@
python page_block.py $i 0.0  100
