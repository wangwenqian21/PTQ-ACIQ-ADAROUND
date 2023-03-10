#!/bin/bash
python inference/inference_sim.py -a resnet50 -b 64 --qtype int8 -qw int8 -pcq_w -pcq_a -c laplace -fr -bcw -vcw -pz 