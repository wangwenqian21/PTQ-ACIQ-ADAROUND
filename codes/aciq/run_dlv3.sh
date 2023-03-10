#!/bin/bash
python inference/inference_sim.py -a deeplabv3 -b 1 --qtype int8 -qw int8 -bcw -vcw -pz -fr -c laplace -pcq_w -pcq_a # --q_off 
# python inference/inference_sim.py -a deeplabv3 -b 1 --qtype int4 -qw int4 -bcw -vcw -pz # -fr -na --q_off  -pcq_w -pcq_a