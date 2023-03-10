#!/bin/bash
python inference/inference_sim.py -a retinanet -b 1 --qtype int8 -qw int8 -bcw -vcw -fr -pz -c laplace -pcq_w -pcq_a # -d # -na --q_off 

