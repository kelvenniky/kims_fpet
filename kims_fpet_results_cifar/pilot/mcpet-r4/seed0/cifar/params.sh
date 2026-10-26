#!/bin/bash
python3 /content/kims_fpet/main.py --dataset cifar --method adaptformer --dim 8 --bit 32 --load_config --seed 0 --model_path /content/drive/MyDrive/fpet_results/pilot/mcpet-r4/seed0 --r_layer 4 --consistency
