#!/usr/bin/env bash

source deactivate
source activate thesis_cluster2

max_p=(0 1 50 90 95)
for p in ${max_p[@]}; do
  for fold in {0..9}; do
    echo "Starting run for fold ${fold}"
    bsub -R "rusage[mem=4096, ngpus_excl_p=1]" -W 04:00 -n 6 \
      "python scripts/train_simple.py --max_epochs 300 --max_p ${p} --res_dir results_maxp --eval_metrics video-b-accuracy/valid --cache_dir ~/.heart_echo --k 10 --fold ${fold} --augment --aug_type 3 --hist --class_balance_per_epoch --optimizer adamw --pretrained --wd 0.001"
    bsub -R "rusage[mem=4096, ngpus_excl_p=1]" -W 04:00 -n 6 \
      "python scripts/train_simple.py --max_epochs 300 --max_p ${p} --res_dir results_maxp --eval_metrics video-b-accuracy/valid --cache_dir ~/.heart_echo --k 10 --fold ${fold} --augment --aug_type 3 --hist --class_balance_per_epoch --optimizer adamw --wd 0.001"
  done
done