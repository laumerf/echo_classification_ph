#!/usr/bin/env bash

source deactivate
source activate thesis_cluster2

#aug_type=(0 1 2 3 4)
aug_type=(6)
for a in ${aug_type[@]}; do
  for fold in {0..9}; do
    echo "Starting run for fold ${fold}"
    bsub -R "rusage[mem=4096, ngpus_excl_p=1]" -W 04:00 -n 6 \
      "python scripts/train_simple.py --max_epochs 300 --num_rand 10 --batch_size 64 --res_dir results_aug --eval_metrics video-b-accuracy/valid --cache_dir ~/.heart_echo --k 10 --fold ${fold} --augment --aug_type ${a} --hist --class_balance_per_epoch --optimizer adamw --wd 1e-3 --pretrained"
    #bsub -R "rusage[mem=4096, ngpus_excl_p=1]" -W 04:00 -n 6 \
    #  "python scripts/train_simple.py --max_epochs 300 --num_rand 10 --batch_size 64 --res_dir results_aug --eval_metrics video-b-accuracy/valid --cache_dir ~/.heart_echo --k 10 --fold ${fold} --augment --aug_type ${a} --hist --class_balance_per_epoch --optimizer adam"
  done
done
