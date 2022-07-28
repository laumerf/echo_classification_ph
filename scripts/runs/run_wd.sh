#!/usr/bin/env bash

source deactivate
source activate thesis_cluster2

wd_val=(0 1e-5 1e-4 1e-3 1e-2 2e-2)
for wd in ${wd_val[@]}; do
  for fold in {0..9}; do
    echo "Starting run for fold ${fold}"
    bsub -R "rusage[mem=2048, ngpus_excl_p=1]" -W 04:00 -n 6 \
      "python scripts/train_simple.py --max_epochs 300 --max_p 95 --batch_size 64 --wd ${wd} --res_dir results_wd --eval_metrics video-b-accuracy/valid --cache_dir ~/.heart_echo --k 10 --fold ${fold} --hist --class_balance_per_epoch --optimizer adam --pretrained"
    bsub -R "rusage[mem=2048, ngpus_excl_p=1]" -W 04:00 -n 6 \
      "python scripts/train_simple.py --max_epochs 300 --max_p 95 --batch_size 64 --wd ${wd} --res_dir results_wd --eval_metrics video-b-accuracy/valid --cache_dir ~/.heart_echo --k 10 --fold ${fold} --hist --class_balance_per_epoch --optimizer adamw --pretrained"
    bsub -R "rusage[mem=2048, ngpus_excl_p=1]" -W 04:00 -n 6 \
      "python scripts/train_simple.py --max_epochs 300 --max_p 95 --batch_size 64 --wd ${wd} --res_dir results_wd --eval_metrics video-b-accuracy/valid --cache_dir ~/.heart_echo --k 10 --fold ${fold} --hist --class_balance_per_epoch --optimizer adam"
    bsub -R "rusage[mem=2048, ngpus_excl_p=1]" -W 04:00 -n 6 \
      "python scripts/train_simple.py --max_epochs 300 --max_p 95 --batch_size 64 --wd ${wd} --res_dir results_wd --eval_metrics video-b-accuracy/valid --cache_dir ~/.heart_echo --k 10 --fold ${fold} --hist --class_balance_per_epoch --optimizer adamw"
  done
done
