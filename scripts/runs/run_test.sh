#!/usr/bin/env bash

source deactivate
source activate thesis_cluster2

for fold in {0..9}; do
  echo "Starting run for fold ${fold}"
    bsub -R "rusage[mem=4096, ngpus_excl_p=1]" -W 04:00 -n 6 \
    "python scripts/train_simple.py --max_epochs 60 --max_p 95 --batch_size 64  --res_dir results_test --eval_metrics video-b-accuracy/valid --cache_dir ~/.heart_echo  --k 10 --fold ${fold} --augment --hist --class_balance_per_epoch --optimizer adam --pretrained"
    #bsub -R "rusage[mem=4096, ngpus_excl_p=1]" -W 04:00 -n 6 \
    #"python scripts/train_simple.py --max_epochs 100 --max_p 95 --res_dir results_test --eval_metrics video-b-accuracy/valid --cache_dir ~/.heart_echo  --k 10 --fold ${fold} --augment --hist --class_balance_per_epoch --optimizer adamw --pretrained"
    #bsub -R "rusage[mem=4096, ngpus_excl_p=1]" -W 04:00 -n 6 \
    #"python scripts/train_simple.py --max_epochs 100 --max_p 95 --aug_type 4 --res_dir results_test --eval_metrics video-b-accuracy/valid --cache_dir ~/.heart_echo  --k 10 --fold ${fold} --augment --hist --class_balance_per_epoch --optimizer adam --pretrained"
    #bsub -R "rusage[mem=4096, ngpus_excl_p=1]" -W 04:00 -n 6 \
    #"python scripts/train_simple.py --max_epochs 100 --max_p 95 --aug_type 4 --res_dir results_test --eval_metrics video-b-accuracy/valid --cache_dir ~/.heart_echo  --k 10 --fold ${fold} --augment --hist --class_balance_per_epoch --optimizer adamw --pretrained"
    #bsub -R "rusage[mem=4096, ngpus_excl_p=1]" -W 04:00 -n 6 \
    #"python scripts/train_simple.py --max_epochs 100 --batch_size 64 --max_p 95 --res_dir results_test --eval_metrics video-b-accuracy/valid --cache_dir ~/.heart_echo  --k 10 --fold ${fold} --augment --hist --class_balance_per_epoch --optimizer adam --pretrained"
    #bsub -R "rusage[mem=4096, ngpus_excl_p=1]" -W 04:00 -n 6 \
    #"python scripts/train_simple.py --max_epochs 300 --batch_size 64 --max_p 95 --res_dir results_test --eval_metrics video-b-accuracy/valid --cache_dir ~/.heart_echo  --k 10 --fold ${fold} --augment --hist --class_balance_per_epoch --optimizer adam --pretrained"
    bsub -R "rusage[mem=4096, ngpus_excl_p=1]" -W 04:00 -n 6 \
    "python scripts/train_simple.py --max_epochs 60 --max_p 95 --batch_size 16 --res_dir results_test --eval_metrics video-b-accuracy/valid --cache_dir ~/.heart_echo  --k 10 --fold ${fold} --augment --hist --class_balance_per_epoch --optimizer adam --pretrained"
    #bsub -R "rusage[mem=4096, ngpus_excl_p=1]" -W 04:00 -n 6 \
    #"python scripts/train_simple.py --max_epochs 30 --max_p 95 --lr 1e-4 --res_dir results_test --eval_metrics video-b-accuracy/valid --cache_dir ~/.heart_echo  --k 10 --fold ${fold} --augment --hist --class_balance_per_epoch --optimizer adam --pretrained"

done
