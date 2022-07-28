# Classifying PH from ECHOS

### Summary of repository
Severity Prediction of PH in newborns using echocardiography (ECHO).
- Trained models can be found in the following [url](https://drive.google.com/drive/folders/10sTERl6dbAxTilWNJqpGt1BVsf6plUhG)
- Main code and classes are located in the *ehco_ph* module.

- Scripts for pre-processing dataset, generating index files (for splitting into train 
and validation set), training models, and analysing results is found in the *scripts* directory.
### Setup
- Create a conda environment with <code>conda env create -f environment.yml</code>
- Activate the conda environment.
- Run <code> pip install -e . </code> (to activate the echo_ph module)

### Description of main scripts and how to run them:
#### Preparing the data for training and evaluation
- Pre-process the dataset: <code>python scripts/data/preprocess_videos.py</code>
- Generate clean labels from excel annotations: <code>python scripts/data/generate_labels.py</code>
- Generate index files with samples acc. to train-val split: <code>python scripts/data/generate_index_files.py</code>
#### Training the models, and saving result files: 
- Script: <code>scripts/train.py</code>
- Example of training temporal severity PH prediction model on the PSAX view:
  -     python scripts/train_simple.py --max_epochs 300 --wd 1e-3 --class_balance_per_epoch --cache_dir ~/.heart_echo --k 10 --fold ${fold} --augment --pretrained --num_rand_frames 10 --model r3d_18 --temporal --label_type 3class --view KAPAP --batch_size 8
- Example of training spatial binary PH detection model on the PLAX view:
  -     python scripts/train_simple.py --max_epochs 300 --wd 1e-3 --class_balance_per_epoch --cache_dir ~/.heart_echo --k 10 --fold ${fold} --augment --pretrained --num_rand_frames 10 --model resnet --label_type 2class_drop_ambiguous --view LA --batch_size 64
#### Evaluating an already trained model
  - Use same script as for training (<code>scripts/train.py</code>), with the same arguments as when you trained the model you are now evaluating.
    - Add the arguments: <code>--load_model</code> and <code>--model_path <path_to_trained_model></code>
    - This will save the result files, holding the raw output, target and sample names.
      - If desired, you can get metric results, by running the <code>scripts/evaluation/get_metrics.py</code> (see next section)
#### Get metrics from raw result files
- <code> python scripts/evaluation/get_metrics.py --res_dir res_dir</code>
- Add <code>--multi_class </code> if any of the models from res_dir are not binary classification.
- Note that the res_dir should be the directory storing the directory of other model(s) results dirs.
#### Visualisations
- Get grad-cam saliency map visualisations for temporal model: 
     - Save 1 clip per video:
     -     python scripts/visualisations/vis_grad_cam_temp.py --model_path  <path_to_trained_model.pt>  --model <model_type> --num_rand_samples 1 --save_video_clip
     - Save full video (feed all frames - but model not trained with this long input): 
     -     python scripts/visualise/vis_grad_cam_temp.py --model_path  <path_to_trained_model.pt>  --model <model_type> --all_frames --save_video --view <view>
- Get grad-cam saliency map visualisations for spatial model:
  - Use  <code>python scripts/visualisations/vis_grad_cam.py </code>
#### Multi-View Majority Vote (MV) and frame level joining of views :
- Mv of 3 views (similar for 5 views):
   -     python scripts/evaluation/multi_view_ensemble.py base_res_dir --res_files file_name_KAPAP file_name_CV file_name_CV --views kapap cv la
- Frame-level joining of 3 views: 
  -     python scripts/evaluation/join_view_models_frame_level.py base_res_dir --res_files file_name_KAPAP file_name_CV file_name_CV --views kapap cv la

