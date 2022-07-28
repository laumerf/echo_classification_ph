import numpy as np
import random
import torch
import os
from torch import cuda, device
from torch import nn, optim, no_grad
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import wandb
from echo_ph.data.echo_dataset import EchoDataset
from echo_ph.data.ph_labels import long_label_type_to_short
from echo_ph.evaluation.metrics import Metrics
from echo_ph.data.transforms import get_transforms
from utils.helpers import get_index_file_path, set_arg_parse_all, get_temp_model, get_spatial_model
import warnings

"""
This script trains the 3D CNNs or spatial CNNs.
"""

TORCH_SEED = 0
torch.manual_seed(TORCH_SEED)  # Fix a seed, to increase reproducibility
torch.cuda.manual_seed(TORCH_SEED)
torch.cuda.manual_seed_all(TORCH_SEED)
np.random.seed(TORCH_SEED)
random.seed(TORCH_SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False
# Generator for same data
g = torch.Generator()
g.manual_seed(TORCH_SEED)

parser = set_arg_parse_all(description=
                           'Train a ML model for classifying PH from ECHO. Make sure to have generated label files '
                           '(placed in project_root/label_files), and index files placed in project_root/index_files')

BASE_MODEL_DIR = 'models'


def get_run_name():
    """
    Returns a 'semi'-unique name according to most important arguments.
    Can be used for model name and tb log names
    :return: run name
    """
    if args.run_id == '':
        run_id = ''
    else:
        run_id = args.run_id + '_'
    if args.k is None:
        k = ''
    else:
        k = '.k' + str(args.k)
    if args.wd is not None:
        wd = '.wd_' + str(args.wd)
    else:
        wd = ''
    if args.temporal:
        start = 'TEMP' + '_cl' + str(args.clip_len) + '_sp' + str(args.period)
    else:
        start = ''
    run_name = start + run_id + args.model + '_' + args.optimizer + '_lt_' + long_label_type_to_short[args.label_type] \
               + k + '.lr_' + str(args.lr) + '.batch_' + str(args.batch_size) + wd + '.me_' + str(args.max_epochs)
    if args.model.endswith('multi_view'):
        run_name += 'join_' + args.join_method
    if args.segm_masks:
        run_name += 'SEGM'
    if args.multi_gpu:
        run_name += 'multi_gpu'
    if args.self_attention:
        run_name += 'self_att'
    if args.map_attention:
        run_name += 'map_att'
    if args.decay_factor > 0.0:
        run_name += str(args.decay_factor)  # only add to description if not default
    if args.decay_patience < 1000:
        run_name += str(args.decay_patience)  # only add to description if not default
    if args.img_size != 224:
        run_name += '_size_' + str(args.img_size)
    if args.pretrained:
        run_name += '_pre'
    if args.augment:
        run_name += '_aug' + str(args.aug_type)
    if args.class_balance_per_epoch:
        run_name += '_bal'
    if args.weight_loss:
        run_name += '_w'
    if args.max_p != 90 and args.num_rand_frames is None:
        run_name += '_p' + str(args.max_p)
    if args.min_expansion:
        run_name += 'MIN'
    if args.num_rand_frames is not None:
        run_name += 'rand_n' + str(args.num_rand_frames)
    if args.crop:
        run_name += '_crop'
    if len(args.view) > 1:
        run_name += '_multi_view'
    elif args.view[0] != 'KAPAP':
            run_name += '_' + args.view[0]
    return run_name


def get_metrics_probs(outputs, sm_prob_1s, targets, samples, binary=True, prefix=''):
    """
    Get metrics per batch
    :param outputs: Model raw outputs (before max)
    :param sm_prob_1s: Soft-max probabilities of the positive class (for binary classification)
    :param targets: Targets / true label - as a tensor OR list
    :param samples: Sample names of the batch
    :param binary: Set to true, in case of binary classification
    :param prefix: What to prefix the metric with - set to 'train' or 'valid' or 'test'
    :return: Dictionary containing the metrics and the model predictions (arg maxed outputs)
    """
    metrics = Metrics(targets, samples, model_outputs=outputs, sm_probs=sm_prob_1s, binary=binary, tb=True)
    all_metrics = metrics.get_per_sample_scores(subset=prefix)  # first get sample metrics only
    subject_metrics = metrics.get_per_subject_scores(subset=prefix)  # then get subject metrics
    all_metrics.update(subject_metrics)  # finally update sample metrics dict with subject metrics, to get all metrics
    return all_metrics


def run_batch(batch, model, criterion=None, binary=False):
    """
    Run a single batch
    :param batch: The data for this batch
    :param model: The seq2seq model
    :param criterion: The criterion for the loss. Set to None during evaluation (no training).
    :param binary: Set to True if this is binary classification.
    :return: The required metrics for this batch, as well as the predictions and targets
    """
    dev = device('cuda' if cuda.is_available() else 'cpu')
    input = batch["frame"]  #.to(dev)
    if len(args.view) == 1:  # single_view
        view = args.view[0]
        input = input[view].to(dev)  # Batch_size, (seq_len), num_channels, w, h => single view
        if args.temporal:
            input = input.transpose(2, 1)  # Reshape to: (batch_size, channels, seq-len, W, H)
    # Else if multi_view, further input processing happens in model
    targets = batch["label"].to(dev)
    sample_names = batch["sample_name"]
    outputs = model(input)
    if args.model == 'saliency_r3d_18':
        outputs, _ = outputs  # latter output is last conv layer -> just need it when evaluate
        attention = None
    elif isinstance(outputs, tuple):
        attention = outputs[1]  # The later value is the attention (for visualisation)
        outputs = outputs[0]  # The prev value is the actual output for predictions
    else:
        attention = None

    # Get loss, if we are training
    if criterion:
        if binary:
            # Convert to one-hot encoding, and convert to float, bc the binary loss supports prob. labels (“soft” labels).
            one_hot_targets = torch.nn.functional.one_hot(targets, num_classes=2).float()
            loss = criterion(outputs, one_hot_targets)
        else:
            loss = criterion(outputs, targets)
    else:  # when just evaluating, no loss
        loss = None
    return loss, outputs, targets, sample_names, attention


def get_base_name(run_name, fold=None, epoch=None):
    """
    Get name base for all results files and models for this run, of this epoch and fold.
    :param run_name: Descriptive name of this model (run)
    :param fold: Current fold
    :param epoch: Current epoch
    :return:
    """
    fold = '' if fold is None else 'fold' + str(fold) + '_'
    epoch = '_final' if epoch is None else '_e' + str(epoch)
    base_name = fold + run_name + epoch
    return base_name


def save_model(model, run_name, epoch_run_name):
    """
    Save model for current epoch, and delete previous models of same run, to save space
    :param model: The model to save
    :param run_name: Descriptive name of this model (run)
    :param epoch_run_name: The name of the run of this epoch
    """
    model_dir = os.path.join(BASE_MODEL_DIR, run_name)
    os.makedirs(model_dir, exist_ok=True)
    model_file_name = epoch_run_name + '.pt'
    # Just before saving the model, delete older versions of the model and results, to save space
    for model_file in os.listdir(model_dir):
        # same model but different epoch
        if model_file.split('_e')[0] == model_file_name.split('_e')[0]:
            os.system(f'rm -r {os.path.join(model_dir, model_file)}')
            res_dir_to_del = os.path.join(BASE_RES_DIR, run_name, model_file[:-3])
            if os.path.exists(res_dir_to_del):
                os.system(f'rm -r {res_dir_to_del}')
    if args.multi_gpu:  # After wrapping the model in nn.DataParallel, original model will be accessible via model.module
        torch.save(model.module.state_dict(), os.path.join(model_dir, model_file_name))
    else:
        torch.save(model.state_dict(), os.path.join(model_dir, model_file_name))


def save_res(run_name, epoch_run_name, target_lst, pred_lst, sample_names, subset='val', attention=[]):
    """
    Save results for validation or train set.
    :param run_name: Descriptive name of this model (run)
    :param epoch_run_name: The name of the run of this epoch
    :param target_lst: List of targets for this subset
    :param pred_lst: List of predictions or raw model outputs for this subset
    :param sample_names: List of sample names for this subset
    :param subset: 'train' or 'val'
    :param attention: (Optional) List of attention weights, for when using attention-based model.
    """
    res_dir = os.path.join(BASE_RES_DIR, run_name, epoch_run_name)
    os.makedirs(res_dir, exist_ok=True)
    if not os.path.exists(res_dir):  # Also add this, bc in cluster sometimes the other one is not working
        os.makedirs(res_dir)
    targ_file_name = 'targets.npy'
    pred_file_name = 'preds.npy'
    sample_file_names = 'samples.npy'
    att_file_names = 'attention.npy'
    prefix = subset + '_'
    np.save(os.path.join(res_dir, prefix + targ_file_name), target_lst)
    np.save(os.path.join(res_dir, prefix + pred_file_name), pred_lst)
    np.save(os.path.join(res_dir, prefix + sample_file_names), sample_names)
    if len(attention) > 0:
        np.save(os.path.join(res_dir, 'train_' + att_file_names), attention)


def evaluate(model, dev, valid_loader, valid_len, run_name, binary=False):
    print("Start evaluating on", valid_len, "validation samples")
    sm = torch.nn.Softmax(dim=-1)
    if os.path.isdir(args.model_name):
        model_paths = os.listdir(args.model_name)
        for path in model_paths:
            if path.startswith('fold' + str(args.fold)):
                model_path = os.path.join(args.model_name, path)
                break
    else:
        model_path = args.model_name
    model.load_state_dict(torch.load(model_path, map_location=dev))
    model.eval()
    epoch_valid_targets = []
    epoch_valid_samples = []
    epoch_valid_outs = []
    epoch_valid_attention = []
    for valid_batch in valid_loader:
        _, val_out, val_targets, val_sample_names, val_att = run_batch(valid_batch, model, None, binary)
        epoch_valid_targets.extend(val_targets)
        epoch_valid_samples.extend(val_sample_names)
        if val_att is not None:
            epoch_valid_attention.append(val_att.cpu().detach().numpy())
        epoch_valid_outs.extend(val_out.cpu().detach().numpy())
    targ_lst_valid = [t.item() for t in epoch_valid_targets]
    base_name = get_base_name(run_name, fold=args.fold, epoch=None)
    save_res(run_name, base_name, targ_lst_valid, epoch_valid_outs, epoch_valid_samples, subset='val',
             attention=epoch_valid_attention)


def train(model, train_loader, valid_loader, data_len, valid_len, tb_writer, run_name, optimizer, weights=None,
          binary=False, use_wandb=False):

    # Set training loss, optimizer and training parameters
    if binary:
        criterion = nn.BCEWithLogitsLoss(weight=weights)  # if weights is None, no weighting is performed
    else:
        criterion = nn.CrossEntropyLoss(weight=weights)  # if weights is None, no weighting is performed
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.decay_factor, patience=args.decay_patience,
                                                     min_lr=args.min_lr, cooldown=args.cooldown)
    best_early_stops = []
    for eval_metric in args.eval_metrics:
        best = float("inf") if 'loss' in eval_metric else -1  # if loss metric, then minimize, else maximize
        best_early_stops.append(best)

    num_val_fails = 0
    print("Start training on", data_len, "training samples, and", valid_len, "validation samples")
    sm = torch.nn.Softmax(dim=-1)
    for epoch in range(args.max_epochs):
        epoch_loss = epoch_valid_loss = 0
        epoch_targets = []
        epoch_outs = []
        epoch_prob_1s = []
        epoch_samples = []
        epoch_attention = []
        epoch_valid_targets = []
        epoch_valid_samples = []
        epoch_valid_outs = []
        epoch_valid_prob_1s = []
        epoch_valid_attention = []

        # TRAIN
        model.train()
        for train_batch in train_loader:
            loss, out, targets, sample_names, att = run_batch(train_batch, model, criterion, binary)
            epoch_samples.extend(sample_names)
            epoch_targets.extend(targets)
            if att is not None:
                epoch_attention.extend(att.cpu().detach().numpy())
            epoch_outs.extend(out.cpu().detach().numpy())
            epoch_prob_1s.extend(sm(out)[:, 1].cpu().detach().numpy())
            epoch_loss += loss.item() * args.batch_size
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

        # VALIDATE
        with no_grad():
            model.eval()
            for valid_batch in valid_loader:
                val_loss, val_out, val_targets, val_sample_names, val_att = run_batch(valid_batch, model, criterion,
                                                                                      binary)
                epoch_valid_samples.extend(val_sample_names)
                epoch_valid_targets.extend(val_targets)
                if val_att is not None:
                    epoch_valid_attention.append(val_att.cpu().detach().numpy())
                epoch_valid_outs.extend(val_out.cpu().detach().numpy())
                epoch_valid_prob_1s.extend(sm(val_out)[:, 1].cpu().detach().numpy())
                epoch_valid_loss += val_loss.item() * args.batch_size

        scheduler.step(epoch_valid_loss / valid_len)  # Update learning rate scheduler

        if epoch % args.log_freq == 0:  # log every xth epoch
            target_lst = [t.item() for t in epoch_targets]
            targ_lst_valid = [t.item() for t in epoch_valid_targets]
            epoch_metrics = get_metrics_probs(epoch_outs, epoch_prob_1s, target_lst, epoch_samples, prefix='train',
                                              binary=binary)
            epoch_valid_metrics = get_metrics_probs(epoch_valid_outs, epoch_valid_prob_1s, targ_lst_valid,
                                                    epoch_valid_samples, prefix='valid', binary=binary)
            print('*** epoch:', epoch, '***')
            print('train_loss:', epoch_loss / data_len)
            print('valid loss:', epoch_valid_loss / valid_len)

            for metric in epoch_metrics:
                print(metric, ":", epoch_metrics[metric])
            for metric in epoch_valid_metrics:
                print(metric, ":", epoch_valid_metrics[metric])

            if args.debug:
                vals, cnts = np.unique(target_lst, return_counts=True)
                print('epoch target distribution')
                for val, cnt in zip(vals, cnts):
                    print(val, ':', cnt)
            else:  # log and save results
                log_dict = {
                    "epoch": epoch,
                    "lr": optimizer.param_groups[0]['lr'],  # Actual learning rate (changes because of scheduler)
                    "loss/valid": epoch_valid_loss / valid_len,
                    "loss/train": epoch_loss / data_len
                }
                log_dict.update(epoch_metrics)
                log_dict.update(epoch_valid_metrics)
                if use_wandb:
                    wandb.log(log_dict)
                for metric_key in log_dict:
                    step = int(epoch / args.log_freq)
                    tb_writer.add_scalar(metric_key, log_dict[metric_key], step)

                if args.early_stop:
                    i = 0
                    saved_model_this_round = False
                    num_fails_this_round = 0
                    # If any of eval metrics improve => save model and reset early stop counter
                    for best_early_stop, eval_metric in zip(best_early_stops, args.eval_metrics):
                        if 'loss' in eval_metric:  # smaller value is better
                            better_res = log_dict[eval_metric] < best_early_stop
                        else:  # larger value is better
                            better_res = log_dict[eval_metric] > best_early_stop

                        if better_res:
                            best_early_stops[i] = log_dict[eval_metric]  # update
                            num_val_fails = 0
                            if not saved_model_this_round:
                                base_name = get_base_name(run_name, fold=args.fold, epoch=epoch)
                                save_model(model, run_name, base_name)
                                save_res(run_name, base_name, target_lst, epoch_outs, epoch_samples, subset='train',
                                         attention=epoch_attention)
                                save_res(run_name, base_name,  targ_lst_valid, epoch_valid_outs, epoch_valid_samples,
                                         subset='val', attention=epoch_valid_attention)
                            saved_model_this_round = True
                        else:
                            num_fails_this_round += 1
                        i += 1

                    if num_fails_this_round == len(args.eval_metrics):  # If no eval metric improved
                        num_val_fails += 1

                    if num_val_fails >= args.early_stop:
                        print('== Early stop training after', num_val_fails,
                              'epochs without validation loss improvement')
                        break

    if not args.debug:
        tb_writer.close()


def main():
    # Set up device, logging, run name, etc.
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Will be training on device', dev)
    run_name = get_run_name()
    print('Run:', run_name)
    use_wandb = False  # Set this to false for now as can't seem to use on cluster
    if not args.debug:
        warnings.simplefilter("ignore")  # Ignore warnings, so they don't fill output log files
        # Initialize  logging if not debug phase
        if use_wandb:
            wandb.init(project='echo_classification', entity='XXXX-6', config={}, mode="offline")
            wandb.config.update(args)
        fold = '' if args.fold is None else 'fold' + str(args.fold) + '_'
        tb_writer = SummaryWriter(log_dir=os.path.join(args.tb_dir, fold + run_name))
    else:
        tb_writer = None

    # Get paths
    binary = True if args.label_type.startswith('2class') else False
    label_path = os.path.join('label_files', 'labels_' + args.label_type + '.pkl')
    train_index_file_path = get_index_file_path(args.k, args.fold, args.label_type, train=True)
    valid_index_file_path = get_index_file_path(args.k, args.fold, args.label_type, train=False)

    size = args.img_size
    views = args.view
    if args.augment and not args.load_model:  # All augmentations
        train_transforms = get_transforms(train_index_file_path, dataset_orig_img_scale=args.scaling_factor, resize=size,
                                          augment=args.aug_type, fold=args.fold, valid=False, view=views,
                                          crop_to_corner=args.crop, segm_mask_only=args.segm_masks,
                                          label_type=args.label_type)
    else:
        train_transforms = get_transforms(train_index_file_path, dataset_orig_img_scale=args.scaling_factor, resize=size,
                                          augment=0, fold=args.fold, valid=False, view=views,
                                          crop_to_corner=args.crop,
                                          segm_mask_only=args.segm_masks)
    valid_transforms = get_transforms(valid_index_file_path, dataset_orig_img_scale=args.scaling_factor, resize=size,
                                      augment=0, fold=args.fold, valid=True, view=views,
                                      crop_to_corner=args.crop,
                                      segm_mask_only=args.segm_masks)
    train_dataset = EchoDataset(train_index_file_path, label_path, videos_dir=args.videos_dir,
                                cache_dir=args.cache_dir,
                                transform=train_transforms, scaling_factor=args.scaling_factor,
                                procs=args.num_workers, visualise_frames=args.visualise_frames,
                                percentile=args.max_p, view=views, min_expansion=args.min_expansion,
                                num_rand_frames=args.num_rand_frames, segm_masks=args.segm_masks,
                                temporal=args.temporal, clip_len=args.clip_len, period=args.period)
    if args.weight_loss:
        class_weights = torch.tensor(train_dataset.class_weights, dtype=torch.float).to(dev)
    else:
        class_weights = None
    valid_dataset = EchoDataset(valid_index_file_path, label_path, videos_dir=args.videos_dir, cache_dir=args.cache_dir,
                                transform=valid_transforms, scaling_factor=args.scaling_factor, procs=args.num_workers,
                                visualise_frames=args.visualise_frames, percentile=args.max_p, view=views,
                                min_expansion=args.min_expansion, num_rand_frames=args.num_rand_frames,
                                segm_masks=args.segm_masks, temporal=args.temporal,
                                clip_len=args.clip_len, period=args.period)
    # For the data loader, if only use 1 worker, set it to 0, so data is loaded on the main process
    num_workers = (0 if args.num_workers == 1 else args.num_workers)

    # Model & Optimizers
    num_classes = len(train_dataset.labels)
    if args.temporal:
        model = get_temp_model(args.model, num_classes, args.pretrained, dev, views=views, size=size,
                               self_att=args.self_attention, map_att=args.map_attention, cl=args.clip_len,
                               join_method=args.join_method)
    else:
        model = get_spatial_model(args.model, num_classes, args.pretrained, views, dev, args.dropout,
                                  join_method=args.join_method)
    if args.multi_gpu:
        print("Training with multiple GPU")
        model = nn.DataParallel(model, dim=0)  # As we have batch-first
    os.makedirs(BASE_MODEL_DIR, exist_ok=True)  # create model results dir, if not exists
    os.makedirs(BASE_RES_DIR, exist_ok=True)  # create results dir, if not exists

    if args.optimizer == 'adam':
        if args.wd is not None:
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        else:  # default => no wd
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:  # optimizer = adamw
        if args.wd is not None:
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        else:  # default, wd=0.02
            optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    if args.class_balance_per_epoch:
        sampler = WeightedRandomSampler(train_dataset.example_weights, train_dataset.num_samples, generator=g)
        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=False, num_workers=num_workers,
                                  sampler=sampler, worker_init_fn=seed_worker, generator=g)  # Sampler is mutually exclusive with shuffle
    else:
        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=num_workers,
                                  worker_init_fn=seed_worker, generator=g)
    valid_loader = DataLoader(valid_dataset, args.batch_size, shuffle=False, num_workers=num_workers)
    if args.load_model:
        evaluate(model, dev, valid_loader, len(valid_dataset), run_name, binary=binary)
    else:
        train(model, train_loader, valid_loader, len(train_dataset), len(valid_dataset), tb_writer, run_name,
              optimizer, weights=class_weights, binary=binary, use_wandb=use_wandb)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == "__main__":
    args = parser.parse_args()
    BASE_RES_DIR = args.res_dir
    main()


