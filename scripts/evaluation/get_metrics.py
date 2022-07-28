import numpy as np
import os
import pandas as pd
from sklearn.metrics import roc_curve
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from matplotlib import pyplot as plt
from echo_ph.evaluation.metrics import Metrics, get_save_classification_report, get_save_confusion_matrix, read_results
from statistics import multimode

parser = ArgumentParser(
    description='Get metrics',
    formatter_class=ArgumentDefaultsHelpFormatter)
# Paths, file name, model names, etc
parser.add_argument('--res_dir', type=str, default=None,
                    help='Set path of directory containing results for all desired runs - if this is desired.')
parser.add_argument('--run_paths', type=str, default=None, nargs='+',
                    help='Set paths to all individual runs that you wish to get results for - if this is desired.')
parser.add_argument('--out_names', type=str, default=None, nargs='+',
                    help='In the case of multiple res-paths, it is optional to provide a shorter name for each run, '
                         'to be used for results')
parser.add_argument('--out_dir', type=str, default='metric_results')
parser.add_argument('--cr',  action='store_true', help='Set this flag to save also classification report per run')
parser.add_argument('--cm',  action='store_true', help='Set this flag to also save confusion matrix per run')
parser.add_argument('--train',  action='store_true', help='Set this flag to save also classification report per run')
parser.add_argument('--only_plot',  action='store_true', help='Set this flag to only plot ROC_AUC')
parser.add_argument('--plot_title',  type=str, default=None, nargs='+', help='title of ROC_AUC plot, if not default')
parser.add_argument('--multi_class', action='store_true', help='Set this flag if not binary classification')
parser.add_argument('--out_type', type=str, default='csv', choices=['csv', 'latex'])


def get_metrics_for_fold(fold_targets, fold_preds, fold_probs, fold_samples, outs):
    """
    Get / collect metrics corresponding to a single fold
    :param fold_targets: Ground truth labels
    :param fold_preds: Model predicted labels
    :param fold_probs: Output probabilities of fold
    :param fold_samples: Sample names
    :return: results dictionary, video_wise_targets, video_wise_probs
    """
    binary = True if not args.multi_class else False
    metrics = Metrics(fold_targets, fold_samples, model_outputs=outs, preds=fold_preds, sm_probs=fold_probs,
                      binary=binary, tb=False)
    all_metrics = metrics.get_per_sample_scores()  # first get sample metrics only
    subject_metrics = metrics.get_per_subject_scores()  # then get subject metrics
    all_metrics.update(subject_metrics)  # finally update sample metrics dict with subject metrics, to get all metrics
    vid_targ, vid_pred, vid_avg_prob, vid_conf, vid_conf_corr, vid_conf_wrong, vid_ids = metrics.get_subject_lists()
    all_metrics.update({'Video CI':  np.mean(vid_conf)})
    all_metrics.update({'Video Corr CI': np.mean(vid_conf_corr)})
    all_metrics.update({'Video Wrong CI': np.mean(vid_conf_wrong)})
    return all_metrics, vid_targ, vid_avg_prob, vid_pred, vid_ids


def get_metrics_for_run(res_base_dir, run_name, out_dir, col, subset='val', get_clf_report=False, get_confusion=False,
                        first=False, out_name=None):
    """
    Get list of metrics (in a string format) for current model / run, averaged over all fold, and per-video
    :param res_base_dir: Directory where results for this model are stored
    :param run_name: Name of this model / run
    :param out_dir: Name of directory to store resulting metrics
    :param col: Colour for this run, for ROC_AUC plot
    :param subset: train or val
    :param get_clf_report: Whether or not to also get classification report (frame-wise & video-wise)
    :param get_confusion: Whether or not to also get confusion matrix (frame-wise & video-wise)
    :param first: Set to true, if this is the first run
    :param out_name: Shorter name to use for saving results for this run, if desired.
    :return: list of metric strings, to be written to csv
    """
    metric_dict = {key: [] for key in metric_list}
    targets = []
    preds = []
    vid_targets = []
    vid_preds = []
    vid_ids = []
    avg_softm_probs = []
    epochs = []
    res_path = os.path.join(res_base_dir, run_name) if res_base_dir is not None else run_name
    # start with first fold, ignore .DS_store and other non-dir files
    fold_paths = [os.path.join(res_path, fold_path) for fold_path in sorted(os.listdir(res_path)) if
                  os.path.isdir(os.path.join(res_path, fold_path))]
    for fold_dir in fold_paths:
        fold_preds, fold_probs, fold_targets, fold_samples, outs = read_results(fold_dir, subset)
        if fold_preds is None:
            print(f'failed for model {os.path.basename(fold_dir)}')
            continue
        results, vid_targ, avg_prob, vid_pred, video_ids = get_metrics_for_fold(fold_targets, fold_preds, fold_probs,
                                                                                fold_samples, outs)
        for metric, val in results.items():
            if metric in metric_dict:
                metric_dict[metric].append(val)

        vid_targets.extend(vid_targ)
        vid_preds.extend(vid_pred)
        vid_ids.extend(video_ids)

        preds.extend(fold_preds)
        targets.extend(fold_targets)
        if avg_prob is not None:
            avg_softm_probs.extend(avg_prob)

    if get_clf_report or get_confusion:
        # For classification report or confusion matrix, don't average over the folds, rather get all unique video
        # (contained in all folds). Note that some videos might appear in more than 1 fold, because cross-validation
        # strategy picks random 20% videos for validation per fold, with replacement.
        # => Thus need to average per video, for those video appearing in more than one fold.
        all_unique_video_res = {}
        for v_id, v_target, v_pred in zip(vid_ids, vid_targets, vid_preds):
            if v_id in all_unique_video_res:
                all_unique_video_res[v_id][1].append(v_pred)  # add current prediction
            else:
                all_unique_video_res[v_id] = [v_target, [v_pred]]  # (video target, list of video predictions)
        # multi-mode returns list of equally most frequent item in a list => so get the most frequent label, as the
        # video label. In case of a tie, pick the higher label (e.g. 50/50 0 and 1, becomes 1).
        vid_preds = [max(multimode(all_unique_video_res[vid_id][1])) for vid_id in all_unique_video_res.keys()]
        vid_targets_unique = [v[0] for v in all_unique_video_res.values()]  # Single target for each unique video
        if get_clf_report:
            # Classification report on a frame-level
            get_save_classification_report(targets, preds, f'{subset}_report_{run_name}.csv',
                                           metric_res_dir=out_dir, epochs=epochs)
            # Classification report on a video-level
            get_save_classification_report(vid_targets_unique, vid_preds, f'{subset}_report_video_{run_name}.csv',
                                           metric_res_dir=out_dir, epochs=epochs)
        if get_confusion:
            get_save_confusion_matrix(targets, preds, f'{subset}_cm_{run_name}.csv', metric_res_dir=out_dir)
            get_save_confusion_matrix(vid_targets_unique, vid_preds, f'{subset}_cm_video_{run_name}.csv',
                                      metric_res_dir=out_dir)

    # ROC_AUC Plotting
    if not args.multi_class:
        if first:  # Plot random baseline, only 1x
            p_fpr, p_tpr, _ = roc_curve(targets, [0 for _ in range(len(targets))], pos_label=1)
            plt.plot(p_fpr, p_tpr, linestyle='--', color='blue', label='random')
        # Get ROC_AUC plot on a video-level, with thresholds referring to probability of frames
        run_label = out_name if out_name is not None else run_name[-25:]
        fpr1, tpr1, thresh1 = roc_curve(vid_targets, avg_softm_probs, pos_label=1, drop_intermediate=False)
        plt.plot(fpr1, tpr1, color=col, label=run_label)

    # Correct res string format, depending on out type
    ret = [] if args.out_type == 'csv' else ''
    for metric_values in metric_dict.values():
        mean = np.mean(metric_values)
        std = np.std(metric_values)
        if args.out_type == 'csv':
            metric_str = f'{mean:.2f} (std: {std:.2f})'
            ret.append(metric_str)
        else:  # latex string
            metric_str = f'& {mean:.2f} $\pm {std:.2f}$'
            ret = ret + ' ' + metric_str
    return ret


def main():
    res_dir = args.res_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    if args.cr:
        os.makedirs(os.path.join(out_dir, 'classification_reports'), exist_ok=True)
    if res_dir is not None: # order by run names
        all_runs = os.listdir(res_dir)
        all_runs = sorted([run for run in all_runs if os.path.isdir(os.path.join(res_dir, run))])
    else:
        all_runs = args.run_paths
    num_runs = len(all_runs)
    val_data = [[] for _ in range(num_runs)]  # list of lists, for each run
    train_data = [[] for _ in range(num_runs)]  # list of lists, for each run
    colorMap = plt.get_cmap('jet', num_runs)
    # Get metrics for each run
    for i, run_name in enumerate(all_runs):
        col = colorMap(i/num_runs)
        out_name = None if args.out_names is None else args.out_names[i]
        res = get_metrics_for_run(res_dir, run_name, out_dir, col, get_clf_report=args.cr, get_confusion=args.cm,
                                  first=(i == 0), out_name=out_name)
        val_data[i] = res
        if args.train:
            res_train = get_metrics_for_run(res_dir, run_name, out_dir, col, subset='train', get_clf_report=args.cr,
                                            get_confusion=args.cm, out_name=out_name)
            train_data[i] = res_train
    if args.out_names:
        df_names = args.out_names
    else:
        df_names = [os.path.basename(run) for run in all_runs]
    if not args.only_plot: # Save results from all runs in one file
        if args.out_type == 'csv':
            df = pd.DataFrame(val_data, index=df_names, columns=metric_list)
        else:  # args.out_type == 'latex'
            df = pd.DataFrame(val_data, index=df_names)
        if args.train:
            if args.out_type == 'csv':
                df_train = pd.DataFrame(train_data, index=df_names, columns=metric_list)
            else:  # args.out_type == 'latex'
                df_train = pd.DataFrame(train_data, index=df_names)
            df = pd.concat([df, df_train], keys=['val', 'train'], axis=1)
        df.to_csv(os.path.join(out_dir, 'summary.csv'), float_format='%.2f')

    # Plotting -> Finalise roc_auc curve
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    title = ' '.join(args.plot_title) if args.plot_title is not None else ''
    plt.title(title)
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(out_dir, 'val_roc_auc_curve'))


if __name__ == "__main__":
    args = parser.parse_args()
    # Maybe have the metric_list as a parameter - depends on the use-case

    # metric_list = ['Video ROC_AUC (weighted)', 'Video F1 (weighted)', 'Video P (weighted)', 'Video R (weighted)',
    #                'Video bACC', 'Video CI', 'Video Corr CI', 'Video Wrong CI']
    metric_list = ['Video ROC_AUC (weighted)', 'Video F1 (weighted)', 'Video P (weighted)', 'Video R (weighted)',
                   'Video bACC', 'Video CI']
    main()









