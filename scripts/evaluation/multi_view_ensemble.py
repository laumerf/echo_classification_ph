from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from echo_ph.evaluation.metrics import read_results, Metrics, get_metric_dict
import os
import numpy as np
import random
from statistics import multimode

"""
This script performs majority vote (ensemble) on models from different views, and reports the results.
Also possible to do weighted average of output probabilities, as a comparison.
"""

parser = ArgumentParser(
    description='Get majority vote results of different views',
    formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('base_dir', help='Path to the base directory where model files are stored')
parser.add_argument('--res_files', help='Path to the results file of kapap view', nargs='+')
parser.add_argument('--views', help='Path to the results file of kapap view', nargs='+')
parser.add_argument('--disagree_method', default='conf', choices=['conf', 'random', 'max', 'psax'],
                    help='What method to use to decide on a prediction when all models disagree')
parser.add_argument('--prob_method', default='conf', choices=['mean', 'conf'],
                    help='What method to use to decide on how to aggregate mean probabilities.'
                         'Mean = take mean prob of all models selected'
                         'Conf = take the prob of the most confident model, of those selected')
parser.add_argument('--no_folds', type=int, default=10, help='Number of folds')
parser.add_argument('--verbose', action='store_true',
                    help='set this flag to have more print statements')
parser.add_argument('--method', default='mv', choices=['mv', 'avg'],
                    help='What method to use to join view predictions. Default is majority vote (mv), but'
                         'can also choose average of outputs weighted by conf (avg)')

SEED = 0
np.random.seed(SEED)
random.seed(SEED)


def weighted_avg(conf_views, mean_probs):
    """
    Get multi view joint prediction, based on weighted average of output probabilities
    :param conf_views: List of confidence for current subject, for all views.
    :param mean_probs: List of mean model probabilities for current subject, for all views.
    :return: The averaged prediction, associated confidence, and probabilities, as well as num models agreeing
    """
    indices = np.where(np.logical_not(np.isnan(conf_views)))[0]
    joined_prob = np.average(mean_probs[indices], axis=0, weights=conf_views[indices])
    joined_pred = np.argmax(joined_prob)
    joined_conf = np.average(conf_views[indices])
    all_models_disagree = False
    return joined_pred, joined_conf, joined_prob, all_models_disagree


def majority_vote(pred_views, conf_views, mean_probs, disagree_method='conf', prob_method='conf'):
    """
    :param pred_views: List of prediction for current subject, for all views.
    :param conf_views: List of confidence for current subject, for all views.
    :param mean_probs: List of mean model probabilities for current subject, for all views.
    :param disagree_method: Method to use to select a prediction when all models disagree. Choices: random, conf, max.
                   random: Choose random prediction,
                   conf: Choose prediction associated to most confident model,
                   max: Choose the higher prediction (i.e. more severe)
    :param prob_method: Method to use to aggregate model probabilities, for ROC_AUC score.
                    mean: Take mean output prob, of all selected model.
                    conf: Take the probabilities of the most confident model of the selected ones.
    :return: The majority vote prediction, associated confidence, and probabilities, as well as num models agreeing
    """
    mv_pred = multimode(pred_views)  # majority vote pred is set to most common class(es) predictions
    num_views = len(pred_views)
    no_unique_preds = len(set(pred_views))
    all_models_disagree = (num_views == no_unique_preds)
    # Only one label is the most common, select this one
    if len(mv_pred) == 1:
        mv_pred = mv_pred[0]
    else:  # Else majority vote gives a tie, pick the results of the most confident model, or acc. to selected strategy
        if disagree_method == 'conf':  # Default approach
            print(conf_views)
            best_idx = np.nanargmax(conf_views)  # max confidence
            print(best_idx)
        elif disagree_method == 'random':
            best_idx = np.random.randint(num_views)
            while np.isnan(pred_views[best_idx]):
                best_idx = np.random.randint(num_views)
        elif disagree_method == 'max':
            best_idx = np.nanargmax(pred_views)  # max pred. value
        else:  # First view, KAPAP (Make sure to give in this order)
            if not np.isnan(pred_views[0]):
                best_idx = 0
            else:
                best_idx = np.nanargmax(pred_views)  # max pred. value
        mv_pred = pred_views[best_idx]

    selected_indexes = np.argwhere(pred_views == mv_pred).squeeze(1)  # Idx of all views that predict this
    selected_conf = conf_views[selected_indexes]
    if prob_method == 'conf':
        max_conf_idx = np.nanargmax(selected_conf)
        view_idx = selected_indexes[max_conf_idx]
        mv_mean_probs = mean_probs[view_idx]
    else:
        mv_mean_probs = np.nanmean(mean_probs[selected_indexes], axis=0)
    mv_conf = np.nanmean(selected_conf)
    return mv_pred, mv_conf, mv_mean_probs, all_models_disagree


# def majority_vote2(pred_views, conf_views, mean_probs, disagree_method='conf', prob_method='conf'):
#     """
#     :param pred_views: List of prediction for current subject, for all views.
#     :param conf_views: List of confidence for current subject, for all views.
#     :param mean_probs: List of mean model probabilities for current subject, for all views.
#     :param disagree_method: Method to use to select a prediction when all models disagree. Choices: random, conf, max.
#                    random: Choose random prediction,
#                    conf: Choose prediction associated to most confident model,
#                    max: Choose the higher prediction (i.e. more severe)
#     :param prob_method: Method to use to aggregate model probabilities, for ROC_AUC score.
#                     mean: Take mean output prob, of all selected model.
#                     conf: Take the probabilities of the most confident model of the selected ones.
#     :return: The majority vote prediction, associated confidence, and probabilities, as well as num models agreeing
#     """
#     mv_pred = multimode(pred_views)  # majority vote pred is set to most common class(es) predictions
#     selected_indexes = []  # Indexes of all views that give the 'selected' prediction (i.e. the majority vote pred)
#     for mv_p in mv_pred:
#         b_idx = np.argwhere(pred_views == mv_p).squeeze(1)
#         selected_indexes.extend(b_idx)  # [0, 2]
#     mv_conf = conf_views[selected_indexes]  # Confidence of the selected views, 0 2
#     if prob_method == 'conf':
#         # Pick the probs of the most confident view, of the 'selected' views:
#         max_conf_idx = np.nanargmax(mv_conf)
#         view_idx = selected_indexes[max_conf_idx]
#         mv_mean_probs = mean_probs[view_idx]
#     else:
#         mv_mean_probs = np.nanmean(mean_probs[selected_indexes], axis=0)
#     mv_conf = np.nanmean(mv_conf)
#     num_views = len(pred_views)
#     no_unique_preds = len(set(pred_views))
#     all_models_disagree = (num_views == no_unique_preds)
#
#     # Only one label is the most common, select this one
#     if len(mv_pred) == 1:
#         return mv_pred[0], mv_conf, mv_mean_probs, all_models_disagree
#     # Else majority vote gives a tie, pick the results of the most confident model, or acc. to selected strategy
#     if disagree_method == 'conf':  # Default approach
#         print(conf_views)
#         best_idx = np.nanargmax(conf_views)  # max confidence
#         print(best_idx)
#     elif disagree_method == 'random':
#         best_idx = np.random.randint(num_views)
#         while np.isnan(pred_views[best_idx]):
#             best_idx = np.random.randint(num_views)
#     elif disagree_method == 'max':
#         best_idx = np.nanargmax(pred_views)  # max pred. value
#     else:  # First view, KAPAP (Make sure to give in this order)
#         if not np.isnan(pred_views[0]):
#             best_idx = 0
#         else:
#             best_idx = np.nanargmax(pred_views)  # max pred. value
#     mv_pred = pred_views[best_idx]
#     mv_conf = conf_views[best_idx]
#     mv_mean_probs = mean_probs[best_idx]
#     return mv_pred, mv_conf, mv_mean_probs, all_models_disagree


def main():
    args = parser.parse_args()
    binary = True if 'binary' in args.base_dir else False
    no_folds = args.no_folds
    res_files = args.res_files  # [args.res_file_kapap, args.res_file_cv, args.res_file_la]
    res_paths = [os.path.join(args.base_dir, res_file) for res_file in res_files]
    views = args.views
    # define metrics of interest
    all_res = {key: [] for key in ['Video ROC_AUC (weighted)', 'Video F1 (weighted)', 'Video P (weighted)',
                                   'Video R (weighted)', 'Video bACC', 'Video CI']}
    cnt_all_disagree = 0
    for fold in range(0, no_folds):
        subj_pred_all_views = {}  # {'818': {'kapap':(pred, ci), 'kaap': (pred, ci)'}}
        subj_targ_all_views = {}  # {'818': 2, '128': 0, ... }
        joint_subj_preds = []
        joint_subj_targets = []
        joint_subj_conf = []
        joint_subj_probs = []
        for res_path, view in zip(res_paths, views):
            fold_path = sorted(os.listdir(res_path))[fold]
            fold_dir = os.path.join(res_path, fold_path)
            print(fold_dir)
            fold_preds, fold_probs, fold_targets, fold_samples, fold_outs = read_results(fold_dir)
            m = Metrics(fold_targets, fold_samples, model_outputs=fold_outs, preds=fold_preds,
                        binary=binary, tb=False)
            subj_targets, subj_preds, subj_mean_probs, subj_confs, corr_conf, wrong_conf, subj_ids, subj_outs = \
                m.get_subject_lists(raw_outputs=True)
            for subj_id, subj_t, subj_p, subj_mp, subj_ci, subj_out in zip(subj_ids, subj_targets, subj_preds,
                                                                           subj_mean_probs, subj_confs, subj_outs):
                # Match video-ids from different views
                if subj_id in subj_pred_all_views:
                    subj_pred_all_views[subj_id][view] = (subj_p, subj_ci, subj_out, subj_mp)
                else:
                    subj_pred_all_views[subj_id] = {}
                    subj_pred_all_views[subj_id][view] = (subj_p, subj_ci, subj_out, subj_mp)
                if subj_id not in subj_targ_all_views:
                    subj_targ_all_views[subj_id] = subj_t

        for key in subj_pred_all_views.keys():
            # Initialise as None, will be filled with correct info if exists
            'Not all views available for all videos'
            preds_all_views = []
            conf_all_views = []
            mean_prob_all_views = []
            for view in views:
                if view in subj_pred_all_views[key]:
                    pred, conf, _, mp = subj_pred_all_views[key][view]
                else:
                    pred = conf = np.nan
                    mp = np.nan if binary else np.full(3, np.nan)
                preds_all_views.append(pred)
                conf_all_views.append(conf)
                mean_prob_all_views.append(mp)

            # Calculate majority vote for the current model
            target = subj_targ_all_views[key]
            if args.method == 'mv': # majority vote
                joint_pred, joint_ci, joint_prob, all_disagree = majority_vote(np.asarray(preds_all_views),
                                                                           np.asarray(conf_all_views),
                                                                           np.asarray(mean_prob_all_views),
                                                                           disagree_method=args.disagree_method,
                                                                           prob_method=args.prob_method)
            else:  # weighted average
                joint_pred, joint_ci, joint_prob, all_disagree = weighted_avg(np.asarray(conf_all_views),
                                                                               np.asarray(mean_prob_all_views))
            if all_disagree:
                cnt_all_disagree += 1
            if args.verbose and all_disagree and joint_pred != target:
                print(f'All models wrong for: subj id {key}')
                print(f't={target}, p={joint_pred}, all_preds: {preds_all_views}, all confs: {conf_all_views}')

            joint_subj_preds.append(joint_pred)
            joint_subj_conf.append(joint_ci)
            joint_subj_targets.append(subj_targ_all_views[key])  # target is always the same
            joint_subj_probs.append(joint_prob)

        subj_res = get_metric_dict(joint_subj_targets, joint_subj_preds, probs=joint_subj_probs, binary=binary,
                                   subset='val', prefix='Video ', tb=False, conf=joint_subj_conf)
        for metric, val in subj_res.items():
            if metric in all_res:
                all_res[metric].append(val)

    if args.verbose:
        print('no. videos where all disagree (aggregated over all folds)', cnt_all_disagree)

    latex_metric_str = ''
    for metric, metric_values in all_res.items():
        mean = np.mean(metric_values)
        std = np.std(metric_values)
        metric_str = f'{mean:.2f} (std: {std:.2f})'
        print(metric, ":", metric_str)
        latex_metric_str += f'& {mean:.2f} $\pm {std:.2f}$'
    print("Latex results")
    print(latex_metric_str)


if __name__ == '__main__':
    main()
