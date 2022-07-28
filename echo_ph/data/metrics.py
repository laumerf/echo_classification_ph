import os
import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, roc_auc_score, classification_report, \
    confusion_matrix, mean_squared_error, precision_score, recall_score
import pandas as pd
import csv
from statistics import multimode


# ==== Functions related to Metrics or Results
def get_metric_dict(targets, preds, probs=None, binary=True, subset='', prefix='', tb=True, regression=False,
                    conf=None):
    """
    Get dictionary of metrics (f1, accuracy, balanced accuracy, roc_auc) and associated values
    :param targets: Targets / labels
    :param preds: Model predictions
    :param probs: Model soft-maxed probabilities for class PH (if binary classification), else None
    :param binary: Set to true if binary classification.
    :param subset: 'valid' or 'train', if it should be specified in metric directory
    :param prefix: If any prefix, in front of all metric-keys in directory (e.g. video-)
    :param tb: Set to true, if calculating metrics for tensorboard during training (has different metric keys)
    :param regression: Set to true, if add regression metrics
    :param conf: Video confidence (ratio of samples agreeing on correct label for a video)
    :return: Metrics directory with various metrics.
    """
    b_acc = balanced_accuracy_score(targets, preds)
    acc = accuracy_score(targets, preds)
    if tb:  # Metric keys for tensor-board (i.e. during training)
        metrics = {prefix + 'f1' + '/' + subset: f1_score(targets, preds, average='micro'),
                   prefix + 'accuracy' + '/' + subset: acc,
                   prefix + 'b-accuracy' + '/' + subset: b_acc,
                   }
    else:  # Metric keys for eval csv files
        if prefix.startswith('Video'):
            metrics = {prefix + 'F1 (micro)': f1_score(targets, preds, average='micro'),
                       prefix + 'F1 (macro)': f1_score(targets, preds, average='macro'),
                       prefix + 'F1 (weighted)': f1_score(targets, preds, average='weighted'),
                       prefix + 'bACC': b_acc,
                       prefix + 'ACC': acc,
                       prefix + 'P (micro)': precision_score(targets, preds, average='micro'),
                       prefix + 'P (macro)': precision_score(targets, preds, average='macro'),
                       prefix + 'P (weighted)': precision_score(targets, preds, average='weighted'),
                       prefix + 'R (micro)': recall_score(targets, preds, average='micro'),
                       prefix + 'R (macro)': recall_score(targets, preds, average='macro'),
                       prefix + 'R (weighted)': recall_score(targets, preds, average='weighted'),
                       prefix + 'CI':  np.mean(conf)}
            if binary:
                metrics.update(
                    {prefix + 'F1, pos': f1_score(targets, preds, average='binary'),
                     prefix + 'F1, neg': f1_score(targets, preds, pos_label=0, average='binary')}
                )
        else:  # For the Frame-wise, only report balanced accuracy.
            metrics = {prefix + 'bACC': b_acc}

    if probs is not None:  # and binary:  # Also get ROC_AUC score on probabilities, for binary classification
        if binary:
            roc_auc = roc_auc_score(targets, probs, average="macro")
            roc_auc_w = roc_auc_score(targets, probs, average="weighted")
        else:
            roc_auc = roc_auc_score(targets, probs, multi_class="ovo", average="macro")
            roc_auc_w = roc_auc_score(targets, probs, multi_class="ovo", average="weighted")
        if tb:
            metrics.update({prefix + 'roc_auc' + '/' + subset: roc_auc})
        else:
            metrics.update({prefix + 'ROC_AUC (macro)': roc_auc})
            metrics.update({prefix + 'ROC_AUC (weighted)': roc_auc_w})
    if regression:
        metrics.update({prefix + 'mse' + '/' + subset: mean_squared_error(targets, preds)})
    return metrics


def read_results(res_dir, subset='val'):
    """
    Read (get) results for model (preds, targets, samples) from numpy files
    :param res_dir: directory of model results
    :param subset: train or val
    :return: list of model predictions, softmax probabilities, targets, sample names, and raw model outputs
    """
    outs = np.load(os.path.join(res_dir, f'{subset}_preds.npy'))
    targets = np.load(os.path.join(res_dir, f'{subset}_targets.npy'))
    samples = np.load(os.path.join(res_dir, f'{subset}_samples.npy'))
    sm = torch.nn.Softmax(dim=-1)
    if len(outs) == 0:
        return None, None, None, None
    if isinstance(outs[0], (list, np.ndarray)):
        preds = np.argmax(outs, axis=1)
        soft_m = np.asarray(sm(torch.tensor(outs)))  # get soft-maxed prob corresponding to class 1
        probs = soft_m[:, 1]
    else:
        preds = outs
        probs = None
    return preds, probs, targets, samples, outs


def get_save_classification_report(targets, preds, file_name, metric_res_dir='results', epochs=None):
    """
    Get classification report for the given targets and predictions, and save it.
    Furthermore, save the epoch that the model stopped training, if epochs is specified.
    :param targets: Ground truth labels
    :param preds: Model predicted labels
    :param file_name: Name of the resulting file
    :param metric_res_dir: Name of the base result directory
    :param epochs: List of epochs per fold, if desired to save this info. Else None
    """
    report = classification_report(targets, preds, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df['support'] = df['support'].astype(int)
    cr_dir = os.path.join(metric_res_dir, 'classification_reports')
    os.makedirs(cr_dir, exist_ok=True)
    file_name = os.path.join(cr_dir, file_name)
    df.to_csv(file_name, float_format='%.2f')
    if epochs is not None:  # Add also epochs info
        with open(file_name, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([])
            writer.writerow(['epochs'] + epochs)


def get_save_confusion_matrix(targets, preds, file_name, metric_res_dir='results'):
    """
    Get classification report for the given targets and predictions, and save it.
    Columns are Predicted labels, Rows are True labels.
    :param targets: Ground truth labels
    :param preds: Model predicted labels
    :param file_name: Name of the resulting file
    :param metric_res_dir: Name of the base result directory
    """
    cm_dir = os.path.join(metric_res_dir, 'confusion_matrix')
    os.makedirs(cm_dir, exist_ok=True)
    file_name = os.path.join(cm_dir, file_name)

    res = confusion_matrix(targets, preds)
    df = pd.DataFrame(res)
    df.to_csv(file_name)


class Metrics():
    def __init__(self, targets, samples, model_outputs=None, preds=None, sm_probs=None, binary=True, tb=True,
                 regression=False):
        """
        :param targets: The targets / labels. Shape: (num_samples)
        :param samples: The sample names. Shape: (num_samples)
        :param model_outputs: (Optional) The raw model outputs, i.e., un-normalized scores (logits), before softmax or
                              sigmoid. If model_outputs is not provided, must provide preds and sm_probs.
                              Shape: (num_samples, num_classes)
        :param preds: (Optional) The model's actual predictions (arg-maxed output)
        :param sm_probs: (Optional) The model's soft-maxed probabilities, for the class of interest (PH).
                                    If not set, it will be calculated from model outputs.)
        :param binary: If binary classification
        :param tb: Set to true if get these scores for tensorboard (metric dict looks different than during eval)
        """
        if preds is None and sm_probs is None:
            if model_outputs is None:
                print("Must provide model outputs, if preds and/or probs not provided")
                return
        self.model_outputs = model_outputs
        self.targets = targets
        self.samples = samples
        self.soft_max = torch.nn.Softmax(dim=-1)
        self.preds = preds
        self.sm_probs = sm_probs
        self.binary = binary
        self.regression = regression
        if not self.binary:  # For multi-class prediction, calculate sm_probs in this class.
            self.sm_probs = None
        self.tb = tb

        # Video-wise lists that will get instantiated when and if needed (or if already provided, then use that)
        self.video_targets = None
        self.video_preds = None
        self.mean_probs_per_video = None
        self.video_outs = None
        self.video_confidence = None
        self.video_corr_confidence = None
        self.video_wrong_confidence = None
        self.video_ids = None

    def get_softmax_probs(self):
        """
        Get softmax probabilities from model outputs (logits). In case of binary classification, get the probs
        for the true class (PH), but in case of multi-class classification, get probs for each class.
        """
        out = self.model_outputs
        if not torch.is_tensor(self.model_outputs):
            out = torch.Tensor(out)
        if self.binary:
            self.sm_probs = self.soft_max(out)[:, 1]
        else:
            self.sm_probs = self.soft_max(out)
        self.sm_probs = [prob.cpu().detach().numpy() for prob in self.sm_probs]
        return self.sm_probs

    def get_preds(self):
        """
        Get model predictions, given its raw outputs (logits)
        """
        if torch.is_tensor(self.model_outputs):
            self.preds = torch.argmax(self.model_outputs, dim=1)
        else:
            self.preds = np.argmax(self.model_outputs, axis=1)
        return self.preds

    def get_per_sample_scores(self, subset=''):
        """
        Get scores (metrics) per sample / frame. Metrics calculated are f1-score, accuracy, balanced accuracy,
        and ROC_AUC score.
        :param subset: 'valid' or 'train' => for output dictionary
        :return: metrics dictionary, where key is metric and value is results / score.
        """
        if self.preds is None:
            self.get_preds()
        # Also get ROC_AUC score on probabilities, for binary classification
        if self.sm_probs is None:
            self.get_softmax_probs()
        prefix = '' if self.tb else 'Frame '
        return get_metric_dict(self.targets, self.preds, probs=self.sm_probs, binary=self.binary,
                               prefix=prefix, subset=subset, tb=self.tb, regression=self.regression)

    def get_per_subject_scores(self, subset=''):
        """
        Get scores (metrics) per subject / video - averaging over all the frames for that video.
        Metrics calculated are f1-score, accuracy, balanced accuracy, and ROC_AUC score.
        :param subset: 'valid' or 'train' => for output dictionary
        :return: metrics dictionary, where key is metric and value is results / score.
        """
        if self.video_targets is None:
            self._set_subject_res_lists()
        prefix = 'video-' if self.tb else 'Video '
        return get_metric_dict(self.video_targets, self.video_preds, probs=self.mean_probs_per_video, binary=self.binary,
                               subset=subset, prefix=prefix, tb=self.tb, regression=self.regression,
                               conf=self.video_confidence)

    def get_subject_lists(self, raw_outputs=False):
        """
        Get and return subject-wise: targts, predictions, average soft-maxed predictions, total confidence,
        confidence of corr predictions, confidence of wrong predictions, video ids, and optionally model outputs.
        :param raw_outputs: Set to true, to also return raw video/subject outputs (not often desired)
        :return: (targets, preds, mean_prob, confidence, corr conf, wrong conf, ids, optionally output) => all per-video
        """
        if self.video_targets is None:
            self._set_subject_res_lists()
        ret = (self.video_targets, self.video_preds, self.mean_probs_per_video, self.video_confidence,
               self.video_corr_confidence, self.video_wrong_confidence, self.video_ids)
        if raw_outputs:
            ret = ret + (self.video_outs,)
        return ret

    def _get_video_dict(self):
        """
        Helper function for getting subject-wise scores. Creates a video dictionary,
        collecting predictions for all frames of a given video.

        :return: Video dict, with video-id as key, and the value is another dict of predictions, targets, and possibly
                 soft-maxed probabilities for this video.
        """
        res_per_video = {}
        for i in range(len(self.samples)):
            vid_id = self.samples[i].split('_')[0]
            target = self.targets[i]
            pred = self.preds[i]
            if self.model_outputs is None:
                out = None
            else:
                out = self.model_outputs[i]
            if vid_id in res_per_video:
                res_per_video[vid_id]['pred'].append(pred)
                if out is not None:
                    res_per_video[vid_id]['out'].append(out)
                res_per_video[vid_id]['prob'].append(self.sm_probs[i])
            else:
                # (target, list of preds, list of raw outs)
                out_lst = [out] if out is not None else None
                res_per_video[vid_id] = {'target': target, 'pred': [pred], 'out': out_lst}
                res_per_video[vid_id].update({'prob': [self.sm_probs[i]]})  # update dict with sm probs
        return res_per_video

    def _set_subject_res_lists(self):
        """
        Get list of targets per subject, predictions per subject (majority vote of frames),
        and avg soft-maxed prob per subject.
        :return:
        """
        if self.preds is None:
            self.get_preds()
        if self.sm_probs is None:
            self.get_softmax_probs()
        res_per_video = self._get_video_dict()
        targets_per_video = []
        preds_per_video = []
        video_confidance = []
        video_corr_confidence = []
        video_wrong_confidence = []
        raw_outs_per_video = None if self.model_outputs is None else []
        mean_probs_per_video = []
        for res in res_per_video.values():
            # Pick the most frequent label for the video (works with binary or multi-labels).
            video_pred = max(multimode(res['pred']))  # In case of a tie, pick the higher label (more PH)
            ratio_corr_pred = res['pred'].count(video_pred) / len(res['pred'])  # Count(corr_pred)/ Total len
            if video_pred == res['target']:
                video_corr_confidence.append(ratio_corr_pred)
            else:
                video_wrong_confidence.append(ratio_corr_pred)
            video_confidance.append(ratio_corr_pred)
            preds_per_video.append(video_pred)
            targets_per_video.append(res['target'])
            if self.model_outputs is not None:
                raw_outs_per_video.append(np.average(res['out'], axis=0))
            if self.binary:
                mean_probs_per_video.append(np.mean(res['prob']))
            else:
                mean_probs_per_video.append(np.mean(res['prob'], axis=0))  # 1 prob for each class, but avg. over frames
        self.video_targets = targets_per_video
        self.video_preds = preds_per_video
        self.mean_probs_per_video = mean_probs_per_video
        self.video_outs = raw_outs_per_video
        self.video_confidence = video_confidance
        self.video_corr_confidence = video_corr_confidence
        self.video_wrong_confidence = video_wrong_confidence
        self.video_ids = list(res_per_video.keys())




