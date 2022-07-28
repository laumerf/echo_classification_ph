import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from echo_ph.models.resnet_3d import Res3DMultiView, Res3DSaliency, Res3DAttention, get_resnet3d_50, get_resnet3d_18
from echo_ph.models.resnets import get_resnet18, ResMultiView
from echo_ph.models.conv_nets import ConvNet, SimpleConvNet

# Various functions that are used multiple times throughout the project, and don't belong to a specific class.


def get_index_file_path(no_folds, curr_fold, label_type, train=True):
    """
    Get index file, with the samples to use for the given fold for given train/valid set.
    :param no_folds: How many folds in total
    :param curr_fold: What is the current fold number
    :param label_type: 3class, 2class, etc.
    :param train: If this is for training set.
    :return: Path to the correct index file.
    """
    idx_dir = 'index_files' if no_folds is None else os.path.join('index_files', 'k' + str(no_folds))
    idx_file_end = '' if curr_fold is None else '_' + str(curr_fold)
    idx_file_base_name = 'train_samples_' if train else 'valid_samples_'
    index_file_path = os.path.join(idx_dir, idx_file_base_name + label_type + idx_file_end + '.npy')
    return index_file_path


def get_temp_model(model_type, num_classes, pretrained, device, views=None,
                   self_att=False, map_att=False, size=256, cl=12, return_last_saliency=True, join_method='sum'):
    """
    Get the correct model based on given parameters, for the temporal case.
    :param model_type: Model type
    :param num_classes: Number of classes
    :param pretrained: True/False
    :param device: Device
    :param views: List of views for training this model.
    :param self_att: Only relevant for attention-based model types.
    :param map_att: Only relevant for attention-based model types.
    :param size: Input frame size
    :param cl: clip length
    :param return_last_saliency: True if return last-layer from saliency model.
    :param join_method: What method to use to join features, inc ase of multi-view model. Options: 'sum', 'concat'
    :return: Model
    """
    if model_type == 'r3d_18_multi_view':
        model = Res3DMultiView(device, num_classes=num_classes, pretrained=pretrained, views=views,
                               join_method=join_method)
    elif model_type == 'saliency_r3d_18':
        model = Res3DSaliency(num_classes=num_classes, pretrained=pretrained, return_last=return_last_saliency)
    elif model_type.endswith('18'):
        if self_att or map_att:
            att_type = 'self' if self_att else 'map'
            model = Res3DAttention(num_classes=num_classes, ch=1, w=size, h=size, t=cl,
                                   att_type=att_type, pretrained=pretrained)
        else:
            model = get_resnet3d_18(num_classes=num_classes, pretrained=pretrained,
                                    model_type=model_type)
    else:  # This is really slow-fast network (TODO: refactor naming)
        model = get_resnet3d_50(num_classes=num_classes, pretrained=pretrained)
    return model.to(device)


def get_spatial_model(model_type, num_classes, pretrained, views, device, dropout, join_method='sum'):
    """
    Get the correct model based on given parameters, for the spatial case.
    :param model_type: Model type
    :param num_classes: Number of classes.
    :param pretrained: True/False
    :param views: List of views for training this model.
    :param device: Device
    :param dropout: If any dropout
    :param join_method: join_method
    :return:
    """
    if model_type == 'resnet':
        model = get_resnet18(num_classes=num_classes, pretrained=pretrained).to(device)
    elif model_type == 'conv':
        model = ConvNet(num_classes=num_classes, dropout_val=dropout).to(device)
    elif model_type == 'resnet2d_multi_view':
        model = ResMultiView(device, num_classes=num_classes, pretrained=pretrained, views=views,
                             join_method=join_method).to(device)
    else:
        model = SimpleConvNet(num_classes=num_classes).to(device)
    return model


def set_arg_parse_all(description, regression=False):
    parser = ArgumentParser(
        description=description,
        formatter_class=ArgumentDefaultsHelpFormatter)
    # Paths, file name, model names, etc
    parser.add_argument('--videos_dir', default=None,
                        help='Path to the directory containing the raw videos - if work on raw videos')
    parser.add_argument('--cache_dir', default=None,
                        help='Path to the directory containing the cached and processed numpy videos - if work on those')
    parser.add_argument('--label_type', default='2class_drop_ambiguous', choices=['2class', '2class_drop_ambiguous',
                                                                                  '3class', '3class_2', '3class_3',
                                                                                  '4class'],
                        help='How many classes for the labels, and in some cases also variations of dropping ambiguous '
                             'labels. Will be used to fetch the correct label file and train and valid index files')
    parser.add_argument('--fold', default=None, type=int,
                        help='In case of k-fold cross-validation, set the current fold for this training.'
                             'Will be used to fetch the relevant k-th train and valid index file')
    parser.add_argument('--k', default=None, type=int,
                        help='In case of k-fold cross-validation, set the k, i.e. how many folds all in all.')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Set name of the model you want to load or train. If None, use the model name as assigned'
                             'by the function get_run_name(), using selected arguments, and optionally unique_run_id')
    parser.add_argument('--run_id', type=str, default='',
                        help='Set a unique_run_id, to identify run if args alone are not enough to identify.'
                             'Default is empty string, i.e. only identify run with arguments.')
    parser.add_argument('--view', nargs='+', type=str, default=['KAPAP'], help='What view (s) to use')
    # choices: 'KAPAP', 'CV', 'KAAP', 'LA', 'KAKL' )

    # Data parameters
    parser.add_argument('--scaling_factor', default=0.25,
                        help='How much to scale (down) the videos, as a ratio of original '
                             'size. Also determines the cache sub-folder')
    parser.add_argument('--img_size', default=224, type=int, help='Size of images (frames) to resize to')
    parser.add_argument('--num_workers', type=int, default=4, help='The number of workers for loading data')
    parser.add_argument('--max_p', type=float, default=90, help='Percentile for max expansion frames')
    parser.add_argument('--min_expansion', action='store_true',
                        help='Percentile for min expansion frames instead of maximum')
    parser.add_argument('--num_rand_frames', type=int, default=None,
                        help='If pick random frames per video (instead of frames corresponding to max/min expansion), '
                             'set the number of frames per video.')
    parser.add_argument('--augment', action='store_true',
                        help='set this flag to apply ALL augmentation transformations to training data')
    parser.add_argument('--aug_type', type=int, default=4,
                        help='What augmentation type to use')

    # Class imbalance
    parser.add_argument('--class_balance_per_epoch', action='store_true',
                        help='set this flag to have ca. equal no. samples of each class per epoch / oversampling')
    parser.add_argument('--weight_loss', action='store_true',
                        help='set this flag to weight loss, according to class imbalance')
    # Training & models parameters
    parser.add_argument('--load_model', action='store_true',
                        help='Set this flag to load an already trained model to predict only, instead of training it.'
                             'If args.model_name is set, load model from that path. Otherwise, get model name acc. to'
                             'function get_run_name(), and load the corresponding model')
    parser.add_argument('--model', default='resnet', choices=['resnet', 'resnet2d_multi_view', 'conv',
                                                              'simple_conv', 'r2plus1d_18', 'mc3_18', 'r3d_18',
                                                              'r3d_18_multi_view', 'r3d_50', 'saliency_r3d_18'],
                        help='What model architecture to use. Note: r3d_50 is actually slow_fast (!)')
    parser.add_argument('--join_method', default='sum', choices=['sum', 'concat'],
                        help='What method to use to join features, relevant only in case of multi-view model.')
    parser.add_argument('--self_attention', action='store_true', help='If use self-attention (non-local block)')
    parser.add_argument('--map_attention', action='store_true', help='If use map-based attention')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout value for those model who use dropout')
    parser.add_argument('--optimizer', default='adamw', choices=['adam', 'adamw'], help='What optimizer to use.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--max_epochs', type=int, default=350, help='Max number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--wd', type=float, default=None,
                        help='Weight decay value. Currently only used in conjunction with '
                             'selecting adamw optimizer. The default for adamw is 1e-2, '
                             '(0.02), when using lr 1e-3 (0.001)')
    parser.add_argument('--decay_factor', type=float, default=0.0, help='Decay lr by this factor for decay on plateau')
    parser.add_argument('--decay_patience', type=int, default=1000,
                        help='Number of epochs to decay lr for decay on plateau')
    parser.add_argument('--min_lr', type=float, default=0.0, help='Min learning rate for reducing lr')
    parser.add_argument('--cooldown', type=float, default=0, help='cool-down for reducing lr on plateau')
    parser.add_argument('--early_stop', type=int, default=100,
                        help='Patience (in no. epochs) for early stopping due to no improvement of valid f1 score')
    parser.add_argument('--pretrained', action='store_true', help='Set this flag to use pre-trained resnet')
    parser.add_argument('--eval_metrics', type=str, default=['video-b-accuracy/valid'], nargs='+',
                        help='Set this the metric you want to use for early stopping - you can choose multiple metrics.'
                             'Choices: f1/valid, loss/valid, b-accuracy/valid, video-f1/valid, video-b-accuracy/valid, '
                             'video-roc_auc/valid')

    # General parameters
    parser.add_argument('--debug', action='store_true',
                        help='set this flag when debugging, to not connect to wandb, etc')
    parser.add_argument('--visualise_frames', action='store_true', help='set this flag to visualise frames')
    parser.add_argument('--log_freq', type=int, default=2,
                        help='How often to log to tensorboard and w&B.')
    parser.add_argument('--tb_dir', type=str, default='tb_runs_cv',
                        help='Tensorboard directory - where tensorboard logs are stored.')

    parser.add_argument('--res_dir', type=str, default='results',
                        help='Name of base directory for results')
    parser.add_argument('--segm_masks', action='store_true', help='set this flag to train only on segmentation masks')
    parser.add_argument('--crop', action='store_true', help='set this flag to crop to corners')

    # Temporal parameters
    parser.add_argument('--temporal', action='store_true', help='set this flag to predict on video clips')
    parser.add_argument('--clip_len', type=int, default=0, help='How many frames to select per video')
    parser.add_argument('--period', type=int, default=1, help='Sample period, sample every n-th frame')
    parser.add_argument('--multi_gpu', action='store_true', help='If use more than one GPU in parallel')

    if regression:
        parser.add_argument('--regression', action='store_true', help='set this flag to use regression')
    return parser

