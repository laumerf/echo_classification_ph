import torch
from torch import nn
from torchvision import models
from echo_ph.models.non_local import NLBlockND, MapBasedAtt
import os


def get_resnet3d_50(num_classes=2, pretrained=True):
    """
    Get a slow 3D-CNN with resnet-50 backbone.
    :param num_classes: 2 for binary classification or 3 for PH severity detection
    :param pretrained: True or False
    :return: The model.
    """
    download_model_path = os.path.expanduser('~/.cache/torch/hub/facebookresearch_pytorchvideo_main')
    model = torch.hub.load(download_model_path, 'slow_r50', source='local', pretrained=pretrained)
    in_channels = 1  # Grayscale
    model.blocks[0].conv = torch.nn.Conv3d(in_channels, model.blocks[0].conv.out_channels, kernel_size=(1, 7, 7),
                                           stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
    num_ftrs = model.blocks[5].proj.in_features
    model.blocks[5].proj = torch.nn.Linear(num_ftrs, num_classes)
    return model


def get_resnet3d_18(num_classes=2, model_type='r3d_18', pretrained=True):
    """
    Get a 3D-CNN with resnet-18 backbone, with different possible variations, as defined by model_type.
    :param num_classes: 2 for binary classification or 3 for PH severity detection
    :param model_type: One of r2plus1d_18, mc3_18, r3d_18
    :param pretrained: True or False
    :return: The model.
    """
    model = models.video.__dict__[model_type](pretrained=pretrained)
    in_channels = 1
    model.stem[0] = torch.nn.Conv3d(in_channels, model.stem[0].out_channels, kernel_size=(1, 7, 7), stride=(1, 2, 2),
                                    padding=(0, 3, 3), bias=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


class Res3DAttention(nn.Module):
    """ Get a 3D-CNN with resnet-18 backbone, with attention on time-step."""
    def __init__(self, num_classes=2, ch=1, w=112, h=112, t=6, att_type='self', pretrained=True):
        """
        :param num_classes: 2 for binary classification or 3 for PH severity detection
        :param ch: Number of channels. Set 1 for grayscale, 3 for colour.
        :param w: Width of frames
        :param h: Height of frames
        :param t: Sequence length
        :param att_type: self or map-based.
        :param pretrained: True or False
        """
        super(Res3DAttention, self).__init__()
        model = models.video.__dict__['r3d_18'](pretrained=pretrained)
        in_channels = 1
        model.stem[0] = torch.nn.Conv3d(in_channels, model.stem[0].out_channels, kernel_size=(1, 7, 7),
                                        stride=(1, 2, 2),
                                        padding=(0, 3, 3), bias=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        self.base_model = model
        if att_type == 'self':
            self.att_block = NLBlockND(in_channels=ch * w * h, dimension=1)
        else:  # 'map-based attention'
            self.att_block = MapBasedAtt(in_channels=ch * w * h, time_steps=t)

    def forward(self, x):
        batch_size, ch, clip_len, w, h = x.shape
        x = x.reshape((batch_size, ch * w * h, clip_len))  # reshape to batch_size, w*h, clip-length
        x, att = self.att_block(x)
        x = x.reshape((batch_size, ch, clip_len, w, h))  # reshape back to 'normal'
        x = self.base_model(x)  # Input: A tensor of size B, T, C, H, W
        return x, att


class Res3DSaliency(nn.Module):
    """
    Get 3D CNN model with resnet-18 backbone, specifically for CAM spatio-temporal saliency map
    visualisation, i.e. returning the last convolutional output for visualisation.
    """
    def __init__(self, num_classes=2, model_type='r3d_18', pretrained=True, return_last=True):
        """
        :param num_classes: 2 for binary classification, 3 for PH severity prediction.
        :param model_type:  One of r2plus1d_18, mc3_18, r3d_18
        :param pretrained: True if use pretrained model.
        :param return_last: True if last conv layer should be returned - for visualisation
        """
        super(Res3DSaliency, self).__init__()
        self.return_last = return_last
        model = models.video.__dict__[model_type](pretrained=pretrained)
        in_channels = 1
        model.stem[0] = torch.nn.Conv3d(in_channels, model.stem[0].out_channels, kernel_size=(1, 7, 7),
                                        stride=(1, 2, 2),
                                        padding=(0, 3, 3), bias=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        self.model_base = nn.Sequential(*[model.stem, model.layer1, model.layer2, model.layer3, model.layer4])
        self.avgpool = model.avgpool
        self.fc = model.fc

    def forward(self, x):
        last_conv_out = self.model_base(x)
        avg_out = self.avgpool(last_conv_out)
        avg_out = avg_out.view(avg_out.size(0), -1)
        pred = self.fc(avg_out)
        if self.return_last:
            return pred, last_conv_out
        else:
            return pred


class Res3DMultiView(nn.Module):
    """
    Get 3D CNN model with resnet-18 backbone, for multi-view input.
    """
    def __init__(self, device, num_classes=2, model_type='r3d_18', views=['KAPAP', 'CV', 'LA'], pretrained=True,
                 join_method='sum'):
        """
        :param device: Device.
        :param num_classes: 2 for binary, 3 for multi-class
        :param model_type: The model type
        :param views: The views to join in embedding space.
        :param pretrained: If the model should be pre-trained.
        :param join_method: What method to use to join features. Set to 'sum' or 'concat'.
        """
        super(Res3DMultiView, self).__init__()
        self.dev = device
        self.views = views
        self.join_method = join_method
        num_views = len(self.views)
        model = models.video.__dict__[model_type](pretrained=pretrained)
        in_channels = 1
        fc_in_ftrs = model.fc.in_features
        model.stem[0] = torch.nn.Conv3d(in_channels, model.stem[0].out_channels, kernel_size=(1, 7, 7),
                                        stride=(1, 2, 2),
                                        padding=(0, 3, 3), bias=False)
        self.fe_model = nn.Sequential(*list(model.children())[:-1])  # All but last layer
        self.fe_model_non_avg = nn.Sequential(*list(model.children())[:-2])  # All but last layer & but avg.pool
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.fc_concat = nn.Linear(fc_in_ftrs * num_views, num_classes)
        self.fc = nn.Linear(fc_in_ftrs, num_classes)

    def forward(self, x):
        # Try concat first, then avg. pool
        all_features = []
        for view in self.views:
            inp = x[view].transpose(2, 1).to(self.dev)
            ftrs = self.fe_model_non_avg(inp)
            all_features.append(ftrs)
        if self.join_method == 'concat':
            joined_ftrs = torch.cat(all_features, dim=1)
        elif self.join_method == 'sum':  # summing / stacking features
            joined_ftrs = torch.stack(all_features, dim=0).sum(dim=0)
        else:
            print('Error - Join method not implemented.')
        ftrs = self.avgpool(joined_ftrs)
        ftrs = ftrs.view(ftrs.size(0), -1)
        if self.join_method == 'concat':
            out = self.fc_concat(ftrs)
        else:  # sum
            out = self.fc(ftrs)
        return out


