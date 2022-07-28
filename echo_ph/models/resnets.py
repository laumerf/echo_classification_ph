import torch
import torch.nn as nn
from torchvision.models import resnet18


def get_resnet18(num_classes=3, pretrained=True):
    """
    Get a Resnet-18 model (2D).
    :param num_classes: 2 for binary classification or 3 for PH severity detection
    :param pretrained: True or False
    :return: The model.
    """
    model = resnet18(pretrained=pretrained)
    in_channels = 1  # Grayscale
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    # Change the output layer to output num_classes instead of 1000 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


class ResMultiView(nn.Module):
    """
    Get Resnet-18 CNN model for multi-view input.
    """
    def __init__(self, device, num_classes=2, views=['KAPAP', 'CV', 'LA'], pretrained=True, join_method='sum'):
        """
        :param device: Device.
        :param num_classes: 2 for binary, 3 for multi-class
        :param views: The views to join in embedding space.
        :param pretrained: If the model should be pre-trained.
        :param join_method: What method to use to join features. Set to 'sum' or 'concat'.
        """
        super(ResMultiView, self).__init__()
        self.dev = device
        self.views = views
        self.join_method = join_method
        num_views = len(self.views)
        model = resnet18(pretrained=pretrained)
        in_channels = 1
        fc_in_ftrs = model.fc.in_features
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fe_model = nn.Sequential(*list(model.children())[:-1])  # All but last layer
        self.fe_model_non_avg = nn.Sequential(*list(model.children())[:-2])  # All but last layer & but avg.pool
        self.fc_concat = nn.Linear(fc_in_ftrs * num_views, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(fc_in_ftrs, num_classes)

    def forward(self, x):
        all_features = []
        for view in self.views:
            inp = x[view].to(self.dev)
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


