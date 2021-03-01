from deep_fake_detect.utils import *
import torch
import torch.nn as nn
from utils import *
from data_utils.utils import *
import torch.nn as nn
import torch.nn.functional as F
from deep_fake_detect.features import *


class DeepFakeDetectModel(nn.Module):
    """
    This is simple model which takes in each frame of video independently and classified them.
    Later the entire video is classified based upon heuristics, which is not done by this model.
    For the frame passed, features are extracted, using given encoder. Then applies AdaptiveAvgPool2d, flattens the
    features and passes to classifier.

    cnn_encoder:
      default: ['tf_efficientnet_b0_ns', 'tf_efficientnet_b7_ns'] # choose either one
    training:
      train_size: 1
      valid_size: 1
      test_size: 1
      params:
        model_name: 'DeepFakeDetectModel'
        label_smoothing: 0.1 # 0 to disable this, or any value less than 1
        train_transform: ['simple', 'complex'] # choose either of the data augmentation
        batch_format: 'simple' # Do not change
        # Adjust epochs, learning_rate, batch_size , fp16, opt_level
        epochs: 5
        learning_rate: 0.001
        batch_size: 4
        fp16: True
        opt_level: 'O1'
        dataset: ['optical', 'plain'] # choose either of the data type
    """

    def __init__(self, frame_dim=None, encoder_name=None):
        super().__init__()

        self.image_dim = frame_dim
        self.num_of_classes = 1
        self.encoder = get_encoder(encoder_name)
        self.encoder_flat_feature_dim, _ = get_encoder_params(encoder_name)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(self.encoder_flat_feature_dim, int(self.encoder_flat_feature_dim * .10)),
            nn.Dropout(0.50),
            nn.ReLU(),
            nn.Linear(int(self.encoder_flat_feature_dim * .10), self.num_of_classes),
        )

    def forward(self, x):
        # x shape = batch_size x color_channels x image_h x image_w
        x = self.encoder.forward_features(x)
        x = self.avg_pool(x).flatten(1)
        x = self.classifier(x)
        return x
