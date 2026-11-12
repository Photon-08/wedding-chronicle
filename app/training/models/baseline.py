import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class FeatureExtractor(nn.Module):

    def __init__(self, num_classes=4):
        super().__init__()

        # Load pretrained ResNet18
        self.feature_extractor = models.resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        

    def forward(self, x):
        return self.feature_extractor(x)

feature_extractor = FeatureExtractor()
feature_extractor.eval()

class BaselineModel(nn.Module): 
    def __init__(self, num_classes):
        super(BaselineModel, self).__init__()
        self.feature_extractor = feature_extractor
        
        self.fc = nn.Linear(1000, num_classes)  # Adjust the input size based on the feature extractor

    def forward(self, x):
        features = self.feature_extractor(x)
        
        features = features.view(features.size(0), -1)  # Flatten the features
        output = self.fc(features)
        return output
