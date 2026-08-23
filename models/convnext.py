import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

class TopologicalConvNeXt(nn.Module):
    def __init__(self, num_classes=10):
        super(TopologicalConvNeXt, self).__init__()
        # Ładujemy architekturę z wagami
        self.base_model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        
        # Dostosowanie klasyfikatora
        self.base_model.classifier[2] = nn.Linear(self.base_model.classifier[2].in_features, num_classes)

    def _prepare_input(self, x):
        # 1. Jeśli obraz jest 1-kanałowy (MNIST/USPS), powielamy do 3 kanałów RGB
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            
        # 2. Jeśli wymiar przestrzenny jest mniejszy niż 64x64, skalujemy go w górę
        if x.shape[2] < 64 or x.shape[3] < 64:
            x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
            
        return x

    def forward(self, x):
        x = self._prepare_input(x)
        return self.base_model(x)

    def forward_features(self, x):
        x = self._prepare_input(x)
        features = []
        for i, layer in enumerate(self.base_model.features):
            x = layer(x)
            if i in [1, 3, 5, 7]:
                pooled = F.adaptive_avg_pool2d(x, (1, 1))
                flattened = pooled.view(pooled.size(0), -1)
                features.append(flattened)
        return features