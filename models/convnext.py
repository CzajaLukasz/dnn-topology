import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

class TopologicalConvNeXt(nn.Module):
    def __init__(self, num_classes=10):
        super(TopologicalConvNeXt, self).__init__()
        # Ładujemy gotowy model z torchvision
        self.base_model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        
        # Zmieniamy ostatnią warstwę klasyfikatora na liczbę naszych klas (np. 10 dla CIFAR10)
        self.base_model.classifier[2] = nn.Linear(self.base_model.classifier[2].in_features, num_classes)

    def forward(self, x):
        # Standardowy forward dla treningu
        return self.base_model(x)

    def forward_features(self, x):
        # Niestandardowy forward do budowy grafu korelacji
        features = []
        
        # Przechodzimy po kolei przez WSZYSTKIE bloki bloku "features"
        for i, layer in enumerate(self.base_model.features):
            x = layer(x)
            
            # Zbieramy cechy po głównych etapach (Stage 1=indeks 1, Stage 2=indeks 3, Stage 3=indeks 5, Stage 4=indeks 7)
            if i in [1, 3, 5, 7]:
                # Stosujemy Adaptive Average Pooling, aby zredukować wymiary HxW do 1x1.
                # Dzięki temu badamy korelacje między CAŁYMI FILTRAMI (kanałami), 
                # a nie pojedynczymi pikselami. Unikniemy awarii pamięci RAM.
                pooled = F.adaptive_avg_pool2d(x, (1, 1))
                
                # Spłaszczamy z (Batch, Channels, 1, 1) do (Batch, Channels)
                flattened = pooled.view(pooled.size(0), -1)
                features.append(flattened)
        
        return features