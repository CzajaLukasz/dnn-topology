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
        features = []
        #print(f"DEBUG: Przechodzę przez {len(self.base_model.features)} warstw...")
        for i, layer in enumerate(self.base_model.features):
            #print(f"Warstwa {i}: {type(layer)}")
            x = layer(x)
            if i in [1, 3, 5, 7]:
                #print(f"DEBUG: Dodaję cechy z warstwy {i}, kształt: {x.shape}")
                pooled = F.adaptive_avg_pool2d(x, (1, 1))
                flattened = pooled.view(pooled.size(0), -1)
                features.append(flattened)
        
        #print(f"DEBUG: Zwracam {len(features)} zestawów cech.")
        return features