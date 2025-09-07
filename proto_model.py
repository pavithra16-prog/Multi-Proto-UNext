import torch
import torch.nn as nn
import torch.nn.functional as F
from archs import UNext  # Import your modified UNext

class UNeXtWithPrototypes(nn.Module):
    # def __init__(self, in_channels=3, num_classes=2, base_c=32, proto_dim=16): # changed from 64
    #     super().__init__()
    #     self.backbone = UNext(in_channels=in_channels, num_classes=num_classes, base_c=base_c)
    #     self.num_classes = num_classes
    #     self.proto_dim = proto_dim

    #     # Learnable class prototypes: shape [2, proto_dim]
    #     self.prototypes = nn.Parameter(torch.randn(num_classes, proto_dim))
    def __init__(self, in_channels=3, num_classes=2, base_c=32, proto_dim=16, num_prototypes=1):
        super().__init__()
        self.backbone = UNext(in_channels=in_channels, num_classes=num_classes, base_c=base_c)
        self.num_classes = num_classes
        self.proto_dim = proto_dim
        self.num_prototypes = num_prototypes
        # shape: [num_classes, num_prototypes, proto_dim]
        self.prototypes = nn.Parameter(torch.randn(num_classes, num_prototypes, proto_dim))


    def forward(self, x, return_features=False):
        # Get features first
        logits, features = self.backbone(x, return_features=True)
        B, C, H, W = features.shape

        # Ensure prototype dimension matches feature channels
        if self.prototypes.shape[2] != C:
            self.prototypes = nn.Parameter(torch.randn(self.num_classes, self.num_prototypes, C, device=features.device))

        return logits, features, self.prototypes

