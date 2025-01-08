import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Tuple

class ResNet2D(nn.Module):
    """
    A ResNet-based backbone for 2D image inputs, returning feature embeddings
    that can be used for downstream classification.
    """

    def __init__(
        self,
        variant: str = "resnet18",
        pretrained: bool = True,
        in_channels: int = 3,
    ):
        """
        Args:
            variant (str): Which ResNet variant to load (e.g., 'resnet18', 'resnet34', 'resnet50', etc.).
            pretrained (bool): Whether to load pretrained ImageNet weights.
            in_channels (int): Number of input channels (default=3 for RGB).
        """
        super().__init__()

        self.variant = variant
        self.in_channels = in_channels

        # Load a standard ResNet from torchvision
        if self.variant == "resnet18":
            if pretrained:
                self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            else:
                self.resnet = models.resnet18()
            self.output_dim = 512
        elif self.variant == "resnet34":
            if pretrained:
                self.resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            else:
                self.resnet = models.resnet34()

            self.output_dim = 512
        elif self.variant == "resnet50":
            if pretrained:
                self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            else:
                self.resnet = models.resnet50()
            self.output_dim = 2048
        else:
            raise ValueError(f"Unsupported ResNet variant: {self.variant}")

        # If input channels != 3, adapt the first conv layer
        if self.in_channels != 3:
            # Create a new Conv2d layer to replace the original first layer
            old_conv = self.resnet.conv1
            self.resnet.conv1 = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias
            )

        # Remove the final FC layer to get raw features
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ResNet (minus the final FC).
        
        Args:
            x (torch.Tensor): Input image tensor of shape [B, in_channels, H, W].

        Returns:
            features (torch.Tensor): Feature embeddings of shape [B, output_dim].
        """
        # Pass through backbone
        features = self.resnet(x)        # shape [B, default_output_dim, 1, 1]
        features = features.view(features.size(0), -1)  # Flatten: [B, default_output_dim]

        return features
