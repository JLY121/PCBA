import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from models.simple import SimpleNet


class EfficientNetB0(SimpleNet):
    """
    EfficientNet-B0 model adapted for CIFAR-100 dataset.
    Based on the timm implementation but adjusted for 32x32 input images.
    """
    
    def __init__(self, num_classes=100, name=None, created_time=None, in_channels=3, pretrained=True):
        super(EfficientNetB0, self).__init__(name=name)
        
        # Try to load pre-trained EfficientNet-B0 from timm
        # If network is not available, fall back to non-pretrained version
        try:
            self.backbone = timm.create_model('efficientnet_b0', pretrained=pretrained)
            print(f"Successfully loaded EfficientNet-B0 with pretrained={pretrained}")
        except Exception as e:
            print(f"Failed to load pretrained model: {e}")
            print("Falling back to non-pretrained version...")
            self.backbone = timm.create_model('efficientnet_b0', pretrained=False)
        
        # Adjust the stem (first conv layer) for CIFAR-100's 32x32 input
        # Original stride=2 would reduce 32x32 to 16x16 too quickly, losing information
        original_conv_stem = self.backbone.conv_stem
        self.backbone.conv_stem = nn.Conv2d(
            in_channels,
            original_conv_stem.out_channels,
            kernel_size=3,
            stride=1,  # Changed from 2 to 1 for CIFAR-100
            padding=1,
            bias=False
        )
        
        # Copy weights from original stem if possible (for the center part)
        if in_channels == 3:
            with torch.no_grad():
                # Copy the center part of the original 3x3 kernel
                self.backbone.conv_stem.weight.data = original_conv_stem.weight.data.clone()
        
        # Replace the classifier head for CIFAR-100 (100 classes)
        num_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(num_features, num_classes)
        
        # Store parameters
        self.num_classes = num_classes
        self.in_channels = in_channels
        
        # Initialize new layers
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize the weights of new layers."""
        # Initialize the new classifier layer
        nn.init.normal_(self.backbone.classifier.weight, 0, 0.01)
        nn.init.zeros_(self.backbone.classifier.bias)
    
    def switch_grads(self, enable=True):
        """Enable or disable gradients for all parameters."""
        for param in self.parameters():
            param.requires_grad_(enable)
    
    def forward(self, x):
        """Forward pass through the model."""
        return self.backbone(x)
    
    def forward_features(self, x):
        """Forward pass through feature extractor only (without classifier)."""
        # Extract features using the backbone's feature extractor
        x = self.backbone.conv_stem(x)
        x = self.backbone.bn1(x)
        x = self.backbone.blocks(x)
        x = self.backbone.conv_head(x)
        x = self.backbone.bn2(x)
        return x
    
    def forward_embedding(self, x):
        """Forward pass to get embeddings (features before classification)."""
        features = self.forward_features(x)
        # Global average pooling
        x = self.backbone.global_pool(features)
        return x
    
    def features(self, x):
        """Extract features and return flattened representation."""
        embedding = self.forward_embedding(x)
        return embedding.view(embedding.size(0), -1)


# Factory functions following the ResNet pattern
def EfficientNetB0_CIFAR100(name=None, created_time=None, num_classes=100, in_channels=3, pretrained=True):
    """
    Create EfficientNet-B0 model adapted for CIFAR-100.
    
    Args:
        name: Model name for identification
        created_time: Creation timestamp
        num_classes: Number of output classes (default: 100 for CIFAR-100)
        in_channels: Number of input channels (default: 3 for RGB)
        pretrained: Whether to use pretrained weights (default: True)
    
    Returns:
        EfficientNetB0 model instance
    """
    model_name = f'{name}_EfficientNet_B0_CIFAR100' if name else 'EfficientNet_B0_CIFAR100'
    return EfficientNetB0(
        num_classes=num_classes,
        name=model_name,
        created_time=created_time,
        in_channels=in_channels,
        pretrained=pretrained
    )


def EfficientNetB0_Custom(name=None, created_time=None, num_classes=10, in_channels=3, pretrained=True):
    """
    Create EfficientNet-B0 model with custom number of classes.
    
    Args:
        name: Model name for identification
        created_time: Creation timestamp
        num_classes: Number of output classes
        in_channels: Number of input channels
        pretrained: Whether to use pretrained weights (default: True)
    
    Returns:
        EfficientNetB0 model instance
    """
    model_name = f'{name}_EfficientNet_B0_Custom' if name else 'EfficientNet_B0_Custom'
    return EfficientNetB0(
        num_classes=num_classes,
        name=model_name,
        created_time=created_time,
        in_channels=in_channels,
        pretrained=pretrained
    )


# # Test function
# def test():
#     """Test the EfficientNet-B0 model with CIFAR-100 input size."""
#     print("Testing EfficientNet-B0 for CIFAR-100...")
    
#     # Test with pretrained=False to avoid network issues
#     model = EfficientNetB0_CIFAR100(pretrained=False)
#     x = torch.randn(1, 3, 32, 32)  # CIFAR-100 input size
    
#     print(f"Input shape: {x.shape}")
    
#     # Test forward pass
#     y = model(x)
#     print(f"Output shape: {y.shape}")
#     print(f"Model name: {model.name}")
    
#     # Test feature extraction
#     features = model.features(x)
#     print(f"Features shape: {features.shape}")
    
#     # Test embedding extraction
#     embedding = model.forward_embedding(x)
#     print(f"Embedding shape: {embedding.shape}")
    
#     # Test with different input sizes
#     print("\nTesting with different batch sizes:")
#     for batch_size in [1, 4, 8]:
#         x_batch = torch.randn(batch_size, 3, 32, 32)
#         y_batch = model(x_batch)
#         print(f"Batch size {batch_size}: {x_batch.shape} -> {y_batch.shape}")
    
#     print("All tests passed!")


# def test_with_pretrained():
#     """Test the EfficientNet-B0 model with pretrained weights (if network is available)."""
#     print("Testing EfficientNet-B0 for CIFAR-100 with pretrained weights...")
    
#     try:
#         model = EfficientNetB0_CIFAR100(pretrained=True)
#         x = torch.randn(1, 3, 32, 32)
#         y = model(x)
#         print(f"Successfully loaded pretrained model!")
#         print(f"Input shape: {x.shape}")
#         print(f"Output shape: {y.shape}")
#     except Exception as e:
#         print(f"Failed to load pretrained model: {e}")
#         print("This is expected if network is not available.")


# if __name__ == '__main__':
#     test()
#     print("\n" + "="*50 + "\n")
#     test_with_pretrained()
