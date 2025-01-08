import torch
from dual_patchnorm import DualPatchNormConfig, DualPatchNorm

def main():
    # Create config
    config = DualPatchNormConfig(
        patch_size=(16, 16),
        hidden_size=768
    )

    # Initialize model
    dual_patchnorm = DualPatchNorm(config)

    # Forward pass with example input 
    batch_size = 4
    img_size = (224, 224)
    x = torch.randn(batch_size, *img_size, 3)
    output = dual_patchnorm(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    main()