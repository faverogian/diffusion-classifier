from nets.unet import UNetCondition2D

# Define model - somewhat aligned with simple diffusion ImageNet 128 model
unet = UNetCondition2D(
    sample_size=256,
    in_channels=10,
    out_channels=10,
    layers_per_block=(2, 2, 2, 2, 4, 2),  # how many ResNet layers to use per UNet block
    block_out_channels=(128, 128, 256, 512, 512, 1024),  # the number of output channels for each UNet block
    down_block_types=(
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
    ),
    up_block_types=(
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
    mid_block_type="UNetMidBlock2DCrossAttn",
    encoder_hid_dim=512,
    encoder_hid_dim_type='text_proj',
    cross_attention_dim=512,
)