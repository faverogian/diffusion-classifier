from nets.dit import DiT
     
# Define model
dit = DiT(
    num_attention_heads=12,
    attention_head_dim=64,
    in_channels=config.image_channels if not config.wavelet_transform else 4*config.image_channels,
    out_channels=config.image_channels if not config.wavelet_transform else 4*config.image_channels,
    num_layers=12,
    dropout=0.0,
    norm_num_groups=32,
    attention_bias=True,
    sample_size=config.image_size if not config.wavelet_transform else config.image_size//2,
    patch_size=config.patch_size,
    activation_fn="gelu-approximate",
    num_embeds_ada_norm=1000,
    upcast_attention=False,
    norm_type="ada_norm_zero",
    norm_elementwise_affine=False,
    norm_eps=1e-5,
)