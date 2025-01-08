from typing import Optional, Tuple, Union
from diffusers import DiTTransformer2DModel

#----------------------------------------------------------------------------
# Adaptation of HuggingFace's DiTTransformer2DModel to use with the 
# DiffusionClassifier pipeline

class DiT(DiTTransformer2DModel):
    def __init__(
            self,
            num_attention_heads: int = 16,
            attention_head_dim: int = 72,
            in_channels: int = 4,
            out_channels: Optional[int] = None,
            num_layers: int = 28,
            dropout: float = 0.0,
            norm_num_groups: int = 32,
            attention_bias: bool = True,
            sample_size: int = 32,
            patch_size: int = 2,
            activation_fn: str = "gelu-approximate",
            num_embeds_ada_norm: Optional[int] = 1000,
            upcast_attention: bool = False,
            norm_type: str = "ada_norm_zero",
            norm_elementwise_affine: bool = False,
            norm_eps: float = 1e-5,
        ):

        super().__init__(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            norm_num_groups=norm_num_groups,
            attention_bias=attention_bias,
            sample_size=sample_size,
            patch_size=patch_size,
            activation_fn=activation_fn,
            num_embeds_ada_norm=num_embeds_ada_norm,
            upcast_attention=upcast_attention,
            norm_type=norm_type,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
        )


    def forward(self, x, noise_labels, encoder_hidden_states=None, ):
        x = super().forward(x, noise_labels, encoder_hidden_states, return_dict=False)
        return x[0]