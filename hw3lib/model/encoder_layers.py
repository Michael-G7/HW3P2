import torch.nn as nn
import torch
from typing import Tuple, Optional
from .sublayers import SelfAttentionLayer, FeedForwardLayer
from .positional_encoding import RotaryPositionalEncoding

'''
TODO: Implement this Module.

This file contains the encoder layer implementation used in transformer architectures:

SelfAttentionEncoderLayer: Used in encoder part of transformers
- Contains self-attention and feed-forward sublayers
- Unlike decoder, does not use causal masking (can attend to all positions)
- Used for tasks like encoding input sequences where bidirectional context is needed

The layer follows a Pre-LN (Layer Normalization) architecture where:
- Layer normalization is applied before each sublayer operation
- Residual connections wrap around each sublayer

Implementation Steps:
1. Initialize the required sublayers in __init__:
   - SelfAttentionLayer for self-attention (no causal mask needed)
   - FeedForwardLayer for position-wise processing

2. Implement the forward pass to:
   - Apply sublayers in the correct order
   - Pass appropriate padding masks (no causal mask needed)
   - Return both outputs and attention weights
'''

class SelfAttentionEncoderLayer(nn.Module):
    '''
    Pre-LN Encoder Layer with self-attention mechanism.
    Used in the encoder part of transformer architectures.
    '''
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1,
                 norm_type: str = 'rmsnorm', ffn_type: str = 'swiglu',
                 rope: Optional[RotaryPositionalEncoding] = None, qk_norm: bool = False):
        '''
        Initialize the SelfAttentionEncoderLayer.
        Args:
            d_model   (int): The dimension of the model.
            num_heads (int): The number of attention heads.
            d_ff      (int): The dimension of the feedforward network.
            dropout (float): The dropout rate.
            norm_type (str): The normalization to use, 'rmsnorm' or 'layernorm'.
            ffn_type  (str): The feed-forward network to use, 'swiglu' or 'gelu'.
            rope (Optional[RotaryPositionalEncoding]): Rotary embeddings, or None to disable.
            qk_norm  (bool): Whether to normalize the queries and keys before scoring.
        '''
        super().__init__()
        self.self_attn = SelfAttentionLayer(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            norm_type=norm_type,
            rope=rope,
            qk_norm=qk_norm,
        )
        self.ffn = FeedForwardLayer(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            norm_type=norm_type,
            ffn_type=ffn_type,
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass for the EncoderLayer.
        Args:
            x (torch.Tensor): The input tensor. shape: (batch_size, seq_len, d_model)   
            key_padding_mask (torch.Tensor): The padding mask for the input. shape: (batch_size, seq_len)

        Returns:
            x (torch.Tensor): The output tensor. shape: (batch_size, seq_len, d_model)
            mha_attn_weights (torch.Tensor): The attention weights. shape: (batch_size, seq_len, seq_len)   
        '''
        x, attention = self.self_attn(x, key_padding_mask=key_padding_mask)
        x = self.ffn(x)
        return x, attention
