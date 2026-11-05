import torch.nn as nn
import torch
from typing import Tuple, Optional
from .sublayers import SelfAttentionLayer, CrossAttentionLayer, FeedForwardLayer
from .positional_encoding import RotaryPositionalEncoding

'''
TODO: Implement this Module.

This file contains the decoder layer implementation used in transformer architectures:

CrossAttentionDecoderLayer: Used in encoder-decoder transformers (like BART)
   - Contains masked self-attention, cross-attention, and feed-forward sublayers
   - Used for tasks like translation where decoder needs to attend to encoder outputs

Each layer follows a Pre-LN (Layer Normalization) architecture where:
- Layer normalization is applied before each sublayer operation
- Residual connections wrap around each sublayer

Implementation Steps:
1. Initialize the required sublayers in __init__:
   - SelfAttentionLayer for masked self-attention
   - CrossAttentionLayer for cross-attention
   - FeedForwardLayer for position-wise processing

2. Implement the forward pass to:
   - Apply sublayers in the correct order
   - Pass appropriate masks to attention layers
   - Return both outputs and attention weights
'''

## -------------------------------------------------------------------------------------------------  
## Decoder Layers
## -------------------------------------------------------------------------------------------------      
class CrossAttentionDecoderLayer(nn.Module):
    '''
    Pre-LN Decoder Layer with masked self-attention, cross-attention, and feed-forward sublayers.
    Used in the encoder-decoder Transformer architecture.
    '''
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1,
                 norm_type: str = 'rmsnorm', ffn_type: str = 'swiglu',
                 rope: Optional[RotaryPositionalEncoding] = None, qk_norm: bool = False):
        '''
        Initialize the CrossAttentionDecoderLayer.
        Args:
            d_model   (int): The dimension of the model.
            num_heads (int): The number of attention heads.
            d_ff      (int): The dimension of the feedforward network.
            dropout (float): The dropout rate.
            norm_type (str): The normalization to use, 'rmsnorm' or 'layernorm'.
            ffn_type  (str): The feed-forward network to use, 'swiglu' or 'gelu'.
            rope (Optional[RotaryPositionalEncoding]): Rotary embeddings for the
                       self-attention sublayer, or None to disable. Cross-attention
                       never uses them.
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
        self.cross_attn = CrossAttentionLayer(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            norm_type=norm_type,
            qk_norm=qk_norm,
        )
        self.ffn = FeedForwardLayer(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            norm_type=norm_type,
            ffn_type=ffn_type,
        )

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, dec_key_padding_mask: Optional[torch.Tensor] = None, enc_key_padding_mask: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Forward pass for the CrossAttentionDecoderLayer.
        Args:
            x (torch.Tensor): The input tensor. shape: (batch_size, dec_seq_len, d_model)
            enc_output (torch.Tensor): The encoder output. shape: (batch_size, enc_seq_len, d_model)
            dec_key_padding_mask (Optional[torch.Tensor]): The padding mask for the decoder input. shape: (batch_size, dec_seq_len)
            enc_key_padding_mask (Optional[torch.Tensor]): The padding mask for the encoder output. shape: (batch_size, enc_seq_len)
            attn_mask (Optional[torch.Tensor]): The self-attention mask for the decoder input. shape: (dec_seq_len, dec_seq_len)
        Returns:
            x (torch.Tensor): The output tensor. shape: (batch_size, dec_seq_len, d_model)
            self_attn_weights (torch.Tensor): The attention weights. shape: (batch_size, dec_seq_len, dec_seq_len)
            cross_attn_weights (torch.Tensor): The attention weights. shape: (batch_size, dec_seq_len, enc_seq_len)
        '''
        x, self_attn_weights = self.self_attn(
            x,
            key_padding_mask=dec_key_padding_mask,
            attn_mask=attn_mask,
        )
        x, cross_attn_weights = self.cross_attn(
            x,
            enc_output,
            key_padding_mask=enc_key_padding_mask,
        )
        x = self.ffn(x)
        return x, self_attn_weights, cross_attn_weights
## -------------------------------------------------------------------------------------------------    
