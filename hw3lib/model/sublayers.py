import torch.nn as nn
import torch
from typing import Tuple, Optional
from .attention import MultiHeadAttention
from .norms import create_norm
from .positional_encoding import RotaryPositionalEncoding

'''
TODO: Implement these Modules.

The file contains four key sublayers used in transformer decoders:
1. SelfAttentionLayer: For masked self-attention
2. CrossAttentionLayer: For cross-attention between encoder and decoder
3. SwiGLU: The gated feed-forward network
4. FeedForwardLayer: For position-wise feed-forward processing

Each layer follows a Pre-LN (Layer Normalization) architecture where:
- Normalization is applied before the main operation
- A residual connection wraps around the operation

The attention and feed-forward layers are configurable:
- norm_type selects the normalization, 'rmsnorm' or 'layernorm'
- ffn_type selects the feed-forward network, 'swiglu' or 'gelu'
- rope, when given, supplies rotary embeddings to the self-attention layers

The attention itself is the MultiHeadAttention you ported from HW3P1, not
PyTorch's, because rotary embeddings have to reach the queries and keys after
the projections and nn.MultiheadAttention performs those internally.
'''

class SelfAttentionLayer(nn.Module):
    '''
    Pre-LN Decoder Sub-Layer 1.
    This layer is responsible for the causally-masked self-attention mechanism.

    Steps to implement:
    1. Initialize the multi-head attention with proper parameters
    2. Initialize normalization for d_model dimensionality using create_norm
    3. Initialize dropout with specified rate
    4. In forward pass:
       a. Store residual connection
       b. Apply pre-normalization
       c. Apply self-attention with masking
       d. Apply residual connection with dropout
       e. Return the output tensor and attention weights
    '''
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0,
                 norm_type: str = 'rmsnorm',
                 rope: Optional[RotaryPositionalEncoding] = None, qk_norm: bool = False):
        '''
        Initialize the SelfAttentionLayer.
        Args:
            d_model   (int): The dimension of the model.
            num_heads (int): The number of attention heads.
            dropout (float): The dropout rate.
            norm_type (str): The normalization to use, 'rmsnorm' or 'layernorm'.
            rope (Optional[RotaryPositionalEncoding]): Rotary embeddings, or None to disable.
            qk_norm  (bool): Whether to normalize the queries and keys before scoring.
        '''
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, dropout, rope=rope, qk_norm=qk_norm)
        self.norm = create_norm(norm_type, d_model)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass for the SelfAttentionLayer.
        Args:
            x (torch.Tensor): Input tensor, shape (batch_size, seq_len, d_model)
            key_padding_mask (Optional[torch.Tensor]): Padding mask, shape (batch_size, seq_len)
            attn_mask (Optional[torch.Tensor]): Attention mask, shape (seq_len, seq_len)

        Returns:
            x (torch.Tensor): Output tensor, shape (batch_size, seq_len, d_model)
            mha_attn_weights (torch.Tensor): Attention weights, shape (batch_size, seq_len, seq_len)
        '''
        residual = x
        normalized = self.norm(x)
        attended, mha_attn_weights = self.mha(
            normalized, normalized, normalized,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )
        return residual + self.dropout(attended), mha_attn_weights

## -------------------------------------------------------------------------------------------------
class CrossAttentionLayer(nn.Module):
    '''
    Pre-LN Decoder Sub-Layer 2.
    This layer is responsible for the cross-attention mechanism between encoder and decoder.

    Note that rotary embeddings are not applied here. The queries and the keys come
    from two different sequences, of two different modalities and lengths, so the
    difference between their positions does not mean anything.

    Steps to implement:
    1. Initialize the multi-head attention with proper parameters
    2. Initialize normalization for d_model dimensionality using create_norm
    3. Initialize dropout with specified rate
    4. In forward pass:
       a. Store residual connection
       b. Apply pre-normalization
       c. Apply cross-attention (query from decoder, key/value from encoder)
       d. Apply residual connection with dropout
       e. Return the output tensor and attention weights (both are needed)
    '''
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0,
                 norm_type: str = 'rmsnorm', qk_norm: bool = False):
        '''
        Initialize the CrossAttentionLayer.
        Args:
            d_model   (int): The dimension of the model.
            num_heads (int): The number of attention heads.
            dropout (float): The dropout rate.
            norm_type (str): The normalization to use, 'rmsnorm' or 'layernorm'.
            qk_norm  (bool): Whether to normalize the queries and keys before scoring.
        '''
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, dropout, qk_norm=qk_norm)
        self.norm = create_norm(norm_type, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, y: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass for the CrossAttentionLayer.
        Args:
            x (torch.Tensor): Decoder input, shape (batch_size, dec_seq_len, d_model)
            y (torch.Tensor): Encoder input, shape (batch_size, enc_seq_len, d_model)
            key_padding_mask (Optional[torch.Tensor]): Padding mask for the encoder input,
                              shape (batch_size, enc_seq_len)
            attn_mask (Optional[torch.Tensor]): Attention mask, shape (dec_seq_len, enc_seq_len)

        Returns:
            x (torch.Tensor): Output tensor, shape (batch_size, dec_seq_len, d_model)
            mha_attn_weights (torch.Tensor): Attention weights, shape (batch_size, dec_seq_len, enc_seq_len)
        '''
        residual = x
        attended, mha_attn_weights = self.mha(
            self.norm(x), y, y,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )
        return residual + self.dropout(attended), mha_attn_weights

## -------------------------------------------------------------------------------------------------
class SwiGLU(nn.Module):
    '''
    Gated feed-forward network.
    Where the GELU network applies one projection and one activation, this one
    applies two projections and gates the first by the second:

        down(silu(gate(x)) * up(x))

    Specification:
    - Three linear layers, none of them with a bias
    - gate and up both map d_model -> d_hidden, down maps d_hidden -> d_model
    - d_hidden is 2/3 of d_ff, rounded down. The gating needs three matrices where
      the GELU network needs two, so without this the two feed-forward types would
      not have comparable parameter counts and your ablations would not be fair

    Steps to implement:
    1. Compute d_hidden from d_ff
    2. Initialize the three linear layers
    3. In forward pass, gate the up projection and project back down
    '''
    def __init__(self, d_model: int, d_ff: int):
        '''
        Initialize the SwiGLU.
        Args:
            d_model (int): The dimension of the model.
            d_ff    (int): The dimension of the equivalent feedforward network.
        '''
        super().__init__()
        self.d_hidden = (2 * d_ff) // 3
        self.gate_proj = nn.Linear(d_model, self.d_hidden, bias=False)
        self.up_proj = nn.Linear(d_model, self.d_hidden, bias=False)
        self.down_proj = nn.Linear(self.d_hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass for the SwiGLU.
        Args:
            x (torch.Tensor): Input tensor, shape (batch_size, seq_len, d_model)

        Returns:
            x (torch.Tensor): Output tensor, shape (batch_size, seq_len, d_model)
        '''
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

## -------------------------------------------------------------------------------------------------
class FeedForwardLayer(nn.Module):
    '''
    Pre-LN Decoder Sub-Layer 3.
    This layer is responsible for the position-wise feed-forward network.

    Steps to implement:
    1. Initialize the feed-forward network according to ffn_type:
       For 'gelu', a Sequential with:
       a. First linear layer: d_model -> d_ff
       b. GELU activation
       c. Dropout
       d. Second linear layer: d_ff -> d_model
       For 'swiglu', the SwiGLU module above
    2. Initialize normalization for d_model dimensionality using create_norm
    3. Initialize dropout with specified rate
    4. In forward pass:
       a. Store residual connection
       b. Apply pre-normalization
       c. Apply feed-forward network with dropout
       d. Add residual connection
       e. Return the output tensor
    '''
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0,
                 norm_type: str = 'rmsnorm', ffn_type: str = 'swiglu'):
        '''
        Initialize the FeedForwardLayer.
        Args:
            d_model (int): The dimension of the model.
            d_ff (int): The dimension of the feedforward network.
            dropout (float): The dropout rate.
            norm_type (str): The normalization to use, 'rmsnorm' or 'layernorm'.
            ffn_type  (str): The feed-forward network to use, 'swiglu' or 'gelu'.
        '''
        super().__init__()
        if ffn_type not in ['swiglu', 'gelu']:
            raise ValueError("ffn_type must be 'swiglu' or 'gelu'")

        if ffn_type == 'swiglu':
            self.ffn = SwiGLU(d_model, d_ff)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
            )
        self.norm = create_norm(norm_type, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass for the FeedForwardLayer.
        Args:
            x (torch.Tensor): Input tensor, shape (batch_size, seq_len, d_model)

        Returns:
            x (torch.Tensor): Output tensor, shape (batch_size, seq_len, d_model)
        '''
        return x + self.dropout(self.ffn(self.norm(x)))
