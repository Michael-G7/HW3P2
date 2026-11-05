import torch.nn as nn
import torch
import random
from typing import Tuple, Optional, Literal
from .masks import PadMask, CausalMask
from .positional_encoding import PositionalEncoding
from .decoder_layers import CrossAttentionDecoderLayer
from .encoder_layers import SelfAttentionEncoderLayer
from .speech_embedding import SpeechEmbedding
from .norms import create_norm
from .positional_encoding import RotaryPositionalEncoding
import warnings
'''
TODO: Implement these Modules.

This file contains the transformer architecture used in this homework:

EncoderDecoderTransformer: Used for ASR (Automatic Speech Recognition) tasks
   - Contains an encoder stack for processing speech features
   - Contains a decoder stack for generating text tokens
   - Uses both self-attention and cross-attention mechanisms
   - Includes CTC auxiliary loss support and optional weight tying

   Key components to implement:
   1. Speech Embedding: Convert speech features to vectors with time reduction
   2. Positional Encoding: Add position information (optional for both encoder/decoder)
   3. Encoder Stack: Process speech features
   4. Decoder Stack: Generate text tokens
   5. CTC Head: For auxiliary CTC loss computation
   6. Output Projection: Convert final representations to logits

   Architecture follows Pre-LN (Layer Normalization) design where:
   - Layer normalization is applied at the start of each sublayer
   - Residual connections wrap around each sublayer
   - Final layer norm is applied before output projection

   Implementation Notes:
   1. The forward pass should handle:
   - Proper masking (padding for encoder, both padding and causal for decoder)
   - Collecting attention weights from all layers
   - Optional layer dropout during training
   - CTC logits computation

   2. The score method should:
   - Handle single token prediction given encoder output
   - Not apply padding masks to decoder inputs
   - Return only the final token's logits
'''

## -------------------------------------------------------------------------------------------------
## Encoder-Decoder Transformer
## -------------------------------------------------------------------------------------------------
class EncoderDecoderTransformer(nn.Module):
    '''
    A Pre-LN Encoder-Decoder Transformer model for ASR tasks.
    '''
    def __init__(
            self,
            input_dim: int,  
            time_reduction: int, 
            reduction_method: Literal['lstm', 'conv', 'both'], 
            num_encoder_layers: int,
            num_encoder_heads: int,
            d_ff_encoder: int, 
            num_decoder_layers: int,
            num_decoder_heads: int,
            d_ff_decoder: int,
            d_model: int,
            dropout: float, 
            max_len: int, 
            num_classes: int,
            weight_tying: bool = False,
            layer_drop_rate: float = 0.0,
            norm_type: Literal['rmsnorm', 'layernorm'] = 'rmsnorm',
            ffn_type: Literal['swiglu', 'gelu'] = 'swiglu',
            qk_norm: bool = False,
            encoder_pos_encoding: Literal['rope', 'sinusoidal', 'none'] = 'rope',
            decoder_pos_encoding: Literal['rope', 'sinusoidal', 'none'] = 'rope',
    ):
        '''
        Initialize the Encoder-Decoder Transformer model.

        Args:
            input_dim: int, dimension of input speech features
            time_reduction: int, stride along time dimension, the amount of reduction to apply to the time dimension
            reduction_method: Literal['lstm', 'conv', 'both'], the source_embedding reduction method
            num_encoder_layers: int, number of encoder layers
            num_encoder_heads: int, number of encoder attention heads
            d_ff_encoder: int, feed-forward dimension for encoder
            num_decoder_layers: int, number of decoder layers
            num_decoder_heads: int, number of decoder attention heads
            d_ff_decoder: int, feed-forward dimension for decoder
            d_model: int, model dimension
            dropout: float, dropout rate
            max_len: int, maximum sequence length this model can handle
            num_classes: int, number of classes
            weight_tying: bool, whether to use weight tying (default: False)
            layer_drop_rate: float, layer drop rate (default: 0.0)
            norm_type: Literal['rmsnorm', 'layernorm'], the normalization to use (default: 'rmsnorm')
            ffn_type: Literal['swiglu', 'gelu'], the feed-forward network to use (default: 'swiglu')
            qk_norm: bool, whether to normalize queries and keys before scoring (default: False)
            encoder_pos_encoding: Literal['rope', 'sinusoidal', 'none'], how the encoder receives position (default: 'rope')
            decoder_pos_encoding: Literal['rope', 'sinusoidal', 'none'], how the decoder receives position (default: 'rope')
        '''
        super().__init__()
        for name, value in [('encoder_pos_encoding', encoder_pos_encoding),
                            ('decoder_pos_encoding', decoder_pos_encoding)]:
            if value not in ['rope', 'sinusoidal', 'none']:
                raise ValueError(f"{name} must be 'rope', 'sinusoidal', or 'none'")

        # Initialize model attributes
        # DO NOT MODIFY THESE ATTRIBUTES
        self.max_len = max_len
        self.layer_drop_rate = layer_drop_rate
        self.num_classes = num_classes
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.encoder_pos_encoding = encoder_pos_encoding
        self.decoder_pos_encoding = decoder_pos_encoding

        # TODO: Create the rotary embeddings, one per stack, or None where the stack
        # does not use them. Each stack rotates over its own head dimension, which is
        # d_model // num_heads for that stack, and every layer in a stack must share
        # the same object rather than building its own.
        self.encoder_rope = (
            RotaryPositionalEncoding(d_model // num_encoder_heads, max_len)
            if encoder_pos_encoding == 'rope' else None
        )
        self.decoder_rope = (
            RotaryPositionalEncoding(d_model // num_decoder_heads, max_len)
            if decoder_pos_encoding == 'rope' else None
        )

        # TODO: Implement __init__:
        # create encoder/decoder stacks, embeddings, positional encoding, dropout, norms, and final linear layer
        # Pass norm_type, ffn_type, qk_norm and the rotary embeddings down to the layers
        self.enc_layers = nn.ModuleList([
            SelfAttentionEncoderLayer(
                d_model=d_model,
                num_heads=num_encoder_heads,
                d_ff=d_ff_encoder,
                dropout=dropout,
                norm_type=norm_type,
                ffn_type=ffn_type,
                rope=self.encoder_rope,
                qk_norm=qk_norm,
            )
            for _ in range(num_encoder_layers)
        ])
        self.dec_layers = nn.ModuleList([
            CrossAttentionDecoderLayer(
                d_model=d_model,
                num_heads=num_decoder_heads,
                d_ff=d_ff_decoder,
                dropout=dropout,
                norm_type=norm_type,
                ffn_type=ffn_type,
                rope=self.decoder_rope,
                qk_norm=qk_norm,
            )
            for _ in range(num_decoder_layers)
        ])

        self.source_embedding = SpeechEmbedding(
            input_dim=input_dim,
            output_dim=d_model,
            time_reduction=time_reduction,
            reduction_method=reduction_method,
            dropout=dropout,
        )
        self.target_embedding = nn.Embedding(num_classes, d_model)
        # Only needed if either stack uses 'sinusoidal'
        self.positional_encoding = (
            PositionalEncoding(d_model, max_len)
            if 'sinusoidal' in (encoder_pos_encoding, decoder_pos_encoding)
            else None
        )
        self.final_linear = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.encoder_norm = create_norm(norm_type, d_model)
        self.decoder_norm = create_norm(norm_type, d_model)

        # TODO: Create the CTC head
        # Use nn.Sequential to create the CTC head
        # CTC head should project the final encoder output from the d_model space to the num_classes space
        # To be compatible with CTCLoss, a log_softmax to the output (See. nn.LogSoftmax)
        self.ctc_head = nn.Sequential(
            nn.Linear(d_model, num_classes),
            nn.LogSoftmax(dim=-1),
        )


        # Weight tying if enabled (extra form of regularization, read more about it)
        if weight_tying:
            self.target_embedding.weight = self.final_linear.weight


    def encode(self, padded_sources: torch.Tensor, source_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict, dict]:
        '''
        Encodes the source features into a sequence of hidden states.
        Args:
            padded_sources: The padded source sequences. shape: (batch_size, src_len, input_dim)
            source_lengths: The lengths of source sequences. shape: (batch_size,)
        Returns:
            x_enc: Encoded representation. shape: (batch_size, src_len, d_model)
            pad_mask_src: Source padding mask. shape: (batch_size, src_len)
            running_att: Dictionary containing encoder self-attention weights
            ctc_inputs: Dictionary of CTC input and source lengths. shape: (src_len, batch_size, d_model), (batch_size,) 
                        Keys: 'log_probs' and 'lengths'
                        Required for CTC loss computation
        '''

        if source_lengths is None:
            source_lengths = torch.full(
                (padded_sources.size(0),), padded_sources.size(1),
                dtype=torch.long, device=padded_sources.device,
            )
        x_enc, x_enc_lengths = self.source_embedding(padded_sources, source_lengths)
        if self.encoder_pos_encoding == 'sinusoidal':
            x_enc = self.positional_encoding(x_enc)
        x_enc = self.dropout(x_enc)
        pad_mask_src = PadMask(x_enc, x_enc_lengths)

        # TODO: Pass through encoder layers and save attention
        running_att = {}
        for i in range(self.num_encoder_layers):
            # Optionally apply LayerDrop during training (More regularization!)
            if self.training and self.layer_drop_rate > 0 and random.random() < self.layer_drop_rate:
                continue
 
            x_enc, attention = self.enc_layers[i](x_enc, key_padding_mask=pad_mask_src)
            running_att[f'layer{i+1}_enc_self'] = attention

        # TODO: final normalization and CTC projection
        x_enc = self.encoder_norm(x_enc)
        ctc_log_probs = self.ctc_head(x_enc).transpose(0, 1)
        ctc_inputs = {'log_probs': ctc_log_probs, 'lengths': x_enc_lengths}
        return x_enc, pad_mask_src, running_att, ctc_inputs

    def decode(
        self, 
        padded_targets: torch.Tensor, 
        encoder_output: torch.Tensor,
        target_lengths: Optional[torch.Tensor] = None,
        pad_mask_src: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        '''
        Decode the target sequence conditioned on the encoder output.
        Args:
            padded_targets: The padded target sequence. shape: (batch_size, tgt_len)
            encoder_output: Output from encoder. shape: (batch_size, src_len, d_model)
            target_lengths: The lengths of target sequences. shape: (batch_size,)
            pad_mask_src: Source padding mask from encoder. shape: (batch_size, src_len)
        Returns:
            seq_out: The output sequence. shape: (batch_size, tgt_len, num_classes)
            running_att: Dictionary containing decoder attention weights
        '''
        # TODO: Implement decode

        # TODO: Create target padding mask on the same device as the input
        pad_mask_tgt = None
        if target_lengths is not None:
            pad_mask_tgt = PadMask(padded_targets, target_lengths)

        if pad_mask_tgt is None and self.training:
            warnings.warn("pad_mask_tgt is None, unless you are using the decoder as a standalone model or doing inference, you should provide target_lengths")

        # TODO: Create causal mask on the same device as the input
        causal_mask = CausalMask(padded_targets)

        # TODO: Apply target embedding, positional encoding, and dropout
        # As in encode, add the sinusoidal encoding here only when decoder_pos_encoding
        # is 'sinusoidal'. Rotary embeddings are applied inside the attention layers.
        x_dec = self.target_embedding(padded_targets)
        if self.decoder_pos_encoding == 'sinusoidal':
            x_dec = self.positional_encoding(x_dec)
        x_dec = self.dropout(x_dec)

        # TODO: Pass through decoder layers and save attention
        running_att = {}
        for i in range(self.num_decoder_layers):
            if self.training and self.layer_drop_rate > 0 and random.random() < self.layer_drop_rate:
                continue
 
            x_dec, self_attn, cross_attn = self.dec_layers[i](
                x_dec,
                encoder_output,
                dec_key_padding_mask=pad_mask_tgt,
                enc_key_padding_mask=pad_mask_src,
                attn_mask=causal_mask,
            )
            running_att[f'layer{i+1}_dec_self'] = self_attn
            running_att[f'layer{i+1}_dec_cross'] = cross_attn

        # TODO: final normalization and projection
        seq_out = self.final_linear(self.decoder_norm(x_dec))
        return seq_out, running_att

    def forward(
        self,
        padded_sources: torch.Tensor,
        padded_targets: torch.Tensor,
        source_lengths: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict, dict]:
        '''
        Forward pass for the encoder-decoder transformer.
        
        Args:
            padded_sources: The padded source sequences. shape: (batch_size, src_len, input_dim)
            padded_targets: The padded target sequences. shape: (batch_size, tgt_len)
            source_lengths: The lengths of source sequences. shape: (batch_size,)
            target_lengths: The lengths of target sequences. shape: (batch_size,)
            
        Returns:
            seq_out: The output sequence logits. shape: (batch_size, tgt_len, num_classes)
            running_att: Dictionary containing all attention weights from both encoder and decoder
            ctc_inputs: Dictionary of CTC input and source lengths. shape: (src_len, batch_size, d_model), (batch_size,) 
                        Keys: 'log_probs' and 'lengths'
                        Required for CTC loss computation
        '''
        # During training, we need target lengths
        if self.training and target_lengths is None:
            raise ValueError("target_lengths must be provided during training")

        if self.training and source_lengths is None:
            raise ValueError("source_lengths must be provided during training")
        
        # TODO: Implement forward

        # TODO: run encoder then decoder
        encoder_output, pad_mask_src, enc_running_att, ctc_inputs = self.encode(
            padded_sources, source_lengths
        )
        seq_out, dec_running_att = self.decode(
            padded_targets, encoder_output, target_lengths, pad_mask_src
        )
        
        # Combine attention dictionaries
        running_att = {**enc_running_att, **dec_running_att}
        
        return seq_out, running_att, ctc_inputs

    def score(self, batch_prompts: torch.Tensor, encoder_output: torch.Tensor, pad_mask_src: torch.Tensor) -> torch.Tensor:
        '''
        Score the next token for given encoder output and prompt.
        Args:
            batch_prompts: tensor of token sequences to score for next token. shape: (batch_size, seq_len)
            encoder_output: encoder output/hidden states. shape: (batch_size, src_len, d_model)
            pad_mask_src: source padding mask. shape: (batch_size, src_len)
        Returns:
            logits: Batch of next token logits. shape: (batch_size, num_classes)
        '''
        if self.training:
            raise ValueError("score method is not supported during training")

        # TODO: Use decode function with no target lengths (no padding mask for targets)
        seq_out, _ = self.decode(batch_prompts, encoder_output, None, pad_mask_src)
        
        # Return only the last token's logits
        return seq_out[:, -1, :]


    def log_param_groups(self, param_groups: list) -> None:
        """Log information about parameter groups."""
        print("\nParameter groups:")
        total_params = 0
        total_trainable = 0
        
        for group in param_groups:
            num_params = sum(p.numel() for p in group['params'])
            trainable = sum(p.numel() for p in group['params'] if p.requires_grad)
            total_params += num_params
            total_trainable += trainable
            
            print(f"\n{group['name']}:")
            print(f"  Parameters: {num_params:,}")
            print(f"  Trainable: {trainable:,}")
            print(f"  LR factor: {group['lr_factor']}")
        
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Total trainable: {total_trainable:,}")
