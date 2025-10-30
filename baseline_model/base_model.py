"""
In this file, we define the base model architecture for our character-level language model using Flax/JAX.
In later model iterations, we may extend or modify this base architecture.

The model follows a GPT-style decoder-only Transformer architecture, consisting of:
- Token embeddings combined with learned positional embeddings
- A stack of Pre-LayerNorm decoder blocks with causal self-attention
- A final LayerNorm layer
- An output projection layer to map hidden states to vocabulary logits 
 
The tensor shape conventions used throughout are as follows:
- B: Batch size
- T: Sequence length (number of tokens)
- D: Hidden size / embedding dimension (d_model)
- V: Vocabulary size
"""

import jax.numpy as jnp
import flax.linen as nn
from flax.linen import attention as attn

class MLP(nn.Module):
    """
    This is the feed-forward network (Multilayer Perceptron) used within each Transformer block.

    Structure: 
    Dense(d_model -> d_model * mlp_ratio), GELU, Dense(d_model * mlp_ratio -> d_model)
    The expansion factor by default is mlp_ratio = 4.

    Args:
        d_model: Hidden size D.
        mlp_ratio: Expansion factor for the intermediate hidden size.
        dropout: Dropout rate to apply after each dense layer.

    Input Shape: (B, T, D)
    Output Shape: (B, T, D)
    """

    d_model: int
    mlp_ratio: int = 4
    dropout: float = 0.0

    def setup(self):

        hidden_dim = self.d_model * self.mlp_ratio
        self.fc1 = nn.Dense(hidden_dim) # Layer of size hidden (D * mlp_ratio)
        self.fc2 = nn.Dense(self.d_model) # Layer of size D
        self.dropout_layer = nn.Dropout(rate=self.dropout) # Dropout layer

    def __call__(self, x, *, deterministic: bool = True):

        x = self.fc1(x) # Expand channel dimension (D -> hidden)
        x = nn.gelu(x) # Apply non-linearity
        x = self.dropout_layer(x, deterministic=deterministic) # Dropout after activation
        x = self.fc2(x) # Project back to D (hidden -> D)
        x = self.dropout_layer(x, deterministic=deterministic) # Dropout after second dense
        return x
    
class SelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism.

    Args:
        n_heads: Number of attention heads.
        dropout: Dropout rate to apply on attention weights.

    Input Shape: (B, T, D)
    Output Shape: (B, T, D)
    """

    n_heads: int
    dropout: float = 0.0

    def setup(self):

        self.attention = nn.SelfAttention( # Single-head self-attention layer
            num_heads=self.n_heads, # Number of attention heads
            use_bias=False, # No bias for QKV projections
            dropout_rate=self.dropout # Dropout rate
        )

    def __call__(self, x, *, mask=None, deterministic: bool = True):

        attn_output = self.attention(x, mask=mask, deterministic=deterministic) # Apply self-attention

        return attn_output
    
class DecoderBlock(nn.Module):
    """
    A single decoder block consisting of:
    - Pre-LayerNorm (Improves training stability)
    - Self-Attention (causal when a causal mask is provided)
    - MLP
    - Residual connections (after attention and MLP sublayers)

    Args:
        d_model: Hidden size D.
        n_heads: Number of attention heads.
        mlp_ratio: Expansion factor for the MLP intermediate hidden size.
        dropout: Dropout rate to apply in attention and MLP.

    Input Shape: (B, T, D)
    Output Shape: (B, T, D)
    """

    d_model: int
    n_heads: int
    mlp_ratio: int = 4
    dropout: float = 0.0

    def setup(self):

        self.ln1 = nn.LayerNorm() # Pre-LayerNorm before self-attention
        self.self_attn = SelfAttention( # Multi-head self-attention layer
            n_heads=self.n_heads,
            dropout=self.dropout
        )
        self.ln2 = nn.LayerNorm() # Pre-LayerNorm before MLP
        self.mlp = MLP(
            d_model=self.d_model,
            mlp_ratio=self.mlp_ratio,
            dropout=self.dropout
        )

    def __call__(self, x, *, mask=None, deterministic: bool = True):

        # Self-Attention block
        h = self.ln1(x) # Pre-LayerNorm
        h = self.self_attn(h, mask=mask, deterministic=deterministic) # Multi Head Self-Attention
        x = x + h  # Residual connection

        # MLP block
        h = self.ln2(x) # Pre-LayerNorm
        h = self.mlp(h, deterministic=deterministic) # MLP
        x = x + h  # Residual connection

        return x
    
class DecoderOnlyTransformer(nn.Module):
    """
    GPT-style decoder-only Transformer for language modeling.

    Components Included in the Model Architecture in order:
    - Token Embeddings (maps token ids to D-dim vectors)
    - Learned Positional Embeddings (adds positional information to token embeddings)
    - n_layers of stacked DecoderBlocks with causal self-attention
    - Final LayerNorm
    - Output projection to vocabulary size (V logits)

    Args:
        vocab_size: Size of the vocabulary V.
        d_model: Hidden size D.
        n_heads: Number of attention heads.
        n_layers: Number of decoder blocks.
        mlp_ratio: Expansion factor for the MLP intermediate hidden size.
        seq_len: Maximum sequence length T.
        dropout: Dropout rate to apply in attention and MLP.
    """

    vocab_size: int
    d_model: int
    n_heads: int
    n_layers: int
    mlp_ratio: int = 4
    seq_len: int = 512
    dropout: float = 0.0

    def setup(self):

        self.token_embedding = nn.Embed( # Token embedding layer
            num_embeddings=self.vocab_size, # Vocabulary size V
            features=self.d_model # Embedding dimension D
        )
 
        self.position_embedding = self.param( # Learned positional embeddings
            'pos_embedding', # Parameter name
            nn.initializers.normal(stddev=0.02), # Normal initialization
            (self.seq_len, self.d_model) # Shape: (T, D)
        )

        self.decoder_blocks = [ # Stacked decoder blocks
            DecoderBlock(
                d_model=self.d_model, # Hidden size D
                n_heads=self.n_heads, # Number of attention heads
                mlp_ratio=self.mlp_ratio, # MLP expansion factor
                dropout=self.dropout # Dropout rate
            ) for _ in range(self.n_layers)
        ]

        self.ln_final = nn.LayerNorm() # Final LayerNorm
        self.output_projection = nn.Dense(self.vocab_size, use_bias=False) # Output projection to vocabulary size V

    def __call__(self, input_ids, *, deterministic: bool = True):

        B, T = input_ids.shape

        # Token and position embeddings
        token_embeds = self.token_embedding(input_ids) # Token embeddings (B, T, D)
        position_embeds = self.position_embedding[:T] # Positional embeddings (T, D)
        x = token_embeds + position_embeds # Combine token and positional embeddings

        # Apply decoder blocks
        causal = attn.make_causal_mask(jnp.ones((B, T), dtype=bool)) # Causal attention mask
        for block in self.decoder_blocks:
            x = block(x, mask=causal, deterministic=deterministic) # Pass through each decoder block. Deterministic for dropout

        x = self.ln_final(x) # Final LayerNorm

        # Output projection to vocabulary size
        logits = self.output_projection(x) # Output logits (B, T, V)

        return logits