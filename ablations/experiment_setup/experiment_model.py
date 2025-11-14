"""
In this file, we define the ablation experiment model architecture for our character-level language model using Flax/JAX.

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

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import attention as attn
from numpy import dtype
from regex import D

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
    dtype: jnp.dtype = jnp.float32

    def setup(self):

        hidden_dim = self.d_model * self.mlp_ratio
        self.fc1 = nn.Dense(hidden_dim, dtype=self.dtype) # Layer of size hidden (D * mlp_ratio)
        self.fc2 = nn.Dense(self.d_model, dtype=self.dtype) # Layer of size D
        self.dropout_layer = nn.Dropout(rate=self.dropout) # Dropout layer

    def __call__(self, x, *, deterministic: bool = True):

        x = self.fc1(x) # Expand channel dimension (D -> hidden)
        x = nn.gelu(x) # Apply non-linearity
        x = self.dropout_layer(x, deterministic=deterministic) # Dropout after activation
        x = self.fc2(x) # Project back to D (hidden -> D)
        x = self.dropout_layer(x, deterministic=deterministic) # Dropout after second dense
        return x
    
class PositionalEncoding(nn.Module):
    """
    Adds positional encodings in accordance to the Transformer architecture.
    This module supports different types of positional encodings including:
    - Learned Positional Embeddings
    - Sinusoidal Positional Encodings
    - Rotary Positional Embeddings (not implemented here)
    - ALiBi (not implemented here)
    - No positional encoding
    
    Args:
        seq_len: Maximum sequence length T.
        d_model: Hidden size D.
        encoding_type: Type of positional encoding (e.g 'learned' or 'sinusoidal')
        dtype: Data type of the positional encodings.
    """

    seq_len: int
    d_model: int
    encoding_type: str = "learned"
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if self.encoding_type == 'learned':
            self.pos_embedding = self.param( # Learned positional embeddings
                'pos_embedding', # Parameter name
                nn.initializers.normal(stddev=0.02), # Normal initialization
                (self.seq_len, self.d_model) # Shape: (T, D)
            )
        elif self.encoding_type == 'sinusoidal':
            # Create sinusoidal positional encodings
            position = jnp.arange(0, self.seq_len)[:, None]
            div_term = jnp.exp(jnp.arange(0, self.d_model, 2) * -(jnp.log(10000.0) / self.d_model))
            pe = jnp.zeros((self.seq_len, self.d_model))
            pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
            pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))

            # Store as non-trainable parameter
            self.pos_embedding = self.variable('constants', 'pos_embedding', lambda: pe.astype(self.dtype))

        elif self.encoding_type in ["none", "rotary", "alibi"]:
            pass  # No positional embeddings needed for these types
        else:
            raise ValueError(f"Unknown encoding_type: {self.encoding_type}")

    def __call__(self, x):

        B, T, D = x.shape

        if self.encoding_type in ['learned', 'sinusoidal']:
            
            if self.encoding_type == 'sinusoidal':
                position_embeds = self.pos_embedding.value[:T]  # Positional embeddings (T, D)
            else:
                position_embeds = self.pos_embedding[:T] # Positional embeddings (T, D)

            x = x + position_embeds[jnp.newaxis, :, :] # Combine token and positional embeddings, broadcast to (B, T, D)
            return x
        elif self.encoding_type in ["none", "rotary", "alibi"]:
            return x
        else:
            raise ValueError(f"Unknown encoding_type: {self.encoding_type}")
        
class Attention(nn.Module):
    """
    Variable attention mechanism for ablation experiments.
    This module implements different attention types including:
    - Multi-Head Attention (MHA)
    - Multi-Query Attention (MQA)
    - Grouped-Query Attention (GQA)

    Args:
        n_heads: Number of attention heads.
        d_model: Hidden size D.
        dropout: Dropout rate to apply on attention weights.
        attention_type: Type of attention mechanism (e.g., 'MHA', 'MQA', etc.)
        pos_encoding: Positional encoding type (e.g., 'none', 'sinusoidal', etc.)
        dtype: Data type of the attention computations.

    Input Shape: (B, T, D)
    Output Shape: (B, T, D)
    """

    n_heads: int
    d_model: int
    attention_type: str = "MHA"
    dropout: float = 0.0
    pos_encoding: str = "none"
    dtype: jnp.dtype = jnp.float32

    def setup(self):

        self.q_proj = nn.Dense(self.d_model, dtype=self.dtype) # Query projection, always has full heads

        # Configure K and V projections based on attention type
        if self.attention_type == "MHA":
            kv_heads = self.n_heads  # Full heads for K and V
        elif self.attention_type == "MQA":
            kv_heads = 1  # Single head for K and V
        elif self.attention_type == "GQA":
            kv_heads = max(1, self.n_heads // 4)  # Grouped heads for K and V
        else:
            raise ValueError(f"Unknown attention_type: {self.attention_type}")
        
        self.kv_heads = kv_heads

        # Keys and values have kv_heads instead of n_heads
        self.k_proj = nn.Dense(self.d_model, dtype=self.dtype) # Key projection
        self.v_proj = nn.Dense(self.d_model, dtype=self.dtype) # Value projection
        self.out_proj = nn.Dense(self.d_model, dtype=self.dtype) # Output projection
        self.dropout_layer = nn.Dropout(rate=self.dropout) # Dropout layer

        # Extra step for ALiBi, precompute slopes
        if self.pos_encoding == "alibi":
            self.slopes = self.variable('constants', 'alibi_slopes', self._get_alibi_slopes, self.n_heads)

    ### define extra RoPE utils ###

    def _rotate_half(self, x):
        # (B, T, H, D) -> rotate last dim
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return jnp.stack([-x2, x1], axis=-1).reshape(x.shape)
    
    def apply_rope(self, q, k):
        _, T, _, D = q.shape # (B, T, H, head_dim)
        half = D // 2

        # RoPE frequencies
        idx = jnp.arange(T)[:, None]  # (T, 1)
        freq = jnp.exp(-jnp.arange(0, half, 2) * (jnp.log(10000.0) / half))  # (half/2,)
        freq = freq[None, None, None, :]  # (1, half/2, 1)

        sin = jnp.sin(idx * freq)[None, :, None, :]  # (T, half/2, 1)
        cos = jnp.cos(idx * freq)[None, :, None, :]  # (T, half/2, 1)

        q1, q2 = q[..., :half], q[..., half:] # Split last dim
        k1, k2 = k[..., :half], k[..., half:] # Split last dim

        q_rotated = q1 * cos + self._rotate_half(q1) * sin
        k_rotated = k1 * cos + self._rotate_half(k1) * sin

        q = jnp.concatenate([q_rotated, q2], axis=-1)
        k = jnp.concatenate([k_rotated, k2], axis=-1)

        return q, k

    ### define extra ALiBi utils ###

    def _get_alibi_slopes(self, n_heads):

        def nearest_power_of_2(x):
            return 2 ** jnp.floor(jnp.log2(x))
        
        m = nearest_power_of_2(n_heads)
        base = 2 ** (-(jnp.arange(m) / m))
        if m < n_heads:
            base = jnp.concatenate([base, jnp.full((n_heads - int(m),), base[-1])])
        return base
    
    def _build_alibi_bias(self, T):
        # slopes: (n_heads, )
        slopes = self.slopes.value[:, None, None]  # (n_heads, 1, 1)
        idx = jnp.arange(T)  # (T, )
        diff = idx[None, :] - idx[:, None]  # (T, T)
        diff = diff[None, :, :]  # (1, T, T)
        alibi_bias = slopes * diff  # (n_heads, T, T)

        return alibi_bias

    def __call__(self, x, mask=None, deterministic: bool = True):

        B, T, D = x.shape
        H = self.n_heads
        KvH = self.kv_heads
        head_dim = D // H

        # Project inputs to Q, K, V
        q = self.q_proj(x).reshape(B, T, H, head_dim)
        k = self.k_proj(x).reshape(B, T, KvH, head_dim)
        v = self.v_proj(x).reshape(B, T, KvH, head_dim)

        # Apply RoPE if specified
        if self.pos_encoding == "rotary":
            q, k = self.apply_rope(q, k)

        # Compute attention scores and outputs based on attention type
        if self.attention_type == "MQA":
            # Expand K and V to match number of heads
            k = jnp.repeat(k, H, axis=2)  # (B, T, H, head_dim)
            v = jnp.repeat(v, H, axis=2)  # (B, T, H, head_dim)

        elif self.attention_type == "GQA":
            # Repeat K and V for grouped heads
            repeat_factor = H // KvH
            k = jnp.repeat(k, repeat_factor, axis=2)  # (B, T, H, head_dim)
            v = jnp.repeat(v, repeat_factor, axis=2)  # (B, T, H, head_dim)

        # Attention computation       
        attention_scores = jnp.einsum('bthd,bThd->bhtT', q, k) / jnp.sqrt(head_dim)  # Scaled dot-product

        # Add ALiBi bias if specified
        if self.pos_encoding == "alibi":
            alibi_bias = self._build_alibi_bias(T)  # (n_heads, T, T)
            attention_scores += alibi_bias[None, :, :, :]  # Broadcast to (B, n_heads, T, T)

        if mask is not None:
            attention_scores = jnp.where(mask, attention_scores, -1e10)  # Apply mask

        attention_weights = nn.softmax(attention_scores, axis=-1)  # Softmax over T
        attention_weights = self.dropout_layer(attention_weights, deterministic=deterministic)  # Dropout on attention weights

        output = jnp.einsum('bhtT,bThd->bthd', attention_weights, v)  # Weighted sum of values
        output = output.reshape(B, T, D)  # Concatenate heads
        output = self.out_proj(output)  # Final linear projection

        return output
    
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
    attention_type: str = "MHA"
    dtype: jnp.dtype = jnp.float32

    def setup(self):

        self.ln1 = nn.LayerNorm(dtype=self.dtype) # Pre-LayerNorm before self-attention
        self.self_attn = Attention( # Multi-head self-attention layer
            n_heads=self.n_heads,
            d_model=self.d_model,
            dropout=self.dropout,
            attention_type=self.attention_type,
            dtype=self.dtype
        )
        self.ln2 = nn.LayerNorm(dtype=self.dtype) # Pre-LayerNorm before MLP
        self.mlp = MLP(
            d_model=self.d_model,
            mlp_ratio=self.mlp_ratio,
            dropout=self.dropout,
            dtype=self.dtype
        )

    def __call__(self, x, *, mask=None, deterministic: bool = True):

        # Self-Attention block
        h = self.ln1(x) # Pre-LayerNorm
        h = self.self_attn(h, mask=mask, deterministic=deterministic) # Self-attention
        x = x + h  # Residual connection

        # MLP block
        h = self.ln2(x) # Pre-LayerNorm
        h = self.mlp(h, deterministic=deterministic) # MLP
        x = x + h  # Residual connection

        return x
    
class DecoderOnlyTransformer(nn.Module):
    """
    GPT-style decoder-only Transformer for ablation testing.

    Components Included in the Model Architecture in order:
    - Token Embeddings (maps token ids to D-dim vectors)
    - Positional Embeddings (adds positional information to token embeddings)
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
        attention_type: Type of attention mechanism (e.g., 'MHA', 'MQA', etc.)
        pos_encoding: Positional encoding type (e.g., 'none', 'sinusoidal', etc.)
        dtype: Data type for model computations.
    """

    vocab_size: int
    d_model: int
    n_heads: int
    n_layers: int
    mlp_ratio: int = 4
    seq_len: int = 512
    dropout: float = 0.0
    attention_type: str = "MHA"
    pos_encoding: str = "learned"
    dtype: jnp.dtype = jnp.float32

    # for auxiliary losses
    num_aux_heads: int = 0
    aux_loss: bool = False

    def setup(self):

        self.token_embedding = nn.Embed( # Token embedding layer
            num_embeddings=self.vocab_size, # Vocabulary size V
            features=self.d_model, # Embedding dimension D
            dtype=self.dtype
        )
 
        self.pos_encoder = PositionalEncoding( # Positional encoding module
            seq_len=self.seq_len, # Maximum sequence length T
            d_model=self.d_model, # Hidden size D
            encoding_type=self.pos_encoding, # Type of positional encoding
            dtype=self.dtype
        )

        self.decoder_blocks = [ # Stacked decoder blocks
            DecoderBlock(
                d_model=self.d_model, # Hidden size D
                n_heads=self.n_heads, # Number of attention heads
                mlp_ratio=self.mlp_ratio, # MLP expansion factor
                dropout=self.dropout, # Dropout rate
                attention_type=self.attention_type, # Type of attention
                dtype=self.dtype
            ) for _ in range(self.n_layers)
        ]

        self.ln_final = nn.LayerNorm(dtype=self.dtype) # Final LayerNorm
        
        self.output_projection = nn.Dense(self.vocab_size, use_bias=False, dtype=self.dtype) # Output projection to vocabulary size V

        if self.num_aux_heads > 0 and self.aux_loss:
            self.aux_lns = [
                nn.LayerNorm(dtype=self.dtype) for _ in range(self.num_aux_heads)
            ]
            self.aux_heads = [
                nn.Dense(self.vocab_size, use_bias=False, dtype=self.dtype) for _ in range(self.num_aux_heads)
            ]

    def __call__(self, input_ids, *, deterministic: bool = True):

        B, T = input_ids.shape

        # Token and position embeddings
        token_embeds = self.token_embedding(input_ids) # Token embeddings (B, T, D)
        x = self.pos_encoder(token_embeds) # Positional embeddings (T, D)

        # Apply decoder blocks
        causal = attn.make_causal_mask(jnp.ones((B, T), dtype=bool)) # Causal attention mask

        intermediates = []

        for i, block in enumerate(self.decoder_blocks):
            x = block(x, mask=causal, deterministic=deterministic) # Pass through each decoder block. Deterministic for dropout

            if self.aux_loss and i >= (self.n_layers - self.num_aux_heads):
                intermediates.append(x)

        x = self.ln_final(x) # Final LayerNorm

        # Output projection to vocabulary size
        logits = self.output_projection(x) # Output logits (B, T, V)

        if not self.aux_loss:
            return {"logits": logits, "aux_logits": None}
        
        else:

            aux_logits = []

            for h, ln, aux_head in zip(intermediates, self.aux_lns, self.aux_heads):
                h_norm = ln(h) # LayerNorm
                aux_logit = aux_head(h_norm) # Auxiliary logits
                aux_logits.append(aux_logit)

            return {"logits": logits, "aux_logits": aux_logits}