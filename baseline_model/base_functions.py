"""
In this file, we define the functions required for training and evaluating our baseline character-level language model.
In later model iterations, we may extend or modify these functions.

Using JAX and Flax for efficient computation, 
The functions are defined based on each component of the training and evaluation pipeline as follows:

1. Data Preprocessing:
    - encode -> Text encoding to integer representation
    - split_train_val -> Train-validation split

2. Model Creation:
	- count_parameters -> Count no. of model parameters for reporting
    - create_train_state -> Initialize model and parameters

3. Training & Evaluation Loop:
	- get_batch -> Create random training / validation batches
    - loss_and_metrics -> Compute loss and accuracy metrics
    - train_step -> Single optimization step with optax optimizer

4. Autoregressive Token Generation:
	- trim_or_pad_context -> Ensure context length matches model block size
	- sample_categorical -> Sample from categorical distribution with temperature
	- generate_tokens -> Main autoregressive token generation function
    - decode -> Decode integer representation back to text
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import base_model as model

# ---------------------------- #
# Data Preprocessing Functions #
# ---------------------------- #

def encode(text, chars_to_int, unknown_token = None):
    """
    Encode text to integer representation

    Args:
    	text: input string
        chars_to_int: dictionary mapping characters to integers
        unknown_token: integer to use for unknown characters (if None, raises error on unknown chars)
    
    Returns:
        - numpy array of integers representing the encoded text
    """

    if unknown_token is not None:
        return np.array([chars_to_int.get(c, unknown_token) for c in text], dtype=np.int32)
    else:
        return np.array([chars_to_int[c] for c in text], dtype=np.int32)

def split_train_val(data, val_fraction=0.1):
    """
    Splits data into training and validation sets.
    We ensure that the split is done at the last whitespace before the split point.
    If no whitespace is found, we split at the exact point.
    
    Args:
		data: string, the full text data
        val_fraction: fraction of data to use for validation
        
    Returns:
		train_data: string, training data
        val_data: string, validation data
    """
    
    split_idx = int(len(data) * (1 - val_fraction)) # Initial split index based on val_fraction
    last_space_idx = data.rfind(' ', 0, split_idx) # Find last whitespace before split_idx
    
    if last_space_idx != -1: # If a whitespace is found, split there
        split_idx = last_space_idx
        
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    return train_data, val_data

# ------------------------ #
# Model Creation Functions #
# ------------------------ #

def count_parameters(params):
    """
    Count the number of parameters in a Flax model.
    
    Args:
        params: Flax model parameters.

    Returns:
        Total number of parameters.
    """

    return sum(jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda x: x.size, params)))

def create_train_state(rng, vocab_size = 27, d_model = 64, n_layers = 2, n_heads = 4, seq_len = 128, mlp_ratio = 4, dropout = 0.0):
    """
    Create initial training state. This includes model initialization and parameter setup.
    
    Args:
		rng: JAX PRNGKey for random number generation
        vocab_size: size of the vocabulary
		d_model: dimension of the model
		n_layers: number of transformer layers
		n_heads: number of attention heads
		seq_len: maximum sequence length
		mlp_ratio: ratio for MLP hidden dimension
		dropout: dropout rate
    
    Returns:
		model: initialized model
        params: initialized model parameters
    """

    m1 = model.DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        mlp_ratio=mlp_ratio,
        seq_len=seq_len,
        dropout=dropout
    )

    # Create dummy input for initialization
    dummy_input = jnp.ones((1, min(16, seq_len)), dtype=jnp.int32)

    # Pass the dummy input through the model to initialize parameters
    variables = m1.init({"params": rng}, dummy_input, deterministic=False)
    params = variables['params'] # Extract parameters from the initialized variables

    return m1, params

# ------------------------------------ #
# Training & Evaluation Loop Functions #
# ------------------------------------ #

def get_batch(text_int, B, T):
    """
    Create a random batch of data from text_int for training.

    Args:
    	text_int: 1D array of token ids.
    	B: batch size (number of sequences).
    	T: sequence length (number of tokens per sequence).

    Returns:
    	x: (B, T) int array input tokens.
    	y: (B, T) int array target tokens.
    """
    
    # choose random starting indices for each sequence in the batch
    ix = np.random.randint(0, len(text_int) - T, size=B)
    
    # inputs are text from i to i+T
    x = np.stack([text_int[i:i+T] for i in ix])
    
    # targets are text from i+1 to i+T+1
    y = np.stack([text_int[i+1:i+T+1] for i in ix])

    return jnp.array(x, dtype=jnp.int32), jnp.array(y, dtype=jnp.int32)

@jax.jit
def loss_and_metrics(logits, targets):
    """
    Compute cross-entropy loss and accuracy.
    Assumes `targets` contains only valid integer class ids in [0, V-1].

    Args:
    	logits: (B, T, V) float array of unnormalized scores.
    	targets: (B, T) integer array with ground-truth class ids.

    Returns:
      	loss: scalar average cross-entropy over all positions.
      	metrics: dict with keys "loss" and "acc" (both scalars).
    """
    
    # Flatten batch/time dims so optax works on shape (N, V) and (N,)
    vocab = logits.shape[-1] # obtain vocabulary size V from last dimension
    flat_logits = logits.reshape(-1, vocab) # Flatten to (B*T, V)
    flat_targets = targets.reshape(-1) # Flatten to (B*T,)

    # Per-position cross-entropy, then mean over all positions
    per_pos = optax.softmax_cross_entropy_with_integer_labels(flat_logits, flat_targets) # Obtain (B*T,) array of losses
    loss = per_pos.mean() # Obtain scalar loss

    # prediction over all positions
    preds = jnp.argmax(logits, axis=-1) # (B, T) predicted class ids
    
    # compute accuracy over only the last position
    is_match = preds == targets # (B, T) boolean array
    
    # Accuracy over all positions
    acc_all = jnp.mean(is_match.astype(jnp.float32))
    
    # Accuracy over only last position
    acc_last = jnp.mean(is_match.astype(jnp.float32)[:,-1])

    return loss, {"loss": loss, "acc": acc_all, "acc_last": acc_last}

def train_step(model, params, opt_state, x, y, tx, *, rng):
    """
    Single optimization step using optax optimizer.

    Args:
		model: Flax/Linen Module with .apply; forward signature like:
			   model.apply({'params': params}, tokens, deterministic=...)
    	params: pytree of model parameters.
    	opt_state: optax optimizer state corresponding to `params`.
    	x: (B, T) int array input tokens.
    	y: (B, T) int array target tokens.
    	tx: optax.GradientTransformation (already initialized).

    Returns:
    	new_params: updated parameters after one gradient step.
    	new_opt_state: updated optimizer state.
    	metrics: dict of scalar metrics (loss, acc).
    """
    
	# define loss function for computing gradients
    def loss_fn(params):
        logits = model.apply({"params": params}, x, deterministic=False, rngs={"dropout": rng})
        loss, metrics = loss_and_metrics(logits, y)
        return loss, metrics

    # compute gradients (loss is scalar, metrics is auxiliary)
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

    # optax update: compute parameter updates and new optimizer state
    updates, new_opt_state = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state, metrics

# JIT compile the train_step function for efficiency
train_step = jax.jit(train_step, static_argnames=('model','tx'))

# ----------------------------------------- #
# Autoregressive token generation Functions #
# ----------------------------------------- #	

def trim_or_pad_context(context , block_size, pad_id = None):
    """
    Ensure `context` has shape (B, block_size) by trimming on the left if too long
    or left-padding if too short.

    Args:
    	context: int32 array (B, S)
    	block_size: target window length
    	pad_id: if provided, use this id for left-padding. If None, pads with 0.

    Returns:
    	ctx: int32 array (B, block_size)
    """
    
    B, S = context.shape # get batch size and current sequence length
    
    # if longer than block, keep the rightmost tokens
    if S > block_size:
        return context[:, -block_size:]

    # if shorter, left-pad to fixed length
    pad_len = block_size - S
    if pad_len == 0:
        return context

    pad_token = 0 if pad_id is None else int(pad_id) # determine pad token id
    pad = jnp.full((B, pad_len), pad_token, dtype=jnp.int32) # (B, pad_len) padding array
    
    return jnp.concatenate([pad, context], axis=1)  # (B, block_size)

def sample_categorical(rng, logits, temperature) -> jnp.ndarray:
    """
    Sample indices from a categorical distribution parameterized by `logits`.

    Args:
      rng: PRNGKey
      logits: float array (B, V), unnormalized scores
      temperature: >0. Lower = greedier, Higher = more random.

    Returns:
      idx: int array (B,)
    """
    
    # clip temperature to avoid divide-by-zero
    temp = jnp.clip(jnp.asarray(temperature, dtype=logits.dtype), 1e-6, None)
    
    # make logits numerically stable
    logits -= logits.max(axis=-1, keepdims=True)
    
    # apply temperature
    logits = logits / temp
    
    return jax.random.categorical(rng, logits, axis=-1).astype(jnp.int32)

def generate_tokens(model, params, rng, context, length, *, block_size, temperature, sample, pad_id, deterministic):
    """
    Generate `length` new tokens autoregressively from a starting `context`.

    Args:
    	model: Flax/Linen Module with .apply; forward signature like:
             model.apply({'params': params}, tokens, deterministic=...)
    	params: model parameter pytree
    	rng: PRNGKey
    	context: int32 array (B, S), S <= block_size (if longer, we crop)
    	length: number of new tokens to generate
    	block_size: model's context window (max sequence length the model attends to)
    	temperature: sampling temperature (>0). Ignored for greedy.
    	sample: True -> sample from distribution; False -> greedy argmax
    	pad_id: if provided, used when left-padding short contexts
    	deterministic: pass through to model (True disables dropout inside model)

    Returns:
    	tokens: int32 array (B, length) — generated ids only (not including the prompt)
    """
    
    # 1) normalize context to fixed window length
    ctx = trim_or_pad_context(context.astype(jnp.int32), block_size, pad_id) # (B, block_size)
    B = ctx.shape[0] # batch size

    # optional rngs for model.apply calls
    def _apply_forward(tokens, rng_in, deterministic):
        if deterministic:
            return model.apply({'params': params}, tokens, deterministic=True)
        else:
            # if model uses dropout at inference, provide rngs
            return model.apply({'params': params}, tokens, deterministic=False, rngs={'dropout': rng_in})

    # 1) one autoregressive step (called repeatedly by lax.scan)
    def _step(carry, _):
        
        rng_loop, cur_ctx = carry # cur_ctx: (B, block_size)
        # forward pass over the full window
        logits = _apply_forward(cur_ctx, rng_loop, deterministic) # (B, block_size, V)
        last_logits = logits[:, -1, :] # (B, V) — distribution for the next token

        # sample or take argmax
        rng_loop, subkey = jax.random.split(rng_loop)
        if sample:
            next_tok = sample_categorical(subkey, last_logits, temperature=temperature)
        else:
            next_tok = jnp.argmax(last_logits, axis=-1).astype(jnp.int32) # greedy

        # slide the window: drop oldest, append new token
        next_tok_col = next_tok[:, None] # (B, 1)
        new_ctx = jnp.concatenate([cur_ctx[:, 1:], next_tok_col], axis=1) # (B, block_size)

        return (rng_loop, new_ctx), next_tok_col # carry, y

    # 2) run the loop for `length` steps (efficiently captures all generated tokens)
    (rng_final, ctx_final), tokens = jax.lax.scan(_step, (rng, ctx), xs=None, length=length)
    
    # tokens shape: (length, B, 1) -> (B, length)
    tokens = tokens.squeeze(-1).transpose(1, 0)
    
    return tokens

def decode(encoded_text, int_to_chars):
    """
    Decode integer representation back to text
    """

    generated_text = ''.join(int_to_chars.get(int(i), '?') for i in list(encoded_text[0]))

    return generated_text