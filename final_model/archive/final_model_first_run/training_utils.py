"""
In this file, we define the functions required for performing ablations character-level language model.

Using JAX and Flax for efficient computation, 
The functions are defined based on each component of the training and evaluation pipeline as follows:

1. Data Preprocessing:
    - encode -> Text encoding to integer representation

2. Model Creation:
	- count_parameters -> Count no. of model parameters for reporting
    - create_train_state -> Initialize model and parameters
    - create_dummy_optimizer -> Create dummy optimizer for throughput benchmarking
    - initialize_optimizer -> Setup optax optimizer with specified hyperparameters

3. Training & Evaluation Loop:
	- get_batch -> Create random training / validation batches
    - loss_and_metrics -> Compute loss and accuracy metrics
    - train_step -> Single optimization step with optax optimizer
    - calculate_throughput -> Benchmark training throughput and estimate max steps within compute budget
    - evaluate_on_test_set -> Evaluate model on test dataset

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
import time
import model_architecture as model

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

def create_train_state(
        rng, 
        *,
        vocab_size = 27,
        d_model = 64,
        n_layers = 2, 
        n_heads = 4, 
        seq_len = 128, 
        mlp_ratio = 4, 
        dropout = 0.0,
        aux_loss = False,
        num_aux_heads = 0,
        mixed_precision = False,
        attention_type = "MHA",
        pos_encoding = "none"
    ):
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
        aux_loss: whether to use auxiliary loss
        num_aux_heads: number of auxiliary heads for intermediate losses
        mixed_precision: whether to use mixed precision (float16) for model parameters
    
    Returns:
		model: initialized model
        params: initialized model parameters
    """

    if mixed_precision:
        dtype = jnp.float16
    else:
        dtype = jnp.float32

    model_obj = model.DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        mlp_ratio=mlp_ratio,
        seq_len=seq_len,
        dropout=dropout,
        attention_type=attention_type,
        pos_encoding=pos_encoding,
        aux_loss=aux_loss,
        num_aux_heads=num_aux_heads,
        dtype=dtype
    )

    # Create dummy input for initialization
    dummy_input = jnp.ones((1, min(16, seq_len)), dtype=jnp.int32)

    # Pass the dummy input through the model to initialize parameters
    variables = model_obj.init({"params": rng}, dummy_input, deterministic=False)

    print(f"[create_train_state] Model initialized with {count_parameters(variables['params']) :,} parameters.")

    params = variables['params'] # Extract parameters from the initialized variables
    constants = variables.get('constants', {}) # Extract constants if any

    return model_obj, params, constants

def create_dummy_optimizer(learning_rate, optimizer_type, weight_decay, params):
    """
    Create a dummy optax optimizer for benchmarking throughput.

    Args:
        learning_rate: learning rate for the optimizer
        lr_schedule: learning rate schedule type
        params: model parameters to initialize the optimizer state 
    Returns:
        optax.GradientTransformation: Initialized optimizer
    """

    dummy_scheduler = optax.constant_schedule(learning_rate)

    if optimizer_type == "adam":
        dummy_optimizer = optax.adam(dummy_scheduler)
    elif optimizer_type == "adamw":
        dummy_optimizer = optax.adamw(dummy_scheduler, weight_decay=weight_decay)
    elif optimizer_type == "sgd":
        dummy_optimizer = optax.sgd(dummy_scheduler, momentum=0.9)

    dummy_opt_state = dummy_optimizer.init(params)

    return dummy_optimizer, dummy_opt_state

def initialize_optimizer(params, iter_max, optimizer_type, lr_schedule, learning_rate, weight_decay, warmup_ratio, grad_clip):
    """
    Initialize the optax optimizer based on the specified type and hyperparameters.

    Args:
        optimizer_type (str): Type of optimizer ('adam', 'adamw', 'sgd').
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay (L2 regularization) factor.
        grad_clip (float): Gradient clipping value (if > 0, gradients are clipped).
    Returns:
        optax.GradientTransformation: Initialized optimizer.
    """

    if lr_schedule == "constant":
        scheduler = optax.constant_schedule(learning_rate)
    elif lr_schedule == "cosine":
        scheduler = optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=iter_max
        )
    elif lr_schedule == "warmup_decay":
        scheduler = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=max(1, int(warmup_ratio * iter_max)),
            decay_steps=iter_max
        )

    if optimizer_type == "adam":
        optimizer = optax.adam(scheduler)
    elif optimizer_type == "adamw":
        optimizer = optax.adamw(scheduler, weight_decay=weight_decay)
    elif optimizer_type == "sgd":
        optimizer = optax.sgd(scheduler, momentum=0.9)

    # Add gradient clipping if specified
    if grad_clip is not None and grad_clip != "none":
        optimizer = optax.chain(
            optax.clip_by_global_norm(grad_clip),
            optimizer
        )

    opt_state = optimizer.init(params)

    print(
        f"[initialize_optimizer] Initialized {optimizer_type} optimizer configurations:"
        f"learning_rate_schedule={lr_schedule}, learning_rate={learning_rate}, weight_decay={weight_decay}, grad_clip={grad_clip}"
        f"weight_decay applied: {'yes' if optimizer_type == 'adamw' else 'no'}, grad_clip applied: {'yes' if grad_clip is not None and grad_clip != 'none' else 'no'}"
        f"warmup_ratio={warmup_ratio}, total_iterations={iter_max}"
    )

    return optimizer, opt_state

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

def loss_and_metrics(logits, targets, loss_type, aux_loss, aux_weight, aux_logits, label_smoothing):
    """
    Compute loss & metrics given model logits and ground-truth targets.
    Assumes `targets` contains only valid integer class ids in [0, V-1].

    This supports three types of loss:
    - 'cross_entropy': compute cross-entropy loss over all positions
    - 'last_position': compute cross-entropy loss only on the last position
    - 'auxiliary_loss': include auxiliary losses from intermediate layers

    Args:
    	logits: (B, T, V) float array of unnormalized scores.
    	targets: (B, T) integer array with ground-truth class ids.
        loss_type: str, type of loss to compute.
        aux_loss: boolean, whether to use auxiliary loss
        aux_weight: float, weight for auxiliary loss
        aux_logits: list of (B, T, V) float arrays for auxiliary losses
        use_label_smoothing: boolean, whether to apply label smoothing
        label_smoothing: float, smoothing factor for label smoothing (default 0.1)

    Returns:
      	loss: scalar average cross-entropy over all positions.
      	metrics: dict with keys "loss" and "acc" (both scalars).
    """

    def smooth_labels(one_hot, smoothing):
        K = one_hot.shape[-1]
        return one_hot * (1 - smoothing) + smoothing / K

    B, T, V = logits.shape # batch size, sequence length, vocabulary size
    losses = []

    ## Main loss computation ##
    if loss_type == 'cross_entropy':
        # Compute cross-entropy loss over all positions
        new_logits = logits.reshape(-1, V) # Flatten to (B*T, V)
        new_targets = targets.reshape(-1) # Flatten to (B*T,)
    
    elif loss_type == 'last_position':
        # Compute cross-entropy loss only on the last position
        new_logits = logits[:, -1, :] # (B, V)
        new_targets = targets[:, -1] # (B,)

    if label_smoothing > 0.0:
        one_hot_targets = jax.nn.one_hot(new_targets, V, dtype=logits.dtype)
        smoothed_targets = smooth_labels(one_hot_targets, label_smoothing)
        main_loss = optax.softmax_cross_entropy(new_logits, smoothed_targets) # Obtain (B*T,) array of losses
        main_loss = main_loss.mean() # Obtain scalar loss
    else:
        main_loss = optax.softmax_cross_entropy_with_integer_labels(new_logits, new_targets) # Obtain (B*T,) array of losses
        main_loss = main_loss.mean() # Obtain scalar loss
    
    losses.append(main_loss)

    ## Auxiliary loss computation ##
    if aux_loss and aux_logits is not None:

        aux_losses = []

        for logits in aux_logits:
            flat = logits.reshape(-1, V) # Flatten to (B*T, V)
            flat_targets = targets.reshape(-1) # Flatten to (B*T,)
            aux = optax.softmax_cross_entropy_with_integer_labels(flat, flat_targets) # Obtain (B*T,) array of losses
            loss = aux.mean()
            aux_losses.append(loss)

        aux_total = sum(aux_losses) / len(aux_losses)
        losses.append(aux_total * aux_weight)

    total_loss = sum(losses) # Final total loss

    ## Accuracy Metrics ##

    # prediction over all positions
    preds = jnp.argmax(logits, axis=-1) # (B, T) predicted class ids
    
    # compute accuracy over only the last position
    is_match = (preds == targets) # (B, T) boolean array
    
    # Accuracy over all positions
    acc_all = jnp.mean(is_match.astype(jnp.float32))
    
    # Accuracy over only last position
    acc_last = jnp.mean(is_match.astype(jnp.float32)[:,-1])

    return total_loss, {"loss": total_loss, "acc": acc_all, "acc_last": acc_last}

def train_step(model, params, constants, opt_state, x, y, tx, rng, loss_type, aux_loss, aux_weight, label_smoothing):
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
        rng: JAX PRNGKey for randomness (e.g., dropout).
        loss_type: str, type of loss to compute.
        aux_loss: boolean, whether to use auxiliary loss
        aux_weight: float, weight for auxiliary loss
        label_smoothing: float, smoothing factor for label smoothing (default 0.1)

    Returns:
    	new_params: updated parameters after one gradient step.
    	new_opt_state: updated optimizer state.
    	metrics: dict of scalar metrics (loss, acc).
    """
    
	# define loss function for computing gradients
    def loss_fn(params):

        output = model.apply({"params": params, "constants": constants}, x, deterministic=False, rngs={"dropout": rng})

        final_logits = output['logits']
        aux_logits = output.get('aux_logits', None)

        loss, metrics = loss_and_metrics(final_logits, y, loss_type, aux_loss, aux_weight, aux_logits, label_smoothing=label_smoothing)

        return loss, metrics

    # compute gradients (loss is scalar, metrics is auxiliary)
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

    # optax update: compute parameter updates and new optimizer state
    updates, new_opt_state = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state, metrics

# JIT compile the train_step function for efficiency
train_step = jax.jit(train_step, static_argnames=('model', 'tx', 'aux_loss', 'loss_type', 'label_smoothing'))

def calculate_throughput(
        max_test_iters, 
        max_test_time,
        model, 
        params, 
        opt_state, 
        optimizer, 
        rng, 
        batch_size, 
        seq_len, 
        compute_budget, 
        train_data,
        loss_type, 
        aux_loss, 
        aux_weight,
        constants,
        label_smoothing
    ):
    """
    Calculate training throughput (tokens/second) and estimate max steps within compute budget.
    
    Args:
        max_iters: maximum number of iterations to run
        max_time: maximum time (in seconds) to run
        model: model
        params: model parameters
        opt_state: optimizer state
        optimizer: optax optimizer
        rng: JAX PRNGKey
        batch_size: batch size B
        seq_len: sequence length T
        compute_budget: compute budget in hours
        train_data: training data (encoded as integers)
        loss_type: str, type of loss to compute.
        aux_loss: boolean, whether to use auxiliary loss
        aux_weight: float, weight for auxiliary loss
        constants: model constants
        label_smoothing: float, smoothing factor for label smoothing (default 0.1)
    """

    time_start = time.time() # Record start time

    test_params = params # Copy of params for testing
    test_opt_state = opt_state # Copy of opt_state for testing

    for it in range(max_test_iters): # Loop for max_iters

        inputs, targets = get_batch(train_data, batch_size, seq_len) # Get batch
        rng, sub = jax.random.split(rng) # Split RNG
        new_params, new_opt_state, _ = train_step( # Perform train step
            model = model, 
            params = test_params, 
            constants = constants,
            opt_state = test_opt_state, 
            x = inputs, 
            y = targets, 
            tx = optimizer,
            rng = sub, 
            loss_type = loss_type, 
            aux_loss = aux_loss, 
            aux_weight = aux_weight,
            label_smoothing = label_smoothing
        )

        # Update params and opt_state
        test_params = new_params
        test_opt_state = new_opt_state

        if time.time() - time_start > max_test_time: # Check time limit
            print(f"Stopping benchmark at iteration {it} due to time limit.")
            break

    t_end = time.time() # Record end time

    elapsed_time = t_end - time_start # Calculate elapsed time
    total_tokens = (it + 1) * batch_size * seq_len # Total tokens processed
    throughput = total_tokens / elapsed_time # Tokens per second

    print(f"Benchmark completed in {elapsed_time:.2f} seconds.")
    print(f"Total tokens processed: {total_tokens}")
    print(f"Throughput: {throughput:.2f} tokens/second")

    time_per_step = elapsed_time / (it + 1) # Time per step
    max_steps = (compute_budget * 60 * 60) // time_per_step # Estimate max steps
    print(f"Estimated max steps within compute budget: {max_steps}")

    return throughput, max_steps

def evaluate_on_test_set(rng, model, params, constants, test_data, batch_size, seq_len, loss_type, aux_loss, aux_weight, label_smoothing, n_batches):
    """
    Evaluate the model on the test dataset.
    The evaluation is done over `n_batches` batches and
    the mean loss and accuracies are reported.

    Args:
        rng: JAX PRNGKey
        model: Flax/Linen Module with .apply; forward signature
        params: model parameters
        constants: model constants
        test_data: encoded test dataset (1D array of token ids)
        batch_size: batch size for evaluation
        seq_len: sequence length for evaluation
        loss_type: str, type of loss to compute.
        aux_loss: boolean, whether to use auxiliary loss
        aux_weight: float, weight for auxiliary loss
        label_smoothing: float, smoothing factor for label smoothing (default 0.1)
    Returns:
        test_loss: scalar test loss
        test_acc: scalar test accuracy over all positions
        last_char_acc_test: scalar test accuracy over last position
    """

    total_loss = []
    total_acc = []
    total_last_char_acc = []

    for _ in range(n_batches):

        test_inputs, test_targets = get_batch(test_data, batch_size, seq_len)

        # Forward pass through the model
        output = model.apply({"params": params, "constants": constants}, test_inputs, deterministic=True, rngs={"dropout": rng})
        test_logits = output['logits']
        aux_logits = output.get('aux_logits', None)

        # Calculate loss and metrics
        test_loss, test_metrics = loss_and_metrics(
                logits = test_logits,
                targets = test_targets,
                loss_type = loss_type,
                aux_loss = aux_loss,
                aux_logits = aux_logits,
                aux_weight = aux_weight,
                label_smoothing = label_smoothing
            )
        
        test_acc = test_metrics['acc']
        last_char_acc_test = test_metrics['acc_last']

        total_loss.append(test_loss)
        total_acc.append(test_acc)
        total_last_char_acc.append(last_char_acc_test)

    print(
        f"[evaluate_on_test_set] Metrics are averaged over {n_batches} batches.\n" 
        f"Mean Test Loss: {np.mean(total_loss):.4f}, Mean Test Accuracy: {np.mean(total_acc):.4f}, Mean Last Character Test Accuracy: {np.mean(total_last_char_acc):.4f}\n"
        f"Min Test Acc: {np.min(total_acc):.4f}, Max Test Accuracy: {np.max(total_acc):.4f}, Max Last Character Test Accuracy: {np.max(total_last_char_acc):.4f}\n"
        f"Max Test Acc: {np.max(total_acc):.4f}, Min Test Accuracy: {np.min(total_acc):.4f}, Min Last Character Test Accuracy: {np.min(total_last_char_acc):.4f}\n"
        f"S.D of Test Acc: {np.std(total_acc):.4f}, S.D of Test Accuracy: {np.std(total_acc):.4f}, S.D of Last Character Test Accuracy: {np.std(total_last_char_acc):.4f}\n"
    )

    return total_loss, total_acc, total_last_char_acc
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

def generate_tokens(model, params, constants, rng, context, length, *, block_size, temperature, sample, pad_id, deterministic):
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
            return model.apply({'params': params, 'constants': constants}, tokens, deterministic=True)
        else:
            # if model uses dropout at inference, provide rngs
            return model.apply({'params': params, 'constants': constants}, tokens, deterministic=False, rngs={'dropout': rng_in})

    # 1) one autoregressive step (called repeatedly by lax.scan)
    def _step(carry, _):
        
        rng_loop, cur_ctx = carry # cur_ctx: (B, block_size)
        # forward pass over the full window
        output = _apply_forward(cur_ctx, rng_loop, deterministic)
        logits = output['logits'] # (B, block_size, V)
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