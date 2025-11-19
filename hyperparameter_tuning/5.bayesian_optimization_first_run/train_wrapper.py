import time
import optuna
import jax
import jax.numpy as jnp
import optax

from hyperparameter_tuning.experiment_setup.experiment_model import DecoderOnlyTransformer
import hyperparameter_tuning.experiment_setup.experiment_utils as fn


def run_full_training(train_data_text, output_file, config, trial=None, global_start_time=None, global_time_limit=None):
    """
    Wrapper for running one hyperparameter trial.
    Returns final validation loss.

    args:
        config (dict): Configuration dictionary for training.
        trial (optuna.trial.Trial, optional): Optuna trial object for pruning.
    
    returns:
        float: Final validation loss after training.
    """

    if global_start_time is not None and global_time_limit is not None:
        elapsed_time = time.time() - global_start_time
        if elapsed_time >= global_time_limit:
            # Log to file
            with open(output_file, 'a') as log_file:
                log_file.write(
                    f"[Trial {trial.number if trial else '-'}] "
                    f"Skipped entirely due to global time limit "
                    f"({elapsed_time/3600:.2f}h > {global_time_limit/3600:.2f}h).\n"
                )
            print(
                f"[Trial {trial.number if trial else '-'}] "
                f"Skipped entirely due to global time limit "
                f"({elapsed_time/3600:.2f}h > {global_time_limit/3600:.2f}h)."
            )

            # Optuna must prune this trial cleanly
            if trial is not None:
                raise optuna.TrialPruned()

            # If trial is None, return a dummy high loss
            return float("inf")

    # ----------------------
    # 1. Extract config
    # ----------------------
    model_cfg = config["model"]
    train_cfg = config["training"]
    throughput_cfg = config["throughput"]

    batch_size = train_cfg["batch_size"]
    seq_len = model_cfg["seq_len"]
    aux_loss = model_cfg["use_auxiliary_loss"]
    aux_weight = model_cfg["aux_weight"]
    compute_budget_hours = throughput_cfg["compute_budget_hours"]
    loss_type = model_cfg["loss_type"]
    val_fraction = train_cfg["val_fraction"]
    
    # ----------------------
    # 2. Load dataset
    # --------------------
    chars = sorted(set(train_data_text))
    chars_to_int = {ch: i for i, ch in enumerate(chars)}

    train_text, val_text = fn.split_train_val(train_data_text, val_fraction=val_fraction)
    train_data = fn.encode(train_text, chars_to_int)
    val_data = fn.encode(val_text, chars_to_int)

    # RNG
    rng = jax.random.PRNGKey(config["seed"])

    # ----------------------
    # 3. Build model
    # ----------------------

    model, params, constants = fn.create_train_state(
        rng,
        vocab_size=model_cfg["vocab_size"],
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        mlp_ratio=model_cfg["mlp_ratio"],
        seq_len=seq_len,
        dropout=model_cfg["dropout"],
        aux_loss=aux_loss,
        num_aux_heads=model_cfg["aux_heads"],
        mixed_precision=model_cfg["mixed_precision"],
        attention_type=model_cfg["attention_type"],
        pos_encoding=model_cfg["pos_encoding"]
    )

    # ----------------------
    # 5. Determine allowable steps
    # ----------------------

    base_lr = train_cfg["learning_rate"]
    weight_decay = model_cfg["weight_decay"]
    lr_schedule = train_cfg["lr_schedule"]
    optimizer_type = train_cfg["optimizer"]
    warmup_ratio = train_cfg.get("warmup_ratio", 0.0)
    label_smoothing = model_cfg.get("label_smoothing", 0.0)

    max_test_iters = throughput_cfg["max_test_iters"]
    max_test_time = throughput_cfg["max_test_time_in_seconds"]
    compute_budget_hours = throughput_cfg["compute_budget_hours"]

    dummy_scheduler = optax.constant_schedule(base_lr)
    dummy_optimizer = optax.adam(dummy_scheduler)
    dummy_opt_state = dummy_optimizer.init(params)

    with open(output_file, 'a') as log_file:
        log_file.write("\n" + "="*70 + "\n")
        print("\n" + "="*70)
        log_file.write(f"\n[Trial {trial.number if trial else '-'}] Starting throughput calculation...\n")
        print(f"\n[Trial {trial.number if trial else '-'}] Starting throughput calculation...")

    _, iter_max = fn.calculate_throughput(
        max_test_iters=max_test_iters,
        max_test_time=max_test_time,      # benchmark length
        model=model,
        params=params,
        opt_state=dummy_opt_state,
        optimizer=dummy_optimizer,
        rng=rng,
        batch_size=batch_size,
        seq_len=seq_len,
        compute_budget=compute_budget_hours,
        train_data=train_data,
        loss_type=loss_type,
        aux_loss=aux_loss,
        aux_weight=aux_weight,
        constants=constants,
        label_smoothing=label_smoothing
    )

    iter_max = int(iter_max)
    iter_max = max(1, iter_max)

    with open(output_file, 'a') as log_file:
        log_file.write(f"[Trial {trial.number if trial else '-'}] Throughput calculation done. "
                       f"Setting iter_max = {iter_max:,}\n")

    # ----------------------
    # 4. Build optimizer
    # ----------------------

    if lr_schedule == "constant":
        scheduler = optax.constant_schedule(base_lr)
    elif lr_schedule == "cosine":
        scheduler = optax.cosine_decay_schedule(
            init_value=base_lr,
            decay_steps=iter_max
        )
    elif lr_schedule == "warmup_decay":
        scheduler = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=base_lr,
            warmup_steps=max(1, int(warmup_ratio * iter_max)),
            decay_steps=iter_max
        )


    if optimizer_type == "adam":
        optimizer = optax.adam(scheduler)
    elif optimizer_type == "adamw":
        optimizer = optax.adamw(scheduler, weight_decay=weight_decay)
    elif optimizer_type == "sgd":
        optimizer = optax.sgd(scheduler, momentum=0.9)

    opt_state = optimizer.init(params)

    # ----------------------
    # 6. Training loop
    # ----------------------
    def eval_step(params, x, y, constants, loss_type, aux_loss, aux_weight, label_smoothing):
        
        output = model.apply({"params": params, "constants": constants}, x, deterministic=True)
        logits = output["logits"]
        aux_logits = output.get("aux_logits", None)

        loss, _ = fn.loss_and_metrics(
            logits=logits,
            targets=y,
            loss_type=loss_type, 
            aux_loss=aux_loss, 
            aux_weight=aux_weight, 
            aux_logits=aux_logits,
            label_smoothing=label_smoothing
        )

        return loss
    
    val_every = max(1, iter_max // 100) # 1% of total iters
    log_every = max(1, iter_max // 20) # 5% of total iters
    num_val_batches = 5 # to average validation loss over to reduce noise
    final_val_loss = None

    with open(output_file, 'a') as log_file:
        log_file.write(f"\n[Trial {trial.number if trial else '-'}] Starting training loop.\n")
        print(f"\n[Trial {trial.number if trial else '-'}] Starting training loop.")
        log_file.write(f"  iter_max = {iter_max:,}\n")
        print(f"  iter_max = {iter_max:,}")
        log_file.write(f"lr_schedule = {lr_schedule}\n")
        print(f"lr_schedule = {lr_schedule}")
        if lr_schedule == 'warmup_decay':
            log_file.write(f". warmup_ratio  = {warmup_ratio}\n")
            print(f". warmup_ratio  = {warmup_ratio}")
        log_file.write(f"  weight_decay = {weight_decay}\n")
        print(f"  weight_decay = {weight_decay}")
        log_file.write(f"  learning_rate = {base_lr}\n\n")
        print(f"  learning_rate = {base_lr}\n")
        log_file.write("="*70 + "\n")
        print("="*70)

    for it in range(iter_max):

        # Check global time limit
        if global_start_time is not None and global_time_limit is not None:
            elapsed_time = time.time() - global_start_time
            if elapsed_time >= global_time_limit:
                with open(output_file, 'a') as log_file:
                    log_file.write(f"[Trial {trial.number if trial else '-'}] Stopped early at step {it} due to global time limit.\n")
                    log_file.write(f"Global time limit reached. ({elapsed_time/3600:.2f}h > {global_time_limit/3600:.2f}h)\n")
                    log_file.write("Pruning current trial.\n")
                    print(f"[Trial {trial.number if trial else '-'}] Stopped early at step {it} due to global time limit.")
                    print(f"Global time limit reached. ({elapsed_time/3600:.2f}h > {global_time_limit/3600:.2f}h)")
                    print("Pruning current trial.")

                if trial is not None:
                    raise optuna.TrialPruned()
                else:
                    return float(final_val_loss) if final_val_loss is not None else float('inf')

        rng, sub = jax.random.split(rng)
        inputs, targets = fn.get_batch(train_data, batch_size, seq_len)

        new_params, new_opt_state, _ = fn.train_step(
            model=model,
            params=params,
            opt_state=opt_state,
            x=inputs,
            y=targets,
            tx=optimizer,
            rng=sub,
            loss_type=loss_type,
            aux_loss=aux_loss,
            aux_weight=aux_weight,
            constants=constants,
            label_smoothing=label_smoothing
        )

        params = new_params
        opt_state = new_opt_state

        # Validation
        if (it % val_every == 0) or (it == iter_max - 1):

            val_losses = []
            for _ in range(num_val_batches):
                val_inputs, val_targets = fn.get_batch(val_data, batch_size, seq_len)
                loss = eval_step(
                    params,
                    val_inputs,
                    val_targets,
                    constants,
                    loss_type,
                    aux_loss,
                    aux_weight,
                    label_smoothing
                )
                val_losses.append(loss)

            val_loss = float(jnp.mean(jnp.stack(val_losses)))
            final_val_loss = val_loss

            if trial is not None:
                trial.report(val_loss, it)
                if trial.should_prune():

                    with open(output_file, 'a') as log_file:
                        log_file.write(f"[Trial {trial.number}] Pruned at step {it}.\n")
                        print(f"[Trial {trial.number}] Pruned at step {it}.")
                    raise optuna.TrialPruned()
                
        # Log the step at every 10% of progress
        if (it % log_every == 0) or (it == iter_max - 1):
            pct = 100 * (it + 1) / iter_max

            with open(output_file, 'a') as log_file:
                log_file.write(f"    Step {it+1:5d}/{iter_max}  ({pct:5.1f}%)"
                               f" | val_loss = {val_loss:.4f}\n")
                print(f"    Step {it+1:5d}/{iter_max}  ({pct:5.1f}%)"
                      f" | val_loss = {val_loss:.4f}")

    with open(output_file, 'a') as log_file:
        log_file.write(f"[Trial {trial.number}] Completed with final val_loss = {val_loss:.4f}\n")
        log_file.write("-"*70 + "\n")    
        print(f"[Trial {trial.number}] Completed with final val_loss = {val_loss:.4f}")
        print("-"*70 + "\n")

    return float(final_val_loss)
