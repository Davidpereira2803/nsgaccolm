import sys
import time
import math
from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from colm import COLM, COLMConfig, ModelConfig, TrainingConfig, RuntimeConfig
from colm._training import _SequenceDataset, _collator, _get_lr, _set_lr
from colm._model import GPT
from colm.constants import RANDOM_SEED

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

class _TimeLimitedCOLM(COLM):

    def set_time_limit(self, seconds: float) -> None:
        # Store a wall-clock time budget for training
        self._time_limit = seconds

    def _train_model(self, train_arr, val_arr, save_path=None, with_evaluation=True):
        # Training loop adapted to stop when the time budget is reached
        torch.manual_seed(RANDOM_SEED)

        time_limit = getattr(self, "_time_limit", None)
        start_time = time.time()

        vocab_size = self.meta["vocab_size"]
        seq_len    = self.meta["sequence_length"]
        bos_idx    = self.meta["bos_token_id"]
        eos_idx    = self.meta["eos_token_id"]

        train_dataset = _SequenceDataset(train_arr, eos_idx)
        val_dataset = (
            _SequenceDataset(val_arr, eos_idx)
            if with_evaluation and len(val_arr) > 0
            else None
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=_collator,
            pin_memory=self.config.runtime.device.startswith("cuda"),
        )

        # Build GPT model from COLM config
        gpt_cfg = self.config.model.to_gpt_config(vocab_size, seq_len, pad_token_id=bos_idx)
        model = GPT(gpt_cfg)
        device = self.config.runtime.device

        init_from = self.config.runtime.init_from
        if init_from != "scratch":
            ckpt_path = Path(init_from)
            if ckpt_path.is_file():
                # Warm-start from checkpoint when provided
                print(f"[COLM] Warm-starting from: {ckpt_path}")
                loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                model.load_state_dict(loaded["state_dict"], strict=True)
            else:
                print(f"[COLM] Checkpoint not found at {ckpt_path} — starting from scratch.")
        else:
            print("[COLM] Starting from scratch.")

        model.to(device)

        if hasattr(torch, "compile"):
            # Optional optimization with torch.compile() for faster training 
            print("[COLM] Compiling model with torch.compile() …")
            model = torch.compile(model)

        model.train()

        # Separate parameters for weight decay handling
        decay_params    = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
        no_decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]

        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params,    "weight_decay": self.config.training.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.config.training.learning_rate,
            betas=(0.9, 0.95),
            fused=device.startswith("cuda"),
        )

        cfg_tr = self.config.training

        global_step   = 0
        best_val_loss = float("inf")
        evals_no_imp  = 0
        stop_training = False
        has_saved     = False

        def cycle(loader):
            # Infinite cycle over the DataLoader, restarting it when exhausted
            while True:
                yield from loader

        data_iter      = cycle(train_loader)
        smoothed_loss  = 0.0
        smoothing      = 0.9
        log_loss_sum   = 0.0
        log_loss_count = 0

        pbar = tqdm(total=cfg_tr.max_iters, desc="Training", unit="step")
        optimizer.zero_grad()

        while global_step < cfg_tr.max_iters and not stop_training:

            lr = _get_lr(global_step, cfg_tr.warmup_steps, cfg_tr.max_iters,
                         cfg_tr.learning_rate, cfg_tr.lr_scheduler_type)
            _set_lr(optimizer, lr)

            step_loss = 0.0
            for _ in range(cfg_tr.gradient_accumulation_steps):
                # Gradient accumulation loop to effectively increase batch size without extra memory usage
                batch          = next(data_iter)
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["labels"].to(device)
                out            = model(input_ids, attention_mask=attention_mask, labels=labels)
                scaled_loss    = out.loss / cfg_tr.gradient_accumulation_steps
                scaled_loss.backward()
                step_loss += out.loss.item()

            step_loss /= cfg_tr.gradient_accumulation_steps

            if cfg_tr.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg_tr.grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            global_step    += 1
            log_loss_sum   += step_loss
            log_loss_count += 1

            # Exponential moving average for logging
            smoothed_loss = (
                step_loss if smoothed_loss == 0.0
                else smoothing * smoothed_loss + (1 - smoothing) * step_loss
            )

            pbar.update(1)

            # Respect the wall-clock time budget if set
            if time_limit is not None and (time.time() - start_time) >= time_limit:
                tqdm.write(f"[COLM] Time limit reached ({time_limit:.0f}s) — stopping after step {global_step}.")
                stop_training = True

            if global_step % cfg_tr.log_interval == 0:
                mean_loss      = log_loss_sum / log_loss_count
                log_loss_sum   = 0.0
                log_loss_count = 0
                tqdm.write(
                    f"[COLM] step {global_step:>6} | "
                    f"loss={mean_loss:.4f} (ema={smoothed_loss:.4f}) | lr={lr:.2e}"
                )

            if global_step % cfg_tr.eval_interval == 0:
                # Peridic evaluation and early-stopping
                if with_evaluation and val_dataset is not None:
                    val_loss = self._evaluate(model, val_dataset, device)
                    tqdm.write(
                        f"[COLM] step {global_step:>6} | "
                        f"val_loss={val_loss:.4f} (best={best_val_loss:.4f})"
                    )
                    improved = val_loss < best_val_loss - cfg_tr.early_stopping_threshold
                    if improved:
                        best_val_loss = val_loss
                        evals_no_imp  = 0
                        if save_path is not None:
                            self._save_checkpoint(model, save_path)
                            has_saved = True
                            tqdm.write(f"[COLM] ✓ val_loss={val_loss:.4f} → {save_path}")
                    else:
                        evals_no_imp += 1
                        tqdm.write(f"[COLM] No improvement {evals_no_imp}/{cfg_tr.early_stopping_patience}")
                        if evals_no_imp >= cfg_tr.early_stopping_patience:
                            tqdm.write("[COLM] Early stopping triggered.")
                            stop_training = True
                else:
                    if save_path is not None:
                        # Save checkpoint at regular intervals even without evaluation
                        self._save_checkpoint(model, save_path)
                        has_saved = True
                        tqdm.write(f"[COLM] step {global_step} → {save_path}")

            model.train()

        pbar.close()
        self.model = model

        if save_path is not None and not has_saved:
            # Ensure a final checkpoint is stored
            self._save_checkpoint(model, save_path)
            print(f"[COLM] Final checkpoint at step {global_step} → {save_path}")

def load_permutations(path: str, n_jobs: int) -> np.ndarray:
    # Read one permutation per line from the specified file, validating format and adjusting for 1-based indexing if needed
    seqs = []
    with open(path, "r", encoding="utf-8") as f:
        for i, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            tokens = [int(x) for x in line.split()]
            if len(tokens) != n_jobs:
                print(f"Skipping line {i}: length {len(tokens)}, expected {n_jobs}")
                continue
            if min(tokens) >= 1:
                tokens = [x - 1 for x in tokens]
            seqs.append(tokens)
    if not seqs:
        raise RuntimeError(f"No valid permutations found in {path}")
    return np.array(seqs, dtype=np.int64)

def main():
    parser = argparse.ArgumentParser(description="Train COLM on raw permutations")
    parser.add_argument("--corpus", type=str, required=True, help="File with permutations (one per line)")
    parser.add_argument("--n-jobs", type=int, default=50)
    parser.add_argument("--output", type=str, default="colm_model_perm.pth")
    parser.add_argument("--max-iters", type=int, default=0, help="Training steps (0 = use --time-limit)")
    parser.add_argument("--time-limit", type=float, default=600.0, help="Wall-clock training budget (s)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=6e-4)
    args = parser.parse_args()

    print(f"Loading permutations from {args.corpus}...")
    data = load_permutations(args.corpus, args.n_jobs)
    print(f"Loaded {len(data)} permutations of length {args.n_jobs}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_iters = args.max_iters if args.max_iters > 0 else 10_000_000

    # Build a compact COLM config used for permutation training
    cfg = COLMConfig(
        model=ModelConfig(n_layer=14, n_head=12, n_embd=768, dropout=0.5),
        training=TrainingConfig(
            batch_size=args.batch_size,
            max_iters=max_iters,
            eval_interval=200,
            log_interval=50,
            learning_rate=args.lr,
        ),
        runtime=RuntimeConfig(device=device, init_from="scratch"),
    )

    colm = _TimeLimitedCOLM(config=cfg)
    if args.max_iters == 0:
        # Use wall-clock time limit for training if max_iters is not set
        colm.set_time_limit(args.time_limit)
        print(f"Training with {args.time_limit:.0f}s time limit (completes current step before stopping)")
    else:
        print(f"Training for {max_iters} iterations")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    colm.train(data, val_split=0.05, save_path=str(out_path))

if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())
    main()
