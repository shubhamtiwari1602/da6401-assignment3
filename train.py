"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
from tqdm import tqdm
import wandb

from model import Transformer, make_src_mask, make_tgt_mask
from dataset import Vocabulary, build_dataloaders
from lr_scheduler import NoamScheduler


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need".
    y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)
    PAD positions receive zero probability and are excluded from the loss.
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits : [batch * tgt_len, vocab_size]
        target : [batch * tgt_len]
        """
        log_probs = torch.log_softmax(logits, dim=-1)

        # Build smooth target distribution
        with torch.no_grad():
            smooth_dist = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 2))
            smooth_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            smooth_dist[:, self.pad_idx] = 0.0
            # zero out rows where target is PAD
            pad_mask = (target == self.pad_idx)
            smooth_dist[pad_mask] = 0.0

        loss = -(smooth_dist * log_probs).sum(dim=-1)
        non_pad = (~pad_mask).sum()
        return loss.sum() / non_pad.clamp(min=1)


# ══════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
    log_grad_norms: bool = False,
) -> float:
    """
    One epoch of training or evaluation.
    Returns average loss over the epoch.
    """
    model.train() if is_train else model.eval()
    total_loss, n_batches = 0.0, 0
    pad_idx = Vocabulary.PAD_IDX

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        pbar = tqdm(data_iter, desc=f"{'Train' if is_train else 'Val'} E{epoch_num+1}", leave=False)
        for src, tgt in pbar:
            src, tgt = src.to(device), tgt.to(device)

            src_mask = make_src_mask(src, pad_idx)
            tgt_in   = tgt[:, :-1]
            tgt_out  = tgt[:, 1:]
            tgt_mask = make_tgt_mask(tgt_in, pad_idx)

            logits = model(src, tgt_in, src_mask, tgt_mask)
            B, T, V = logits.shape
            loss = loss_fn(logits.reshape(B * T, V), tgt_out.reshape(B * T))

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                if log_grad_norms:
                    qk_norm = 0.0
                    cnt = 0
                    for layer in model.encoder.layers:
                        for pname, p in layer.self_attn.named_parameters():
                            if p.grad is not None and ("W_q" in pname or "W_k" in pname):
                                qk_norm += p.grad.norm().item()
                                cnt += 1
                    if cnt:
                        wandb.log({"grad_norm/qk": qk_norm / cnt})

                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            total_loss += loss.item()
            n_batches  += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(n_batches, 1)


# ══════════════════════════════════════════════════════════════════════
#  GREEDY DECODING
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.
    Returns [1, out_len] including start_symbol; stops at end_symbol.
    """
    model.eval()
    with torch.no_grad():
        memory = model.encode(src, src_mask)          # [1, src_len, d_model]
        ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys, pad_idx=Vocabulary.PAD_IDX).to(device)
            logits   = model.decode(memory, src_mask, ys, tgt_mask)  # [1, T, V]
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1, 1]
            ys = torch.cat([ys, next_tok], dim=1)
            if next_tok.item() == end_symbol:
                break

    return ys


# ══════════════════════════════════════════════════════════════════════
#  BLEU EVALUATION
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Corpus-level BLEU score (0–100) using evaluate.bleu.
    """
    from evaluate import load as eval_load
    bleu_metric = eval_load("bleu")

    model.eval()
    sos = Vocabulary.SOS_IDX
    eos = Vocabulary.EOS_IDX
    pad = Vocabulary.PAD_IDX

    predictions, references = [], []
    with torch.no_grad():
        for src, tgt in tqdm(test_dataloader, desc="BLEU eval", leave=False):
            src = src.to(device)
            src_mask = make_src_mask(src, pad).to(device)
            ys = greedy_decode(model, src, src_mask, max_len, sos, eos, device)

            pred_ids = ys[0].tolist()
            pred_toks = [tgt_vocab.lookup_token(i) for i in pred_ids
                         if i not in (sos, eos, pad)]

            ref_ids  = tgt[0].tolist()
            ref_toks = [tgt_vocab.lookup_token(i) for i in ref_ids
                        if i not in (sos, eos, pad)]

            predictions.append(" ".join(pred_toks))
            references.append([" ".join(ref_toks)])

    result = bleu_metric.compute(predictions=predictions, references=references)
    return result["bleu"] * 100.0


# ══════════════════════════════════════════════════════════════════════
#  CHECKPOINT UTILITIES
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    torch.save({
        "epoch": epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "model_config": {
            "src_vocab_size": model.src_embed.num_embeddings,
            "tgt_vocab_size": model.tgt_embed.num_embeddings,
            "d_model":   model.d_model,
            "N":         len(model.encoder.layers),
            "num_heads": model.encoder.layers[0].self_attn.num_heads,
            "d_ff":      model.encoder.layers[0].ffn.linear1.out_features,
            "dropout":   model.encoder.layers[0].dropout.p,
        },
    }, path)


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["epoch"]


# ══════════════════════════════════════════════════════════════════════
#  EXPERIMENT HELPERS
# ══════════════════════════════════════════════════════════════════════

def _build_model(cfg, src_vocab_size, tgt_vocab_size):
    return Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=cfg["d_model"],
        N=cfg["N"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        dropout=cfg["dropout"],
        learned_pos_enc=cfg.get("learned_pos_enc", False),
    )


def _train_run(
    cfg,
    train_loader,
    val_loader,
    test_loader,
    src_vocab,
    tgt_vocab,
    device,
    run_name: str,
    ckpt_path: str,
    log_grad_norms: bool = False,
    use_label_smoothing: bool = True,
):
    """Generic training run used by all experiments."""
    model = _build_model(cfg, len(src_vocab), len(tgt_vocab)).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0,
        betas=(0.9, 0.98), eps=1e-9
    )

    if cfg.get("fixed_lr"):
        scheduler = None
        for pg in optimizer.param_groups:
            pg["lr"] = cfg["fixed_lr"]
    else:
        scheduler = NoamScheduler(optimizer, d_model=cfg["d_model"], warmup_steps=cfg["warmup_steps"])

    smoothing = 0.1 if use_label_smoothing else 0.0
    loss_fn = LabelSmoothingLoss(len(tgt_vocab), Vocabulary.PAD_IDX, smoothing=smoothing)

    best_val_loss = float("inf")
    for epoch in range(cfg["epochs"]):
        train_loss = run_epoch(
            train_loader, model, loss_fn, optimizer, scheduler,
            epoch_num=epoch, is_train=True, device=device,
            log_grad_norms=(log_grad_norms and epoch == 0),
        )
        val_loss = run_epoch(
            val_loader, model, loss_fn, None, None,
            epoch_num=epoch, is_train=False, device=device,
        )

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"[{run_name}] Epoch {epoch+1}/{cfg['epochs']}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  lr={lr_now:.2e}")
        wandb.log({
            f"{run_name}/train_loss": train_loss,
            f"{run_name}/val_loss":   val_loss,
            f"{run_name}/lr":         lr_now,
            "epoch": epoch + 1,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, ckpt_path)

    # load best and evaluate BLEU
    load_checkpoint(ckpt_path, model)
    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device)
    print(f"[{run_name}] Test BLEU: {bleu:.2f}")
    wandb.log({f"{run_name}/test_bleu": bleu})
    return model, bleu


# ══════════════════════════════════════════════════════════════════════
#  ATTENTION MAP VISUALIZATION
# ══════════════════════════════════════════════════════════════════════

def log_attention_maps(model, src, src_vocab, tgt_vocab, device):
    """Extract last encoder layer attention weights and log heatmaps to W&B."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")

    model.eval()
    src = src.to(device)
    src_mask = make_src_mask(src, Vocabulary.PAD_IDX).to(device)

    with torch.no_grad():
        model.encode(src, src_mask)

    last_layer = model.encoder.layers[-1]
    attn_w = last_layer.self_attn.attn_weights  # [1, h, S, S]
    if attn_w is None:
        return

    src_tokens = [src_vocab.lookup_token(i.item()) for i in src[0]
                  if i.item() not in (Vocabulary.PAD_IDX,)]
    num_heads = attn_w.size(1)
    figs = []
    for h in range(num_heads):
        w = attn_w[0, h].cpu().numpy()[:len(src_tokens), :len(src_tokens)]
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(w, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(src_tokens))); ax.set_xticklabels(src_tokens, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(src_tokens))); ax.set_yticklabels(src_tokens, fontsize=8)
        ax.set_title(f"Encoder Head {h+1}")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        figs.append(wandb.Image(fig, caption=f"Encoder Head {h+1}"))
        plt.close(fig)
    wandb.log({"attention_maps": figs})


# ══════════════════════════════════════════════════════════════════════
#  PREDICTION CONFIDENCE LOGGING
# ══════════════════════════════════════════════════════════════════════

def log_prediction_confidence(model, val_loader, tgt_vocab, device, run_name, n_batches=50):
    """Log avg softmax prob of the correct token for a subset of val batches."""
    model.eval()
    confidences = []
    pad_idx = Vocabulary.PAD_IDX

    with torch.no_grad():
        for i, (src, tgt) in enumerate(val_loader):
            if i >= n_batches:
                break
            src, tgt = src.to(device), tgt.to(device)
            tgt_in  = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            src_mask = make_src_mask(src, pad_idx)
            tgt_mask = make_tgt_mask(tgt_in, pad_idx)
            logits = model(src, tgt_in, src_mask, tgt_mask)
            probs  = torch.softmax(logits, dim=-1)
            B, T, V = probs.shape
            probs_flat  = probs.reshape(B * T, V)
            target_flat = tgt_out.reshape(B * T)
            mask = target_flat != pad_idx
            correct_probs = probs_flat[mask].gather(1, target_flat[mask].unsqueeze(1)).squeeze(1)
            confidences.append(correct_probs.mean().item())

    wandb.log({f"{run_name}/prediction_confidence": sum(confidences) / len(confidences)})


# ══════════════════════════════════════════════════════════════════════
#  EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment(args=None) -> None:
    """
    Full experiment suite: baseline + 4 ablations, all logged to W&B.
    """
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--exp",         type=str, default="all",
                            choices=["all", "baseline", "exp1", "exp2", "exp3", "exp4", "exp5"])
        parser.add_argument("--epochs",      type=int,   default=15)
        parser.add_argument("--batch_size",  type=int,   default=128)
        parser.add_argument("--d_model",     type=int,   default=256)
        parser.add_argument("--N",           type=int,   default=3)
        parser.add_argument("--num_heads",   type=int,   default=8)
        parser.add_argument("--d_ff",        type=int,   default=512)
        parser.add_argument("--dropout",     type=float, default=0.1)
        parser.add_argument("--warmup_steps",type=int,   default=4000)
        parser.add_argument("--num_workers", type=int,   default=0)
        parser.add_argument("--device",      type=str,   default="")
        args = parser.parse_args()

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs("checkpoints", exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────
    print("Loading datasets…")
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = build_dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    print(f"  src_vocab={len(src_vocab)}  tgt_vocab={len(tgt_vocab)}")

    base_cfg = dict(
        d_model=args.d_model, N=args.N, num_heads=args.num_heads,
        d_ff=args.d_ff, dropout=args.dropout,
        warmup_steps=args.warmup_steps, epochs=args.epochs,
    )

    wandb.init(project="da6401-a3", config=vars(args), reinit="finish_previous")

    # ═══════════════════════════════════════════════════════
    # Exp 2.1 — Noam Scheduler vs Fixed LR
    # ═══════════════════════════════════════════════════════
    if args.exp in ("all", "exp1"):
        print("\n=== Exp 2.1: Noam vs Fixed LR ===")
        _train_run(base_cfg, train_loader, val_loader, test_loader,
                   src_vocab, tgt_vocab, device,
                   run_name="noam", ckpt_path="checkpoints/noam.pt")

        fixed_cfg = {**base_cfg, "fixed_lr": 1e-4}
        _train_run(fixed_cfg, train_loader, val_loader, test_loader,
                   src_vocab, tgt_vocab, device,
                   run_name="fixed_lr", ckpt_path="checkpoints/fixed_lr.pt")

    # ═══════════════════════════════════════════════════════
    # Exp 2.2 — Ablation: scaling factor 1/√dk
    # ═══════════════════════════════════════════════════════
    if args.exp in ("all", "exp2"):
        print("\n=== Exp 2.2: Scaling factor ablation ===")
        # Monkey-patch scaled_dot_product_attention to skip scaling
        import model as model_module
        _orig_sdpa = model_module.scaled_dot_product_attention

        def _no_scale_sdpa(Q, K, V, mask=None):
            import math
            d_k = Q.size(-1)
            scores = torch.matmul(Q, K.transpose(-2, -1))   # no /sqrt(d_k)
            if mask is not None:
                scores = scores.masked_fill(mask, float("-inf"))
            import torch.nn.functional as F
            attn_w = F.softmax(scores, dim=-1)
            attn_w = torch.nan_to_num(attn_w, nan=0.0)
            return torch.matmul(attn_w, V), attn_w

        # With scaling (baseline, first 1000 steps only for grad norm logging)
        _train_run({**base_cfg, "epochs": min(args.epochs, 5)},
                   train_loader, val_loader, test_loader,
                   src_vocab, tgt_vocab, device,
                   run_name="with_scale", ckpt_path="checkpoints/with_scale.pt",
                   log_grad_norms=True)

        # Without scaling
        model_module.scaled_dot_product_attention = _no_scale_sdpa
        _train_run({**base_cfg, "epochs": min(args.epochs, 5)},
                   train_loader, val_loader, test_loader,
                   src_vocab, tgt_vocab, device,
                   run_name="no_scale", ckpt_path="checkpoints/no_scale.pt",
                   log_grad_norms=True)
        model_module.scaled_dot_product_attention = _orig_sdpa

    # ═══════════════════════════════════════════════════════
    # Exp 2.3 — Attention rollout & head specialization
    # ═══════════════════════════════════════════════════════
    if args.exp in ("all", "exp3"):
        print("\n=== Exp 2.3: Attention maps ===")
        model, _ = _train_run(base_cfg, train_loader, val_loader, test_loader,
                               src_vocab, tgt_vocab, device,
                               run_name="attn_vis", ckpt_path="checkpoints/attn_vis.pt")
        # grab one sentence and log attention maps for each head
        src_sample, _ = next(iter(val_loader))
        log_attention_maps(model, src_sample[:1], src_vocab, tgt_vocab, device)

    # ═══════════════════════════════════════════════════════
    # Exp 2.4 — Sinusoidal vs Learned Positional Encoding
    # ═══════════════════════════════════════════════════════
    if args.exp in ("all", "exp4"):
        print("\n=== Exp 2.4: Sinusoidal vs Learned PE ===")
        _train_run(base_cfg, train_loader, val_loader, test_loader,
                   src_vocab, tgt_vocab, device,
                   run_name="sin_pe", ckpt_path="checkpoints/sin_pe.pt")
        learned_cfg = {**base_cfg, "learned_pos_enc": True}
        _train_run(learned_cfg, train_loader, val_loader, test_loader,
                   src_vocab, tgt_vocab, device,
                   run_name="learned_pe", ckpt_path="checkpoints/learned_pe.pt")

    # ═══════════════════════════════════════════════════════
    # Exp 2.5 — Label smoothing eps=0.1 vs eps=0.0
    # ═══════════════════════════════════════════════════════
    if args.exp in ("all", "exp5"):
        print("\n=== Exp 2.5: Label Smoothing ablation ===")
        model_smooth, _ = _train_run(base_cfg, train_loader, val_loader, test_loader,
                                      src_vocab, tgt_vocab, device,
                                      run_name="smooth_0.1", ckpt_path="checkpoints/smooth.pt",
                                      use_label_smoothing=True)
        model_no_smooth, _ = _train_run(base_cfg, train_loader, val_loader, test_loader,
                                         src_vocab, tgt_vocab, device,
                                         run_name="smooth_0.0", ckpt_path="checkpoints/no_smooth.pt",
                                         use_label_smoothing=False)
        log_prediction_confidence(model_smooth,    val_loader, tgt_vocab, device, "smooth_0.1")
        log_prediction_confidence(model_no_smooth, val_loader, tgt_vocab, device, "smooth_0.0")

    # ═══════════════════════════════════════════════════════
    # Baseline (standalone best-model training)
    # ═══════════════════════════════════════════════════════
    if args.exp in ("baseline",):
        print("\n=== Baseline training ===")
        _train_run(base_cfg, train_loader, val_loader, test_loader,
                   src_vocab, tgt_vocab, device,
                   run_name="baseline", ckpt_path="checkpoints/best.pt")

    wandb.finish()
    print("\nAll experiments complete.")


if __name__ == "__main__":
    run_training_experiment()
