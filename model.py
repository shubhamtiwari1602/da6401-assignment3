"""
model.py — Transformer Architecture
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  └─────────────────────────────────────────────────────────────────┘
"""

import math
import copy
import json
import os
import re
import subprocess
import sys
import importlib
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
#  STANDALONE ATTENTION FUNCTION
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V

    Args:
        Q    : (..., seq_q, d_k)
        K    : (..., seq_k, d_k)
        V    : (..., seq_k, d_v)
        mask : broadcastable to (..., seq_q, seq_k); True → masked out

    Returns:
        output  : (..., seq_q, d_v)
        attn_w  : (..., seq_q, seq_k)
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    attn_w = F.softmax(scores, dim=-1)
    # replace nan rows (fully masked) with 0 so they don't propagate
    attn_w = torch.nan_to_num(attn_w, nan=0.0)
    output = torch.matmul(attn_w, V)
    return output, attn_w


# ══════════════════════════════════════════════════════════════════════
#  MASK HELPERS
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(src: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """
    Padding mask for the encoder.

    Returns:
        [batch, 1, 1, src_len]  True = PAD (masked out)
    """
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """
    Combined padding + causal mask for the decoder.

    Returns:
        [batch, 1, tgt_len, tgt_len]  True = masked out
    """
    batch, tgt_len = tgt.size()
    pad_mask  = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)                     # [B,1,1,T]
    look_ahead = torch.triu(
        torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=tgt.device), diagonal=1
    )                                                                           # [T,T]
    return pad_mask | look_ahead                                               # [B,1,T,T]


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention — §3.2.2 of "Attention Is All You Need".
    Does NOT use torch.nn.MultiheadAttention.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

        # last attention weights — used for visualization
        self.attn_weights = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, S, d_model] → [B, h, S, d_k]"""
        B, S, _ = x.size()
        return x.view(B, S, self.num_heads, self.d_k).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = query.size(0)

        Q = self._split_heads(self.W_q(query))   # [B, h, S_q, d_k]
        K = self._split_heads(self.W_k(key))
        V = self._split_heads(self.W_v(value))

        attn_out, attn_w = scaled_dot_product_attention(Q, K, V, mask)
        self.attn_weights = attn_w.detach()

        # concat heads → [B, S_q, d_model]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.W_o(attn_out)


# ══════════════════════════════════════════════════════════════════════
#  POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding — §3.5.
    Registered as a buffer (non-trainable).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)                       # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                                      # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : [batch, seq_len, d_model] → same shape"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#  FEED-FORWARD NETWORK
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂  — §3.3"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """
    x → [Self-Attention → Add & Norm] → [FFN → Add & Norm]
    Uses Post-LayerNorm (as in the original paper).
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn       = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        # Self-attention sub-layer
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        # FFN sub-layer
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


# ══════════════════════════════════════════════════════════════════════
#  DECODER LAYER
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    x → [Masked Self-Attn → Add & Norm]
      → [Cross-Attn(memory) → Add & Norm]
      → [FFN → Add & Norm]
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(p=dropout)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Masked self-attention
        sa_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(sa_out))
        # Cross-attention over encoder memory
        ca_out = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout(ca_out))
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """Stack of N identical EncoderLayer modules with final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N identical DecoderLayer modules with final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
#  FULL TRANSFORMER
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence tasks.
    Supports optional learned positional embeddings for experiment 2.4.
    """

    def __init__(
        self,
        src_vocab_size: Optional[int] = None,
        tgt_vocab_size: Optional[int] = None,
        d_model:   int   = 256,
        N:         int   = 3,
        num_heads: int   = 8,
        d_ff:      int   = 512,
        dropout:   float = 0.1,
        learned_pos_enc: bool = False,
        max_len:   int   = 5000,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.max_len = max_len
        self.learned_pos_enc = learned_pos_enc
        self._inference_ready = src_vocab_size is None or tgt_vocab_size is None

        if self._inference_ready:
            src_vocab_size, tgt_vocab_size = self._load_inference_assets()

        enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)

        self.encoder = Encoder(enc_layer, N)
        self.decoder = Decoder(dec_layer, N)

        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)

        if learned_pos_enc:
            self.src_pos = nn.Embedding(max_len, d_model)
            self.tgt_pos = nn.Embedding(max_len, d_model)
            self.pos_dropout = nn.Dropout(p=dropout)
        else:
            self.pos_enc = PositionalEncoding(d_model, dropout, max_len)

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        self._init_weights()

        if self._inference_ready:
            self._load_inference_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _src_encoding(self, src: torch.Tensor) -> torch.Tensor:
        x = self.src_embed(src) * math.sqrt(self.d_model)
        if self.learned_pos_enc:
            pos = torch.arange(src.size(1), device=src.device).unsqueeze(0)
            x = self.pos_dropout(x + self.src_pos(pos))
        else:
            x = self.pos_enc(x)
        return x

    def _tgt_encoding(self, tgt: torch.Tensor) -> torch.Tensor:
        x = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        if self.learned_pos_enc:
            pos = torch.arange(tgt.size(1), device=tgt.device).unsqueeze(0)
            x = self.pos_dropout(x + self.tgt_pos(pos))
        else:
            x = self.pos_enc(x)
        return x

    # ── AUTOGRADER HOOKS ──────────────────────────────────────────────

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(self._src_encoding(src), src_mask)

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        dec_out = self.decoder(self._tgt_encoding(tgt), memory, src_mask, tgt_mask)
        return self.fc_out(dec_out)

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    # ── SUBMISSION INFERENCE HELPERS ─────────────────────────────────

    def _ensure_package(self, import_name: str, package_name: Optional[str] = None) -> None:
        try:
            importlib.import_module(import_name)
            return
        except ImportError:
            pass

        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", package_name or import_name],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        importlib.import_module(import_name)

    def _load_inference_assets(self) -> Tuple[int, int]:
        self._ensure_package("datasets")
        self._ensure_package("spacy")

        from datasets import load_dataset
        from dataset import Vocabulary
        import spacy

        self.src_vocab = Vocabulary()
        self.tgt_vocab = Vocabulary()

        try:
            self.nlp_de = spacy.load("de_core_news_sm")
        except OSError:
            try:
                subprocess.run(
                    [sys.executable, "-m", "spacy", "download", "de_core_news_sm"],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self.nlp_de = spacy.load("de_core_news_sm")
            except Exception:
                self.nlp_de = spacy.blank("de")

        try:
            self.nlp_en = spacy.load("en_core_web_sm")
        except OSError:
            try:
                subprocess.run(
                    [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self.nlp_en = spacy.load("en_core_web_sm")
            except Exception:
                self.nlp_en = spacy.blank("en")

        train_split = load_dataset("bentrevett/multi30k", split="train")
        src_tokens = [[tok.text.lower() for tok in self.nlp_de.tokenizer(ex["de"])] for ex in train_split]
        tgt_tokens = [[tok.text.lower() for tok in self.nlp_en.tokenizer(ex["en"])] for ex in train_split]

        self.src_vocab.build(src_tokens)
        self.tgt_vocab.build(tgt_tokens)
        if len(self.src_vocab) != 7853 or len(self.tgt_vocab) != 5893:
            raise RuntimeError(
                f"Inference vocab mismatch: got src={len(self.src_vocab)}, tgt={len(self.tgt_vocab)}; "
                "expected src=7853, tgt=5893 from the trained checkpoint."
            )
        return len(self.src_vocab), len(self.tgt_vocab)

    def _load_submission_config(self) -> dict:
        cfg_path = Path(__file__).with_name("submission_config.json")
        if not cfg_path.exists():
            return {}
        with cfg_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _resolve_checkpoint_path(self) -> Path:
        cfg = self._load_submission_config()

        local_candidates = [
            cfg.get("local_checkpoint"),
            os.environ.get("DA6401_A3_LOCAL_CHECKPOINT"),
            "kaggle_deploy/output/sin_pe.pt",
            "checkpoints/sin_pe.pt",
            "sin_pe.pt",
        ]
        for candidate in local_candidates:
            if not candidate:
                continue
            path = Path(candidate)
            if not path.is_absolute():
                path = Path(__file__).resolve().parent / path
            if path.exists():
                return path

        file_id = cfg.get("gdrive_file_id") or os.environ.get("DA6401_A3_GDRIVE_FILE_ID")
        file_url = cfg.get("gdrive_url") or os.environ.get("DA6401_A3_GDRIVE_URL")
        if not file_id and not file_url:
            raise FileNotFoundError(
                "No checkpoint found. Add a local checkpoint path or set a Google Drive file id/url "
                "in submission_config.json or the DA6401_A3_GDRIVE_FILE_ID/DA6401_A3_GDRIVE_URL env vars."
            )

        download_dir = Path(__file__).resolve().parent / ".cache"
        download_dir.mkdir(exist_ok=True)
        destination = download_dir / "submission_checkpoint.pt"
        if destination.exists():
            return destination

        self._download_checkpoint(destination, file_id=file_id, file_url=file_url)
        return destination

    def _download_checkpoint(
        self,
        destination: Path,
        file_id: Optional[str] = None,
        file_url: Optional[str] = None,
    ) -> None:
        self._ensure_package("gdown")
        import gdown

        if file_id:
            gdown.download(id=file_id, output=str(destination), quiet=True)
        else:
            gdown.download(url=file_url, output=str(destination), quiet=True)

        if not destination.exists():
            raise FileNotFoundError(f"Checkpoint download failed: {destination}")

    def _load_inference_weights(self) -> None:
        ckpt_path = self._resolve_checkpoint_path()
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.load_state_dict(state_dict, strict=True)

    def _detokenize_english(self, tokens) -> str:
        text = " ".join(tokens)
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        text = re.sub(r"\(\s+", "(", text)
        text = re.sub(r"\s+\)", ")", text)
        text = re.sub(r"\s+'", "'", text)
        return text.strip()

    def infer(self, german_sentence: str) -> str:
        if not self._inference_ready:
            raise RuntimeError("infer() is only available on a submission/inference-initialized Transformer().")

        from dataset import Vocabulary

        self.eval()
        device = next(self.parameters()).device

        src_tokens = [tok.text.lower() for tok in self.nlp_de.tokenizer(german_sentence)]
        src_ids = [Vocabulary.SOS_IDX] + self.src_vocab.lookup_indices(src_tokens) + [Vocabulary.EOS_IDX]
        src = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_mask = make_src_mask(src, Vocabulary.PAD_IDX).to(device)

        with torch.no_grad():
            memory = self.encode(src, src_mask)
            ys = torch.tensor([[Vocabulary.SOS_IDX]], dtype=torch.long, device=device)

            for _ in range(self.max_len - 1):
                tgt_mask = make_tgt_mask(ys, Vocabulary.PAD_IDX).to(device)
                logits = self.decode(memory, src_mask, ys, tgt_mask)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                ys = torch.cat([ys, next_token], dim=1)
                if next_token.item() == Vocabulary.EOS_IDX:
                    break

        pred_ids = ys[0].tolist()
        pred_tokens = [
            self.tgt_vocab.lookup_token(idx)
            for idx in pred_ids
            if idx not in (Vocabulary.SOS_IDX, Vocabulary.EOS_IDX, Vocabulary.PAD_IDX)
        ]
        return self._detokenize_english(pred_tokens)
