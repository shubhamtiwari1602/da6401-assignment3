"""
Kaggle training script for DA6401 Assignment 3.
Runs on T4/P100 GPU. Clones the public repo, installs deps, runs experiments.
"""
import os
import subprocess
import sys
import shutil

REPO_URL = "https://github.com/shubhamtiwari1602/da6401-assignment3.git"
REPO_DIR = "/kaggle/working/repo"
OUT_DIR  = "/kaggle/working"

import torch
print(f"PyTorch {torch.__version__}  CUDA: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

# ── Install deps ────────────────────────────────────────────────────
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "wandb", "datasets", "spacy", "evaluate", "sacrebleu"], check=True)
subprocess.run([sys.executable, "-m", "spacy", "download", "de_core_news_sm"], check=True)
subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"],  check=True)

# ── W&B login via secret ────────────────────────────────────────────
# Store your W&B API key as a Kaggle secret named WANDB_API_KEY
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
os.environ["WANDB_API_KEY"] = secrets.get_secret("WANDB_API_KEY")

# ── Clone repo ──────────────────────────────────────────────────────
if os.path.exists(REPO_DIR):
    shutil.rmtree(REPO_DIR)
subprocess.run(["git", "clone", "--depth=1", REPO_URL, REPO_DIR], check=True)
os.chdir(REPO_DIR)
os.makedirs("checkpoints", exist_ok=True)

# ── Run all experiments ─────────────────────────────────────────────
# Smaller model for Kaggle (fits comfortably in T4 memory, trains fast)
cmd = (
    f"{sys.executable} train.py "
    f"--exp all "
    f"--epochs 15 "
    f"--batch_size 256 "
    f"--d_model 256 "
    f"--N 3 "
    f"--num_heads 8 "
    f"--d_ff 512 "
    f"--warmup_steps 4000 "
    f"--num_workers 2"
)
print(f"\n{'='*60}\n{cmd}\n{'='*60}", flush=True)
subprocess.run(cmd, shell=True, check=True)

# ── Copy checkpoints to output ──────────────────────────────────────
for ckpt in os.listdir("checkpoints"):
    src = os.path.join("checkpoints", ckpt)
    dst = os.path.join(OUT_DIR, ckpt)
    shutil.copy(src, dst)
    print(f"[OUTPUT] {ckpt}: {os.path.getsize(dst)/1e6:.1f} MB", flush=True)

print("\nDone. Checkpoints in /kaggle/working/", flush=True)
