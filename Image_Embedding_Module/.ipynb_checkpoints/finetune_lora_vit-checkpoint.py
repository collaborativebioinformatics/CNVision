#!/usr/bin/env python3
# finetune_lora_vit.py
import os, math, random, argparse, numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold, StratifiedKFold, GroupShuffleSplit
from sklearn.metrics import average_precision_score
from transformers import AutoImageProcessor, AutoModel
from peft import LoraConfig, get_peft_model
from huggingface_hub import login

token = os.environ.get("HUGGINGFACE_HUB_TOKEN") 
if token:
    login(token=token)

# -------------------- Args --------------------
ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="meta.csv")
ap.add_argument("--model_id", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--last_k", type=int, default=3, help="LoRA on last K transformer blocks")
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--group_by", default="sample_id",
                help="Grouping column for CV: sample_id or event_id")
ap.add_argument("--include_mlp", action="store_true", help="also adapt MLP up/down proj")
ap.add_argument("--lr_head", type=float, default=1e-3)
ap.add_argument("--lr_lora", type=float, default=5e-4)
ap.add_argument("--weight_decay", type=float, default=0.01)
ap.add_argument("--grad_accum", type=int, default=1)
ap.add_argument("--seed", type=int, default=1337)
ap.add_argument("--log_csv", default=None, help="write per-epoch metrics here")
ap.add_argument("--out", default="vitb16_lora_test.ckpt")
args = ap.parse_args()

# -------------------- Seeds/Backend --------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

# -------------------- Helpers --------------------
def _first_tensor_like(x):
    import numpy as np, torch
    if torch.is_tensor(x): return x
    if isinstance(x, np.ndarray): return torch.from_numpy(x)
    return None

def load_pt_any(path):
    """Return (1,3,H,W) float in [0,1], or raise with path context."""
    try:
        obj = torch.load(path, map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"{path} :: torch.load failed :: {e}")

    t = _first_tensor_like(obj)
    if t is None and isinstance(obj, dict):
        # common keys
        for k in ("pixel_values","image","img","x","array","arr","tensor","data","inputs"):
            if k in obj:
                t = _first_tensor_like(obj[k]); 
                if t is not None: break
        if t is None:
            for v in obj.values():
                t = _first_tensor_like(v)
                if t is not None: break
    if t is None and isinstance(obj, (list, tuple)) and obj:
        t = _first_tensor_like(obj[0])

    if t is None:
        raise RuntimeError(f"{path} :: no tensor-like data inside")

    x = t
    if x.ndim == 4: x = x[0]
    if x.ndim == 2:
        x = x.unsqueeze(0).repeat(3,1,1)
    elif x.ndim == 3:
        if x.shape[0] not in (1,3,4) and x.shape[-1] in (1,3,4):
            x = x.permute(2,0,1)
        if x.shape[0] == 1: x = x.repeat(3,1,1)
        if x.shape[0] == 4: x = x[:3]
    else:
        raise RuntimeError(f"{path} :: unexpected ndim={x.ndim}")

    x = x.float()
    if x.max() > 1.5: x = x/255.0
    return x.unsqueeze(0)  # (1,3,H,W)

# --- dataset that returns None on bad file ---
class PTDataset(Dataset):
    def __init__(self, df, processor, train=True):
        self.df, self.proc, self.train = df.reset_index(drop=True), processor, train
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        try:
            chw = load_pt_any(r.path).squeeze(0)  # (3,H,W)
        except Exception as e:
            # Log once per failure
            print(f"[BAD] {e}")
            return None  # tell collate to drop it
        # tiny geometric jitter for train
        if self.train:
            C,H,W = chw.shape; dh, dw = int(0.05*H), int(0.05*W)
            if dh>0 and dw>0:
                import random
                top = random.randint(0, dh); left = random.randint(0, dw)
                chw = chw[:, top:H-(dh-top), left:W-(dw-left)]
        hwc = chw.permute(1,2,0).numpy()
        pv  = self.proc(images=hwc, return_tensors="pt")["pixel_values"].squeeze(0)
        return (pv, int(r.label))

def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return {"pixel_values": torch.empty(0,3,224,224)}, torch.empty(0)
    px = torch.stack([b[0] for b in batch], dim=0)
    y  = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    return {"pixel_values": px}, y

def get_feats(backbone, px):
    """
    Return per-image features for the classifier head.
    Prefer CLS token; fall back to pooler_output only if it's non-degenerate.
    """
    out = backbone(pixel_values=px, return_dict=True)
    feats = getattr(out, "pooler_output", None)
    if feats is None:
        return out.last_hidden_state[:, 0, :]  # CLS

    # If pooler exists but is effectively constant, use CLS
    # (avoid tiny-std bias-only behavior)
    try:
        stdval = feats.std().detach().cpu().item()
    except Exception:
        stdval = 0.0
    if stdval < 1e-6:
        return out.last_hidden_state[:, 0, :]  # CLS

    return feats


# -------------------- Model & LoRA target modules --------------------
proc  = AutoImageProcessor.from_pretrained(args.model_id)
base  = AutoModel.from_pretrained(args.model_id).to(device)
feat_dim = base.config.hidden_size if hasattr(base.config, "hidden_size") else 768

# discover #layers
try:
    n_layers = base.config.num_hidden_layers
except Exception:
    idx = []
    for n,_ in base.named_modules():
        m = re.search(r"^layer\.(\d+)\.", n)
        if m: idx.append(int(m.group(1)))
    n_layers = max(idx) + 1

last = list(range(max(0, n_layers-args.last_k), n_layers))
target_modules = sum([
    [f"layer.{i}.attention.q_proj",
     f"layer.{i}.attention.k_proj",
     f"layer.{i}.attention.v_proj",
     f"layer.{i}.attention.o_proj"]
    for i in last
], [])
if args.include_mlp:
    target_modules += sum([
        [f"layer.{i}.mlp.up_proj", f"layer.{i}.mlp.down_proj"]
        for i in last
    ], [])
assert target_modules, "target_modules is empty; check layer names."

lcfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=target_modules,
    bias="none",
    task_type="FEATURE_EXTRACTION"
)

# freeze base, wrap with PEFT
for p in base.parameters(): p.requires_grad = False
base = get_peft_model(base, lcfg)
base.print_trainable_parameters()

# small classifier head
head = nn.Sequential(
    nn.LayerNorm(feat_dim),
    nn.Linear(feat_dim, 256), nn.GELU(),
    nn.Linear(256, 1)
).to(device)

# -------------------- Train/Eval helpers --------------------
def run_fold(df, tr, te, fi):
    # reset head
    for m in head.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)
    # reset LoRA params
    if hasattr(base, "reset_lora_parameters"): base.reset_lora_parameters()

    ds_tr = PTDataset(df.iloc[tr], proc, train=True)
    ds_te = PTDataset(df.iloc[te], proc, train=False)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                       pin_memory=True, collate_fn=collate, persistent_workers=(args.workers>0), prefetch_factor=2)
    dl_te = DataLoader(ds_te, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                       pin_memory=True, collate_fn=collate, persistent_workers=(args.workers>0), prefetch_factor=2)

    # loss with class imbalance
    y_tr = df.iloc[tr]["label"].values.astype(int)
    n_pos, n_neg = (y_tr==1).sum(), (y_tr==0).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos,1)], device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # two param groups
    params = [
        {"params": [p for p in base.parameters() if p.requires_grad], "lr": args.lr_lora},
        {"params": head.parameters(), "lr": args.lr_head},
    ]
    opt = torch.optim.AdamW(params, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda',enabled=True)

    best_ap, best = -1.0, None
    total_steps = args.epochs * math.ceil(len(ds_tr)/args.batch)
    global_step = 0

    for epoch in range(args.epochs):
        base.train(); head.train()
        tr_loss_sum, tr_n = 0.0, 0

        for batch in dl_tr:
            px = batch[0]["pixel_values"].to(device, non_blocking=True)
            y  = batch[1].to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=True):
                feats  = get_feats(base, px)
                logits = head(feats).squeeze(1)
                raw_loss = bce(logits, y)               # ← loss for logging
                loss = raw_loss / args.grad_accum       # ← scaled for backward only

            scaler.scale(loss).backward()
            if (global_step + 1) % args.grad_accum == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            global_step += 1

            # accumulate train loss weighted by batch size
            bs = y.numel()
            tr_loss_sum += raw_loss.item() * bs
            tr_n += bs

        # eval AUPRC and val loss
        base.eval(); head.eval()
        preds, gold = [], []
        va_loss_sum, va_n = 0.0, 0

        with torch.no_grad(), torch.amp.autocast('cuda', enabled=True):
            for batch in dl_te:
                px = batch[0]["pixel_values"].to(device, non_blocking=True)
                y  = batch[1].to(device, non_blocking=True)
                feats = get_feats(base, px)
                logits = head(feats).squeeze(1)
                p  = torch.sigmoid(logits)

                # accumulate val loss on the SAME criterion (pos_weight)
                raw_val = bce(logits, y)
                bs = y.numel()
                va_loss_sum += raw_val.item() * bs
                va_n += bs

                preds.append(p.float().cpu()); gold.append(y.float().cpu())

        train_loss = tr_loss_sum / max(tr_n, 1)
        val_loss   = va_loss_sum / max(va_n, 1)
        p = torch.cat(preds).numpy(); y = torch.cat(gold).numpy()
        ap = average_precision_score(y, p)

        print(f"epoch {epoch+1}/{args.epochs} | "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"AUPRC={ap:.4f} | pos_weight={pos_weight.item():.2f}")
        if args.log_csv:
            import csv, os
            header = ["fold", "epoch", "train_loss", "val_loss", "val_auprc", "pos_weight"]
            row = [fi, epoch+1, train_loss, val_loss, ap, float(pos_weight.item())]  # 'fi' is the fold index from outer loop
            write_header = not os.path.exists(args.log_csv)
            with open(args.log_csv, "a", newline="") as f:
                w = csv.writer(f)
                if write_header: w.writerow(header)
                w.writerow(row)


        if ap > best_ap:
            best_ap, best = ap, {
                "peft_state_dict": base.state_dict(),
                "head_state_dict": head.state_dict()
            }

    return best_ap, best

# -------------------- CV & save --------------------
def build_folds(df, folds=5, group_by="sample_id", seed=1337):
    y = df["label"].values.astype(int)

    def _grouped(col):
        groups = df[col].values
        uniq = df[col].nunique()
        if uniq >= 2:
            n = min(folds, uniq)
            print(f"Using GroupKFold(n_splits={n}) on '{col}' with {uniq} groups")
            gkf = GroupKFold(n_splits=n)
            return list(gkf.split(df, y, groups))
        return None

    # 1) try requested group
    fs = _grouped(group_by)
    if fs: return fs

    # 2) try event_id as fallback
    if "event_id" in df.columns:
        fs = _grouped("event_id")
        if fs: return fs

    # 3) no grouping possible -> stratified k-fold (tile-level)
    min_class = df["label"].value_counts().min()
    if min_class >= 2:
        n = min(folds, min_class)
        n = max(n, 2)
        print(f"Using StratifiedKFold(n_splits={n}) (no valid groups)")
        skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=seed)
        return list(skf.split(df, y))

    # 4) absolute fallback: repeated GroupShuffleSplit on whatever we have
    print("Very few positives/negatives; doing 5x GroupShuffleSplit with test_size=0.2")
    gss = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=seed)
    groups = df[group_by].values if group_by in df else np.arange(len(df))
    return list(gss.split(df, y, groups))


df = pd.read_csv(args.csv)
assert {"path","label","sample_id"}.issubset(df.columns), "meta.csv must have path,label,sample_id"
folds = build_folds(df, folds=args.folds, group_by=args.group_by, seed=args.seed)

scores = []
best_global = (-1, None)
for fi,(tr,te) in enumerate(folds, 1):
    print(f"\n=== Fold {fi}/{len(folds)} (grouped by {args.group_by}) ===")
    ap, state = run_fold(df, tr, te, fi)
    scores.append(ap)
    if ap > best_global[0]: best_global = (ap, state)

print(f"\nCV AUPRC mean±sd: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

# Save best fold (adapters + head + config)
ckpt = {
    "model_id": args.model_id,
    "feat_dim": feat_dim,
    "last_k": args.last_k,
    "include_mlp": bool(args.include_mlp),
    "target_modules": target_modules,
    "lora_config": {
        "r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
        "target_modules": target_modules, "bias": "none", "task_type": "FEATURE_EXTRACTION"
    },
    "peft_state_dict": best_global[1]["peft_state_dict"],
    "head_state_dict": best_global[1]["head_state_dict"],
    "cv_scores": scores
}
torch.save(ckpt, args.out)
print(f"[OK] saved {args.out}")
