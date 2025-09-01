# /Evaluation/ablation.py
# Ablation (permutation / zero) on TFT external variables with robust feature-dimension alignment.
# Works across experiments with different #historical/#future feature counts.

import os, re, glob, csv
from typing import List, Tuple, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ----------------- utils -----------------
from collections import Counter

def _infer_hist_fut_feature_counts(model, all_inputs):
   
    enc = getattr(getattr(model, "hparams", None), "num_encoder_steps", None)
    tot = getattr(getattr(model, "hparams", None), "total_time_steps", None)
    dec = (tot - enc) if (enc is not None and tot is not None) else None

    def _walk(x):
        if isinstance(x, (list, tuple)):
            for e in x: 
                yield from _walk(e)
        elif isinstance(x, dict):
            for v in x.values():
                yield from _walk(v)
        elif torch.is_tensor(x) and x.dim() == 3:
            yield x

    hist_cands, fut_cands = [], []
    for t in _walk(all_inputs):
        T, N = int(t.size(1)), int(t.size(2))
        if enc is not None and T == enc:
            hist_cands.append(N)
        elif dec is not None and T == dec:
            fut_cands.append(N)

    def _most_common(xs):
        return Counter(xs).most_common(1)[0][0] if xs else None

    return _most_common(hist_cands), _most_common(fut_cands)

def assert_loader_matches_model(model, loader):

    device = next(model.parameters()).device
    for batch in loader:

        if isinstance(batch, (list, tuple)):
            pass
        batch = move_to(batch, device)
        pack = unpack_batch_for_io(batch)
        if hasattr(model, "_prepare_tft_inputs"):
            all_inputs, _ = model._prepare_tft_inputs(pack)
        else:
            (xk, xo, xs), _ = pack
            all_inputs = (xk, xo, xs)

        n_hist_in, n_fut_in = _infer_hist_fut_feature_counts(model, all_inputs)
        n_hist_need = int(model.hist_var_proj.shape[0])
        n_fut_need  = int(model.fut_var_proj.shape[0])

        msg = (f"[Ablation preflight] inferred from loader -> "
               f"hist={n_hist_in}, fut={n_fut_in} ; "
               f"model expects -> hist={n_hist_need}, fut={n_fut_need}")
        print(msg)

        bad = ((n_hist_in is not None and n_hist_in != n_hist_need) or
               (n_fut_in  is not None and n_fut_in  != n_fut_need))
        if bad:
            raise RuntimeError(
                "Loader/model feature-count mismatch.\n"
                f"{msg}\n"
                "This happens when you use a checkpoint trained with a DIFFERENT set/order of variables. "
                "You must build the dataloader with the SAME column_definition / data_formatter as that checkpoint.\n"
        
            )
        break

def latest_ckpt(root: str) -> str:
    files = glob.glob(os.path.join(root, "checkpoints", "*.ckpt")) + glob.glob(os.path.join(root, "*.ckpt"))
    if not files:
        raise FileNotFoundError(f"No .ckpt found under: {root}")
    return max(files, key=os.path.getmtime)

def move_to(x, device):
    if torch.is_tensor(x): return x.to(device)
    if isinstance(x, (list, tuple)): return type(x)(move_to(xx, device) for xx in x)
    if isinstance(x, dict): return {k: move_to(v, device) for k, v in x.items()}
    return x

def unpack_batch_for_io(batch):
    """
    Try to unpack batch into ((x_known, x_observed, x_static), y).
    If your model expects a different structure, adapt here.
    """
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        x, y = batch
        if isinstance(x, (list, tuple)) and len(x) >= 2:
            x_known    = x[0] if len(x) > 0 else None
            x_observed = x[1] if len(x) > 1 else None
            x_static   = x[2] if len(x) > 2 else None
            return (x_known, x_observed, x_static), y
    if isinstance(batch, (list, tuple)) and len(batch) == 2 and isinstance(batch[0], (list, tuple)):
        return batch
    raise RuntimeError("Cannot unpack batch. Please adapt unpack_batch_for_io() to your data format.")

# --- optional helpers (kept for compatibility; not required by main path) ---
def _align_last_dim(x: torch.Tensor, target_n: int):
    if x is None:
        return None
    if x.size(-1) == target_n:
        return x
    if x.size(-1) > target_n:
        return x[..., :target_n]  # trim
    pad = target_n - x.size(-1)
    return F.pad(x, (0, pad))     # pad zeros at feature dim

def _align_pack_to_model(model, pack):
    # pack = ((xk, xo, xs), y)
    (xk, xo, xs), y = pack
    Nh = int(model.hist_var_proj.shape[0])  # expected #hist regular vars
    Nf = int(model.fut_var_proj.shape[0])   # expected #fut  regular vars
    xk2 = _align_last_dim(xk, Nf) if torch.is_tensor(xk) else None
    xo2 = _align_last_dim(xo, Nh) if torch.is_tensor(xo) else None
    return ((xk2, xo2, xs), y)
# ---------------------------------------------------------------------------

def _pad_or_trim_last(x: torch.Tensor, target_n: int):
    """Pad (zeros) or trim the last dim to target_n."""
    if x is None or not torch.is_tensor(x):
        return x
    n = x.size(-1)
    if n == target_n:
        return x
    if n > target_n:
        return x[..., :target_n]
    # n < target_n: pad zeros on the last dim
    pad = target_n - n
    return F.pad(x, (0, pad))

def _align_all_inputs_to_model(model, all_inputs):
    """
    After model._prepare_tft_inputs(pack), align any 3D tensors' last dim
    to the model's expected #features:
      - tensors with time dim == enc_len --> align to Nh (hist_var_proj.shape[0])
      - tensors with time dim == dec_len --> align to Nf (fut_var_proj.shape[0])
    Fallback heuristics are applied if enc/dec are unavailable.
    """
    Nh = int(model.hist_var_proj.shape[0])  # expected historical feature count
    Nf = int(model.fut_var_proj.shape[0])   # expected future     feature count

    enc_len = getattr(getattr(model, "hparams", None), "num_encoder_steps", None)
    tot_len = getattr(getattr(model, "hparams", None), "total_time_steps", None)
    dec_len = (tot_len - enc_len) if (enc_len is not None and tot_len is not None) else None

    def _align_tensor(t: torch.Tensor):
        if not torch.is_tensor(t) or t.dim() != 3:
            return t
        T = t.size(1)
        # Prefer exact enc/dec matches
        if enc_len is not None and T == enc_len:
            return _pad_or_trim_last(t, Nh)
        if dec_len is not None and T == dec_len:
            return _pad_or_trim_last(t, Nf)
        # Fallback: pick the closer of Nh / Nf
        n = t.size(-1)
        target = Nh if abs(n - Nh) <= abs(n - Nf) else Nf
        return _pad_or_trim_last(t, target)

    def _walk(x):
        if isinstance(x, (list, tuple)):
            xs = [ _walk(e) for e in x ]
            return type(x)(xs)
        if isinstance(x, dict):
            return {k: _walk(v) for k, v in x.items()}
        return _align_tensor(x)

    return _walk(all_inputs)

def names_from_column_definition(model) -> Tuple[List[str], List[str]]:
    """
    Extract OBSERVED and KNOWN variable names by scanning model/data_formatter column definitions.
    Returns: (observed_names, known_names)
    """
    coldefs: Iterable = []
    # Try common places
    if hasattr(model, "column_definition") and getattr(model, "column_definition"):
        coldefs = model.column_definition
    elif hasattr(model, "data_formatter"):
        fmt = model.data_formatter
        if hasattr(fmt, "get_column_definition"):
            coldefs = fmt.get_column_definition()
        elif hasattr(fmt, "_column_definition"):
            coldefs = fmt._column_definition

    observed, known = [], []
    for it in coldefs or []:
        try:
            name, role = it[0], str(it[2])
        except Exception:
            continue
        if "OBSERVED_INPUT" in role:
            observed.append(name)
        elif "KNOWN_INPUT" in role:
            known.append(name)
    return observed, known

def eval_batch_loss(model, batch, bidx: int) -> float:
    """
    Compute loss without calling model.training_step to avoid einsum shape errors across experiments.
    Steps:
      1) unpack batch -> pack
      2) model._prepare_tft_inputs(pack) -> all_inputs, y
      3) align rank-3 tensors inside all_inputs to model's expected feature counts
      4) forward & compute the same loss as training_step
    """
    pack = unpack_batch_for_io(batch)

    # Prepare inputs (repo-specific logic lives here)
    if hasattr(model, "_prepare_tft_inputs"):
        all_inputs, y = model._prepare_tft_inputs(pack)
    else:
        # Fallback: assume forward accepts packed tensors directly
        (xk, xo, xs), y = pack
        all_inputs = (xk, xo, xs)

    # Align rank-3 tensors in all_inputs to match hist/fut proj sizes
    all_inputs = _align_all_inputs_to_model(model, all_inputs)

    # Forward (TFT forward expects a single 'all_inputs' object)
    y_hat = model(all_inputs)  # [B, T_dec, Q]

    # Align targets like training_step
    if hasattr(model, "_align_targets_to_decoder"):
        y_true = model._align_targets_to_decoder(y, y_hat)
    else:
        y_true = y
        if torch.is_tensor(y_true) and y_true.dim() == 2:
            y_true = y_true.unsqueeze(-1).expand(-1, y_hat.size(1), y_hat.size(2))

    # Use the same criterion if available
    if hasattr(model, "train_criterion"):
        loss = model.train_criterion.apply(y_hat, y_true)
    else:
        loss = torch.mean((y_hat - y_true) ** 2)

    return float(loss.detach().cpu().item())


def perturb_batch(pack, var_name: str, observed_names: List[str], known_names: List[str],
                  mode: str = "permute", seed: int = 42):
    """
    Return a perturbed deep copy of batch pack=((xk, xo, xs), y) by modifying one variable:
      - 'permute': shuffle across batch dimension for that variable slice.
      - 'zero': set that variable slice to zero.
    """
    (xk, xo, xs), y = pack
    rng = np.random.default_rng(seed)

    xk_p = xk.clone() if torch.is_tensor(xk) else None
    xo_p = xo.clone() if torch.is_tensor(xo) else None
    xs_p = xs.clone() if torch.is_tensor(xs) else None

    # OBSERVED: shape [B, Th, Vh]
    if torch.is_tensor(xo_p) and var_name in observed_names:
        j = observed_names.index(var_name)
        B = xo_p.shape[0]
        if mode == "permute":
            perm = torch.as_tensor(rng.permutation(B), device=xo_p.device)
            xo_p[:, :, j] = xo_p[perm, :, j]
        elif mode == "zero":
            xo_p[:, :, j] = 0.0

    # KNOWN: shape [B, Tk, Vk]
    if torch.is_tensor(xk_p) and var_name in known_names:
        j = known_names.index(var_name)
        B = xk_p.shape[0]
        if mode == "permute":
            perm = torch.as_tensor(rng.permutation(B), device=xk_p.device)
            xk_p[:, :, j] = xk_p[perm, :, j]
        elif mode == "zero":
            xk_p[:, :, j] = 0.0

    return ((xk_p, xo_p, xs_p), y)

def extract_version_suffix(version_dir: str) -> str:
    """
    Extract the suffix after 'version' (e.g., '.../version_20' -> '20').
    Fallback to the basename if no match.
    """
    base = os.path.basename(version_dir.rstrip("/"))
    m = re.search(r"version[_\-]?(.+)$", base)
    return m.group(1) if m else base

# ----------------- core ablation -----------------
def run_ablation(model,
                 loader,
                 observed_names: List[str],
                 known_names: List[str],
                 mode: str = "permute",
                 max_batches: int = 0,
                 var_list: Optional[List[str]] = None,
                 device: Optional[torch.device] = None,
                 seed: int = 42):
    """
    Run baseline and per-variable ablation. Returns (rows, baseline, n_batches).
    rows: list of tuples (var_name, perturbed_mean_loss, delta, impact_percent)
    """
    if device is None:
        device = torch.device("cpu")

    # Discover target variables
    ext_vars_all = (known_names or []) + (observed_names or [])
    if not ext_vars_all:
        raise RuntimeError("No external variables (KNOWN/OBSERVED) found from column_definition.")

    if var_list is not None and len(var_list) > 0:
        target_vars = [v for v in var_list if v in ext_vars_all]
        if not target_vars:
            raise RuntimeError("None of the variables in VAR_LIST matched KNOWN/OBSERVED names.")
    else:
        target_vars = ext_vars_all

    # Baseline over (partial) validation set
    baseline_sum, n_batches = 0.0, 0
    model.eval()
    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            if (max_batches > 0) and (b_idx >= max_batches): break
            batch = move_to(batch, device)
            # Try packing once to ensure compatibility; ignore if model expects raw batches
            try:
                _ = unpack_batch_for_io(batch)
            except RuntimeError:
                pass
            baseline_sum += eval_batch_loss(model, batch, b_idx)
            n_batches += 1
    if n_batches == 0:
        raise RuntimeError("Validation loader produced 0 batches. Check your dataloader.")
    baseline = baseline_sum / n_batches

    # Per-variable perturbation
    rows = []
    with torch.no_grad():
        for var in target_vars:
            pert_sum, cnt = 0.0, 0
            for b_idx, batch in enumerate(loader):
                if (max_batches > 0) and (b_idx >= max_batches): break
                batch = move_to(batch, device)
                try:
                    pack = unpack_batch_for_io(batch)
                    pert = perturb_batch(pack, var, observed_names, known_names, mode=mode, seed=seed)
                    batch_pert = pert   # ((xk, xo, xs), y)
                except Exception:
                    continue
                pert_sum += eval_batch_loss(model, batch_pert, b_idx)
                cnt += 1
            if cnt == 0:
                continue
            pert_mean = pert_sum / cnt
            delta = pert_mean - baseline
            pct = 0.0 if abs(baseline) < 1e-12 else 100.0 * delta / abs(baseline)
            rows.append((var, pert_mean, delta, pct))

    # Sort by impact percent (descending)
    rows.sort(key=lambda x: x[3], reverse=True)
    return rows, baseline, n_batches

def save_ablation_csv(rows, out_dir: str, csv_name: str):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, csv_name)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variable", "perturbed_mean_loss", "delta", "impact_percent"])
        for var, pert_mean, delta, pct in rows:
            w.writerow([var, f"{pert_mean:.6f}", f"{delta:.6f}", f"{pct:.4f}"])
    return csv_path


def _clean_names_for_ticks(names):
    """Replace '-' and '_' with spaces and convert to Title Case."""
    return [str(n).replace("-", " ").replace("_", " ").title() for n in names]

def plot_ablation(rows,
                  baseline: float,
                  out_dir: str,
                  version_dir: str,
                  n_batches: int,
                  mode: str):
    """
    Create two figures and save to out_dir. Title includes the parsed version suffix.
    Styling rules:
      - Color-blind-friendly colors (different from blue/orange used elsewhere)
      - Bars: Teal (#76B7B2); Baseline bar: Gray (#BAB0AC)
      - Impact line: Purple (#B07AA1)
      - Variable names cleaned for readability
    """
    os.makedirs(out_dir, exist_ok=True)
    ver = extract_version_suffix(version_dir)

    # Unpack rows
    vars_  = [r[0] for r in rows]
    losses = np.array([r[1] for r in rows], dtype=float)
    deltas = np.array([r[2] for r in rows], dtype=float)
    pcts   = np.array([r[3] for r in rows], dtype=float)

    # Clean names for tick labels
    tick_vars = _clean_names_for_ticks(vars_)

    # --- Figure 1: baseline vs perturbed mean loss ---
    fig_w = max(10, 0.35 * (len(vars_) + 1))
    plt.rcParams.update({
        "figure.figsize": (fig_w, 6),
        "savefig.dpi": 300,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 8,
        "ytick.labelsize": 10,
        "font.family": "DejaVu Sans",
    })

    # Prepare data and colors: first bar is baseline (gray), others are perturbed (teal)
    values_f1 = np.concatenate([[baseline], losses])
    colors_f1 = ["#BAB0AC"] + ["#76B7B2"] * len(losses)

    plt.figure()
    plt.bar(range(len(values_f1)), values_f1, color=colors_f1, alpha=0.95)
    plt.xticks(range(len(values_f1)), ["Baseline"] + tick_vars, rotation=75, ha="right")
    plt.xlabel("Variables (Masked/Permuted)")
    plt.ylabel("Mean Loss")
    plt.title(f"Baseline vs. Perturbed Losses | mode={mode} | version={ver} | batches={n_batches}")
    plt.tight_layout()
    f1_png = os.path.join(out_dir, "losses_baseline_vs_perturbed.png")
    f1_svg = os.path.join(out_dir, "losses_baseline_vs_perturbed.svg")
    plt.savefig(f1_png, bbox_inches="tight")
    plt.savefig(f1_svg, bbox_inches="tight")
    plt.close()

    # --- Figure 2: absolute degradation (bars) and impact percent (line) ---
    x = np.arange(len(vars_))
    fig, ax1 = plt.subplots(figsize=(fig_w, 6))

    # Bars: deltas in teal
    ax1.bar(x, deltas, color="#76B7B2", alpha=0.95)
    ax1.set_xlabel("Variables")
    ax1.set_ylabel("Δ Loss (Perturbed - Baseline)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(tick_vars, rotation=75, ha="right")

    # Line: pcts in purple with markers
    ax2 = ax1.twinx()
    ax2.plot(x, pcts, marker="o", linewidth=2.0, color="#B07AA1")
    ax2.set_ylabel("Impact (%)")

    fig.suptitle(f"Degradation & Impact (%) | mode={mode} | version={ver} | batches={n_batches}")
    fig.tight_layout()
    f2_png = os.path.join(out_dir, "impact_delta_and_percent.png")
    f2_svg = os.path.join(out_dir, "impact_delta_and_percent.svg")
    fig.savefig(f2_png, bbox_inches="tight")
    fig.savefig(f2_svg, bbox_inches="tight")
    plt.close()

    return f1_png, f1_svg, f2_png, f2_svg
