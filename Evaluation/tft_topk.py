# tft_topk.py
# Collect and rank Temporal Fusion Transformer VSN (variable selection) weights over a dataloader.

import os, re, glob
from typing import List, Tuple, Iterable, Optional
from collections import Counter

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path



# ---------- Basics ----------
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
    Try to unpack a batch into ((x_known, x_observed, x_static), y).
    Adapt here if your repo uses a different batch structure.
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

def extract_version_suffix(version_dir: str) -> str:
    """
    Extract the suffix after 'version' (e.g., '.../version_20' -> '20').
    Fallback to the basename if no match.
    """
    base = os.path.basename(version_dir.rstrip("/"))
    m = re.search(r"version[_\-]?(.+)$", base)
    return m.group(1) if m else base


# ---------- Introspection & safety checks ----------
def _infer_hist_fut_feature_counts(model, all_inputs):
    """
    From all_inputs (after model._prepare_tft_inputs), infer (#hist_features, #future_features)
    by looking at rank-3 tensors whose time dimension equals enc_len or dec_len.
    """
    enc = getattr(getattr(model, "hparams", None), "num_encoder_steps", None)
    tot = getattr(getattr(model, "hparams", None), "total_time_steps", None)
    dec = (tot - enc) if (enc is not None and tot is not None) else None

    def _walk(x):
        if isinstance(x, (list, tuple)):
            for e in x: yield from _walk(e)
        elif isinstance(x, dict):
            for v in x.values(): yield from _walk(v)
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
    """
    Take one batch, run _prepare_tft_inputs, infer (#hist, #fut) from data, and compare to model params.
    Raise a clear error if mismatched (VSN depends on feature counts).
    """
    device = next(model.parameters()).device
    for batch in loader:
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

        msg = (f"[TopK preflight] inferred from loader -> hist={n_hist_in}, fut={n_fut_in} ; "
               f"model expects -> hist={n_hist_need}, fut={n_fut_need}")
        print(msg)

        bad = ((n_hist_in is not None and n_hist_in != n_hist_need) or
               (n_fut_in  is not None and n_fut_in  != n_fut_need))
        if bad:
            raise RuntimeError(
                "Loader/model feature-count mismatch.\n"
                f"{msg}\n"
                "This checkpoint was trained with a DIFFERENT set/order of variables. "
                "You must use the dataloader built with the SAME column_definition/data_formatter as the checkpoint.\n"
                "Fix by: (1) Using model.val_dataloader() / train_dataloader() from the same experiment config, "
                "or (2) Switching to a matching checkpoint."
            )
        break


# ---------- Capture & aggregation ----------
def extract_vsn_weights(out):
    """
    Try to get VSN selection weights from a module's forward output.
    Common patterns: (selected_values, selection_weights) or any rank>=2 tensor as weights.
    """
    if isinstance(out, (list, tuple)) and len(out) >= 2 and torch.is_tensor(out[1]):
        return out[1]
    if torch.is_tensor(out) and out.dim() >= 2:
        return out
    if isinstance(out, (list, tuple)):
        for e in out:
            if torch.is_tensor(e) and e.dim() >= 2:
                return e
    raise RuntimeError("Cannot infer VSN weights from module output; adapt extract_vsn_weights().")

class WeightedAverager:
    """Accumulate sum over all dims except last (variable dim), plus element count, to compute a global mean."""
    def __init__(self):
        self.sum = None
        self.count = 0
    def add(self, w: torch.Tensor):
        dims = tuple(range(w.dim()-1))     # reduce over all but last
        s = w.sum(dim=dims)                # [V]
        c = int(w.numel() // w.shape[-1])  # samples aggregated per variable
        s = s.detach().to("cpu")
        if self.sum is None:
            self.sum = s
        else:
            self.sum += s
        self.count += c
    def mean(self):
        if self.sum is None or self.count == 0:
            return None
        return (self.sum / float(self.count))  # [V] (CPU tensor)

def register_vsn_hooks(model):
    """
    Register forward hooks on the three VSN modules if present.
    Returns: (hooks[], captures dict)
    """
    captures = {}
    def make_hook(key):
        def _hook(mod, inp, out):
            try:
                captures[key] = extract_vsn_weights(out)  # [*, V]
            except Exception:
                captures[key] = None
        return _hook

    names = ["static_vsn", "temporal_historical_vsn", "temporal_future_vsn"]
    hooks = []
    for name in names:
        if hasattr(model, name):
            hooks.append(getattr(model, name).register_forward_hook(make_hook(name)))
    return hooks, captures

def collect_vsn_means(model, loader, max_batches: int = 0):
    """
    Iterate dataloader, run forward to trigger VSN hooks, and aggregate mean weights.
    Returns: (w_static_mean, w_hist_mean, w_future_mean) as 1D CPU tensors or None.
    """
    hooks, captures = register_vsn_hooks(model)
    avg_static, avg_hist, avg_future = WeightedAverager(), WeightedAverager(), WeightedAverager()

    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            if max_batches and b_idx >= max_batches:
                break
            batch = move_to(batch, next(model.parameters()).device)

            # Preferred path: use _prepare_tft_inputs then forward(all_inputs)
            ok = False
            try:
                pack = unpack_batch_for_io(batch)
                all_inputs, _ = model._prepare_tft_inputs(pack) if hasattr(model, "_prepare_tft_inputs") else (batch[0], None)
                _ = model(all_inputs)
                ok = True
            except Exception:
                # fallback: try training_step to trigger hooks
                try:
                    _ = model.training_step(batch, b_idx)
                    ok = True
                except Exception:
                    pass

            if not ok:
                # last resort: try model(x) if batch is (x, y)
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    try:
                        _ = model(batch[0])
                        ok = True
                    except Exception as e:
                        raise RuntimeError(f"Forward failed; adapt calling pattern. Error: {e}")
                else:
                    raise RuntimeError("Forward failed; adapt calling pattern to your repo.")

            # aggregate
            if isinstance(captures.get("static_vsn"), torch.Tensor):
                avg_static.add(captures["static_vsn"])
            if isinstance(captures.get("temporal_historical_vsn"), torch.Tensor):
                avg_hist.add(captures["temporal_historical_vsn"])
            if isinstance(captures.get("temporal_future_vsn"), torch.Tensor):
                avg_future.add(captures["temporal_future_vsn"])

    for h in hooks:
        h.remove()

    return avg_static.mean(), avg_hist.mean(), avg_future.mean()


# ---------- Names & printing ----------
def names_from_column_definition(model, n_static: int, n_hist: int, n_future: int):
    """
    Best-effort mapping from model.column_definition to variable names.
    Returns (static_names, hist_names, future_names). If counts mismatch, pad with generic names.
    """
    static_names, known_names, observed_names = [], [], []
    coldef: Iterable = getattr(model, "column_definition", None)

    if coldef is not None:
        for item in coldef:
            try:
                name, itype = item[0], item[2]
                is_ = str(itype)
            except Exception:
                continue
            if "STATIC_INPUT" in is_:
                static_names.append(name)
            elif "KNOWN_INPUT" in is_:
                known_names.append(name)
            elif "OBSERVED_INPUT" in is_:
                observed_names.append(name)

    def fit(names, n, prefix):
        names = names or []
        if n is None or n <= 0: return []
        if len(names) >= n:     return names[:n]
        return names + [f"{prefix}_{i}" for i in range(len(names), n)]

    s_names = fit(static_names, n_static, "static")
    h_all   = (observed_names or []) + (known_names or [])   # encoder uses observed + known
    h_names = fit(h_all, n_hist, "hist")
    f_names = fit(known_names, n_future, "future")           # decoder uses known only
    return s_names, h_names, f_names

def topk_print(weights_1d: torch.Tensor, names: List[str], title: str, k: int):
    """Print Top-K entries from a 1D tensor of weights (CPU)."""
    if weights_1d is None: return
    w = weights_1d.detach().cpu().numpy()
    n = len(w); k = max(1, min(k, n))
    idx = np.argsort(w)[-k:][::-1]
    print(f"\n=== {title} — Top-{k} ===")
    for i in idx:
        nm = names[i] if i < len(names) else f"var_{i}"
        print(f"{w[i]:10.6f}  {nm}")


# ---------- Plotting ----------
def _to_np(x):
    if x is None: return None
    if hasattr(x, "detach"): x = x.detach().cpu().numpy()
    return np.asarray(x)

# Fixed, color-blind-friendly palettes (academic style)
PALETTES = {
    "historical": {
        "solid": "#4E79A7",              # dark blue (default)
        "alt":   ["#4E79A7", "#A0CBE8"], # dark blue + light blue (optional stripes)
    },
    "future": {
        "solid": "#F28E2B",              # dark orange (default)
        "alt":   ["#F28E2B", "#FFBE7D"], # dark orange + light orange (optional stripes)
    },
    "default": {
        "solid": "#4E79A7",
        "alt":   ["#4E79A7", "#A0CBE8"],
    }
}

def _infer_section(section: str | None, title: str, filename: str, out_dir: str) -> str:
    """Infer section (historical/future/default) based on explicit argument or text in title/path"""
    candidates = " ".join([str(section or ""), title, filename, out_dir]).lower()
    if "histor" in candidates:
        return "historical"
    if "future" in candidates or "ahead" in candidates:
        return "future"
    return "default"

def _clean_names(names):
    """Replace '-' and '_' with spaces, convert to Title Case"""
    return [str(n).replace("-", " ").replace("_", " ").title() for n in names]


# Color-blind friendly academic palettes
_PALETTES = {
    "historical": {"solid": "#4E79A7", "alt": ["#4E79A7", "#A0CBE8"]}, # blue family
    "future":     {"solid": "#F28E2B", "alt": ["#F28E2B", "#FFBE7D"]}, # orange family
    "default":    {"solid": "#4E79A7", "alt": ["#4E79A7", "#A0CBE8"]},
}

def _infer_section_auto(title: str, filename: str, out_dir: str) -> str:
    """
    Automatically infer section type (historical/future/default) 
    based on keywords in title, filename, or output directory.
    - future: contains "future", "ahead", or "decoder"
    - historical: contains "histor", "past", or "encoder"
    - otherwise: default
    """
    s = " ".join([title, filename, out_dir]).lower()
    if any(k in s for k in ["future", "ahead", "decoder"]):
        return "future"
    if any(k in s for k in ["histor", "past", "encoder"]):
        return "historical"
    return "default"

def _clean_names(names):
    """Replace '-' and '_' with spaces and convert to Title Case."""
    return [str(n).replace("-", " ").replace("_", " ").title() for n in names]

def plot_var_bars(
    names,
    weights,
    title,
    filename,
    out_dir="Image/TopK",
    topk: int | None = None,
    dpi: int = 300,
    label_rot: int = 75,
    font_xtick: int = 8,
    use_alt_stripes: bool = False,
    override_color: str | None = None,
):
    """
    Plot variable importance bar chart with consistent style:
      - Automatically infer section (historical=blue, future=orange) 
        and apply corresponding color scheme
      - Default: all bars in the same section use one solid color
      - Optionally: alternating two colors within the same palette 
        (use_alt_stripes=True)
      - Variable names are cleaned (remove hyphen/underscore, Title Case)
      - No change required in main program
    """
    if weights is None or len(weights) == 0:
        return
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Align names with weights length
    n = len(weights)
    names = list(names) + [f"Var {i}" for i in range(len(names), n)] if len(names) < n else list(names[:n])

    # Sort by descending weight
    order = np.argsort(weights)[::-1]
    weights_sorted = np.array(weights)[order]
    names_sorted   = [names[i] for i in order]

    # Keep top-k if specified
    if isinstance(topk, int) and topk > 0:
        weights_sorted = weights_sorted[:topk]
        names_sorted   = names_sorted[:topk]

    # Clean variable names
    names_sorted = _clean_names(names_sorted)

    # Infer section and get palette
    sec = _infer_section_auto(title, filename, out_dir)
    pal = _PALETTES.get(sec, _PALETTES["default"])

    # Determine bar colors
    if override_color:
        bar_colors = [override_color] * len(weights_sorted)
    elif use_alt_stripes:
        base = pal["alt"]
        bar_colors = (base * (len(weights_sorted)//len(base) + 1))[:len(weights_sorted)]
    else:
        bar_colors = [pal["solid"]] * len(weights_sorted)

    # Set figure style
    fig_w = max(8, 0.35 * len(names_sorted))
    plt.rcParams.update({
        "figure.figsize": (fig_w, 6),
        "savefig.dpi": dpi,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": font_xtick,
        "ytick.labelsize": 10,
        "font.family": "DejaVu Sans",
    })

    # Plot
    plt.figure()
    plt.bar(range(len(weights_sorted)), weights_sorted, color=bar_colors, alpha=0.95)
    plt.title(title)
    plt.xlabel("Variables")
    plt.ylabel("Mean Selection Weight")
    plt.xticks(range(len(names_sorted)), names_sorted, rotation=label_rot, ha="right")
    plt.tight_layout()

    # Save to files
    png = os.path.join(out_dir, f"{filename}.png")
    svg = os.path.join(out_dir, f"{filename}.svg")
    plt.savefig(png, bbox_inches="tight")
    plt.savefig(svg, bbox_inches="tight")
    plt.close()

    # Debug print (optional)
    # print(f"[palette] section={sec}, solid={pal['solid']}, alt={pal['alt'][:2]}")
    # print(f"[Saved] {png} | {svg}")

    return png, svg
