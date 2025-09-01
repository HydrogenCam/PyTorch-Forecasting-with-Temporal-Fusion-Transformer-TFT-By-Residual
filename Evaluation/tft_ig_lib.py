# tft_ig_lib.py
# Utility functions for Integrated Gradients (IG) on TFT external variables.

import os, glob
import numpy as np
import torch
import matplotlib.pyplot as plt
import math



# ---------- FS & import helpers ----------
def latest_ckpt(version_dir: str) -> str:
    """Return the most recent .ckpt path under a Lightning version dir."""
    files = glob.glob(os.path.join(version_dir, "checkpoints", "*.ckpt")) + glob.glob(os.path.join(version_dir, "*.ckpt"))
    if not files:
        raise FileNotFoundError(f"No .ckpt found under: {version_dir}")
    return max(files, key=os.path.getmtime)

def move_to(x, device: torch.device):
    """Recursively move nested tensors to a device."""
    if torch.is_tensor(x): return x.to(device)
    if isinstance(x, (list, tuple)): return type(x)(move_to(xx, device) for xx in x)
    if isinstance(x, dict): return {k: move_to(v, device) for k, v in x.items()}
    return x

def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


# ---------- Column definition / dataloader helpers ----------
def names_from_column_definition(model):
    """
    Extract variable names from model.column_definition.
    Returns (observed_names, known_names).
    """
    observed, known = [], []
    coldef = getattr(model, "column_definition", None)
    if coldef is None:
        return observed, known
    for it in coldef:
        try:
            name, itype = it[0], str(it[2])
        except Exception:
            continue
        if "OBSERVED_INPUT" in itype:
            observed.append(name)
        elif "KNOWN_INPUT" in itype:
            known.append(name)
    return observed, known

def unpack_batch_for_io(batch):
    """
    Try to unpack a batch into ((x_known, x_observed, x_static), y).
    Adapt here if your repo uses a different structure.
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
    raise RuntimeError("Cannot unpack batch; adapt unpack_batch_for_io() to your dataloader format.")

def select_single_sample(packed_batch, idx: int):
    """
    Slice batch dim to keep only sample `idx` (retain batch dim=1).
    packed_batch: ((xk, xo, xs), y)
    """
    (xk, xo, xs), y = packed_batch
    def sel(t):
        if t is None: return None
        if t.dim() == 0: return t
        return t[idx:idx+1]
    return (sel(xk), sel(xo), sel(xs)), sel(y)

def build_batch_from_parts(xk, xo, xs, y):
    """Rebuild batch nesting for model calls."""
    return ((xk, xo, xs), y)


# ---------- IG core ----------
def find_first_tensor(x):
    """Find the first torch.Tensor in nested outputs."""
    if torch.is_tensor(x): return x
    if isinstance(x, (list, tuple)):
        for e in x:
            t = find_first_tensor(e)
            if t is not None: return t
    if isinstance(x, dict):
        for _, e in x.items():
            t = find_first_tensor(e)
            if t is not None: return t
    return None

def forward_scalar(model, batch):
    """
    Preferred and ONLY safe path for IG:
      model._prepare_tft_inputs(batch) -> model(all_inputs) -> mean(first_tensor).
    We avoid calling model(batch) directly and avoid training_step fallback,
    because they may perform in-place masking on leaf tensors that require grad.
    """
    if not hasattr(model, "_prepare_tft_inputs"):
        raise RuntimeError(
            "Model lacks _prepare_tft_inputs; please adapt forward_scalar() to build "
            "the exact (hist_regular, fut_regular, x_static[, hist_cate, fut_cate]) tuple for your forward."
        )
    all_inputs, _ = model._prepare_tft_inputs(batch)
    out = model(all_inputs)
    t = find_first_tensor(out)
    if t is None:
        raise RuntimeError("Forward returned no tensor to reduce.")
    return t.mean()

def integrated_gradients_single_input(
    model,
    base_batch,
    which: str = "known",     # "known" or "observed"
    steps: int = 32,
    zero_baseline: bool = True,
    baseline_eps: float = 1e-6,
    device: torch.device = torch.device("cpu"),
):
    """
    Compute IG for ONE input group on a single-sample batch.
    Returns CPU tensor [1, T, V] or None if that input is missing.
    """
    (xk, xo, xs), y = base_batch
    x = xk if which == "known" else xo
    if x is None:
        return None
    assert x.dim() == 3 and x.shape[0] == 1, "Expect [1, T, V] single-sample input."

    x0 = torch.zeros_like(x) if zero_baseline else torch.full_like(x, baseline_eps)
    diff = (x - x0).detach()
    grad_sum = torch.zeros_like(x, device=device)

    for s in range(1, steps + 1):
        alpha = float(s) / float(steps)
        x_scaled = x0 + alpha * diff
        x_scaled.requires_grad_(True)

        xk_in = x_scaled if which == "known" else xk
        xo_in = x_scaled if which == "observed" else xo

        batch_scaled = build_batch_from_parts(xk_in, xo_in, xs, y)
        model.zero_grad(set_to_none=True)
        scalar = forward_scalar(model, batch_scaled)  # <— only safe path
        scalar.backward()

        if x_scaled.grad is None:
            raise RuntimeError("No gradient on scaled input; ensure forward depends on this input.")
        grad_sum += x_scaled.grad.detach()

    avg_grad = grad_sum / float(steps)
    attributions = diff * avg_grad  # [1, T, V]
    return attributions.detach().cpu()


# ---------- Ranking & plotting ----------
def topk_indices_by_total_abs(attr_1T_V: torch.Tensor, k: int) -> list[int]:
    """Return indices of top-k variables by sum_t |IG|."""
    A = attr_1T_V.squeeze(0).abs().sum(dim=0).numpy()  # [V]
    k = min(k, len(A))
    idx = np.argsort(A)[-k:][::-1]
    return idx.tolist()

def plot_time_curves(attr_1T_V: torch.Tensor, names: list[str], idxs: list[int], title: str, out_png: str,
                     types_map: dict[str, str] | None = None, col_wrap: int = 2, sharey: bool = False):
    """
    Faceted time-series IG curves by variable types:
      - Keep the original signature but add optional 'types_map', 'col_wrap', 'sharey'.
      - If 'types_map' is None, types are inferred by keyword rules (price/fossil/renewable/weather/other).
      - One subplot per type; all subplots are composed into a single figure.
    """

    if attr_1T_V is None or len(idxs) == 0:
        return

    # Convert to numpy [T, V]
    A = attr_1T_V.squeeze(0).numpy()
    T = A.shape[0]
    t_axis = np.arange(T)

    # Helper: clean variable display names
    def _clean(n: str) -> str:
        return str(n).replace("-", " ").replace("_", " ").title()

    # Infer or use provided types
    def _infer_type(nm: str) -> str:
        s = nm.lower()
        if any(k in s for k in ["price", "tariff"]):
            return "Price"
        if any(k in s for k in ["coal", "gas", "oil", "fossil", "nuclear", "peat", "shale"]):
            return "Fossil"
        if any(k in s for k in ["wind", "solar", "hydro", "biomass", "geothermal", "marine", "renewable"]):
            return "Renewable"
        if any(k in s for k in ["temp", "rain", "snow", "cloud", "humidity", "pressure", "weather"]):
            return "Weather"
        return "Other"

    # Build mapping: variable name -> type
    name2type = {}
    for j in idxs:
        nm = names[j] if j < len(names) else f"var_{j}"
        if types_map is not None and nm in types_map:
            name2type[nm] = types_map[nm]
        else:
            name2type[nm] = _infer_type(nm)

    # Group selected indices by type
    groups: dict[str, list[tuple[str, int]]] = {}
    for j in idxs:
        nm = names[j] if j < len(names) else f"var_{j}"
        tp = name2type[nm]
        groups.setdefault(tp, []).append((nm, j))

    # Figure layout: one facet per type
    types = sorted(groups.keys())
    n_types = len(types)
    n_cols = max(1, col_wrap)
    n_rows = math.ceil(n_types / n_cols)

    # Width scales with number of subplots; height per row ~4.5
    fig_w = max(8.0, 5.0 * n_cols)
    fig_h = max(4.5, 4.5 * n_rows)

    # Color palettes per type (color-blind friendly families)
    type_color = {
        "Price":     ["#4E79A7", "#A0CBE8", "#2F5597"],
        "Fossil":    ["#E15759", "#FF9DA7", "#B63C43"],
        "Renewable": ["#59A14F", "#8CD17D", "#2E7D32"],
        "Weather":   ["#B07AA1", "#D4A6C8", "#7B6194"],
        "Other":     ["#9C755F", "#BAB0AC", "#7A5C47"],
    }

    # Create figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), sharex=True, sharey=sharey)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.reshape(n_rows, n_cols)

    # Plot each type on its subplot
    for k, tp in enumerate(types):
        r, c = divmod(k, n_cols)
        ax = axes[r, c]
        series = groups[tp]
        palette = type_color.get(tp, type_color["Other"])
        # cycle colors
        colors = (palette * (len(series)//len(palette) + 1))[:len(series)]

        for (nm, j), col in zip(series, colors):
            ax.plot(t_axis, A[:, j], label=_clean(nm), linewidth=1.8, alpha=0.95, color=col)

        ax.axhline(0.0, color="#888888", linewidth=1.0, alpha=0.6)
        ax.set_title(f"{tp} (n={len(series)})", fontsize=11)
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Attribution (IG)")
        # Compact legend per facet
        ax.legend(fontsize=8, loc="upper left", frameon=False, ncol=1)

    # Hide any unused axes (if n_types not multiple of n_cols)
    for k in range(n_types, n_rows * n_cols):
        r, c = divmod(k, n_cols)
        axes[r, c].axis("off")

    # Global title and layout
    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()

    # Save as PNG and SVG
    plt.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.savefig(out_png.replace(".png", ".svg"), bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"Saved: {out_png}  |  {out_png.replace('.png', '.svg')}")
