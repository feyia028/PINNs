# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# ============================================================
# pinn_airfoil_ablation_v2.py
#

#   - DATA      : pure supervised surrogate fit to CFD (x,y)->(u,v,p)
#   - BC_DATA   : supervised + boundary conditions
#   - PHYS_DATA : supervised + BCs + Navier–Stokes residual (PINN-style)
#

import os
import time
import json
import math
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt


# ----------------------------
# DEVICE
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
torch.set_default_dtype(DTYPE)

#Checks for GPU
#All numbers inside the network will be 32-bit floats.

# ----------------------------
# USER CONFIG (EDIT THESE)
# ----------------------------
RHO = 1.225  # kg/m^3

# Output folder for this script/run
OUT_DIR = "runs_ablation_v7"

# Modes to run for each case
MODES_TO_RUN = ["DATA", "BC_DATA", "PHYS_DATA"]  

# Training defaults - number of epochs and learning rate
# NOTE: older parts of the code expect a global name `EPOCHS`.
# Keep both for backwards-compatibility.
EPOCHS_MAX = 2600
EPOCHS = EPOCHS_MAX

# Physics curriculum (PDE loss ramp)
PHYS_PRETRAIN_EPOCHS = 800  # longer data+BC stabilization before physics on CPU
PHYS_RAMP_EPOCHS = 1200   # slower physics ramp (CPU friendly)

PDE_START = PHYS_PRETRAIN_EPOCHS
PDE_RAMP  = PHYS_RAMP_EPOCHS

BC_RAMP_EPOCHS = 1400  # ramp boundary losses to avoid wrecking DATA fit
N_COL_WALL = 3000         # extra near-wall collocation points (taken from FLUID domain points) to resolve boundary-layer physics
LR = 3e-4

N_DATA = 16000  # stronger supervised anchor
N_COL = 2000   # lower collocation (CPU) + importance sampling
N_COL_POOL = 50000
N_BC = 1200

# Mini-batch sizes (CPU: reduces step cost; improves stability)
BATCH_DATA = 4096
BATCH_BC   = 1024

# Early stopping: stop if no improvement for PATIENCE checks
PATIENCE = 260
MIN_DELTA = 5e-7

# Save checkpoint every N epochs (so you always get outputs)
SAVE_EVERY = 200

# Loss weights (tunable)
W_DATA = 3.0
W_IN = 0.7
W_WALL = 0.9
W_OUT = 0.25
W_FAR = 0.15
W_PDE = 0.06  # gentle physics (CPU)

# Network size 
NET_WIDTH = 192
NET_DEPTH = 7

ACT_FN = "silu"  # silu tends to optimize better than tanh for this surrogate/PINN mix  # activation for MLP: tanh is common for PINNs
# ---- Accuracy/quality improvements (safe defaults) ----
# Fourier features help represent sharp gradients / wakes.
USE_FOURIER = True
FOURIER_BANDS = 6          # 4*B features per (x,y) if using simple per-dim sin/cos bands
FOURIER_SCALE = 1.0        # scales (x,y) inside the sinusoid


# Add a speed supervision term (helps speed field + Cp indirectly)
LAMBDA_SPEED_DATA = 0.25
LAMBDA_P_DATA = 0.25        # keep pressure anchored (prevents p collapse)

# Optimisation stability
MAX_GRAD_NORM = 1.0         # gradient clipping
USE_LR_SCHEDULER = True     # cosine LR schedule over Adam phase

# Optional LBFGS refinement after Adam (usually helps PINNs)
USE_LBFGS = True
LBFGS_MODES = ["BC_DATA", "PHYS_DATA"]  # skip DATA; save time
LBFGS_STEPS = 60
LBFGS_LR = 1.0

# Contour resolution 
CONTOUR_NX = 400
CONTOUR_NY = 250

# Random seed for repeatability
SEED = 42

CASES = [
    ("Re1e5_AoA0", 1e5, 1.46, 0.0,
     "domain_points_Re1e5_AoA0.csv", "inlet_Re1e5_AoA0.csv", "outlet_Re1e5_AoA0.csv",
     "airfoil_wall.csv", "farfield_walls.csv", "cp_Re1e5_AoA0.csv"),

    ("Re5e5_AoA0", 5e5, 7.30, 0.0,
     "domain_points_Re5e5_AoA0.csv", "inlet_Re5e5_AoA0.csv", "outlet_Re5e5_AoA0.csv",
     "airfoil_wall.csv", "farfield_walls.csv", "cp_Re5e5_AoA0.csv"),

    ("Re1e5_AoA8", 1e5, 1.445, 0.203,
     "domain_points_Re1e5_AoA8.csv", "inlet_Re1e5_AoA8.csv", "outlet_Re1e5_AoA8.csv",
     "airfoil_wall.csv", "farfield_walls.csv", "cp_Re1e5_AoA8.csv"),

    ("Re5e5_AoA8", 5e5, 7.23, 1.02,
     "domain_points_Re5e5_AoA8.csv", "inlet_Re5e5_AoA8.csv", "outlet_Re5e5_AoA8.csv",
     "airfoil_wall.csv", "farfield_walls.csv", "cp_Re5e5_AoA8.csv"),
]


# ============================================================
# Utility: Error handling functions
# ============================================================
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed")]


def normalise_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = drop_unnamed(df)
    df.columns = [str(c).strip().lower() for c in df.columns]
    # common ANSYS name variants
    df = df.rename(columns={
        "x-coordinate": "x",
        "y-coordinate": "y",
        "x velocity": "u",
        "y velocity": "v",
        "static pressure": "p",
        "pressure": "p",
    })
    return df


def ensure_xy(df: pd.DataFrame) -> pd.DataFrame:
    df = normalise_cols(df)
    if "x" not in df.columns or "y" not in df.columns:
        raise ValueError(f"Boundary file missing x/y. Found columns: {df.columns.tolist()}")
    return df[["x", "y"]].dropna()


def ensure_domain(df: pd.DataFrame) -> pd.DataFrame:
    df = normalise_cols(df)
    needed = ["x", "y", "u", "v", "p"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Domain file missing {missing}. Found columns: {df.columns.tolist()}")
    return df[needed].dropna()


def read_cp(cp_csv: str) -> Optional[pd.DataFrame]:
    if not isinstance(cp_csv, str) or not os.path.exists(cp_csv):
        return None
    df = normalise_cols(pd.read_csv(cp_csv))
    # try to detect cp and x columns
    if "cp" not in df.columns:
        for c in df.columns:
            if "cp" in c:
                df = df.rename(columns={c: "cp"})
                break
    if "x" not in df.columns:
        for c in df.columns:
            if c.startswith("x"):
                df = df.rename(columns={c: "x"})
                break
    if "x" in df.columns and "cp" in df.columns:
        return df[["x", "cp"]].dropna()
    return None


def sample_df(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n >= len(df):
        return df
    return df.sample(n=n, random_state=seed, replace=False)


def minmax_scale(a: np.ndarray, amin: float, amax: float) -> np.ndarray:
    # scales to [-1, 1]
    return 2.0 * (a - amin) / (amax - amin + 1e-12) - 1.0


def to_t(x: np.ndarray) -> torch.Tensor:
    return torch.tensor(x, dtype=DTYPE, device=DEVICE)


# ============================================================
# Utility: Percentage accuracy (robust)
# ============================================================
# Accuracy definition: 100 * (1 - RMSE / std(true))
#  - 100% = perfect
#  - 0%   = no better than predicting the mean
#  - <0%  = worse than predicting the mean
def accuracy_pct(true: np.ndarray, pred: np.ndarray) -> float:
    """Return a *bounded* accuracy percentage in [0, 100].

    We define accuracy via a normalized RMSE against the RMS magnitude of the true
    signal (more stable than std when the field is near-constant):

        nRMSE = RMSE / (RMS(true) + eps)
        accuracy = 100 * (1 - nRMSE)

    Values below 0 are clipped to 0% (worse-than-baseline), and above 100 to 100%.
    """
    rmse_val = float(np.sqrt(np.mean((true - pred) ** 2)))
    rms_true = float(np.sqrt(np.mean(true ** 2)) + 1e-12)
    acc = 100.0 * (1.0 - rmse_val / rms_true)
    # keep it interpretable for reports
    return float(np.clip(acc, 0.0, 100.0))


# ============================================================
# Model: simple MLP Multi-Layer Perceptron
# ============================================================
class FourierFeatures(nn.Module):
    """Simple deterministic Fourier features for (x,y).
    Produces [sin(2^k*pi*s*x), cos(...), sin(...y), cos(...y)] for k=0..B-1.
    """
    def __init__(self, bands: int = 6, scale: float = 1.0):
        super().__init__()
        self.bands = int(bands)
        self.scale = float(scale)
        # frequencies: 1,2,4,...,2^(B-1)
        freqs = (2.0 ** torch.arange(self.bands)).float() * math.pi
        self.register_buffer("freqs", freqs)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        # xy: (N,2)
        x = xy[:, 0:1] * self.scale
        y = xy[:, 1:2] * self.scale
        # (N,B)
        fx = x * self.freqs[None, :]
        fy = y * self.freqs[None, :]
        feats = torch.cat([torch.sin(fx), torch.cos(fx), torch.sin(fy), torch.cos(fy)], dim=1)
        return feats


class MLP(nn.Module):
    """
    Inputs:  (x,y) in scaled coordinates [-1,1]
    Outputs: (u_nd, v_nd, p_nd) nondimensional
    """
    def __init__(
        self,
        width: int = NET_WIDTH,
        depth: int = NET_DEPTH,
        act: str = ACT_FN,
        use_fourier: bool = USE_FOURIER,
        fourier_bands: int = FOURIER_BANDS,
        fourier_scale: float = FOURIER_SCALE,
    ):
        super().__init__()
        self.use_fourier = bool(use_fourier)
        self.ff = FourierFeatures(fourier_bands, fourier_scale) if self.use_fourier else None

        in_dim = 2 + (4 * int(fourier_bands) if self.use_fourier else 0)

        acts = {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
        }
        a = acts.get(act.lower(), nn.Tanh())

        layers = []
        layers.append(nn.Linear(in_dim, width))
        layers.append(a)
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(a)
        layers.append(nn.Linear(width, 3))
        self.net = nn.Sequential(*layers)

        # Xavier init tends to be stable for PINNs
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_fourier:
            feats = self.ff(x)
            x = torch.cat([x, feats], dim=1)
        return self.net(x)


# ============================================================
# Warm-start helper: reuse DATA weights for BC_DATA / PHYS_DATA
# ============================================================
def maybe_load_data_warmstart(model: nn.Module, out_dir: str, case_name: str, mode: str) -> bool:
    """If running BC_DATA or PHYS_DATA, load the best DATA model (if it exists).
    This preserves the strong supervised surrogate fit and prevents BC/PDE from destroying it."""
    if mode not in ["BC_DATA", "PHYS_DATA"]:
        return False
    data_path = os.path.join(out_dir, f"{case_name}_DATA", "model_best.pt")
    if os.path.exists(data_path):
        try:
            model.load_state_dict(torch.load(data_path, map_location=DEVICE))
            print(f"[WARMSTART] Loaded DATA weights for {case_name} -> {mode}")
            return True
        except Exception as e:
            print(f"[WARMSTART] Failed to load DATA weights: {e}")
    return False


# ============================================================
# Physics: Navier–Stokes residuals (steady, incompressible, 2D)
# ============================================================
def gradients(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    dy/dx using autograd
    create_graph=True allows the computation second derivatives later
    """
    return autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]


def pde_residuals(
    model: nn.Module,
    XY: torch.Tensor,
    Re: float,
    sx: float,
    sy: float,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Steady incompressible Navier–Stokes in nondimensional form:

    continuity:   u_x + v_y = 0

    momentum-x:   u*u_x + v*u_y + p_x - (1/Re)*(u_xx + u_yy) = 0
    momentum-y:   u*v_x + v*v_y + p_y - (1/Re)*(v_xx + v_yy) = 0

    IMPORTANT NOTE ABOUT SCALING:
    -----------------------------
    - Network inputs are min-max scaled to [-1, 1] => autograd derivatives are w.r.t (x_hat, y_hat).
      We apply chain-rule correction with sx = d(x_hat)/dx and sy = d(y_hat)/dy.

    - The model is trained to output *standardized* nondimensional variables:
          z = (y - mean) / std
      For physics, we must work with the physical nondimensional variables y:
          y = z*std + mean
      Autograd derivatives are taken through this affine transform automatically when we form y first.
    """
    XY.requires_grad_(True)

    # Model outputs standardized nondim quantities (u_nd, v_nd, p_nd)
    out_z = model(XY)

    # De-standardize to nondimensional physical quantities
    out = out_z * y_std + y_mean
    u = out[:, 0:1]
    v = out[:, 1:2]
    p = out[:, 2:3]

    # first derivatives w.r.t. scaled inputs (x_hat, y_hat)
    gu_hat = gradients(u, XY)  # [N,2] => u_xhat, u_yhat
    gv_hat = gradients(v, XY)
    gp_hat = gradients(p, XY)

    # chain-rule: convert to physical derivatives (x,y)
    u_x = gu_hat[:, 0:1] * sx
    u_y = gu_hat[:, 1:2] * sy
    v_x = gv_hat[:, 0:1] * sx
    v_y = gv_hat[:, 1:2] * sy
    p_x = gp_hat[:, 0:1] * sx
    p_y = gp_hat[:, 1:2] * sy

    # second derivatives w.r.t. scaled inputs then chain-rule to physical second derivatives
    u_xx_hat = gradients(gu_hat[:, 0:1], XY)[:, 0:1]  # d/dxhat (u_xhat)
    u_yy_hat = gradients(gu_hat[:, 1:2], XY)[:, 1:2]  # d/dyhat (u_yhat)
    v_xx_hat = gradients(gv_hat[:, 0:1], XY)[:, 0:1]
    v_yy_hat = gradients(gv_hat[:, 1:2], XY)[:, 1:2]

    u_xx = u_xx_hat * (sx ** 2)
    u_yy = u_yy_hat * (sy ** 2)
    v_xx = v_xx_hat * (sx ** 2)
    v_yy = v_yy_hat * (sy ** 2)

    r_c = u_x + v_y
    r_u = (u * u_x + v * u_y) + p_x - (1.0 / Re) * (u_xx + u_yy)
    r_v = (u * v_x + v * v_y) + p_y - (1.0 / Re) * (v_xx + v_yy)

    return r_c, r_u, r_v


# ============================================================
# Cp helper
# ============================================================
def cp_from_p(p: np.ndarray, p_inf: float, rho: float, Uinf: float) -> np.ndarray:
    q = 0.5 * rho * (Uinf**2 + 1e-12)
    return (p - p_inf) / q


# ============================================================
# Plot helpers
# ============================================================
def save_loss_plot(losses: List[float], path_png: str) -> None:
    plt.figure()
    plt.plot(losses)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Total loss")
    plt.title("Training loss (log scale)")
    plt.tight_layout()
    plt.savefig(path_png, dpi=300)
    plt.close()


def save_scatter(true_vals: np.ndarray, pred_vals: np.ndarray, path_png: str, title: str, xlabel: str, ylabel: str) -> None:
    plt.figure()
    plt.scatter(true_vals, pred_vals, s=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path_png, dpi=300)
    plt.close()


def save_contour(
    xs: np.ndarray, ys: np.ndarray, field: np.ndarray,
    path_png: str, title: str, cbar_label: str,
    vmin: Optional[float] = None, vmax: Optional[float] = None,
    airfoil_csv: str = "airfoil_wall.csv"
) -> None:
    """
    Poster-quality contour plot with ANSYS-like styling:
    - jet colormap
    - smooth gradients
    - fixed limits when provided for fair true/pred comparison
    - white airfoil mask
    - high-resolution export
    """
    plt.figure(figsize=(10, 4), dpi=150)
    cf = plt.contourf(xs, ys, field, levels=100, cmap="jet", vmin=vmin, vmax=vmax)

    try:
        airfoil = normalise_cols(pd.read_csv(airfoil_csv))
        if "x" in airfoil.columns and "y" in airfoil.columns:
            plt.fill(airfoil["x"].values, airfoil["y"].values, color="white", zorder=10)
    except Exception:
        pass

    cbar = plt.colorbar(cf)
    cbar.set_label(cbar_label)
    cbar.ax.tick_params(labelsize=10)

    plt.title(title)
    plt.axis("equal")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path_png, dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close()


# ============================================================
# Core: Run one case + one mode
# ============================================================
def run_one(case: Tuple, mode: str) -> Dict:
    """
    mode options:
      - DATA
      - BC_DATA
      - PHYS_DATA
    """
    name, Re, Ux, Uy, dom_csv, in_csv, out_csv, wall_csv, far_csv, cp_csv = case

    # --- output directory for this run
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{name}_{mode}")
    os.makedirs(out_path, exist_ok=True)

    # --- load data
    dom = ensure_domain(pd.read_csv(dom_csv))
    inlet = ensure_xy(pd.read_csv(in_csv))
    outlet = ensure_xy(pd.read_csv(out_csv))
    wall = ensure_xy(pd.read_csv(wall_csv))
    far = None
    if isinstance(far_csv, str) and os.path.exists(far_csv):
        far = ensure_xy(pd.read_csv(far_csv))
    cp_df = read_cp(cp_csv)

    # --- scaling bounds for x,y -> [-1,1]
    x_min, x_max = float(dom["x"].min()), float(dom["x"].max())
    y_min, y_max = float(dom["y"].min()), float(dom["y"].max())
    # --- scaling factors for chain-rule corrections in PDE residuals
    # Inputs are scaled to x_hat,y_hat in [-1,1] via minmax_scale(). Autograd derivatives are w.r.t. x_hat,y_hat.
    # We need sx = d(x_hat)/dx and sy = d(y_hat)/dy to map derivatives back to physical space in the PDE residual.
    sx = 2.0 / (x_max - x_min + 1e-12)
    sy = 2.0 / (y_max - y_min + 1e-12)

    def scale_xy(df_xy: pd.DataFrame) -> torch.Tensor:
        xs = minmax_scale(df_xy["x"].values.astype(np.float64), x_min, x_max)
        ys = minmax_scale(df_xy["y"].values.astype(np.float64), y_min, y_max)
        return to_t(np.stack([xs, ys], axis=1))

    # --- freestream magnitude
    Uinf = float(math.sqrt(Ux**2 + Uy**2))

    # --- estimate p_inf from "outer ring" of domain points (far-field region)
    # This helps Cp scaling if ANSYS pressure is gauge-ish.
    XYs = np.stack([
        minmax_scale(dom["x"].values, x_min, x_max),
        minmax_scale(dom["y"].values, y_min, y_max)
    ], axis=1)
    cent = XYs.mean(axis=0, keepdims=True)
    r = np.linalg.norm(XYs - cent, axis=1)
    k_far = min(500, max(100, len(dom) // 200))
    idx_far = np.argsort(r)[-k_far:]
    p_inf = float(dom["p"].values[idx_far].mean())

    # --- build nondimensional targets for supervised training
    # u_nd = u / Uinf, v_nd = v / Uinf
    # p_nd = (p - p_inf) / (rho*Uinf^2)
    dom = dom.copy()
    dom["u_nd"] = dom["u"].values / (Uinf + 1e-12)
    dom["v_nd"] = dom["v"].values / (Uinf + 1e-12)
    dom["p_nd"] = (dom["p"].values - p_inf) / (RHO * (Uinf**2 + 1e-12))

    # --- sample fixed sets for fairness in comparisons (same size each run)
    dom_dat = sample_df(dom, N_DATA, seed=SEED + 2)
    dom_col = sample_df(dom, N_COL, seed=SEED + 1)

    X_dat = scale_xy(dom_dat[["x", "y"]])
    Y_dat = to_t(dom_dat[["u_nd", "v_nd", "p_nd"]].values.astype(np.float64))

    # --- output normalization (helps pressure learning + keeps losses balanced)
    y_mean = Y_dat.mean(dim=0, keepdim=True)
    y_std = Y_dat.std(dim=0, keepdim=True).clamp_min(1e-6)
    Y_dat_n = (Y_dat - y_mean) / y_std

    X_col = scale_xy(dom_col[["x", "y"]])

    # --- PDE collocation importance sampling pool (FASTER + BETTER)
    # Uniform random collocation often misses the near-wall region where gradients are steep.
    # Instead of evaluating PDE residual on a huge set each epoch (slow), we build a pool once,
    # compute a distance-to-wall for each pool point, then sample collocation points with a bias
    # toward the wall (but still keeping some farfield coverage).
    #
    # This improves quality and reduces runtime versus using very large N_COL.
    try:
        from scipy.spatial import cKDTree
        wall_pts = wall[["x", "y"]].values.astype(np.float64)
        dom_pool = sample_df(dom, min(N_COL_POOL, len(dom)), seed=SEED + 123)
        pool_xy = dom_pool[["x", "y"]].values.astype(np.float64)
        tree = cKDTree(wall_pts)
        dists, _ = tree.query(pool_xy, k=1)
        chord = float(wall["x"].max() - wall["x"].min()) if ("x" in wall.columns) else float(x_max - x_min)
        eps = max(1e-6, 0.02 * chord)
        # weights: mix of uniform + near-wall emphasis
        w_np = 0.25 + 0.75 * np.exp(-dists / eps)
        w_np = w_np / (w_np.sum() + 1e-12)
        X_pool_t = scale_xy(dom_pool[["x", "y"]])
        w_pool_t = torch.tensor(w_np, dtype=DTYPE, device=DEVICE)
    except Exception:
        # fallback: no scipy; just use uniform collocation from dom_col
        X_pool_t = X_col
        w_pool_t = torch.ones((X_pool_t.shape[0],), dtype=DTYPE, device=DEVICE)
        w_pool_t = w_pool_t / (w_pool_t.sum() + 1e-12)
    # --- boundary sampling
    X_in = scale_xy(sample_df(inlet, N_BC, seed=SEED + 3))
    X_out = scale_xy(sample_df(outlet, N_BC, seed=SEED + 4))
    X_wall = scale_xy(sample_df(wall, N_BC, seed=SEED + 5))
    X_far = None
    if far is not None:
        X_far = scale_xy(sample_df(far, N_BC, seed=SEED + 6))

    # --- nondimensional inlet target
    uin = float(Ux / (Uinf + 1e-12))
    vin = float(Uy / (Uinf + 1e-12))

    # Normalized BC targets in the same space as the network outputs
    uin_n = (uin - float(y_mean[0,0].item())) / float(y_std[0,0].item())
    vin_n = (vin - float(y_mean[0,1].item())) / float(y_std[0,1].item())
    uwall_n = (0.0 - float(y_mean[0,0].item())) / float(y_std[0,0].item())
    vwall_n = (0.0 - float(y_mean[0,1].item())) / float(y_std[0,1].item())

    # --- model + optimiser
    model = MLP(width=NET_WIDTH, depth=NET_DEPTH, act=ACT_FN, use_fourier=USE_FOURIER, fourier_bands=FOURIER_BANDS, fourier_scale=FOURIER_SCALE).to(DEVICE)
    # Warm-start from DATA model when running BC/PINN modes
    _ = maybe_load_data_warmstart(model, OUT_DIR, name, mode)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-6)
    scaler = GradScaler('cuda', enabled=(DEVICE == 'cuda'))

    # Reduce learning rate gradually for stable refinement
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS) if USE_LR_SCHEDULER else None

    # --- training loop with early stopping and guaranteed periodic saves
    losses: List[float] = []
    best_loss = float("inf")
    best_epoch = 0
    no_improve = 0

    t0 = time.time()

    #small run config for traceability
    run_config = {
        "case": name,
        "mode": mode,
        "Re": float(Re),
        "Ux": float(Ux), "Uy": float(Uy), "Uinf": float(Uinf),
        "device": DEVICE,
        "torch_version": torch.__version__,
        "python": sys.version.split()[0],
        "epochs_max": EPOCHS_MAX,
        "lr": LR,
        "N_DATA": N_DATA,
        "N_COL": N_COL,
        "N_BC": N_BC,
        "weights": {
            "W_DATA": W_DATA, "W_IN": W_IN, "W_WALL": W_WALL, "W_OUT": W_OUT, "W_FAR": W_FAR, "W_PDE": W_PDE
        },
        "scaling": {"p_inf": p_inf, "rho": RHO, "y_mean": [float(x) for x in y_mean.squeeze().tolist()], "y_std": [float(x) for x in y_std.squeeze().tolist()]}
    }
    with open(os.path.join(out_path, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    # Don't early-stop before curriculum phases have a chance to run
    if mode == "PHYS_DATA":
        MIN_EPOCHS_BEFORE_EARLYSTOP = int(PDE_START + 400)
    elif mode == "BC_DATA":
        MIN_EPOCHS_BEFORE_EARLYSTOP = int(BC_RAMP_EPOCHS + 200)
    else:
        MIN_EPOCHS_BEFORE_EARLYSTOP = 600

    for ep in range(1, EPOCHS_MAX + 1):
        opt.zero_grad(set_to_none=True)

        # Ramp boundary losses so DATA fit is not destroyed early
        bc_eff = float(min(1.0, ep / (BC_RAMP_EPOCHS + 1e-12)))

        # Mixed precision on GPU speeds up training substantially
        with autocast("cuda", enabled=(DEVICE == "cuda")):
            total_loss = 0.0

            # -----------------------------
            # -----------------------------
            # 1) DATA LOSS (Surrogate fit)
            # -----------------------------
            # The network acts like a "surrogate model" for CFD:
            #   it learns (x,y) -> (u_nd, v_nd, p_nd)
            #
            # NOTE:
            # The model outputs are *standardized* (z = (y-mean)/std) to balance u/v/p losses.
            # For any physical quantity (speed, PDE residuals, evaluation), we de-standardize first.
            if mode in ["DATA", "BC_DATA", "PHYS_DATA"]:
                idx_d = torch.randperm(X_dat.shape[0], device=DEVICE)[:min(BATCH_DATA, X_dat.shape[0])]
                Xd = X_dat[idx_d]
                Yd = Y_dat[idx_d]
                Yd_n = Y_dat_n[idx_d]
                pred_z = model(Xd)

                # main supervised loss in standardized space (balances u/v/p automatically)
                l_data_uvp = F.smooth_l1_loss(pred_z, Yd_n, beta=0.5)

                # extra supervision on speed magnitude (computed in nondimensional physical space)
                pred_y = pred_z * y_std + y_mean
                u_pred, v_pred = pred_y[:, 0:1], pred_y[:, 1:2]
                u_true, v_true = Yd[:, 0:1], Yd[:, 1:2]
                sp_pred = torch.sqrt(u_pred**2 + v_pred**2 + 1e-12)
                sp_true = torch.sqrt(u_true**2 + v_true**2 + 1e-12)
                l_data_speed = F.smooth_l1_loss(sp_pred, sp_true, beta=0.5)

                l_data_p = F.smooth_l1_loss(pred_y[:, 2:3], Yd[:, 2:3], beta=0.5)
                l_data = l_data_uvp + LAMBDA_SPEED_DATA * l_data_speed + LAMBDA_P_DATA * l_data_p
                total_loss = total_loss + W_DATA * l_data
            else:
                l_data = torch.tensor(0.0, device=DEVICE)

            # -----------------------------
            # -----------------------------
            # 2) BOUNDARY CONDITION LOSSES
            # -----------------------------
            # Inlet: enforce u=uin, v=vin
            # Wall: no-slip u=v=0
            # Outlet: zero-gradient (Neumann) for u,v,p
            # Farfield: velocity ~ inlet velocity and pressure ~ 0 (nondimensional gauge)
            #
            # NOTE: BC losses are enforced in the model's standardized output space.
            if mode in ["BC_DATA", "PHYS", "PHYS_DATA"]:
                idx_in = torch.randperm(X_in.shape[0], device=DEVICE)[:min(BATCH_BC, X_in.shape[0])]
                pred_in_z = model(X_in[idx_in])
                l_in = (pred_in_z[:, 0:1] - uin_n).pow(2).mean() + (pred_in_z[:, 1:2] - vin_n).pow(2).mean()
                total_loss = total_loss + bc_eff * W_IN * l_in

                idx_w = torch.randperm(X_wall.shape[0], device=DEVICE)[:min(BATCH_BC, X_wall.shape[0])]
                pred_wall_z = model(X_wall[idx_w])
                l_wall = (pred_wall_z[:, 0:1] - uwall_n).pow(2).mean() + (pred_wall_z[:, 1:2] - vwall_n).pow(2).mean()
                total_loss = total_loss + bc_eff * W_WALL * l_wall

                # Outlet Neumann BC: d()/dx = 0 (computed on de-standardized fields)
                idx_o = torch.randperm(X_out.shape[0], device=DEVICE)[:min(BATCH_BC, X_out.shape[0])]
                XY_out = X_out[idx_o].clone().detach().requires_grad_(True)
                out_o_z = model(XY_out)
                out_o = out_o_z * y_std + y_mean
                uo = out_o[:, 0:1]; vo = out_o[:, 1:2]; po = out_o[:, 2:3]
                guo = gradients(uo, XY_out)
                gvo = gradients(vo, XY_out)
                gpo = gradients(po, XY_out)
                du_dx = guo[:, 0:1] * sx
                dv_dx = gvo[:, 0:1] * sx
                dp_dx = gpo[:, 0:1] * sx
                l_out = du_dx.pow(2).mean() + dv_dx.pow(2).mean() + dp_dx.pow(2).mean()
                total_loss = total_loss + bc_eff * W_OUT * l_out

                if X_far is not None:
                    idx_f = torch.randperm(X_far.shape[0], device=DEVICE)[:min(BATCH_BC, X_far.shape[0])]
                    pred_far_z = model(X_far[idx_f])
                    p0_n = (0.0 - float(y_mean[0,2].item())) / float(y_std[0,2].item())
                    l_far = (pred_far_z[:, 0:1] - uin_n).pow(2).mean() + (pred_far_z[:, 1:2] - vin_n).pow(2).mean() + (pred_far_z[:, 2:3] - p0_n).pow(2).mean()
                    total_loss = total_loss + bc_eff * W_FAR * l_far
                else:
                    l_far = torch.tensor(0.0, device=DEVICE)
            else:
                l_in = l_wall = l_out = l_far = torch.tensor(0.0, device=DEVICE)

            # -----------------------------
            # 3) PHYSICS LOSS (PINN term)
            # -----------------------------
            # PDE residual encourages the network outputs to satisfy NS.
            # This is the expensive part (autograd + second derivatives).
            # ---- Curriculum: ramp physics on earlier (helps avoid over-smooth surrogate fits) ----
            if mode in ["PHYS", "PHYS_DATA"] and ep > PDE_START:
                # Sample collocation points from the pool with near-wall bias
                if X_pool_t.shape[0] <= N_COL:
                    Xc = X_pool_t
                else:
                    idx = torch.multinomial(w_pool_t, num_samples=N_COL, replacement=False)
                    Xc = X_pool_t[idx]

                rc, ru, rv = pde_residuals(model, Xc, Re=Re, sx=sx, sy=sy, y_mean=y_mean, y_std=y_std)
                # balance continuity slightly higher to suppress nonphysical swirling modes
                l_pde = rc.pow(2).mean() + ru.pow(2).mean() + rv.pow(2).mean()
                w_pde_eff = W_PDE * float(min(1.0, (ep - PDE_START) / (PDE_RAMP + 1e-12)))
                total_loss = total_loss + w_pde_eff * l_pde
            else:
                l_pde = torch.tensor(0.0, device=DEVICE)

        # backprop + step
        scaler.scale(total_loss).backward()
        # AMP-safe grad clipping
        scaler.unscale_(opt)
        if MAX_GRAD_NORM is not None and MAX_GRAD_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

        scaler.step(opt)
        scaler.update()

        if scheduler is not None:
            scheduler.step()
        loss_val = float(total_loss.detach().item())
        losses.append(loss_val)

        # status printing 
        if ep == 1 or ep % 200 == 0:
            print(
                f"[{name} | {mode}] ep={ep:5d} loss={loss_val:.3e} "
                f"data={float(l_data.item()):.2e} bc(in/wall/out/far)={float(l_in.item()):.2e}/{float(l_wall.item()):.2e}/{float(l_out.item()):.2e}/{float(l_far.item()):.2e} "
                f"pde={float(l_pde.item()):.2e} (w={float(w_pde_eff) if 'w_pde_eff' in locals() else 0.0:.2e})"
            )

        # early stopping bookkeeping
        if loss_val < best_loss - MIN_DELTA:
            best_loss = loss_val
            best_epoch = ep
            no_improve = 0
            # save best model immediately
            torch.save(model.state_dict(), os.path.join(out_path, "model_best.pt"))
        else:
            no_improve += 1

        # periodic save so you ALWAYS have outputs
        if ep % SAVE_EVERY == 0:
            np.savetxt(os.path.join(out_path, "loss.txt"), np.array(losses))
            save_loss_plot(losses, os.path.join(out_path, "loss.png"))
            # quick heartbeat metrics
            heartbeat = {"epoch": ep, "best_loss": best_loss, "best_epoch": best_epoch}
            with open(os.path.join(out_path, "heartbeat.json"), "w") as f:
                json.dump(heartbeat, f, indent=2)

        if (ep >= MIN_EPOCHS_BEFORE_EARLYSTOP) and (no_improve >= PATIENCE):
            print(f"[EARLY STOP] {name} {mode} stopped at ep={ep} (best ep={best_epoch}, best loss={best_loss:.3e})")
            break


    # ------------------------------------------------------------
    # Optional LBFGS refinement (often improves accuracy noticeably)
    # ------------------------------------------------------------
    # We run a short LBFGS phase *after* Adam converges. This tends to:
    #  - tighten BC satisfaction
    #  - sharpen wakes / pressure gradients
    #  - reduce residual "blobs" in PINN fields
    #
    # LBFGS is run in full precision (no AMP).
    if USE_LBFGS and (mode in LBFGS_MODES):
        # Fix a deterministic collocation batch for LBFGS (LBFGS expects a stable objective).
        if mode in ["PHYS", "PHYS_DATA"] and X_pool_t.shape[0] > 0:
            if X_pool_t.shape[0] <= N_COL:
                Xc_lbfgs = X_pool_t
            else:
                # sample once with the same near-wall bias as Adam
                idx_lb = torch.multinomial(w_pool_t, num_samples=N_COL, replacement=False)
                Xc_lbfgs = X_pool_t[idx_lb]
        else:
            Xc_lbfgs = None

        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=LBFGS_LR,
            max_iter=LBFGS_STEPS,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        def closure():
            lbfgs.zero_grad(set_to_none=True)
            total = torch.tensor(0.0, device=DEVICE)

            # Data term
            if mode in ["DATA", "BC_DATA", "PHYS_DATA"]:
                pred_z = model(X_dat)
                l_uvp = (pred_z - Y_dat_n).pow(2).mean()
                pred_y = pred_z * y_std + y_mean
                u_pred, v_pred = pred_y[:, 0:1], pred_y[:, 1:2]
                u_true, v_true = Yd[:, 0:1], Yd[:, 1:2]
                sp_pred = torch.sqrt(u_pred**2 + v_pred**2 + 1e-12)
                sp_true = torch.sqrt(u_true**2 + v_true**2 + 1e-12)
                l_sp = (sp_pred - sp_true).pow(2).mean()
                l_d = l_uvp + LAMBDA_SPEED_DATA * l_sp
                total = total + W_DATA * l_d

            # BC terms
            if mode in ["BC_DATA", "PHYS", "PHYS_DATA"]:
                pred_in_z = model(X_in)
                l_in_ = (pred_in_z[:, 0:1] - uin_n).pow(2).mean() + (pred_in_z[:, 1:2] - vin_n).pow(2).mean()
                total = total + W_IN * l_in_

                pred_wall_z = model(X_wall)
                l_wall_ = (pred_wall_z[:, 0:1] - uwall_n).pow(2).mean() + (pred_wall_z[:, 1:2] - vwall_n).pow(2).mean()
                total = total + W_WALL * l_wall_

                XY_out = X_out.clone().detach().requires_grad_(True)
                out_o_z = model(XY_out)
                out_o = out_o_z * y_std + y_mean
                uo = out_o[:, 0:1]; vo = out_o[:, 1:2]; po = out_o[:, 2:3]
                guo = gradients(uo, XY_out)
                gvo = gradients(vo, XY_out)
                gpo = gradients(po, XY_out)
                du_dx = guo[:, 0:1] * sx
                dv_dx = gvo[:, 0:1] * sx
                dp_dx = gpo[:, 0:1] * sx
                l_out_ = du_dx.pow(2).mean() + dv_dx.pow(2).mean() + dp_dx.pow(2).mean()
                total = total + W_OUT * l_out_

                if X_far is not None:
                    pred_far_z = model(X_far)
                    p0_n = (0.0 - float(y_mean[0,2].item())) / float(y_std[0,2].item())
                    l_far_ = (pred_far_z[:, 0:1] - uin_n).pow(2).mean() + (pred_far_z[:, 1:2] - vin_n).pow(2).mean() + (pred_far_z[:, 2:3] - p0_n).pow(2).mean()
                    total = total + W_FAR * l_far_

            # PDE term (full strength)
            if mode in ["PHYS", "PHYS_DATA"] and (Xc_lbfgs is not None):
                rc, ru, rv = pde_residuals(model, Xc_lbfgs, Re=Re, sx=sx, sy=sy, y_mean=y_mean, y_std=y_std)
                l_p = 2.0 * rc.pow(2).mean() + ru.pow(2).mean() + rv.pow(2).mean()
                total = total + W_PDE * l_p

            total.backward()
            return total

        # run LBFGS steps
        try:
            lbfgs_loss = lbfgs.step(closure)
            print(f"[LBFGS] {name} {mode} done | final_loss={lbfgs_loss.detach().item():.3e}")
        except Exception as e:
            print(f"[LBFGS] {name} {mode} skipped due to error: {e}")

    epochs_ran = ep
    train_time = time.time() - t0

    # ============================================================
    # EVALUATION (creates files every run)
    # ============================================================
    # Predict full domain in dimensional units for plotting/metrics
    XY_all = scale_xy(dom[["x", "y"]])
    with torch.no_grad():
        pred_nd = (model(XY_all) * y_std + y_mean).detach().cpu().numpy()

    # Convert nondim -> dimensional
    u_pred = pred_nd[:, 0] * Uinf
    v_pred = pred_nd[:, 1] * Uinf
    p_pred = pred_nd[:, 2] * (RHO * Uinf**2) + p_inf

    u_true = dom["u"].values
    v_true = dom["v"].values
    p_true = dom["p"].values

    sp_true = np.sqrt(u_true**2 + v_true**2)
    sp_pred = np.sqrt(u_pred**2 + v_pred**2)

    def rmse(a, b) -> float:
        return float(np.sqrt(np.mean((a - b)**2)))

    def mae(a, b) -> float:
        return float(np.mean(np.abs(a - b)))

    # PDE residual diagnostics on a subset (cheap-ish)
    # For DATA runs, PDE residual will likely be large; for PHYS_DATA it should reduce.
    pde_diag = {}
    try:
        X_diag = scale_xy(sample_df(dom, min(1000, len(dom)), seed=SEED + 99)[["x", "y"]])
        with torch.no_grad():
            pass
        # PDE needs gradients => no torch.no_grad here
        rc, ru, rv = pde_residuals(model, X_diag, Re=Re, sx=sx, sy=sy, y_mean=y_mean, y_std=y_std)
        pde_diag = {
            "pde_rmse_continuity": float(torch.sqrt((rc**2).mean()).detach().cpu().item()),
            "pde_rmse_mom_u": float(torch.sqrt((ru**2).mean()).detach().cpu().item()),
            "pde_rmse_mom_v": float(torch.sqrt((rv**2).mean()).detach().cpu().item()),
            "pde_mean_abs_continuity": float(rc.abs().mean().detach().cpu().item()),
            "pde_mean_abs_mom_u": float(ru.abs().mean().detach().cpu().item()),
            "pde_mean_abs_mom_v": float(rv.abs().mean().detach().cpu().item()),
        }
    except Exception as e:
        pde_diag = {"pde_diag_error": str(e)}

    # Inference timing (one forward pass on full domain)
    t_inf0 = time.time()
    with torch.no_grad():
        _ = model(XY_all)
    inf_time = time.time() - t_inf0

    metrics = {
        "case": name,
        "mode": mode,
        "Re": float(Re),
        "Ux": float(Ux), "Uy": float(Uy), "Uinf": float(Uinf),
        "device": DEVICE,
        "torch_version": torch.__version__,
        "python": sys.version.split()[0],
        "epochs_ran": int(epochs_ran),
        "epochs_max": int(EPOCHS_MAX),
        "train_time_s": float(train_time),
        "sec_per_epoch": float(train_time / max(1, epochs_ran)),
        "inference_time_s_full_domain": float(inf_time),
        "p_inf_used": float(p_inf),
        "best_loss": float(best_loss),
        "best_epoch": int(best_epoch),
        "rmse_u": rmse(u_true, u_pred),
        "rmse_v": rmse(v_true, v_pred),
        "rmse_p": rmse(p_true, p_pred),
        "rmse_speed": rmse(sp_true, sp_pred),
        "mae_u": mae(u_true, u_pred),
        "mae_v": mae(v_true, v_pred),
        "mae_p": mae(p_true, p_pred),
        "mae_speed": mae(sp_true, sp_pred),
        **pde_diag
    }

    # ------------------------------------------------------------
    # Percentage accuracy metrics (robust, avoids MAPE blow-ups)
    # ------------------------------------------------------------
    metrics["acc_u_pct"] = accuracy_pct(u_true, u_pred)
    metrics["acc_v_pct"] = accuracy_pct(v_true, v_pred)
    metrics["acc_p_pct"] = accuracy_pct(p_true, p_pred)
    metrics["acc_speed_pct"] = accuracy_pct(sp_true, sp_pred)

    metrics["acc_overall_pct"] = float(np.mean([
        metrics["acc_u_pct"],
        metrics["acc_v_pct"],
        metrics["acc_p_pct"],
        metrics["acc_speed_pct"]
    ]))

    print(f"[ACCURACY] {name} {mode} | Overall={metrics['acc_overall_pct']:.2f}% "
          f"(u={metrics['acc_u_pct']:.2f}%, v={metrics['acc_v_pct']:.2f}%, "
          f"p={metrics['acc_p_pct']:.2f}%, speed={metrics['acc_speed_pct']:.2f}%)")

    # --- Always save metrics + loss plots
    with open(os.path.join(out_path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    np.savetxt(os.path.join(out_path, "loss.txt"), np.array(losses))
    save_loss_plot(losses, os.path.join(out_path, "loss.png"))

    # --- Scatter plot: speed (CFD vs model)
    save_scatter(
        sp_true, sp_pred,
        os.path.join(out_path, "scatter_speed.png"),
        title=f"Speed scatter: {name} ({mode})",
        xlabel="CFD speed [m/s]",
        ylabel="Model speed [m/s]"
    )

    # ============================================================
    # Flow-field contour plots 
    # ============================================================
    # Plot predicted speed and predicted pressure on a grid.
    # Note: This is a visualisation grid, not the CFD mesh.
    xg = np.linspace(x_min, x_max, CONTOUR_NX)
    yg = np.linspace(y_min, y_max, CONTOUR_NY)
    XG, YG = np.meshgrid(xg, yg)

    XG_s = minmax_scale(XG.reshape(-1), x_min, x_max)
    YG_s = minmax_scale(YG.reshape(-1), y_min, y_max)
    XYG = to_t(np.stack([XG_s, YG_s], axis=1))

    with torch.no_grad():
        predg_nd = (model(XYG) * y_std + y_mean).detach().cpu().numpy()

    ug = predg_nd[:, 0].reshape(CONTOUR_NY, CONTOUR_NX) * Uinf
    vg = predg_nd[:, 1].reshape(CONTOUR_NY, CONTOUR_NX) * Uinf
    pg = predg_nd[:, 2].reshape(CONTOUR_NY, CONTOUR_NX) * (RHO * Uinf**2) + p_inf
    spg = np.sqrt(ug**2 + vg**2)

    speed_vmin = float(np.nanmin(sp_true))
    speed_vmax = float(np.nanmax(sp_true))
    pressure_vmin = float(np.nanmin(p_true))
    pressure_vmax = float(np.nanmax(p_true))

    save_contour(
        XG, YG, spg,
        os.path.join(out_path, "contour_speed.png"),
        title=f"Predicted speed contour: {name} ({mode})",
        cbar_label="Speed [m/s]",
        vmin=speed_vmin,
        vmax=speed_vmax,
        airfoil_csv=wall_csv
    )

    save_contour(
        XG, YG, pg,
        os.path.join(out_path, "contour_pressure.png"),
        title=f"Predicted pressure contour: {name} ({mode})",
        cbar_label="Pressure [Pa]",
        vmin=pressure_vmin,
        vmax=pressure_vmax,
        airfoil_csv=wall_csv
    )
    
        # ============================================================
    # TRUE CFD CONTOURS (Ground Truth for comparison)
    # ============================================================
    # These use the original CFD domain values (u_true, v_true, p_true)
    # and interpolate them onto the same visualization grid used above.
    #
    # This allows direct visual comparison between:
    #   - contour_speed.png          (PINN prediction)
    #   - contour_speed_TRUE.png     (CFD truth)
    #
    #   - contour_pressure.png       (PINN prediction)
    #   - contour_pressure_TRUE.png  (CFD truth)
    
    from scipy.interpolate import griddata
    
    # Interpolate CFD values onto visualization grid
    points = np.column_stack((dom["x"].values, dom["y"].values))
    
    # Speed (true CFD)
    sp_true_grid = griddata(
        points,
        sp_true,
        (XG, YG),
        method="linear"
    )
    
    # Pressure (true CFD)
    p_true_grid = griddata(
        points,
        p_true,
        (XG, YG),
        method="linear"
    )
    
    # Replace NaNs (outside convex hull) for clean plotting
    sp_true_grid = np.nan_to_num(sp_true_grid)
    p_true_grid = np.nan_to_num(p_true_grid)
    
    # Save TRUE speed contour
    save_contour(
        XG, YG, sp_true_grid,
        os.path.join(out_path, "contour_speed_TRUE.png"),
        title=f"CFD TRUE speed contour: {name}",
        cbar_label="Speed [m/s]",
        vmin=speed_vmin,
        vmax=speed_vmax,
        airfoil_csv=wall_csv
    )

    # Save TRUE pressure contour
    save_contour(
        XG, YG, p_true_grid,
        os.path.join(out_path, "contour_pressure_TRUE.png"),
        title=f"CFD TRUE pressure contour: {name}",
        cbar_label="Pressure [Pa]",
        vmin=pressure_vmin,
        vmax=pressure_vmax,
        airfoil_csv=wall_csv
    )

    # ============================================================
    # Cp comparison (if available)
    # ============================================================
    # Predicts pressure at wall points (airfoil surface),
    # convert to Cp and compare with  exported Cp(x) from ANSYS.
    if cp_df is not None and len(cp_df) > 10:
        wall_xy = wall.copy()

        # predict p on wall points
        Xw = scale_xy(wall_xy)
        with torch.no_grad():
            pw_nd = model(Xw)[:, 2:3].detach().cpu().numpy().reshape(-1)
        pw = pw_nd * (RHO * Uinf**2) + p_inf
        Cp_pred = cp_from_p(pw, p_inf, RHO, Uinf)

        # match CFD Cp to wall points.
        # The old nearest-x mapping + unsorted wall points can create a "spider web" Cp plot.
        # Here we:
        #   1) sort CFD Cp by x and interpolate onto wall x
        #   2) split wall into upper/lower surfaces using y sign
        #   3) sort each branch by x before plotting
        x_wall = wall_xy["x"].values
        y_wall = wall_xy["y"].values

        x_cp = cp_df["x"].values
        cp_true_raw = cp_df["cp"].values

        # sort Cp data by x for stable interpolation
        order_cp = np.argsort(x_cp)
        x_cp_s = x_cp[order_cp]
        cp_cp_s = cp_true_raw[order_cp]

        # handle potential duplicate x values in Cp file by averaging (prevents np.interp issues)
        # (keeps behaviour robust without deleting any user comments elsewhere)
        x_unique, inv = np.unique(x_cp_s, return_inverse=True)
        cp_accum = np.zeros_like(x_unique, dtype=np.float64)
        cp_count = np.zeros_like(x_unique, dtype=np.float64)
        np.add.at(cp_accum, inv, cp_cp_s)
        np.add.at(cp_count, inv, 1.0)
        cp_unique = cp_accum / np.maximum(cp_count, 1.0)

        # interpolate CFD Cp onto each wall x
        Cp_true = np.interp(x_wall, x_unique, cp_unique)

        metrics["rmse_Cp"] = rmse(Cp_true, Cp_pred)
        metrics["mae_Cp"] = mae(Cp_true, Cp_pred)

        # plot Cp (split upper/lower + sort)
        plt.figure()

        up = y_wall >= 0.0
        lo = ~up

        # upper surface
        if np.any(up):
            o = np.argsort(x_wall[up])
            plt.plot(x_wall[up][o], Cp_true[up][o], label="CFD Cp (upper)")
            plt.plot(x_wall[up][o], Cp_pred[up][o], label=f"{mode} Cp (upper)")

        # lower surface
        if np.any(lo):
            o = np.argsort(x_wall[lo])
            plt.plot(x_wall[lo][o], Cp_true[lo][o], label="CFD Cp (lower)")
            plt.plot(x_wall[lo][o], Cp_pred[lo][o], label=f"{mode} Cp (lower)")

        plt.gca().invert_yaxis()
        plt.xlabel("x [m]")
        plt.ylabel("Cp")
        plt.title(f"Cp comparison: {name} ({mode})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_path, "cp_compare.png"), dpi=300)
        plt.close()

        # re-save metrics with Cp included
        with open(os.path.join(out_path, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    print(f"[DONE] {name} {mode} | epochs={epochs_ran} train_time={train_time:.1f}s | rmse_speed={metrics['rmse_speed']:.4g}")
    print(f"       outputs -> {out_path}")

    return metrics


# ============================================================
# Main: run all cases + modes, write summary CSV/JSON
# ============================================================
def main():
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    all_metrics: List[Dict] = []
    for case in CASES:
        for mode in MODES_TO_RUN:
            # Optional safety: PHYS-only tends to be painful on CPU
            if mode == "PHYS" and DEVICE == "cpu":
                print("[SKIP] PHYS mode on CPU is usually too slow/unstable. Remove this skip if you want.")
                continue

            try:
                m = run_one(case, mode)
                all_metrics.append(m)
            except Exception as e:
                print(f"[ERROR] {case[0]} {mode} failed: {e}")
                # still record failure so your summary shows it happened
                all_metrics.append({
                    "case": case[0], "mode": mode, "error": str(e)
                })

    # Save summary JSON
    summary_json = os.path.join(OUT_DIR, "summary_metrics.json")
    with open(summary_json, "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Save summary CSV (only rows that have rmse_speed)
    rows = []
    for m in all_metrics:
        if "rmse_speed" in m:
            rows.append(m)
    if rows:
        df = pd.DataFrame(rows)
        summary_csv = os.path.join(OUT_DIR, "summary_metrics.csv")
        df.to_csv(summary_csv, index=False)
        print(f"\nWrote: {summary_json}")
        print(f"Wrote: {summary_csv}")

        # Print quick ranking
        df2 = df.sort_values("rmse_speed")
        print("\nTop by rmse_speed (lower is better):")
        for _, r in df2.head(8).iterrows():
            print(f"{r['case']:>12}  {r['mode']:<9}  rmse_speed={r['rmse_speed']:.4g}  train_s={r['train_time_s']:.1f}")
    else:
        print(f"\nWrote: {summary_json}")
        print("No successful runs to write CSV.")


if __name__ == "__main__":
    main()
    
    
    
#train MLP
#make the space along the airfoil smaller
#use one learning model