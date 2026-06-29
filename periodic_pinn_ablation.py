# -*- coding: utf-8 -*-
"""
periodic_pinn_ablation.py

Optimised periodic-benchmark workflow for three model types:
- DATA
- BC_DATA
- PHYS_DATA
"""

import os, json, math, time
from typing import List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt

try:
    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
torch.set_default_dtype(DTYPE)

OUT_DIR = "runs_periodic_v3"
SEED = 42

CASES = [
    ("periodic_coarse", "periodic_base_full_coarse.csv"),
    ("periodic_medium", "periodic_base_full.csv"),
    ("periodic_fine", "periodic_base_full_fine.csv"),
]
MODES = ["DATA", "BC_DATA", "PHYS_DATA"]

RHO = 998.2
MU = 0.001003
CP = 4182.0
K_TH = 0.6
L_CHAR = 0.01

EPOCHS = 2600
VAL_FRAC = 0.15
PATIENCE = 320
MIN_DELTA = 1e-7
SAVE_EVERY = 200
N_BC = 1400
N_COL = 3500
N_COL_POOL = 60000
BATCH_DATA = 4096
BATCH_BC = 1024

WIDTH = 224
DEPTH = 7
FOURIER_BANDS = 8
MAX_GRAD_NORM = 1.0

BC_RAMP_EPOCHS = 900
PDE_START = 900
PDE_RAMP = 1200

CFG = {
    "DATA": {
        "n_data": 24000, "lr": 2.5e-4,
        "w_data": 3.8, "w_periodic": 0.0, "w_sym": 0.0, "w_pde": 0.0,
        "w_press": 0.25, "w_temp": 0.45, "w_energy": 0.0,
    },
    "BC_DATA": {
        "n_data": 22000, "lr": 2.0e-4,
        "w_data": 3.6, "w_periodic": 0.12, "w_sym": 0.06, "w_pde": 0.0,
        "w_press": 0.25, "w_temp": 0.45, "w_energy": 0.0,
    },
    "PHYS_DATA": {
        "n_data": 18000, "lr": 1.6e-4,
        "w_data": 3.4, "w_periodic": 0.08, "w_sym": 0.04, "w_pde": 0.012,
        "w_press": 0.22, "w_temp": 0.35, "w_energy": 0.20,
    },
}

def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def to_t(x):
    return torch.tensor(x, dtype=DTYPE, device=DEVICE)

def sample_df(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n >= len(df):
        return df.copy()
    return df.sample(n=n, random_state=seed, replace=False).copy()

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed")]
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns={
        "x-coordinate":"x","y-coordinate":"y",
        "x velocity":"u","y velocity":"v",
        "static pressure":"p","pressure":"p",
        "static temperature":"t","temperature":"t",
    })
    need = ["x","y","u","v","p","t"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns: {miss}")
    df = df[need].replace([np.inf, -np.inf], np.nan).dropna()
    return df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

def scale_arr(x, xmin, xmax):
    return 2.0*(x-xmin)/(xmax-xmin+1e-12)-1.0

def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))
def mae(a,b): return float(np.mean(np.abs(a-b)))
def acc_pct(a,b):
    return float(np.clip(100.0*(1.0 - np.sqrt(np.mean((a-b)**2))/(np.sqrt(np.mean(a**2))+1e-12)), 0.0, 100.0))

def split_train_val(df: pd.DataFrame, val_frac: float = VAL_FRAC):
    n_val = max(1, int(len(df) * val_frac))
    return df.iloc[n_val:].copy(), df.iloc[:n_val].copy()

def save_loss_plot(train_losses, val_losses, path):
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def save_scatter(y_true, y_pred, path, title, xlabel, ylabel):
    plt.figure()
    plt.scatter(y_true, y_pred, s=2)
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([mn,mx],[mn,mx],"k--",lw=1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def save_contour(
    X, Y, Z, path, title, cbar,
    vmin=None, vmax=None,
    cmap="viridis",
    mask_nan=True
):
    plt.figure(figsize=(10, 4), dpi=150)
    Z_plot = np.array(Z, copy=True)
    if mask_nan:
        Z_plot = np.ma.masked_invalid(Z_plot)

    cf = plt.contourf(X, Y, Z_plot, levels=100, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar_obj = plt.colorbar(cf)
    cbar_obj.set_label(cbar)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close()


class FourierFeatures(nn.Module):
    def __init__(self, bands=FOURIER_BANDS):
        super().__init__()
        self.register_buffer("freqs", (2.0**torch.arange(bands).float())*math.pi)
    def forward(self, x):
        fx = x[:,0:1]*self.freqs[None,:]
        fy = x[:,1:2]*self.freqs[None,:]
        return torch.cat([torch.sin(fx),torch.cos(fx),torch.sin(fy),torch.cos(fy)], dim=1)

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.ff = FourierFeatures()
        in_dim = 2 + 4*FOURIER_BANDS
        layers = [nn.Linear(in_dim, WIDTH), nn.SiLU()]
        for _ in range(DEPTH):
            layers += [nn.Linear(WIDTH, WIDTH), nn.SiLU()]
        layers += [nn.Linear(WIDTH, 4)]
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.net(torch.cat([x, self.ff(x)], dim=1))

def grad(y, x):
    return autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True, only_inputs=True)[0]

def extract_boundaries(df, tol_scale=2.5e-3):
    x_min, x_max = float(df["x"].min()), float(df["x"].max())
    y_min, y_max = float(df["y"].min()), float(df["y"].max())
    tol_x = max(1e-12, tol_scale*(x_max-x_min))
    tol_y = max(1e-12, tol_scale*(y_max-y_min))
    left = df[np.abs(df["x"]-x_min) <= tol_x].copy()
    right = df[np.abs(df["x"]-x_max) <= tol_x].copy()
    bottom = df[np.abs(df["y"]-y_min) <= tol_y].copy()
    top = df[np.abs(df["y"]-y_max) <= tol_y].copy()
    return left, right, bottom, top

def pair_periodic_by_y(left, right, n_pairs=N_BC, seed=SEED):
    if len(left)==0 or len(right)==0:
        return left.iloc[:0].copy(), right.iloc[:0].copy()
    if SCIPY_OK:
        tree = cKDTree(right[["y"]].values.astype(np.float64))
        left_sub = left.sample(min(n_pairs, len(left)), random_state=seed).copy().reset_index(drop=True)
        _, idx = tree.query(left_sub[["y"]].values.astype(np.float64), k=1)
        right_sub = right.iloc[idx].copy().reset_index(drop=True)
        return left_sub, right_sub
    left_sub = left.sample(min(n_pairs, len(left)), random_state=seed).sort_values("y").reset_index(drop=True)
    right_s = right.sort_values("y").reset_index(drop=True)
    idx = np.linspace(0, len(right_s)-1, num=len(left_sub)).astype(int)
    return left_sub, right_s.iloc[idx].reset_index(drop=True)

def pde_residuals(model, X, y_mean, y_std, u_scale, p_scale, t_scale, t_offset, sx, sy):
    X.requires_grad_(True)
    out_z = model(X)
    out = out_z*y_std + y_mean
    u = out[:,0:1]*u_scale
    v = out[:,1:2]*u_scale
    p = out[:,2:3]*p_scale
    T = out[:,3:4]*t_scale + t_offset
    gu, gv, gp, gT = grad(u,X), grad(v,X), grad(p,X), grad(T,X)
    u_x, u_y = gu[:,0:1]*sx, gu[:,1:2]*sy
    v_x, v_y = gv[:,0:1]*sx, gv[:,1:2]*sy
    p_x, p_y = gp[:,0:1]*sx, gp[:,1:2]*sy
    T_x, T_y = gT[:,0:1]*sx, gT[:,1:2]*sy
    u_xx = grad(gu[:,0:1], X)[:,0:1]*(sx**2)
    u_yy = grad(gu[:,1:2], X)[:,1:2]*(sy**2)
    v_xx = grad(gv[:,0:1], X)[:,0:1]*(sx**2)
    v_yy = grad(gv[:,1:2], X)[:,1:2]*(sy**2)
    T_xx = grad(gT[:,0:1], X)[:,0:1]*(sx**2)
    T_yy = grad(gT[:,1:2], X)[:,1:2]*(sy**2)
    nu = MU/RHO
    alpha = K_TH/(RHO*CP)
    r_c = u_x + v_y
    r_u = u*u_x + v*u_y + (1.0/RHO)*p_x - nu*(u_xx + u_yy)
    r_v = u*v_x + v*v_y + (1.0/RHO)*p_y - nu*(v_xx + v_yy)
    r_T = u*T_x + v*T_y - alpha*(T_xx + T_yy)
    return r_c, r_u, r_v, r_T

def warmstart(model, case_name, mode):
    if mode not in ["BC_DATA","PHYS_DATA"]:
        return
    p = os.path.join(OUT_DIR, f"{case_name}_DATA", "model_best.pt")
    if os.path.exists(p):
        try:
            model.load_state_dict(torch.load(p, map_location=DEVICE))
            print(f"[WARMSTART] Loaded DATA weights for {case_name}->{mode}")
        except Exception as e:
            print(f"[WARMSTART] Failed: {e}")

def run_one(case_name, csv_file, mode):
    cfg = CFG[mode]
    out_path = os.path.join(OUT_DIR, f"{case_name}_{mode}")
    os.makedirs(out_path, exist_ok=True)

    df = clean_df(pd.read_csv(csv_file))
    x_min, x_max = float(df["x"].min()), float(df["x"].max())
    y_min, y_max = float(df["y"].min()), float(df["y"].max())
    sx = 2.0/(x_max-x_min+1e-12)
    sy = 2.0/(y_max-y_min+1e-12)

    def scale_xy(dsub):
        xs = scale_arr(dsub["x"].values.astype(np.float64), x_min, x_max)
        ys = scale_arr(dsub["y"].values.astype(np.float64), y_min, y_max)
        return to_t(np.stack([xs,ys], axis=1))

    speed = np.sqrt(df["u"].values**2 + df["v"].values**2)
    u_scale = max(float(np.max(speed)), 1e-6)
    p_offset = float(df["p"].mean())
    p_scale = max(float(df["p"].std()), 1e-6)
    t_offset = float(df["t"].mean())
    t_scale = max(float(df["t"].std()), 1e-6)

    df = df.copy()
    df["u_nd"] = df["u"]/u_scale
    df["v_nd"] = df["v"]/u_scale
    df["p_nd"] = (df["p"]-p_offset)/p_scale
    df["t_nd"] = (df["t"]-t_offset)/t_scale

    df_train, df_val = split_train_val(df, VAL_FRAC)
    df_train = sample_df(df_train, min(cfg["n_data"], len(df_train)), seed=SEED)

    X_train = scale_xy(df_train[["x","y"]])
    Y_train = to_t(df_train[["u_nd","v_nd","p_nd","t_nd"]].values.astype(np.float64))
    X_val = scale_xy(df_val[["x","y"]])
    Y_val = to_t(df_val[["u_nd","v_nd","p_nd","t_nd"]].values.astype(np.float64))

    y_mean = Y_train.mean(dim=0, keepdim=True)
    y_std = Y_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    Y_train_n = (Y_train - y_mean)/y_std
    Y_val_n = (Y_val - y_mean)/y_std

    df_pool = sample_df(df, min(N_COL_POOL, len(df)), seed=SEED+10)
    X_pool = scale_xy(df_pool[["x","y"]])

    left, right, bottom, top = extract_boundaries(df)
    left, right = pair_periodic_by_y(left, right, n_pairs=N_BC, seed=SEED)
    bottom = sample_df(bottom, min(N_BC,len(bottom)), seed=SEED) if len(bottom) else bottom
    top = sample_df(top, min(N_BC,len(top)), seed=SEED) if len(top) else top

    X_left = scale_xy(left[["x","y"]]) if len(left) else None
    X_right = scale_xy(right[["x","y"]]) if len(right) else None
    X_bottom = scale_xy(bottom[["x","y"]]) if len(bottom) else None
    X_top = scale_xy(top[["x","y"]]) if len(top) else None

    model = PINN().to(DEVICE)
    warmstart(model, case_name, mode)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=2e-6)
    scaler = GradScaler('cuda', enabled=(DEVICE == 'cuda'))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.6, patience=120, min_lr=1e-6)

    train_losses, val_losses = [], []
    best_val, best_epoch, no_improve = float("inf"), 0, 0
    t0 = time.time()

    with open(os.path.join(out_path, "run_config.json"), "w") as f:
        json.dump({
            "case": case_name, "mode": mode, "csv": csv_file, "device": DEVICE,
            "epochs": EPOCHS, "u_scale": u_scale, "p_offset": p_offset, "p_scale": p_scale,
            "t_offset": t_offset, "t_scale": t_scale, "cfg": cfg
        }, f, indent=2)

    min_epochs_before_stop = 900 if mode == "DATA" else 1200

    for ep in range(1, EPOCHS+1):
        model.train()
        opt.zero_grad(set_to_none=True)
        bc_eff = min(1.0, ep/(BC_RAMP_EPOCHS+1e-12))

        with autocast("cuda", enabled=(DEVICE == "cuda")):
            idx = torch.randperm(X_train.shape[0], device=DEVICE)[:min(BATCH_DATA, X_train.shape[0])]
            Xd, Yd, Yd_n = X_train[idx], Y_train[idx], Y_train_n[idx]

            pred_z = model(Xd)
            pred_y = pred_z*y_std + y_mean

            l_data_uvpt = F.smooth_l1_loss(pred_z, Yd_n, beta=0.25)
            uv_true, uv_pred = Yd[:,:2], pred_y[:,:2]
            sp_true = torch.sqrt((uv_true**2).sum(dim=1, keepdim=True)+1e-12)
            sp_pred = torch.sqrt((uv_pred**2).sum(dim=1, keepdim=True)+1e-12)
            l_speed = F.smooth_l1_loss(sp_pred, sp_true, beta=0.25)
            l_press = F.smooth_l1_loss(pred_y[:,2:3], Yd[:,2:3], beta=0.25)
            l_temp = F.smooth_l1_loss(pred_y[:,3:4], Yd[:,3:4], beta=0.25)
            l_data = l_data_uvpt + 0.30*l_speed + cfg["w_press"]*l_press + cfg["w_temp"]*l_temp
            total_loss = cfg["w_data"]*l_data

            if mode in ["BC_DATA","PHYS_DATA"]:
                if X_left is not None and X_right is not None and len(X_left)>0 and len(X_right)>0:
                    idxb = torch.randperm(X_left.shape[0], device=DEVICE)[:min(BATCH_BC, X_left.shape[0])]
                    pred_l = model(X_left[idxb]); pred_r = model(X_right[idxb])
                    l_periodic = F.mse_loss(pred_l, pred_r)
                    total_loss = total_loss + bc_eff*cfg["w_periodic"]*l_periodic
                else:
                    l_periodic = torch.tensor(0.0, device=DEVICE)

                l_sym = torch.tensor(0.0, device=DEVICE)
                for Xsym in [X_bottom, X_top]:
                    if Xsym is None or len(Xsym)==0:
                        continue
                    idxs = torch.randperm(Xsym.shape[0], device=DEVICE)[:min(BATCH_BC, Xsym.shape[0])]
                    XYs = Xsym[idxs].clone().detach().requires_grad_(True)
                    out_s = model(XYs)*y_std + y_mean
                    u_s, v_s, p_s, t_s = out_s[:,0:1], out_s[:,1:2], out_s[:,2:3], out_s[:,3:4]
                    gu, gp, gT = grad(u_s, XYs), grad(p_s, XYs), grad(t_s, XYs)
                    du_dy, dp_dy, dT_dy = gu[:,1:2]*sy, gp[:,1:2]*sy, gT[:,1:2]*sy
                    l_sym = l_sym + v_s.pow(2).mean() + du_dy.pow(2).mean() + 0.25*dp_dy.pow(2).mean() + 0.25*dT_dy.pow(2).mean()
                total_loss = total_loss + bc_eff*cfg["w_sym"]*l_sym
            else:
                l_periodic = torch.tensor(0.0, device=DEVICE)
                l_sym = torch.tensor(0.0, device=DEVICE)

            if mode == "PHYS_DATA" and ep > PDE_START:
                if X_pool.shape[0] <= N_COL:
                    Xc = X_pool
                else:
                    idc = torch.randperm(X_pool.shape[0], device=DEVICE)[:N_COL]
                    Xc = X_pool[idc]
                rc, ru, rv, rT = pde_residuals(model, Xc, y_mean, y_std, u_scale, p_scale, t_scale, t_offset, sx, sy)
                l_pde = rc.pow(2).mean() + ru.pow(2).mean() + rv.pow(2).mean() + cfg["w_energy"]*rT.pow(2).mean()
                w_pde_eff = cfg["w_pde"]*min(1.0, (ep-PDE_START)/(PDE_RAMP+1e-12))
                total_loss = total_loss + w_pde_eff*l_pde
            else:
                l_pde = torch.tensor(0.0, device=DEVICE)
                w_pde_eff = 0.0

        scaler.scale(total_loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        scaler.step(opt)
        scaler.update()

        train_loss = float(total_loss.detach().item())
        train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            val_pred_z = model(X_val)
            val_loss = float(F.smooth_l1_loss(val_pred_z, Y_val_n, beta=0.25).item())
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if ep == 1 or ep % 200 == 0:
            print(f"[{case_name} | {mode}] ep={ep:4d} train={train_loss:.3e} val={val_loss:.3e} "
                  f"data={float(l_data.item()):.2e} periodic={float(l_periodic.item()):.2e} "
                  f"sym={float(l_sym.item()):.2e} pde={float(l_pde.item()):.2e} (w={float(w_pde_eff):.2e})")

        if val_loss < best_val - MIN_DELTA:
            best_val, best_epoch, no_improve = val_loss, ep, 0
            torch.save(model.state_dict(), os.path.join(out_path, "model_best.pt"))
        else:
            no_improve += 1

        if ep % SAVE_EVERY == 0:
            np.savetxt(os.path.join(out_path, "loss_train.txt"), np.array(train_losses))
            np.savetxt(os.path.join(out_path, "loss_val.txt"), np.array(val_losses))
            save_loss_plot(train_losses, val_losses, os.path.join(out_path, "loss_train_val.png"))
            with open(os.path.join(out_path, "heartbeat.json"), "w") as f:
                json.dump({"epoch": ep, "best_val": best_val, "best_epoch": best_epoch}, f, indent=2)

        if ep >= min_epochs_before_stop and no_improve >= PATIENCE:
            print(f"[EARLY STOP] {case_name} {mode} at ep={ep} (best ep={best_epoch}, best val={best_val:.3e})")
            break

    best_path = os.path.join(out_path, "model_best.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))

    return model, df, y_mean, y_std, u_scale, p_offset, p_scale, t_offset, t_scale, x_min, x_max, y_min, y_max, train_losses, val_losses, best_val, best_epoch, time.time()-t0, out_path

def evaluate_and_save(model, df, y_mean, y_std, u_scale, p_offset, p_scale, t_offset, t_scale, x_min, x_max, y_min, y_max, train_losses, val_losses, best_val, best_epoch, train_time, out_path):
    def scale_xy(dsub):
        xs = scale_arr(dsub["x"].values.astype(np.float64), x_min, x_max)
        ys = scale_arr(dsub["y"].values.astype(np.float64), y_min, y_max)
        return to_t(np.stack([xs, ys], axis=1))

    X_all = scale_xy(df[["x","y"]])
    model.eval()
    with torch.no_grad():
        pred_nd = (model(X_all)*y_std + y_mean).cpu().numpy()

    u_pred = pred_nd[:,0]*u_scale
    v_pred = pred_nd[:,1]*u_scale
    p_pred = pred_nd[:,2]*p_scale + p_offset
    T_pred = pred_nd[:,3]*t_scale + t_offset

    u_true, v_true, p_true, T_true = df["u"].values, df["v"].values, df["p"].values, df["t"].values
    sp_true = np.sqrt(u_true**2 + v_true**2)
    sp_pred = np.sqrt(u_pred**2 + v_pred**2)

    metrics = {
        "rmse_u": rmse(u_true,u_pred), "rmse_v": rmse(v_true,v_pred), "rmse_p": rmse(p_true,p_pred), "rmse_T": rmse(T_true,T_pred), "rmse_speed": rmse(sp_true,sp_pred),
        "mae_u": mae(u_true,u_pred), "mae_v": mae(v_true,v_pred), "mae_p": mae(p_true,p_pred), "mae_T": mae(T_true,T_pred), "mae_speed": mae(sp_true,sp_pred),
        "acc_u": acc_pct(u_true,u_pred), "acc_v": acc_pct(v_true,v_pred), "acc_p": acc_pct(p_true,p_pred), "acc_T": acc_pct(T_true,T_pred), "acc_speed": acc_pct(sp_true,sp_pred),
        "train_time_s": float(train_time), "best_val": float(best_val), "best_epoch": int(best_epoch),
    }
    with open(os.path.join(out_path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame({
        "x": df["x"], "y": df["y"],
        "u_true": u_true, "u_pred": u_pred,
        "v_true": v_true, "v_pred": v_pred,
        "p_true": p_true, "p_pred": p_pred,
        "T_true": T_true, "T_pred": T_pred,
        "speed_true": sp_true, "speed_pred": sp_pred,
        "abs_err_u": np.abs(u_true-u_pred), "abs_err_v": np.abs(v_true-v_pred),
        "abs_err_p": np.abs(p_true-p_pred), "abs_err_T": np.abs(T_true-T_pred), "abs_err_speed": np.abs(sp_true-sp_pred),
    }).to_csv(os.path.join(out_path, "predictions_full_domain.csv"), index=False)

    save_scatter(sp_true, sp_pred, os.path.join(out_path, "scatter_speed.png"), "Speed scatter", "CFD speed [m/s]", "Pred speed [m/s]")
    save_scatter(T_true, T_pred, os.path.join(out_path, "scatter_temperature.png"), "Temperature scatter", "CFD T [K]", "Pred T [K]")
    save_loss_plot(train_losses, val_losses, os.path.join(out_path, "loss_train_val.png"))

    xg = np.linspace(x_min, x_max, 240)
    yg = np.linspace(y_min, y_max, 140)
    XG, YG = np.meshgrid(xg, yg)
    XY = to_t(np.stack([scale_arr(XG.flatten(), x_min, x_max), scale_arr(YG.flatten(), y_min, y_max)], axis=1))
    with torch.no_grad():
        predg = (model(XY)*y_std + y_mean).cpu().numpy()

    ug = (predg[:,0]*u_scale).reshape(YG.shape)
    vg = (predg[:,1]*u_scale).reshape(YG.shape)
    pg = (predg[:,2]*p_scale + p_offset).reshape(YG.shape)
    Tg = (predg[:,3]*t_scale + t_offset).reshape(YG.shape)
    spg = np.sqrt(ug**2 + vg**2)

    if SCIPY_OK:
        pts = np.column_stack((df["x"].values, df["y"].values))
        sp_true_grid = griddata(pts, sp_true, (XG, YG), method="linear")
        p_true_grid = griddata(pts, p_true, (XG, YG), method="linear")
        T_true_grid = griddata(pts, T_true, (XG, YG), method="linear")

        speed_vmin = float(np.nanmin(sp_true_grid))
        speed_vmax = float(np.nanmax(sp_true_grid))
        pressure_vmin = float(np.nanmin(p_true_grid))
        pressure_vmax = float(np.nanmax(p_true_grid))
        temp_vmin = float(np.nanmin(T_true_grid))
        temp_vmax = float(np.nanmax(T_true_grid))

        speed_abs_err = np.abs(sp_true_grid - spg)
        pressure_abs_err = np.abs(p_true_grid - pg)
        temp_abs_err = np.abs(T_true_grid - Tg)

        save_contour(
            XG, YG, spg,
            os.path.join(out_path, "contour_speed.png"),
            "Predicted Speed", "Speed",
            vmin=speed_vmin, vmax=speed_vmax, cmap="viridis", mask_nan=True
        )
        save_contour(
            XG, YG, pg,
            os.path.join(out_path, "contour_pressure.png"),
            "Predicted Pressure", "Pressure",
            vmin=pressure_vmin, vmax=pressure_vmax, cmap="viridis", mask_nan=True
        )
        save_contour(
            XG, YG, Tg,
            os.path.join(out_path, "contour_temperature.png"),
            "Predicted Temperature", "Temperature",
            vmin=temp_vmin, vmax=temp_vmax, cmap="viridis", mask_nan=True
        )

        save_contour(
            XG, YG, sp_true_grid,
            os.path.join(out_path, "contour_speed_TRUE.png"),
            "CFD TRUE Speed", "Speed",
            vmin=speed_vmin, vmax=speed_vmax, cmap="viridis", mask_nan=True
        )
        save_contour(
            XG, YG, p_true_grid,
            os.path.join(out_path, "contour_pressure_TRUE.png"),
            "CFD TRUE Pressure", "Pressure",
            vmin=pressure_vmin, vmax=pressure_vmax, cmap="viridis", mask_nan=True
        )
        save_contour(
            XG, YG, T_true_grid,
            os.path.join(out_path, "contour_temperature_TRUE.png"),
            "CFD TRUE Temperature", "Temperature",
            vmin=temp_vmin, vmax=temp_vmax, cmap="viridis", mask_nan=True
        )

        save_contour(
            XG, YG, speed_abs_err,
            os.path.join(out_path, "contour_speed_abs_error.png"),
            "Absolute Speed Error", "|error|",
            vmin=0.0, vmax=float(np.nanmax(speed_abs_err)), cmap="magma", mask_nan=True
        )
        save_contour(
            XG, YG, pressure_abs_err,
            os.path.join(out_path, "contour_pressure_abs_error.png"),
            "Absolute Pressure Error", "|error|",
            vmin=0.0, vmax=float(np.nanmax(pressure_abs_err)), cmap="magma", mask_nan=True
        )
        save_contour(
            XG, YG, temp_abs_err,
            os.path.join(out_path, "contour_temperature_abs_error.png"),
            "Absolute Temperature Error", "|error|",
            vmin=0.0, vmax=float(np.nanmax(temp_abs_err)), cmap="magma", mask_nan=True
        )
    else:
        save_contour(XG, YG, spg, os.path.join(out_path, "contour_speed.png"), "Predicted Speed", "Speed")
        save_contour(XG, YG, pg, os.path.join(out_path, "contour_pressure.png"), "Predicted Pressure", "Pressure")
        save_contour(XG, YG, Tg, os.path.join(out_path, "contour_temperature.png"), "Predicted Temperature", "Temperature")

    print(f"[SAVED] Results -> {out_path}")
    return metrics

def main():
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    all_metrics = []
    for case_name, csv_file in CASES:
        for mode in MODES:
            try:
                out = run_one(case_name, csv_file, mode)
                metrics = evaluate_and_save(*out)
                metrics["case"] = case_name
                metrics["mode"] = mode
                all_metrics.append(metrics)
            except Exception as e:
                print(f"ERROR: {case_name} {mode} -> {e}")
                all_metrics.append({"case": case_name, "mode": mode, "error": str(e)})
    with open(os.path.join(OUT_DIR, "summary_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    pd.DataFrame(all_metrics).to_csv(os.path.join(OUT_DIR, "summary_metrics.csv"), index=False)
    print("ALL DONE")

if __name__ == "__main__":
    main()
