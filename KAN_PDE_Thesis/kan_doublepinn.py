import math
import copy
import logging
import numpy as np
import time
import json
from dataclasses import dataclass, asdict
from typing import Tuple
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.autograd as autograd
import matplotlib.pyplot as plt

# =============== efficient-KAN ============================
# Assicurati di avere:
# pip install "git+https://github.com/Blealtan/efficient-kan.git"
from efficient_kan import KAN

# =============== LOGGING ============================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# =============== DEVICE ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

if device.type == 'cuda':
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# =============== CONFIGURAZIONE CENTRALIZZATA ==================
@dataclass
class Config:
    """Configurazione centralizzata per tutti i parametri del modello."""
    # Parametri fisici Fokker-Planck
    a: float = 0.3
    b: float = 0.5
    sigma: float = 0.5
    x_min: float = -2.2
    x_max: float = 2.2
    
    # Architettura reti
    layers_D: Tuple[int, ...] = (1, 2, 2, 1)
    layers_B: Tuple[int, ...] = (1, 2, 2, 1)
    
    # Training
    N_int: int = 128
    max_epochs: int = 1200
    phase0_epochs: int = 100  # pre-training BC
    phase1_epochs: int = 150  # curriculum uniforme
    lr: float = 1e-3
    grad_clip_max_norm: float = 1.0
    
    # Normalizzazione e sampling
    N_norm: int = 400
    phase2_sampling_min_factor: int = 4
    phase2_sampling_max_factor: int = 16
    
    # ALM (Augmented Lagrangian Method)
    rho_alm_init: float = 10.0
    rho_max: float = 865.0
    eta_rho: float = 1.5
    Lambda_clip: float = 10.0
    k_update_lambda: int = 10
    h_update_rho: int = 80
    
    # Pesi loss
    w_pde: float = 1.0
    w_norm: float = 0.5
    w_pos: float = 0.1
    
    # Schedulers
    gamma_min: float = 0.0
    gamma_max: float = 0.1
    w_bc_min: float = 0.2
    w_bc_max: float = 1.0
    
    # Role loss
    alpha_int: float = 1e-3
    alpha_bd: float = 1e-3
    tau: float = 0.6
    
    # Multi-seed
    seeds: Tuple[int, ...] = (42, 44, 46, 48, 50)
    
    @property
    def D(self) -> float:
        """Coefficiente di diffusione."""
        return self.sigma**2 / 2.0
    
    @property
    def phase2_epochs(self) -> int:
        """Epoche rimanenti per Phase 2."""
        return self.max_epochs - self.phase0_epochs - self.phase1_epochs


config = Config()
logger.info(f"Config: max_epochs={config.max_epochs}, N_int={config.N_int}, lr={config.lr}")


def mu(x: torch.Tensor) -> torch.Tensor:
    """Drift: a x - b x^3."""
    return config.a * x - config.b * x**3


# soluzione esatta non normalizzata
def rho_unnorm(x: torch.Tensor) -> torch.Tensor:
    z = (1.0 / (2.0 * config.sigma**2)) * (2.0 * config.a * x**2 - config.b * x**4)
    return torch.exp(z)


def rho_exact(x: torch.Tensor) -> torch.Tensor:
    """Densità esatta normalizzata su [x_min, x_max]."""
    with torch.no_grad():
        rho_u = rho_unnorm(x)
        x1d = x.view(-1)
        r1d = rho_u.view(-1)
        Z = torch.trapz(r1d, x1d)
        return rho_u / (Z + 1e-8)


# =============== TENSORI PRE-ALLOCATI ==================
x_norm = torch.linspace(config.x_min, config.x_max, config.N_norm).view(-1, 1).to(device)
x_bc = torch.tensor([[config.x_min], [config.x_max]], dtype=torch.float32).to(device)
g_bc = torch.zeros_like(x_bc)  # rho = 0 agli estremi


# =============== LOSS COMPONENTS ===========================
def fp_residual(
    netD, netB, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Residuo FP stazionario:
    0 = -((mu(x) * rho(x))_x) + D * rho_xx
    """
    x = x.clone().detach().to(device)
    x.requires_grad_(True)

    rhoD = netD(x)
    rhoB = netB(x)
    rho = rhoD + rhoB

    ones = torch.ones_like(rho)

    # (mu * rho)_x
    mu_rho = mu(x) * rho
    d_mu_rho_dx = autograd.grad(
        mu_rho,
        x,
        grad_outputs=torch.ones_like(mu_rho),
        create_graph=True,
    )[0]

    # rho_x, rho_xx
    rho_x = autograd.grad(
        rho,
        x,
        grad_outputs=ones,
        create_graph=True,
    )[0]
    rho_xx = autograd.grad(
        rho_x,
        x,
        grad_outputs=torch.ones_like(rho_x),
        create_graph=True,
    )[0]

    res = -d_mu_rho_dx + config.D * rho_xx
    return res, rhoD, rhoB, rho


def boundary_alm_loss(
    netD,
    netB,
    lambda_bc: torch.Tensor,
    rho_alm: float,
    w_bc: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Loss di ALM per i vincoli di bordo."""
    rhoD_bc = netD(x_bc)
    rhoB_bc = netB(x_bc)
    rho_bc = rhoD_bc + rhoB_bc
    c = rho_bc - g_bc  # violazione BC

    # w_bc(t) modula il peso dei vincoli al bordo
    L_alm = w_bc * torch.mean(lambda_bc * c + 0.5 * rho_alm * c**2)
    return L_alm, c.detach()


def role_loss(
    x: torch.Tensor,
    rhoD: torch.Tensor,
    rhoB: torch.Tensor,
) -> torch.Tensor:
    """
    L_role = alpha_int * E[w_bd * |rhoD|^2] + alpha_bd * E[w_in * |rhoB|^2]
    con w_bd = exp(-d/tau), d distanza minima dal bordo.
    """
    d_left = (x - config.x_min).abs()
    d_right = (config.x_max - x).abs()
    d = torch.minimum(d_left, d_right)

    w_bd = torch.exp(-d / config.tau)
    w_in = 1.0 - w_bd

    L = config.alpha_int * torch.mean(w_bd * (rhoD**2)) \
        + config.alpha_bd * torch.mean(w_in * (rhoB**2))
    return L


def gamma_schedule(
    epoch_idx: int,
    T: int,
    gamma_min: float = 0.0,
    gamma_max: float = 0.1,
) -> float:
    """Scheduler cosinusoidale per gamma in [gamma_min, gamma_max]."""
    t = float(epoch_idx)
    return gamma_min + 0.5 * (gamma_max - gamma_min) * (
        1.0 + math.cos(math.pi * t / T)
    )


def w_bc_schedule(
    epoch_idx: int,
    T: int,
    w_min: float = 0.2,
    w_max: float = 1.0,
) -> float:
    """
    Peso dei vincoli di bordo che scende linearmente da w_max a w_min
    nell'arco di T epoche.
    """
    t = float(epoch_idx)
    frac = min(max(t / T, 0.0), 1.0)
    return w_max - (w_max - w_min) * frac


def norm_and_pos_loss(netD, netB) -> tuple[torch.Tensor, torch.Tensor]:
    """Penalizza negatività e normalizzazione non unitaria."""
    rho_all = netD(x_norm) + netB(x_norm)

    # penalità negatività
    neg = torch.relu(-rho_all)
    L_pos = torch.mean(neg**2)

    rho_flat = rho_all.view(-1)
    x_flat = x_norm.view(-1)
    integral = torch.trapz(rho_flat, x_flat)
    L_norm = (integral - 1.0) ** 2
    return L_norm, L_pos


# =============== SAMPLING INTERNI ==========================
def sample_uniform(N: int) -> torch.Tensor:
    """Campionamento uniforme su [x_min, x_max]."""
    x = config.x_min + (config.x_max - config.x_min) * torch.rand(N, 1)
    return x.to(device)


def sample_interior_phase1(N: int) -> torch.Tensor:
    """Fase 1: collocazione uniforme."""
    return sample_uniform(N)


def sample_interior_phase2(N: int, netD, netB, epoch_ratio: float) -> torch.Tensor:
    """
    Residual-based sampling (Phase 2) con sampling adattivo:
    - genera N_cand >> N con distribuzione uniforme
    - valuta il residuo FP
    - seleziona i N punti con residuo più alto
    - N_cand aumenta progressivamente durante Phase 2
    
    Args:
        N: numero di punti da campionare
        netD, netB: reti neurali
        epoch_ratio: progresso in Phase 2, da 0.0 a 1.0
    """
    # Aumenta progressivamente il fattore di campionamento
    N_factor = int(
        config.phase2_sampling_min_factor + 
        (config.phase2_sampling_max_factor - config.phase2_sampling_min_factor) * epoch_ratio
    )
    N_cand = N_factor * N
    
    x_cand = sample_uniform(N_cand)
    # campiona N_cand punti uniformi nel dominio
    res_cand, _, _, _ = fp_residual(netD, netB, x_cand)
    r_val = (res_cand**2).detach().view(-1)

    # top-N punti per residuo
    _, idx = torch.topk(r_val, k=N, largest=True, sorted=False)
    x_sel = x_cand[idx]
    return x_sel


# =============== TRAINING PER UN SINGOLO SEED =============
def train_for_seed(seed: int):
    start_time = time.time()
    
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)

    # nuovi modelli per questo seed
    netD = KAN(list(config.layers_D)).to(device)
    netB = KAN(list(config.layers_B)).to(device)

    # Conta parametri
    params_D = sum(p.numel() for p in netD.parameters())
    params_B = sum(p.numel() for p in netB.parameters())
    total_params = params_D + params_B

    # ALM variables
    lambda_bc = torch.zeros_like(g_bc, device=device)
    rho_alm = config.rho_alm_init

    optimizer = torch.optim.Adam(
        list(netD.parameters()) + list(netB.parameters()),
        lr=config.lr,
    )

    logger.info(f"\n[Seed {seed}] Inizio training Dual-PINN efficient-KAN (epoche = {config.max_epochs})")
    logger.info(f"[Seed {seed}] Parametri: netD={params_D}, netB={params_B}, totale={total_params}")

    # ---- PHASE 0: pre-training BC ----
    for epoch in range(1, config.phase0_epochs + 1):
        optimizer.zero_grad()
        rhoD_bc = netD(x_bc)
        rhoB_bc = netB(x_bc)
        rho_bc = rhoD_bc + rhoB_bc
        L_bc0 = torch.mean((rho_bc - g_bc) ** 2)
        L_bc0.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            list(netD.parameters()) + list(netB.parameters()),
            max_norm=config.grad_clip_max_norm
        )
        
        optimizer.step()

        if epoch == 1 or epoch % 20 == 0 or epoch == config.phase0_epochs:
            logger.info(f"[Seed {seed} | Epoch {epoch:4d} | Phase 0] L_bc0={L_bc0.item():.3e}")

    # ---- PHASE 1 + PHASE 2 ----
    for epoch in range(config.phase0_epochs + 1, config.max_epochs + 1):
        # determina fase
        if epoch <= config.phase0_epochs + config.phase1_epochs:
            phase = 1
        else:
            phase = 2

        # sampling interno
        if phase == 1:
            x_int = sample_interior_phase1(config.N_int)
        else:
            # Calcola progresso in Phase 2 (0.0 all'inizio, 1.0 alla fine)
            epoch_in_phase2 = epoch - (config.phase0_epochs + config.phase1_epochs)
            epoch_ratio = min(epoch_in_phase2 / config.phase2_epochs, 1.0)
            x_int = sample_interior_phase2(config.N_int, netD, netB, epoch_ratio)

        optimizer.zero_grad()

        # PDE residual
        res, rhoD, rhoB, _ = fp_residual(netD, netB, x_int)
        L_pde = torch.mean(res**2)

        # ALM BC
        epoch_rel = epoch - (config.phase0_epochs + 1)
        w_bc = w_bc_schedule(epoch_rel, config.max_epochs - config.phase0_epochs)
        L_alm, c_bc = boundary_alm_loss(netD, netB, lambda_bc, rho_alm, w_bc)

        # normalizzazione + positività
        L_norm, L_pos = norm_and_pos_loss(netD, netB)

        # ruolo duale
        gamma = gamma_schedule(epoch - 1, config.max_epochs)
        L_role = role_loss(x_int, rhoD, rhoB)

        # Loss totale con pesi da config
        L_total = (
            config.w_pde * L_pde
            + L_alm
            + gamma * L_role
            + config.w_norm * L_norm
            + config.w_pos * L_pos
        )

        L_total.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            list(netD.parameters()) + list(netB.parameters()),
            max_norm=config.grad_clip_max_norm
        )
        
        optimizer.step()

        # update lambda (ALM)
        if epoch % config.k_update_lambda == 0:
            with torch.no_grad():
                lambda_bc[:] = torch.clamp(
                    lambda_bc + rho_alm * c_bc,
                    -config.Lambda_clip,
                    config.Lambda_clip,
                )

        # aumenta progressivamente rho_alm
        if epoch % config.h_update_rho == 0:
            rho_alm = min(rho_alm * config.eta_rho, config.rho_max)

        # logging
        if epoch == config.phase0_epochs + 1 or epoch % 50 == 0 or epoch == config.max_epochs:
            logger.info(
                f"[Seed {seed} | Epoch {epoch:4d} | Phase {phase}] "
                f"L_tot={L_total.item():.3e} "
                f"PDE={L_pde.item():.3e} "
                f"ALM={L_alm.item():.3e} "
                f"Role={L_role.item():.3e} "
                f"Norm={L_norm.item():.3e} "
                f"Pos={L_pos.item():.3e} "
                f"w_bc={w_bc:.3f} gamma={gamma:.3f} rho_alm={rho_alm:.1f}"
            )
        
        # Memory management per GPU
        if device.type == 'cuda' and epoch % 100 == 0:
            torch.cuda.empty_cache()

    # ===== VALUTAZIONE SU GRIGLIA =====
    x_eval = torch.linspace(config.x_min, config.x_max, 400).view(-1, 1).to(device)
    with torch.no_grad():
        rho_pred = netD(x_eval) + netB(x_eval)
        Z_pred = torch.trapz(rho_pred.view(-1), x_eval.view(-1))
        rho_pred_n = rho_pred / (Z_pred + 1e-8)
        rho_true = rho_exact(x_eval)

        diff = rho_pred_n - rho_true
        rel_L2 = torch.norm(diff) / torch.norm(rho_true)
        acc_L2 = 1.0 - rel_L2
        mae = torch.mean(torch.abs(diff))
        rmse = torch.sqrt(torch.mean(diff**2))

    # BC L2
    with torch.no_grad():
        rhoD_bc = netD(x_bc)
        rhoB_bc = netB(x_bc)
        rho_bc = rhoD_bc + rhoB_bc
        bc_l2 = torch.mean((rho_bc - g_bc) ** 2)

    # PDE residual L2 su griglia
    res_test, _, _, _ = fp_residual(netD, netB, x_eval)
    pde_l2 = torch.mean(res_test**2)

    training_time = time.time() - start_time
    
    metrics = {
        "RelL2": rel_L2.item(),
        "AccL2": acc_L2.item(),
        "MAE": mae.item(),
        "RMSE": rmse.item(),
        "BC_L2": bc_l2.item(),
        "PDE_L2": pde_l2.item(),
        "training_time": training_time,
        "params_D": params_D,
        "params_B": params_B,
        "total_params": total_params,
    }

    logger.info(f"\n[Seed {seed}] metrics: {metrics}")
    logger.info(f"[Seed {seed}] Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    # Memory cleanup
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return netD, netB, metrics


# =============== LOOP MULTI-SEED ==========================
logger.info(f"\n{'='*60}")
logger.info(f"INIZIO TRAINING MULTI-SEED: {config.seeds}")
logger.info(f"{'='*60}\n")

total_start_time = time.time()

all_metrics = []

best_relL2 = float("inf")
best_seed = None
best_stateD = None
best_stateB = None
best_metrics_best = None

for seed in config.seeds:
    netD, netB, metrics = train_for_seed(seed)
    all_metrics.append(metrics)

    # aggiorna "miglior modello"
    if metrics["RelL2"] < best_relL2:
        best_relL2 = metrics["RelL2"]
        best_seed = seed
        best_stateD = copy.deepcopy(netD.state_dict())
        best_stateB = copy.deepcopy(netB.state_dict())
        best_metrics_best = metrics.copy()

# =============== STATISTICHE MULTI-SEED ===================
total_time = time.time() - total_start_time

keys = ["RelL2", "AccL2", "MAE", "RMSE", "BC_L2", "PDE_L2", "training_time"]
vals = {k: np.array([m[k] for m in all_metrics]) for k in keys}

logger.info("\n===== STATISTICHE MULTI-SEED (media ± std) =====")
for k in keys:
    mean = vals[k].mean()
    std = vals[k].std()
    logger.info(f"{k:12s}: {mean: .3e} ± {std: .3e}")

logger.info(f"\nTempo totale: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

# =============== MODELLO MIGLIORE: RIPRISTINA & PLOT ======
logger.info(
    f"\nMiglior seed: {best_seed} "
    f"(RelL2 = {best_relL2:.3e}, "
    f"Accuracy L2 = {best_metrics_best['AccL2']*100:.2f}%)"
)

# ricrea reti e carica i pesi migliori
netD_best = KAN(list(config.layers_D)).to(device)
netB_best = KAN(list(config.layers_B)).to(device)
netD_best.load_state_dict(best_stateD)
netB_best.load_state_dict(best_stateB)

# valutazione finale best model
x_plot = torch.linspace(config.x_min, config.x_max, 400).view(-1, 1).to(device)
with torch.no_grad():
    rho_pred = netD_best(x_plot) + netB_best(x_plot)
    Z_pred = torch.trapz(rho_pred.view(-1), x_plot.view(-1))
    rho_pred_n = rho_pred / (Z_pred + 1e-8)
    rho_true = rho_exact(x_plot)

rel_L2_best = torch.norm(rho_pred_n - rho_true) / torch.norm(rho_true)
acc_L2_best = 1.0 - rel_L2_best

logger.info(f"\n[Best model] Relative L2 error = {rel_L2_best.item():.6e}")
logger.info(f"[Best model] Accuracy in L2 (%) = {100.0 * acc_L2_best.item():.4f}")

# Final memory cleanup
if device.type == 'cuda':
    torch.cuda.empty_cache()
    logger.info(f"GPU Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# =============== PLOT BEST MODEL ==========================
# Crea directory per i risultati
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plot_filename = output_dir / f"fokker_planck_best_seed{best_seed}_{timestamp}.png"

x_np = x_plot.cpu().numpy()
rho_true_np = rho_true.cpu().numpy()
rho_pred_np = rho_pred_n.cpu().numpy()

plt.figure(figsize=(8, 5))
plt.plot(x_np, rho_true_np, label="rho_exact (due picchi)", linewidth=2)
plt.plot(
    x_np,
    rho_pred_np,
    "--",
    label=f"Dual-PINN KAN (seed {best_seed})",
    linewidth=2,
)
plt.xlabel("x")
plt.ylabel("rho(x)")
plt.title("Fokker-Planck 1D (double-well): modello migliore (RelL2 minimo)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
logger.info(f"\nGrafico salvato in: {plot_filename}")
plt.close()

# =============== SALVA RISULTATI IN FILE JSON E TXT =======
results_json = {
    "timestamp": timestamp,
    "device": str(device),
    "config": {
        "max_epochs": config.max_epochs,
        "N_int": config.N_int,
        "lr": config.lr,
        "layers_D": list(config.layers_D),
        "layers_B": list(config.layers_B),
        "phase0_epochs": config.phase0_epochs,
        "phase1_epochs": config.phase1_epochs,
        "phase2_epochs": config.phase2_epochs,
        "grad_clip_max_norm": config.grad_clip_max_norm,
        "seeds": list(config.seeds),
    },
    "best_model": {
        "seed": best_seed,
        "metrics": best_metrics_best,
    },
    "all_seeds_metrics": all_metrics,
    "statistics": {
        k: {
            "mean": float(vals[k].mean()),
            "std": float(vals[k].std()),
            "min": float(vals[k].min()),
            "max": float(vals[k].max()),
        }
        for k in keys
    },
    "total_training_time_seconds": total_time,
    "total_training_time_minutes": total_time / 60.0,
}

json_filename = output_dir / f"results_{timestamp}.json"
with open(json_filename, 'w') as f:
    json.dump(results_json, f, indent=2)
logger.info(f"Risultati JSON salvati in: {json_filename}")

# TXT file (readable format)
txt_filename = output_dir / f"results_{timestamp}.txt"
with open(txt_filename, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("DUAL-PINN KAN TRAINING RESULTS - FOKKER-PLANCK 1D\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Timestamp: {timestamp}\n")
    f.write(f"Device: {device}\n\n")
    
    f.write("CONFIGURATION:\n")
    f.write("-" * 70 + "\n")
    f.write(f"  Total epochs: {config.max_epochs}\n")
    f.write(f"  Phase 0 (BC pre-training): {config.phase0_epochs}\n")
    f.write(f"  Phase 1 (uniform sampling): {config.phase1_epochs}\n")
    f.write(f"  Phase 2 (adaptive sampling): {config.phase2_epochs}\n")
    f.write(f"  Interior points: {config.N_int}\n")
    f.write(f"  Learning rate: {config.lr}\n")
    f.write(f"  Gradient clipping: {config.grad_clip_max_norm}\n")
    f.write(f"  netD architecture: {config.layers_D}\n")
    f.write(f"  netB architecture: {config.layers_B}\n")
    f.write(f"  Total parameters: {best_metrics_best['total_params']}\n")
    f.write(f"    - netD: {best_metrics_best['params_D']}\n")
    f.write(f"    - netB: {best_metrics_best['params_B']}\n")
    f.write(f"  Seeds: {config.seeds}\n\n")
    
    f.write("BEST MODEL:\n")
    f.write("-" * 70 + "\n")
    f.write(f"  Seed: {best_seed}\n")
    f.write(f"  Relative L2 Error: {best_metrics_best['RelL2']:.6e}\n")
    f.write(f"  Accuracy L2 (%): {best_metrics_best['AccL2']*100:.4f}%\n")
    f.write(f"  MAE: {best_metrics_best['MAE']:.6e}\n")
    f.write(f"  RMSE: {best_metrics_best['RMSE']:.6e}\n")
    f.write(f"  BC L2: {best_metrics_best['BC_L2']:.6e}\n")
    f.write(f"  PDE L2: {best_metrics_best['PDE_L2']:.6e}\n")
    f.write(f"  Training time: {best_metrics_best['training_time']:.2f}s ({best_metrics_best['training_time']/60:.2f}min)\n\n")
    
    f.write("MULTI-SEED STATISTICS (mean ± std):\n")
    f.write("-" * 70 + "\n")
    for k in keys:
        mean = vals[k].mean()
        std = vals[k].std()
        min_val = vals[k].min()
        max_val = vals[k].max()
        f.write(f"  {k:15s}: {mean:.6e} ± {std:.6e}  [min: {min_val:.6e}, max: {max_val:.6e}]\n")
    
    f.write(f"\nTIMING:\n")
    f.write("-" * 70 + "\n")
    f.write(f"  Total time: {total_time:.2f}s ({total_time/60:.2f}min)\n")
    f.write(f"  Average time per seed: {vals['training_time'].mean():.2f}s\n\n")
    
    f.write("RESULTS PER SEED:\n")
    f.write("-" * 70 + "\n")
    for i, (seed, metrics) in enumerate(zip(config.seeds, all_metrics)):
        f.write(f"\nSeed {seed}:\n")
        f.write(f"  RelL2: {metrics['RelL2']:.6e}, AccL2: {metrics['AccL2']*100:.4f}%\n")
        f.write(f"  MAE: {metrics['MAE']:.6e}, RMSE: {metrics['RMSE']:.6e}\n")
        f.write(f"  BC_L2: {metrics['BC_L2']:.6e}, PDE_L2: {metrics['PDE_L2']:.6e}\n")
        f.write(f"  Time: {metrics['training_time']:.2f}s\n")
    
    f.write("\n" + "=" * 70 + "\n")

logger.info(f"Risultati TXT salvati in: {txt_filename}")
logger.info(f"\n{'='*60}")
logger.info(f"COMPLETATO! Tutti i file salvati in: {output_dir}/")
logger.info(f"{'='*60}")
