#!/usr/bin/env python3
"""
pinn_baseline.py  — PINN baseline usando DeepXDE (PyTorch backend).
Resuelve las mismas 4 ecuaciones que PI-NSGA-II y guarda resultados
en el mismo formato CSV para ser comparados directamente.

Uso:
    DDE_BACKEND=pytorch python3 pinn_baseline.py [--epochs 10000] [--runs 1]
"""

import os
import sys
import time
import argparse
import csv
import warnings
warnings.filterwarnings("ignore")

os.environ.setdefault("DDE_BACKEND", "pytorch")
os.environ["CUDA_VISIBLE_DEVICES"] = "" # Force CPU to avoid kernel image errors

import numpy as np
import torch
import deepxde as dde

dde.config.set_default_float("float64")

_pi = np.pi

# ── Directorios ────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

METHOD = "PINN"
GRID_N = 50  # resolución del grid de evaluación (50×50 = 2500 puntos)

# ── Ecuaciones exactas (para comparar) ────────────────────────────────────────
def exact_laplace(x, y):
    return np.sin(np.pi * x) * np.sinh(np.pi * y) / np.sinh(np.pi)

def exact_poisson(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

def exact_helmholtz(x, y, k=1.0):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

def exact_schrodinger(x, y):
    r2 = (x - 0.5)**2 + (y - 0.5)**2
    return np.exp(-2.0 * r2)

# ── Configuración de red ───────────────────────────────────────────────────────
def make_net(layers=4, width=64):
    return dde.nn.FNN(
        [2] + [width] * layers + [1],
        activation="tanh",
        kernel_initializer="Glorot normal",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# LAPLACE:  u_xx + u_yy = 0  en [0,1]²
#   BC: u(x,0)=0, u(x,1)=sin(πx), u(0,y)=0, u(1,y)=0
#   Exacta: sin(πx)·sinh(πy)/sinh(π)
# ═══════════════════════════════════════════════════════════════════════════════
def solve_laplace(n_domain, n_boundary, epochs, lr):
    geom = dde.geometry.Rectangle([0, 0], [1, 1])

    def pde(x, u):
        u_xx = dde.grad.hessian(u, x, i=0, j=0)
        u_yy = dde.grad.hessian(u, x, i=1, j=1)
        return u_xx + u_yy

    def bc_func(x):
        return np.sin(np.pi * x[:, 0:1]) * np.sinh(np.pi * x[:, 1:2]) / np.sinh(np.pi)

    bc = dde.icbc.DirichletBC(geom, bc_func, lambda _, on_bnd: on_bnd)
    data = dde.data.PDE(geom, pde, bc, num_domain=n_domain, num_boundary=n_boundary, num_test=500)
    net  = make_net()
    model = dde.Model(data, net)
    print("    [Adam] Entrenando...")
    model.compile("adam", lr=lr, loss_weights=[1, 100])
    model.train(iterations=epochs, display_every=1000)
    
    print("    [L-BFGS] Refinando...")
    dde.optimizers.config.set_LBFGS_options(maxiter=2000)
    model.compile("L-BFGS")
    model.train()
    return model, exact_laplace

# ═══════════════════════════════════════════════════════════════════════════════
# POISSON:  u_xx + u_yy = f   donde f = -2π²·sin(πx)·sin(πy)
#   BC: u = 0 en ∂Ω
#   Exacta: sin(πx)·sin(πy)
# ═══════════════════════════════════════════════════════════════════════════════
def solve_poisson(n_domain, n_boundary, epochs, lr):
    geom = dde.geometry.Rectangle([0, 0], [1, 1])

    def pde(x, u):
        u_xx = dde.grad.hessian(u, x, i=0, j=0)
        u_yy = dde.grad.hessian(u, x, i=1, j=1)
        f    = -2 * _pi**2 * torch.sin(_pi * x[:, 0:1]) * torch.sin(_pi * x[:, 1:2])
        return u_xx + u_yy - f

    bc   = dde.icbc.DirichletBC(geom, lambda x: np.zeros((len(x), 1)), lambda _, on_bnd: on_bnd)
    data = dde.data.PDE(geom, pde, bc, num_domain=n_domain, num_boundary=n_boundary, num_test=500)
    net  = make_net()
    model = dde.Model(data, net)
    print("    [Adam] Entrenando...")
    model.compile("adam", lr=lr, loss_weights=[1, 100])
    model.train(iterations=epochs, display_every=1000)
    
    print("    [L-BFGS] Refinando...")
    dde.optimizers.config.set_LBFGS_options(maxiter=2000)
    model.compile("L-BFGS")
    model.train()
    return model, lambda x, y: exact_poisson(x, y)

# ═══════════════════════════════════════════════════════════════════════════════
# HELMHOLTZ:  u_xx + u_yy + k²u = f    k=1
#   f = (2π² - k²)·sin(πx)·sin(πy)
#   BC: u = 0 en ∂Ω
#   Exacta: sin(πx)·sin(πy)
# ═══════════════════════════════════════════════════════════════════════════════
def solve_helmholtz(n_domain, n_boundary, epochs, lr, k=1.0):
    geom = dde.geometry.Rectangle([0, 0], [1, 1])

    def pde(x, u):
        u_xx = dde.grad.hessian(u, x, i=0, j=0)
        u_yy = dde.grad.hessian(u, x, i=1, j=1)
        f    = (2 * _pi**2 - k**2) * torch.sin(_pi * x[:, 0:1]) * torch.sin(_pi * x[:, 1:2])
        return u_xx + u_yy + k**2 * u - f

    bc   = dde.icbc.DirichletBC(geom, lambda x: np.zeros((len(x), 1)), lambda _, on_bnd: on_bnd)
    data = dde.data.PDE(geom, pde, bc, num_domain=n_domain, num_boundary=n_boundary, num_test=500)
    net  = make_net()
    model = dde.Model(data, net)
    print("    [Adam] Entrenando...")
    model.compile("adam", lr=lr, loss_weights=[1, 100])
    model.train(iterations=epochs, display_every=1000)
    
    print("    [L-BFGS] Refinando...")
    dde.optimizers.config.set_LBFGS_options(maxiter=2000)
    model.compile("L-BFGS")
    model.train()
    return model, lambda x, y: exact_helmholtz(x, y, k)

# ═══════════════════════════════════════════════════════════════════════════════
# SCHRÖDINGER:  -∇²u + V(x,y)·u = E·u    E=4, V=4r²
#   ⟺  u_xx + u_yy + (E - V)·u = 0
#   BC: u ≈ 0 en bordes (Dirichlet)
#   Exacta: exp(-2r²), r² = (x-0.5)² + (y-0.5)²
# ═══════════════════════════════════════════════════════════════════════════════
def solve_schrodinger(n_domain, n_boundary, epochs, lr):
    geom = dde.geometry.Rectangle([0, 0], [1, 1])
    E    = 4.0

    def pde(x, u):
        u_xx = dde.grad.hessian(u, x, i=0, j=0)
        u_yy = dde.grad.hessian(u, x, i=1, j=1)
        r2   = (x[:, 0:1] - 0.5)**2 + (x[:, 1:2] - 0.5)**2
        V    = 4.0 * r2
        return u_xx + u_yy + (E - V) * u

    def bc_func(x):
        r2 = (x[:, 0:1] - 0.5)**2 + (x[:, 1:2] - 0.5)**2
        return np.exp(-2.0 * r2)

    bc   = dde.icbc.DirichletBC(geom, bc_func, lambda _, on_bnd: on_bnd)
    data = dde.data.PDE(geom, pde, bc, num_domain=n_domain, num_boundary=n_boundary, num_test=500)
    net  = make_net()
    model = dde.Model(data, net)
    print("    [Adam] Entrenando...")
    model.compile("adam", lr=lr, loss_weights=[1, 100])
    model.train(iterations=epochs, display_every=1000)
    
    print("    [L-BFGS] Refinando...")
    dde.optimizers.config.set_LBFGS_options(maxiter=2000)
    model.compile("L-BFGS")
    model.train()
    return model, exact_schrodinger

# ── Evaluación sobre una grilla uniforme 50×50 ────────────────────────────────
def eval_on_grid(model, exact_fn, pde_name, run_dir):
    xs = np.linspace(0, 1, GRID_N)
    ys = np.linspace(0, 1, GRID_N)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.column_stack([xx.ravel(), yy.ravel()])

    u_approx = model.predict(pts).ravel()
    u_exact  = np.array([exact_fn(p[0], p[1]) for p in pts])

    mse_domain = float(np.mean((u_approx - u_exact)**2))

    # Guardar grid CSV
    grid_path = os.path.join(run_dir, f"grid_{pde_name}_{METHOD}.csv")
    with open(grid_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "u_exact", "u_approx"])
        for i in range(len(pts)):
            w.writerow([f"{pts[i,0]:.6f}", f"{pts[i,1]:.6f}",
                        f"{u_exact[i]:.6f}", f"{u_approx[i]:.6f}"])
    return mse_domain

def eval_boundary_mse(model, exact_fn):
    """MSE en los 4 bordes del cuadrado unitario."""
    t = np.linspace(0, 1, 80)
    bnd_pts = np.concatenate([
        np.column_stack([t,        np.zeros_like(t)]),   # y=0
        np.column_stack([t,        np.ones_like(t)]),    # y=1
        np.column_stack([np.zeros_like(t), t]),          # x=0
        np.column_stack([np.ones_like(t),  t]),          # x=1
    ])
    u_approx = model.predict(bnd_pts).ravel()
    u_exact  = np.array([exact_fn(p[0], p[1]) for p in bnd_pts])
    return float(np.mean((u_approx - u_exact)**2))

# ── Guardar pareto CSV (PINN no tiene frente Pareto, es un único punto) ───────
def save_pareto_csv(pde_name, mse_dom, mse_bnd, run_dir):
    path = os.path.join(run_dir, f"{pde_name}_{METHOD.lower()}_pareto.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "pde", "mse_domain", "mse_boundary", "rank"])
        w.writerow([METHOD, pde_name, f"{mse_dom:.10f}", f"{mse_bnd:.10f}", 1])

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="PINN baseline with DeepXDE")
    parser.add_argument("--epochs",     type=int,   default=10000, help="Training iterations (Adam)")
    parser.add_argument("--n_domain",   type=int,   default=2500,  help="Collocation points inside domain")
    parser.add_argument("--n_boundary", type=int,   default=200,   help="Boundary points")
    parser.add_argument("--lr",         type=float, default=1e-3,  help="Adam learning rate")
    parser.add_argument("--runs",       type=int,   default=1,     help="Independent runs (for stats)")
    args = parser.parse_args()

    problems = [
        ("Laplace",     solve_laplace),
        ("Poisson",     solve_poisson),
        ("Helmholtz",   solve_helmholtz),
        ("Schrodinger", solve_schrodinger),
    ]

    all_rows = []

    for run_idx in range(args.runs):
        run_dir = RESULTS_DIR if args.runs == 1 else os.path.join(RESULTS_DIR, f"run_{run_idx}")
        os.makedirs(run_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  PINN Baseline  (run {run_idx+1}/{args.runs})")
        print(f"  Epochs={args.epochs}  N_dom={args.n_domain}  N_bnd={args.n_boundary}")
        print(f"{'='*60}")

        for pde_name, solver_fn in problems:
            print(f"\n── {pde_name} ──")
            t0 = time.time()
            model, exact_fn = solver_fn(args.n_domain, args.n_boundary, args.epochs, args.lr)
            rt = time.time() - t0

            mse_dom = eval_on_grid(model, exact_fn, pde_name, run_dir)
            mse_bnd = eval_boundary_mse(model, exact_fn)

            print(f"  Domain MSE={mse_dom:.4e}  |  BC MSE={mse_bnd:.4e}  |  Time={rt:.1f}s")

            save_pareto_csv(pde_name, mse_dom, mse_bnd, run_dir)

            all_rows.append({
                "run":                run_idx,
                "method":             METHOD,
                "pde":                pde_name,
                "pareto_size":        1,
                "best_mse_domain":    mse_dom,
                "best_mse_boundary":  mse_bnd,
                "mean_mse_domain":    mse_dom,
                "mean_mse_boundary":  mse_bnd,
                "hypervolume":        0.0,   # PINN no tiene frente Pareto
                "runtime_s":          rt,
            })

    # Guardar summary CSV para integración con stats_analysis.py
    if args.runs > 1:
        summary_path = os.path.join(RESULTS_DIR, "pinn_summary.csv")
        fieldnames = ["run", "method", "pde", "pareto_size",
                      "best_mse_domain", "best_mse_boundary",
                      "mean_mse_domain", "mean_mse_boundary",
                      "hypervolume", "runtime_s"]
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_rows)
        print(f"\n  Guardado: {summary_path}")
        print("  Combina con all_runs_summary.csv para análisis estadístico.")
    else:
        # Agregar a comparison_summary.csv (run único)
        comp_path = os.path.join(RESULTS_DIR, "comparison_summary.csv")
        if os.path.exists(comp_path):
            # Leer y agregar sin duplicar
            with open(comp_path, "r") as f:
                existing = f.read()
            existing_lines = [l for l in existing.strip().split("\n") if METHOD not in l]
            with open(comp_path, "w") as f:
                f.write("\n".join(existing_lines) + "\n")
                for row in all_rows:
                    f.write(f"{METHOD},{row['pde']},{1},{row['best_mse_domain']:.10f},"
                            f"{row['best_mse_boundary']:.10f},{row['mean_mse_domain']:.10f},"
                            f"{row['mean_mse_boundary']:.10f},{row['hypervolume']:.10f},{row['runtime_s']:.4f}\n")
            print(f"\n  Actualizado: {comp_path}")

        # También agregar a all_runs_summary.csv para stats_analysis.py
        all_runs_path = os.path.join(RESULTS_DIR, "all_runs_summary.csv")
        file_exists = os.path.exists(all_runs_path)
        with open(all_runs_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["run", "method", "pde", "pareto_size",
                                             "best_mse_domain", "best_mse_boundary",
                                             "mean_mse_domain", "mean_mse_boundary",
                                             "hypervolume", "runtime_s"])
            if not file_exists:
                w.writeheader()
            for row in all_rows:
                row_to_write = row.copy()
                row_to_write["method"] = METHOD
                w.writerow(row_to_write)
        print(f"  Añadido a: {all_runs_path}")

    print("\n  ✅ PINN baseline completado.")
    print("     Ejecuta: python3 plot_pareto.py  para ver los resultados.")

if __name__ == "__main__":
    main()
