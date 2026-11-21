#!/usr/bin/env python3
"""PINN de referencia (Raissi et al. 2019) para la suite de PISR-NSGA-II.

Un trabajo = un problema x una semilla. Entrena una red densa que minimiza el
residuo de la EDP en puntos de colocacion mas el error de los datos de borde, y
mide el error contra la referencia en los MISMOS puntos y con la MISMA
referencia que solution_mse de PISR-NSGA-II.

Los datos (puntos de validacion, referencia, frontera) los escribe el binario:
    ./build_release/PISR-NSGA-II --export-reference results/reference
Los residuos de abajo son copia de PDEProblem::pde_residual_ad
(src/pde_problems.cpp); si cambia una ecuacion alla, hay que cambiarla aca.

Salidas, en results/run_<k>/ (o results/ con --run -1):
    <EDP>_pinn.csv        metricas (method=PINN)
    <EDP>_pinn_pred.csv   prediccion en los puntos de validacion (entrada de PySR)
    <EDP>_pinn.pt         pesos de la red (para extrapolacion)
"""
import argparse, json, math, os, sys, time

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PI = math.pi


# ─── Problema ─────────────────────────────────────────────────────────────────
class Problem:
    def __init__(self, ref_dir, label):
        with open(os.path.join(ref_dir, label + ".json")) as f:
            self.meta = json.load(f)
        df = pd.read_csv(os.path.join(ref_dir, label + ".csv"))
        self.label = label
        self.pde = self.meta["pde"]
        self.dim = self.meta["dim"]
        self.unsteady = self.meta["unsteady"]
        self.cplx = self.meta["complex"]
        self.order = self.meta["spatial_order"]
        self.k2 = self.meta["k2"]
        # Columnas de entrada de la red: x, [y], [t]
        self.cols = ["x"] + (["y"] if self.dim == 2 else []) + (["t"] if self.unsteady else [])
        self.dom = df[df.kind == "dom"].reset_index(drop=True)
        self.bnd = df[df.kind == "bnd"].reset_index(drop=True)
        self.ivp = df[df.kind == "ivp"].reset_index(drop=True)
        self.n_out = 2 if self.cplx else 1

    def inputs(self, frame):
        return torch.tensor(frame[self.cols].to_numpy(), dtype=torch.float64)


# ─── Red ──────────────────────────────────────────────────────────────────────
class MLP(torch.nn.Module):
    def __init__(self, n_in, n_out, width, depth):
        super().__init__()
        layers, d = [], n_in
        for _ in range(depth):
            layers += [torch.nn.Linear(d, width), torch.nn.Tanh()]
            d = width
        layers.append(torch.nn.Linear(d, n_out))
        self.net = torch.nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, X):
        return self.net(X)


def grad(f, X, i):
    """d f / d X[:, i] (f: N x 1), conservando el grafo."""
    if not f.requires_grad:  # f constante respecto de X
        return torch.zeros_like(f)
    g = torch.autograd.grad(f, X, torch.ones_like(f), create_graph=True, allow_unused=True)[0]
    return torch.zeros_like(f) if g is None else g[:, i:i + 1]


# ─── Residuos (copia de pde_residual_ad) ──────────────────────────────────────
def residual(P, X, U):
    """Lista de componentes del residuo (1 si el campo es real, 2 si complejo)."""
    ix = 0
    iy = 1 if P.dim == 2 else None
    it = (2 if P.dim == 2 else 1) if P.unsteady else None
    x = X[:, ix:ix + 1]
    y = X[:, iy:iy + 1] if iy is not None else torch.zeros_like(x)

    def lap(u):
        ux = grad(u, X, ix)
        L = grad(ux, X, ix)
        if P.dim == 2:
            L = L + grad(grad(u, X, iy), X, iy)
        return L

    u = U[:, 0:1]
    name = P.pde
    if name == "Navier-Stokes" or name == "Navier-Stokes-Unsteady":
        nu = P.k2
        px, py = grad(u, X, ix), grad(u, X, iy)
        pxx, pyy, pxy = grad(px, X, ix), grad(py, X, iy), grad(px, X, iy)
        pxxx, pxyy = grad(pxx, X, ix), grad(pyy, X, ix)
        pxxy, pyyy = grad(pxx, X, iy), grad(pyy, X, iy)
        biharm = grad(pxxx, X, ix) + 2.0 * grad(pxxy, X, iy) + grad(pyyy, X, iy)
        if name == "Navier-Stokes":
            return [py * (pxxx + pxyy) - px * (pxxy + pyyy) - nu * biharm]
        w_t = -(grad(pxx, X, it) + grad(pyy, X, it))
        w_x, w_y = -(pxxx + pxyy), -(pxxy + pyyy)
        return [w_t + py * w_x + (-px) * w_y - nu * (-biharm)]

    if name == "Schrodinger":
        k = P.dim * P.k2
        return [lap(U[:, 0:1]) + k * U[:, 0:1], lap(U[:, 1:2]) + k * U[:, 1:2]]
    if name == "Schrodinger-Unsteady":
        a, b = U[:, 0:1], U[:, 1:2]
        # i (a + i b)_t + Lap(a + i b)
        return [lap(a) - grad(b, X, it), grad(a, X, it) + lap(b)]

    L = lap(u)
    if name == "Laplace":
        return [L]
    if name == "Poisson":
        src = PI ** 2 * torch.sin(PI * x) if P.dim == 1 else 2 * PI ** 2 * torch.sin(PI * x) * torch.sin(PI * y)
        return [L + src]
    if name == "HarmonicOscillator":
        V = x ** 2 if P.dim == 1 else x ** 2 + y ** 2
        return [L - V * u + P.dim * u]
    if name == "Airy":
        return [L - (x if P.dim == 1 else x + y) * u]
    if name == "Fisher":
        return [L + u * (1.0 - u)]
    if name == "Duffing":
        f = torch.sin(PI * x) if P.dim == 1 else torch.sin(PI * x) * torch.sin(PI * y)
        return [L + u + u ** 3 - f]
    if name == "Thomas-Fermi":
        r = torch.abs(x) if P.dim == 1 else torch.sqrt(x ** 2 + y ** 2)
        eps = 1e-12
        return [L - (torch.abs(u) + eps) ** 1.5 / torch.sqrt(r + eps)]
    if name == "Lane-Emden":
        # u'' + (2/x) u' + u^3 = 0, multiplicada por x (misma ecuacion, sin 1/x
        # en la funcion de perdida).
        return [x * L + 2.0 * grad(u, X, ix) + x * u ** 3]
    if name == "Troesch":
        return [L - 3.0 * torch.sinh(3.0 * u)]
    if name == "Ginzburg-Landau":
        return [L + u - u ** 3]
    if name == "Painleve-I":
        return [L - (u ** 2 + x)]
    raise ValueError(f"EDP sin residuo PINN: {name}")


def boundary_loss(P, model, Xb, gb, dnb, nrm):
    U = model(Xb)
    loss = ((U - gb) ** 2).sum(dim=1).mean()
    if P.order > 2 and nrm is not None:
        mask = (nrm.abs().sum(dim=1) > 0)
        if mask.any():
            ux, uy = grad(U[:, 0:1], Xb, 0), grad(U[:, 0:1], Xb, 1)
            dn = nrm[:, 0:1] * ux + nrm[:, 1:2] * uy
            loss = loss + ((dn - dnb[:, 0:1])[mask] ** 2).mean()
    return loss


def ivp_loss(P, model, X0, u0, du0):
    U = model(X0)
    loss = ((U - u0) ** 2).sum(dim=1).mean()
    for c in range(P.n_out):
        loss = loss + ((grad(U[:, c:c + 1], X0, 0) - du0[:, c:c + 1]) ** 2).mean()
    return loss


# ─── Entrenamiento ────────────────────────────────────────────────────────────
def train(P, seed, adam_iters, lbfgs_iters, n_col, width, depth, w_bnd, log_every, budget_s=math.inf):
    """Adam y luego L-BFGS. budget_s: tiempo maximo de pared; Adam usa hasta el
    75% y L-BFGS el resto. Devuelve (modelo, iteraciones Adam, iteraciones L-BFGS)."""
    t_start = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = MLP(len(P.cols), P.n_out, width, depth).double()

    # Datos de borde / valor inicial
    if len(P.ivp):
        X0 = P.inputs(P.ivp).requires_grad_(True)
        u0 = torch.tensor(P.ivp[["u_re", "u_im"]].to_numpy()[:, :P.n_out], dtype=torch.float64)
        du0 = torch.tensor(P.ivp[["dn_re", "dn_im"]].to_numpy()[:, :P.n_out], dtype=torch.float64)
        bnd = lambda: ivp_loss(P, model, X0, u0, du0)
    else:
        Xb_all = P.inputs(P.bnd)
        gb_all = torch.tensor(P.bnd[["u_re", "u_im"]].to_numpy()[:, :P.n_out], dtype=torch.float64)
        dnb_all = torch.tensor(P.bnd[["dn_re", "dn_im"]].to_numpy()[:, :P.n_out], dtype=torch.float64)
        nrm_all = None
        if P.order > 2:
            bx, by = P.bnd.x.to_numpy(), P.bnd.y.to_numpy()
            nx = np.where(bx == 0, -1.0, np.where(bx == 1, 1.0, 0.0))
            ny = np.where(nx != 0, 0.0, np.where(by == 0, -1.0, np.where(by == 1, 1.0, 0.0)))
            nrm_all = torch.tensor(np.stack([nx, ny], 1), dtype=torch.float64)
        # Lote de frontera: se re-muestrea junto con la colocacion.
        B = {}

        def resample_bnd():
            n = len(Xb_all)
            idx = torch.randperm(n, generator=gen)[:min(n, 1000)]
            B["X"] = Xb_all[idx].clone().requires_grad_(P.order > 2)
            B["g"], B["dn"] = gb_all[idx], dnb_all[idx]
            B["n"] = nrm_all[idx] if nrm_all is not None else None
        resample_bnd()
        bnd = lambda: boundary_loss(P, model, B["X"], B["g"], B["dn"], B["n"])

    def collocation():
        X = torch.rand(n_col, len(P.cols), generator=gen, dtype=torch.float64)
        return X.requires_grad_(True)

    def loss_fn(Xc):
        R = residual(P, Xc, model(Xc))
        l_dom = sum((r ** 2).mean() for r in R)
        return l_dom + w_bnd * bnd(), l_dom

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, adam_iters // 3), gamma=0.3)
    Xc = collocation()
    it = 0
    for it in range(adam_iters):
        if time.time() - t_start > 0.75 * budget_s:
            break
        if it % 250 == 0:
            Xc = collocation()
            if len(P.ivp) == 0:
                resample_bnd()
        opt.zero_grad()
        loss, l_dom = loss_fn(Xc)
        if not torch.isfinite(loss):
            break
        loss.backward()
        opt.step()
        sched.step()
        if log_every and it % log_every == 0:
            print(f"  [adam {it:6d}] loss={loss.item():.3e} dom={l_dom.item():.3e}", flush=True)

    n_adam = it + 1 if adam_iters > 0 else 0
    n_lbfgs = 0
    if lbfgs_iters > 0:
        Xc = collocation()
        chunk = 50
        lb = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=chunk, history_size=50,
                               tolerance_grad=1e-12, tolerance_change=1e-15, line_search_fn="strong_wolfe")
        snapshot = [p.detach().clone() for p in model.parameters()]

        def closure():
            lb.zero_grad()
            loss, _ = loss_fn(Xc)
            loss.backward()
            return loss
        try:
            # Por tramos, para poder cortar por tiempo (el estado de L-BFGS se conserva).
            while n_lbfgs < lbfgs_iters and time.time() - t_start < budget_s:
                lb.step(closure)
                n_lbfgs += chunk
                if lb.state[lb._params[0]].get("n_iter", 0) < n_lbfgs:  # convergio antes del tramo
                    break
        except RuntimeError:
            pass
        n_lbfgs = int(lb.state[lb._params[0]].get("n_iter", 0))
        final = loss_fn(Xc)[0]
        if not torch.isfinite(final):  # L-BFGS divergio: volver a los pesos de Adam
            with torch.no_grad():
                for p, s in zip(model.parameters(), snapshot):
                    p.copy_(s)
    return model, n_adam, min(n_lbfgs, lbfgs_iters)


def evaluate(P, model):
    Xd = P.inputs(P.dom).requires_grad_(True)
    U = model(Xd)
    R = residual(P, Xd, U)
    mse_dom = float(sum((r ** 2).mean() for r in R).detach())
    u = U.detach().numpy()
    ref = P.dom[["u_re", "u_im"]].to_numpy()
    if P.cplx:
        sol = float(np.mean((u[:, 0] - ref[:, 0]) ** 2 + (u[:, 1] - ref[:, 1]) ** 2))
    else:
        # Campo real: misma metrica que PISR (|u - ref|^2, ref con parte imaginaria 0)
        sol = float(np.mean((u[:, 0] - ref[:, 0]) ** 2 + ref[:, 1] ** 2))
    if len(P.ivp):
        X0 = P.inputs(P.ivp).requires_grad_(True)
        U0 = model(X0)
        mse_bnd = float(((U0 - torch.tensor(P.ivp[["u_re", "u_im"]].to_numpy()[:, :P.n_out])) ** 2).sum(1).mean().detach())
    else:
        Ub = model(P.inputs(P.bnd)).detach().numpy()
        mse_bnd = float(np.mean(((Ub - P.bnd[["u_re", "u_im"]].to_numpy()[:, :P.n_out]) ** 2).sum(1)))
    return sol, mse_dom, mse_bnd, u


def out_dir(run):
    d = os.path.join(ROOT, "results") if run < 0 else os.path.join(ROOT, "results", f"run_{run}")
    os.makedirs(d, exist_ok=True)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True)
    ap.add_argument("--run", type=int, default=-1, help="indice de semilla; -1 escribe en results/")
    ap.add_argument("--ref-dir", default=os.path.join(ROOT, "results", "reference"))
    ap.add_argument("--adam", type=int, default=10000)
    ap.add_argument("--lbfgs", type=int, default=2000)
    ap.add_argument("--width", type=int, default=50)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--w-bnd", type=float, default=10.0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--budget-min", type=float, default=30.0, help="tiempo maximo por trabajo (minutos)")
    ap.add_argument("--test", action="store_true", help="pocas iteraciones, para probar la tuberia")
    ap.add_argument("--force", action="store_true", help="recalcular aunque ya exista el resultado")
    ap.add_argument("--log-every", type=int, default=0)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    d = out_dir(a.run)
    metrics = os.path.join(d, f"{a.problem}_pinn.csv")
    if os.path.exists(metrics) and not a.force:
        print(f"[PINN] {a.problem} run {a.run}: ya existe, se omite")
        return
    P = Problem(a.ref_dir, a.problem)
    if a.test:
        a.adam, a.lbfgs = 300, 50
    n_col = {1: 1000, 2: 2000, 3: 2000}[len(P.cols)]
    if P.order > 2:
        n_col = 500  # derivadas de orden 4: ~0.3 s por iteracion con 500 puntos
    seed = 1000 + max(a.run, 0)

    t0 = time.time()
    model, n_adam, n_lbfgs = train(P, seed, a.adam, a.lbfgs, n_col, a.width, a.depth, a.w_bnd,
                                   a.log_every, a.budget_min * 60.0)
    rt = time.time() - t0
    sol, mse_dom, mse_bnd, u = evaluate(P, model)
    n_params = sum(p.numel() for p in model.parameters())

    pred = P.dom[P.cols].copy()
    pred["u_re"] = u[:, 0]
    pred["u_im"] = u[:, 1] if P.cplx else 0.0
    pred.to_csv(os.path.join(d, f"{a.problem}_pinn_pred.csv"), index=False)
    torch.save({"state": model.state_dict(), "width": a.width, "depth": a.depth,
                "cols": P.cols, "n_out": P.n_out}, os.path.join(d, f"{a.problem}_pinn.pt"))
    pd.DataFrame([{"method": "PINN", "pde": a.problem, "seed": seed, "solution_mse": sol,
                   "mse_domain": mse_dom, "mse_boundary": mse_bnd, "runtime_s": rt,
                   "n_params": n_params, "adam_iters": n_adam, "lbfgs_iters": n_lbfgs,
                   "budget_min": a.budget_min}]).to_csv(metrics, index=False)
    print(f"[PINN] {a.problem} run {a.run}: solution_mse={sol:.3e} residuo={mse_dom:.3e} "
          f"borde={mse_bnd:.3e} t={rt:.1f}s", flush=True)


if __name__ == "__main__":
    main()
