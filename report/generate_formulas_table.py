#!/usr/bin/env python3
import os

# Rutas dinámicas relativas a la ubicación del script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
TABLES_DIR = os.path.join(SCRIPT_DIR, "tables")

def get_formula(pde, dim, method):
    label = f"{pde}_{dim}D"
    import glob
    pattern = os.path.join(RESULTS_DIR, "**", f"expr_{label}_{method}.tex")
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        return "N/A"
    
    path = matches[0]
    with open(path, "r") as f:
        content = f.read().strip()
        # Eliminar signos de dólar y posibles prefijos de fórmula
        content = content.replace("$$", "").replace("$", "")
        for prefix in ["\\hat{u}(x,y) =", "\\hat{u}(x) ="]:
            content = content.replace(prefix, "")
        return content.strip()
    return "N/A"

def main():
    os.makedirs(TABLES_DIR, exist_ok=True)
    pdes = ["Laplace", "Poisson", "Helmholtz", "Schrodinger"]
    dims = [1, 2]
    
    exacts = {
        ("Laplace", 1): r"x",
        ("Laplace", 2): r"\frac{\sin(\pi x) \sinh(\pi y)}{\sinh(\pi)}",
        ("Poisson", 1): r"\sin(\pi x)",
        ("Poisson", 2): r"\sin(\pi x) \sin(\pi y)",
        ("Helmholtz", 1): r"\sin(\pi x)",
        ("Helmholtz", 2): r"\sin(\pi x) \sin(\pi y)",
        ("Schrodinger", 1): r"e^{i \pi x}",
        ("Schrodinger", 2): r"e^{i \pi (x+y)}",
        ("Airy", 1): r"\text{Ai}(x)",
        ("HarmonicOscillator", 1): r"e^{-0.5 x^2}",
        ("HarmonicOscillator", 2): r"e^{-0.5 (x^2 + y^2)}",
        ("NonlinearPoisson", 2): r"1 / (1 + x^2 + y^2)",
        ("Liouville", 2): r"1 / (1 + x^2 + y^2)",
        ("Sine-Gordon", 2): r"\sin(\pi x) \sin(\pi y)",
        ("Navier-Stokes", 2): r"y - \frac{e^{\lambda x}}{2 \pi \text{Re}} \sin(2 \pi y)",
        ("Navier-Stokes-Unsteady", 2): r"\sin(\pi x) \cos(\pi y) e^{-0.1 t}",
        ("Fisher", 1): r"\text{Numerical}",
        ("Fisher", 2): r"\text{Numerical}",
        ("Duffing", 1): r"\text{Numerical}",
        ("Duffing", 2): r"\text{Numerical}",
        ("ThomasFermi", 1): r"\text{Numerical}",
        ("ThomasFermi", 2): r"\text{Numerical}",
        ("Bratu", 2): r"\ln(2/\cosh^2(r))",
        ("Allen-Cahn", 2): r"\tanh(r/\epsilon)",
        ("Lane-Emden", 1): r"1 - x^2/6"
    }
    
    lines = [
        r"\begin{table*}[ht]",
        r"  \centering",
        r"  \caption{Comparison of Symbolic Expressions (1D and 2D).}",
        r"  \label{tab:formulas}",
        r"  \resizebox{\textwidth}{!}{",
        r"  \begin{tabular}{llp{14cm}}",
        r"    \toprule",
        r"    \textbf{PDE} & \textbf{Method} & \textbf{Symbolic Expression} $\hat{u}$ \\",
        r"    \midrule"
    ]
    
    for pde in ["Laplace", "Poisson", "Helmholtz", "Schrodinger", "Airy", "HarmonicOscillator", "Fisher", "Duffing", "ThomasFermi", "NonlinearPoisson", "Liouville", "Sine-Gordon", "Navier-Stokes", "Navier-Stokes-Unsteady", "Bratu", "Allen-Cahn", "Lane-Emden"]:
        for d in dims:
            if pde in ["NonlinearPoisson", "Liouville", "Sine-Gordon", "Navier-Stokes", "Navier-Stokes-Unsteady", "Bratu", "Allen-Cahn"] and d == 1: continue
            if pde == "Lane-Emden" and d == 2: continue
            p_pi = get_formula(pde, d, "PI-NSGA-II")
            ex_eq = exacts.get((pde, d), "N/A")
            
            lines.append(rf"    \multirow{{2}}{{*}}{{{pde} ({d}D)}}")
            lines.append(rf"    & PI-NSGA-II & ${p_pi}$ \\")
            lines.append(rf"    & \textbf{{Exact}} & $\mathbf{{{ex_eq}}}$ \\")
            lines.append(r"    \midrule")
            
    lines += [r"    \bottomrule", r"  \end{tabular}", r"  }", r"\end{table*}"]
    
    outpath = os.path.join(TABLES_DIR, "formulas_table.tex")
    with open(outpath, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Guardado: {outpath}")

if __name__ == "__main__":
    main()
