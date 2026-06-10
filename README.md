# PI-NSGA-II vs Koza BNF vs PINNs: Physics-Informed Multi-Objective Symbolic Regression & Neural Solvers on PDEs

This repository contains the C++ and Python codebase, benchmark evaluation suite, and automated publication-quality reporting system comparing three paradigms for solving differential equations:
1. **PI-NSGA-II**: Our proposed Physics-Informed Multi-Objective Symbolic Regression solver using exact Automatic Differentiation (AD).
2. **DeepXDE**: A deep neural network baseline powered by DeepXDE, optimized for execution on consumer GPUs.

---

## Key Features

### 1. The Solvers
* **PI-NSGA-II (Ours)**:
  * Expressions are evaluated using **Exact Automatic Differentiation (AD)** based on second-order chain rules.
  * Zero truncation error allows discovery of exact PDE residual gradients.
  * Rich operator library: `+`, `-`, `*`, `/`, `sin`, `cos`, `exp`, `log`, `sinh`, `cosh`, `tanh`, `sqrt`, `atan`.
  * Real-valued Ephemeral Random Constants (ERCs) with Gaussian mutation.
* **DeepXDE (Baseline)**:
  * Deep neural networks built using the DeepXDE framework.
  * Memory-optimized execution using VRAM-conserving techniques (mixed-precision training, gradient accumulation, and reduced hidden layers) to prevent CUDA out-of-memory errors on limited VRAM hardware.

### 2. Multi-Objective & Hypervolume Selection
* Evaluates candidates across three objective dimensions:
  1. $\mathcal{L}_{\text{dom}}$: Interior domain PDE residual (Mean Squared Error).
  2. $\mathcal{L}_{\text{bc}}$: Boundary condition compliance (Mean Squared Error).
  3. **Complexity**: Symbolic node/complexity metric.
* Analyzes Pareto fronts using a **3D Hypervolume (HV)** metric to determine structural convergence.

### 3. Comprehensive Benchmarks (13 Equations)
* **Elliptic PDEs**: Laplace ($\nabla^2 u = 0$), Poisson ($\nabla^2 u = f$), Helmholtz ($\nabla^2 u + k^2 u = f$), Nonlinear Poisson ($\nabla^2 u + u^2 = f$), Liouville ($\nabla^2 u = e^u$), Sine-Gordon ($\nabla^2 u = \sin(u)$), and Navier-Stokes ($\psi_y (\nabla^2 \psi)_x - \psi_x (\nabla^2 \psi)_y = \nu \nabla^4 \psi$).
* **ODEs & Systems**: Schrödinger ($-u'' + V u = E u$), Airy ($u'' = x\,u$), Harmonic Oscillator ($u'' = (x^2-1)u$), Fisher ($\nabla^2 u + u(1-u) = 0$), Duffing ($\nabla^2 u + u + u^3 = 0$), and Thomas-Fermi ($\nabla^2 u = u^2 / (x+y+0.5)$).

---

## Directory Structure

```
.
├── include/              # C++ Header files
│   ├── common.hpp        # Shared symbolic regression configs
│   ├── nsga2.hpp         # Core NSGA-II sorting and selection
│   ├── pde_problems.hpp  # Analytical and numerical boundary definitions
│   ├── tree_node.hpp     # Expression tree & Exact AD chain rule
│   ├── koza_bnf.hpp      # Grammatical evolution (Finite Difference)
│   └── pi_solver.hpp     # Physics-Informed Symbolic Regression (Exact AD)
├── src/                  # C++ Source files
├── main.cpp              # C++ Main entry point
├── pinn_baseline.py      # DeepXDE baseline execution (PyTorch backend)
├── plot_solutions.py     # 3D surface and 1D curve plotting pipeline
├── plot_pareto.py        # Pareto front and Hypervolume graphing
├── stats_analysis.py     # Multi-run statistics & Wilcoxon testing
├── run_pipeline.sh       # Automated C++/Python runner script
└── report/               # LaTeX templates and compilation files
    ├── generate_report.py
    ├── results.tex       # Master LaTeX document
    └── figures/          # Output vector PDF graphics
```

---

## Installation & Usage

### 1. Prerequisites
Ensure you have a C++17 compiler, CMake, and a Python 3 environment.

```bash
# Set up Python virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Build C++ Core
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel $(nproc)
```

### 3. Run DeepXDE Baseline
```bash
python3 pinn_baseline.py --only Laplace_2D
```
*Use `--only <equation_name>` to restrict training, or run without flags to train DeepXDE baselines on all 13 problems.*

### 4. Run the Full Evaluation & Compile Report
The automated pipeline executes the symbolic search runs, trains DeepXDE baselines, regenerates all vector PDF graphics, and compiles the LaTeX PDF:
```bash
./run_pipeline.sh
```
The compiled output is saved as `report/results.pdf`.
