# PI-NSGA-II vs Koza BNF: Physics-Informed Multi-Objective Symbolic Regression

This repository contains the full C++ implementation, evaluation pipeline, and automated reporting system for the paper: **"PI-NSGA-II: Physics-Informed Multi-Objective Symbolic Regression — A Comparison Against Koza's Grammatical Evolution on Elliptic PDEs"**.

## Overview

The goal of this project is to discover closed-form, human-interpretable symbolic solutions to Partial Differential Equations (PDEs). We benchmark our proposed approach (**PI-NSGA-II**) against a classical Genetic Programming baseline (**Koza BNF**).

### Key Features
- **PI-NSGA-II (Ours)**: 
  - Individuals are evaluated via **Exact Automatic Differentiation (AD)** using the second-order chain rule. 
  - Zero truncation error, enabling the algorithm to find perfect PDE residual gradients.
  - Supports a rich set of operators including `sinh`, `cosh`, `tanh`, `exp`, `log`, `sqrt`, and `atan`.
  - Floating-point Ephemeral Random Constants (ERCs) with Gaussian mutation.
- **Koza BNF (Baseline)**:
  - Grammatical evolution mapping integer codon strings to symbolic expressions using a BNF grammar.
  - Relies on **Finite Difference (FD)** for the Laplacian, which introduces $O(h^2)$ truncation error and requires 5 evaluations per collocation point.
- **Bi-objective Optimization**: Both methods use NSGA-II to simultaneously minimize:
  1. $\mathcal{L}_{\text{dom}}$: Interior PDE residual (Mean Squared Error).
  2. $\mathcal{L}_{\text{bc}}$: Boundary Condition error (Mean Squared Error).
- **100% Reproducible**: All experiments use strict deterministic seeding (`seed = 1000 * (run_id + 1)`) ensuring zero variance across re-runs of the same benchmark.

## Dependencies

- **C++17** or higher
- **CMake** (>= 3.10)
- **Python 3.8+** (for statistical analysis and plotting)
  - `numpy`, `pandas`, `matplotlib`, `scipy`
- **LaTeX** distribution (`pdflatex`) to compile the paper report.

## Quick Start

### 1. Build the Project
We use CMake to build the C++ core:
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel $(nproc)
```

### 2. One-Click Reproducibility Pipeline
We provide a bash script that automatically runs the multi-run benchmark, extracts statistical features, evaluates single-run grids for 3D visualization, extracts symbolic expressions, plots everything, and compiles the final LaTeX PDF.

```bash
./run_pipeline.sh
```

### 3. Manual Step-by-Step Execution
If you prefer to run the stages manually:

**Step 3.1: Multi-Run Benchmark ($N=20$)**
Runs the algorithm 20 times for statistically significant comparisons:
```bash
./build/pi_nsga2 --runs 20
python3 stats_analysis.py
```

**Step 3.2: Single-Run Validation & Extraction**
Runs a single detailed iteration to dump evaluation grids (`grid_*.csv`) and exact symbolic math equations (`expr_*.tex`):
```bash
./build/pi_nsga2
python3 plot_solutions.py
```

**Step 3.3: Visualizations & Tables**
Generates Pareto fronts, sensitivity analysis, box plots, and method comparison bar charts:
```bash
python3 plot_pareto.py
python3 report/generate_formulas_table.py
python3 report/generate_report.py
```

**Step 3.4: Compile LaTeX PDF**
Compiles all the generated `.png` figures and `.tex` tables into a standalone paper-ready document:
```bash
cd report
pdflatex results.tex
```

*Final outputs, including images and CSV tables, will be available in the `results/` and `report/` directories.*

## Directory Structure
```
.
├── include/              # C++ Header files
│   ├── common.hpp        # Shared configurations (Pop size, Gens, etc.)
│   ├── nsga2.hpp         # Core NSGA-II selection/sorting logic
│   ├── pde_problems.hpp  # Definitions of Laplace, Poisson, Helmholtz
│   ├── tree_node.hpp     # Expression tree & Exact AD implementation
│   ├── koza_bnf.hpp      # Koza's baseline solver
│   └── pi_solver.hpp     # PI-NSGA-II solver
├── src/                  # C++ Source files
├── main.cpp              # Orchestrator and CLI
├── run_pipeline.sh       # Automated benchmarking and plotting pipeline
├── plot_pareto.py        # 2D Pareto, Dominance Map, and Bar charting
├── plot_solutions.py     # 3D Surface reconstruction of PDEs
├── stats_analysis.py     # Multi-run statistics & Wilcoxon testing
└── report/               # LaTeX templates and automated reporting
    ├── generate_report.py
    ├── generate_formulas_table.py
    ├── paper_template.tex
    └── results.tex       # Master document for PDF rendering
```

## Benchmark Problems

The methods are tested on three canonical elliptic PDEs defined on $\Omega = [0,1]^2$:
1. **Laplace Equation** ($\nabla^2 u = 0$)
2. **Poisson Equation** ($\nabla^2 u = f(x,y)$)
3. **Helmholtz Equation** ($\nabla^2 u + k^2 u = f(x,y)$)
