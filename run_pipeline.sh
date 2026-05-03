#!/bin/bash
set -e
cd /home/uwo/Projects/PI-NSGA-II

echo "Step 1: Stats Analysis"
python3 stats_analysis.py

echo "Step 2: Single run for 3D grids"
./build/pi_nsga2 > single_run.log

echo "Step 3: Plot Pareto"
python3 plot_pareto.py

echo "Step 4: Plot Solutions 3D"
python3 plot_solutions.py

echo "Step 5: Generate Report tables and copy figures"
python3 report/generate_report.py

echo "Step 6: Compile LaTeX PDF"
cd report
pdflatex -interaction=nonstopmode results.tex > /dev/null
pdflatex -interaction=nonstopmode results.tex > /dev/null
echo "PIPELINE COMPLETED SUCCESSFULLY"
