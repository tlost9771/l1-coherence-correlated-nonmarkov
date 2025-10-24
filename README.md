# Quantum Coherence – Figures & Scripts

Generate all figures for the paper (Pauli, GAD, JC, PMME, μ_c). PDFs land in `./figures/`.

## Local usage
```bash
pip install -r requirements.txt
python make_figures.py
# or
make install
make figures
```

## LaTeX
```tex
\usepackage{graphicx}
\graphicspath{{figures/}}
```

## GitHub
```bash
git init && git add . && git commit -m "init"
git branch -M main
git remote add origin https://github.com/<OWNER>/<REPO>.git
git push -u origin main
```
