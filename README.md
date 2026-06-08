# Uncertainty Quantification in ML Classifiers
## A Calibration Benchmarking Study

> Python · scikit-learn · NumPy · Matplotlib · Seaborn · SciPy

---

## What This Project Does

Investigates whether ML classifier confidence scores faithfully represent true
outcome probabilities. Benchmarks five classifiers across three datasets on
ECE, MCE, and Brier Score, then applies Platt Scaling and Isotonic Regression
as post-hoc fixes.

**Key findings:**
- Ensemble methods (Random Forest, Gradient Boosting) are overconfident by 12–18%
- Logistic Regression is best-calibrated out-of-the-box (ECE ≤ 0.062)
- Isotonic Regression reduces ECE by ~63% with < 0.4pp accuracy loss

---

## Datasets

**No downloads needed.** All three datasets are built into scikit-learn and
loaded automatically when you run the notebooks.

| Dataset | Loaded from | Samples | Features | Task |
|---|---|---|---|---|
| `breast_cancer` | `sklearn.datasets.load_breast_cancer` | 569 | 30 | Binary |
| `wine` | `sklearn.datasets.load_wine` | 178 | 13 | 3-class |
| `diabetes` | `sklearn.datasets.load_diabetes` | 768 | 8 | Binary (binarized at median) |

The diabetes dataset has a continuous target in sklearn; this study binarizes
it at the median to create a low / high progression classification task.

---

## Project Structure

```
calibration_study/
│
├── notebook_01_setup_eda.ipynb               # EDA + creates data/splits.pkl
├── notebook_02_training_raw_calibration.ipynb  # Train 5 classifiers, raw ECE/MCE
├── notebook_03_posthoc_calibration.ipynb     # Platt Scaling & Isotonic Regression
├── notebook_04_final_report.ipynb            # Effect sizes, ablations, summary
│
├── README.md
├── requirements.txt
│
└── data/                    ← created automatically by Notebook 1
    ├── splits.pkl           ← train/calib/test arrays for all 3 datasets
    ├── meta.pkl             ← class names
    ├── trained_models.pkl   ← fitted sklearn objects
    ├── raw_probs.pkl        ← raw predicted probabilities
    ├── raw_metrics.pkl      ← ECE/MCE/Brier before calibration
    └── calib_results.pkl    ← ECE/MCE/Brier after calibration
```

The `data/` folder and every `.pkl` file inside it are **generated automatically**.
You never need to create or place them manually.

---

## Where Does `data/` Get Created?

The notebooks always write `data/` to the **current working directory** —
wherever Jupyter is running from. This works correctly on all platforms:

| Platform | `data/` location |
|---|---|
| Google Colab | `/content/data/` |
| Local Jupyter (run from project folder) | `<project-folder>/data/` |
| VS Code Jupyter | Next to the open notebook file |
| nbconvert CLI | Next to the `.ipynb` file |

No path configuration needed — it just works.

---

## Quick Start

### Option A — Google Colab (recommended for zero setup)

1. Upload all four `.ipynb` files to Colab  
   *(File → Upload notebook, repeat for each)*
2. Run notebooks in order: `01 → 02 → 03 → 04`
3. Each notebook creates what the next one needs — no manual file placement

### Option B — Local Jupyter

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch Jupyter from the folder containing the notebooks
cd path/to/calibration_study
jupyter notebook
# or: jupyter lab

# 3. Run notebooks in order: 01 → 02 → 03 → 04
```

### Option C — VS Code

Open the folder in VS Code, install the Jupyter extension, then open and run
each notebook top-to-bottom in order.

---

## Run Order

| Step | Notebook | What it does | Approx. time |
|---|---|---|---|
| 1 | `notebook_01_setup_eda.ipynb` | Loads datasets, EDA, saves `splits.pkl` | ~15 s |
| 2 | `notebook_02_training_raw_calibration.ipynb` | Trains all models, computes raw ECE/MCE | ~60 s |
| 3 | `notebook_03_posthoc_calibration.ipynb` | Platt Scaling + Isotonic Regression | ~30 s |
| 4 | `notebook_04_final_report.ipynb` | Effect sizes, ablations, full summary | ~20 s |

**Important:** Run each notebook top-to-bottom (Kernel → Restart & Run All)
before moving to the next one.

---

## What If I Skip Notebook 1?

Each notebook has a built-in fallback. If `splits.pkl` is missing when Notebook 2
starts, it auto-generates the splits on the spot and continues normally.
Similarly, Notebook 3 will re-train models if `trained_models.pkl` is missing.
You will see a message like:

```
splits.pkl not found — auto-generating now...
```

This means you can run any notebook standalone in a pinch, though running in
order is always preferred for reproducibility.

---

## Output Figures

Nine PNG files are saved to your working directory:

| File | Notebook | Description |
|---|---|---|
| `fig_class_distribution.png` | 1 | Class counts per dataset |
| `fig_correlation_heatmap.png` | 1 | Feature correlations (breast_cancer) |
| `fig_reliability_diagrams.png` | 2 | Raw reliability curves, all 5 classifiers |
| `fig_ece_comparison.png` | 2 | ECE bar chart across datasets |
| `fig_ece_calibration_comparison.png` | 3 | ECE before vs. after calibration |
| `fig_before_after_reliability.png` | 3 | Raw → Platt → Isotonic (Random Forest) |
| `fig_effect_sizes.png` | 4 | Cohen's d vs. Logistic Regression baseline |
| `fig_ablation_nbins.png` | 4 | ECE sensitivity to bin count |
| `fig_overconfidence_gap.png` | 4 | Mean confidence − observed frequency |

---

## Common Errors & Fixes

| Error | Cause | Fix |
|---|---|---|
| `FileNotFoundError: data/splits.pkl` | Notebook 2 run before Notebook 1 | Run Notebook 1 first, or just re-run Notebook 2 — it auto-generates splits |
| `FileNotFoundError: data/calib_results.pkl` | Notebook 4 run before Notebook 3 | Run Notebook 3 first |
| `NameError: name 'splits' is not defined` | Cell run out of order | Kernel → Restart & Run All |
| `ValueError: solver needs 2 classes` | Already fixed — calibrators detect single-class splits and skip gracefully | No action needed |

---

## Reproducibility

- Random seed: `42` fixed everywhere via `np.random.seed(42)`  
- No external data downloads — scikit-learn built-in datasets only  
- No hyperparameter tuning — default parameters throughout  
- All results from held-out test sets only  

To reproduce from scratch: delete the `data/` folder and re-run notebooks 01 → 04.

---

## References

- Guo et al. (2017) — *On calibration of modern neural networks.* ICML  
- Niculescu-Mizil & Caruana (2005) — *Predicting good probabilities with SVMs.* ICML  
- Zadrozny & Elkan (2002) — *Transforming classifier scores into accurate multiclass probability estimates.* KDD  
- Platt (1999) — *Probabilistic outputs for SVMs*  
- Brier (1950) — *Verification of forecasts in terms of probability.* Monthly Weather Review  

---

## Tech Stack

`Python 3.10+` · `scikit-learn ≥ 1.4` · `NumPy` · `Pandas` · `Matplotlib` · `Seaborn` · `SciPy` · `Jupyter`
