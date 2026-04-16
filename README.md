# Lasso Coefficient Comparison

A small Python project that simulates **many time-series features** and runs LASSO regression using both **scikit-learn** and **R's `glmnet` package** (via `rpy2`).

It includes:
- A **command-line script** (`main.py`) that fits and evaluates the R model on a panel of simulated series.
- An interactive **Streamlit app** (`app.py`) that lets you compare coefficients and predictions side-by-side.

## What it does

Instead of using lagged values of a single time series, this project simulates **many independent time-series features** (`x1`, `x2`, ..., `xN`). Only a random subset of them are truly informative — the rest are noise. LASSO's job is to identify the informative series and shrink the rest to exactly zero.

Both scikit-learn and `glmnet` are configured with the **same penalty** and **no automatic standardization**, so the coefficient comparison is truly apples-to-apples. The scikit-learn `Lasso` also uses tightened convergence settings (`max_iter=20000`, `tol=1e-6`, `selection='random'`) to match `glmnet`'s coordinate-descent behavior as closely as possible.

## Prerequisites

You need **both** a Python and an R environment installed on your machine.

### Python
- **Python 3.11** is required.  
  *Why?* The version of R on this machine (4.1.2) is incompatible with `rpy2 >= 3.6`. We therefore pin `rpy2` to `< 3.6`, which does not support Python 3.13.
- [`uv`](https://docs.astral.sh/uv/) is used for dependency management and virtual environments.

### R
- **R >= 4.0** (tested on R 4.1.2).
- The R package **`glmnet`** must be installed. You can install it from an R console or directly from the shell:

```bash
R -e "install.packages('glmnet', repos='https://cloud.r-project.org/')"
```

## Installation

1. Clone or navigate into the project directory:

```bash
cd lasso-test
```

2. Ensure Python 3.11 is active (the repo already contains a `.python-version` file):

```bash
uv python install 3.11   # only if you don't have it yet
```

3. Install Python dependencies:

```bash
uv sync
```

This will create a `.venv` and install:
- `scikit-learn`, `numpy`, `pandas`
- `rpy2` (< 3.6)
- `streamlit`, `plotly`

## Project structure

```
lasso-test/
├── .python-version      # pinned to 3.11
├── pyproject.toml       # uv project manifest
├── README.md            # this file
├── main.py              # CLI script (rpy2 + glmnet only)
├── app.py               # Streamlit comparison app (sklearn vs glmnet)
└── uv.lock              # uv lockfile
```

## Solver Matching Note

When the penalty (`alpha` / `lambda`) and standardization are matched, scikit-learn and R `glmnet` produce coefficient estimates that differ by less than `1e-3` and identical sparsity patterns in most random seeds. Any visible discrepancies in the app are therefore due to genuine solver differences (coordinate-descent path, warm-start strategy, strong-rule screening) rather than mismatched hyper-parameters.

## Usage

### 1. Command-line script (`main.py`)

Runs a quick end-to-end pipeline:
1. Simulates `n_features` time-series (AR processes).
2. Randomly selects `n_informative` of them to drive the target `y`.
3. Splits train / test temporally.
4. Fits LASSO via **R `glmnet`** with a fixed penalty.
5. Prints evaluation metrics, true-positive count, and top coefficients.

```bash
uv run main.py
```

Example output:
```
Informative series: ['series_12', 'series_22', 'series_14', 'series_11', 'series_19']
Non-zero coefficients: 10/50
Intercept: -0.0071
MSE:  1.1705
MAE:  0.8948
R^2:  0.9546
True positives: 5/5
False positives: 5
```

### 2. Streamlit comparison app (`app.py`)

Launch the interactive UI:

```bash
uv run streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

**What you'll see:**
- **Sidebar controls** for the simulation:
  - `Samples` – length of each time series
  - `Number of X series` – total features generated
  - `Informative series` – how many truly affect `y`
  - `Noise σ` and `Random seed`
- **Shared penalty slider** that controls `alpha` for both models.
- **Metrics cards** (MSE, MAE, R², non-zero count) for both scikit-learn and R glmnet.
- **Interactive coefficient chart** comparing ground-truth coefficients against scikit-learn and R glmnet estimates.
- **Coefficient table** with exact values and differences.
- **Prediction preview** plotting actual vs. predicted values on the first 20 test points.

## Thread safety note

Because the app runs inside Streamlit's threaded server, the R execution is wrapped with `rpy2.rinterface_lib.openrlib.rlock`. **R is not thread-safe**; without this lock, concurrent requests would crash the embedded R process.

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `undefined symbol: R_existsVarInFrame` or `symbol 'R_getVar' not found` | `rpy2 >= 3.6` incompatible with R 4.1.2 | Already handled by pinning `rpy2<3.6` in `pyproject.toml` |
| `R glmnet execution failed` | Missing `glmnet` R package | Run `R -e "install.packages('glmnet', repos='https://cloud.r-project.org/')"` |
| Coefficients don't match | Different standardization or penalty values | Ensure you are using `app.py` (not `main.py`), which shares the same `alpha` and disables `glmnet` standardization |
| `ImportError: cannot import name 'globalenv' from 'rpy2.robjects'` | Very old `rpy2` version | Use `uv sync` to install the exact pinned versions |

## License

MIT (or whatever you prefer — this is a demo project).
