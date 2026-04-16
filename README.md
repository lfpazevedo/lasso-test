# Lasso Coefficient Comparison

A small Python project that simulates time-series data and runs LASSO regression using both **scikit-learn** and **R's `glmnet` package** (via `rpy2`).

It includes:
- A **command-line script** (`main.py`) that fits and evaluates the R model.
- An interactive **Streamlit app** (`app.py`) that lets you compare coefficients and predictions side-by-side.

## Why this exists

Comparing scikit-learn and `glmnet` can be tricky because the two libraries use different defaults for standardization and penalty parametrization. This project fixes those parameters so both solvers receive the **exact same objective function**, making the coefficient comparison truly apples-to-apples.

Key matching decisions in the app:
- **Shared penalty** (`alpha` / `lambda`) controlled by a single sidebar widget.
- **`standardize = FALSE`** in `glmnet` to match scikit-learn's default behavior (no automatic feature scaling).
- `alpha = 1` in `glmnet` to enforce a pure L1 (LASSO) penalty, identical to scikit-learn's `Lasso`.

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
└── app.py               # Streamlit comparison app (sklearn vs glmnet)
```

## Usage

### 1. Command-line script (`main.py`)

Runs a quick end-to-end pipeline:
1. Simulates a time series (trend + seasonality + noise).
2. Builds lag features.
3. Splits train / test temporally.
4. Fits LASSO via **R `glmnet`** using cross-validation (`lambda.1se`).
5. Prints evaluation metrics and top coefficients.

```bash
uv run main.py
```

Example output:
```
Non-zero coefficients: 8/20
Intercept: 1.1706
MSE:  7.4183
MAE:  2.2432
R^2:  0.8547
```

### 2. Streamlit comparison app (`app.py`)

Launch the interactive UI:

```bash
uv run streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

**What you'll see:**
- **Sidebar controls** for the simulation (samples, lags, noise, trend, seasonality) and a shared penalty slider.
- **Metrics cards** (MSE, MAE, R², non-zero count) for both scikit-learn and R glmnet.
- **Interactive coefficient chart** comparing every lag coefficient + intercept.
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
# lasso-test
