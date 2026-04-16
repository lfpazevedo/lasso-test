import logging
import numpy as np
import pandas as pd
from rpy2.robjects import globalenv as ro_globalenv, r as ro_r
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import numpy2ri, pandas2ri
import rpy2.robjects as ro
from rpy2.rinterface_lib import openrlib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")


def simulate_many_series(n_samples=500, n_features=50, n_informative=5, noise_std=1.0, seed=42):
    """Simulate many time-series features; only n_informative truly affect y."""
    rng = np.random.default_rng(seed)

    # Generate AR(1) time-series features
    X = np.zeros((n_samples, n_features))
    for j in range(n_features):
        phi = rng.uniform(-0.5, 0.8)
        eps = rng.normal(0, 1, size=n_samples)
        X[0, j] = eps[0]
        for t in range(1, n_samples):
            X[t, j] = phi * X[t - 1, j] + eps[t]

    # Pick informative features and assign random true coefficients
    informative_idx = rng.choice(n_features, size=n_informative, replace=False)
    true_coefs = np.zeros(n_features)
    true_coefs[informative_idx] = rng.uniform(-3, 3, size=n_informative)

    y = X @ true_coefs + rng.normal(0, noise_std, size=n_samples)

    df_X = pd.DataFrame(X, columns=[f"series_{i+1}" for i in range(n_features)])
    return df_X, pd.Series(y, name="target"), true_coefs, informative_idx


def run_lasso_regression(X_df: pd.DataFrame, y_series: pd.Series, alpha: float = 0.1) -> pd.DataFrame:
    """Runs LASSO regression using R's glmnet package via rpy2."""
    logging.debug("--- Starting R LASSO Regression ---")
    try:
        with openrlib.rlock:
            with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
                ro_globalenv["py_X"] = X_df.values
                ro_globalenv["py_y"] = y_series.values
                ro_globalenv["py_alpha"] = float(alpha)

                r_script = """
                library(glmnet)
                X_mat <- as.matrix(py_X)
                y_vec <- as.numeric(py_y)
                fit <- glmnet(X_mat, y_vec, alpha = 1, family = "gaussian", standardize = FALSE)
                coef_sparse <- coef(fit, s = py_alpha)
                coef_matrix <- as.matrix(coef_sparse)
                coef_matrix
                """

                logging.debug("Executing glmnet in R...")
                r_coefs = ro_r(r_script)

                features = ["Intercept"] + X_df.columns.tolist()
                coefs_df = pd.DataFrame(r_coefs, index=features, columns=["Coefficient"])
                logging.debug("LASSO Regression finished successfully.")
                return coefs_df

    except Exception as e:
        logging.error(f"AN ERROR OCCURRED during R LASSO execution: {e}", exc_info=True)
        return None


def main():
    # 1. Simulate many X series
    n_samples = 500
    n_features = 50
    n_informative = 5
    alpha = 0.1

    X, y_target, true_coefs, informative_idx = simulate_many_series(
        n_samples=n_samples, n_features=n_features, n_informative=n_informative, noise_std=1.0, seed=42
    )
    print(f"Informative series: {[f'series_{i+1}' for i in informative_idx]}")

    # 2. Train/test split
    split_idx = int(0.8 * len(X))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y_target.iloc[:split_idx], y_target.iloc[split_idx:]

    # 3. Fit LASSO via R/glmnet
    coefs_df = run_lasso_regression(X_train, y_train, alpha=alpha)
    if coefs_df is None:
        print("R LASSO execution failed.")
        return

    # 4. Predict and evaluate in Python using returned coefficients
    intercept = coefs_df.loc["Intercept", "Coefficient"]
    beta = coefs_df.loc[X_train.columns, "Coefficient"].values
    y_pred = intercept + X_test.values @ beta

    nonzero = np.count_nonzero(beta)
    print(f"Non-zero coefficients: {nonzero}/{n_features}")
    print(f"Intercept: {intercept:.4f}")
    print(f"MSE:  {mean_squared_error(y_test, y_pred):.4f}")
    print(f"MAE:  {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"R^2:  {r2_score(y_test, y_pred):.4f}")

    # 5. Check recovery of true features
    selected_idx = np.where(beta != 0)[0]
    true_set = set(informative_idx)
    selected_set = set(selected_idx)
    print(f"True positives: {len(true_set & selected_set)}/{n_informative}")
    print(f"False positives: {len(selected_set - true_set)}")

    print("\nTop 5 estimated coefficients by absolute value:")
    print(coefs_df.reindex(coefs_df["Coefficient"].abs().sort_values(ascending=False).index).head())


if __name__ == "__main__":
    main()
