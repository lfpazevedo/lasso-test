import logging
import numpy as np
import pandas as pd
from rpy2.robjects import globalenv as ro_globalenv, r as ro_r
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import numpy2ri, pandas2ri
import rpy2.robjects as ro
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")


def simulate_time_series(n_samples=1000, trend=0.02, seasonality_period=50, noise_std=2.0, seed=42):
    """Simulate a time series with trend, seasonality, and Gaussian noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples)
    trend_component = trend * t
    seasonal_component = 10 * np.sin(2 * np.pi * t / seasonality_period)
    noise = rng.normal(0, noise_std, size=n_samples)
    y = trend_component + seasonal_component + noise
    return y


def build_lag_features_df(y, n_lags=20):
    """Build supervised-learning dataset from time-series using lag features."""
    X, targets = [], []
    for i in range(n_lags, len(y)):
        X.append(y[i - n_lags : i])
        targets.append(y[i])
    columns = [f"lag_{i+1}" for i in range(n_lags)]
    return pd.DataFrame(np.array(X), columns=columns), pd.Series(np.array(targets), name="target")


def run_lasso_regression(X_df: pd.DataFrame, y_series: pd.Series) -> pd.DataFrame:
    """Runs LASSO regression using R's glmnet package via rpy2."""
    logging.debug("--- Starting R LASSO Regression ---")
    try:
        with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
            ro_globalenv["py_X"] = X_df.values
            ro_globalenv["py_y"] = y_series.values

            r_script = """
            library(glmnet)

            X_mat <- as.matrix(py_X)
            y_vec <- as.numeric(py_y)

            cv_fit <- cv.glmnet(X_mat, y_vec,
                                alpha = 1,
                                family = "gaussian",
                                standardize = TRUE,
                                nfolds = 10)

            coef_sparse <- coef(cv_fit, s = "lambda.1se")
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
    # 1. Simulate data
    y = simulate_time_series(n_samples=1000, trend=0.02, seasonality_period=50, noise_std=2.0, seed=42)

    # 2. Build lag features as a DataFrame
    n_lags = 20
    X, y_target = build_lag_features_df(y, n_lags=n_lags)

    # 3. Train/test split (respect temporal order)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y_target.iloc[:split_idx], y_target.iloc[split_idx:]

    # 4. Fit LASSO via R/glmnet
    coefs_df = run_lasso_regression(X_train, y_train)
    if coefs_df is None:
        print("R LASSO execution failed.")
        return

    # 5. Predict and evaluate in Python using returned coefficients
    intercept = coefs_df.loc["Intercept", "Coefficient"]
    beta = coefs_df.loc[X_train.columns, "Coefficient"].values
    y_pred = intercept + X_test.values @ beta

    nonzero = np.count_nonzero(beta)
    print(f"Non-zero coefficients: {nonzero}/{n_lags}")
    print(f"Intercept: {intercept:.4f}")
    print(f"MSE:  {mean_squared_error(y_test, y_pred):.4f}")
    print(f"MAE:  {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"R^2:  {r2_score(y_test, y_pred):.4f}")

    print("\nFirst 5 predictions vs actual:")
    for pred, actual in zip(y_pred[:5], y_test.values[:5]):
        print(f"  Pred: {pred:8.3f} | Actual: {actual:8.3f}")

    # Optional: print a few coefficients
    print("\nTop 5 coefficients by absolute value:")
    print(coefs_df.reindex(coefs_df["Coefficient"].abs().sort_values(ascending=False).index).head())


if __name__ == "__main__":
    main()
