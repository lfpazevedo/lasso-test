import logging
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from rpy2.robjects import globalenv as ro_globalenv, r as ro_r
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import numpy2ri, pandas2ri
import rpy2.robjects as ro
from rpy2.rinterface_lib import openrlib

logging.basicConfig(level=logging.WARNING)


def simulate_time_series(n_samples=1000, trend=0.02, seasonality_period=50, noise_std=2.0, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples)
    trend_component = trend * t
    seasonal_component = 10 * np.sin(2 * np.pi * t / seasonality_period)
    noise = rng.normal(0, noise_std, size=n_samples)
    y = trend_component + seasonal_component + noise
    return y


def build_lag_features_df(y, n_lags=20):
    X, targets = [], []
    for i in range(n_lags, len(y)):
        X.append(y[i - n_lags : i])
        targets.append(y[i])
    columns = [f"lag_{i+1}" for i in range(n_lags)]
    return pd.DataFrame(np.array(X), columns=columns), pd.Series(np.array(targets), name="target")


def run_sklearn_lasso(X_train, y_train, X_test, y_test, alpha=0.1):
    model = Lasso(alpha=alpha, max_iter=10000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    coefs = pd.Series(model.coef_, index=X_train.columns, name="Coefficient")
    coefs["Intercept"] = model.intercept_
    return {
        "model": model,
        "coefs": coefs,
        "pred": y_pred,
        "mse": mean_squared_error(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
        "nonzero": np.count_nonzero(model.coef_),
    }


def run_r_glmnet(X_df: pd.DataFrame, y_series: pd.Series, alpha: float):
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
                # match sklearn default behavior (no standardize)
                fit <- glmnet(X_mat, y_vec, alpha = 1, family = "gaussian", standardize = FALSE)
                coef_sparse <- coef(fit, s = py_alpha)
                coef_matrix <- as.matrix(coef_sparse)
                list(coef = coef_matrix)
                """

                r_result = ro_r(r_script)
                r_coefs = r_result["coef"]

                features = ["Intercept"] + X_df.columns.tolist()
                coefs = pd.Series(r_coefs.flatten(), index=features, name="Coefficient")
                return {"coefs": coefs, "lambda": alpha}
    except Exception as e:
        logging.error("R glmnet execution failed", exc_info=True)
        st.error(f"R glmnet execution failed: {e}")
        return None


def evaluate_r_model(coefs, X_test, y_test):
    intercept = coefs["Intercept"]
    beta = coefs[X_test.columns].values
    y_pred = intercept + X_test.values @ beta
    return {
        "pred": y_pred,
        "mse": mean_squared_error(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
        "nonzero": np.count_nonzero(beta),
    }


def main():
    st.set_page_config(page_title="Lasso Comparison", layout="wide")
    st.title("🧪 Lasso Coefficient Comparison")
    st.markdown("**scikit-learn** vs **R `glmnet` (via rpy2)** on simulated time-series lag features")

    with st.sidebar:
        st.header("⚙️ Simulation Settings")
        n_samples = st.slider("Samples", 200, 2000, 1000, 100)
        n_lags = st.slider("Lag features", 5, 50, 20, 1)
        noise_std = st.slider("Noise σ", 0.0, 10.0, 2.0, 0.5)
        trend = st.number_input("Trend", value=0.02, step=0.01)
        seasonality_period = st.number_input("Seasonality period", value=50, step=1)
        seed = st.number_input("Random seed", value=42, step=1)

        st.header("🔧 Shared Settings")
        lasso_alpha = st.number_input("Penalty (alpha / lambda)", value=0.1, step=0.05, min_value=0.0, format="%.3f")

        run_btn = st.button("▶️ Run Models", type="primary")

    if not run_btn:
        st.info("Click **Run Models** in the sidebar to start.")
        return

    with st.spinner("Simulating data & fitting models..."):
        y = simulate_time_series(n_samples, trend, seasonality_period, noise_std, seed)
        X, y_target = build_lag_features_df(y, n_lags)
        split_idx = int(0.8 * len(X))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y_target.iloc[:split_idx], y_target.iloc[split_idx:]

        sk_result = run_sklearn_lasso(X_train, y_train, X_test, y_test, lasso_alpha)
        r_result = run_r_glmnet(X_train, y_train, lasso_alpha)
        if r_result is None:
            return
        r_metrics = evaluate_r_model(r_result["coefs"], X_test, y_test)

    # Metrics
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🐍 scikit-learn Lasso")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MSE", f"{sk_result['mse']:.4f}")
        m2.metric("MAE", f"{sk_result['mae']:.4f}")
        m3.metric("R²", f"{sk_result['r2']:.4f}")
        m4.metric("Non-zero", f"{sk_result['nonzero']}/{n_lags}")
    with col2:
        st.subheader("🇷 R glmnet (lambda = Penalty)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MSE", f"{r_metrics['mse']:.4f}")
        m2.metric("MAE", f"{r_metrics['mae']:.4f}")
        m3.metric("R²", f"{r_metrics['r2']:.4f}")
        m4.metric("Non-zero", f"{r_metrics['nonzero']}/{n_lags}")
        st.caption(f"λ = {r_result['lambda']:.4f}")

    # Coefficient comparison chart
    st.divider()
    st.subheader("📊 Coefficient Comparison")

    comp_df = pd.DataFrame({
        "Feature": sk_result["coefs"].index,
        "sklearn": sk_result["coefs"].values,
        "glmnet": r_result["coefs"].values,
    })
    # Sort by absolute sklearn coefficient for nicer visuals
    comp_df = comp_df.reindex(comp_df["sklearn"].abs().sort_values(ascending=False).index)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=comp_df["Feature"],
        y=comp_df["sklearn"],
        name="scikit-learn",
        marker_color="#636EFA",
    ))
    fig.add_trace(go.Bar(
        x=comp_df["Feature"],
        y=comp_df["glmnet"],
        name="R glmnet",
        marker_color="#EF553B",
    ))
    fig.update_layout(
        barmode="group",
        xaxis_title="Feature",
        yaxis_title="Coefficient",
        template="plotly_white",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Table
    st.subheader("📋 Coefficient Table")
    comp_df["Difference"] = comp_df["sklearn"] - comp_df["glmnet"]
    st.dataframe(comp_df.style.format({"sklearn": "{:,.4f}", "glmnet": "{:,.4f}", "Difference": "{:,.4f}"}), use_container_width=True)

    # Prediction preview
    st.divider()
    st.subheader("🔮 Prediction Preview (first 20 test points)")
    preview = pd.DataFrame({
        "Actual": y_test.values[:20],
        "sklearn": sk_result["pred"][:20],
        "glmnet": r_metrics["pred"][:20],
    })
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(y=preview["Actual"], mode="lines+markers", name="Actual", line=dict(color="black", width=2)))
    fig2.add_trace(go.Scatter(y=preview["sklearn"], mode="lines+markers", name="scikit-learn", line=dict(color="#636EFA", width=2)))
    fig2.add_trace(go.Scatter(y=preview["glmnet"], mode="lines+markers", name="R glmnet", line=dict(color="#EF553B", width=2)))
    fig2.update_layout(template="plotly_white", height=400, xaxis_title="Index", yaxis_title="Value")
    st.plotly_chart(fig2, use_container_width=True)


if __name__ == "__main__":
    main()
