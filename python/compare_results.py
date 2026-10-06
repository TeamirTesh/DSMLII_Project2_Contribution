"""Compare the ScalaTion and statsmodels results (run both of them first).

Writes results/software_comparison.csv:
  - path_max_abs_diff and path_median_abs_diff: |ScalaTion coef - statsmodels coef|
    on the scaled predictors, over the 60 lambda values. The penalty is the same
    at every point, so a gap comes from the solver, not from tuning.
  - final_*: the same gap between the two final tuned models.
"""

import numpy as np
import pandas as pd

import regularized_regression as rr
from load_autompg import DATA_DIR

RESULTS = DATA_DIR.parent / "results"


def main() -> None:
    rows = []
    for model in ("ridge", "lasso"):
        sm_path = pd.read_csv(RESULTS / f"statsmodels_path_{model}.csv")
        st_path = pd.read_csv(RESULTS / f"scalation_path_{model}.csv")
        assert np.allclose(sm_path["alpha"], st_path["alpha"]), "alpha grids differ"
        gap = (sm_path.iloc[:, 2:] - st_path.iloc[:, 2:]).abs().max(axis=1)
        rows.append({"model": model.capitalize(), "metric": "path_max_abs_diff", "value": gap.max()})
        rows.append({"model": model.capitalize(), "metric": "path_median_abs_diff", "value": gap.median()})

    sm_c = pd.read_csv(RESULTS / "statsmodels_coefficients_standardized.csv", index_col="term")
    st_c = pd.read_csv(RESULTS / "scalation_coefficients_standardized.csv", index_col="term")
    sm_s = pd.read_csv(RESULTS / "statsmodels_summary.csv").set_index("model")
    st_s = pd.read_csv(RESULTS / "scalation_summary.csv").set_index("model")
    for model in ("OLS", "Ridge", "Lasso"):
        rows.append({"model": model, "metric": "final_max_abs_coef_diff",
                     "value": (sm_c[model] - st_c[model]).abs().max()})
        rows.append({"model": model, "metric": "final_test_rmse_diff",
                     "value": abs(sm_s.loc[model, "test_rmse"] - st_s.loc[model, "test_rmse"])})
        rows.append({"model": model, "metric": "final_test_r2_diff",
                     "value": abs(sm_s.loc[model, "test_r2"] - st_s.loc[model, "test_r2"])})

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "software_comparison.csv", index=False)
    print(out.pivot(index="metric", columns="model", values="value").to_string(float_format=lambda v: f"{v:.3e}"))

    fold_rmse().to_csv(RESULTS / "statsmodels_cv_fold_rmse.csv", index_label="fold")


def fold_rmse() -> pd.DataFrame:
    """CV RMSE of OLS, Ridge and Lasso in each fold, at the alphas statsmodels picked."""
    df = rr.load_autompg_clean()
    X, y = df[rr.FEATURES].to_numpy(float), df[rr.TARGET].to_numpy(float)
    train, _ = rr.split_indices(len(df))
    Xt, yt = X[train], y[train]
    summ = pd.read_csv(RESULTS / "statsmodels_summary.csv").set_index("model")
    out = {"OLS": [], "Ridge": [], "Lasso": []}
    for fold in rr.cv_folds(len(yt)):
        tr = np.setdiff1d(np.arange(len(yt)), fold)
        mu, sd = rr.fit_scaler(Xt[tr])
        A, B = rr.design(Xt[tr], mu, sd), rr.design(Xt[fold], mu, sd)
        fits = {"OLS": rr.fit_ols(A, yt[tr]),
                "Ridge": rr.fit_ridge(A, yt[tr], summ.loc["Ridge", "alpha"]),
                "Lasso": rr.fit_lasso(A, yt[tr], summ.loc["Lasso", "alpha"])}
        for name, b in fits.items():
            out[name].append(float(np.sqrt(np.mean((yt[fold] - B @ b) ** 2))))
    table = pd.DataFrame(out)
    print("\nper-fold CV RMSE (statsmodels):\n", table.round(4).to_string())
    return table


if __name__ == "__main__":
    main()
