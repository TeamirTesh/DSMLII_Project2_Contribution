"""Ridge and Lasso on UCI AutoMPG with statsmodels.

Steps:
  1. Split 80/20 into train and test rows (seeded). The test rows and the 5 CV
     folds are saved to data/test_indices.csv and data/cv_folds.csv so the
     ScalaTion run uses the same split and folds.
  2. Scale the predictors with training-set mean and sample std (ddof=1, same as
     ScalaTion's NormForm). The intercept is never penalized.
  3. Pick alpha with 5-fold CV on the training rows, using OLS.fit_regularized.
  4. Fit OLS, Ridge and Lasso on the training rows and score them on the training
     and test rows.

statsmodels minimizes 0.5*RSS/n + alpha*((1-L1_wt)*|b|_2^2/2 + L1_wt*|b|_1),
with L1_wt=0 for Ridge and L1_wt=1 for Lasso. ScalaTion uses RSS + lam*|b|^2 for
Ridge and 0.5*RSS + lam*|b|_1 for Lasso. The two match when lam = n_train * alpha.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from load_autompg import DATA_DIR, load_autompg_clean

RESULTS_DIR = DATA_DIR.parent / "results"
TARGET = "mpg"
# origin is left out, same as ScalaTion's Example_AutoMPG (it is a category code).
FEATURES = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year"]
ALPHAS = np.logspace(-4, 1, 60)
N_FOLDS = 5
SEED = 42
TEST_FRAC = 0.2


def split_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(SEED)
    test = np.sort(rng.choice(n, size=int(n * TEST_FRAC), replace=False))
    train = np.setdiff1d(np.arange(n), test)
    return train, test


def fit_scaler(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return X.mean(axis=0), X.std(axis=0, ddof=1)


def design(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return sm.add_constant((X - mu) / sd, has_constant="add")


def penalty(alpha: float, p: int) -> np.ndarray:
    return np.r_[0.0, np.full(p, alpha)]


def fit_ols(Xc: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.asarray(sm.OLS(y, Xc).fit().params)


def fit_ridge(Xc: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    res = sm.OLS(y, Xc).fit_regularized(alpha=penalty(alpha, Xc.shape[1] - 1), L1_wt=0.0)
    return np.asarray(res.params)


def fit_lasso(Xc: np.ndarray, y: np.ndarray, alpha: float, max_restarts: int = 100) -> np.ndarray:
    """The statsmodels solver never looks at a coefficient again once it hits zero,
    so one call can stop too early. Each restart checks every coefficient on its
    first pass, so we restart from the last answer until it stops changing."""
    pen = penalty(alpha, Xc.shape[1] - 1)
    b = fit_ols(Xc, y)
    for _ in range(max_restarts):
        res = sm.OLS(y, Xc).fit_regularized(
            alpha=pen, L1_wt=1.0, maxiter=1000, cnvrg_tol=1e-12, start_params=b
        )
        new = np.asarray(res.params)
        if np.max(np.abs(new - b)) < 1e-10:
            return new
        b = new
    raise RuntimeError(f"Lasso did not converge for alpha={alpha}")


FITTERS = {"Ridge": fit_ridge, "Lasso": fit_lasso}


def rmse(y: np.ndarray, yp: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yp) ** 2)))


def r2(y: np.ndarray, yp: np.ndarray) -> float:
    return float(1.0 - np.sum((y - yp) ** 2) / np.sum((y - y.mean()) ** 2))


def cv_folds(n_train: int) -> list[np.ndarray]:
    return np.array_split(np.random.RandomState(SEED).permutation(n_train), N_FOLDS)


def cv_rmse(X: np.ndarray, y: np.ndarray, alpha: float, model: str) -> float:
    mses = []
    for fold in cv_folds(len(y)):
        tr = np.setdiff1d(np.arange(len(y)), fold)
        mu, sd = fit_scaler(X[tr])
        b = FITTERS[model](design(X[tr], mu, sd), y[tr], alpha)
        mses.append(np.mean((y[fold] - design(X[fold], mu, sd) @ b) ** 2))
    return float(np.sqrt(np.mean(mses)))


def to_original_units(b_std: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    slopes = b_std[1:] / sd
    return np.r_[b_std[0] - np.sum(slopes * mu), slopes]


def evaluate(name, b, alpha, n_train, Xtr, ytr, Xte, yte, cv):
    ptr, pte = Xtr @ b, Xte @ b
    return {
        "software": "statsmodels",
        "model": name,
        "alpha": alpha,
        "lambda_equiv": np.nan if alpha is None else n_train * alpha,
        "n_zero_coefs": int(np.sum(np.abs(b[1:]) < 1e-8)),
        "cv_rmse": cv,
        "train_r2": r2(ytr, ptr),
        "train_rmse": rmse(ytr, ptr),
        "test_r2": r2(yte, pte),
        "test_rmse": rmse(yte, pte),
        "test_mae": float(np.mean(np.abs(yte - pte))),
    }


def main() -> None:
    df = load_autompg_clean()
    X, y = df[FEATURES].to_numpy(float), df[TARGET].to_numpy(float)
    train, test = split_indices(len(df))
    pd.DataFrame({"row": test}).to_csv(DATA_DIR / "test_indices.csv", index=False)
    fold_of = np.empty(len(train), dtype=int)
    for k, idx in enumerate(cv_folds(len(train))):
        fold_of[idx] = k
    pd.DataFrame({"train_pos": np.arange(len(train)), "row": train, "fold": fold_of}).to_csv(
        DATA_DIR / "cv_folds.csv", index=False)
    Xtr_raw, ytr, Xte_raw, yte = X[train], y[train], X[test], y[test]
    n_train = len(train)

    mu, sd = fit_scaler(Xtr_raw)
    Xtr, Xte = design(Xtr_raw, mu, sd), design(Xte_raw, mu, sd)
    names = ["intercept"] + FEATURES

    cv_curve = pd.DataFrame({"alpha": ALPHAS, "lambda_equiv": n_train * ALPHAS})
    for model in FITTERS:
        cv_curve[f"{model.lower()}_cv_rmse"] = [cv_rmse(Xtr_raw, ytr, a, model) for a in ALPHAS]
    best = {m: float(ALPHAS[cv_curve[f"{m.lower()}_cv_rmse"].idxmin()]) for m in FITTERS}

    coefs = {"OLS": fit_ols(Xtr, ytr)}
    for model, fitter in FITTERS.items():
        coefs[model] = fitter(Xtr, ytr, best[model])

    rows = []
    for model, b in coefs.items():
        alpha = best.get(model)
        cv = float(cv_curve[f"{model.lower()}_cv_rmse"].min()) if alpha else cv_rmse(Xtr_raw, ytr, 0.0, "Ridge")
        rows.append(evaluate(model, b, alpha, n_train, Xtr, ytr, Xte, yte, cv))
    summary = pd.DataFrame(rows)

    coef_std = pd.DataFrame(coefs, index=names)
    coef_orig = pd.DataFrame({m: to_original_units(b, mu, sd) for m, b in coefs.items()}, index=names)

    selected = [0] + [j + 1 for j in range(len(FEATURES)) if abs(coefs["Lasso"][j + 1]) > 1e-8]
    post = sm.OLS(ytr, Xtr[:, selected]).fit()
    ci = post.conf_int()
    post_table = pd.DataFrame(
        {"coef": post.params, "std_err": post.bse, "t": post.tvalues, "p_value": post.pvalues,
         "ci_low": ci[:, 0], "ci_high": ci[:, 1]},
        index=[names[j] for j in selected],
    )

    paths = {}
    for model, fitter in FITTERS.items():
        path = pd.DataFrame([fitter(Xtr, ytr, a)[1:] for a in ALPHAS], columns=FEATURES)
        path.insert(0, "lambda_equiv", n_train * ALPHAS)
        path.insert(0, "alpha", ALPHAS)
        paths[model] = path

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS_DIR / "statsmodels_summary.csv", index=False)
    coef_std.to_csv(RESULTS_DIR / "statsmodels_coefficients_standardized.csv", index_label="term")
    coef_orig.to_csv(RESULTS_DIR / "statsmodels_coefficients_original_units.csv", index_label="term")
    cv_curve.to_csv(RESULTS_DIR / "statsmodels_cv_curve.csv", index=False)
    post_table.to_csv(RESULTS_DIR / "statsmodels_lasso_selected_ols.csv", index_label="term")
    for model, path in paths.items():
        path.to_csv(RESULTS_DIR / f"statsmodels_path_{model.lower()}.csv", index=False)
    (RESULTS_DIR / "statsmodels_ols_summary.txt").write_text(
        str(sm.OLS(ytr, Xtr).fit().summary(xname=names, yname=TARGET))
    )

    pd.set_option("display.width", 200)
    print(f"train rows = {n_train}, test rows = {len(test)}")
    print(f"selected alpha: Ridge = {best['Ridge']:.5f} (lambda_equiv {n_train * best['Ridge']:.3f}), "
          f"Lasso = {best['Lasso']:.5f} (lambda_equiv {n_train * best['Lasso']:.3f})")
    print("\nStandardized coefficients (training set):")
    print(coef_std.round(4).to_string())
    print("\nCoefficients in original units:")
    print(coef_orig.round(5).to_string())
    print("\nSummary:")
    print(summary.drop(columns=["software"]).round(4).to_string(index=False))
    print("\nOLS refit on the Lasso-selected variables (inference is post-selection, optimistic):")
    print(post_table.round(4).to_string())


if __name__ == "__main__":
    main()
