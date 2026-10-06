"""Makes the LaTeX tables from results/*.csv and writes them to report/tables/."""

import pandas as pd

from load_autompg import DATA_DIR

RESULTS = DATA_DIR.parent / "results"
TABDIR = DATA_DIR.parent / "report" / "tables"
SOFTWARE = [("statsmodels", "statsmodels"), ("scalation", "ScalaTion")]
MODELS = ["OLS", "Ridge", "Lasso"]


def fmt(v: float, nd: int = 4) -> str:
    return "n/a" if pd.isna(v) else f"{v:.{nd}f}".replace("-", "$-$")


def tex_name(term: str) -> str:
    return term.replace("_", r"\_")


def write(name: str, text: str) -> None:
    (TABDIR / name).write_text(text, encoding="utf-8")


def summary_table() -> None:
    lines = [r"\begin{tabular}{@{}llrrrrrrr@{}}", r"\toprule",
             r"Software & Model & $\lambda$ & Zero coef. & CV RMSE & Train $R^2$ & Test $R^2$ & Test RMSE & Test MAE \\",
             r"\midrule"]
    for key, label in SOFTWARE:
        s = pd.read_csv(RESULTS / f"{key}_summary.csv").set_index("model")
        for m in MODELS:
            r = s.loc[m]
            lines.append(
                f"{label} & {m} & {fmt(r['lambda_equiv'], 2)} & {int(r['n_zero_coefs'])}/6 & {fmt(r['cv_rmse'])} & "
                f"{fmt(r['train_r2'])} & {fmt(r['test_r2'])} & {fmt(r['test_rmse'])} & {fmt(r['test_mae'])} \\\\")
        if key != SOFTWARE[-1][0]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("summary.tex", "\n".join(lines))


def coefficient_table(suffix: str, nd: int) -> None:
    tabs = {key: pd.read_csv(RESULTS / f"{key}_coefficients_{suffix}.csv", index_col="term") for key, _ in SOFTWARE}
    lines = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r" & \multicolumn{3}{c}{statsmodels} & \multicolumn{3}{c}{ScalaTion} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Term & OLS & Ridge & Lasso & OLS & Ridge & Lasso \\", r"\midrule"]
    for term in tabs["statsmodels"].index:
        cells = [fmt(tabs[key].loc[term, m], nd) for key, _ in SOFTWARE for m in MODELS]
        lines.append(f"{tex_name(term)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(f"coefficients_{suffix}.tex", "\n".join(lines))


def lasso_refit_table() -> None:
    t = pd.read_csv(RESULTS / "statsmodels_lasso_selected_ols.csv", index_col="term")
    lines = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r"Term & Coef. & Std.\ err. & $t$ & $p$-value & 95\% CI low & 95\% CI high \\", r"\midrule"]
    for term, r in t.iterrows():
        lines.append(f"{tex_name(term)} & {fmt(r['coef'])} & {fmt(r['std_err'])} & {fmt(r['t'], 2)} & "
                     f"{fmt(r['p_value'])} & {fmt(r['ci_low'])} & {fmt(r['ci_high'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("lasso_refit.tex", "\n".join(lines))


def comparison_table() -> None:
    c = pd.read_csv(RESULTS / "software_comparison.csv")
    piv = c.pivot(index="metric", columns="model", values="value")
    labels = [("path_max_abs_diff", r"Max coefficient gap over the whole $\lambda$ grid"),
              ("path_median_abs_diff", r"Median coefficient gap over the grid"),
              ("final_max_abs_coef_diff", r"Max coefficient gap, tuned final models"),
              ("final_test_rmse_diff", r"Test-RMSE gap, tuned final models")]
    lines = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule", r"Quantity & OLS & Ridge & Lasso \\", r"\midrule"]
    for key, label in labels:
        cells = [("n/a" if pd.isna(piv.loc[key, m]) else f"{piv.loc[key, m]:.1e}") for m in MODELS]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("comparison.tex", "\n".join(lines))


def data_summary_table() -> None:
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    import regularized_regression as rr

    df = rr.load_autompg_clean()
    train, _ = rr.split_indices(len(df))
    mu, sd = rr.fit_scaler(df[rr.FEATURES].to_numpy(float)[train])
    z = rr.design(df[rr.FEATURES].to_numpy(float)[train], mu, sd)
    lines = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
             r"Variable & Mean & Std.\ dev. & Min & Max & VIF \\", r"\midrule"]
    r = df[rr.TARGET]
    lines.append(f"mpg (response) & {r.mean():.2f} & {r.std():.2f} & {r.min():.1f} & {r.max():.1f} & n/a \\\\")
    for j, feat in enumerate(rr.FEATURES):
        c = df[feat]
        lines.append(f"{tex_name(feat)} & {c.mean():.2f} & {c.std():.2f} & {c.min():.1f} & {c.max():.1f} & "
                     f"{variance_inflation_factor(z, j + 1):.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("data_summary.tex", "\n".join(lines))


def scalation_native_tables() -> None:
    """Reads ScalaTion's own fit numbers and console output from results/scalation_run_log.txt."""
    import re

    text = (RESULTS / "scalation_run_log.txt").read_text(encoding="utf-8")
    blocks = re.split(r"\n(?=\| (?:RidgeRegression|LassoRegression|Regression) )", text)
    blocks = [b for b in blocks if b.startswith("| ") and "fitMap" in b]
    labels = {"RidgeRegression": "Ridge", "LassoRegression": "Lasso", "Regression": "OLS"}
    lines = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r"Model & $R^2$ & Adj.\ $R^2$ & SSE & RMSE & MAE & $m$ \\", r"\midrule"]
    excerpt = []
    for block in blocks:
        kind = re.match(r"\| (\w+)", block).group(1)
        qof = dict(re.findall(r"(\w+) -> (-?[\d.]+)", re.search(r"fitMap\s+qof = LinkedHashMap\((.*)\)", block).group(1)))
        lines.append(f"{labels[kind]} & {float(qof['rSq']):.4f} & {float(qof['rSqBar']):.4f} & {float(qof['sse']):.1f} & "
                     f"{float(qof['rmse']):.4f} & {float(qof['mae']):.4f} & {int(float(qof['m']))} \\\\")
        for ln in block.splitlines():
            if any(k in ln for k in ("hparameter", "features", "fitMap", "[Ljava", "fname =", "ERROR")):
                continue
            if ln.strip() and set(ln.strip()) != {"-"}:
                excerpt.append(re.sub(r"\s*\t+\s*", "  ", ln.rstrip()))
            if "F-statistic" in ln:
                break
        excerpt.append("")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("scalation_native_qof.tex", "\n".join(lines))
    write("scalation_excerpt.txt", "\n".join(excerpt))


if __name__ == "__main__":
    TABDIR.mkdir(parents=True, exist_ok=True)
    scalation_native_tables()
    data_summary_table()
    summary_table()
    coefficient_table("standardized", 4)
    coefficient_table("original_units", 5)
    lasso_refit_table()
    comparison_table()
    print(f"tables written to {TABDIR}")
