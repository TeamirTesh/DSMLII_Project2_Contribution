"""Makes the report figures from results/*.csv and writes them to report/figures/."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from load_autompg import DATA_DIR  # noqa: E402

RESULTS = DATA_DIR.parent / "results"
FIGDIR = DATA_DIR.parent / "report" / "figures"
FEATURES = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year"]
COLORS = dict(zip(FEATURES, ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2"]))
SOFTWARE = [("statsmodels", "statsmodels"), ("scalation", "ScalaTion")]

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150})


def chosen_lambda(software: str, model: str) -> float:
    return float(pd.read_csv(RESULTS / f"{software}_summary.csv").set_index("model").loc[model, "lambda_equiv"])


def cv_curves() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, model in zip(axes, ("Ridge", "Lasso")):
        for (key, label), style in zip(SOFTWARE, ("-", "--")):
            cv = pd.read_csv(RESULTS / f"{key}_cv_curve.csv")
            col = f"{model.lower()}_cv_rmse"
            ax.plot(cv["lambda_equiv"], cv[col], style, label=label, lw=1.6)
            i = cv[col].idxmin()
            ax.plot(cv["lambda_equiv"][i], cv[col][i], "o", ms=5, color=ax.lines[-1].get_color())
        ols = pd.read_csv(RESULTS / "statsmodels_summary.csv").set_index("model").loc["OLS", "cv_rmse"]
        ax.axhline(ols, color="gray", lw=0.9, ls=":", label="OLS (no penalty)")
        ax.set_xscale("log")
        ax.set_xlabel(r"penalty $\lambda = n_{train}\,\alpha$")
        ax.set_title(model)
    axes[0].set_ylabel("5-fold CV RMSE (mpg)")
    axes[0].legend(frameon=False, fontsize=8)
    ymin, ymax = axes[0].get_ylim()
    axes[0].set_ylim(ymin, min(ymax, 4.6))
    fig.tight_layout()
    fig.savefig(FIGDIR / "cv_curves.pdf")
    plt.close(fig)


def coefficient_paths() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True, sharey="row")
    for r, model in enumerate(("Ridge", "Lasso")):
        for c, (key, label) in enumerate(SOFTWARE):
            ax = axes[r, c]
            path = pd.read_csv(RESULTS / f"{key}_path_{model.lower()}.csv")
            for feat in FEATURES:
                ax.plot(path["lambda_equiv"], path[feat], color=COLORS[feat], lw=1.5, label=feat)
            ax.axvline(chosen_lambda(key, model), color="black", ls="--", lw=0.9)
            ax.axhline(0, color="gray", lw=0.6)
            ax.set_xscale("log")
            ax.set_title(f"{model} - {label}")
            if r == 1:
                ax.set_xlabel(r"penalty $\lambda = n_{train}\,\alpha$")
        axes[r, 0].set_ylabel("standardized coefficient")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIGDIR / "coefficient_paths.pdf")
    plt.close(fig)


def lasso_zoom() -> None:
    small = ["cylinders", "displacement", "horsepower", "acceleration"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, (key, label) in zip(axes, SOFTWARE):
        path = pd.read_csv(RESULTS / f"{key}_path_lasso.csv")
        for feat in small:
            ax.plot(path["lambda_equiv"], path[feat], color=COLORS[feat], lw=1.5, label=feat)
        ax.axvline(chosen_lambda(key, "Lasso"), color="black", ls="--", lw=0.9)
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_xscale("log")
        ax.set_xlim(0.1, 3000)
        ax.set_title(f"Lasso, small predictors - {label}")
        ax.set_xlabel(r"penalty $\lambda = n_{train}\,\alpha$")
    axes[0].set_ylabel("standardized coefficient")
    axes[0].legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGDIR / "lasso_zoom.pdf")
    plt.close(fig)


def coefficient_bars() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    width = 0.26
    for ax, (key, label) in zip(axes, SOFTWARE):
        coef = pd.read_csv(RESULTS / f"{key}_coefficients_standardized.csv", index_col="term").drop(index="intercept")
        x = np.arange(len(coef))
        for k, (model, color) in enumerate(zip(("OLS", "Ridge", "Lasso"), ("#9d9d9d", "#4c78a8", "#e45756"))):
            ax.bar(x + (k - 1) * width, coef[model], width, label=model, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(coef.index, fontsize=7, rotation=35, ha="right")
        ax.axhline(0, color="black", lw=0.7)
        ax.set_title(label)
    axes[0].set_ylabel("standardized coefficient")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "coefficient_bars.pdf")
    plt.close(fig)


if __name__ == "__main__":
    FIGDIR.mkdir(parents=True, exist_ok=True)
    cv_curves()
    coefficient_paths()
    lasso_zoom()
    coefficient_bars()
    print(f"figures written to {FIGDIR}")
