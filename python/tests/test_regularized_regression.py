"""Run from Project2/python:  python -m unittest discover -s tests -v"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from load_autompg import load_autompg_clean  # noqa: E402
from regularized_regression import (  # noqa: E402
    ALPHAS, FEATURES, TARGET, design, fit_lasso, fit_ols, fit_ridge, fit_scaler,
    split_indices, to_original_units,
)


class RegularizedRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = load_autompg_clean()
        cls.df = df
        cls.X = df[FEATURES].to_numpy(float)
        cls.y = df[TARGET].to_numpy(float)
        cls.train, cls.test = split_indices(len(df))
        cls.Xr, cls.yt = cls.X[cls.train], cls.y[cls.train]
        cls.mu, cls.sd = fit_scaler(cls.Xr)
        cls.Xc = design(cls.Xr, cls.mu, cls.sd)
        cls.n = len(cls.yt)

    def test_dataset_shape_and_cleanliness(self):
        self.assertEqual(self.df.shape[0], 392)
        self.assertFalse(self.df[FEATURES + [TARGET]].isna().any().any())
        self.assertAlmostEqual(self.y.sum(), 9190.8, places=6)

    def test_split_is_disjoint_complete_and_deterministic(self):
        self.assertEqual((len(self.train), len(self.test)), (314, 78))
        self.assertEqual(len(np.intersect1d(self.train, self.test)), 0)
        self.assertEqual(len(np.union1d(self.train, self.test)), 392)
        again_train, again_test = split_indices(392)
        np.testing.assert_array_equal(self.test, again_test)

    def test_standardized_columns(self):
        np.testing.assert_allclose(self.Xc[:, 1:].mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(self.Xc[:, 1:].std(axis=0, ddof=1), 1.0, atol=1e-10)

    def test_ridge_matches_closed_form(self):
        for alpha in (0.001, 0.05, 1.0):
            D = np.diag(np.r_[0.0, np.ones(len(FEATURES))])
            expected = np.linalg.solve(self.Xc.T @ self.Xc + self.n * alpha * D, self.Xc.T @ self.yt)
            np.testing.assert_allclose(fit_ridge(self.Xc, self.yt, alpha), expected, atol=1e-7)

    def test_ridge_matches_sklearn(self):
        from sklearn.linear_model import Ridge
        alpha = 0.3
        sk = Ridge(alpha=self.n * alpha).fit(self.Xc[:, 1:], self.yt)
        np.testing.assert_allclose(fit_ridge(self.Xc, self.yt, alpha),
                                   np.r_[sk.intercept_, sk.coef_], atol=1e-6)

    def test_lasso_matches_sklearn_across_the_alpha_grid(self):
        from sklearn.linear_model import Lasso
        for alpha in ALPHAS[::6]:
            sk = Lasso(alpha=alpha, tol=1e-14, max_iter=10**6).fit(self.Xc[:, 1:], self.yt)
            np.testing.assert_allclose(fit_lasso(self.Xc, self.yt, alpha),
                                       np.r_[sk.intercept_, sk.coef_], atol=1e-6,
                                       err_msg=f"alpha={alpha}")

    def test_lasso_satisfies_kkt_conditions(self):
        for alpha in (0.01, 0.05, 0.2, 0.5):
            b = fit_lasso(self.Xc, self.yt, alpha)
            grad = -(self.Xc.T @ (self.yt - self.Xc @ b)) / self.n
            self.assertAlmostEqual(grad[0], 0.0, places=7)
            for j in range(1, len(b)):
                if abs(b[j]) > 1e-8:
                    self.assertAlmostEqual(grad[j], -alpha * np.sign(b[j]), places=6)
                else:
                    self.assertLessEqual(abs(grad[j]), alpha + 1e-6)

    def test_intercept_is_not_penalized(self):
        for fit, alpha in ((fit_ridge, 5.0), (fit_lasso, 0.3)):
            self.assertAlmostEqual(fit(self.Xc, self.yt, alpha)[0], self.yt.mean(), places=8)

    def test_tiny_penalty_recovers_ols(self):
        np.testing.assert_allclose(fit_ridge(self.Xc, self.yt, 1e-10), fit_ols(self.Xc, self.yt), atol=1e-5)

    def test_huge_lasso_penalty_zeroes_every_slope(self):
        b = fit_lasso(self.Xc, self.yt, 1e3)
        np.testing.assert_allclose(b[1:], 0.0, atol=1e-12)

    def test_original_unit_coefficients_give_identical_predictions(self):
        b_std = fit_ridge(self.Xc, self.yt, 0.05)
        b_orig = to_original_units(b_std, self.mu, self.sd)
        raw_design = np.c_[np.ones(len(self.Xr)), self.Xr]
        np.testing.assert_allclose(raw_design @ b_orig, self.Xc @ b_std, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
