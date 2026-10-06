# Ridge and Lasso on AutoMPG

Ridge and Lasso on the UCI AutoMPG data in ScalaTion and in statsmodels. Each package also has an OLS fit for comparison.

```
python/      statsmodels code, comparison, figure and table scripts, unit tests
scalation/   ScalaTion program (RegularizedRegression_AutoMPG.scala)
data/        autompg_clean.csv (392 rows), test_indices.csv and cv_folds.csv (shared split and folds)
results/     CSV results and run logs from both packages
report/      main.tex, figures/, tables/, main.pdf
```

## Run the statsmodels part (Python 3.12)

```bash
cd python
pip install -r requirements.txt
python regularized_regression.py        # writes results/statsmodels_*.csv, data/test_indices.csv, data/cv_folds.csv
python -m unittest discover -s tests -v # 11 tests
```

## Run the ScalaTion part (JDK 21, sbt, Scala 3.8.3)

Run the Python script first. The ScalaTion program reads `data/test_indices.csv` and `data/cv_folds.csv`.

```bash
git clone https://github.com/scalation/scalation_2.0
cp scalation/RegularizedRegression_AutoMPG.scala scalation_2.0/src/main/scala/scalation/modeling/
cd scalation_2.0
P2_DIR=<full path to this Project2 folder> sbt "runMain scalation.modeling.regularizedRegression_AutoMPG"
```

## Make the comparison, figures, tables and PDF

```bash
cd python
python compare_results.py
python make_figures.py
python make_tables.py
cd ../report && tectonic main.tex    # or run pdflatex twice
```

## Notes

- `origin` is not used as a predictor, same as ScalaTion's `Example_AutoMPG`. That leaves 6 predictors.
- The two packages use different penalty scales. They match when `lambda = n_train * alpha`.
- The statsmodels Lasso solver can stop too early, so `fit_lasso` restarts it until the answer stops changing. The tests check it against scikit-learn.
- ScalaTion's `LassoAdmm` has loose fixed tolerances, so its Lasso coefficients can be off by about 0.07 (scaled units).
