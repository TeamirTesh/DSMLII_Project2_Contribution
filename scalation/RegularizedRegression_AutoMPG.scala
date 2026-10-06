//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Ridge and Lasso on the UCI AutoMPG data in ScalaTion. Same steps as
 *  python/regularized_regression.py (statsmodels):
 *    - train/test rows and the 5 CV folds come from data/test_indices.csv and data/cv_folds.csv
 *    - predictors are scaled with the training rows (NormForm) and y is centered, so the
 *      intercept is the training mean and is not penalized
 *    - the penalty comes from 5-fold CV over the same 60 values, with lambda = n_train * alpha
 *
 *  > P2_DIR=<path to Project2> sbt "runMain scalation.modeling.regularizedRegression_AutoMPG"
 */

package scalation
package modeling

import java.io.PrintWriter
import scala.io.Source
import scala.math.{abs, pow, sqrt}

import scalation.mathstat._
import scalation.optimization.LassoAdmm

import Example_AutoMPG._

object RegAutoMPGUtil:

    type Fitter = (MatrixD, VectorD) => (MatrixD => VectorD)

    val nFolds = 5
    val alphas: Array [Double] = Array.tabulate (60)(i => pow (10.0, -4.0 + 5.0 * i / 59.0))
    val csvNames = Array ("cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Read one integer column of a CSV file that has a header row.
     */
    def readInts (file: String, col: Int): Array [Int] =
        val src = Source.fromFile (file)
        try src.getLines ().drop (1).map (_.split (",")(col).trim.toInt).toArray
        finally src.close ()

    def writeFile (path: String, lines: Seq [String]): Unit =
        val pw = new PrintWriter (path)
        try lines.foreach (pw.println)
        finally pw.close ()

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build and train a Ridge or Lasso model (scaled x, centered y) for the given lambda.
     *  Both classes share RidgeRegression.hp, so lambda is set right before the model is built.
     */
    def fitModel (ridge: Boolean, lam: Double, xx: MatrixD, yy: VectorD): Predictor =
        RidgeRegression.hp("lambda") = lam
        val mod: Predictor =
            if ridge then RidgeRegression.rescale (xx, yy, x_fname)
            else
                LassoAdmm.reset                                            // no warm start carried between fits
                LassoRegression.rescale (xx, yy, x_fname)
        mod.train ()
        mod

    def penalizedFitter (ridge: Boolean, lam: Double): Fitter = (xx, yy) =>
        val mod = fitModel (ridge, lam, xx, yy)
        (z: MatrixD) => mod.predict (z)

    val olsFitter: Fitter = (xx, yy) =>
        val xf  = new NormForm (xx)
        val mod = new Regression (VectorD.one (xx.dim) +^: xf.f (xx), yy)
        mod.train ()
        (z: MatrixD) => mod.predict (VectorD.one (z.dim) +^: xf.f (z))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** 5-fold CV RMSE using the folds saved by the Python script
     *  (the scaling is redone inside each training fold).
     */
    def cvRmse (fitter: Fitter, xtr: MatrixD, ytr: VectorD, fold: Array [Int]): Double =
        val mses = for f <- 0 until nFolds yield
            val tr  = fold.indices.filter (fold(_) != f).toArray
            val va  = fold.indices.filter (fold(_) == f).toArray
            val e   = ytr(va) - fitter (xtr(tr), ytr(tr))(xtr(va))
            e.normSq / e.dim
        sqrt (mses.sum / nFolds)

    def rmse (y: VectorD, yp: VectorD): Double = sqrt ((y - yp).normSq / y.dim)
    def r2 (y: VectorD, yp: VectorD): Double   = 1.0 - (y - yp).normSq / (y - y.mean).normSq

    def mae (y: VectorD, yp: VectorD): Double =
        var s = 0.0
        for i <- y.indices do s += abs (y(i) - yp(i))
        s / y.dim

    /** ScalaTion's own quality-of-fit report and coefficient table for a Ridge/Lasso model.
     */
    def nativeReport (mod: Predictor): Unit =
        val (_, qof) = mod.test ()
        println (mod.report (qof))
        mod match
            case m: RidgeRegression => println (m.summary ())
            case m: LassoRegression => println (m.summary ())
            case _                  => ()

end RegAutoMPGUtil


@main def regularizedRegression_AutoMPG (): Unit =

    import RegAutoMPGUtil._

    val root     = sys.env.getOrElse ("P2_DIR", ".")
    val resDir   = s"$root/results"
    val testIdx  = readInts (s"$root/data/test_indices.csv", 0)
    val trainRow = readInts (s"$root/data/cv_folds.csv", 1)
    val fold     = readInts (s"$root/data/cv_folds.csv", 2)

    banner ("Data check (must match the Python/statsmodels run)")
    println (s"rows = ${x.dim}, predictors = ${x.dim2}, x_fname = ${x_fname.mkString (", ")}")
    println (s"sum(mpg) = ${y.sum}  (expected 9190.8),  sum(weight) = ${x(?, 3).sum}  (expected 1167213.0)")
    assert (abs (y.sum - 9190.8) < 1e-6 && abs (x(?, 3).sum - 1167213.0) < 1e-6, "dataset differs from UCI AutoMPG")

    val testSet  = testIdx.toSet
    val trainIdx = Array.range (0, x.dim).filterNot (testSet.contains)
    assert (trainIdx.sameElements (trainRow), "train rows differ from data/cv_folds.csv")
    val (xtr, ytr) = (x(trainIdx), y(trainIdx))
    val (xte, yte) = (x(testIdx), y(testIdx))
    val ntr = ytr.dim
    println (s"train rows = $ntr, test rows = ${yte.dim}, CV folds = $nFolds")

    val mu = xtr.mean
    val sd = xtr.stdev

    banner ("5-fold CV over the lambda grid (lambda = n_train * alpha)")
    val cvRidge = alphas.map (a => cvRmse (penalizedFitter (true,  ntr * a), xtr, ytr, fold))
    val cvLasso = alphas.map (a => cvRmse (penalizedFitter (false, ntr * a), xtr, ytr, fold))
    val iR = cvRidge.indexOf (cvRidge.min)
    val iL = cvLasso.indexOf (cvLasso.min)
    val cvOls = cvRmse (olsFitter, xtr, ytr, fold)
    println (s"Ridge: best alpha = ${alphas(iR)}, lambda = ${ntr * alphas(iR)}, cv_rmse = ${cvRidge(iR)}")
    println (s"Lasso: best alpha = ${alphas(iL)}, lambda = ${ntr * alphas(iL)}, cv_rmse = ${cvLasso(iL)}")
    println (s"OLS:   cv_rmse = $cvOls")

    banner ("Final models on the training set")
    val ridge = fitModel (true,  ntr * alphas(iR), xtr, ytr)
    val lasso = fitModel (false, ntr * alphas(iL), xtr, ytr)
    val olsF  = olsFitter (xtr, ytr)

    banner (s"RidgeRegression (lambda = ${ntr * alphas(iR)}): ScalaTion native report")
    nativeReport (ridge)
    banner (s"LassoRegression (lambda = ${ntr * alphas(iL)}): ScalaTion native report")
    nativeReport (lasso)

    val xf   = new NormForm (xtr)
    val olsM = new Regression (VectorD.one (ntr) +^: xf.f (xtr), ytr, Array ("intercept") ++ x_fname)
    olsM.train ()
    banner ("Regression (OLS baseline): ScalaTion native report")
    val (_, qofOls) = olsM.test ()
    println (olsM.report (qofOls))
    println (olsM.summary ())

    val coefOls   = olsM.parameter
    val coefRidge = ytr.mean +: ridge.parameter
    val coefLasso = ytr.mean +: lasso.parameter

    def toOriginal (b: VectorD): VectorD =
        val slopes = VectorD (for j <- 0 until x.dim2 yield b(j + 1) / sd(j))
        var icpt = b(0)
        for j <- 0 until x.dim2 do icpt -= slopes(j) * mu(j)
        icpt +: slopes

    val predictors: Seq [(String, Option [Double], Double, VectorD, MatrixD => VectorD)] = Seq (
        ("OLS",   None,                    cvOls,       coefOls,   olsF),
        ("Ridge", Some (alphas(iR)),       cvRidge(iR), coefRidge, penalizedFitter (true,  ntr * alphas(iR))(xtr, ytr)),
        ("Lasso", Some (alphas(iL)),       cvLasso(iL), coefLasso, penalizedFitter (false, ntr * alphas(iL))(xtr, ytr)))

    val summaryRows = for (name, alpha, cv, coef, pred) <- predictors yield
        val (ptr, pte) = (pred (xtr), pred (xte))
        val nZero = (1 until coef.dim).count (j => abs (coef(j)) < 1e-8)
        val (aStr, lStr) = alpha match
            case Some (a) => (s"$a", s"${ntr * a}")
            case None     => ("", "")
        s"ScalaTion,$name,$aStr,$lStr,$nZero,$cv,${r2 (ytr, ptr)},${rmse (ytr, ptr)},${r2 (yte, pte)},${rmse (yte, pte)},${mae (yte, pte)}"

    writeFile (s"$resDir/scalation_summary.csv",
        "software,model,alpha,lambda_equiv,n_zero_coefs,cv_rmse,train_r2,train_rmse,test_r2,test_rmse,test_mae" +: summaryRows)

    val terms = "intercept" +: csvNames.toSeq
    def coefRows (conv: VectorD => VectorD): Seq [String] =
        val cols = Seq (coefOls, coefRidge, coefLasso).map (conv)
        "term,OLS,Ridge,Lasso" +: terms.indices.map (i => s"${terms(i)},${cols.map (c => c(i)).mkString (",")}")
    writeFile (s"$resDir/scalation_coefficients_standardized.csv",   coefRows (identity))
    writeFile (s"$resDir/scalation_coefficients_original_units.csv", coefRows (toOriginal))

    writeFile (s"$resDir/scalation_cv_curve.csv",
        "alpha,lambda_equiv,ridge_cv_rmse,lasso_cv_rmse" +:
        alphas.indices.map (i => s"${alphas(i)},${ntr * alphas(i)},${cvRidge(i)},${cvLasso(i)}"))

    banner ("Coefficient paths over the lambda grid")
    for (name, ridgeFlag) <- Seq (("ridge", true), ("lasso", false)) do
        val rows = alphas.map { a =>
            val b = fitModel (ridgeFlag, ntr * a, xtr, ytr).parameter
            s"$a,${ntr * a},${b.indices.map (b(_)).mkString (",")}" }
        writeFile (s"$resDir/scalation_path_$name.csv", ("alpha,lambda_equiv," + csvNames.mkString (",")) +: rows.toSeq)

    banner ("Summary (standardized coefficients; intercept = training mean of mpg)")
    println ("term          OLS         Ridge       Lasso")
    for i <- terms.indices do
        println (f"${terms(i)}%-13s ${coefOls(i)}%10.4f  ${coefRidge(i)}%10.4f  ${coefLasso(i)}%10.4f")
    summaryRows.foreach (println)
    println (s"results written to $resDir")

end regularizedRegression_AutoMPG
