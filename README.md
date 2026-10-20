# Robust Covariance Matrix Estimation & Minimum Variance Portfolio

## 📌 Overview

This research project explores the robust estimation of the financial returns' covariance matrix $\Sigma$ and its application to **minimum-variance portfolio** construction.

Real financial returns deviate significantly from the Gaussian model by exhibiting extreme events (*heavy tails*) and time-varying volatility. The standard sample covariance estimator is highly sensitive to outliers, compromising risk management and portfolio stability. This project implements and compares robust estimators (Catoni, *Median-of-Means*, Shrinkage) to ensure superior out-of-sample stability.

## 🎯 Objectives

* **Implement Robust Estimators:** Adapt recent statistical methods (Catoni, MoM) to the multivariate covariance matrix setting.

* **Evaluate on Simulated Data:** Compare estimation error $\|\hat{\Sigma} - \Sigma\|$ against a known ground truth (Gaussian vs. heavy-tailed Student-$t$ distributions).

* **Backtest on Real Data:** Measure weight stability, turnover, and out-of-sample volatility reduction of the minimum-variance portfolio under real market conditions.

* **Identify Key Regimes:** Determine in which market configurations and asset universes robust approaches provide the most significant gain.

## 🔬 Research Methodology

1. **Synthetic Benchmark:**
   Generate controlled time series with a known matrix $\Sigma$ to evaluate convergence rates and resistance to outliers.

2. **Empirical Validation (Out-of-Sample):**
   Rolling-window time-series backtesting on historical stock and index data to test robustness against regime shifts.

3. **Constrained Optimization:**
   Solve the minimum-variance allocation problem:

   $$
   \min_{w} w^T \hat{\Sigma} w \quad \text{s.t.} \quad \sum_{i=1}^{p} w_i = 1, \quad w_i \ge 0
   $$

## 👥 Project Context

* **Project:** PI² No. 161 – *Robust Estimation of Covariance Matrix*

* **Institution:** ESILV (École Supérieure d'Ingénieurs Léonard de Vinci)

* **Advisor:** Sirine Louati
