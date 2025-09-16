#!/usr/bin/env python3
"""
fit_binned_lognormal.py

Read a CSV with columns `id` and `polya_len` (integer tail lengths).
For each id, fit a binned (integer) lognormal distribution by MLE and
write a CSV with columns at least: id, mu, sigma.

Usage:
    python fit_binned_lognormal.py --input input.csv --output params.csv
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("binned_lognormal")

INPUT_DTYPES = {
    "read_id": str,
    "reference": str,
    "sample": str,
    "tail_len": float,
    "polya_len": int,
    "template": str,
    "polyA tail from SS of PCR": float,
    "polyA tail from SS of IVT": float,
    "A's at end": float,
    "true_length": int,
    "method": str,
    "run": int,
    "method_run": str,
    "sample_id": int,
}


# -------------------------
# Data classes
# -------------------------
@dataclass
class MixtureResults:
    n_components: int
    weights: np.ndarray
    mu_params: np.ndarray
    sigma_params: np.ndarray
    log_likelihood: float
    aic: float
    bic: float
    converged: bool
    n_iter: int


# -------------------------
# Core utilities
# -------------------------
def binned_lognormal_pmf(k: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """
    Compute discrete PMF P(K = k) for integer k >= 1 using
    lognormal CDF differences:
        P(k) = F(k + 0.5) - F(k - 0.5)
    For k == 1 we use left edge = 0.5 (works for positive integer data).
    Returns an array of probabilities (same shape as k).
    """
    k = np.asarray(k, dtype=float)
    # left and right edges for each bin
    left = np.maximum(k - 0.5, 0.0)  # do not go below 0
    right = k + 0.5
    cdf_right = stats.lognorm.cdf(right, s=sigma, scale=np.exp(mu))
    cdf_left = stats.lognorm.cdf(left, s=sigma, scale=np.exp(mu))
    pmf = cdf_right - cdf_left
    # numerical floor to avoid zero probabilities (prevents -inf logs)
    pmf = np.maximum(pmf, 1e-15)
    return pmf


def neg_log_likelihood_from_counts(
    params: np.ndarray, values: np.ndarray, counts: np.ndarray
) -> float:
    """
    Negative log-likelihood for binned lognormal given unique integer values
    and counts.
    params: [mu, sigma] with sigma > 0.
    values: integer k array
    counts: counts for each k
    """
    mu, sigma = params
    if sigma <= 0 or not np.isfinite(mu) or not np.isfinite(sigma):
        return 1e12
    pmf = binned_lognormal_pmf(values, mu, sigma)
    ll = np.sum(counts * np.log(pmf))
    if not np.isfinite(ll):
        return 1e12
    return -ll


def fit_binned_lognormal_to_sample(data: np.ndarray) -> Dict[str, Any]:
    """
    Fit a binned lognormal to the 1D array `data` of integer positive values.
    Returns dict with keys: mu, sigma, log_likelihood, converged, n_obs.
    If too few observations, returns NaNs and converged=False.
    """
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    data = data[data >= 1]  # enforce positive integers
    n = data.size

    # MLE
    log_data = np.log(data)
    mu_hat = float(np.mean(log_data))
    sigma_hat = float(np.std(log_data, ddof=1))
    if sigma_hat <= 0 or not np.isfinite(sigma_hat):
        sigma_hat = max(0.1, 0.1 * abs(mu_hat) + 0.1)

    return {
        "mu": mu_hat,
        "sigma": sigma_hat,
        "n_obs": int(n),
    }


# -------------------------
# EM for mixture of binned lognormals
# -------------------------
def initialize_mixture_params(
    values: np.ndarray, counts: np.ndarray, n_components: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Initialize weights, mu, sigma for mixture using quantiles of log-values.
    """
    weights = np.full(n_components, 1.0 / n_components)
    log_vals = np.log(values)
    # choose initial mus as quantiles
    quantiles = np.linspace(0.1, 0.9, n_components)
    mu_init = np.quantile(log_vals, quantiles)
    sigma_global = (
        float(np.std(np.repeat(log_vals, counts.astype(int)), ddof=1))
        if np.sum(counts) > 1
        else 0.5
    )
    sigma_init = np.full(
        n_components, max(0.3, sigma_global / np.sqrt(n_components))
    )
    return weights, mu_init.astype(float), sigma_init.astype(float)


def fit_lognormal_mixture(
    values: np.ndarray,
    counts: np.ndarray,
    n_components: int = 2,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> MixtureResults:
    """
    Fit mixture of binned lognormals by EM on unique integer values with
    counts.
    Returns MixtureResults.
    """
    values = np.asarray(values, dtype=float)
    counts = np.asarray(counts, dtype=float)
    N = float(np.sum(counts))
    if N <= 0:
        raise ValueError("No data provided to mixture fitter")

    w, mu, sigma = initialize_mixture_params(values, counts, n_components)

    prev_ll = -np.inf
    converged = False
    for it in range(1, max_iter + 1):
        # E-step: compute responsibilities r[i,j]
        # shape (m, k) where m = len(values)
        pmf_ij = np.zeros((len(values), n_components))
        for j in range(n_components):
            pmf_ij[:, j] = binned_lognormal_pmf(values, mu[j], sigma[j])

        weighted = pmf_ij * w[np.newaxis, :]
        denom = weighted.sum(axis=1, keepdims=True)
        denom = np.maximum(denom, 1e-15)
        r = weighted / denom  # responsibilities for each unique value

        # M-step: update weights, mu, sigma using counts as
        # observation multiplicities
        # Effective counts per component
        effective_counts = counts[:, np.newaxis] * r  # shape (m, k)
        Nk = effective_counts.sum(axis=0)  # size k
        Nk = np.maximum(Nk, 1e-8)
        w = Nk / Nk.sum()
        # Update mu_j and sigma_j via weighted mean and variance of log(values)
        log_vals = np.log(np.maximum(values, 1e-12))
        for j in range(n_components):
            weights_j = effective_counts[:, j]
            mu_j = np.sum(weights_j * log_vals) / Nk[j]
            var_j = np.sum(weights_j * (log_vals - mu_j) ** 2) / Nk[j]
            sigma_j = np.sqrt(max(var_j, 1e-6))
            mu[j] = float(mu_j)
            sigma[j] = float(max(sigma_j, 1e-3))

        comp_mix = (w[np.newaxis, :] * pmf_ij).sum(axis=1)
        comp_mix = np.maximum(comp_mix, 1e-15)
        ll = float(np.sum(counts * np.log(comp_mix)))

        if np.abs(ll - prev_ll) < tol:
            converged = True
            break
        prev_ll = ll

    k_params = 3 * n_components - 1  # (weights-1) + mus + sigmas
    aic = -2.0 * ll + 2.0 * k_params
    bic = -2.0 * ll + k_params * np.log(N)

    return MixtureResults(
        n_components=n_components,
        weights=w,
        mu_params=mu.copy(),
        sigma_params=sigma.copy(),
        log_likelihood=ll,
        aic=aic,
        bic=bic,
        converged=converged,
        n_iter=it,
    )


def remove_outliers(values: np.ndarray, iqr_factor: float = 1.5) -> np.ndarray:
    """
    Remove outliers from the array `values` using a threshold of
    `sigma_threshold`.
    """
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - iqr_factor * iqr
    upper_bound = q3 + iqr_factor * iqr
    mask = (values >= lower_bound) & (values <= upper_bound)
    return values[mask]


# -------------------------
# High-level per-group processing
# -------------------------
def process_group(
    values: np.ndarray,
    *,
    max_components: int = 3,
    group_name: str = None,
    outdir: str = None,
) -> Dict[str, Any]:
    """
    Fit single-component, fit mixtures up to max_components

    Returns a dict of results.
    """
    no_outliers = remove_outliers(np.asarray(values))
    base = fit_binned_lognormal_to_sample(no_outliers)

    vals_all, counts_all = np.unique(
        np.asarray(no_outliers).astype(int), return_counts=True
    )
    mixture_results = []
    for k in range(2, max_components + 1):
        mr = fit_lognormal_mixture(vals_all, counts_all, n_components=k)
        mixture_results.append(mr)

    fig_path = None
    if not mixture_results:
        mus = [base["mu"]] * max_components
        sigmas = [base["sigma"]] * max_components
    else:
        mus = [base["mu"]] + [
            mr.mu_params[int(np.argmax(mr.weights))] for mr in mixture_results
        ]
        sigmas = [base["sigma"]] + [
            mr.sigma_params[int(np.argmax(mr.weights))]
            for mr in mixture_results
        ]

        fig_path = os.path.join(outdir, f"all_fits_{group_name}.png")
        n_plots = len(mixture_results) + 1
        plt.figure(figsize=(n_plots * 5, 5))
        x = np.arange(min(no_outliers), max(no_outliers) + 1)
        base_pdf = stats.lognorm.pdf(
            x, base["sigma"], loc=0, scale=np.exp(base["mu"])
        )

        plt.subplot(1, len(mixture_results) + 1, 1)
        plt.hist(
            no_outliers,
            bins=np.arange(no_outliers.min(), no_outliers.max() + 1),
            density=True,
            alpha=0.5,
            label="Data",
        )
        plt.plot(
            np.arange(no_outliers.min(), no_outliers.max() + 1),
            base_pdf,
            "r-",
            linewidth=2,
            label="Base fit",
        )
        plt.title(f"fit_idx=1")
        plt.legend()
        plt.grid(True, alpha=0.3)

        for i, (mr) in enumerate(mixture_results):
            plt.subplot(1, len(mixture_results) + 1, i + 2)
            plt.hist(
                no_outliers,
                bins=np.arange(no_outliers.min(), no_outliers.max() + 1),
                density=True,
                alpha=0.5,
                label="Data",
            )
            this_majority_idx = int(np.argmax(mr.weights))
            mu = mr.mu_params[this_majority_idx]
            sigma = mr.sigma_params[this_majority_idx]
            pdf = stats.lognorm.pdf(x, sigma, loc=0, scale=np.exp(mu))
            plt.plot(
                x, pdf, "r-", linewidth=2, label=f"Mixture fit (k={i + 1})"
            )
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.title(f"fit_idx={i + 2}")

        fig_mixtures = plt.gcf()
        fig_mixtures.suptitle(f"Sample: {group_name}")
        fig_mixtures.savefig(fig_path)
        plt.close()

    out = {
        "mixture_fig_path": fig_path,
    }
    for i in range(len(mus)):
        out[f"mu{i + 1}"] = mus[i]
        out[f"sigma{i + 1}"] = sigmas[i]
    return out


# -------------------------
# File-level orchestration
# -------------------------
def process_file(
    in_path: str,
    out_path: str,
    id_col: str = "sample_id",
    value_col: str = "polya_len",
    max_components: int = 3,
) -> None:
    """
    Read input CSV, group by id_col, perform mixture detection and filtering,
    write output CSV with results.
    """
    df = pd.read_csv(in_path, dtype=INPUT_DTYPES)
    if not {id_col, value_col}.issubset(df.columns):
        raise ValueError(
            f"Input CSV must contain columns: {id_col}, {value_col}"
        )

    df = df[[id_col, value_col]].dropna()
    df[value_col] = df[value_col].astype(float)

    results = []
    groups = df.groupby(id_col)
    logger.info("Processing %d groups...", groups.ngroups)

    for name, sub in groups:
        vals = sub[value_col].values
        res = process_group(
            vals,
            max_components=max_components,
            group_name=name,
            outdir=os.path.dirname(out_path),
        )
        out_row = {id_col: name}
        out_row.update(res)
        results.append(out_row)

    out_df = pd.DataFrame(results)
    out_df.to_csv(out_path, index=False)
    logger.info("Wrote results for %d groups to %s", len(out_df), out_path)


# -------------------------
# CLI
# -------------------------
def main():
    p = argparse.ArgumentParser(
        description="Fit binned lognormal (mixture) per id"
    )
    p.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input CSV path (columns: id, polya_len)",
    )
    p.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output CSV path (will contain id, mu, sigma, ...)",
    )
    p.add_argument("--id-col", default="sample_id", help="Name of id column")
    p.add_argument(
        "--value-col", default="polya_len", help="Name of length column"
    )
    p.add_argument(
        "--max-components",
        type=int,
        default=3,
        help="Max mixture components to try (default 3)",
    )
    args = p.parse_args()

    process_file(
        args.input,
        args.output,
        id_col=args.id_col,
        value_col=args.value_col,
        max_components=args.max_components,
    )


if __name__ == "__main__":
    main()
