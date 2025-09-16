#!/usr/bin/env python3
"""
predict.py

Load a saved model artifact (pickle) and predict true_length with 95% CI for
rows in an input CSV.

Input CSV must contain columns: id, mu, sigma

Output CSV contains columns: id, predicted_length, ci_lo, ci_hi
"""

import argparse
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


# -------------------------
# Utilities
# -------------------------
def build_design(
    mu: np.ndarray, sigma: np.ndarray, degree: int = 1
) -> Tuple[np.ndarray, List[str]]:
    """
    Create polynomial design (no intercept)
    Returns X (n x p) and feature_names list.
    """
    X = np.column_stack([mu, sigma])
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    feature_names = poly.get_feature_names_out(["mu", "sigma"]).tolist()
    return X_poly, feature_names


def load_model_artifact(path: str) -> dict:
    """
    Load a model artifact pickle and normalize to a dict with keys:
      'name' -> model name string (OLS, GLM_Log, GLM_Log_ENetSel)
      'degree' -> polynomial degree (int)
      'fitted' -> statsmodels results object (required)
      'feature_names' -> optional list of feature names
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(p, "rb") as f:
        obj = pickle.load(f)

    out = {}
    out["name"] = obj.get("name")
    out["degree"] = obj.get("degree")
    out["fitted"] = obj.get("fitted")
    out["feature_names"] = obj.get("feature_names")
    return out


def assemble_X_sm_for_params(
    X_full: np.ndarray,
    feature_names: List[str],
    param_index: Optional[List[str]],
) -> np.ndarray:
    """
    Build a design matrix X_sm whose column ordering matches
    param_index (list of parameter names).
    If param_index is None, return intercept + X_full (positional mapping).
    """
    n = X_full.shape[0]
    if param_index is None:
        return np.hstack([np.ones((n, 1)), X_full])

    cols = []
    for pname in param_index:
        if pname in ("const", "Intercept"):
            cols.append(np.ones((n, 1)))
        else:
            if pname in feature_names:
                idx = feature_names.index(pname)
                cols.append(X_full[:, idx: idx + 1])
            else:
                # try positional fallback: x1,x2... mapping
                if pname.startswith("x") and pname[1:].isdigit():
                    pos = int(pname[1:]) - 1
                    if 0 <= pos < X_full.shape[1]:
                        cols.append(X_full[:, pos: pos + 1])
                        continue
                # try mu/sigma when degree==1
                if pname in ("mu", "sigma") and len(feature_names) >= 2:
                    try:
                        idx = feature_names.index(pname)
                        cols.append(X_full[:, idx: idx + 1])
                        continue
                    except ValueError:
                        pass
                # last resort: zeros
                cols.append(np.zeros((n, 1)))
    if cols:
        return np.hstack(cols)
    else:
        return np.ones((n, 1))


# -------------------------
# Prediction logic
# -------------------------
def predict_with_intervals(
    artifact: dict, df_in: pd.DataFrame
) -> pd.DataFrame:
    """
    Input:
      artifact: dict from load_model_artifact()
      df_in: DataFrame with columns id, mu, sigma
    Returns:
      DataFrame with columns id, predicted_length, ci_lo, ci_hi
    """
    fitted = artifact.get("fitted")
    saved_feature_names = artifact.get("feature_names")

    if fitted is None:
        raise ValueError(
            "Artifact does not contain a fitted model/result object under ",
            "'fitted' key.",
        )

    mu = df_in["mu"].to_numpy(dtype=float)
    sigma = df_in["sigma"].to_numpy(dtype=float)
    X_full, feature_names = build_design(mu, sigma, 1)

    if saved_feature_names:
        if len(saved_feature_names) != X_full.shape[1]:
            raise ValueError(
                (
                    "Saved artifact.feature_names length does not match ",
                    "constructed design. Ensure artifact degree matches.",
                )
            )
        feature_names = saved_feature_names

    try:
        params = fitted.params
        param_index = list(params.index) if hasattr(params, "index") else None
        params_arr = np.asarray(params)
    except Exception:
        # fallback
        params_arr = np.asarray(getattr(fitted, "params", None))
        param_index = None

    X_sm = assemble_X_sm_for_params(X_full, feature_names, param_index)
    n = X_sm.shape[0]

    preds = np.zeros(n)
    cis_lo = np.zeros(n)
    cis_hi = np.zeros(n)

    # GLM with log link: compute linear predictor and delta-method intervals
    Xb = X_sm @ params_arr
    mu_hat = np.exp(Xb)
    # cov, var_mean_eta
    try:
        cov = fitted.cov_params()
    except Exception:
        cov = None
    if cov is not None:
        try:
            var_mean_eta = np.einsum("ij,jk,ik->i", X_sm, cov, X_sm)
        except Exception:
            var_mean_eta = np.full(n, 1e-8)
    else:
        var_mean_eta = np.full(n, 1e-8)
    var_mean_y = (mu_hat**2) * var_mean_eta
    try:
        res_scale = float(getattr(fitted, "scale", np.nan))
        if np.isnan(res_scale):
            res_scale = 0.0
    except Exception:
        res_scale = 0.0
    var_pred = var_mean_y + res_scale
    se_pred = np.sqrt(np.maximum(var_pred, 1e-12))
    preds = mu_hat
    cis_lo = np.maximum(preds - 1.96 * se_pred, 1e-12)
    cis_hi = preds + 1.96 * se_pred

    out = pd.DataFrame(
        {
            "sample_id": df_in["sample_id"].values,
            "predicted_length": preds,
            "ci_lo": cis_lo,
            "ci_hi": cis_hi,
        }
    )

    return out


# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Predict true_length with 95% CI from saved model and "
            "input CSV (id,mu,sigma)."
        ),
    )
    parser.add_argument(
        "--model",
        "-m",
        required=True,
        help=(
            "Path to pickled model artifact (dict or object or statsmodels ",
            "result).",
        ),
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to input CSV with columns: id, mu, sigma",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path to output CSV to write predictions.",
    )
    args = parser.parse_args()

    artifact = load_model_artifact(args.model)

    df = pd.read_csv(args.input)
    required = {"sample_id", "mu", "sigma"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Input CSV must contain columns: {', '.join(required)}"
        )

    out_df = predict_with_intervals(artifact, df)
    out_df.to_csv(args.output, index=False)
    print(f"Wrote predictions to {args.output}")
    print(out_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
