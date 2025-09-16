import argparse

import numpy as np
import pandas as pd


def create_best_fit_csv(
    df_all: pd.DataFrame, df_best: pd.DataFrame
) -> pd.DataFrame:
    """
    Create best fit CSV from input CSV (sample_id,mu1,sigma1,mu2,sigma2,...).
    """
    df_all = df_all.merge(df_best, on="sample_id", how="left")
    mu_columns = [c for c in df_all.columns if c.startswith("mu")]
    sigma_columns = [c for c in df_all.columns if c.startswith("sigma")]
    mu_vals = df_all[mu_columns].values
    sigma_vals = df_all[sigma_columns].values
    fit_idx = df_all["fit_idx"].values.astype(int) - 1
    mu_vals = mu_vals[np.arange(mu_vals.shape[0]), fit_idx]
    sigma_vals = sigma_vals[np.arange(sigma_vals.shape[0]), fit_idx]
    df_all["mu"] = mu_vals
    df_all["sigma"] = sigma_vals
    return df_all[["sample_id", "mu", "sigma"]]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create best fit CSV from input CSV ",
            "(sample_id,mu1,sigma1,mu2,sigma2,...).",
        ),
    )
    parser.add_argument(
        "--all",
        required=True,
        help=(
            "Path to input CSV with columns: ",
            "sample_id, mu1, sigma1, mu2, ...",
        ),
    )
    parser.add_argument(
        "--best",
        required=True,
        help=(
            "Path to input CSV with columns: ",
            "sample_id, fit_idx. fit_idx should correspond to the ",
            "suffix of the mu and sigma columns in the all CSV.",
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Path to output CSV to write best fit csv with ",
            "columns: id, mu, sigma.",
        ),
    )
    args = parser.parse_args()

    # load input CSV
    df_all = pd.read_csv(args.all)
    required = {"sample_id", "mu1", "sigma1"}
    if not required.issubset(df_all.columns):
        raise ValueError(
            f"Input CSV must contain columns: {', '.join(required)}"
        )

    df_best = pd.read_csv(args.best)
    required = {"sample_id", "fit_idx"}
    if not required.issubset(df_best.columns):
        raise ValueError(
            f"Input CSV must contain columns: {', '.join(required)}"
        )

    out_df = create_best_fit_csv(df_all, df_best)
    out_df.to_csv(args.output, index=False)
    print(f"Wrote best fit csv to {args.output}")
    print(out_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
