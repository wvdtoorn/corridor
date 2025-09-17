#!/bin/bash

set -eu

my_dir="$(dirname $(readlink -f $0))"
data_dir="$(dirname "$my_dir")/data"
tmp_dir="$my_dir/tmp"

mkdir -p "$tmp_dir"
fit_binned_lognormal -i "$data_dir/dorado_polya.csv.gz" -o "$tmp_dir/fit_binned_lognormal.csv"
create_best_fit_csv \
    --all "$tmp_dir/fit_binned_lognormal.csv" \
    --best "$my_dir/assets/best_fit_idx.csv" \
    --output "$tmp_dir/best_fit.csv"
predict -i "$tmp_dir/best_fit.csv" -o "$tmp_dir/predictions.csv"
diff <(sort "$tmp_dir/predictions.csv") <(sort "$my_dir/assets/expected_predictions.csv") \
    || (echo "Predictions do not match expected values" && exit 1)
echo "Predictions match expected values"