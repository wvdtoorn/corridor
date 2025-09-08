conda activate corridor
python corridor/fit_binned_lognormal.py -i data/dorado_polya.csv.gz -o test/fit_binned_lognormal.csv
python corridor/create_best_fit_csv.py --all test/fit_binned_lognormal.csv --best test/assets/best_fit_idx.csv --output test/best_fit.csv
python corridor/predict.py -m corridor/models/rna004_130bps_sup@v5.1.0_v1.0.0.pkl -i test/best_fit.csv -o test/predictions.csv