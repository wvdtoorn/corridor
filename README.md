# Corridor

Corridor is a tool for analyzing polyA tail length estimates from `dorado`.
We provide a pre-trained model for the `dorado` basecalling model `rna004_130bps_sup@v5.1.0`.

## Installation

Currently, `corridor` is not a package, but a collection of scripts.

Install and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate corridor
```


## Usage

**0. Basecall your data**

Basecall your data with `dorado` (model `rna004_130bps_sup@v5.1.0`) with the `--estimate-poly-a` option enabled.

**1. Parse BAM file**

```bash
python corridor/parse_bam_polya.py path/to/bam/file -o path/to/dorado_polya.csv
```

**1.b (Optional) Update the sample name in the output CSV.**

If your BAM file has multiple samples, you should update the `sample_id` column in the output CSV based on the `read_id` column and your demultiplexing results. This is necessary for the next step.

**2. Fit binned lognormal model**

```bash
python corridor/fit_binned_lognormal.py -i path/to/dorado_polya.csv -o path/to/fit_binned_lognormal.csv
```

This creates a figure per sample in the `path/to` directory as well as a CSV file with the results. You should check the figure(s) to determine the minimal reasonable fit, i.e. the left most plot that matches the majority of the data while ignoring low and high value noise. We currently do not support predictions per mode, for multi-modal polya tail length distributions.

**3. Create best fit CSV**

Based on the results from the previous step, you should create a CSV file with the best fit for each sample:

```csv
sample_id,fit_idx
sample1,1
sample2,2
sample3,1
...
```

An example of such a CSV file for the included dataset is provided in `test/assets/best_fit_idx.csv`.

Then, run the following command:

```bash
python corridor/create_best_fit_csv.py --all path/to/fit_binned_lognormal.csv --best path/to/best_fit_idx.csv --output path/to/best_fit.csv
```

**4. Predict true length with 95% CI**

```bash
python corridor/predict.py -m corridor/models/rna004_130bps_sup@v5.1.0_v1.0.0.pkl -i path/to/best_fit.csv -o path/to/predictions.csv
```

The output CSV will have the following structure:

```csv

sample_id,predicted_length,ci_lo,ci_hi
sample1,15,10,20
sample2,20,15,25
sample3,29,24,34
...
```

## End-to-end test with real data

```
bash test/run_e2e_test.sh
```