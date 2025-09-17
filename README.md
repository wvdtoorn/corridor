# Corridor

Corridor is a tool for analyzing polyA tail length estimates from `dorado`.
We provide a pre-trained model for the `dorado` basecalling model `rna004_130bps_sup@v5.1.0`.

## Installation

Best use a separate Python virtual environment, which can be created
and activated like this:

```bash
my_venv_path=~/.virtualenvs/corridor  # Choose as you wish
mkdir -p "$my_venv_path"
python -m venv "$my_venv_path"

. "$my_venv_path/bin/activate"
```

Now you can install the software in the virtual environment run:

```bash
git clone https://github.com/wvdtoorn/corridor.git
cd corridor
pip install .
```

After that the repository is not needed anymore. For a development
setup do an _editable_ installation instead and keep the repository:

```bash
pip install -e .
```


## Usage

**0. Basecall your data**

Basecall your data with `dorado` (model `rna004_130bps_sup@v5.1.0`) with the `--estimate-poly-a` option enabled.

**1. Parse BAM file**

```bash
parse_bam_polya path/to/bam/file -o path/to/dorado_polya.csv
```

**1.b (Optional) Update the sample name in the output CSV.**

If your BAM file has multiple samples, you should update the `sample_id` column in the output CSV based on the `read_id` column and your demultiplexing results. This is necessary for the next step.

**2. Fit binned lognormal model**

```bash
fit_binned_lognormal -i path/to/dorado_polya.csv -o path/to/fit_binned_lognormal.csv
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
create_best_fit_csv --all path/to/fit_binned_lognormal.csv --best path/to/best_fit_idx.csv --output path/to/best_fit.csv
```

**4. Predict true length with 95% CI**

```bash
predict -i path/to/best_fit.csv -o path/to/predictions.csv
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
