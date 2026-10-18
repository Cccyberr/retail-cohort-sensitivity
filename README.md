# Reproducibility package: Trailing-Window Sensitivity of First-Observed-Purchase Cohort Activity

## Contents

- `load.py` reads the two sheets of the UCI file and writes one pickle.
- `analysis_c2.py` is the frozen analysis script (no model fitting, no hypothesis tests). Seeds are fixed inside the script: primary 20261016, check 20261017, permutation 20261018. The number of bootstrap pairs B defaults to 1000 and can be overridden with the environment variable `C2_B` (used for the supplementary B=10000 run).
- `analysis_output_PRIMARY_B1000.txt` is the frozen primary output. Every number in the paper's Methods and Results comes from it, except the B=10,000 statements, which come from the supplementary output.
- `analysis_SUPPLEMENTARY_B10000_REGENERATED.txt` is the supplementary run (B=10,000 for both seeds). It is not the primary analysis, and its values must not be mixed with the primary results.
- `SUPPLEMENTARY_README.txt` explains how the supplementary output was produced.
- `plot_figures.py` regenerates the three data figures from the primary output. Without a manuscript, it runs the figure-generation, rank and nesting checks. If an optional manuscript `.tex` is supplied as a second argument, it also cross-checks the plotted values against the manuscript's numbers.
- `verification_log.txt` contains the recorded verification log.

## Dataset

The dataset is not redistributed here. Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D (CC BY 4.0; 1,067,371 rows across two sheets).

Download the dataset from UCI and extract `online_retail_II.xlsx` into the repository directory.

## Steps

1. Load the dataset:

   ```bash
   python3 load.py online_retail_II.xlsx all.pkl
   ```

2. Run the primary analysis:

   ```bash
   python3 -I analysis_c2.py all.pkl > my_primary_output.txt
   ```

   Compare `my_primary_output.txt` with `analysis_output_PRIMARY_B1000.txt` using `cmp` or SHA-256.

3. Optional supplementary run:

   ```bash
   C2_B=10000 python3 -I analysis_c2.py all.pkl > my_supp_output.txt
   ```

   Compare `my_supp_output.txt` with `analysis_SUPPLEMENTARY_B10000_REGENERATED.txt`.

4. Regenerate the figures:

   ```bash
   mkdir -p figures
   python3 plot_figures.py analysis_output_PRIMARY_B1000.txt
   ```

   The manuscript `.tex` is not included in this package. To run the additional manuscript cross-check, supply its path as an optional second argument.

## Clean-room test

Linux, fresh directory, using the UCI Online Retail II ZIP supplied by the authors: `load.py` took about 50 seconds; `analysis_c2.py` took about 35 seconds and its output was byte-identical to the frozen primary output. The `C2_B=10000` run took about 3 minutes and was byte-identical to the supplementary output. `plot_figures.py` ran successfully without a manuscript.

## Environment

Reported environment: Python 3.13.16, pandas 3.0.5, NumPy 2.5.3, SciPy 1.18.1 and openpyxl 3.1.5 on Linux.

Bootstrap draws depend on the NumPy random generator version. If your environment differs, small differences in resampled summaries are possible; compare against the frozen output and report the versions used.

## Checksums (SHA-256)

- `analysis_c2.py`: `698921a1de3f43af81f69a83649c0e0cf56543a477952305800a61801660451e`
- `load.py`: `e231c5c823ad22a77ef981b6edc2fa655751c1ac2c48ff9885760964954be8dc`
- `SUPPLEMENTARY_README.txt`: `2c95177c06eafd2cba94d6e53a9b64652a4acaa5daac7e5746f99c456b644605`
- `analysis_SUPPLEMENTARY_B10000_REGENERATED.txt`: `89624299a507798e1394a0f168fe3d134e76e03d1533982734570185f1a05a63`
- `analysis_output_PRIMARY_B1000.txt`: `66ec75027440533aa0ea1cd91762c6ee68638767dcb83fe3a3542a33c0ad4783`
- `verification_log.txt`: `05c393ca5a6a2dcb2b9ca74e9f12543a9d2e64d8586ca433c4170ddf9724825b`
- `plot_figures.py`: `28d5ea2ab29b3f94a2c628fb1e7a06531749e5b19a2a296548feee686358a01e`

## Repository, license and archive DOI

Repository: https://github.com/Cccyberr/retail-cohort-sensitivity

License: To be specified after the authors decide on the appropriate license and confirm the rights to distribute the included materials.

Archive DOI: Not yet assigned. If a DOI is created through Zenodo or another archive, add the DOI here.
