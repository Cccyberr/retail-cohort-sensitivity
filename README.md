# Reproducibility package: trailing-window sensitivity of first-observed-purchase cohort activity (UCI Online Retail II)

Contents
- `load.py` reads the two sheets of the UCI file and writes one pickle.
- `analysis_c2.py` is the frozen analysis script (no model fitting, no hypothesis tests). Seeds are fixed inside the script: primary 20261016, check 20261017, permutation 20261018. The number of bootstrap pairs B defaults to 1000 and can be overridden with the environment variable `C2_B` (used only for the supplementary B=10000 run).
- `outputs/analysis_output_PRIMARY_B1000.txt` is the frozen primary output: every number in the paper's Methods and Results comes from it, except the B=10,000 statements, which come from the supplementary file.
- `outputs/analysis_SUPPLEMENTARY_B10000_REGENERATED.txt` is the supplementary run (B=10,000 for both seeds). It is not the primary analysis and its values must not be mixed with the primary ones. `outputs/SUPPLEMENTARY_README.txt` explains how it was produced.
- `figures_script/plot_figures.py` regenerates the three data figures from the primary output and cross-checks them against the manuscript tables. `outputs/verification_log.txt` is its log.

Dataset (not redistributed here)
Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D (CC BY 4.0; 1,067,371 rows across two sheets). Download it from UCI and unzip to obtain `online_retail_II.xlsx`.

Steps
1. `python3 load.py online_retail_II.xlsx all.pkl`
2. `python3 -I analysis_c2.py all.pkl > my_primary_output.txt` then compare with `outputs/analysis_output_PRIMARY_B1000.txt` (`cmp` or sha256).
3. Optional supplementary run: `C2_B=10000 python3 -I analysis_c2.py all.pkl > my_supp_output.txt`.
4. mkdir -p figures
python3 plot_figures.py analysis_output_PRIMARY_B1000.txt
Clean-room test (Linux, fresh directory, using the UCI Online Retail II zip supplied by the authors): `load.py` about 50 s; `analysis_c2.py` about 35 s and its output is byte-identical to the frozen primary output; the `C2_B=10000` run takes about 3 minutes and is byte-identical to the supplementary output; `plot_figures.py` runs without a manuscript.

Environment used for the reported run: Python 3.13.16, pandas 3.0.5, numpy 2.5.3, scipy 1.18.1, openpyxl 3.1.5 (Linux). Bootstrap draws depend on the numpy random generator version; if your environment differs, small differences in resampled summaries are possible, so compare against the frozen output and report the versions you used.

Checksums (sha256)
698921a1de3f43af81f69a83649c0e0cf56543a477952305800a61801660451e  analysis_c2.py
e231c5c823ad22a77ef981b6edc2fa655751c1ac2c48ff9885760964954be8dc  load.py
2c95177c06eafd2cba94d6e53a9b64652a4acaa5daac7e5746f99c456b644605  outputs/SUPPLEMENTARY_README.txt
89624299a507798e1394a0f168fe3d134e76e03d1533982734570185f1a05a63  outputs/analysis_SUPPLEMENTARY_B10000_REGENERATED.txt
66ec75027440533aa0ea1cd91762c6ee68638767dcb83fe3a3542a33c0ad4783  outputs/analysis_output_PRIMARY_B1000.txt
05c393ca5a6a2dcb2b9ca74e9f12543a9d2e64d8586ca433c4170ddf9724825b  outputs/verification_log.txt
28d5ea2ab29b3f94a2c628fb1e7a06531749e5b19a2a296548feee686358a01e  figures_script/plot_figures.py

Not yet decided by the authors: the repository host, the software license, and the archive DOI (for example a Zenodo deposit). Add them here after the repository exists.
