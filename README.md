# ParamBoost: Gradient Boosted Piecewise Cubic Polynomials

This repository contains the code used to obtain results for the ICML 2026 submission "ParamBoost: Gradient Boosted Piecewise Cubic Polynomials".

We used `uv` as python package manager. To obtain the benchmark results, simply run:

```
uv run benchmark_exp.py
```

Note that due to conflicts with python packages, EBM needs to be run separately in its own environment. This can be done, for example, with mamba:

```
mamba create -n interpret python==3.13
mamba activate interpret
pip install -r requirements.txt
```
Then run:
```
python benchmark_exp_ebm.py
```
Where the lines from `benchmark_exp.py` that would result in ImportError have been commented out for convenience.

To obtain the case study results, simply run:

```
uv run case_study/lpmc.py
```

Note that the datasets are already splitted in train--val--test sets. To obtain new splits, simply run `split_dataset.py` with a different seed.

Finally, note that some datasets had to be omitted because they were too large to be pushed on github. They can be downloaded from the following source:

- MSD: https://archive.ics.uci.edu/dataset/203/yearpredictionmsd
- Diabetes: https://github.com/vduong143/CAT-KDD-2024/tree/main
- Fraud: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud


