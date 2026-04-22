# ParamBoost: Gradient Boosted Piecewise Cubic Polynomials

This repository contains the code used to obtain results for our [arXiv paper "ParamBoost: Gradient Boosted Piecewise Cubic Polynomials"](https://arxiv.org/abs/2604.18864).

We used `uv` as Python package manager. To obtain the benchmark results, simply run:

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
Due to some numerical errors, the results may vary slightly from one run to another, but should be consistent overall.

Note that the datasets need to be split in train--val--test sets before running the benchmark/case study scripts. To obtain new splits, simply run `split_dataset.py`.

Finally, note that some datasets had to be omitted because they were too large to be pushed on GitHub. They can be downloaded from the following source:

- MSD: https://archive.ics.uci.edu/dataset/203/yearpredictionmsd
- Diabetes: https://github.com/vduong143/CAT-KDD-2024/tree/main
- Fraud: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Then they need to be put in the [data](data/) folder. The loading paths in `split_dataset.py` might need to be updated.


