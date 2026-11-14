Reference implementations used ONLY by `python run_cell.py --check-ref`, which compares the
re-implemented SPAM and AdamS optimizers with the authors' code. They are never used for training.

- `adams_official.py`: verbatim copy of `new_optimizers.py` from https://github.com/pkuhuzhang/AdamS (Apache-2.0).
- `spam_official.py`: NOT included in this repository. To run `--check-ref`, download
  `galore_torch/SPAM.py` from https://github.com/TianjinYellow/SPAM-Optimizer and save it here
  as `spam_official.py`.
