# Simtools

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This project analyzes how many packs and coins are needed to complete a collection of gold cards in FIFA 26 using Monte Carlo simulation.

## Project Organization

```
project/
├── configs/              # yaml/json for seeds and global parameters (optional)
│
├── data/                 # reserved for datasets (not strictly needed yet)
│   ├── raw/              # original, read-only inputs
│   ├── interim/          # cleaned / transformed data
│   └── processed/        # final data used for analysis
│
├── figures/              # additional png/pdf plots outside of reports (optional)
│
├── notebooks/            # exploration & demos (e.g. Jupyter notebooks)
│   └── fifa_packs.ipynb  # main notebook for the pack-opening project
│
├── reports/              # tables, short text outputs, generated reports
│   └── figures/          # plots used in the report
│       ├── packs_vs_cards.png
│       └── coins_vs_cards.png
    └── ABOUT.txt         # Short project description (authors, course, requirements)
│
├── src/                  # core functions / models
│   ├── __init__.py
│   ├── simulation.py # Monte Carlo pack-opening
    ├── plots_packs_coins.py # plot results       
│
├── tests/                # optional unit tests for simulation and plotting code
│
├── requirements.txt      # Python dependencies to reproduce the environment
└── README.md             # high-level project description, setup and usage instructions


```

--------

