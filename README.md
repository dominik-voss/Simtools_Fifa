# Simtools

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This project uses Monte Carlo simulation to study the Coupon Collector’s problem in the context of FIFA Ultimate Team pack openings. It estimates how many premium gold packs, coins, and real money are needed to collect all gold cards, under realistic but simplified drop probabilities derived from EA’s disclosed odds and rating distributions. The simulation models different collection stages (e.g. 50, 200, 500, 1900 cards), computes sample means, standard errors, and 95% confidence intervals for the required number of packs, and visualizes how the effort grows over the collection progress. The results show that completing the full gold card set requires tens of thousands of packs and an amount of time or money that is practically unattainable for normal players, highlighting the quasi-gambling character of the lootbox system.

## Reproducibility
The simulation environment and all library versions are documented in ABOUT.txt, including Python version, operating system, and core dependencies (NumPy, SciPy, pandas, Matplotlib).
​
This setup ensures that the simulation can be rerun with identical software versions across different systems, making all reported results reproducible as long as the same code and configuration from ABOUT.txt are used.


## Project Organization

```
project/
├── notebooks/            # exploration & demos (Jupyter notebook)
│   └── fifa_packs.ipynb  # main notebook for the pack-opening project
│
├── reports/              # tables, short text outputs, generated reports
│   └── figures/          # plots used in the report
│       ├── packs_vs_cards.png # plotted number of packs that are needed
│       └── coins_vs_cards.png # plotted number of coins that are needed
    └── ABOUT.txt         # Short project description (authors, course, requirements)
│
├── src/                  # core functions / models
│   ├── simulation.py # Monte Carlo pack-opening
    ├── plots_packs_coins.py # plot results       
│
├── requirements.txt      # Python dependencies to reproduce the environment
└── README.md             # high-level project description, setup and usage instructions


```

--------

