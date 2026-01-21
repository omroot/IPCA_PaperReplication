# IPCA Paper Replication

Replication and extension of the paper **"Instrumented Principal Component Analysis"** by Kelly, Pruitt, and Su (2020), published in the *Journal of Financial Economics*.

## Author

**Oualid Missaoui**

## Overview

This project implements the Instrumented Principal Component Analysis (IPCA) methodology for asset pricing applications. IPCA addresses a fundamental problem in asset pricing by using observable firm characteristics as instruments for time-varying factor loadings, allowing the model to explain cross-sectional return variation more effectively than traditional factor models.

The repository includes:
- Core IPCA estimation algorithm
- Monte Carlo simulations to validate theoretical results (Paper Section 7)
- Empirical asset pricing application using real market data (Paper Section 8.2)
- Out-of-sample evaluation framework

## Project Structure

```
IPCA_PaperReplication/
├── src/                        # Core Python implementation
│   ├── ipca.py                 # Main IPCA estimation class
│   ├── simulate_ipca_data.py   # Monte Carlo data generation
│   └── _deprecated/            # Additional scripts and reference code
│       ├── original_ipca.py    # Alternative/reference implementation
│       ├── reproduce_asset_pricing.py  # Standalone asset pricing analysis
│       ├── full_oos_evaluation.py      # Extended evaluation framework
│       └── diagnostic_summary.py       # Diagnostic tools
│
├── notebooks/                  # Jupyter notebooks (numbered for sequence)
│   ├── 01_ipca_simulation_study.ipynb   # Monte Carlo simulations
│   ├── 02_asset_pricing_replication.ipynb # Empirical asset pricing
│   └── results/                # Output figures and JSON results
│
├── data/                       # Essential data files
│   ├── crsp_monthly_returns.csv         # Stock-level monthly returns
│   ├── datashare.csv                    # Comprehensive market data
│   └── _deprecated/            # Additional datasets
│       ├── characteristics_data_feb2017.csv # Alternative characteristics
│       ├── permno_list.csv              # CRSP permanent security IDs
│       ├── stage1_osbap_0k_volume_2025.parquet
│       └── macro/                       # Goyal-Welch macro predictors
│
├── docs/                       # Documentation
│   ├── bib/                    # Original paper PDFs
│   └── methodology/            # Simulation methodology guide
│
├── requirements.txt            # Python dependencies
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

## IPCA Model

The IPCA model is defined as:

```
x_{i,t} = β_{i,t}' f_t + ε_{i,t}
β_{i,t} = Γ' c_{i,t}
```

Where:
- `x_{i,t}`: Returns for asset i at time t
- `β_{i,t}`: Time-varying factor loadings
- `f_t`: Latent factors
- `c_{i,t}`: Observable firm characteristics
- `Γ`: Characteristic-to-loading mapping matrix

## Usage

### Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run the notebooks in order:**
```bash
cd notebooks
jupyter lab
```
   - Start with `01_ipca_simulation_study.ipynb` to understand IPCA methodology
   - Then run `02_asset_pricing_replication.ipynb` for empirical results

### Using the IPCA Class

```python
import sys
sys.path.append('src')
from ipca import ipca

# Initialize and fit model
model = ipca(returns, characteristics)
results = model.fit(K=4, OOS=True)

# Access results
gamma = results['Gamma']        # Characteristic loadings
factors = results['Factor']     # Estimated factors
r2 = results['xfits']['R2_Total']  # R-squared
```

## Data Requirements

The notebooks require only two data files (included):
- `data/datashare.csv` - Main characteristics and identifiers
- `data/crsp_monthly_returns.csv` - Monthly stock return data

The simulation notebook generates its own synthetic data.

## References

Kelly, B., Pruitt, S., & Su, Y. (2019). Characteristics are covariances: A unified model of risk and return. *Journal of Financial Economics*, 134(3), 501-524.

## License

This project is for academic and research purposes.
