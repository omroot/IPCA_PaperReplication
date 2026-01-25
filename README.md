# IPCA Paper Replication

Replication of the paper **["Characteristics are Covariances: A Unified Model of Risk and Return"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2983919)** by Kelly, Pruitt, and Su (2019), published in the *Journal of Financial Economics*.

## Author

**Oualid Missaoui**

## Overview

This project implements the Instrumented Principal Component Analysis (IPCA) methodology using Alternating Least Squares (ALS) for estimation. IPCA uses observable firm characteristics as instruments for time-varying factor loadings, explaining cross-sectional return variation more effectively than traditional factor models.

The repository includes:
- ALS-based IPCA estimation with convergence tracking
- Monte Carlo simulation study validating the estimator
- Empirical asset pricing application replicating Table 2 of the paper (1985-2015)
- Automatic data downloading from Dropbox

## Project Structure

```
IPCA_PaperReplication/
├── src/
│   ├── als_ipca.py             # ALS IPCA estimator
│   ├── simulate_ipca_data.py   # Synthetic data generation
│   ├── data_utils.py           # Data preparation and R² computation
│   └── data_loader.py          # Auto-downloads data from Dropbox if missing
│
├── notebooks/
│   ├── 01_als_ipca_simulation_study.ipynb  # Monte Carlo simulation study
│   └── 02_als_ipca_asset_pricing.ipynb     # Empirical asset pricing replication
│
├── data/                       # Downloaded automatically on first run
│   ├── crsp_monthly_returns.csv
│   └── datashare.csv
│
├── docs/
│   ├── bib/                    # Reference papers
│   └── methodology/            # Methodology notes
│
├── requirements.txt
└── README.md
```

## IPCA Model

The IPCA model:

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

The ALS algorithm alternates between updating factors given `Γ` and updating `Γ` given factors until convergence.

## Usage

### Quick Start

1. **Clone and install dependencies:**
```bash
git clone <repo-url>
cd IPCA_PaperReplication
pip install -r requirements.txt
```

2. **Download the data** (required for asset pricing notebook):
```bash
python -m src.data_loader
```
This downloads `datashare.csv` and `crsp_monthly_returns.csv` from Dropbox into `data/`.

3. **Run the notebooks:**
```bash
jupyter lab notebooks/
```
   - `01_als_ipca_simulation_study.ipynb` — validates the ALS estimator on synthetic data
   - `02_als_ipca_asset_pricing.ipynb` — replicates the empirical results

> **Note:** The asset pricing notebook will also download the data automatically on first run if not present.

### Using the ALS IPCA Estimator

```python
from als_ipca import ALSIPCA

model = ALSIPCA(num_assets=N, num_fact=K, num_charact=L, win_len=T)
Gamma, objective_history = model.fit(data, max_iter=1000, tol=1e-6)

results = model.get_results()
# results['Gamma']   — L x K loading matrix
# results['factors'] — K x T factor estimates
# results['Lambda']  — K x 1 mean factor (risk premia)
```

## Data

The asset pricing notebook requires two CSV files in `data/`. They are downloaded automatically from Dropbox on the first run via `src/data_loader.py`. You can also trigger the download manually:

```bash
python -m src.data_loader
```

## References

Kelly, B., Pruitt, S., & Su, Y. (2019). [Characteristics are covariances: A unified model of risk and return](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2983919). *Journal of Financial Economics*, 134(3), 501-524.

## License

This project is for academic and research purposes.
