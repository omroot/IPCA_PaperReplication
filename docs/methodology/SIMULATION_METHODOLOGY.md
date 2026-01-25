# IPCA Monte Carlo Simulation Methodology

This document explains the logic behind the synthetic data generation for IPCA (Instrumented Principal Component Analysis) Monte Carlo simulations, following Section 7 of Kelly, Pruitt, and Su (2020).

## Table of Contents
1. [The IPCA Model](#the-ipca-model)
2. [Data Generating Process](#data-generating-process)
3. [Step-by-Step Generation](#step-by-step-generation)
4. [Calibration Strategy](#calibration-strategy)
5. [Identification and Rotation](#identification-and-rotation)
6. [Monte Carlo Procedure](#monte-carlo-procedure)
7. [Implementation Details](#implementation-details)

---

## The IPCA Model

### Core Equations

The IPCA model decomposes panel data as:

```
x_{i,t} = β_{i,t}' f_t + ε_{i,t}
```

where:
- `x_{i,t}` is the observed outcome for entity `i` at time `t`
- `f_t` is a `K×1` vector of latent factors
- `β_{i,t}` is a `K×1` vector of entity-time-specific factor loadings
- `ε_{i,t}` is idiosyncratic error

The key innovation of IPCA is modeling time-varying loadings as:

```
β_{i,t} = Γ' c_{i,t}
```

where:
- `c_{i,t}` is an `L×1` vector of observable characteristics (instruments)
- `Γ` is an `L×K` matrix mapping characteristics to loadings

### Combined Model

Substituting the loading equation into the main model:

```
x_{i,t} = c_{i,t}' Γ f_t + ε_{i,t}
```

This is the fundamental equation we simulate.

---

## Data Generating Process

The simulation follows the paper's DGP with these components:

### 1. Factors (`f_t`)

Factors follow a VAR(1) (Vector Autoregression) process:

```
f_t = Φ_f f_{t-1} + u_t,    u_t ~ N(0, Σ_f)
```

where:
- `Φ_f` is the `K×K` autoregressive coefficient matrix
- `Σ_f` is the `K×K` innovation covariance matrix
- In our implementation: `Φ_f = φ_f × I_K` (diagonal with persistence `φ_f ≈ 0.9`)

**Intuition**: Factors represent systematic risk drivers (like market risk, size, value). VAR(1) captures their persistence and potential cross-factor dynamics.

### 2. Characteristics (`c_{i,t}`)

Each entity's characteristics follow independent panel VAR(1) processes:

```
c_{i,t} = Φ_c c_{i,t-1} + v_{i,t},    v_{i,t} ~ N(0, Σ_c)
```

where:
- `Φ_c` is the `L×L` autoregressive coefficient matrix
- `Σ_c` is the `L×L` innovation covariance matrix
- In our implementation: `Φ_c = φ_c × I_L` with `φ_c ≈ 0.95`

**Intuition**: Characteristics (like size, book-to-market, momentum) are persistent but evolve over time. Each firm has its own characteristic trajectory.

### 3. Mapping Matrix (`Γ`)

The `L×K` matrix `Γ` is the key object of interest:
- Fixed across the simulation (the "truth" we try to estimate)
- Maps characteristics to factor loadings
- In the paper, calibrated from empirical asset pricing estimates

### 4. Errors (`ε_{i,t}`)

Idiosyncratic errors are i.i.d. normal:

```
ε_{i,t} ~ N(0, σ²_ε)
```

where `σ²_ε` is calibrated to achieve a target R² (typically 20%).

---

## Step-by-Step Generation

```
┌─────────────────────────────────────────────────────────────┐
│                    SIMULATION PIPELINE                       │
└─────────────────────────────────────────────────────────────┘

Step 1: Initialize Γ (L×K)
         │
         ▼
Step 2: Generate factors f_t via VAR(1)
         │         T periods
         │         K factors
         ▼
Step 3: Generate characteristics c_{i,t} via panel VAR(1)
         │         N entities × T periods × L characteristics
         ▼
Step 4: Compute factor loadings β_{i,t} = Γ' c_{i,t}
         │         N × T × K loadings
         ▼
Step 5: Calibrate error variance σ²_ε for target R²
         │
         ▼
Step 6: Generate errors ε_{i,t} ~ N(0, σ²_ε)
         │
         ▼
Step 7: Compute outcomes x_{i,t} = β_{i,t}' f_t + ε_{i,t}
         │
         ▼
Step 8: Format as panel DataFrame for IPCA estimation
```

### Detailed Steps

#### Step 1: Initialize Γ

```python
# Random initialization (or use empirical calibration)
Gamma = np.random.randn(L, K) * scale
```

The scale factor controls the magnitude of factor loadings.

#### Step 2: Generate Factors

```python
# VAR(1) with burn-in period
for t in range(1, T + burn_in):
    f[t] = phi_f * f[t-1] + innovation[t]
```

Burn-in discards initial transients for stationarity.

#### Step 3: Generate Characteristics

```python
# Independent VAR(1) for each entity
for i in range(N):
    for t in range(1, T + burn_in):
        c[i,t] = phi_c * c[i,t-1] + innovation[i,t]
```

#### Step 4: Compute Loadings

```python
# β_{i,t} = Γ' c_{i,t}
for i in range(N):
    for t in range(T):
        beta[i,t] = Gamma.T @ characteristics[i,t]
```

#### Step 5-6: Calibrate and Generate Errors

```python
# Compute signal variance
signal_variance = Var(β_{i,t}' f_t)

# Solve for error variance given target R²
# R² = signal_var / (signal_var + error_var)
# => error_var = signal_var * (1 - R²) / R²
error_variance = signal_variance * (1 - target_r2) / target_r2
```

#### Step 7: Generate Outcomes

```python
x[i,t] = beta[i,t] @ factors[t] + error[i,t]
```

---

## Calibration Strategy

### Why 20% R²?

The paper calibrates to empirical asset pricing: individual stock returns have approximately 15-20% of variance explained by systematic factors. This is realistic:
- Too high R² (>50%): Unrealistically predictable returns
- Too low R² (<5%): Signal drowned in noise, hard to estimate

### Error Variance Derivation

Given the model `x = signal + ε`:

```
Var(x) = Var(signal) + Var(ε)

R² = Var(signal) / Var(x)
   = Var(signal) / (Var(signal) + Var(ε))

Solving for Var(ε):
Var(ε) = Var(signal) × (1 - R²) / R²
```

For R² = 0.20:
```
Var(ε) = Var(signal) × 0.80 / 0.20 = 4 × Var(signal)
```

So the error variance is 4× the signal variance to achieve 20% R².

---

## Identification and Rotation

### The Rotation Problem

IPCA identifies `Γ` only up to a rotation matrix `H`:

```
If (Γ, f_t) solves the model, so does (Γ H, H⁻¹ f_t)
```

This means:
- The estimated `Γ̂` may be a rotated version of true `Γ`
- We cannot directly compare `Γ̂` to `Γ`

### Procrustes Solution

To compare estimates to truth, we solve the orthogonal Procrustes problem:

```
min_H ||Γ - Γ̂ H||_F   subject to H'H = I
```

Solution via SVD:
```python
# Compute M = Γ̂' Γ
# SVD: M = U S V'
# Optimal rotation: H = U V'
M = Gamma_hat.T @ Gamma_true
U, S, Vt = svd(M)
H_optimal = U @ Vt

# Align estimate
Gamma_hat_aligned = Gamma_hat @ H_optimal
```

### Normalization Schemes

The paper discusses two normalizations:

1. **Θ_X Normalization**: First K rows of Γ form identity matrix
   - Interpretable factor loadings
   - Common in empirical work

2. **Θ_Y Normalization**: Factors orthonormal (F'F/T = I)
   - Useful for asymptotic theory
   - Factors have unit variance

---

## Monte Carlo Procedure

### Paper Specification (Section 7)

| Parameter | Value |
|-----------|-------|
| Simulations | 200 |
| Entities (N) | 200 |
| Time periods (T) | 200 |
| Factors (K) | 2 |
| Characteristics (L) | 10 |
| Target R² | 20% |

### Procedure

```
For sim = 1, ..., 200:
    1. Generate data with fixed true Γ
    2. Estimate Γ̂ via IPCA
    3. Align Γ̂ to Γ using Procrustes
    4. Store estimation errors: Γ - Γ̂_aligned

Compute:
    - Mean and std of elementwise errors
    - Distribution of ||Γ - Γ̂||_F (Frobenius norm)
    - Compare to asymptotic theory predictions
```

### Expected Results

The paper's asymptotic theory predicts:
- `√(NT) (Γ̂ - Γ)` converges to normal distribution
- Estimation errors decrease at rate `1/√(NT)`
- Monte Carlo should confirm these theoretical predictions

---

## Implementation Details

### File Structure

```
simulate_ipca_data.py
├── generate_var1_process()      # Single VAR(1) series
├── generate_panel_var1()        # Panel of VAR(1) processes
├── calibrate_error_variance()   # R² calibration
├── generate_ipca_data()         # Main data generation
├── run_single_simulation()      # Single IPCA estimation
├── compute_gamma_error()        # Procrustes alignment
└── run_monte_carlo()            # Full MC study
```

### Usage Examples

#### Generate Synthetic Data

```python
from simulate_ipca_data import generate_ipca_data

data = generate_ipca_data(
    N=200,          # entities
    T=200,          # time periods
    K=2,            # factors
    L=10,           # characteristics
    target_r2=0.20, # calibration target
    seed=42         # reproducibility
)

# Access components
X = data['X']                    # outcomes (N, T)
factors = data['factors']        # true factors (T, K)
chars = data['characteristics']  # characteristics (N, T, L)
Gamma = data['Gamma']            # true Gamma (L, K)
df = data['df']                  # formatted for IPCA
```

#### Run Monte Carlo Study

```python
from simulate_ipca_data import run_monte_carlo

results = run_monte_carlo(
    n_simulations=200,
    N=200,
    T=200,
    K=2,
    L=10,
    verbose=True
)

# Examine results
print(f"Mean RMSE: {results['summary']['mean_rmse']:.4f}")
print(f"Std RMSE: {results['summary']['std_rmse']:.4f}")
```

### Computational Considerations

| N | T | L | K | Approx. Time per Simulation |
|---|---|---|---|---------------------------|
| 100 | 100 | 5 | 2 | ~5 seconds |
| 200 | 200 | 10 | 2 | ~30 seconds |
| 200 | 200 | 10 | 4 | ~60 seconds |

For 200 simulations with N=T=200, expect ~2-3 hours total runtime.

---

## References

- Kelly, B., Pruitt, S., & Su, Y. (2020). Instrumented Principal Component Analysis. *Journal of Financial Economics*.
- Section 7: Monte Carlo Simulations (pages 28-30)
- Section 8.2: Asset Pricing Application

---

## Appendix: Mathematical Notation Summary

| Symbol | Dimension | Description |
|--------|-----------|-------------|
| `x_{i,t}` | scalar | Outcome for entity i at time t |
| `f_t` | K×1 | Latent factor vector |
| `β_{i,t}` | K×1 | Factor loadings for entity i at time t |
| `c_{i,t}` | L×1 | Observable characteristics |
| `Γ` | L×K | Characteristic-to-loading mapping |
| `ε_{i,t}` | scalar | Idiosyncratic error |
| `Φ_f` | K×K | Factor VAR coefficient |
| `Φ_c` | L×L | Characteristic VAR coefficient |
| `N` | scalar | Number of entities |
| `T` | scalar | Number of time periods |
