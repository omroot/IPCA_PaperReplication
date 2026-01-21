"""
Synthetic Data Generation for IPCA Monte Carlo Simulations

Based on Section 7 of Kelly, Pruitt, Su (2020)
"Instrumented Principal Component Analysis"

This module generates synthetic panel data following the IPCA model:
    x_{i,t} = β_{i,t}' f_t + ε_{i,t}
    β_{i,t} = Γ' c_{i,t}

where:
    - x_{i,t}: observed outcome for entity i at time t
    - f_t: K-dimensional latent factors (time-varying)
    - β_{i,t}: entity-specific factor loadings (time-varying)
    - c_{i,t}: L-dimensional observable characteristics/instruments
    - Γ: L×K mapping from characteristics to loadings
    - ε_{i,t}: idiosyncratic error
"""

import numpy as np
import pandas as pd
from scipy import linalg
from typing import Optional, Dict, Any, Union, Tuple


def generate_var1_process(T: int, n_vars: int, phi: Union[float, np.ndarray], 
                         sigma_u: np.ndarray, burn_in: int = 100) -> np.ndarray:
    """
    Generate a VAR(1) process: y_t = φ y_{t-1} + u_t

    Parameters
    ----------
    T : int
        Number of time periods to generate
    n_vars : int
        Number of variables in the VAR
    phi : float or ndarray
        AR coefficient(s). If scalar, applies to all variables.
        If array, should be (n_vars, n_vars) transition matrix.
    sigma_u : ndarray
        Covariance matrix of innovations (n_vars, n_vars)
    burn_in : int
        Number of initial observations to discard

    Returns
    -------
    ndarray
        Array of shape (T, n_vars) containing the VAR process
    """
    total_T = T + burn_in

    # Handle scalar phi
    if np.isscalar(phi):
        phi_matrix = phi * np.eye(n_vars)
    else:
        phi_matrix = np.asarray(phi)

    # Initialize
    y = np.zeros((total_T, n_vars))

    # Generate innovations
    innovations = np.random.multivariate_normal(
        mean=np.zeros(n_vars),
        cov=sigma_u,
        size=total_T
    )

    # Simulate VAR(1)
    for t in range(1, total_T):
        y[t] = phi_matrix @ y[t-1] + innovations[t]

    # Discard burn-in
    return y[burn_in:]


def generate_panel_var1(N: int, T: int, n_vars: int, phi: Union[float, np.ndarray], 
                        sigma_u: np.ndarray, burn_in: int = 50) -> np.ndarray:
    """
    Generate panel VAR(1) process for characteristics.

    Each entity i follows: c_{i,t} = φ c_{i,t-1} + u_{i,t}

    Parameters
    ----------
    N : int
        Number of entities (cross-sectional dimension)
    T : int
        Number of time periods
    n_vars : int
        Number of characteristics per entity (L)
    phi : float or ndarray
        AR coefficient(s)
    sigma_u : ndarray
        Covariance matrix of innovations
    burn_in : int
        Burn-in periods to discard

    Returns
    -------
    ndarray
        Array of shape (N, T, n_vars) containing characteristics
    """
    characteristics = np.zeros((N, T, n_vars))

    for i in range(N):
        characteristics[i] = generate_var1_process(T, n_vars, phi, sigma_u, burn_in)

    return characteristics


def calibrate_error_variance(Gamma: np.ndarray, factors: np.ndarray, 
                           characteristics: np.ndarray, target_r2: float = 0.20) -> float:
    """
    Calibrate error variance to achieve target R².

    Given the signal x = c'Γf, we want:
        R² = Var(signal) / Var(x) = Var(signal) / (Var(signal) + Var(ε))

    Solving for Var(ε):
        Var(ε) = Var(signal) * (1 - R²) / R²

    Parameters
    ----------
    Gamma : ndarray
        L×K mapping matrix
    factors : ndarray
        T×K factor matrix
    characteristics : ndarray
        N×T×L characteristics array
    target_r2 : float
        Target R² (default 0.20 as in the paper)

    Returns
    -------
    float
        Calibrated error standard deviation
    """
    N, T, L = characteristics.shape
    K = factors.shape[1]

    # Compute signal for all observations
    signals = []
    for i in range(N):
        for t in range(T):
            c_it = characteristics[i, t]  # L-vector
            f_t = factors[t]               # K-vector
            beta_it = Gamma.T @ c_it       # K-vector (loadings)
            signal = beta_it @ f_t         # scalar
            signals.append(signal)

    signal_var = np.var(signals)

    # Var(ε) = Var(signal) * (1 - R²) / R²
    error_var = signal_var * (1 - target_r2) / target_r2
    error_std = np.sqrt(error_var)

    return error_std


def generate_ipca_data(
    N: int = 200,
    T: int = 200,
    K: int = 2,
    L: int = 10,
    phi_f: float = 0.9,
    phi_c: float = 0.95,
    target_r2: float = 0.20,
    Gamma: Optional[np.ndarray] = None,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generate synthetic panel data following the IPCA model.

    Model:
        x_{i,t} = β_{i,t}' f_t + ε_{i,t}
        β_{i,t} = Γ' c_{i,t}

    Data Generating Process (following Section 7 of Kelly, Pruitt, Su 2020):
        1. Factors f_t follow a VAR(1) process
        2. Characteristics c_{i,t} follow entity-specific VAR(1) processes
        3. Errors ε_{i,t} are i.i.d. N(0, σ²) calibrated to target R²
        4. Outcomes computed from the model equation

    Parameters
    ----------
    N : int
        Number of entities (cross-sectional dimension). Default 200.
    T : int
        Number of time periods. Default 200.
    K : int
        Number of latent factors. Default 2.
    L : int
        Number of observable characteristics. Default 10.
    phi_f : float
        AR(1) coefficient for factors. Default 0.9.
    phi_c : float
        AR(1) coefficient for characteristics. Default 0.95.
    target_r2 : float
        Target R² for calibrating error variance. Default 0.20.
    Gamma : ndarray or None
        L×K mapping matrix. If None, generated randomly.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary containing:
        - 'X': Panel data array (N, T) of outcomes
        - 'factors': Factor array (T, K)
        - 'characteristics': Characteristics array (N, T, L)
        - 'Gamma': True Gamma matrix (L, K)
        - 'betas': True betas array (N, T, K)
        - 'errors': Error array (N, T)
        - 'params': Dictionary of simulation parameters
        - 'df': DataFrame formatted for IPCA estimation
    """
    if seed is not None:
        np.random.seed(seed)

    # =========================================================================
    # Step 1: Generate Gamma matrix (if not provided)
    # =========================================================================
    if Gamma is None:
        # Random Gamma with some structure
        # Scale to have reasonable magnitudes
        Gamma = np.random.randn(L, K) * 0.5
    else:
        Gamma = np.asarray(Gamma)
        assert Gamma.shape == (L, K), f"Gamma must be ({L}, {K}), got {Gamma.shape}"

    # =========================================================================
    # Step 2: Generate factors via VAR(1)
    # =========================================================================
    # Factor covariance matrix (identity for simplicity)
    sigma_f = np.eye(K)
    factors = generate_var1_process(T, K, phi_f, sigma_f)

    # =========================================================================
    # Step 3: Generate characteristics via panel VAR(1)
    # =========================================================================
    # Characteristic covariance matrix
    # Add some correlation structure
    sigma_c = np.eye(L)
    for i in range(L):
        for j in range(L):
            if i != j:
                sigma_c[i, j] = 0.3 * np.exp(-0.5 * abs(i - j))

    characteristics = generate_panel_var1(N, T, L, phi_c, sigma_c)

    # =========================================================================
    # Step 4: Compute betas (factor loadings)
    # =========================================================================
    # β_{i,t} = Γ' c_{i,t}  => shape (N, T, K)
    betas = np.zeros((N, T, K))
    for i in range(N):
        for t in range(T):
            betas[i, t] = Gamma.T @ characteristics[i, t]

    # =========================================================================
    # Step 5: Calibrate and generate errors
    # =========================================================================
    error_std = calibrate_error_variance(Gamma, factors, characteristics, target_r2)
    errors = np.random.randn(N, T) * error_std

    # =========================================================================
    # Step 6: Generate outcomes
    # =========================================================================
    # x_{i,t} = β_{i,t}' f_t + ε_{i,t}
    X = np.zeros((N, T))
    for i in range(N):
        for t in range(T):
            X[i, t] = betas[i, t] @ factors[t] + errors[i, t]

    # =========================================================================
    # Step 7: Create DataFrame for IPCA estimation
    # =========================================================================
    df = _create_ipca_dataframe(X, characteristics, N, T, L)

    # =========================================================================
    # Compile results
    # =========================================================================
    params = {
        'N': N,
        'T': T,
        'K': K,
        'L': L,
        'phi_f': phi_f,
        'phi_c': phi_c,
        'target_r2': target_r2,
        'error_std': error_std,
        'seed': seed
    }

    results = {
        'X': X,
        'factors': factors,
        'characteristics': characteristics,
        'Gamma': Gamma,
        'betas': betas,
        'errors': errors,
        'params': params,
        'df': df
    }

    return results


def _create_ipca_dataframe(X: np.ndarray, characteristics: np.ndarray, 
                          N: int, T: int, L: int) -> pd.DataFrame:
    """
    Create a DataFrame formatted for IPCA estimation.

    The IPCA class expects:
    - MultiIndex with (date, entity_id)
    - First column: outcome variable
    - Remaining columns: characteristics
    """
    rows = []

    for t in range(T):
        for i in range(N):
            row = {
                'date': t,
                'entity': i,
                'outcome': X[i, t]
            }
            for l in range(L):
                row[f'char_{l+1}'] = characteristics[i, t, l]
            rows.append(row)

    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime('2000-01-01') + pd.to_timedelta(df['date'] * 30, unit='D')
    df['date'] = df['date'].dt.to_period('M')
    df = df.set_index(['date', 'entity'])

    return df


def run_single_simulation(N: int = 200, T: int = 200, K: int = 2, L: int = 10, 
                         seed: Optional[int] = None, verbose: bool = True) -> Dict[str, Any]:
    """
    Run a single IPCA simulation and return estimation results.

    Parameters
    ----------
    N, T, K, L : int
        Simulation dimensions
    seed : int
        Random seed
    verbose : bool
        Print progress information

    Returns
    -------
    dict
        Contains true parameters, estimates, and estimation errors
    """
    from ipca import ipca

    # Generate data
    if verbose:
        print(f"Generating data: N={N}, T={T}, K={K}, L={L}")

    data = generate_ipca_data(N=N, T=T, K=K, L=L, seed=seed)

    # Fit IPCA
    if verbose:
        print("Fitting IPCA model...")

    model = ipca(RZ=data['df'], return_column='outcome', add_constant=False)
    results = model.fit(
        K=K,
        OOS=False,
        R_fit=True,
        dispIters=verbose,
        dispItersInt=100,
        maxIters=2000,
        minTol=1e-5
    )

    # Extract estimated Gamma
    Gamma_hat = results['Gamma'].values if 'Gamma' in results else None

    # Compute estimation error (accounting for rotation indeterminacy)
    if Gamma_hat is not None:
        Gamma_true = data['Gamma']
        estimation_error = compute_gamma_error(Gamma_true, Gamma_hat)
    else:
        estimation_error = None

    return {
        'true_Gamma': data['Gamma'],
        'estimated_Gamma': Gamma_hat,
        'estimation_error': estimation_error,
        'results': results,
        'data': data
    }


def compute_gamma_error(Gamma_true: np.ndarray, Gamma_hat: np.ndarray) -> Dict[str, Any]:
    """
    Compute estimation error for Gamma, accounting for rotation indeterminacy.

    IPCA identifies Gamma only up to a rotation matrix H, so we find the
    optimal rotation that minimizes ||Gamma_true - Gamma_hat @ H||_F

    Parameters
    ----------
    Gamma_true : ndarray
        True Gamma matrix (L, K)
    Gamma_hat : ndarray
        Estimated Gamma matrix (L, K)

    Returns
    -------
    dict
        Contains rotation matrix H, rotated estimate, and Frobenius error
    """
    # Solve Procrustes problem: find H that minimizes ||Gamma_true - Gamma_hat @ H||
    # Solution: H = V @ U' where Gamma_hat' @ Gamma_true = U @ S @ V'

    M = Gamma_hat.T @ Gamma_true
    U, S, Vt = linalg.svd(M)
    H = U @ Vt

    # Rotate estimate
    Gamma_hat_rotated = Gamma_hat @ H

    # Compute error
    error_matrix = Gamma_true - Gamma_hat_rotated
    frobenius_error = linalg.norm(error_matrix, 'fro')
    elementwise_errors = error_matrix.flatten()

    return {
        'H': H,
        'Gamma_hat_rotated': Gamma_hat_rotated,
        'frobenius_error': frobenius_error,
        'elementwise_errors': elementwise_errors,
        'rmse': np.sqrt(np.mean(elementwise_errors**2))
    }


def run_monte_carlo(
    n_simulations: int = 200,
    N: int = 200,
    T: int = 200,
    K: int = 2,
    L: int = 10,
    base_seed: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run Monte Carlo simulation study as in Section 7 of the paper.

    Parameters
    ----------
    n_simulations : int
        Number of Monte Carlo replications. Default 200.
    N, T, K, L : int
        Simulation dimensions
    base_seed : int
        Base random seed (incremented for each simulation)
    verbose : bool
        Print progress

    Returns
    -------
    dict
        Contains all estimation errors and summary statistics
    """
    from ipca import ipca

    all_errors = []
    all_frobenius = []
    all_rmse = []

    # Use a fixed Gamma across all simulations (like in the paper)
    np.random.seed(base_seed)
    true_Gamma = np.random.randn(L, K) * 0.5

    for sim in range(n_simulations):
        if verbose and (sim + 1) % 10 == 0:
            print(f"Simulation {sim + 1}/{n_simulations}")

        seed = base_seed + sim + 1

        # Generate data with fixed Gamma
        data = generate_ipca_data(
            N=N, T=T, K=K, L=L,
            Gamma=true_Gamma,
            seed=seed
        )

        # Fit IPCA
        model = ipca(RZ=data['df'], return_column='outcome', add_constant=False)

        try:
            results = model.fit(
                K=K,
                OOS=False,
                R_fit=False,
                dispIters=False,
                maxIters=2000,
                minTol=1e-5
            )

            # Compute errors
            Gamma_hat = results['Gamma'].values
            error_info = compute_gamma_error(true_Gamma, Gamma_hat)

            all_errors.append(error_info['elementwise_errors'])
            all_frobenius.append(error_info['frobenius_error'])
            all_rmse.append(error_info['rmse'])

        except Exception as e:
            if verbose:
                print(f"  Simulation {sim + 1} failed: {e}")
            continue

    # Compile results
    all_errors = np.array(all_errors)

    summary = {
        'n_successful': len(all_frobenius),
        'n_failed': n_simulations - len(all_frobenius),
        'mean_frobenius': np.mean(all_frobenius),
        'std_frobenius': np.std(all_frobenius),
        'mean_rmse': np.mean(all_rmse),
        'std_rmse': np.std(all_rmse),
        'elementwise_mean': np.mean(all_errors, axis=0),
        'elementwise_std': np.std(all_errors, axis=0),
    }

    if verbose:
        print("\n" + "="*60)
        print("MONTE CARLO RESULTS")
        print("="*60)
        print(f"Successful simulations: {summary['n_successful']}/{n_simulations}")
        print(f"Mean Frobenius error: {summary['mean_frobenius']:.4f}")
        print(f"Std Frobenius error: {summary['std_frobenius']:.4f}")
        print(f"Mean RMSE: {summary['mean_rmse']:.4f}")
        print(f"Std RMSE: {summary['std_rmse']:.4f}")

    return {
        'true_Gamma': true_Gamma,
        'all_errors': all_errors,
        'all_frobenius': all_frobenius,
        'all_rmse': all_rmse,
        'summary': summary,
        'params': {'N': N, 'T': T, 'K': K, 'L': L, 'n_simulations': n_simulations}
    }


# =============================================================================
# Example usage and testing
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("IPCA Synthetic Data Generation - Demo")
    print("="*60)

    # Generate a single dataset
    print("\n1. Generating synthetic data...")
    data = generate_ipca_data(N=100, T=100, K=2, L=5, seed=42)

    print(f"\nGenerated data summary:")
    print(f"  Outcomes shape: {data['X'].shape}")
    print(f"  Factors shape: {data['factors'].shape}")
    print(f"  Characteristics shape: {data['characteristics'].shape}")
    print(f"  Gamma shape: {data['Gamma'].shape}")
    print(f"  DataFrame shape: {data['df'].shape}")
    print(f"  Target R²: {data['params']['target_r2']}")
    print(f"  Calibrated error std: {data['params']['error_std']:.4f}")

    # Verify R²
    signal_var = np.var(data['X'] - data['errors'])
    total_var = np.var(data['X'])
    actual_r2 = signal_var / total_var
    print(f"  Actual R²: {actual_r2:.4f}")

    print("\n2. True Gamma matrix:")
    print(data['Gamma'])

    print("\n3. DataFrame head (for IPCA input):")
    print(data['df'].head(10))

    # Run single simulation with IPCA estimation
    print("\n" + "="*60)
    print("4. Running single IPCA estimation...")
    print("="*60)

    try:
        sim_result = run_single_simulation(N=100, T=100, K=2, L=5, seed=123)

        print("\nTrue Gamma:")
        print(sim_result['true_Gamma'])

        print("\nEstimated Gamma (after rotation alignment):")
        if sim_result['estimation_error'] is not None:
            print(sim_result['estimation_error']['Gamma_hat_rotated'])
            print(f"\nFrobenius error: {sim_result['estimation_error']['frobenius_error']:.4f}")
            print(f"RMSE: {sim_result['estimation_error']['rmse']:.4f}")
    except ImportError:
        print("Note: ipca module not found. Skipping estimation demo.")
    except Exception as e:
        print(f"Note: Estimation demo failed: {e}")
