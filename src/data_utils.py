"""
Data utilities for IPCA asset pricing analysis.

Functions for loading, preparing, and evaluating IPCA models on real data.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict


def next_month(yyyymm: int) -> int:
    """Calculate next month's YYYYMM integer."""
    year = yyyymm // 100
    month = yyyymm % 100
    if month == 12:
        return (year + 1) * 100 + 1
    else:
        return year * 100 + (month + 1)


def rank_transform(df: pd.DataFrame, char_cols: List[str],
                   time_col: str = 'YYYYMM') -> pd.DataFrame:
    """
    Transform characteristics to cross-sectional ranks in [-0.5, 0.5].

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with characteristics
    char_cols : list
        List of characteristic column names
    time_col : str
        Name of time column for grouping

    Returns
    -------
    pd.DataFrame
        DataFrame with rank-transformed characteristics
    """
    df_ranked = df.copy()
    for col in char_cols:
        df_ranked[col] = df.groupby(time_col)[col].transform(
            lambda x: (x.rank(method='average', na_option='keep') - 1) / max(x.count() - 1, 1) - 0.5
            if x.count() > 1 else 0
        )
        df_ranked[col] = df_ranked[col].fillna(0)
    return df_ranked


def create_balanced_panel(df: pd.DataFrame, char_cols: List[str],
                          min_obs_per_stock: int = 60,
                          max_stocks: int = 500,
                          add_intercept: bool = True,
                          verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, List, List, int, int, int]:
    """
    Create a balanced panel by selecting stocks with sufficient observations.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: permno, YYYYMM, ret, and characteristic columns
    char_cols : list
        List of characteristic column names
    min_obs_per_stock : int
        Minimum observations required per stock
    max_stocks : int
        Maximum number of stocks to include
    add_intercept : bool
        Whether to add intercept column to characteristics
    verbose : bool
        Print progress information

    Returns
    -------
    rets : np.ndarray (T, N)
        Returns array
    Z : np.ndarray (T, N, L)
        Characteristics array
    times : list
        List of time periods (YYYYMM)
    stocks : list
        List of stock identifiers (permno)
    T : int
        Number of time periods
    N : int
        Number of stocks
    L : int
        Number of characteristics (including intercept if added)
    """
    # Count observations per stock
    stock_counts = df.groupby('permno').size()
    valid_stocks = stock_counts[stock_counts >= min_obs_per_stock].index

    if verbose:
        print(f"Stocks with >= {min_obs_per_stock} observations: {len(valid_stocks)}")

    df_filtered = df[df['permno'].isin(valid_stocks)].copy()

    # Pivot to wide format to find stocks present in all periods
    times = sorted(df_filtered['YYYYMM'].unique())
    ret_pivot = df_filtered.pivot_table(index='YYYYMM', columns='permno', values='ret', aggfunc='first')

    # Find stocks present in all periods
    complete_stocks = ret_pivot.dropna(axis=1).columns.tolist()

    if verbose:
        print(f"Stocks present in all {len(times)} periods: {len(complete_stocks)}")

    if len(complete_stocks) < 100:
        # Use stocks with most observations
        stock_presence = ret_pivot.notna().sum()
        complete_stocks = stock_presence.nlargest(max_stocks).index.tolist()

        if verbose:
            print(f"Using top {max_stocks} stocks by presence: {len(complete_stocks)}")

        # Filter to times where we have data for these stocks
        df_balanced = df_filtered[df_filtered['permno'].isin(complete_stocks)].copy()
        ret_pivot = df_balanced.pivot_table(index='YYYYMM', columns='permno', values='ret', aggfunc='first')

        # Drop times with any missing
        valid_times = ret_pivot.dropna(axis=0).index.tolist()

        if verbose:
            print(f"Time periods with complete data: {len(valid_times)}")

        df_balanced = df_balanced[df_balanced['YYYYMM'].isin(valid_times)]
        times = valid_times
        stocks = complete_stocks
    else:
        # Limit to max_stocks if we have too many
        if len(complete_stocks) > max_stocks:
            complete_stocks = complete_stocks[:max_stocks]
            if verbose:
                print(f"Limiting to {max_stocks} stocks")

        df_balanced = df_filtered[df_filtered['permno'].isin(complete_stocks)].copy()
        stocks = complete_stocks

    # Sort for consistent ordering
    df_balanced = df_balanced.sort_values(['YYYYMM', 'permno'])

    T = len(times)
    N = len(stocks)
    L = len(char_cols) + (1 if add_intercept else 0)

    if verbose:
        print(f"\nCreating arrays: T={T}, N={N}, L={L}")

    # Create arrays
    rets = np.zeros((T, N))
    Z = np.zeros((T, N, L))

    stock_to_idx = {s: i for i, s in enumerate(sorted(stocks))}
    time_to_idx = {t: i for i, t in enumerate(times)}

    for _, row in df_balanced.iterrows():
        t_idx = time_to_idx.get(row['YYYYMM'])
        s_idx = stock_to_idx.get(row['permno'])

        if t_idx is not None and s_idx is not None:
            rets[t_idx, s_idx] = row['ret']
            if add_intercept:
                Z[t_idx, s_idx, :-1] = row[char_cols].values
                Z[t_idx, s_idx, -1] = 1.0
            else:
                Z[t_idx, s_idx, :] = row[char_cols].values

    # Fill any remaining NaN with 0
    Z = np.nan_to_num(Z, nan=0.0)
    rets = np.nan_to_num(rets, nan=0.0)

    return rets, Z, times, stocks, T, N, L


def compute_r2(rets: np.ndarray, Z: np.ndarray, Gamma, factors, Lambda) -> Tuple[float, float]:
    """
    Compute Total R² and Predictive R².

    Parameters
    ----------
    rets : np.ndarray (T, N)
        Returns array
    Z : np.ndarray (T, N, L)
        Characteristics array
    Gamma : np.ndarray or pd.DataFrame (L, K)
        Factor loading matrix
    factors : np.ndarray or pd.DataFrame (K, T)
        Time-varying factors
    Lambda : np.ndarray or pd.Series (K,)
        Mean factors (risk premia)

    Returns
    -------
    r2_total : float
        Total R² using time-varying factors
    r2_pred : float
        Predictive R² using mean factors (Lambda)
    """
    T, N = rets.shape

    # Flatten actual returns
    actual = rets.flatten()

    # Convert to numpy if needed
    Gamma_np = Gamma.values if hasattr(Gamma, 'values') else Gamma
    Lambda_np = Lambda.values if hasattr(Lambda, 'values') else Lambda

    # Compute fitted values
    fitted_total = np.zeros((T, N))
    fitted_pred = np.zeros((T, N))

    for t in range(T):
        Z_t = Z[t, :, :]  # (N, L)
        Beta_t = Z_t @ Gamma_np  # (N, K)

        # Time-varying factors
        if hasattr(factors, 'iloc'):
            f_t = factors.iloc[:, t].values
        else:
            f_t = factors[:, t]
        fitted_total[t, :] = Beta_t @ f_t

        # Constant factors (Lambda)
        fitted_pred[t, :] = Beta_t @ Lambda_np

    fitted_total = fitted_total.flatten()
    fitted_pred = fitted_pred.flatten()

    # R² vs zero benchmark
    ss_total = np.sum(actual**2)

    r2_total = 1 - np.sum((actual - fitted_total)**2) / ss_total
    r2_pred = 1 - np.sum((actual - fitted_pred)**2) / ss_total

    return r2_total, r2_pred


def compute_oos_r2(df_oos: pd.DataFrame, char_cols: List[str],
                   Gamma, factors, Lambda, times: List,
                   add_intercept: bool = True) -> Tuple[float, float]:
    """
    Compute out-of-sample R² for new observations.

    Parameters
    ----------
    df_oos : pd.DataFrame
        Out-of-sample data with columns: ret, YYYYMM, and characteristics
    char_cols : list
        List of characteristic column names
    Gamma : np.ndarray or pd.DataFrame (L, K)
        Estimated factor loading matrix
    factors : np.ndarray or pd.DataFrame (K, T)
        Estimated time-varying factors
    Lambda : np.ndarray or pd.Series (K,)
        Estimated mean factors
    times : list
        List of time periods from training sample
    add_intercept : bool
        Whether intercept was included in characteristics

    Returns
    -------
    r2_total : float
        Total R² using time-varying factors
    r2_pred : float
        Predictive R² using mean factors
    """
    actual = df_oos['ret'].values

    # Prepare characteristics
    Z_oos = df_oos[char_cols].values
    if add_intercept:
        Z_oos = np.hstack([Z_oos, np.ones((len(df_oos), 1))])

    # Convert to numpy if needed
    Gamma_np = Gamma.values if hasattr(Gamma, 'values') else Gamma
    Lambda_np = Lambda.values if hasattr(Lambda, 'values') else Lambda

    # Compute loadings
    Beta_oos = Z_oos @ Gamma_np  # (n_obs, K)

    # Fitted values
    fitted_total = np.zeros(len(df_oos))
    fitted_pred = np.zeros(len(df_oos))

    time_to_idx = {t: i for i, t in enumerate(times)}
    yyyymm_values = df_oos['YYYYMM'].values

    for i in range(len(df_oos)):
        t = yyyymm_values[i]

        # Predictive fit (using Lambda)
        fitted_pred[i] = Beta_oos[i] @ Lambda_np

        # Total fit (using time-varying factors if available)
        if t in time_to_idx:
            t_idx = time_to_idx[t]
            if hasattr(factors, 'iloc'):
                f_t = factors.iloc[:, t_idx].values
            else:
                f_t = factors[:, t_idx]
            fitted_total[i] = Beta_oos[i] @ f_t
        else:
            fitted_total[i] = fitted_pred[i]

    # R² vs zero benchmark
    ss_total = np.sum(actual**2)

    r2_total = 1 - np.sum((actual - fitted_total)**2) / ss_total
    r2_pred = 1 - np.sum((actual - fitted_pred)**2) / ss_total

    return r2_total, r2_pred
