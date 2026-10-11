"""
Two-way Analysis of Variance (ANOVA) decomposition without replication.
Computes data seed SS, build seed SS, residual (interaction) SS, variance shares, and F-statistics.
"""

import numpy as np


def compute_two_way_anova(matrix: np.ndarray) -> dict:
    """
    Computes two-way ANOVA decomposition for a 2D matrix (rows = factor A / data seed, cols = factor B / build seed).
    
    Args:
        matrix: 2D numpy array of shape (I, J), where I >= 2 and J >= 2.
        
    Returns:
        dict with:
            grand_mean, ss_total, ss_data, ss_build, ss_error,
            share_data_percent, share_build_percent, share_error_percent,
            f_data, f_build
    """
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError(f"Matrix must be 2D, got {mat.ndim}D")
    I, J = mat.shape
    if I < 2 or J < 2:
        raise ValueError(f"Matrix must have at least 2 rows and 2 columns, got shape ({I}, {J})")

    grand_mean = float(np.mean(mat))
    ss_total = float(np.sum((mat - grand_mean) ** 2))

    if ss_total == 0.0:
        # Zero variance: all values are identical
        return {
            "grand_mean": grand_mean,
            "ss_total": 0.0,
            "ss_data": 0.0,
            "ss_build": 0.0,
            "ss_error": 0.0,
            "share_data_percent": 0.0,
            "share_build_percent": 0.0,
            "share_error_percent": 0.0,
            "f_data": 0.0,
            "f_build": 0.0,
        }

    row_means = np.mean(mat, axis=1)  # shape (I,)
    col_means = np.mean(mat, axis=0)  # shape (J,)

    ss_data = float(J * np.sum((row_means - grand_mean) ** 2))
    ss_build = float(I * np.sum((col_means - grand_mean) ** 2))
    ss_error = float(np.sum((mat - row_means[:, None] - col_means[None, :] + grand_mean) ** 2))

    # Guard against float rounding
    ss_error = max(0.0, ss_error)

    share_data_percent = float(100.0 * (ss_data / ss_total))
    share_build_percent = float(100.0 * (ss_build / ss_total))
    share_error_percent = float(100.0 * (ss_error / ss_total))

    df_data = I - 1
    df_build = J - 1
    df_error = (I - 1) * (J - 1)

    ms_data = ss_data / df_data
    ms_build = ss_build / df_build
    ms_error = ss_error / df_error if df_error > 0 else 0.0

    f_data = float(ms_data / ms_error) if ms_error > 0 else 0.0
    f_build = float(ms_build / ms_error) if ms_error > 0 else 0.0

    return {
        "grand_mean": grand_mean,
        "ss_total": ss_total,
        "ss_data": ss_data,
        "ss_build": ss_build,
        "ss_error": ss_error,
        "share_data_percent": share_data_percent,
        "share_build_percent": share_build_percent,
        "share_error_percent": share_error_percent,
        "f_data": f_data,
        "f_build": f_build,
    }
