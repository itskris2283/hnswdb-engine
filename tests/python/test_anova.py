import pytest
import numpy as np
from benchmarks.anova import compute_two_way_anova


def test_anova_pure_row_effect():
    """When only rows (data seeds) vary and columns are identical, data share should be 100%."""
    # 5 rows, 5 columns: each row has a different constant
    row_effects = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    mat = np.tile(row_effects[:, None], (1, 5))  # (5, 5)

    res = compute_two_way_anova(mat)
    assert np.isclose(res["share_data_percent"], 100.0, atol=1e-5)
    assert np.isclose(res["share_build_percent"], 0.0, atol=1e-5)
    assert np.isclose(res["share_error_percent"], 0.0, atol=1e-5)
    assert np.isclose(res["ss_build"], 0.0, atol=1e-5)
    assert np.isclose(res["ss_error"], 0.0, atol=1e-5)


def test_anova_pure_col_effect():
    """When only columns (build seeds) vary, build share should be 100%."""
    col_effects = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    mat = np.tile(col_effects[None, :], (5, 1))

    res = compute_two_way_anova(mat)
    assert np.isclose(res["share_data_percent"], 0.0, atol=1e-5)
    assert np.isclose(res["share_build_percent"], 100.0, atol=1e-5)
    assert np.isclose(res["share_error_percent"], 0.0, atol=1e-5)


def test_anova_equal_shares():
    """Construct data with equal row and col effect sum-of-squares and zero residual."""
    effects = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    mat = effects[:, None] + effects[None, :]

    res = compute_two_way_anova(mat)
    assert np.isclose(res["share_data_percent"], 50.0, atol=1e-5)
    assert np.isclose(res["share_build_percent"], 50.0, atol=1e-5)
    assert np.isclose(res["share_error_percent"], 0.0, atol=1e-5)


def test_anova_known_three_way_split():
    """Construct data where row SS = 20, col SS = 30, residual SS = 50 -> 20%, 30%, 50% shares."""
    # Orthogonal construction for 3x3 matrix:
    # row means centered, col means centered, interaction
    alpha = np.array([np.sqrt(2), -np.sqrt(2), 0.0])  # sum = 0, sum sq = 4 -> 3*4 = 12
    # Let's generate a general matrix and scale components
    np.random.seed(42)
    A = np.random.randn(5, 1)
    A -= A.mean()
    B = np.random.randn(1, 5)
    B -= B.mean()
    E = np.random.randn(5, 5)
    E = E - E.mean(axis=1, keepdims=True) - E.mean(axis=0, keepdims=True) + E.mean()

    # Scale components
    # SS_A = 5 * sum(A^2), SS_B = 5 * sum(B^2), SS_E = sum(E^2)
    A = A / np.sqrt(5 * np.sum(A**2)) * np.sqrt(20.0)
    B = B / np.sqrt(5 * np.sum(B**2)) * np.sqrt(30.0)
    E = E / np.sqrt(np.sum(E**2)) * np.sqrt(50.0)

    mat = 10.0 + A + B + E
    res = compute_two_way_anova(mat)

    assert np.isclose(res["share_data_percent"], 20.0, atol=1e-3)
    assert np.isclose(res["share_build_percent"], 30.0, atol=1e-3)
    assert np.isclose(res["share_error_percent"], 50.0, atol=1e-3)


def test_anova_invalid_inputs():
    """Invalid dimensions or non-2D matrices must raise ValueError."""
    with pytest.raises(ValueError):
        compute_two_way_anova(np.array([1, 2, 3]))

    with pytest.raises(ValueError):
        compute_two_way_anova(np.zeros((1, 5)))

    with pytest.raises(ValueError):
        compute_two_way_anova(np.zeros((5, 1)))
