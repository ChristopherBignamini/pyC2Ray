from pathlib import Path

import numpy as np
import pytest

try:
    from pyc2ray.lib import libasoratest as asoratest
except ImportError:
    pytest.skip("libasoratest.so missing, skipping tests", allow_module_level=True)


def test_cinterp(data_dir: Path) -> None:
    rng = np.random.default_rng(seed=42)
    N = 11
    dens = rng.random((N, N, N), dtype=np.float64)

    cdens, path = asoratest.cinterp(dens)
    expected_output = np.load(data_dir / "cinterp_output.npz")

    assert np.allclose(cdens, expected_output["cdens"])
    assert np.allclose(path, expected_output["path"])

    expected_output.close()


def linthrd2cart(q: int, s: int) -> tuple[int, int, int]:
    """Reference function for shell mapping to cartesian coordinates"""
    if s == 0:
        return q, 0, 0

    s_top = 2 * q * (q + 1) + 1
    if s == s_top:
        return q - 1, 0, -1

    def get_ij(q: int, s: int) -> tuple[int, int]:
        j, i = divmod(s - 1, 2 * q)
        i += j - q
        if i + j > q:
            i -= q
            j -= q + 1
        return i, j

    if s < s_top:
        sgn = 1
        i, j = get_ij(q, s)
    else:
        sgn = -1
        i, j = get_ij(q - 1, s - s_top)

    return i, j, sgn * (q - abs(i) - abs(j))


@pytest.mark.parametrize("q", range(0, 50))
def test_shell_mapping(q: int) -> None:
    q_max = 4 * q**2 + 2 if q > 0 else 1
    for s in range(q_max):
        ijk = linthrd2cart(q, s)
        assert ijk == asoratest.linthrd2cart(q, s)
        assert (q, s) == asoratest.cart2linthrd(*ijk)


def test_cells_in_shell() -> None:
    assert asoratest.cells_in_shell(0) == 1
    for q in range(1, 50):
        assert asoratest.cells_in_shell(q) == 4 * q**2 + 2


def test_cells_to_shell() -> None:
    q_tot = 1
    assert asoratest.cells_to_shell(0) == q_tot
    for q in range(1, 50):
        q_tot += 4 * q**2 + 2
        assert asoratest.cells_to_shell(q) == q_tot
