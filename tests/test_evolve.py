from typing import cast
from unittest.mock import patch

import astropy.constants as cst
import astropy.units as u
import numpy as np

from pyc2ray.evolve import evolve3D
from pyc2ray.radiation.blackbody import BlackBodySource
from pyc2ray.radiation.common import make_tau_table


def call_evolve3D(use_gpu: bool = False, use_mpi: int | None = None):
    # Define some random sources
    N = 32

    rng = np.random.default_rng(918)
    src_pos = rng.integers(0, N, size=(3, 10), dtype=np.int32)
    src_flux = rng.uniform(1e10, 1e14, size=10).astype(np.float64)
    src_flux *= 1e-46

    shape = (N, N, N)
    ndens = np.empty(shape, order="F")
    xh = np.full(shape, 1.2e-3, order="F")
    temp = np.full(shape, 1e4, order="F")
    clump = np.full(shape, 1.0, dtype=np.float64, order="F")

    minlogtau, maxlogtau, num_tau = -20.0, 4.0, 20000
    tau, dlogtau = make_tau_table(minlogtau, maxlogtau, num_tau)
    freq_min, freq_max = (
        (13.598 * u.eV / cst.h).to("Hz").value,
        (54.416 * u.eV / cst.h).to("Hz").value,
    )
    sig = 6.30e-18
    radsource = BlackBodySource(1e5, False, freq_min, sig)
    photo_thin_table, photo_thick_table = radsource.make_photo_table(
        tau, freq_min, freq_max, 1e48
    )

    rank = 0 if use_mpi is None else cast(int, use_mpi)

    colh0 = 1.3e-8 * 0.83 * 1.0 / 13.598**2
    temph0 = 13.598 / (cst.k_B * u.K).to("eV").value

    return evolve3D(
        dt=1e3,
        dr=(1 * u.Mpc).cgs.value / N,
        src_flux=src_flux,
        src_pos=src_pos,
        src_batch_size=8,
        use_gpu=use_gpu,
        max_subbox=1000,
        subboxsize=128,
        loss_fraction=1e-2,
        use_mpi=bool(use_mpi),
        rank=rank,
        nprocs=8,
        temp=temp,
        ndens=ndens,
        xh=xh,
        clump=clump,
        photo_thin_table=photo_thin_table,
        photo_thick_table=photo_thick_table,
        minlogtau=minlogtau,
        dlogtau=dlogtau,
        R_max_LLS=15.0,
        convergence_fraction=1e-4,
        sig=sig,
        bh00=2.59e-13,
        albpow=-0.7,
        colh0=colh0,
        temph0=temph0,
        abu_c=7.1e-7,
    )


@patch("pyc2ray.lib.libasora")
@patch("pyc2ray.lib.libc2ray")
def test_evolve3D_no_gpu_no_mpi(mock_c2ray, mock_asora):
    call_evolve3D()
    mock_asora.assert_not_called()
    mock_c2ray.assert_called()
