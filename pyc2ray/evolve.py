"""This file contains the main time-evolution subroutine, which updates
the ionization state of the whole grid over one timestep, using the
C2Ray method.

The raytracing step can use either the sequential (subbox, cubic)
technique which runs in Fortran on the CPU or the accelerated technique,
which runs using the ASORA library on the GPU.

When using the latter, some notes apply:
For performance reasons, the program minimizes the frequency at which
data is moved between the CPU and the GPU (this is a big bottleneck).
In particular, the radiation tables, which in principle shouldn't change
over the run of a simulation, need to be copied separately to the GPU
using the photo_table_to_device() method of the module. This is done
automatically when using the C2Ray subclasses but must be done manually
if for some reason you are calling the evolve3D routine directly without
using the C2Ray subclasses.

This file defines two variants of evolve3D: The reference, single-gpu
version, and a MPI version which enables usage on multiple GPU nodes.
"""

import array
import logging
import time

import numpy as np
from mpi4py import MPI

from pyc2ray.asora_core import is_device_init
from pyc2ray.load_extensions import libasora, libc2ray
from pyc2ray.utils.logutils import disable_newline
from pyc2ray.utils.other_utils import display_seconds, distribute_jobs
from pyc2ray.utils.sourceutils import FloatArray, IntArray, format_sources

__all__ = ["evolve3D"]

logger = logging.getLogger(__name__)


def evolve3D(
    dt: float,
    dr: float,
    src_flux: FloatArray,
    src_pos: IntArray,
    src_batch_size: int,
    use_gpu: bool,
    max_subbox: int,
    subboxsize: int,
    loss_fraction: float,
    use_mpi: bool,
    rank: int,
    nprocs: int,
    temp: FloatArray,
    ndens: FloatArray,
    xh: FloatArray,
    clump: FloatArray,
    photo_thin_table: FloatArray,
    photo_thick_table: FloatArray,
    minlogtau: float,
    dlogtau: float,
    R_max_LLS: float,
    convergence_fraction: float,
    sig: float,
    bh00: float,
    albpow: float,
    colh0: float,
    temph0: float,
    abu_c: float,
) -> tuple[FloatArray, FloatArray]:
    """Evolves the ionization fraction over one timestep for the whole grid

    Warning: Calling this function with use_gpu = True assumes that the radiation tables have previously been
    copied to the GPU using photo_table_to_device()

    Parameters
    ----------
    dt
        Timestep in seconds
    dr
        Cell dimension in each direction in cm.
    src_flux
        Array containing the total ionizing flux of each source, normalized by S_star (1e48 by default).
    src_pos
        Array containing the 3D grid position of each source, in Fortran indexing (from 1).
    use_gpu
        Whether or not to use the GPU-accelerated ASORA library for raytracing.
    max_subbox
        Maximum subbox to raytrace when using CPU cubic raytracing. Has no effect when use_gpu is true.
    subboxsize
        ...
    loss_fraction
        Fraction of remaining photons below we stop ray-tracing (subbox technique). Has no effect when use_gpu is true.
    temp
        The initial temperature of each cell in K.
    ndens
        The hydrogen number density of each cell in cm^-3.
    xh
        The initial ionized fraction of each cell.
    photo_thin_table
        Tabulated values of the integral ∫L_v*e^(-τ_v)/hv. When using GPU, this table needs to have been copied to the GPU
        in a separate (previous) step, using photo_table_to_device().
    minlogtau
        Base 10 log of the minimum value of the table in τ (excluding τ = 0).
    dlogtau
        Step size of the logτ-table.
    R_max_LLS
        Value of maximum comoving distance for photons from source (type 3 LLS in original C2Ray). This value is
        given in cell units, but doesn't need to be an integer.
    convergence_fraction
        Which fraction of the cells can be left unconverged to improve performance (usually ~ 1e-4).
    sig
        Constant photoionization cross-section of hydrogen in cm^2.
    bh00
        Hydrogen recombination parameter at 10^4 K in the case B OTS approximation.
    albpow
        Power-law index for the H recombination parameter.
    colh0
        Hydrogen collisional ionization parameter.
    temph0
        Hydrogen ionization energy expressed in K.
    abu_c
        Carbon abundance.

    Returns
    -------
    xh_new : 3D-array of dtype float
        The updated ionization fraction of each cell at the end of the timestep.
    phi_ion : 3D-array of dtype float
        Photoionization rate of each cell due to all sources.
    """

    if use_gpu and not is_device_init():
        raise RuntimeError(
            "GPU not initialized. Please initialize it by calling device_init(N)"
        )

    # Problem dimensions
    N = temp.shape[0]  # Mesh size
    num_cells = N * N * N  # Number of cells/points
    num_src = src_flux.shape[0]  # Number of sources
    num_tau = photo_thin_table.shape[0]

    # Convergence Criteria
    conv_criterion = min(int(convergence_fraction * num_cells), (num_src - 1) / 3)

    # Initialize convergence metrics
    prev_sum_xh1_int: float = 2 * num_cells
    prev_sum_xh0_int: float = 2 * num_cells
    converged = False
    if rank != 0:
        xh_new = np.empty_like(xh)

    # initialize average and intermediate results to values at beginning of timestep
    xh_av = np.copy(xh)
    xh_intermed = np.copy(xh)

    # When using GPU raytracing, data has to be reshaped & reformatted and copied to the device
    if use_gpu:
        # Format input data for the CUDA extension module (flat arrays, C-types,etc)
        xh_av_flat = np.ravel(xh).astype(np.float64, copy=True)

        # Get portion of source data for this rank
        if use_mpi:
            chunk = distribute_jobs(num_src, nprocs, rank)

            # overwrite number of sources
            num_src = chunk.stop - chunk.start
            srcpos_flat, normflux_flat = format_sources(
                src_pos[:, chunk], src_flux[chunk]
            )

            logger.info(f"...rank={rank:n} has {num_src:n} sources.")
        else:
            srcpos_flat, normflux_flat = format_sources(src_pos, src_flux)

        # Copy positions & fluxes of sources to the GPU in advance
        assert libasora is not None
        libasora.source_data_to_device(srcpos_flat, normflux_flat)

        # Initialize Flat Column density & ionization rate arrays.
        # These are used to store the output of the raytracing module.
        phi_ion_flat = np.zeros(num_cells, dtype=np.float64)

        # Copy density field to GPU once at the beginning of timestep (!! do_all_sources assumes this !!)
        assert libasora is not None
        # FIXME: is copy necessary?
        ndens_flat = np.ravel(ndens).astype(np.float64, copy=True)
        libasora.density_to_device(ndens_flat)
        if use_mpi:
            logger.info("Copied source data to device.")
        else:
            logger.info(f"Rank {rank} copied source data to device.")

    # -----------------------------------------------------------
    # Start Evolve step, Iterate until convergence in <x> and <y>
    # -----------------------------------------------------------
    logger.info(f"""Calling evolve3D...
dr [Mpc]: {dr / 3.086e24:.3e}
dt [years]: {dt / 3.15576e07:.3e}
Running on {num_src:n} source(s), total normalized ionizing flux: {src_flux.sum():.2e}
Mean density (cgs): {ndens.mean():.3e}, Mean ionized fraction: {xh.mean():.3e}
Convergence Criterion (Number of points): {conv_criterion: n}
""")

    if rank == 0:
        # Iteration counter
        n_count = 0

    while not converged:
        # --------------------
        # (1): Raytracing Step
        # --------------------
        with disable_newline():
            logger.info("Doing Raytracing...")

        time_start = time.perf_counter()

        # Do the raytracing part for each source. This computes the cumulative ionization rate for each cell.
        if use_gpu:
            # Use GPU raytracing
            assert libasora is not None
            libasora.do_all_sources(
                R_max_LLS,
                sig,
                dr,
                xh_av_flat,
                phi_ion_flat,
                num_src,
                N,
                minlogtau,
                dlogtau,
                num_tau,
                src_batch_size,
            )
        else:
            # Set rates to 0. When using ASORA, this is done internally by the library (directly on the GPU)
            phi_ion = np.zeros((N, N, N), order="F")
            # So far in evolve we ignore heating (not considered in chemistry),
            # but the raytracing function requires heating tables as argument
            phi_heat = np.zeros((N, N, N), order="F")
            coldensh_out = np.zeros((N, N, N), order="F")
            # Use CPU raytracing with subbox optimization
            nsubbox, photonloss = libc2ray.raytracing.do_all_sources(
                src_flux,
                src_pos,
                max_subbox,
                subboxsize,
                coldensh_out,
                sig,
                dr,
                ndens,
                xh_av,
                phi_ion,
                phi_heat,
                loss_fraction,
                photo_thin_table,
                photo_thick_table,
                np.zeros(num_tau),
                np.zeros(num_tau),  # Eventually we'll add heating tables here
                minlogtau,
                dlogtau,
                R_max_LLS,
            )

        time_end = time.perf_counter()
        logger.info(f"  took {display_seconds(time_end - time_start)}")

        # Since chemistry (ODE solving) is done on the CPU in Fortran, flattened CUDA arrays need to be reshaped
        if use_gpu:
            phi_ion = np.reshape(phi_ion_flat, (N, N, N))
        else:
            logger.info(
                f"Average number of subboxes: {nsubbox / num_src:n}, Total photon loss: {photonloss:.3e}"
            )

        if use_mpi:
            # Collect results from the different MPI processors
            if rank == 0:
                MPI.COMM_WORLD.Reduce(
                    MPI.IN_PLACE, [phi_ion, MPI.DOUBLE], op=MPI.SUM, root=0
                )
            else:
                MPI.COMM_WORLD.Reduce([phi_ion, MPI.DOUBLE], None, op=MPI.SUM, root=0)
            MPI.COMM_WORLD.Bcast([phi_ion, MPI.DOUBLE], root=0)

        if rank == 0:
            # ---------------------
            # (2): ODE Solving Step
            # ---------------------
            with disable_newline():
                logger.info("Doing Chemistry...")

            time_start = time.perf_counter()

            # Apply the global rates to compute the updated ionization fraction
            conv_flag = libc2ray.chemistry.global_pass(
                dt,
                ndens,
                temp,
                xh,
                xh_av,
                xh_intermed,
                phi_ion,
                clump,
                bh00,
                albpow,
                colh0,
                temph0,
                abu_c,
            )
            # TODO: the line below is the same function but completely in python
            # (much slower then the fortran version, due to a lot of loops)
            # xh_intermed, xh_av, conv_flag = global_pass(
            #     dt, ndens, temp, xh, xh_av, xh_intermed, phi_ion,
            #     clump, bh00, albpow, colh0, temph0, abu_c,
            # )

            time_end = time.perf_counter()
            logger.info(f"  took {display_seconds(time_end - time_start)}")

            # ----------------------------
            # (3): Test Global Convergence
            # ----------------------------
            sum_xh1_int = np.sum(xh_intermed)
            sum_xh0_int = xh_intermed.size - sum_xh1_int  # = np.sum(1 - xh_intermed)

            if sum_xh1_int > 0.0:
                rel_change_xh1 = np.abs((sum_xh1_int - prev_sum_xh1_int) / sum_xh1_int)
            else:
                rel_change_xh1 = 1.0

            if sum_xh0_int > 0.0:
                rel_change_xh0 = np.abs((sum_xh0_int - prev_sum_xh0_int) / sum_xh0_int)
            else:
                rel_change_xh0 = 1.0

            # Display convergence
            logger.info(
                f"Number of non-converged points: {conv_flag} of {num_cells} ({conv_flag / num_cells * 100: .3f} % ), "
                f"Relative change in ionfrac: {rel_change_xh1: .2e}",
            )

            converged = (conv_flag < conv_criterion) or (
                (rel_change_xh1 < convergence_fraction)
                and (rel_change_xh0 < convergence_fraction)
            )
            n_count += 1

            # Set previous metrics to current ones and repeat if not converged
            prev_sum_xh1_int = sum_xh1_int
            prev_sum_xh0_int = sum_xh0_int

            # Finally, when using GPU, need to reshape x back for the next ASORA call
            if use_gpu and not converged:
                # FIXME: already flat?
                xh_av_flat = np.ravel(xh_av)

        if use_mpi:
            # broadcast ionised fraction field
            # TODO: use lower case version of MPI functions for compatibility with python objects?
            MPI.COMM_WORLD.Bcast([xh_av_flat, MPI.DOUBLE], root=0)
            MPI.COMM_WORLD.Bcast([xh_intermed, MPI.DOUBLE], root=0)

            # convert the bool variable to bit
            # converged_array = array.array("i", [converged])
            converged_array = array.array("i", [int(converged)])

            # braodcast convergence to the other ranks
            MPI.COMM_WORLD.Bcast(converged_array, root=0)
            if rank != 0:
                converged = bool(converged_array[0])

    if rank == 0:
        # When converged, return the updated ionization fractions at the end of the timestep
        logger.info(
            f"Multiple source convergence reached after {n_count} ray-tracing iterations."
        )
        xh_new = xh_intermed

    if use_mpi:
        # braodcast final result
        MPI.COMM_WORLD.Bcast([xh_new, MPI.DOUBLE], root=0)

    return xh_new, phi_ion
