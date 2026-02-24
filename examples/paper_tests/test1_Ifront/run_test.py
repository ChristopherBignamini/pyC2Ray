import logging
import sys
sys.path.append("../../../")
import pyc2ray as pc2r
import time
import argparse

# ======================================================================
# t_evol = 500 Myr
# COARSE TIME: dt = t_evol / 10  = 50 Myr
# FINE TIME:   dt = t_evol / 100 = 5  Myr
# ======================================================================

# Global parameters
parser = argparse.ArgumentParser()
parser.add_argument("-mode",type=str,default="coarse")
args = parser.parse_args()

mode = str(args.mode)
paramfile = "parameters.yml"

if mode == "coarse":
    numzred = 10
elif mode == "fine":
    numzred = 100
else:
    raise RuntimeError("Unknown mode")

num_steps_between_slices = 1        # Number of timesteps between redshift slices
t_evol = 5e8 # years

# Create C2Ray object
sim = pc2r.C2Ray_Test(paramfile)
logger = logging.getLogger("pyc2ray.c2ray_test")

# Generate redshift list (test case)
zred_array = sim.generate_redshift_array(numzred+1,t_evol/numzred)

# Read sources
srcpos, srcstrength = sim.read_sources("source.txt",1)

# Measure time
tinit = time.time()

# Loop over redshifts
for k in range(len(zred_array)-1):
    zi = zred_array[k]       # Start redshift
    zf = zred_array[k+1]     # End redshift

    logger.info("=================================")
    logger.info(f"Doing redshift {zi:.3f} to {zf:.3f}")
    logger.info("=================================")

    # Compute timestep of current redshift slice
    dt = sim.set_timestep(zi,zf,num_steps_between_slices)

    # Write output
    sim.write_output(zi)

    # Set density (when cosmological is false, zi has no effect)
    sim.density_init(zi)

    # Set redshift to current slice redshift
    sim.zred = zi

    # Loop over timesteps
    for t in range(num_steps_between_slices):
        tnow = time.time()
        logger.info(
            f"--- Timestep {t+1:n}. Redshift: z = {sim.zred : .3f} Wall clock time: {tnow - tinit : .3f} seconds ---"
        )

        # Evolve Cosmology: increment redshift and scale physical quantities (density, proper cell size, etc.)
        # If cosmology is disabled in parameter, this step does nothing
        sim.cosmo_evolve(dt)

        # Evolve the simulation: raytrace -> photoionization rates -> chemistry -> until convergence
        sim.evolve3D(dt, srcstrength, srcpos)

# Write final output
sim.write_output(zf)
logger.info(f"Done. Final time: {time.time() - tinit : .3f} seconds")