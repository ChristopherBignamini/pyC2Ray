import logging
import sys
sys.path.append("../../../")
import pyc2ray as pc2r
import numpy as np
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpu",action='store_true')
parser.add_argument("--plot",action='store_true')
args = parser.parse_args()

# Global parameters
num_steps_between_slices = 10
numzred = 2
paramfile = "parameters.yml"

# Create C2Ray object
sim = pc2r.C2Ray_Test(paramfile)

# Get logger
logger = logging.getLogger("pyc2ray.c2ray_test")

# Generate redshift list (test case)
zred_array = sim.generate_redshift_array(numzred,1e7)

# Read sources
numsrc = 1
srcpos, srcflux = sim.read_sources("src.txt",numsrc)

# Measure time
tinit = time.time()

# Setup density
avgdens = 1e-3
shadow_pos = np.array([76,76,63])
shadow_radius = 8
shadow_fact = 6
ndens = avgdens*np.ones((sim.N,sim.N,sim.N))
for i in range(0,sim.N):
    for j in range(0,sim.N):
        for k in range(0,sim.N):
            if (i-shadow_pos[0])**2 + (j-shadow_pos[1])**2 + (k-shadow_pos[2])**2 < shadow_radius**2:
                ndens[i,j,k] = shadow_fact*avgdens

for k in range(len(zred_array)-1):

    # Compute timestep of current redshift slice
    zi = zred_array[k]
    zf = zred_array[k+1]
    dt = sim.set_timestep(zi,zf,num_steps_between_slices)

    # Write output
    sim.write_output(zi)

    # Set density field (could be an actual cosmological field here)
    # TODO: this has to set the comoving density which is then scaled to the
    # correct redshift. In the timesteps, the density is then "diluted" gradually
    #sim.set_constant_average_density(avgdens,0) 
    sim.ndens = ndens

    logger.info("=================================")
    logger.info(f"Doing redshift {zi:.3f} to {zf:.3f}")
    logger.info("=================================")
    # Do num_steps_between_slices timesteps
    for t in range(num_steps_between_slices):
        tnow = time.time()
        logger.info(f"\n --- Timestep {t+1:n}. Redshift: z = {sim.zred : .3f} Wall clock time: {tnow - tinit : .3f} seconds --- \n")

        # Evolve Cosmology: increment redshift and scale physical quantities (density, proper cell size, etc.)
        # If cosmology is disabled in parameter, this step does nothing (checked internally by the class)
        sim.cosmo_evolve(dt)

        # Evolve the simulation: raytrace -> photoionization rates -> chemistry -> until convergence
        sim.evolve3D(dt, srcflux, srcpos)

if args.plot:
    import matplotlib.pyplot as plt
    plt.imshow(sim.xh[:,:,63],norm='log',cmap='jet')
    plt.colorbar()
    plt.show()
    plt.imshow(sim.phi_ion[:,:,63],norm='log',cmap='inferno')
    plt.show()

# Write final output
sim.write_output(zf)
logger.info(f"Done. Final time: {time.time() - tinit : .3f} seconds")