import numpy as np
import healpy as hp
import os
from os.path import join as opj
from cmb_utils import LAT_noise_info, SAT_noise_info

os.environ["OMP_NUM_THREADS"] = "16"
parentdir = '../'

### main noise map generation starts here
Nsim_start = 1
Nsim_end = 1

# running with MPI
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    myid, nproc = comm.Get_rank(), comm.Get_size()
except ImportError:
    myid, nproc = 0, 1 
 
print(myid)

######################################
##### LAT noise maps 
### LAT noise produce T, Q and U
           
outdir = parentdir + "/outputs/lat_noise/" # path to output noise maps

## make sure the path exists
try :
    os.makedirs(outdir)
except : print('exists')
    
random_seed = 10000
N_splits = 1
    
for sim in range(Nsim_start+myid, Nsim_end+1, nproc):
    LAT_noise = LAT_noise_info(splits=N_splits, lmin=50)  # sky coverage and cut off l_min
    for n in range(1,N_splits+1):
        for a, f in enumerate(LAT_noise.freqs):
            lat_noise_map = LAT_noise.get_noise_map(frequency=f, seed = n*random_seed + 10*sim + a)
            hp.write_map(opj(outdir,"lat_noise_%dGHz_split_%d_sid_%03d.fits") %(f, n, sim), lat_noise_map,
                             overwrite=True, dtype=np.float32)
            del lat_noise_map
    del LAT_noise
       
######################################
##### SAT noise maps 
### SAT noise only produce Q and U (only polarisation is needed for r constrain)
         
outdir = parentdir + "outputs/sat_noise" # path to output noise maps
## make sure the path exists

try :
    os.makedirs(outdir)
except : print('exists')

random_seed = 20000
    
for sim in range(Nsim_start+myid, Nsim_end+1, nproc):
    SAT_noise = SAT_noise_info(lmin=20)  # sky coverage and cut off l_min
    for fid, f in enumerate(SAT_noise.freqs):
        sat_noise_map = SAT_noise.get_noise_map(frequency=f, seed = random_seed + 10*sim + fid)
        hp.write_map(opj(outdir, 'sat_noise_%dGHz_sid_%03d.fits') %(f, sim), 
                        sat_noise_map, overwrite=True, dtype=np.float32)        
        del sat_noise_map
    del SAT_noise  
