#!/usr/bin/env python
import numpy as np
import healpy as hp
import os,sys
import lenspyx
from os.path import join as opj
from cmb_utils import camb_clfile

os.environ["OMP_NUM_THREADS"] = "16"   # set OMP_NUM_THREADS

Nsim_start = 1
Nsim_end = 1

# running with MPI
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    myid, nproc = comm.Get_rank(), comm.Get_size()
except ImportError:
    myid, nproc = 0, 1

print(myid,nproc)

unlen = True # True or False (specify to store unlensed fields)
parentdir = "/home/dekak/Desktop/simons_sims/"
outdir = parentdir + "/outputs/cmb_maps/"# directory to store output maps 'cmb_maps'
cls_dir = parentdir + "/camb_cls/"  # directory of theory cls
kk_path = "/home/dekak/Desktop/simons_sims/outputs/"   # path to convergence map directory


## make sure the output path exists
try :
    os.makedirs(outdir)
except : print('exists')


NSIDE = 2048   # nside of output maps
seed = 123456  # random seed

# lenspyx geometry info
geom_info = ('healpix', {'nside':NSIDE})

lmax = 4096    # lmax for output maps (lensed)
synlmax = 5000  # lmax of unlensed field to generate lensed field
if synlmax is None: synlmax = 8*NSIDE

lmax_unl = synlmax 
lmax_len = lmax # store maps with this lmax
    
# Read the input spectra (in CAMB format)
cl_unl = camb_clfile(opj(cls_dir, 'camb_unlensed_cls.dat'))
Lmax = len(cl_unl['tt']) - 1
print('unlensed lmax : ',Lmax)
# Read the input convergence spectra
data = np.loadtxt(opj(cls_dir,'cl_kk_lmax_7000.dat'))
clkk = data[0] if data.ndim > 1 else data  

pixwin = hp.pixwin(2048, lmax=synlmax)

teb_cl = np.zeros((4, Lmax+1))
teb_cl[0,2:] = cl_unl['tt'][2:]
teb_cl[1,2:] = cl_unl['ee'][2:] 
teb_cl[2,2:] = cl_unl['bb'][2:]
teb_cl[3,2:] = cl_unl['te'][2:]

for sim in range(Nsim_start+myid, Nsim_end+1, nproc):

    # Simulate a CMB and lensing field
    print("Simulation %d of %d" % (sim, Nsim_end - Nsim_start + 1))
    print("Simulating cmb")
    np.random.seed(seed + sim)
    
    ulm = hp.synalm(teb_cl, lmax=synlmax, new=True)

    # if you need primordial cmb maps (not lensed)
    if unlen is True:  
        print("Writing unlensed T,Q,U maps")
        TQU = hp.alm2map(ulm, nside=NSIDE, lmax=lmax_unl, mmax=lmax_unl, pol=True)
        hp.write_map(outdir + "ucmb_lmax_%d_sid_%03d.fits" % (lmax_unl, sim),
                       TQU , overwrite=True, dtype=np.float64)
        del TQU
        
    print("loading lensing field")
    #kappa = hp.read_map(opj(kk_path, 'map_kk_nside_2048_fullsky_nsim_%d.fits') %(sim), field=0)
    #klm = hp.map2alm(kappa, synlmax, iter=3, use_pixel_weights=True)
    #klm = hp.almxfl(klm, 1/pixwin, inplace=False)  # divided by pixel window function
    
    klm = hp.synalm(clkk, lmax=synlmax)
    
    wll = np.zeros(synlmax+1)
    wll[2:] = 2/np.sqrt(np.arange(2,synlmax+1)*np.arange(3,synlmax+2))
    
    # obtain deflection field for Lenspyx
    dlm = hp.almxfl(klm, wll, inplace=False)
    
    del klm , wll, kappa    
     
    # apply lensing 
    print("Apply lensing deflections")
    TQU_len = lenspyx.alm2lenmap(ulm, dlm, geometry=geom_info, verbose=1, epsilon=1e-6)
    del ulm, dlm
    
    print("Writing full-sky lensed T,Q,U maps")
    hp.write_map(outdir + "lcmb_lmax_%d_sid_%03d.fits" %(lmax_len, sim), TQU_len, dtype=np.float64, overwrite=True)
    
    del TQU_len #, so_coverage
    
del cl_unl, teb_cl
